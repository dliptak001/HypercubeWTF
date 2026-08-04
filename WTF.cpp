#include "WTF.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

// ---------------------------------------------------------------------------
// Persistent collect thread pool
//
// Background workers live for the WTF lifetime. Each ForEach is fork-join:
// the calling thread is tid 0; workers 1..nthreads-1 take the other chunks.
// Extra parked workers (pool larger than this job) wait out the generation
// without touching active_.
// ---------------------------------------------------------------------------

struct WTF::CollectPool
{
    explicit CollectPool(size_t background_workers)
    {
        workers_.reserve(background_workers);
        for (size_t i = 0; i < background_workers; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }

    ~CollectPool()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    CollectPool(const CollectPool&) = delete;
    CollectPool& operator=(const CollectPool&) = delete;

    [[nodiscard]] size_t NumThreads() const { return workers_.size() + 1; }

    /// @p func(tid, begin, end) over [0, count). Blocks until all done.
    template <typename F>
    void ForEach(size_t count, size_t nthreads, F&& func)
    {
        if (count == 0)
            return;

        nthreads = std::max<size_t>(1, std::min({nthreads, count, NumThreads()}));
        if (nthreads == 1)
        {
            func(size_t{0}, size_t{0}, count);
            return;
        }

        const size_t chunk = (count + nthreads - 1) / nthreads;
        const int bg = static_cast<int>(nthreads - 1);

        {
            std::lock_guard lock(mutex_);
            exception_ = nullptr;
            job_nthreads_ = nthreads;
            active_.store(bg);
            for_func_ = [&func, chunk, count, nthreads](size_t tid) {
                if (tid >= nthreads)
                    return;
                const size_t b = tid * chunk;
                if (b >= count)
                    return;
                func(tid, b, std::min(b + chunk, count));
            };
            ++generation_;
        }
        cv_work_.notify_all();

        std::exception_ptr caller_ex;
        try
        {
            func(size_t{0}, size_t{0}, std::min(chunk, count));
        }
        catch (...)
        {
            caller_ex = std::current_exception();
        }

        {
            std::unique_lock lock(mutex_);
            cv_done_.wait(lock, [this] { return active_.load() == 0; });
            for_func_ = nullptr;
            if (caller_ex)
            {
                exception_ = nullptr;
                std::rethrow_exception(caller_ex);
            }
            if (exception_)
                std::rethrow_exception(exception_);
        }
    }

private:
    void WorkerLoop(size_t tid)
    {
        size_t local_gen = 0;
        std::function<void(size_t)> fn;
        size_t job_nt = 0;
        while (true)
        {
            {
                std::unique_lock lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ > local_gen; });
                if (stop_)
                    return;
                local_gen = generation_;
                fn = for_func_;
                job_nt = job_nthreads_;
            }

            // Pool may be larger than this job; only tid in [1, job_nt) work.
            if (tid < job_nt && fn)
            {
                try
                {
                    fn(tid);
                }
                catch (...)
                {
                    std::lock_guard elock(mutex_);
                    if (!exception_)
                        exception_ = std::current_exception();
                }

                if (active_.fetch_sub(1) == 1)
                {
                    // Synchronize with ForEach's wait (lost-wakeup guard).
                    { std::lock_guard lock(mutex_); }
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::function<void(size_t)> for_func_;
    std::exception_ptr exception_;
    size_t generation_ = 0;
    size_t job_nthreads_ = 0;
    bool stop_ = false;
    alignas(64) std::atomic<int> active_{0};
};

// ---------------------------------------------------------------------------

WTF::WTF(const WTFConfig& cfg)
    : ic_seed_(cfg.ic_seed),
      collect_threads_pref_(cfg.episode.collect_threads),
      readout_cfg_(cfg.readout),
      reservoir_cfg_(cfg.reservoir)
{
    reservoir_ = Reservoir::Create(cfg.reservoir);
    n_ = reservoir_->Size();
    M_ = reservoir_->HistoryDepth();

    T_ = cfg.episode.T == 0 ? n_ : cfg.episode.T;
    if (T_ == 0)
        throw std::invalid_argument("WTF: episode T must be > 0");

    B_ = cfg.episode.readout_slices;
    if (B_ == 0 || (B_ & (B_ - 1)) != 0)
        throw std::invalid_argument("WTF: readout_slices (B) must be a power of two >= 1");
    if (B_ > M_)
        throw std::invalid_argument("WTF: readout_slices (B) must be <= history_depth (M)");

    size_t log2_B = 0;
    for (size_t b = B_; b > 1; b >>= 1)
        ++log2_B;
    const size_t expected_readout_dim = reservoir_->Dim() + log2_B;
    if (readout_cfg_.dim == 0)
        readout_cfg_.dim = expected_readout_dim;
    else if (readout_cfg_.dim != expected_readout_dim)
        throw std::invalid_argument(
            "WTF: readout.dim must be 0 (auto) or reservoir.dim + log2(B)");

    if (readout_cfg_.num_outputs < 1)
        throw std::invalid_argument("WTF: readout.num_outputs must be >= 1");

    readout_ = std::make_unique<Readout>(readout_cfg_);
    if (readout_->NumFeatures() != FeatureSize())
        throw std::logic_error("WTF: readout NumFeatures does not match B*N");

    s0_.assign(n_ * M_, 0.0f);
    std::mt19937_64 rng(mix64(ic_seed_ ^ 0x5343000000000001ULL));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : s0_)
        v = dist(rng);

    drive_.assign(n_, 0.0f);
    last_features_.clear();
    ClearCollected();

    // Worker 0 aliases the primary reservoir + drive_ (no second weight copy).
    CollectWorker primary;
    primary.res = reservoir_.get();
    primary.drive_ptr = drive_.data();
    primary.field.assign(n_, 0.0f);
    collect_workers_.push_back(std::move(primary));
}

WTF::~WTF() = default;

void WTF::PackEndFeaturesFrom(const Reservoir& res, std::span<float> out) const
{
    if (out.size() != FeatureSize())
        throw std::logic_error("WTF::PackEndFeaturesFrom: out size mismatch");
    for (size_t b = 0; b < B_; ++b)
    {
        const float* slice = res.SliceAt(b);
        std::memcpy(out.data() + b * n_, slice, n_ * sizeof(float));
    }
}

void WTF::RunEpisodeOn(Reservoir& res, float* drive,
                       std::span<const float> x, std::span<float> out_features) const
{
    if (x.size() != n_)
        throw std::invalid_argument(
            "WTF::RunEpisode: x.size() must equal N = 2^dim");
    if (drive == nullptr)
        throw std::logic_error("WTF::RunEpisodeOn: null drive");

    // s0_ is immutable after construction; concurrent LoadInitialCondition only
    // reads it (each reservoir memcpy's into its own history buffers).
    res.LoadInitialCondition(s0_.data(), s0_.size());

    const size_t n_mask = n_ - 1;
    size_t c = 0;
    for (size_t pass = 0; pass < T_; ++pass)
    {
        for (size_t v = 0; v < n_; ++v)
            drive[v] = x[(v ^ c) & n_mask];
        res.InjectInputField(drive, n_);
        res.Step();
        ++c;
    }

    PackEndFeaturesFrom(res, out_features);
}

void WTF::RunEpisode(std::span<const float> x)
{
    last_features_.resize(FeatureSize());
    RunEpisodeOn(*reservoir_, drive_.data(), x, last_features_);
}

void WTF::ClearCollected()
{
    collected_features_.clear();
    collected_labels_.clear();
    collected_targets_.clear();
    num_collected_ = 0;
}

void WTF::RequireClassification() const
{
    if (readout_cfg_.task != ReadoutTask::Classification)
        throw std::invalid_argument(
            "WTF: classification API used but task is Regression");
}

void WTF::RequireRegression() const
{
    if (readout_cfg_.task != ReadoutTask::Regression)
        throw std::invalid_argument(
            "WTF: regression API used but task is Classification");
}

void WTF::AppendFeatures(std::span<const float> x)
{
    RunEpisode(x);

    const size_t f = FeatureSize();
    const size_t off = num_collected_ * f;
    collected_features_.resize(off + f);
    std::memcpy(collected_features_.data() + off, last_features_.data(),
                f * sizeof(float));
    ++num_collected_;
}

void WTF::CollectEpisode(std::span<const float> x, int class_label)
{
    RequireClassification();
    if (class_label < 0 || class_label >= readout_cfg_.num_outputs)
        throw std::invalid_argument(
            "WTF::CollectEpisode: class_label must be in [0, num_outputs)");

    AppendFeatures(x);
    collected_labels_.push_back(class_label);
}

void WTF::CollectEpisode(std::span<const float> x, std::span<const float> target)
{
    RequireRegression();
    if (target.size() != static_cast<size_t>(readout_cfg_.num_outputs))
        throw std::invalid_argument(
            "WTF::CollectEpisode: target size must equal num_outputs");

    AppendFeatures(x);
    collected_targets_.insert(collected_targets_.end(), target.begin(), target.end());
}

size_t WTF::ResolveCollectThreads(size_t count) const
{
    if (count == 0)
        return 1;
    size_t n = collect_threads_pref_;
    if (n == 0)
    {
        const unsigned hw = std::thread::hardware_concurrency();
        n = (hw == 0) ? 1 : static_cast<size_t>(hw);
    }
    if (n < 1)
        n = 1;
    return std::min(n, count);
}

void WTF::EnsureCollectWorkers(size_t n)
{
    if (n <= collect_workers_.size())
        return;

    // Owning thread only (before pool work). Build clones in parallel — each
    // Reservoir::Create re-draws weights + SR estimate (non-trivial cost).
    const size_t already = collect_workers_.size();
    const size_t need = n - already;
    std::vector<CollectWorker> fresh(need);
    std::exception_ptr ex;
    std::mutex ex_mu;

    auto build_one = [&](size_t k) {
        try
        {
            ReservoirConfig rc = reservoir_cfg_;
            rc.verbose = false;
            CollectWorker w;
            w.owned = Reservoir::Create(rc);
            w.res = w.owned.get();
            w.drive.assign(n_, 0.0f);
            w.drive_ptr = w.drive.data();
            w.field.assign(n_, 0.0f);
            fresh[k] = std::move(w);
        }
        catch (...)
        {
            std::lock_guard lock(ex_mu);
            if (!ex)
                ex = std::current_exception();
        }
    };

    if (need == 1)
    {
        build_one(0);
    }
    else
    {
        std::vector<std::thread> thr;
        thr.reserve(need);
        for (size_t k = 0; k < need; ++k)
            thr.emplace_back([&, k] { build_one(k); });
        for (auto& t : thr)
            t.join();
    }
    if (ex)
        std::rethrow_exception(ex);

    collect_workers_.reserve(n);
    for (auto& w : fresh)
        collect_workers_.push_back(std::move(w));
}

void WTF::EnsureCollectPool(size_t nthreads)
{
    if (nthreads <= 1)
        return;
    const size_t want_bg = nthreads - 1;
    if (!collect_pool_ || collect_pool_->NumThreads() < nthreads)
        collect_pool_ = std::make_unique<CollectPool>(want_bg);
}

void WTF::CollectFeaturesParallel(
    size_t count,
    const float* fields_flat,
    const std::function<void(size_t, std::span<float>)>& fill_field)
{
    if (count == 0)
        return;

    const bool use_flat = (fields_flat != nullptr);
    const bool use_fill = static_cast<bool>(fill_field);
    if (use_flat == use_fill)
        throw std::logic_error(
            "WTF::CollectFeaturesParallel: need exactly one field source");

    const size_t f = FeatureSize();
    const size_t base = num_collected_;
    collected_features_.resize((base + count) * f);

    const size_t nw = ResolveCollectThreads(count);
    EnsureCollectWorkers(nw);
    EnsureCollectPool(nw);

    auto run_range = [&](size_t tid, size_t begin, size_t end) {
        CollectWorker& w = collect_workers_[tid];
        for (size_t i = begin; i < end; ++i)
        {
            std::span<const float> x;
            if (use_flat)
            {
                x = std::span<const float>(fields_flat + i * n_, n_);
            }
            else
            {
                fill_field(i, std::span<float>(w.field.data(), n_));
                x = std::span<const float>(w.field.data(), n_);
            }

            // Write end-state features straight into the collected matrix.
            float* feat_out = collected_features_.data() + (base + i) * f;
            RunEpisodeOn(*w.res, w.drive_ptr, x,
                         std::span<float>(feat_out, f));
        }
    };

    try
    {
        if (nw <= 1 || !collect_pool_)
            run_range(0, 0, count);
        else
            collect_pool_->ForEach(count, nw, run_range);
    }
    catch (...)
    {
        collected_features_.resize(base * f);
        throw;
    }

    num_collected_ = base + count;
}

// ---------------------------------------------------------------------------
// Public bulk collect entry points
// ---------------------------------------------------------------------------

void WTF::CollectEpisodes(size_t count,
                          std::span<const int> labels,
                          const std::function<void(size_t, std::span<float>)>& fill_field)
{
    RequireClassification();
    if (labels.size() != count)
        throw std::invalid_argument(
            "WTF::CollectEpisodes: labels.size() must equal count");
    if (count == 0)
        return;
    if (!fill_field)
        throw std::invalid_argument("WTF::CollectEpisodes: null fill_field");

    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= readout_cfg_.num_outputs)
            throw std::invalid_argument(
                "WTF::CollectEpisodes: class_label must be in [0, num_outputs)");
    }

    // Labels are tiny: copy serially so the parallel path only writes features.
    const size_t base = num_collected_;
    collected_labels_.resize(base + count);
    std::memcpy(collected_labels_.data() + base, labels.data(),
                count * sizeof(int));

    try
    {
        CollectFeaturesParallel(count, /*fields_flat=*/nullptr, fill_field);
    }
    catch (...)
    {
        collected_labels_.resize(base);
        throw;
    }
}

void WTF::CollectEpisodes(std::span<const float> fields_flat,
                          std::span<const int> labels)
{
    RequireClassification();
    const size_t count = labels.size();
    if (fields_flat.size() != count * n_)
        throw std::invalid_argument(
            "WTF::CollectEpisodes: fields_flat size must be count * N");
    if (count == 0)
        return;

    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= readout_cfg_.num_outputs)
            throw std::invalid_argument(
                "WTF::CollectEpisodes: class_label must be in [0, num_outputs)");
    }

    const size_t base = num_collected_;
    collected_labels_.resize(base + count);
    std::memcpy(collected_labels_.data() + base, labels.data(),
                count * sizeof(int));

    try
    {
        CollectFeaturesParallel(count, fields_flat.data(), /*fill_field=*/{});
    }
    catch (...)
    {
        collected_labels_.resize(base);
        throw;
    }
}

void WTF::CollectEpisodes(size_t count,
                          std::span<const float> targets_flat,
                          const std::function<void(size_t, std::span<float>)>& fill_field)
{
    RequireRegression();
    const size_t no = static_cast<size_t>(readout_cfg_.num_outputs);
    if (targets_flat.size() != count * no)
        throw std::invalid_argument(
            "WTF::CollectEpisodes: targets_flat size must be count * num_outputs");
    if (count == 0)
        return;
    if (!fill_field)
        throw std::invalid_argument("WTF::CollectEpisodes: null fill_field");

    const size_t base = num_collected_;
    collected_targets_.resize((base + count) * no);
    std::memcpy(collected_targets_.data() + base * no, targets_flat.data(),
                count * no * sizeof(float));

    try
    {
        CollectFeaturesParallel(count, /*fields_flat=*/nullptr, fill_field);
    }
    catch (...)
    {
        collected_targets_.resize(base * no);
        throw;
    }
}

void WTF::CollectEpisodes(std::span<const float> fields_flat,
                          std::span<const float> targets_flat)
{
    RequireRegression();
    const size_t no = static_cast<size_t>(readout_cfg_.num_outputs);
    if (no == 0 || fields_flat.size() % n_ != 0)
        throw std::invalid_argument(
            "WTF::CollectEpisodes: fields_flat size must be a multiple of N");
    const size_t count = fields_flat.size() / n_;
    if (targets_flat.size() != count * no)
        throw std::invalid_argument(
            "WTF::CollectEpisodes: targets_flat size must be count * num_outputs");
    if (count == 0)
        return;

    const size_t base = num_collected_;
    collected_targets_.resize((base + count) * no);
    std::memcpy(collected_targets_.data() + base * no, targets_flat.data(),
                count * no * sizeof(float));

    try
    {
        CollectFeaturesParallel(count, fields_flat.data(), /*fill_field=*/{});
    }
    catch (...)
    {
        collected_targets_.resize(base * no);
        throw;
    }
}

void WTF::TrainOnCollected()
{
    if (num_collected_ == 0)
        throw std::invalid_argument("WTF::TrainOnCollected: no samples collected");

    if (readout_cfg_.task == ReadoutTask::Classification)
        readout_->Train(collected_features_.data(), collected_labels_.data(),
                        num_collected_);
    else
        readout_->Train(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
}

std::vector<float> WTF::Predict(std::span<const float> x)
{
    RunEpisode(x);
    std::vector<float> out(readout_->NumOutputs());
    readout_->PredictRaw(last_features_.data(), out.data());
    return out;
}

int WTF::PredictClass(std::span<const float> x)
{
    if (readout_cfg_.task != ReadoutTask::Classification)
        throw std::invalid_argument("WTF::PredictClass: task is not Classification");
    RunEpisode(x);
    return readout_->PredictClass(last_features_.data());
}

double WTF::AccuracyOnCollected() const
{
    if (num_collected_ == 0)
        throw std::invalid_argument("WTF::AccuracyOnCollected: no samples");
    if (readout_cfg_.task != ReadoutTask::Classification)
        throw std::invalid_argument("WTF::AccuracyOnCollected: not classification");
    return readout_->Accuracy(collected_features_.data(), collected_labels_.data(),
                              num_collected_);
}

double WTF::R2OnCollected() const
{
    if (num_collected_ == 0)
        throw std::invalid_argument("WTF::R2OnCollected: no samples");
    if (readout_cfg_.task != ReadoutTask::Regression)
        throw std::invalid_argument("WTF::R2OnCollected: not regression");
    return readout_->R2(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
}
