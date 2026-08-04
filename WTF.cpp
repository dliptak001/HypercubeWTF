#include "WTF.h"

#include <cstring>
#include <random>
#include <stdexcept>

namespace {

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

WTF::WTF(const WTFConfig& cfg)
    : ic_seed_(cfg.ic_seed),
      readout_cfg_(cfg.readout)
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
}

void WTF::PackEndFeatures()
{
    last_features_.resize(B_ * n_);
    for (size_t b = 0; b < B_; ++b)
    {
        const float* slice = reservoir_->SliceAt(b);
        std::memcpy(last_features_.data() + b * n_, slice, n_ * sizeof(float));
    }
}

void WTF::RunEpisode(std::span<const float> x)
{
    if (x.size() != n_)
        throw std::invalid_argument(
            "WTF::RunEpisode: x.size() must equal N = 2^dim");

    reservoir_->LoadInitialCondition(s0_.data(), s0_.size());

    const size_t n_mask = n_ - 1;
    size_t c = 0;
    for (size_t pass = 0; pass < T_; ++pass)
    {
        for (size_t v = 0; v < n_; ++v)
            drive_[v] = x[(v ^ c) & n_mask];
        reservoir_->InjectInputField(drive_.data(), n_);
        reservoir_->Step();
        ++c;
    }

    PackEndFeatures();
}

void WTF::ClearCollected()
{
    collected_features_.clear();
    collected_targets_.clear();
    num_collected_ = 0;
}

void WTF::RequireTargetSize(std::span<const float> target) const
{
    const size_t need = (readout_cfg_.task == ReadoutTask::Classification)
                            ? 1u
                            : static_cast<size_t>(readout_cfg_.num_outputs);
    if (target.size() != need)
        throw std::invalid_argument(
            "WTF::CollectEpisode: target size does not match task "
            "(classification: 1 class index; regression: num_outputs)");

    // Classification: reject non-integral or out-of-range class indices before
    // they become silent static_cast<int> mistakes inside HCNN training.
    if (readout_cfg_.task == ReadoutTask::Classification)
    {
        const float y = target[0];
        const int cls = static_cast<int>(y);
        if (y != static_cast<float>(cls) || cls < 0
            || cls >= readout_cfg_.num_outputs)
        {
            throw std::invalid_argument(
                "WTF::CollectEpisode: class index must be an integer in "
                "[0, num_outputs)");
        }
    }
}

void WTF::CollectEpisode(std::span<const float> x, std::span<const float> target)
{
    RequireTargetSize(target);
    RunEpisode(x);

    const size_t f = FeatureSize();
    const size_t off = num_collected_ * f;
    collected_features_.resize(off + f);
    std::memcpy(collected_features_.data() + off, last_features_.data(),
                f * sizeof(float));

    collected_targets_.insert(collected_targets_.end(), target.begin(), target.end());
    ++num_collected_;
}

void WTF::TrainOnCollected()
{
    if (num_collected_ == 0)
        throw std::invalid_argument("WTF::TrainOnCollected: no samples collected");
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
    return readout_->Accuracy(collected_features_.data(), collected_targets_.data(),
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
