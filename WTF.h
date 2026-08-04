#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

/// Episode / product knobs for @ref WTF.
struct EpisodeConfig
{
    /// Drive-pass count. 0 means “use N after construction” (default T = N).
    size_t T = 0;

    /// End-of-episode delay-line ages packed into features (default 1).
    size_t readout_slices = 1; // B

    /// Parallel workers for @ref WTF::CollectEpisodes (each owns a Reservoir
    /// with the same frozen weights). 0 = auto (hardware_concurrency),
    /// 1 = serial, N = N workers. Single-sample @ref CollectEpisode is always
    /// serial on the primary reservoir.
    ///
    /// Worker 0 reuses the primary reservoir (no extra weight copy). Workers
    /// 1..N-1 are full clones. A persistent thread pool is kept for the WTF
    /// lifetime so bulk collects do not re-spawn OS threads each call.
    size_t collect_threads = 0;
};

/// Full product configuration.
struct WTFConfig
{
    ReservoirConfig reservoir{};
    ReadoutConfig readout{};
    EpisodeConfig episode{};

    /// Separate seed for frozen episode IC s0 (N × M). Not the reservoir weight seed.
    uint64_t ic_seed = 1;
};

/// @brief HypercubeWTF: static length-N field → driven reservoir orbit → HCNN.
///
/// Episode: @ref RunEpisode / @ref CollectEpisode. Batch train: @ref TrainOnCollected.
/// Inference: @ref Predict / @ref PredictClass (each runs a fresh episode).
///
/// Bulk collection (@ref CollectEpisodes) fans out independent episodes across
/// worker reservoirs — episode start does not depend on prior episode state.
class WTF
{
public:
    explicit WTF(const WTFConfig& cfg);
    ~WTF();

    WTF(const WTF&) = delete;
    WTF& operator=(const WTF&) = delete;

    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t T() const { return T_; }
    [[nodiscard]] size_t B() const { return B_; }
    [[nodiscard]] size_t M() const { return M_; }
    [[nodiscard]] size_t FeatureSize() const { return B_ * n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }
    /// Configured collect-thread preference (0 = auto). Actual workers used on
    /// a given bulk collect are min(resolved, sample_count).
    [[nodiscard]] size_t CollectThreads() const { return collect_threads_pref_; }

    [[nodiscard]] const Reservoir& reservoir() const { return *reservoir_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }
    [[nodiscard]] const ReadoutConfig& readout_config() const { return readout_cfg_; }

    /// Drive one episode. @p x must have size N. Does not modify @p x.
    /// After return, @ref LastFeatures holds the end-of-episode pack.
    void RunEpisode(std::span<const float> x);

    /// End features from the most recent successful @ref RunEpisode (length B*N).
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    /// Drop all samples collected for batch training.
    /// Does not free collect-worker reservoirs or the collect thread pool.
    void ClearCollected();

    /// @ref RunEpisode then append features + class label (classification task).
    /// @p class_label must be in [0, NumOutputs()). Serial (primary reservoir).
    void CollectEpisode(std::span<const float> x, int class_label);

    /// @ref RunEpisode then append features + regression targets.
    /// @p target must have length NumOutputs(). Serial (primary reservoir).
    void CollectEpisode(std::span<const float> x, std::span<const float> target);

    /// Parallel bulk collect (classification). @p fields_flat is sample-major,
    /// length count * N; @p labels length count. Appends count episodes.
    /// Fields are read zero-copy (drive remap still uses per-worker scratch).
    void CollectEpisodes(std::span<const float> fields_flat,
                         std::span<const int> labels);

    /// Parallel bulk collect (classification) with a pack/fill callback.
    /// For each sample index i, @p fill_field(i, field) must write N floats
    /// into @p field (thread-safe for concurrent calls on distinct i).
    void CollectEpisodes(size_t count,
                         std::span<const int> labels,
                         const std::function<void(size_t, std::span<float>)>& fill_field);

    /// Parallel bulk collect (regression). @p fields_flat length count * N;
    /// @p targets_flat length count * NumOutputs() (sample-major).
    void CollectEpisodes(std::span<const float> fields_flat,
                         std::span<const float> targets_flat);

    /// Parallel bulk collect (regression) with pack/fill callback.
    void CollectEpisodes(size_t count,
                         std::span<const float> targets_flat,
                         const std::function<void(size_t, std::span<float>)>& fill_field);

    /// Batch-train the HCNN on all collected episodes (requires NumCollected() > 0).
    void TrainOnCollected();

    /// Episode + readout forward; returns NumOutputs() floats (logits or regression).
    [[nodiscard]] std::vector<float> Predict(std::span<const float> x);

    /// Episode + argmax class (classification task only).
    [[nodiscard]] int PredictClass(std::span<const float> x);

    /// Accuracy of the trained readout on the collected set (classification).
    [[nodiscard]] double AccuracyOnCollected() const;

    /// R² on the collected set (regression).
    [[nodiscard]] double R2OnCollected() const;

private:
    /// One episode runner. Worker 0 aliases the primary reservoir + drive_
    /// (no second weight copy). Workers 1.. use owned clones.
    struct CollectWorker
    {
        Reservoir* res = nullptr;         // non-owning view
        std::unique_ptr<Reservoir> owned; // null for primary alias
        std::vector<float> drive;         // length N; empty if using drive_
        std::vector<float> field;         // length N pack scratch
        float* drive_ptr = nullptr;       // → drive_ or drive.data()
    };

    /// Persistent fork-join pool (see WTF.cpp).
    struct CollectPool;

    void PackEndFeaturesFrom(const Reservoir& res, std::span<float> out) const;
    void RunEpisodeOn(Reservoir& res, float* drive,
                      std::span<const float> x, std::span<float> out_features) const;
    void AppendFeatures(std::span<const float> x);
    void RequireClassification() const;
    void RequireRegression() const;

    [[nodiscard]] size_t ResolveCollectThreads(size_t count) const;
    void EnsureCollectWorkers(size_t n);
    void EnsureCollectPool(size_t nthreads);

    /// Parallel feature generation into collected_features_[base .. base+count).
    /// Labels/targets must already be sized for the new samples.
    /// @p fields_flat XOR @p fill_field (exactly one). On throw, feature buffer
    /// is rolled back to @p base rows; caller rolls back labels/targets.
    void CollectFeaturesParallel(
        size_t count,
        const float* fields_flat,
        const std::function<void(size_t, std::span<float>)>& fill_field);

    size_t n_ = 0;
    size_t M_ = 0;
    size_t T_ = 0;
    size_t B_ = 1;
    uint64_t ic_seed_ = 0;
    size_t collect_threads_pref_ = 0; // raw config (0 = auto)

    std::unique_ptr<Reservoir> reservoir_;
    std::unique_ptr<Readout> readout_;
    ReadoutConfig readout_cfg_{};
    ReservoirConfig reservoir_cfg_{}; // for cloning collect workers

    std::vector<float> s0_;
    std::vector<float> drive_;
    std::vector<float> last_features_;

    std::vector<float> collected_features_; // num_collected_ * FeatureSize()
    std::vector<int> collected_labels_;     // classification: one int per episode
    std::vector<float> collected_targets_;  // regression: NumOutputs() floats each
    size_t num_collected_ = 0;

    std::vector<CollectWorker> collect_workers_;
    std::unique_ptr<CollectPool> collect_pool_;
};
