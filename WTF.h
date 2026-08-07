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
    /// with the same frozen weights). 0 = auto (desktop-friendly: leave 1–2
    /// cores free for the OS/UI), 1 = serial, K = K workers (pin for max burn).
    /// Single-sample @ref CollectEpisode is always serial on the primary.
    ///
    /// Auto policy: max(1, hw − 1), or max(1, hw − 2) when hw ≥ 8.
    /// Worker 0 reuses the primary reservoir (no extra weight copy). Workers
    /// 1..K-1 are full clones. A persistent thread pool is kept for the WTF
    /// lifetime (grows to the high-water mark; does not shrink) so bulk collects
    /// do not re-spawn OS threads each call.
    size_t collect_threads = 0;

    /// Train/collect-only i.i.d. Gaussian noise on the length-N field before
    /// the episode (σ of N(0,σ) per vertex). 0 = off (default; zero cost).
    /// Applied on @ref CollectEpisode / @ref CollectEpisodes only — not on
    /// @ref RunEpisode / @ref Predict / @ref PredictClass. Fresh draw every
    /// collected sample (deterministic from ic_seed + sample index).
    /// Name stresses train path: this is not test/eval noise (see demo knobs).
    float train_input_noise_sigma = 0.0f;

    /// If true, skip the reservoir orbit: readout features are the length-N
    /// packed field itself (host packing / PackMode still applies). Requires
    /// readout_slices B == 1. Train input noise still applies when σ > 0.
    bool bypass_reservoir = false;
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
/// Typical lifecycle: @ref CollectEpisode / @ref CollectEpisodes →
/// @ref TrainOnCollected → @ref Predict / @ref PredictClass.
/// Episode: @ref RunEpisode (or collect APIs). Inference each runs a fresh episode
/// (no train-input noise).
///
/// Bulk collection (@ref CollectEpisodes) fans out independent episodes across
/// worker reservoirs — episode start does not depend on prior episode state.
///
/// One @ref WTF instance is not thread-safe for concurrent public calls from
/// multiple host threads. Parallelism is internal to bulk @ref CollectEpisodes.
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

    /// Drive one episode (or copy field → features when @ref EpisodeConfig::bypass_reservoir).
    /// @p x must have size N. Does not modify @p x.
    /// After return, @ref LastFeatures holds the feature pack (B*N).
    void RunEpisode(std::span<const float> x);

    /// Feature pack (length B*N) from the most recent successful path that runs
    /// an episode into the primary buffer: @ref RunEpisode, serial
    /// @ref CollectEpisode, @ref Predict, or @ref PredictClass.
    /// Not updated by bulk @ref CollectEpisodes (features go only into the
    /// collected training set).
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    /// True when the reservoir orbit is skipped (field → readout).
    [[nodiscard]] bool BypassReservoir() const { return bypass_reservoir_; }

    /// Drop all samples collected for batch training.
    /// Does not free collect-worker reservoirs or the collect thread pool.
    void ClearCollected();

    /// One collect episode, then append features + class label (classification).
    /// Applies @ref EpisodeConfig::train_input_noise_sigma when > 0, then the
    /// same episode path as @ref RunEpisode (including bypass). Updates
    /// @ref LastFeatures. @p class_label must be in [0, NumOutputs()).
    /// Serial (primary reservoir).
    void CollectEpisode(std::span<const float> x, int class_label);

    /// One collect episode, then append features + regression targets.
    /// Same noise / episode / @ref LastFeatures rules as the classification
    /// overload. @p target must have length NumOutputs(). Serial (primary).
    void CollectEpisode(std::span<const float> x, std::span<const float> target);

    /// Parallel bulk collect (classification). @p fields_flat is sample-major,
    /// length count * N; @p labels length count. Appends count episodes.
    /// Fields are read zero-copy (drive remap still uses per-worker scratch).
    /// Does not update @ref LastFeatures.
    void CollectEpisodes(std::span<const float> fields_flat,
                         std::span<const int> labels);

    /// Parallel bulk collect (classification) with a pack/fill callback.
    /// For each sample index i, @p fill_field(i, field) must write N floats
    /// into @p field (thread-safe for concurrent calls on distinct i).
    /// Does not update @ref LastFeatures.
    void CollectEpisodes(size_t count,
                         std::span<const int> labels,
                         const std::function<void(size_t, std::span<float>)>& fill_field);

    /// Parallel bulk collect (regression). @p fields_flat length count * N;
    /// @p targets_flat length count * NumOutputs() (sample-major).
    /// Does not update @ref LastFeatures.
    void CollectEpisodes(std::span<const float> fields_flat,
                         std::span<const float> targets_flat);

    /// Parallel bulk collect (regression) with pack/fill callback.
    /// Does not update @ref LastFeatures.
    void CollectEpisodes(size_t count,
                         std::span<const float> targets_flat,
                         const std::function<void(size_t, std::span<float>)>& fill_field);

    /// Batch-train the HCNN on all collected episodes (requires NumCollected() > 0).
    /// Does not clear the collected set — call again to retrain, or
    /// @ref ClearCollected first to start over.
    void TrainOnCollected();

    /// Fresh episode + readout forward; returns NumOutputs() floats (logits or
    /// regression). No train-input noise. Updates @ref LastFeatures.
    [[nodiscard]] std::vector<float> Predict(std::span<const float> x);

    /// Fresh episode + argmax class (classification task only).
    /// No train-input noise. Updates @ref LastFeatures.
    [[nodiscard]] int PredictClass(std::span<const float> x);

    /// Accuracy of the trained readout on the collected (training) set —
    /// not a test-set metric (classification).
    [[nodiscard]] double AccuracyOnCollected() const;

    /// R² on the collected (training) set — not a test-set metric (regression).
    [[nodiscard]] double R2OnCollected() const;

    /// Episode IC seed (construction-time; not the reservoir weight seed).
    [[nodiscard]] uint64_t IcSeed() const { return ic_seed_; }

    /// Train/collect-only field noise σ (0 = off).
    [[nodiscard]] float TrainInputNoiseSigma() const { return train_input_noise_sigma_; }

    // ── Readout persistence (thin forwards; same blob/HCNW contract as Readout) ──

    [[nodiscard]] bool IsReadoutTrained() const { return readout_->IsTrained(); }
    [[nodiscard]] std::vector<double> GetReadoutWeights() const { return readout_->Weights(); }
    void SetReadoutWeights(std::vector<double> weights,
                           ReadoutLoadMode mode = ReadoutLoadMode::Eval)
    {
        readout_->SetState(std::move(weights), mode);
    }
    void SaveReadoutHcnnModel(const std::string& path_stem) const
    {
        readout_->SaveHcnnModel(path_stem);
    }
    void LoadReadoutHcnnModel(const std::string& path_stem,
                              ReadoutLoadMode mode = ReadoutLoadMode::Eval)
    {
        readout_->LoadHcnnModel(path_stem, mode);
    }
    [[nodiscard]] std::string ReadoutArchSummary() const { return readout_->ArchSummary(); }
    [[nodiscard]] int ReadoutBestEpoch() const { return readout_->BestEpoch(); }

private:
    /// One episode runner. Worker 0 aliases the primary reservoir + drive_
    /// (no second weight copy). Workers 1.. use owned clones.
    struct CollectWorker
    {
        Reservoir* res = nullptr;         // non-owning view
        std::unique_ptr<Reservoir> owned; // null for primary alias
        std::vector<float> drive;         // length N; empty if using drive_
        std::vector<float> field;         // length N pack scratch
        std::vector<float> noise;         // length N if train_input_noise_sigma_ > 0
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
    float train_input_noise_sigma_ = 0.0f;  // train/collect-only; 0 = off
    bool bypass_reservoir_ = false;   // field → features, no orbit

    std::unique_ptr<Reservoir> reservoir_;
    std::unique_ptr<Readout> readout_;
    ReadoutConfig readout_cfg_{};
    ReservoirConfig reservoir_cfg_{}; // for cloning collect workers

    std::vector<float> s0_;
    std::vector<float> drive_;
    std::vector<float> last_features_;
    std::vector<float> noise_field_; // serial CollectEpisode scratch when σ > 0

    std::vector<float> collected_features_; // num_collected_ * FeatureSize()
    std::vector<int> collected_labels_;     // classification: one int per episode
    std::vector<float> collected_targets_;  // regression: NumOutputs() floats each
    size_t num_collected_ = 0;

    std::vector<CollectWorker> collect_workers_;
    std::unique_ptr<CollectPool> collect_pool_;
};
