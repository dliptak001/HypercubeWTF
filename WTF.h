#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <cstddef>
#include <cstdint>
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
class WTF
{
public:
    explicit WTF(const WTFConfig& cfg);

    WTF(const WTF&) = delete;
    WTF& operator=(const WTF&) = delete;

    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t T() const { return T_; }
    [[nodiscard]] size_t B() const { return B_; }
    [[nodiscard]] size_t M() const { return M_; }
    [[nodiscard]] size_t FeatureSize() const { return B_ * n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }

    [[nodiscard]] const Reservoir& reservoir() const { return *reservoir_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }
    [[nodiscard]] const ReadoutConfig& readout_config() const { return readout_cfg_; }

    /// Drive one episode. @p x must have size N. Does not modify @p x.
    /// After return, @ref LastFeatures holds the end-of-episode pack.
    void RunEpisode(std::span<const float> x);

    /// End features from the most recent successful @ref RunEpisode (length B*N).
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    /// Drop all samples collected for batch training.
    void ClearCollected();

    /// @ref RunEpisode then append features + class label (classification task).
    /// @p class_label must be in [0, NumOutputs()).
    void CollectEpisode(std::span<const float> x, int class_label);

    /// @ref RunEpisode then append features + regression targets.
    /// @p target must have length NumOutputs().
    void CollectEpisode(std::span<const float> x, std::span<const float> target);

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
    void PackEndFeatures();
    void AppendFeatures(std::span<const float> x);
    void RequireClassification() const;
    void RequireRegression() const;

    size_t n_ = 0;
    size_t M_ = 0;
    size_t T_ = 0;
    size_t B_ = 1;
    uint64_t ic_seed_ = 0;

    std::unique_ptr<Reservoir> reservoir_;
    std::unique_ptr<Readout> readout_;
    ReadoutConfig readout_cfg_{};

    std::vector<float> s0_;
    std::vector<float> drive_;
    std::vector<float> last_features_;

    std::vector<float> collected_features_; // num_collected_ * FeatureSize()
    std::vector<int> collected_labels_;     // classification: one int per episode
    std::vector<float> collected_targets_;  // regression: NumOutputs() floats each
    size_t num_collected_ = 0;
};
