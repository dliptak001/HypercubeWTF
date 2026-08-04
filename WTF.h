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

/// @brief HypercubeWTF: static length-N field → driven reservoir orbit → end features.
///
/// Phase 2: @ref RunEpisode loads frozen s0, drives affine field translation for
/// T passes, packs B end slices into @ref LastFeatures. Train/predict is Phase 3.
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

    /// Feature length after an episode: B * N.
    [[nodiscard]] size_t FeatureSize() const { return B_ * n_; }

    [[nodiscard]] const Reservoir& reservoir() const { return *reservoir_; }
    [[nodiscard]] const ReadoutConfig& readout_config() const { return readout_cfg_; }

    /// Drive one episode. @p x must have size N (throws otherwise). Does not
    /// modify @p x. After return, @ref LastFeatures holds the end-of-episode pack.
    void RunEpisode(std::span<const float> x);

    /// End features from the most recent successful @ref RunEpisode (length B*N).
    /// Empty if no episode has been run yet.
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

private:
    void PackEndFeatures();

    size_t n_ = 0;
    size_t M_ = 0;
    size_t T_ = 0;
    size_t B_ = 1;
    uint64_t ic_seed_ = 0;

    std::unique_ptr<Reservoir> reservoir_;
    ReadoutConfig readout_cfg_{};

    /// Frozen IC length N×M; drawn once at construction.
    std::vector<float> s0_;

    /// Scratch staging field (length N) for translated x.
    std::vector<float> drive_;

    /// Last packed end features (length B*N).
    std::vector<float> last_features_;
};
