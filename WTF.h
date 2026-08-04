#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

/// Episode / product knobs for @ref WTF (Phase 1 skeleton — drive path lands in Phase 2).
struct EpisodeConfig
{
    /// Drive-pass count. 0 means “use N after construction” (default T = N).
    size_t T = 0;

    /// End-of-episode delay-line ages packed into the readout (default 1).
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
/// Phase 1: constructible shell only (reservoir + readout + s0 allocation).
/// Episode drive / train / predict arrive in later phases.
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

    [[nodiscard]] const Reservoir& reservoir() const { return *reservoir_; }
    [[nodiscard]] const ReadoutConfig& readout_config() const { return readout_cfg_; }

    // Phase 2+: RunEpisode, TrainOnCollected, Predict

private:
    size_t n_ = 0;
    size_t M_ = 0;
    size_t T_ = 0;
    size_t B_ = 1;
    uint64_t ic_seed_ = 0;

    std::unique_ptr<Reservoir> reservoir_;
    ReadoutConfig readout_cfg_{};

    /// Frozen IC length N×M; drawn once at construction.
    std::vector<float> s0_;
};
