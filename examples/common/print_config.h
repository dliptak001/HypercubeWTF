#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Reservoir.h"
#include "WTF.h"

#include <algorithm>
#include <cstdio>

namespace wtf_ex {

inline void PrintReservoirConfig(const ReservoirConfig& r,
                                 float realized_sr = -1.0f)
{
    const size_t N = (r.dim < 8 * sizeof(size_t)) ? (size_t{1} << r.dim) : 0;
    std::printf(
        "reservoir: dim=%zu N=%zu M=%zu seed=%llu "
        "SR_target=%.6g leak=%.6g in_scale=%.6g bias_scale=%.6g",
        r.dim, N, r.history_depth,
        static_cast<unsigned long long>(r.seed),
        static_cast<double>(r.spectral_radius),
        static_cast<double>(r.leak_rate),
        static_cast<double>(r.input_scaling),
        static_cast<double>(r.bias_scaling));
    if (realized_sr >= 0.0f)
        std::printf(" SR_realized=%.6g", static_cast<double>(realized_sr));
    std::printf("\n");
    std::fflush(stdout);
}

/// Live readout shape after WTF construction (dim resolved; weight count from net).
inline void PrintReadoutConfig(const Readout& ro)
{
    const ReadoutConfig& r = ro.GetConfig();
    const int d = static_cast<int>(r.dim);
    // Match Readout::make_layer_specs: 0 = auto min(dim-2, 2), at least 1.
    int layers = (r.num_layers > 0) ? r.num_layers : std::min(d - 2, 2);
    layers = std::max(layers, 1);

    const char* pool = !r.use_pooling
                           ? "off"
                           : (r.pool_type == ReadoutPoolType::Avg) ? "avg" : "max";

    const char* act = "tanh";
    switch (r.activation) {
        case ReadoutActivation::TANH:       act = "tanh"; break;
        case ReadoutActivation::RELU:       act = "relu"; break;
        case ReadoutActivation::LEAKY_RELU: act = "leaky_relu"; break;
        case ReadoutActivation::NONE:       act = "none"; break;
    }

    std::printf("readout: layers=%d weights=%zu pooling=%s activation=%s lr_max=%.6g\n",
                layers, ro.Weights().size(), pool, act,
                static_cast<double>(r.lr_max));
    std::fflush(stdout);
}

/// Reservoir + episode + readout knobs after WTF construction (N/T/B are live).
inline void PrintWtfHeader(const char* demo_name, const WTF& wtf,
                           const WTFConfig& cfg)
{
    std::printf("%s: N=%zu T=%zu B=%zu M=%zu ic_seed=%llu collect_threads=%zu%s "
                "input_noise_sigma=%.4g bypass_reservoir=%s\n",
                demo_name, wtf.N(), wtf.T(), wtf.B(), wtf.M(),
                static_cast<unsigned long long>(cfg.ic_seed),
                cfg.episode.collect_threads,
                cfg.episode.collect_threads == 0
                    ? " (auto: leave 1-2 cores free)"
                    : "",
                static_cast<double>(cfg.episode.input_noise_sigma),
                cfg.episode.bypass_reservoir ? "true" : "false");
    PrintReservoirConfig(cfg.reservoir, wtf.reservoir().GetRealizedSpectralRadius());
    PrintReadoutConfig(wtf.readout());
    std::fflush(stdout);
}

} // namespace wtf_ex
