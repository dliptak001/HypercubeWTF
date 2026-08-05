#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Reservoir.h"
#include "WTF.h"

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

/// Reservoir + episode knobs after WTF construction (N/T/B are live).
inline void PrintWtfHeader(const char* demo_name, const WTF& wtf,
                           const WTFConfig& cfg)
{
    std::printf("%s: N=%zu T=%zu B=%zu M=%zu ic_seed=%llu collect_threads=%zu%s\n",
                demo_name, wtf.N(), wtf.T(), wtf.B(), wtf.M(),
                static_cast<unsigned long long>(cfg.ic_seed),
                cfg.episode.collect_threads,
                cfg.episode.collect_threads == 0
                    ? " (auto: leave 1-2 cores free)"
                    : "");
    PrintReservoirConfig(cfg.reservoir, wtf.reservoir().GetRealizedSpectralRadius());
    std::fflush(stdout);
}

} // namespace wtf_ex
