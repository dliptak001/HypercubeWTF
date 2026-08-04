// Phase 2 smoke: construct, RunEpisode, determinism, isolation, size check.
#include "WTF.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {

WTFConfig MakeCfg()
{
    WTFConfig cfg;
    cfg.reservoir.dim = 5; // N = 32
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.seed = 1;
    cfg.reservoir.verbose = false;
    cfg.ic_seed = 2;
    cfg.episode.T = 0;              // → N
    cfg.episode.readout_slices = 1; // B = 1
    cfg.readout.dim = 0;
    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    return cfg;
}

std::vector<float> MakeField(size_t n, float fill)
{
    std::vector<float> x(n, 0.0f);
    for (size_t i = 0; i < n / 2; ++i)
        x[i] = fill * (static_cast<float>(i + 1) / static_cast<float>(n));
    return x;
}

bool Near(std::span<const float> a, std::span<const float> b, float eps = 1e-6f)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::fabs(a[i] - b[i]) > eps)
            return false;
    }
    return true;
}

} // namespace

int main()
{
    try
    {
        WTF wtf(MakeCfg());
        if (wtf.N() != 32 || wtf.T() != 32 || wtf.B() != 1 || wtf.M() != 4)
        {
            std::fprintf(stderr, "sizes: N=%zu T=%zu B=%zu M=%zu\n",
                         wtf.N(), wtf.T(), wtf.B(), wtf.M());
            return 1;
        }

        auto x = MakeField(wtf.N(), 1.0f);
        const auto x_copy = x;

        // Wrong size throws.
        try
        {
            std::vector<float> bad(wtf.N() + 1, 0.0f);
            wtf.RunEpisode(bad);
            std::fprintf(stderr, "expected throw on size mismatch\n");
            return 1;
        }
        catch (const std::invalid_argument&)
        {
        }

        wtf.RunEpisode(x);
        if (x != x_copy)
        {
            std::fprintf(stderr, "caller x was mutated\n");
            return 1;
        }
        if (wtf.LastFeatures().size() != wtf.FeatureSize())
        {
            std::fprintf(stderr, "feature size mismatch\n");
            return 1;
        }
        const std::vector<float> feat_a(wtf.LastFeatures().begin(),
                                        wtf.LastFeatures().end());

        // Determinism: same x again → same features.
        wtf.RunEpisode(x);
        if (!Near(wtf.LastFeatures(), feat_a))
        {
            std::fprintf(stderr, "determinism failed\n");
            return 1;
        }

        // Isolation: different sample in between does not change re-run of x.
        auto y = MakeField(wtf.N(), -0.7f);
        wtf.RunEpisode(y);
        const std::vector<float> feat_y(wtf.LastFeatures().begin(),
                                        wtf.LastFeatures().end());
        if (Near(feat_y, feat_a))
        {
            std::fprintf(stderr, "expected different features for different x\n");
            return 1;
        }
        wtf.RunEpisode(x);
        if (!Near(wtf.LastFeatures(), feat_a))
        {
            std::fprintf(stderr, "isolation failed after intervening episode\n");
            return 1;
        }

        // T > N (wrap): still runs and yields features of correct size.
        {
            WTFConfig cfg = MakeCfg();
            cfg.episode.T = wtf.N() + 5;
            WTF w2(cfg);
            w2.RunEpisode(x);
            if (w2.LastFeatures().size() != w2.FeatureSize())
            {
                std::fprintf(stderr, "T>N feature size bad\n");
                return 1;
            }
        }

        // Cross-instance determinism (same seeds).
        {
            WTF w3(MakeCfg());
            w3.RunEpisode(x);
            if (!Near(w3.LastFeatures(), feat_a))
            {
                std::fprintf(stderr, "cross-instance determinism failed\n");
                return 1;
            }
        }

        std::printf("wtf_smoke: ok episode N=%zu T=%zu B=%zu features=%zu\n",
                    wtf.N(), wtf.T(), wtf.B(), wtf.FeatureSize());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_smoke: %s\n", e.what());
        return 1;
    }
}
