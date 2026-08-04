// Smoke: episode contract + synthetic 2-class train/predict.
#include "WTF.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
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
    cfg.episode.T = 0; // → N
    cfg.episode.readout_slices = 1;
    cfg.readout.dim = 0;
    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 80;
    cfg.readout.batch_size = 16;
    cfg.readout.num_threads = 1;
    cfg.readout.restore_best_epoch = false;
    return cfg;
}

std::vector<float> MakeField(size_t n, int label)
{
    // Class 0: positive ramp in low half; class 1: negative ramp.
    std::vector<float> x(n, 0.0f);
    const float sign = (label == 0) ? 1.0f : -1.0f;
    for (size_t i = 0; i < n / 2; ++i)
        x[i] = sign * (0.2f + 0.8f * static_cast<float>(i) / static_cast<float>(n));
    return x;
}

bool Near(std::span<const float> a, std::span<const float> b, float eps = 1e-6f)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > eps)
            return false;
    return true;
}

bool ExpectThrow(const char* label, auto&& fn)
{
    try
    {
        fn();
        std::fprintf(stderr, "expected throw: %s\n", label);
        return false;
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
}

} // namespace

int main()
{
    try
    {
        // ----- Episode contract (charter §4) -----
        {
            WTF wtf(MakeCfg());
            if (wtf.N() != 32 || wtf.T() != wtf.N() || wtf.B() != 1 || wtf.M() != 4
                || wtf.FeatureSize() != wtf.N())
            {
                std::fprintf(stderr, "defaults: N=%zu T=%zu B=%zu M=%zu F=%zu\n",
                             wtf.N(), wtf.T(), wtf.B(), wtf.M(), wtf.FeatureSize());
                return 1;
            }

            auto x = MakeField(wtf.N(), 0);
            const auto x_copy = x;

            if (!ExpectThrow("x.size != N", [&] {
                    std::vector<float> bad(wtf.N() + 1, 0.0f);
                    wtf.RunEpisode(bad);
                }))
                return 1;

            wtf.RunEpisode(x);
            if (x != x_copy)
            {
                std::fprintf(stderr, "caller x mutated\n");
                return 1;
            }
            const std::vector<float> feat_a(wtf.LastFeatures().begin(),
                                            wtf.LastFeatures().end());
            wtf.RunEpisode(x);
            if (!Near(wtf.LastFeatures(), feat_a))
            {
                std::fprintf(stderr, "determinism failed\n");
                return 1;
            }
            auto y = MakeField(wtf.N(), 1);
            wtf.RunEpisode(y);
            wtf.RunEpisode(x);
            if (!Near(wtf.LastFeatures(), feat_a))
            {
                std::fprintf(stderr, "isolation failed (residual state / IC)\n");
                return 1;
            }
        }

        // T > N: field address wraps; longer orbit is deterministic and differs from T = N.
        {
            auto cfg_n = MakeCfg();
            auto cfg_2n = MakeCfg();
            cfg_n.episode.T = 0; // N
            cfg_2n.episode.T = 64; // 2N
            WTF w_n(cfg_n);
            WTF w_2n(cfg_2n);
            if (w_n.T() != w_n.N() || w_2n.T() != 2 * w_2n.N())
            {
                std::fprintf(stderr, "T config: T_n=%zu T_2n=%zu N=%zu\n",
                             w_n.T(), w_2n.T(), w_n.N());
                return 1;
            }
            auto x = MakeField(w_n.N(), 0);
            w_n.RunEpisode(x);
            w_2n.RunEpisode(x);
            const std::vector<float> f_n(w_n.LastFeatures().begin(),
                                         w_n.LastFeatures().end());
            const std::vector<float> f_2n(w_2n.LastFeatures().begin(),
                                          w_2n.LastFeatures().end());
            if (Near(f_n, f_2n))
            {
                std::fprintf(stderr, "T>N: features identical to T=N (orbit stuck?)\n");
                return 1;
            }
            w_2n.RunEpisode(x);
            if (!Near(w_2n.LastFeatures(), f_2n))
            {
                std::fprintf(stderr, "T>N: not deterministic\n");
                return 1;
            }
        }

        // B = 2: feature pack is ages 0..1; readout dim auto = dim + 1.
        {
            auto cfg = MakeCfg();
            cfg.episode.readout_slices = 2;
            cfg.readout.dim = 0;
            WTF wtf(cfg);
            if (wtf.B() != 2 || wtf.FeatureSize() != 2 * wtf.N())
            {
                std::fprintf(stderr, "B=2 sizes: B=%zu F=%zu N=%zu\n",
                             wtf.B(), wtf.FeatureSize(), wtf.N());
                return 1;
            }
            if (wtf.readout().NumFeatures() != wtf.FeatureSize())
            {
                std::fprintf(stderr, "B=2 readout NumFeatures mismatch\n");
                return 1;
            }
            auto x = MakeField(wtf.N(), 0);
            wtf.RunEpisode(x);
            const auto feats = wtf.LastFeatures();
            const float* a0 = wtf.reservoir().SliceAt(0);
            const float* a1 = wtf.reservoir().SliceAt(1);
            if (std::memcmp(feats.data(), a0, wtf.N() * sizeof(float)) != 0
                || std::memcmp(feats.data() + wtf.N(), a1, wtf.N() * sizeof(float)) != 0)
            {
                std::fprintf(stderr, "B=2 pack does not match SliceAt(0..1)\n");
                return 1;
            }
        }

        // Invalid B / IC seed splits s0 from weight seed.
        {
            auto bad_b = MakeCfg();
            bad_b.episode.readout_slices = 3; // not power of two
            if (!ExpectThrow("B not power of two", [&] { WTF w(bad_b); }))
                return 1;

            auto bad_bm = MakeCfg();
            bad_bm.episode.readout_slices = 8; // > M=4
            if (!ExpectThrow("B > M", [&] { WTF w(bad_bm); }))
                return 1;

            auto c0 = MakeCfg();
            auto c1 = MakeCfg();
            c1.ic_seed = c0.ic_seed + 99; // same reservoir seed, different IC
            WTF w0(c0);
            WTF w1(c1);
            auto x = MakeField(w0.N(), 0);
            w0.RunEpisode(x);
            w1.RunEpisode(x);
            if (Near(w0.LastFeatures(), w1.LastFeatures()))
            {
                std::fprintf(stderr, "ic_seed: features identical (s0 not separate?)\n");
                return 1;
            }
        }

        // ----- Phase 3 train / predict -----
        {
            WTF wtf(MakeCfg());
            constexpr int kPerClass = 24;
            for (int rep = 0; rep < kPerClass; ++rep)
            {
                auto x0 = MakeField(wtf.N(), 0);
                // Mild per-sample jitter so we are not memorizing one vector.
                x0[rep % (wtf.N() / 2)] += 0.01f * static_cast<float>(rep);
                wtf.CollectEpisode(x0, /*class_label=*/0);

                auto x1 = MakeField(wtf.N(), 1);
                x1[rep % (wtf.N() / 2)] -= 0.01f * static_cast<float>(rep);
                wtf.CollectEpisode(x1, /*class_label=*/1);
            }

            if (wtf.NumCollected() != static_cast<size_t>(2 * kPerClass))
            {
                std::fprintf(stderr, "collect count %zu\n", wtf.NumCollected());
                return 1;
            }

            // Out-of-range class index must fail at collect.
            {
                auto x = MakeField(wtf.N(), 0);
                if (!ExpectThrow("class index OOR",
                                 [&] { wtf.CollectEpisode(x, 2); }))
                    return 1;
            }

            wtf.TrainOnCollected();
            const double acc = wtf.AccuracyOnCollected();
            if (acc < 0.85)
            {
                std::fprintf(stderr, "train accuracy too low: %.3f\n", acc);
                return 1;
            }

            auto x0 = MakeField(wtf.N(), 0);
            auto x1 = MakeField(wtf.N(), 1);
            if (wtf.PredictClass(x0) != 0)
            {
                std::fprintf(stderr, "PredictClass(0) failed\n");
                return 1;
            }
            if (wtf.PredictClass(x1) != 1)
            {
                std::fprintf(stderr, "PredictClass(1) failed\n");
                return 1;
            }

            std::printf("wtf_smoke: ok contract+train acc=%.3f N=%zu T=%zu\n",
                        acc, wtf.N(), wtf.T());
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_smoke: %s\n", e.what());
        return 1;
    }
}
