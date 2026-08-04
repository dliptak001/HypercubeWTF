// Smoke: episode path + synthetic 2-class train/predict.
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

} // namespace

int main()
{
    try
    {
        // ----- Phase 2 episode checks -----
        {
            WTF wtf(MakeCfg());
            auto x = MakeField(wtf.N(), 0);
            const auto x_copy = x;

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
                std::fprintf(stderr, "isolation failed\n");
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
                float t0 = 0.0f;
                wtf.CollectEpisode(x0, std::span<const float>(&t0, 1));

                auto x1 = MakeField(wtf.N(), 1);
                x1[rep % (wtf.N() / 2)] -= 0.01f * static_cast<float>(rep);
                float t1 = 1.0f;
                wtf.CollectEpisode(x1, std::span<const float>(&t1, 1));
            }

            if (wtf.NumCollected() != static_cast<size_t>(2 * kPerClass))
            {
                std::fprintf(stderr, "collect count %zu\n", wtf.NumCollected());
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

            std::printf("wtf_smoke: ok episode+train acc=%.3f N=%zu T=%zu\n",
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
