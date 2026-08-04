/// Synthetic multi-class demo: invent length-N fields (pad/low style patterns),
/// drive WTF episodes, train readout, score held-out accuracy.
///
/// No image pipeline. CI-friendly. Edit knobs in DemoConfig.

#include "WTF.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {

struct DemoConfig
{
    size_t dim = 6;              // N = 64
    size_t history_depth = 4;
    size_t T = 0;                // 0 → N
    size_t B = 1;
    int num_classes = 4;
    int train_per_class = 40;
    int test_per_class = 20;
    int epochs = 60;
    int batch_size = 16;
    uint64_t reservoir_seed = 1;
    uint64_t ic_seed = 2;
    uint64_t readout_seed = 3;
};

/// Class-conditioned fields already in R^N (values in [-1, 1]).
/// Patterns live in low addresses; high half is pad (−1) so pack story is
/// pad/low without image packing.
std::vector<float> MakePattern(size_t n, int label, int rep)
{
    std::vector<float> x(n, -1.0f);
    const size_t half = n / 2;
    const float jitter = 0.02f * static_cast<float>(rep % 7);

    switch (label % 4)
    {
    case 0: // rising ramp in low half
        for (size_t i = 0; i < half; ++i)
            x[i] = -0.5f + jitter
                   + 1.5f * static_cast<float>(i) / static_cast<float>(half);
        break;
    case 1: // falling ramp
        for (size_t i = 0; i < half; ++i)
            x[i] = 1.0f - jitter
                   - 1.5f * static_cast<float>(i) / static_cast<float>(half);
        break;
    case 2: // block of +1 in first quarter
        for (size_t i = 0; i < n / 4; ++i)
            x[i] = 0.85f + 0.1f * jitter;
        break;
    default: // alternating sign in low half
        for (size_t i = 0; i < half; ++i)
            x[i] = ((i + static_cast<size_t>(rep)) & 1u) ? (0.7f + jitter)
                                                          : (-0.7f - jitter);
        break;
    }
    for (float& v : x)
    {
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
    }
    return x;
}

WTFConfig MakeWtfConfig(const DemoConfig& d)
{
    WTFConfig cfg;
    cfg.reservoir.dim = d.dim;
    cfg.reservoir.history_depth = d.history_depth;
    cfg.reservoir.seed = d.reservoir_seed;
    cfg.reservoir.verbose = false;
    cfg.ic_seed = d.ic_seed;
    cfg.episode.T = d.T;
    cfg.episode.readout_slices = d.B;
    cfg.readout.dim = 0;
    cfg.readout.num_outputs = d.num_classes;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = d.epochs;
    cfg.readout.batch_size = d.batch_size;
    cfg.readout.seed = d.readout_seed;
    cfg.readout.num_threads = 1;
    cfg.readout.restore_best_epoch = false;
    return cfg;
}

} // namespace

int main()
{
    try
    {
        const DemoConfig demo{};
        WTF wtf(MakeWtfConfig(demo));

        std::printf("wtf_synth: N=%zu T=%zu B=%zu classes=%d train=%d/class\n",
                    wtf.N(), wtf.T(), wtf.B(), demo.num_classes,
                    demo.train_per_class);

        auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < demo.num_classes; ++c)
        {
            for (int r = 0; r < demo.train_per_class; ++r)
            {
                auto x = MakePattern(wtf.N(), c, r);
                wtf.CollectEpisode(x, c);
            }
        }
        wtf.TrainOnCollected();
        const double train_acc = wtf.AccuracyOnCollected();
        auto t1 = std::chrono::steady_clock::now();

        size_t correct = 0;
        size_t total = 0;
        for (int c = 0; c < demo.num_classes; ++c)
        {
            for (int r = 0; r < demo.test_per_class; ++r)
            {
                // Offset rep so test fields are not identical to train.
                auto x = MakePattern(wtf.N(), c, r + 1000);
                if (wtf.PredictClass(x) == c)
                    ++correct;
                ++total;
            }
        }
        const double test_acc =
            static_cast<double>(correct) / static_cast<double>(total);
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("wtf_synth: train_acc=%.3f test_acc=%.3f (%zu/%zu) collect+train=%.2fs\n",
                    train_acc, test_acc, correct, total, secs);

        if (test_acc < 0.75)
        {
            std::fprintf(stderr, "wtf_synth: test accuracy too low\n");
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_synth: %s\n", e.what());
        return 1;
    }
}
