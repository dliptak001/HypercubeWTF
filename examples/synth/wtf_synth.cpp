/// @file wtf_synth.cpp
/// @brief Synthetic multi-class fields → WTF episode → train readout → held-out test.
/// CI-friendly (no data files). Edit knobs in the sections below.

#include "WTF.h"

#include <chrono>
#include <cstdio>
#include <vector>

// =============================================================================
// WTF configuration — primary knobs for this demo (edit here)
// =============================================================================

static constexpr int kNumClasses = 4;

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim           = 6; // N = 64
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.seed          = 1;
    cfg.reservoir.verbose       = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 2;

    // Episode: T = 0 means default T = N; B = end-of-episode slices
    cfg.episode.T              = 0;
    cfg.episode.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.dim                = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs        = kNumClasses;
    cfg.readout.task               = ReadoutTask::Classification;
    cfg.readout.epochs             = 60;
    cfg.readout.batch_size         = 16;
    cfg.readout.seed               = 3;
    cfg.readout.num_threads        = 1;
    cfg.readout.restore_best_epoch = false;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr int kTrainPerClass = 40;
static constexpr int kTestPerClass  = 20;
static constexpr double kMinTestAcc = 0.75;

// =============================================================================
// Helpers
// =============================================================================

/// Class-conditioned fields already in R^N (values in [-1, 1]).
/// Patterns in low addresses; high half pad (−1) — pad/low story, no images.
static std::vector<float> MakePattern(size_t n, int label, int rep)
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

// =============================================================================

int main()
{
    try
    {
        const WTFConfig cfg = MakeWTFConfig();
        WTF wtf(cfg);

        std::printf("wtf_synth: N=%zu T=%zu B=%zu classes=%d train=%d/class\n",
                    wtf.N(), wtf.T(), wtf.B(), kNumClasses, kTrainPerClass);

        auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < kNumClasses; ++c)
        {
            for (int r = 0; r < kTrainPerClass; ++r)
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
        for (int c = 0; c < kNumClasses; ++c)
        {
            for (int r = 0; r < kTestPerClass; ++r)
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

        if (test_acc < kMinTestAcc)
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
