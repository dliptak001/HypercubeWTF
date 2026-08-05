/// @file wtf_synth.cpp
/// @brief Synthetic multi-class fields → WTF episode → train readout → held-out test.
/// CI-friendly (no data files). Edit knobs in the sections below.

#include "WTF.h"
#include "done_beep.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// =============================================================================
// WTF configuration — primary knobs for this demo (edit here)
// =============================================================================

static constexpr int kNumClasses = 6;

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim           = 7; // N = 128 — room for harder spatial structure
    cfg.reservoir.history_depth = 8;
    cfg.reservoir.seed          = 1;
    cfg.reservoir.verbose       = false;
    cfg.reservoir.input_scaling = 0.03f; // milder drive → less trivial separation

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 2;

    // Episode: T = 0 means default T = N; B = end-of-episode slices
    cfg.episode.T              = 0;
    cfg.episode.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.dim                = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs        = kNumClasses;
    cfg.readout.task               = ReadoutTask::Classification;
    cfg.readout.epochs             = 100;
    cfg.readout.batch_size         = 32;
    // Cosine LR: peak → floor = lr_max * lr_min_frac over lr_decay_epochs (0 = epochs)
    cfg.readout.lr_max             = 0.0015f;
    cfg.readout.lr_min_frac        = 0.01f;
    cfg.readout.lr_decay_epochs    = 0;
    cfg.readout.weight_decay       = 0.0f;
    cfg.readout.seed               = 3;
    cfg.readout.num_threads        = 1;
    cfg.readout.restore_best_epoch = false;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr int kTrainPerClass = 64;
static constexpr int kTestPerClass  = 32;
static constexpr double kMinTestAcc = 0.70; // harder task; still a CI gate
static constexpr float kNoiseStd    = 0.18f;

// =============================================================================
// Helpers
// =============================================================================

static uint32_t Mix32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/// Deterministic noise in ~[-amp, amp] from (label, rep, index).
static float DetNoise(int label, int rep, size_t i, float amp)
{
    const uint32_t h = Mix32(static_cast<uint32_t>(label) * 0x9E3779B9u
                             ^ static_cast<uint32_t>(rep) * 0x85EBCA6Bu
                             ^ static_cast<uint32_t>(i));
    // Map to (-1, 1)
    const float u = (static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu)) * 2.0f
                    - 1.0f;
    return amp * u;
}

/// Harder multi-class fields: shared pad/low layout, class-specific multi-tone
/// carriers + sparse peaks + noise. Classes are not trivial ramps.
static std::vector<float> MakePattern(size_t n, int label, int rep)
{
    if (label < 0 || label >= kNumClasses)
        throw std::invalid_argument("MakePattern: label out of range");

    std::vector<float> x(n, -1.0f);
    const size_t half = n / 2;
    constexpr float kPi = 3.14159265358979323846f;

    // Base: multi-frequency carrier in low half (class sets frequency set).
    const float f0 = 0.55f + 0.35f * static_cast<float>(label);
    const float f1 = 1.10f + 0.40f * static_cast<float>((label * 3) % kNumClasses);
    const float phase = 0.17f * static_cast<float>(rep % 11);

    for (size_t i = 0; i < half; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(half);
        float v = 0.55f * std::sin(2.0f * kPi * f0 * t + phase);
        v += 0.35f * std::sin(2.0f * kPi * f1 * t - 0.5f * phase);
        // Shared slow ramp (all classes) so pure DC/ramp is not a cue.
        v += 0.15f * (2.0f * t - 1.0f);
        v += DetNoise(label, rep, i, kNoiseStd);
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        x[i] = v;
    }

    // Class-dependent sparse peaks in the high half (pad region otherwise −1).
    const size_t n_peaks = 3 + static_cast<size_t>(label % 3);
    for (size_t p = 0; p < n_peaks; ++p)
    {
        const size_t idx =
            half + ((p * 17u + static_cast<size_t>(label) * 13u + static_cast<size_t>(rep) * 3u)
                    % (n - half));
        x[idx] = (p % 2u == 0) ? 0.9f : -0.85f;
        x[idx] += DetNoise(label, rep, idx, 0.05f);
        if (x[idx] > 1.0f)
            x[idx] = 1.0f;
        if (x[idx] < -1.0f)
            x[idx] = -1.0f;
    }

    return x;
}

// =============================================================================

int main()
{
    int exit_code = 1;
    try
    {
        const WTFConfig cfg = MakeWTFConfig();
        if (cfg.readout.num_outputs != kNumClasses)
            throw std::logic_error("readout.num_outputs must match kNumClasses");

        WTF wtf(cfg);
        wtf_ex::PrintWtfHeader("wtf_synth", wtf, cfg);
        std::printf("wtf_synth: classes=%d train=%d/class noise=%.2f\n",
                    kNumClasses, kTrainPerClass, kNoiseStd);

        auto t0 = std::chrono::steady_clock::now();
        const size_t n_train =
            static_cast<size_t>(kNumClasses) * static_cast<size_t>(kTrainPerClass);
        std::vector<int> train_labels(n_train);
        for (size_t i = 0; i < n_train; ++i)
        {
            const int c = static_cast<int>(i / static_cast<size_t>(kTrainPerClass));
            train_labels[i] = c;
        }
        wtf.CollectEpisodes(
            n_train, train_labels,
            [&](size_t i, std::span<float> out_field) {
                const int c = static_cast<int>(i / static_cast<size_t>(kTrainPerClass));
                const int r = static_cast<int>(i % static_cast<size_t>(kTrainPerClass));
                auto x = MakePattern(wtf.N(), c, r);
                std::memcpy(out_field.data(), x.data(), x.size() * sizeof(float));
            });
        wtf.TrainOnCollected();
        // Full collected buffer (no holdout in this demo) — not a separate test set.
        const double acc_on_collected = wtf.AccuracyOnCollected();
        auto t1 = std::chrono::steady_clock::now();

        size_t correct = 0;
        size_t total = 0;
        for (int c = 0; c < kNumClasses; ++c)
        {
            for (int r = 0; r < kTestPerClass; ++r)
            {
                // Fresh noise/phase stream (rep offset) for held-out fields.
                auto x = MakePattern(wtf.N(), c, r + 10'000);
                if (wtf.PredictClass(x) == c)
                    ++correct;
                ++total;
            }
        }
        const double test_acc =
            static_cast<double>(correct) / static_cast<double>(total);
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("wtf_synth: acc_on_collected=%.3f test_acc=%.3f (%zu/%zu) "
                    "collect+train=%.2fs\n",
                    acc_on_collected, test_acc, correct, total, secs);
        std::fflush(stdout);

        if (test_acc < kMinTestAcc)
        {
            std::fprintf(stderr, "wtf_synth: test accuracy too low (need >= %.2f)\n",
                         kMinTestAcc);
        }
        else
        {
            exit_code = 0;
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_synth: %s\n", e.what());
    }
    wtf_ex::DoneBeep();
    return exit_code;
}
