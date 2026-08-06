/// @file wtf_synth.cpp
/// @brief Synthetic multi-class fields → WTF episode → train readout → test.
///
/// CI / fast gate (no data files). Six classes of length-N cube fields:
/// multi-tone carriers in the low half, sparse peaks in the high half, plus
/// deterministic noise. Not a vision claim — a portable orbit → HCNN smoke.
///
/// Pipeline: MakePattern → CollectEpisodes → TrainOnCollected →
/// test MakePattern (rep offset so fields are not the training draws) →
/// PredictClass.
///
/// Default path: frozen reservoir orbit (bypass_reservoir = false).
/// Knobs: MakeWTFConfig() = product WTFConfig; k* below = demo-only.
/// Sibling: examples/mnist/wtf_mnist.cpp (IDX data; study A/Bs).

#include "WTF.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

// =============================================================================
// Demo task size (also sets readout.num_outputs)
// =============================================================================

static constexpr int kNumClasses = 6;

// =============================================================================
// WTF configuration — primary knobs for this demo (edit here)
// =============================================================================
//
// Live MakeWTFConfig() snapshot (keep in sync with the body below):
//   reservoir: dim=7 N=128  M=8  seed=1
//              SR_target=0.999  leak=1  in_scale=0.03  bias_scale=0.003
//   episode:   T=0(=N)  B=1  ic_seed=2  train_input_noise_sigma=0  bypass=false
//   readout:   seed=3  dim=0(auto)  num_outputs=6  epochs=100  batch=32
//              lr_max=0.0015  lr_min_frac=0.01  threads=1  restore_best=false
// Demo: train=64/class test=32/class field_noise_amp=0.18 CI_floor=0.70
// =============================================================================

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics) — smaller cube than MNIST for a fast CI gate
    cfg.reservoir.dim             = 7; // N = 128
    cfg.reservoir.history_depth   = 8;
    cfg.reservoir.seed            = 1;
    cfg.reservoir.spectral_radius = 0.999f;
    cfg.reservoir.leak_rate       = 1.0f;
    cfg.reservoir.input_scaling   = 0.03f; // milder drive → less trivial separation
    cfg.reservoir.bias_scaling    = 0.003f;
    cfg.reservoir.verbose         = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 2;

    // Episode: T = 0 means default T = N after construction
    cfg.episode.T                       = 0;
    cfg.episode.readout_slices          = 1;
    cfg.episode.train_input_noise_sigma = 0.0f;
    cfg.episode.bypass_reservoir        = false;

    // Readout (trainable HCNN)
    cfg.readout.seed               = 3;
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
    cfg.readout.num_threads        = 1; // pin for CI-ish determinism
    cfg.readout.restore_best_epoch = false; // full buffer; no best-epoch holdout

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr int kTrainPerClass = 64;
static constexpr int kTestPerClass  = 32;  // test draws per class
static constexpr double kMinTestAcc = 0.70; // CI gate (task is easy under this recipe)
static constexpr float kNoiseStd    = 0.18f; // deterministic amp inside MakePattern

// Test reps start here so they never reuse training (label, rep) pairs.
static constexpr int kTestRepBase = 10'000;

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

/// Fill a length-N field for (label, rep). Classes share pad/low layout but use
/// class-specific multi-tone carriers + sparse peaks + noise (not trivial ramps).
static void FillPattern(std::span<float> x, int label, int rep)
{
    if (label < 0 || label >= kNumClasses)
        throw std::invalid_argument("FillPattern: label out of range");
    if (x.empty())
        throw std::invalid_argument("FillPattern: empty field");

    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i)
        x[i] = -1.0f;

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
        std::printf(
            "wtf_synth: classes=%d train=%d/class test=%d/class field_noise=%.2f\n",
            kNumClasses, kTrainPerClass, kTestPerClass, static_cast<double>(kNoiseStd));
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        const size_t n_train =
            static_cast<size_t>(kNumClasses) * static_cast<size_t>(kTrainPerClass);
        std::vector<int> train_labels(n_train);
        for (size_t i = 0; i < n_train; ++i)
        {
            const int c = static_cast<int>(i / static_cast<size_t>(kTrainPerClass));
            train_labels[i] = c;
        }

        std::printf("Collecting %zu episodes (parallel)...\n", n_train);
        std::fflush(stdout);
        wtf.CollectEpisodes(
            n_train, train_labels,
            [&](size_t i, std::span<float> out_field) {
                const int c = static_cast<int>(i / static_cast<size_t>(kTrainPerClass));
                const int r = static_cast<int>(i % static_cast<size_t>(kTrainPerClass));
                FillPattern(out_field, c, r);
            });
        std::printf("Training readout on %zu episodes...\n", wtf.NumCollected());
        std::fflush(stdout);
        wtf.TrainOnCollected();
        // Full collected buffer (restore_best_epoch is off in this demo).
        const double acc_on_collected = wtf.AccuracyOnCollected();
        auto t1 = std::chrono::steady_clock::now();

        size_t correct = 0;
        size_t total = 0;
        std::vector<float> field(wtf.N());
        for (int c = 0; c < kNumClasses; ++c)
        {
            for (int r = 0; r < kTestPerClass; ++r)
            {
                FillPattern(field, c, r + kTestRepBase);
                if (wtf.PredictClass(field) == c)
                    ++correct;
                ++total;
            }
        }
        auto t2 = std::chrono::steady_clock::now();

        const double test_acc =
            static_cast<double>(correct) / static_cast<double>(total);
        const double secs_collect_train =
            std::chrono::duration<double>(t1 - t0).count();
        const double secs_test =
            std::chrono::duration<double>(t2 - t1).count();
        const double secs_total =
            std::chrono::duration<double>(t2 - t0).count();

        std::printf("wtf_synth: acc_on_collected=%.3f test_acc=%.3f (%zu/%zu)\n"
                    "wtf_synth: time %.2f+%.2f=%.2fs (collect+train|test|total)\n",
                    acc_on_collected, test_acc, correct, total,
                    secs_collect_train, secs_test, secs_total);
        std::fflush(stdout);

        if (test_acc < kMinTestAcc)
        {
            std::fprintf(stderr,
                         "wtf_synth: test accuracy too low (need >= %.2f)\n",
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
    return exit_code;
}
