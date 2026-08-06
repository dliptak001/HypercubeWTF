/// @file wtf_mnist.cpp
/// @brief MNIST → pack length-N field → WTF episode → train end-state readout.
///
/// Packing is example-owned and uses vendored HCNN SpatialEmbed modes:
///   PadLow       — full 28x28 in [0,784), pad [784,N) (needs N >= 784).
///   PadLowCenter — full 28x28 + centered crop in the tail (default; dim=10
///                  fills N exactly with a 15x16 center at (6,6)).
///
/// Train collect only: optional HCNNSpatialAug on 28x28, then pack.
/// Test is never geometrically augmented. Optional demo-only i.i.d. Gaussian on
/// the packed test field (kTestNoiseSigma) is an eval protocol — not WTF core
/// and not episode.train_input_noise_sigma (collect-only). Prefer one noise source at
/// a time when comparing (leave collect/aug noise at 0 for clean test-noise A/B).
///
/// Data: this repo's data/ (discovered via cwd / exe / source tree).
/// Edit knobs in the sections below.

#include "WTF.h"
#include "done_beep.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "pack_field.h"
#include "print_config.h"

#include "HCNNSpatialAug.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Pack mode (demo task — maps to HCNNSpatialEmbedMode)
// =============================================================================

enum class PackMode
{
    PadLow,       // HCNNSpatialEmbedMode::PadLow
    PadLowCenter, // HCNNSpatialEmbedMode::PadLowCenter (default)
};

static hcnn::HCNNSpatialEmbedMode ToEmbedMode(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow:       return hcnn::HCNNSpatialEmbedMode::PadLow;
    case PackMode::PadLowCenter: return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
    }
    return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
}

static const char* PackModeName(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow:       return "PadLow";
    case PackMode::PadLowCenter: return "PadLowCenter";
    }
    return "?";
}

// =============================================================================
// WTF configuration — primary knobs for this demo (edit here)
// =============================================================================
//
// Live MakeWTFConfig() snapshot (keep this comment in sync when you edit below):
//   reservoir: dim=10 N=1024  M=4  seed=13871537636959942979
//              SR_target=0.4  leak=0.5  in_scale=0.005  bias_scale=0
//   episode:   T=20  B=1  ic_seed=12  train_input_noise_sigma=0  bypass_reservoir=false
//   readout:   seed=42  dim=0(auto)  num_outputs=10  num_layers=1  channel_growth=1
//              pooling=max  conv_channels=16  activation=NONE  task=classification
//              epochs=100  batch_size=64  lr_max=0.0015  lr_min_frac=0.01
//              lr_decay_epochs=0  weight_decay=0  num_threads=0(auto)
//              restore_best_epoch=true  optimizer=Adam  best_epoch_holdout_frac=0.1
//
// Demo pack / noise (outside WTFConfig; see k* below): pack=PadLowCenter,
//   train=60000 test=10000, aug=off, kTestNoiseSigma / kTestNoiseSeedBase.
//
// Measured bypass vs reservoir (see examples/mnist/WhiteNoiseFilter.md):
//   clean: reservoir ≈ bypass (test_acc≈0.979)
//   AWGN σ=0.5 on packed field (multi-seed): reservoir≈0.93 vs bypass≈0.85
// =============================================================================

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim           = 10; // N = 1024; PadLow/PadLowCenter need dim >= 10
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.seed          = 13871537636959942979ull;
    cfg.reservoir.spectral_radius = 0.4;
    cfg.reservoir.input_scaling = 0.005;
    cfg.reservoir.leak_rate  = 0.5;
    cfg.reservoir.bias_scaling  = 0.0;
    cfg.reservoir.verbose       = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 12;

    // Episode: T = 0 means default T = N
    cfg.episode.T              = 20;
    cfg.episode.readout_slices = 1;
    // Keep 0 when train aug N(0,σ) is on — do not stack two noise sources.
    cfg.episode.train_input_noise_sigma = 0.0f;
    // true = pack field → readout (no orbit); PackMode still applies. Needs B=1.
    cfg.episode.bypass_reservoir  = true;

    // Readout (trainable HCNN)
    cfg.readout.seed                    = 42;
    cfg.readout.dim                     = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs             = 10;
    cfg.readout.num_layers              = 1;
    cfg.readout.channel_growth          = 1;
    cfg.readout.use_pooling             = true;
    cfg.readout.conv_channels           = 16;
    cfg.readout.activation              = ReadoutActivation::NONE;
    cfg.readout.task                    = ReadoutTask::Classification;
    cfg.readout.epochs                  = 100; //100 for orbit; 20 for bypass;
    cfg.readout.batch_size              = 64;
    // Cosine LR: peak → floor = lr_max * lr_min_frac over lr_decay_epochs (0 = epochs)
    cfg.readout.lr_max                  = 0.0015f;
    cfg.readout.lr_min_frac             = 0.01f;
    cfg.readout.lr_decay_epochs         = 0;
    cfg.readout.weight_decay            = 0.0f;
    cfg.readout.num_threads             = 0; // 0 = HCNN auto
    cfg.readout.restore_best_epoch      = true;
    cfg.readout.optimizer               = ReadoutOptimizer::Adam;
    cfg.readout.best_epoch_holdout_frac = 0.1f; // tail of collected buffer only

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr PackMode kPack       = PackMode::PadLowCenter;
static constexpr size_t kMaxTrain     = 60000; // short default; campaign: 20000 / 0=all
static constexpr size_t kMaxTest      = 10000;
static constexpr float kPad           = -1.0f;
static constexpr int kImgSide         = 28;
static constexpr int kImgPixels       = kImgSide * kImgSide;
static constexpr double kMinTestAcc   = 0.50; // soft CI floor

// Train-only 2D aug (HCNNSpatialAug), then PackMode. Test is never augmented.
// Report line format (when on):
//   aug=rot+/-12+scale[0.9,1.1]+shift+/-2+shear_x+/-0.15+shear_y+/-0+elastic(a=0,s=5)+N(0,0.03)
static constexpr bool  kTrainAug         = false;
static constexpr float kAugRotDegMax     = 12.0f;
static constexpr float kAugScaleMin      = 0.9f;
static constexpr float kAugScaleMax      = 1.1f;
static constexpr int   kAugShiftMax      = 2;
static constexpr float kAugShearXMax     = 0.15f;
static constexpr float kAugShearYMax     = 0.0f;
static constexpr float kAugElasticAlpha  = 0.0f; // 0 = off
static constexpr float kAugElasticSigma  = 5.0f; // ignored when alpha == 0
static constexpr float kAugNoiseSigma    = 0.03f;
static constexpr unsigned kAugSeedBase   = 0xC0FFEEu;

// Test-only protocol: N(0,σ) on the packed length-N field after PackSample,
// before PredictClass. 0 = off (default). Independent of train aug and of
// episode.train_input_noise_sigma. Deterministic per sample from kTestNoiseSeedBase.
// High-noise bypass A/B: try σ in ~0.3–0.5 on [-1,1]-ish packed fields.
static constexpr float    kTestNoiseSigma    = 0.0f;
static constexpr unsigned kTestNoiseSeedBase = 0x7E57u; //0x7E57u;

// =============================================================================
// Helpers
// =============================================================================

static hcnn::HCNNSpatialAugConfig MakeTrainAugConfig()
{
    if (!kTrainAug)
        return hcnn::HCNNSpatialAugConfig::None();

    hcnn::HCNNSpatialAugConfig ac;
    ac.rot_deg_max   = kAugRotDegMax;
    ac.scale_min     = kAugScaleMin;
    ac.scale_max     = kAugScaleMax;
    ac.shift_max     = kAugShiftMax;
    ac.shear_x_max   = kAugShearXMax;
    ac.shear_y_max   = kAugShearYMax;
    ac.elastic_alpha = kAugElasticAlpha;
    ac.elastic_sigma = kAugElasticSigma;
    ac.noise_sigma   = kAugNoiseSigma;
    ac.value_min     = -1.0f;
    ac.value_max     = 1.0f;
    ac.border_value  = kPad;
    ac.enabled       = true;
    return ac;
}

/// One report line; knobs drive the string so the log matches the live config.
static void PrintAugReport()
{
    if (!kTrainAug)
    {
        std::printf("wtf_mnist: aug=off\n");
        return;
    }
    // Keep format stable for scans / diffs.
    std::printf(
        "wtf_mnist: aug=rot+/-%g+scale[%g,%g]+shift+/-%d+shear_x+/-%g+"
        "shear_y+/-%g+elastic(a=%g,s=%g)+N(0,%g)\n",
        static_cast<double>(kAugRotDegMax),
        static_cast<double>(kAugScaleMin),
        static_cast<double>(kAugScaleMax),
        kAugShiftMax,
        static_cast<double>(kAugShearXMax),
        static_cast<double>(kAugShearYMax),
        static_cast<double>(kAugElasticAlpha),
        static_cast<double>(kAugElasticSigma),
        static_cast<double>(kAugNoiseSigma));
}

static void PrintTestNoiseReport()
{
    if (kTestNoiseSigma <= 0.0f)
        std::printf("wtf_mnist: test_noise=off\n");
    else
        std::printf("wtf_mnist: test_noise=N(0,%g) on packed field seed_base=0x%X\n",
                    static_cast<double>(kTestNoiseSigma), kTestNoiseSeedBase);
}

/// Test path: raw 28x28 → pack (no geometric aug). Optional field noise is
/// applied in the eval loop after this returns.
static void PackSample(const wtf_ex::MnistSample& s,
                       std::span<float> field,
                       const hcnn::HCNNSpatialEmbedder& emb)
{
    if (static_cast<int>(s.pixels.size()) != kImgPixels)
        throw std::runtime_error("expected 28x28 MNIST sample");
    wtf_ex::PackMnist28(s.pixels.data(), emb, field);
}

/// In-place i.i.d. Gaussian on a packed field. No-op when kTestNoiseSigma <= 0.
static void AddTestFieldNoise(std::span<float> field, size_t sample_index)
{
    if (kTestNoiseSigma <= 0.0f)
        return;
    std::mt19937 rng(kTestNoiseSeedBase
                     + static_cast<unsigned>(sample_index) * 9973u);
    std::normal_distribution<float> dist(0.0f, kTestNoiseSigma);
    for (float& v : field)
        v += dist(rng);
}

/// Train collect path: optional aug on 28x28, then pack.
/// Parallel-safe: thread_local 784 scratch; per-sample RNG (no shared engine).
static void PackTrainSample(const wtf_ex::MnistSample& s,
                            std::span<float> field,
                            const hcnn::HCNNSpatialEmbedder& emb,
                            const hcnn::HCNNSpatialAugmenter& aug,
                            size_t sample_index)
{
    if (static_cast<int>(s.pixels.size()) != kImgPixels)
        throw std::runtime_error("expected 28x28 MNIST sample");

    const float* pixels = s.pixels.data();
    if (kTrainAug && aug.config().enabled && !aug.config().is_identity())
    {
        // Geometry requires in != out; one scratch buffer per worker thread.
        thread_local std::vector<float> scratch(static_cast<size_t>(kImgPixels));
        if (scratch.size() != static_cast<size_t>(kImgPixels))
            scratch.assign(static_cast<size_t>(kImgPixels), 0.0f);

        std::mt19937 rng(kAugSeedBase
                         + static_cast<unsigned>(sample_index) * 9973u);
        aug.apply(s.pixels.data(), scratch.data(), kImgSide, kImgSide, rng);
        pixels = scratch.data();
    }
    wtf_ex::PackMnist28(pixels, emb, field);
}

// =============================================================================

int main(int argc, char** argv)
{
    int exit_code = 1;
    try
    {
        const WTFConfig cfg = MakeWTFConfig();
        const size_t N_field = size_t{1} << cfg.reservoir.dim;

        if (N_field < static_cast<size_t>(kImgPixels))
        {
            std::fprintf(stderr,
                         "wtf_mnist: pack needs N >= 784 (dim >= 10), got N=%zu\n",
                         N_field);
        }
        else
        {
            const char* argv0 = (argc > 0) ? argv[0] : nullptr;
            const auto data_dir = wtf_ex::FindMnistDataDir(argv0);
            const auto data_str =
                std::filesystem::absolute(data_dir).lexically_normal().make_preferred().string();

            std::printf("wtf_mnist: loading IDX from %s\n", data_str.c_str());
            std::fflush(stdout);

            auto train = wtf_ex::LoadMnist(
                (data_dir / "train-images-idx3-ubyte").string(),
                (data_dir / "train-labels-idx1-ubyte").string(), kMaxTrain);
            auto test = wtf_ex::LoadMnist(
                (data_dir / "t10k-images-idx3-ubyte").string(),
                (data_dir / "t10k-labels-idx1-ubyte").string(), kMaxTest);

            WTF wtf(cfg);
            wtf_ex::PrintWtfHeader("wtf_mnist", wtf, cfg);
            std::vector<float> field(wtf.N());

            const auto emb = wtf_ex::MakeMnistEmbedder(
                static_cast<int>(cfg.reservoir.dim), ToEmbedMode(kPack), kPad);
            if (static_cast<size_t>(emb.capacity()) != wtf.N())
                throw std::logic_error("embed capacity does not match WTF N");

            // One augmenter for the run (const config; concurrent apply OK).
            const hcnn::HCNNSpatialAugmenter train_aug(MakeTrainAugConfig());

            const auto plan = emb.plan(kImgSide, kImgSide);
            std::printf("wtf_mnist: pack=%s train=%zu test=%zu epochs=%d\n",
                        PackModeName(kPack), train.size(), test.size(),
                        cfg.readout.epochs);
            if (kPack == PackMode::PadLowCenter)
            {
                std::printf(
                    "wtf_mnist: PadLowCenter full=%dx%d@ [0,%d) "
                    "center=%dx%d@(%d,%d) pattern=%d N=%d\n",
                    plan.height_in, plan.width_in, kImgPixels,
                    plan.crop_h, plan.crop_w, plan.crop_row0, plan.crop_col0,
                    plan.pattern_length, plan.N);
            }
            else
            {
                std::printf("wtf_mnist: PadLow pattern=%d N=%d pad_tail=%d\n",
                            plan.pattern_length, plan.N,
                            plan.N - plan.pattern_length);
            }
            PrintAugReport();
            PrintTestNoiseReport();
            std::fflush(stdout);

            auto t0 = std::chrono::steady_clock::now();
            std::vector<int> train_labels(train.size());
            for (size_t i = 0; i < train.size(); ++i)
                train_labels[i] = train.samples[i].label;

            std::printf("Collecting %zu episodes (parallel)...\n", train.size());
            std::fflush(stdout);
            wtf.CollectEpisodes(
                train.size(), train_labels,
                [&](size_t i, std::span<float> out_field) {
                    PackTrainSample(train.samples[i], out_field, emb, train_aug, i);
                });
            std::printf("Training readout on %zu episodes...\n", wtf.NumCollected());
            std::fflush(stdout);
            wtf.TrainOnCollected();
            // Whole collected buffer (includes best-epoch holdout tail when enabled).
            const double acc_on_collected = wtf.AccuracyOnCollected();
            auto t1 = std::chrono::steady_clock::now();

            size_t correct = 0;
            for (size_t i = 0; i < test.size(); ++i)
            {
                PackSample(test.samples[i], field, emb); // no geometric aug
                AddTestFieldNoise(field, i);             // optional eval protocol
                if (wtf.PredictClass(field) == test.samples[i].label)
                    ++correct;
            }
            auto t2 = std::chrono::steady_clock::now();
            const double test_acc =
                static_cast<double>(correct) / static_cast<double>(test.size());
            const double secs_collect_train =
                std::chrono::duration<double>(t1 - t0).count();
            const double secs_test =
                std::chrono::duration<double>(t2 - t1).count();
            const double secs_total =
                std::chrono::duration<double>(t2 - t0).count();

            std::printf("wtf_mnist: acc_on_collected=%.3f test_acc=%.3f (%zu/%zu)\n"
                        "wtf_mnist: time %.1f+%.1f=%.1fs (collect+train|test|total)\n",
                        acc_on_collected, test_acc, correct, test.size(),
                        secs_collect_train, secs_test, secs_total);
            std::fflush(stdout);

            if (test_acc < kMinTestAcc)
            {
                std::fprintf(stderr,
                             "wtf_mnist: test accuracy too low (need >= %.2f)\n",
                             kMinTestAcc);
            }
            else
            {
                exit_code = 0;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in HypercubeWTF/data/:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
                     "See data/README.md\n");
    }
    wtf_ex::DoneBeep();
    return exit_code;
}
