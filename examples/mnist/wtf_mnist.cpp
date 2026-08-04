/// @file wtf_mnist.cpp
/// @brief MNIST → pack length-N field → WTF episode → train end-state readout.
///
/// Packing is example-owned. DualPlane uses vendored HCNN SpatialEmbed
/// (ink || |grad| on low addresses + pad). Pad/low writes 784 pixels into
/// verts [0,784) and pads the rest (requires N >= 784 → dim >= 10).
///
/// Data: this repo's data/ (discovered via cwd / exe / source tree).
/// Edit knobs in the sections below.

#include "WTF.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "pack_field.h"
#include "print_config.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Pack mode (demo task — not part of WTFConfig)
// =============================================================================

enum class PackMode
{
    DualPlane, // multi-view field (default)
    PadLow,    // raw 784 in low addresses + pad
};

// =============================================================================
// WTF configuration — primary knobs for this demo (edit here)
// =============================================================================

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim           = 10; // N = 1024; DualPlane S≈22; PadLow needs dim >= 10
    cfg.reservoir.history_depth = 8;
    cfg.reservoir.seed          = 1;
    cfg.reservoir.spectral_radius = 0.999;
    cfg.reservoir.input_scaling = 0.015;
    cfg.reservoir.verbose       = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 12;

    // Episode: T = 0 means default T = N
    cfg.episode.T              = 128;
    cfg.episode.readout_slices = 2;

    // Readout (trainable HCNN)
    cfg.readout.dim                     = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs             = 10;
    cfg.readout.num_layers              = 1;
    cfg.readout.use_pooling             = true;
    cfg.readout.conv_channels            = 8;
    cfg.readout.task                    = ReadoutTask::Classification;
    cfg.readout.activation              = ReadoutActivation::TANH;
    cfg.readout.epochs                  = 40;
    cfg.readout.batch_size              = 32;
    cfg.readout.seed                    = 42;
    cfg.readout.num_threads             = 0; // 0 = HCNN auto
    cfg.readout.restore_best_epoch      = true;
    cfg.readout.momentum                = 0.9f;
    cfg.readout.best_epoch_holdout_frac = 0.1f; // tail of collected buffer only

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr PackMode kPack       = PackMode::DualPlane;
static constexpr size_t kMaxTrain     = 2000; // short default; campaign: 20000 / 0=all
static constexpr size_t kMaxTest      = 500;
static constexpr float kPad           = -1.0f;
static constexpr int kImgSide         = 28;
static constexpr int kImgPixels       = kImgSide * kImgSide;
static constexpr double kMinTestAcc   = 0.50; // soft CI floor

// =============================================================================
// Helpers
// =============================================================================

static void PackSample(const wtf_ex::MnistSample& s,
                       std::span<float> field,
                       PackMode pack,
                       const hcnn::HCNNSpatialEmbedder* dual_emb)
{
    if (static_cast<int>(s.pixels.size()) != kImgPixels)
        throw std::runtime_error("expected 28x28 MNIST sample");

    if (pack == PackMode::DualPlane)
    {
        if (dual_emb == nullptr)
            throw std::logic_error("DualPlane pack requires embedder");
        wtf_ex::PackDualPlane28(s.pixels.data(), *dual_emb, field);
    }
    else
    {
        if (field.size() < static_cast<size_t>(kImgPixels))
            throw std::runtime_error("PadLow requires N >= 784 (use dim >= 10)");
        wtf_ex::PackPadLow(s.pixels, field, kPad);
    }
}

// =============================================================================

int main(int argc, char** argv)
{
    try
    {
        const WTFConfig cfg = MakeWTFConfig();

        if (kPack == PackMode::PadLow
            && (size_t{1} << cfg.reservoir.dim) < static_cast<size_t>(kImgPixels))
        {
            std::fprintf(stderr, "wtf_mnist: PadLow needs dim >= 10\n");
            return 1;
        }

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

        // One DualPlane embedder for the whole run (not per sample).
        std::optional<hcnn::HCNNSpatialEmbedder> dual_emb;
        if (kPack == PackMode::DualPlane)
            dual_emb = wtf_ex::MakeDualPlaneEmbedder(
                static_cast<int>(cfg.reservoir.dim), kPad);
        const hcnn::HCNNSpatialEmbedder* emb_ptr =
            dual_emb ? &*dual_emb : nullptr;

        const char* pack_name = (kPack == PackMode::DualPlane) ? "DualPlane" : "PadLow";
        std::printf("wtf_mnist: pack=%s train=%zu test=%zu epochs=%d\n",
                    pack_name, train.size(), test.size(), cfg.readout.epochs);
        if (dual_emb)
        {
            const auto plan = dual_emb->plan(kImgSide, kImgSide);
            std::printf("wtf_mnist: DualPlane S=%d pattern=%d pad_tail=%d\n",
                        plan.plane_side, plan.pattern_length,
                        plan.N - plan.pattern_length);
        }
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < train.size(); ++i)
        {
            PackSample(train.samples[i], field, kPack, emb_ptr);
            wtf.CollectEpisode(field, train.samples[i].label);
            if ((i + 1) % 500 == 0)
            {
                std::printf("  collected %zu / %zu\n", i + 1, train.size());
                std::fflush(stdout);
            }
        }
        std::printf("Training readout on %zu episodes...\n", wtf.NumCollected());
        std::fflush(stdout);
        wtf.TrainOnCollected();
        // Whole collected buffer (includes best-epoch holdout tail when enabled).
        const double acc_on_collected = wtf.AccuracyOnCollected();
        auto t1 = std::chrono::steady_clock::now();

        size_t correct = 0;
        for (size_t i = 0; i < test.size(); ++i)
        {
            PackSample(test.samples[i], field, kPack, emb_ptr);
            if (wtf.PredictClass(field) == test.samples[i].label)
                ++correct;
        }
        const double test_acc =
            static_cast<double>(correct) / static_cast<double>(test.size());
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("wtf_mnist: acc_on_collected=%.3f test_acc=%.3f (%zu/%zu) "
                    "collect+train=%.1fs\n",
                    acc_on_collected, test_acc, correct, test.size(), secs);
        std::fflush(stdout);

        if (test_acc < kMinTestAcc)
        {
            std::fprintf(stderr,
                         "wtf_mnist: test accuracy too low (need >= %.2f)\n",
                         kMinTestAcc);
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in HypercubeWTF/data/:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
                     "See data/README.md\n");
        return 1;
    }
}
