/// @file wtf_mnist.cpp
/// @brief MNIST → pack length-N field → WTF episode → train end-state readout.
///
/// Packing is example-owned. DualPlane uses vendored HCNN SpatialEmbed
/// (ink || |grad| on low addresses + pad). Pad/low writes 784 pixels into
/// verts [0,784) and pads the rest (requires N >= 784 → dim >= 10).
///
/// Data: IDX under this repo's data/ only. Edit knobs in the sections below.

#include "WTF.h"
#include "mnist_idx.h"
#include "pack_field.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
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
    cfg.reservoir.verbose      = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 2;

    // Episode: T = 0 means default T = N
    cfg.episode.T              = 0;
    cfg.episode.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.dim                     = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs             = 10;
    cfg.readout.task                    = ReadoutTask::Classification;
    cfg.readout.epochs                  = 40;
    cfg.readout.batch_size              = 32;
    cfg.readout.seed                    = 42;
    cfg.readout.num_threads             = 0; // 0 = HCNN auto
    cfg.readout.restore_best_epoch      = true;
    cfg.readout.best_epoch_holdout_frac = 0.1f;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr PackMode kPack       = PackMode::DualPlane;
static constexpr size_t kMaxTrain     = 20000; // 0 = all (slow: episode cost × T)
static constexpr size_t kMaxTest      = 500;
static constexpr float kPad           = -1.0f;
static constexpr int kImgSide         = 28;
static constexpr int kImgPixels       = kImgSide * kImgSide;

// =============================================================================
// Helpers
// =============================================================================

/// This repository's data/ only (examples/mnist/ → repo root → data/).
static std::filesystem::path ResolveDataDir()
{
    // __FILE__ = .../examples/mnist/wtf_mnist.cpp
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
           / "data";
}

static void PackSample(const WTF& wtf, const wtf_ex::MnistSample& s,
                       std::vector<float>& field)
{
    if (static_cast<int>(s.pixels.size()) != kImgPixels)
        throw std::runtime_error("expected 28x28 MNIST sample");

    if (kPack == PackMode::DualPlane)
    {
        wtf_ex::PackDualPlane28(s.pixels.data(), static_cast<int>(wtf.reservoir().Dim()),
                                field, kPad);
    }
    else
    {
        if (field.size() < static_cast<size_t>(kImgPixels))
            throw std::runtime_error("PadLow requires N >= 784 (use dim >= 10)");
        wtf_ex::PackPadLow(s.pixels, field, kPad);
    }
}

// =============================================================================

int main()
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

        const auto data_dir = ResolveDataDir();
        const auto data_str =
            std::filesystem::absolute(data_dir).lexically_normal().make_preferred().string();

        std::printf("wtf_mnist: loading IDX from %s\n", data_str.c_str());
        auto train = wtf_ex::LoadMnist(
            (data_dir / "train-images-idx3-ubyte").string(),
            (data_dir / "train-labels-idx1-ubyte").string(), kMaxTrain);
        auto test = wtf_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(), kMaxTest);

        WTF wtf(cfg);
        std::vector<float> field(wtf.N());

        const char* pack_name = (kPack == PackMode::DualPlane) ? "DualPlane" : "PadLow";
        std::printf("wtf_mnist: pack=%s N=%zu T=%zu train=%zu test=%zu epochs=%d\n",
                    pack_name, wtf.N(), wtf.T(), train.size(), test.size(),
                    cfg.readout.epochs);

        if (kPack == PackMode::DualPlane)
        {
            auto emb = wtf_ex::MakeDualPlaneEmbedder(
                static_cast<int>(cfg.reservoir.dim), kPad);
            const auto plan = emb.plan(kImgSide, kImgSide);
            std::printf("wtf_mnist: DualPlane S=%d pattern=%d pad_tail=%d\n",
                        plan.plane_side, plan.pattern_length,
                        plan.N - plan.pattern_length);
        }

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < train.size(); ++i)
        {
            PackSample(wtf, train.samples[i], field);
            wtf.CollectEpisode(field, train.samples[i].label);
            if ((i + 1) % 500 == 0)
                std::printf("  collected %zu / %zu\n", i + 1, train.size());
        }
        std::printf("Training readout on %zu episodes...\n", wtf.NumCollected());
        wtf.TrainOnCollected();
        const double train_acc = wtf.AccuracyOnCollected();
        auto t1 = std::chrono::steady_clock::now();

        size_t correct = 0;
        for (size_t i = 0; i < test.size(); ++i)
        {
            PackSample(wtf, test.samples[i], field);
            if (wtf.PredictClass(field) == test.samples[i].label)
                ++correct;
        }
        const double test_acc =
            static_cast<double>(correct) / static_cast<double>(test.size());
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        std::printf("wtf_mnist: train_acc=%.3f test_acc=%.3f (%zu/%zu) "
                    "collect+train=%.1fs\n",
                    train_acc, test_acc, correct, test.size(), secs);
        std::printf("wtf_mnist: note — not an HCNN-mnist bake-off; product is "
                    "field orbit + end-state readout.\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in this repo's data/:\n"
                     "  data/train-images-idx3-ubyte\n"
                     "  data/train-labels-idx1-ubyte\n"
                     "  data/t10k-images-idx3-ubyte\n"
                     "  data/t10k-labels-idx1-ubyte\n"
                     "See data/README.md\n");
        return 1;
    }
}
