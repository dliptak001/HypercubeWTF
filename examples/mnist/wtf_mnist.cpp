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
#include "done_beep.h"
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
//
// BEST RUN SO FAR (record only — do not overwrite live knobs below):
//   test_acc=0.968 (968/1000)  acc_on_collected=0.982  collect+train=164.8s
//   dim=10 N=1024  M=1  T=32  B=1  ic_seed=12
//   reservoir.seed=13871537636959942979  SR_target=0.95  (realized ~0.9507)
//   input_scaling=0.015  leak=1  bias_scale=0.003 (default)
//   pack=DualPlane  S=22 pattern=968 pad_tail=56
//   train=60000  test=1000  epochs=40  batch_size=32  lr_max=0.001
//   readout.seed=42  num_layers=1  conv_channels=16  activation=NONE
//   restore_best_epoch=true  best_epoch_holdout_frac=0.1
// Other seeds tried (weaker): 13769974450290969021 (0.964),
//   6963774647319908809 (0.963), 5330595307729750981 (0.962).
// =============================================================================

static WTFConfig MakeWTFConfig()
{
    WTFConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim           = 10; // N = 1024; DualPlane S≈22; PadLow needs dim >= 10
    cfg.reservoir.history_depth = 2;
    cfg.reservoir.seed          = 13871537636959942979ull;//13871537636959942979ull(test_acc=0.965); 13769974450290969021 (test_acc=0.964); 6963774647319908809ull (test_acc=0.963); 5330595307729750981ull(test_acc=0.962);
    cfg.reservoir.spectral_radius = 0.95;
    cfg.reservoir.input_scaling = 0.015;
    cfg.reservoir.verbose       = false;

    // Episode IC (separate from weight seed)
    cfg.ic_seed = 12;

    // Episode: T = 0 means default T = N
    cfg.episode.T              = 32;
    cfg.episode.readout_slices = 2;

    // Readout (trainable HCNN)
    cfg.readout.seed                    = 42;
    cfg.readout.dim                     = 0; // auto = dim + log2(B)
    cfg.readout.num_outputs             = 10;
    cfg.readout.num_layers              = 1;
    cfg.readout.use_pooling             = true;
    cfg.readout.conv_channels           = 16;
    cfg.readout.task                    = ReadoutTask::Classification;
    cfg.readout.activation              = ReadoutActivation::NONE;
    cfg.readout.epochs                  = 40;
    cfg.readout.batch_size              = 32;
    // Cosine LR: peak → floor = lr_max * lr_min_frac over lr_decay_epochs (0 = epochs)
    cfg.readout.lr_max                  = 0.001;
    cfg.readout.lr_min_frac             = 0.01f;
    cfg.readout.lr_decay_epochs         = 0;
    cfg.readout.weight_decay            = 0.0f;
    cfg.readout.num_threads             = 0; // 0 = HCNN auto
    cfg.readout.restore_best_epoch      = true;
    cfg.readout.momentum                = 0.9f; // SGD only; ignored by Adam
    cfg.readout.optimizer               = ReadoutOptimizer::Adam;;  // TODO - try sdg...
    cfg.readout.best_epoch_holdout_frac = 0.1f; // tail of collected buffer only

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of WTFConfig)
// =============================================================================

static constexpr PackMode kPack       = PackMode::DualPlane;
static constexpr size_t kMaxTrain     = 60000; // short default; campaign: 20000 / 0=all
static constexpr size_t kMaxTest      = 1000;
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
    int exit_code = 1;
    try
    {
        const WTFConfig cfg = MakeWTFConfig();

        if (kPack == PackMode::PadLow
            && (size_t{1} << cfg.reservoir.dim) < static_cast<size_t>(kImgPixels))
        {
            std::fprintf(stderr, "wtf_mnist: PadLow needs dim >= 10\n");
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
            std::vector<int> train_labels(train.size());
            for (size_t i = 0; i < train.size(); ++i)
                train_labels[i] = train.samples[i].label;

            std::printf("Collecting %zu episodes (parallel)...\n", train.size());
            std::fflush(stdout);
            wtf.CollectEpisodes(
                train.size(), train_labels,
                [&](size_t i, std::span<float> out_field) {
                    PackSample(train.samples[i], out_field, kPack, emb_ptr);
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
