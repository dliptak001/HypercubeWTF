/// MNIST demo for HypercubeWTF.
///
/// Pipeline (product hinge):
///   28×28 digit [-1,1]
///     → pack length-N field  (DualPlane default; optional pad/low raw 784)
///     → WTF episode orbit (frozen reservoir)
///     → train HCNN readout on end state only
///
/// Packing is example-owned. DualPlane uses vendored HCNN SpatialEmbed
/// (ink || |grad| on low addresses + pad). Pad/low writes 784 pixels into
/// verts [0,784) and pads the rest (requires N >= 784 → dim >= 10).
///
/// Data: IDX files under this repo's data/ only (not shipped in git).
/// Edit DemoConfig for subset size / dim / pack mode.

#include "WTF.h"
#include "mnist_idx.h"
#include "pack_field.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

enum class PackMode
{
    DualPlane, // effective multi-view field (default)
    PadLow,    // raw 784 in low addresses + pad
};

struct DemoConfig
{
    // ----- Data -----
    size_t max_train = 20000; // 0 = all (slow: episode cost × T)
    size_t max_test = 500;
    PackMode pack = PackMode::DualPlane;

    // ----- Reservoir / episode -----
    size_t dim = 10; // N=1024; DualPlane S≈22; PadLow needs dim>=10
    size_t history_depth = 8;
    size_t T = 0; // 0 → N
    size_t B = 1;
    uint64_t reservoir_seed = 1;
    uint64_t ic_seed = 2;

    // ----- Readout -----
    int epochs = 40;
    int batch_size = 32;
    uint64_t readout_seed = 42;
    int num_threads = 0; // 0 = HCNN auto
};

static constexpr float kPad = -1.0f;
static constexpr int kSide = 28;
static constexpr int kPixels = kSide * kSide;

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
    cfg.readout.num_outputs = 10;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = d.epochs;
    cfg.readout.batch_size = d.batch_size;
    cfg.readout.seed = d.readout_seed;
    cfg.readout.num_threads = static_cast<size_t>(d.num_threads);
    cfg.readout.restore_best_epoch = true;
    cfg.readout.best_epoch_holdout_frac = 0.1f;
    return cfg;
}

/// Always this repository's data/ (sibling of examples/), never another project.
std::filesystem::path ResolveDataDir()
{
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "data";
}

void PackSample(const DemoConfig& d, const wtf_ex::MnistSample& s,
                std::vector<float>& field)
{
    if (static_cast<int>(s.pixels.size()) != kPixels)
        throw std::runtime_error("expected 28x28 MNIST sample");

    if (d.pack == PackMode::DualPlane)
    {
        wtf_ex::PackDualPlane28(s.pixels.data(), static_cast<int>(d.dim), field, kPad);
    }
    else
    {
        if (field.size() < static_cast<size_t>(kPixels))
            throw std::runtime_error(
                "PadLow requires N >= 784 (use dim >= 10)");
        wtf_ex::PackPadLow(s.pixels, field, kPad);
    }
}

} // namespace

int main()
{
    try
    {
        const DemoConfig demo{};
        if (demo.pack == PackMode::PadLow && (size_t{1} << demo.dim) < 784)
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
            (data_dir / "train-labels-idx1-ubyte").string(), demo.max_train);
        auto test = wtf_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(), demo.max_test);

        WTF wtf(MakeWtfConfig(demo));
        std::vector<float> field(wtf.N());

        const char* pack_name =
            (demo.pack == PackMode::DualPlane) ? "DualPlane" : "PadLow";
        std::printf("wtf_mnist: pack=%s N=%zu T=%zu train=%zu test=%zu epochs=%d\n",
                    pack_name, wtf.N(), wtf.T(), train.size(), test.size(),
                    demo.epochs);

        if (demo.pack == PackMode::DualPlane)
        {
            auto emb = wtf_ex::MakeDualPlaneEmbedder(static_cast<int>(demo.dim), kPad);
            const auto plan = emb.plan(kSide, kSide);
            std::printf("wtf_mnist: DualPlane S=%d pattern=%d pad_tail=%d\n",
                        plan.plane_side, plan.pattern_length,
                        plan.N - plan.pattern_length);
        }

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < train.size(); ++i)
        {
            PackSample(demo, train.samples[i], field);
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
            PackSample(demo, test.samples[i], field);
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
