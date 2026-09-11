#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"
#include "RamanPaths.h"
#include "RamanScore.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr int kTrainSamples = 10000;
static constexpr int kTestSamples = 2000;
static constexpr bool kSkipTrain = false;

static BaselineExtractor* s_ex = nullptr;
static const RamanSplit* s_train = nullptr;

static void EpochTick(int epoch, double)
{
    const double train_rmse = RamanRmseOnCollected(*s_ex, *s_train);
    std::printf("wtf_raman: epoch=%d train_rmse=%.6f\n", epoch, train_rmse);
    std::fflush(stdout);
}

// One normalized spectrum through WTF::RunEpisode, then mean |value| of the
// drive and of the end-of-orbit features the readout sees. ~1 is a live
// field, ~0 is crushed. lr_max * N * mean|f| is the number to keep an eye
// on: the readout goes unstable when it climbs much past ~0.1.
static void ReportFeatureScale(const char* name, BaselineExtractor& ex,
                               std::span<const float> xn)
{
    auto mean_abs = [](std::span<const float> x) {
        if (x.empty())
            return 0.0;
        double a = 0.0;
        for (float v : x)
            a += std::fabs(static_cast<double>(v));
        return a / static_cast<double>(x.size());
    };
    ex.wtf().RunEpisode(xn);
    const double mean_x = mean_abs(xn);
    const double mean_f = mean_abs(ex.wtf().LastFeatures());
    const double lr = static_cast<double>(ex.config().readout.lr_max);
    std::printf("%s: feature scale after one normalized train spectrum "
                "(mean |value| over N=%zu; ~1 is a live field, ~0 is crushed)\n",
                name, ex.N());
    std::printf("%s:   Reservoir drive (normalized spectrum)   mean|x|=%.4g\n",
                name, mean_x);
    std::printf("%s:   Readout features (end-of-orbit state)    mean|f|=%.4g\n",
                name, mean_f);
    std::printf("%s:   lr_max * N * mean|f| = %.4g\n",
                name, lr * static_cast<double>(ex.N()) * mean_f);
    std::fflush(stdout);
}

int main()
{
    int exit_code = 1;
    try
    {
        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "kTrainSamples / kTestSamples must be >= 0 (0 = whole split)");
        }

        const std::filesystem::path root(kRamanDataRoot);
        const auto train = LoadRamanSplit(
            root / "Training", static_cast<size_t>(kTrainSamples));
        const auto test = LoadRamanSplit(
            root / "Validation", static_cast<size_t>(kTestSamples));

        if (train.count == 0)
            throw std::runtime_error("empty training split");

        const std::string stem(kRamanModelStem);

        WTFConfig cfg = MakeBaseConfig();
        if (!kSkipTrain)
            cfg.readout.epoch_tick = EpochTick;

        BaselineExtractor ex(cfg);
        if (ex.N() != kN)
            throw std::logic_error("extractor N must equal kN");
        s_ex = &ex;
        s_train = &train;

        std::printf("wtf_raman: train=%zu val=%zu skip_train=%s\n",
                    train.count, test.count,
                    kSkipTrain ? "true" : "false");
        wtf_ex::PrintWtfHeader("wtf_raman", ex.wtf(), ex.config());
        {
            std::vector<float> xn(kN);
            const auto nrm = RamanNorm::FromSpectrum(train.Spectrum(0));
            nrm.Apply(train.Spectrum(0), xn);
            ReportFeatureScale("wtf_raman", ex, xn);
        }
        {
            // Same resolve as HCNN ThreadPool: 0 = auto (hw-1 workers + caller),
            // 1 = caller only, N > 1 = N workers + caller.
            const unsigned hw = std::thread::hardware_concurrency();
            const size_t knob = ex.config().readout.num_threads;
            size_t pool_nt = 1;
            if (knob == 0)
                pool_nt = (hw > 1) ? static_cast<size_t>(hw) : 1;
            else if (knob > 1)
                pool_nt = knob + 1;
            std::printf("wtf_raman: hw=%u pool_nt=%zu\n", hw, pool_nt);
            std::fflush(stdout);
        }

        if (kSkipTrain)
        {
            ex.LoadReadout(stem);
            std::printf("wtf_raman: loaded %s.hcnw\n", stem.c_str());
            std::fflush(stdout);
        }
        else
        {
            ex.Collect(train);
            std::printf("wtf_raman: collected\n");
            std::fflush(stdout);

            const auto t0 = std::chrono::steady_clock::now();
            ex.Train();
            const auto t1 = std::chrono::steady_clock::now();
            const double train_secs =
                std::chrono::duration<double>(t1 - t0).count();
            std::printf("wtf_raman: trained time=%.2fs best_epoch=%d\n",
                        train_secs, ex.wtf().ReadoutBestEpoch());
            std::fflush(stdout);

            ex.SaveReadout(stem);
            {
                BaselineExtractor check(cfg);
                check.LoadReadout(stem);
                std::vector<float> a(kN), b(kN);
                const auto probe = train.Spectrum(0);
                ex.Predict(probe, a);
                check.Predict(probe, b);
                for (size_t i = 0; i < kN; ++i)
                {
                    if (std::fabs(a[i] - b[i]) > 1e-3f)
                    {
                        throw std::runtime_error(
                            "wtf_raman: saved readout failed reload check");
                    }
                }
            }
            std::printf("wtf_raman: saved %s.hcnw %s.arch.json\n",
                        stem.c_str(), stem.c_str());
            std::fflush(stdout);
        }

        const double train_rmse = RamanRmse(ex, train);
        const double val_rmse = RamanRmse(ex, test);
        std::printf("wtf_raman: train_rmse=%.6f val_rmse=%.6f\n",
                    train_rmse, val_rmse);
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_raman: %s\n", e.what());
    }
    return exit_code;
}
