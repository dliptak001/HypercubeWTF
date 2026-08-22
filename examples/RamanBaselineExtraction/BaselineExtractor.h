#pragma once

#include "WTF.h"

#include <span>
#include <string>

struct RamanSplit;

constexpr size_t kDim = 11;
constexpr size_t kN = size_t{1} << kDim;

inline WTFConfig MakeBaseConfig()
{
    WTFConfig cfg;

    // The normalized spectrum ([-1, 1], see RamanNorm.h) is the reservoir
    // drive. There is no gain stage in front of the orbit and none behind
    // it: reservoir.input_scaling sets how hard the field drives the cube,
    // and the end-of-orbit state goes to the readout as is. wtf_raman prints
    // mean |feature| after one episode so that scale can be watched.
    cfg.reservoir.dim = kDim;
    cfg.reservoir.seed = 13871537636959942979ull;
    cfg.reservoir.spectral_radius = 0.95f;
    cfg.reservoir.history_depth = 8;
    cfg.reservoir.leak_rate = 1.0f;
    cfg.reservoir.input_scaling = 0.045f;
    cfg.reservoir.bias_scaling = 0.001f;

    cfg.episode.T = 60;
    cfg.episode.readout_slices = 1;
    cfg.episode.collect_threads = 0; // auto
    cfg.ic_seed = 1;

    cfg.readout.dim = 0; // auto: reservoir.dim + log2(readout_slices)
    cfg.readout.epochs = 60;
    cfg.readout.num_outputs = static_cast<int>(kN);
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.batch_size = 48;
    cfg.readout.conv_channels = 1;
    cfg.readout.channel_growth = 1;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = false;
    cfg.readout.lr_max = 0.003f;
    cfg.readout.lr_min_frac = 0.04f;
    cfg.readout.restore_best_epoch = true;

    return cfg;
}

class BaselineExtractor
{
public:
    BaselineExtractor();
    explicit BaselineExtractor(const WTFConfig& cfg);
    ~BaselineExtractor() = default;

    BaselineExtractor(const BaselineExtractor&) = delete;
    BaselineExtractor& operator=(const BaselineExtractor&) = delete;
    BaselineExtractor(BaselineExtractor&&) = delete;
    BaselineExtractor& operator=(BaselineExtractor&&) = delete;

    [[nodiscard]] size_t Dim() const { return wtf_.reservoir().Dim(); }
    [[nodiscard]] size_t N() const { return wtf_.N(); }
    [[nodiscard]] size_t NumOutputs() const { return wtf_.NumOutputs(); }

    [[nodiscard]] const WTFConfig& config() const { return cfg_; }
    [[nodiscard]] WTF& wtf() { return wtf_; }
    [[nodiscard]] const WTF& wtf() const { return wtf_; }

    void Collect(const RamanSplit& split);
    void Train();
    void Predict(std::span<const float> spectrum, std::span<float> baseline);

    void SaveReadout(const std::string& path_stem) const;
    void LoadReadout(const std::string& path_stem);

private:
    WTFConfig cfg_;
    WTF wtf_;
};
