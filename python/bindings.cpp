// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Thin pybind11 surface for HypercubeWTF. Ergonomics (shape checks, fit,
// pickle, docs) live in hypercube_wtf/__init__.py.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../WTF.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

namespace {

void require_field_size(size_t got, size_t n, const char* what)
{
    if (got != n)
        throw std::invalid_argument(
            std::string(what) + " size (" + std::to_string(got)
            + ") must equal N (" + std::to_string(n) + ")");
}

} // namespace

// Single de-templated WTF binding. Hypercube dim is a runtime constructor
// argument (cfg.reservoir.dim), so one C++ type and one Python class serve
// every dimension 5–16 — no per-DIM instantiations.
PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeWTF: frozen hypercube reservoir orbit + HypercubeCNN on end state";
    m.attr("__version__") = "0.1.0";

    py::class_<WTF>(m, "_WTF")
        // ── Construction ──
        // All reservoir + episode + readout parameters fixed at construction.
        .def(py::init([](size_t dim, uint64_t seed, float spectral_radius,
                         float input_scaling, float leak_rate, size_t history_depth,
                         bool verbose, float bias_scaling, uint64_t ic_seed,
                         size_t episode_T, size_t readout_slices,
                         size_t collect_threads, float train_input_noise_sigma,
                         bool bypass_reservoir,
                         int readout_num_outputs, const char* readout_task,
                         int readout_num_layers, int readout_conv_channels,
                         int readout_epochs, int readout_batch_size,
                         float readout_lr_max, float readout_lr_min_frac,
                         int readout_lr_decay_epochs, float readout_weight_decay,
                         float readout_momentum, const char* readout_activation,
                         uint64_t readout_seed, size_t readout_num_threads,
                         bool readout_restore_best_epoch,
                         float readout_best_epoch_holdout_frac,
                         bool readout_use_pooling) {
            WTFConfig cfg;
            cfg.reservoir.dim             = dim;
            cfg.reservoir.seed            = seed;
            cfg.reservoir.spectral_radius = spectral_radius;
            cfg.reservoir.input_scaling   = input_scaling;
            cfg.reservoir.leak_rate       = leak_rate;
            cfg.reservoir.history_depth   = history_depth;
            cfg.reservoir.verbose         = verbose;
            cfg.reservoir.bias_scaling    = bias_scaling;
            cfg.ic_seed                   = ic_seed;
            cfg.episode.T                       = episode_T;
            cfg.episode.readout_slices          = readout_slices;
            cfg.episode.collect_threads         = collect_threads;
            cfg.episode.train_input_noise_sigma = train_input_noise_sigma;
            cfg.episode.bypass_reservoir        = bypass_reservoir;
            cfg.readout.num_outputs = readout_num_outputs;
            cfg.readout.task = (std::strcmp(readout_task, "classification") == 0)
                                   ? ReadoutTask::Classification
                                   : ReadoutTask::Regression;
            if (std::strcmp(readout_task, "classification") != 0
                && std::strcmp(readout_task, "regression") != 0) {
                throw std::invalid_argument(
                    std::string("readout_task must be 'classification' or "
                                "'regression' (got '")
                    + readout_task + "')");
            }
            cfg.readout.num_layers      = readout_num_layers;
            cfg.readout.conv_channels   = readout_conv_channels;
            cfg.readout.epochs          = readout_epochs;
            cfg.readout.batch_size      = readout_batch_size;
            cfg.readout.lr_max          = readout_lr_max;
            cfg.readout.lr_min_frac     = readout_lr_min_frac;
            cfg.readout.lr_decay_epochs = readout_lr_decay_epochs;
            cfg.readout.weight_decay    = readout_weight_decay;
            cfg.readout.momentum        = readout_momentum;
            if      (std::strcmp(readout_activation, "relu") == 0)
                cfg.readout.activation = ReadoutActivation::RELU;
            else if (std::strcmp(readout_activation, "leaky_relu") == 0)
                cfg.readout.activation = ReadoutActivation::LEAKY_RELU;
            else if (std::strcmp(readout_activation, "none") == 0)
                cfg.readout.activation = ReadoutActivation::NONE;
            else if (std::strcmp(readout_activation, "tanh") == 0)
                cfg.readout.activation = ReadoutActivation::TANH;
            else
                throw std::invalid_argument(
                    std::string("readout_activation must be one of "
                                "'tanh', 'relu', 'leaky_relu', 'none' (got '")
                    + readout_activation + "')");
            cfg.readout.seed                    = readout_seed;
            cfg.readout.num_threads             = readout_num_threads;
            cfg.readout.restore_best_epoch      = readout_restore_best_epoch;
            cfg.readout.best_epoch_holdout_frac = readout_best_epoch_holdout_frac;
            cfg.readout.use_pooling             = readout_use_pooling;
            return std::make_unique<WTF>(cfg);
        }),
            py::arg("dim"),
            py::arg("seed")                     = 7934791766227647176ULL,
            py::arg("spectral_radius")          = 0.999f,
            py::arg("input_scaling")            = 0.02f,
            py::arg("leak_rate")                = 1.0f,
            py::arg("history_depth")            = 16ULL,
            py::arg("verbose")                  = false,
            py::arg("bias_scaling")             = 0.003f,
            py::arg("ic_seed")                  = 1ULL,
            py::arg("episode_T")                = 100ULL,
            py::arg("readout_slices")           = 1ULL,
            py::arg("collect_threads")          = 0ULL,
            py::arg("train_input_noise_sigma")  = 0.0f,
            py::arg("bypass_reservoir")         = false,
            py::arg("readout_num_outputs")      = 1,
            py::arg("readout_task")             = "regression",
            py::arg("readout_num_layers")       = 1,
            py::arg("readout_conv_channels")    = 16,
            py::arg("readout_epochs")           = 200,
            py::arg("readout_batch_size")       = 32,
            py::arg("readout_lr_max")           = 0.0015f,
            py::arg("readout_lr_min_frac")      = 0.01f,
            py::arg("readout_lr_decay_epochs")  = 0,
            py::arg("readout_weight_decay")     = 0.0f,
            py::arg("readout_momentum")         = 0.9f,
            py::arg("readout_activation")       = "tanh",
            py::arg("readout_seed")             = 42ULL,
            py::arg("readout_num_threads")      = 0ULL,
            py::arg("readout_restore_best_epoch") = true,
            py::arg("readout_best_epoch_holdout_frac") = 0.0f,
            py::arg("readout_use_pooling")      = true)

        // ── Episode ──
        .def("run_episode", [](WTF& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            self.RunEpisode({static_cast<const float*>(buf.ptr), self.N()});
        }, py::arg("x"),
           "Drive one episode (or bypass copy). Updates last_features.")

        .def("last_features", [](const WTF& self) {
            auto span = self.LastFeatures();
            py::array_t<float> arr(span.size());
            if (!span.empty())
                std::memcpy(arr.mutable_data(), span.data(),
                            span.size() * sizeof(float));
            return arr;
        }, "Feature pack (B*N) from the last serial episode path.")

        .def("clear_collected", &WTF::ClearCollected,
             "Drop all samples collected for batch training.")

        // ── Collect (serial) ──
        .def("collect_episode_class", [](WTF& self, FloatArray x, int class_label) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            self.CollectEpisode(
                {static_cast<const float*>(buf.ptr), self.N()}, class_label);
        }, py::arg("x"), py::arg("class_label"),
           "Serial collect one classification episode.")

        .def("collect_episode_reg", [](WTF& self, FloatArray x, FloatArray target) {
            auto xbuf = x.request();
            auto tbuf = target.request();
            require_field_size(static_cast<size_t>(xbuf.size), self.N(), "field");
            if (static_cast<size_t>(tbuf.size) != self.NumOutputs())
                throw std::invalid_argument(
                    "target size (" + std::to_string(tbuf.size)
                    + ") must equal num_outputs (" + std::to_string(self.NumOutputs())
                    + ")");
            py::gil_scoped_release release;
            self.CollectEpisode(
                {static_cast<const float*>(xbuf.ptr), self.N()},
                {static_cast<const float*>(tbuf.ptr),
                 static_cast<size_t>(tbuf.size)});
        }, py::arg("x"), py::arg("target"),
           "Serial collect one regression episode.")

        // ── Collect (bulk) ──
        .def("collect_episodes_class", [](WTF& self, FloatArray fields, IntArray labels) {
            auto fbuf = fields.request();
            auto lbuf = labels.request();
            const size_t n = self.N();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(lbuf.size) != count)
                throw std::invalid_argument(
                    "labels length (" + std::to_string(lbuf.size)
                    + ") must equal sample count (" + std::to_string(count) + ")");
            py::gil_scoped_release release;
            self.CollectEpisodes(
                {static_cast<const float*>(fbuf.ptr), total},
                {static_cast<const int*>(lbuf.ptr), count});
        }, py::arg("fields"), py::arg("labels"),
           "Bulk parallel collect (classification). fields: (count, N) or flat count*N.")

        .def("collect_episodes_reg", [](WTF& self, FloatArray fields, FloatArray targets) {
            auto fbuf = fields.request();
            auto tbuf = targets.request();
            const size_t n = self.N();
            const size_t k = self.NumOutputs();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(tbuf.size) != count * k)
                throw std::invalid_argument(
                    "targets size (" + std::to_string(tbuf.size)
                    + ") must equal count * num_outputs ("
                    + std::to_string(count * k) + ")");
            py::gil_scoped_release release;
            self.CollectEpisodes(
                {static_cast<const float*>(fbuf.ptr), total},
                {static_cast<const float*>(tbuf.ptr),
                 static_cast<size_t>(tbuf.size)});
        }, py::arg("fields"), py::arg("targets"),
           "Bulk parallel collect (regression). fields: (count, N); "
           "targets: (count, num_outputs) or flat.")

        // ── Train / predict ──
        .def("train", [](WTF& self) {
            py::gil_scoped_release release;
            self.TrainOnCollected();
        }, "Batch-train the HCNN on all collected episodes.")

        .def("predict", [](WTF& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            std::vector<float> out;
            {
                py::gil_scoped_release release;
                out = self.Predict({static_cast<const float*>(buf.ptr), self.N()});
            }
            py::array_t<float> arr(out.size());
            std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(float));
            return arr;
        }, py::arg("x"),
           "Fresh episode + readout forward; returns (num_outputs,) float32.")

        .def("predict_class", [](WTF& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            return self.PredictClass({static_cast<const float*>(buf.ptr), self.N()});
        }, py::arg("x"),
           "Fresh episode + argmax class (classification only).")

        .def("accuracy_on_collected", &WTF::AccuracyOnCollected,
             "Train-set accuracy on collected episodes (not a test metric).")

        .def("r2_on_collected", &WTF::R2OnCollected,
             "Train-set R² on collected episodes (not a test metric).")

        // ── Properties ──
        .def_property_readonly("N", &WTF::N)
        .def_property_readonly("T", &WTF::T)
        .def_property_readonly("B", &WTF::B)
        .def_property_readonly("M", &WTF::M)
        .def_property_readonly("feature_size", &WTF::FeatureSize)
        .def_property_readonly("num_collected", &WTF::NumCollected)
        .def_property_readonly("num_outputs", &WTF::NumOutputs)
        .def_property_readonly("collect_threads", &WTF::CollectThreads)
        .def_property_readonly("bypass_reservoir", &WTF::BypassReservoir)
        .def_property_readonly("ic_seed", &WTF::IcSeed)
        .def_property_readonly("train_input_noise_sigma", &WTF::TrainInputNoiseSigma)
        .def_property_readonly("dim", [](const WTF& self) {
            return self.reservoir().Dim();
        })
        .def_property_readonly("seed", [](const WTF& self) {
            return self.reservoir().GetConfig().seed;
        })
        .def_property_readonly("spectral_radius", [](const WTF& self) {
            return self.reservoir().GetConfig().spectral_radius;
        })
        .def_property_readonly("realized_spectral_radius", [](const WTF& self) {
            return self.reservoir().GetRealizedSpectralRadius();
        })
        .def_property_readonly("input_scaling", [](const WTF& self) {
            return self.reservoir().GetConfig().input_scaling;
        })
        .def_property_readonly("leak_rate", [](const WTF& self) {
            return self.reservoir().GetConfig().leak_rate;
        })
        .def_property_readonly("history_depth", [](const WTF& self) {
            return self.reservoir().GetConfig().history_depth;
        })
        .def_property_readonly("bias_scaling", [](const WTF& self) {
            return self.reservoir().GetConfig().bias_scaling;
        })
        .def_property_readonly("readout_task", [](const WTF& self) {
            return self.readout_config().task == ReadoutTask::Classification
                       ? "classification"
                       : "regression";
        })
        .def_property_readonly("readout_best_epoch", &WTF::ReadoutBestEpoch)

        // ── Persistence helpers ──
        .def("_get_readout_state", [](const WTF& self) -> py::dict {
            py::dict d;
            d["is_trained"] = self.IsReadoutTrained();
            auto w = self.GetReadoutWeights();
            d["weights"] = py::array_t<double>(
                {static_cast<py::ssize_t>(w.size())}, w.data());
            return d;
        })
        .def("_set_readout_state", [](WTF& self, py::dict d) {
            if (!d.contains("is_trained") || !d["is_trained"].cast<bool>())
                return;
            auto w = d["weights"].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
            std::vector<double> weights(w.data(), w.data() + w.size());
            ReadoutLoadMode mode = ReadoutLoadMode::Eval;
            if (d.contains("mode")) {
                const auto ms = d["mode"].cast<std::string>();
                if (ms == "resume_train" || ms == "ResumeTrain")
                    mode = ReadoutLoadMode::ResumeTrain;
            }
            self.SetReadoutWeights(std::move(weights), mode);
        })
        .def("save_readout_hcnn_model",
             &WTF::SaveReadoutHcnnModel,
             py::arg("path_stem"),
             "Write portable stem.hcnw + stem.arch.json for the HCNN readout.")
        .def("load_readout_hcnn_model",
             [](WTF& self, const std::string& path_stem, const std::string& mode) {
                 ReadoutLoadMode m = ReadoutLoadMode::Eval;
                 if (mode == "resume_train" || mode == "ResumeTrain")
                     m = ReadoutLoadMode::ResumeTrain;
                 self.LoadReadoutHcnnModel(path_stem, m);
             },
             py::arg("path_stem"),
             py::arg("mode") = "eval",
             "Load stem.hcnw (+ arch sidecar) into the live readout.")
        .def("readout_arch_summary",
             &WTF::ReadoutArchSummary,
             "Human-readable HCNN readout architecture and parameter counts.")
        ;
}
