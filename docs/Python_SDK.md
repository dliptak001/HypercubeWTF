# HypercubeWTF Python SDK

Python bindings for the HypercubeWTF C++ library: a fixed Boolean-hypercube
reservoir driven for a short **episode** per static field, plus a trainable
HypercubeCNN readout on the **end state**, exposed as one `WTF` class.

Deep dive on the C++ core: [CPP_SDK.md](CPP_SDK.md).

Package version: **0.1.0** (`hypercube_wtf.__version__`). Early library —
APIs and defaults can move.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)

## Installation

### From PyPI (when published)

```bash
pip install hypercube-wtf
```

Import as `import hypercube_wtf as hw` (PyPI name `hypercube-wtf`).

### From source

Requirements: Python 3.10+, C++23 compiler, CMake 3.20+, scikit-build-core,
pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeWTF.git
cd HypercubeWTF/python
pip install .
```

On Windows with MinGW (e.g. CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

(from the `python/` directory after install, or point pytest at
`python/tests` from the repo root)

### Examples

For a first try, the [Quick start](#quick-start) below is enough after
`pip install`. Longer demos live on GitHub under
[`python/examples/`](../python/examples/README.md) (not part of the pip install).
From a clone of the repo:

```bash
python python/examples/synthetic_classification.py
```

## Quick start

```python
import numpy as np
import hypercube_wtf as hw

dim = 7
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((128, N), dtype=np.float32)
labels = rng.integers(0, 4, size=128)

wtf = hw.WTF(
    dim=dim,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
wtf.fit(fields, labels)

print(wtf.accuracy_on_collected())  # train-set only
print(wtf.predict_class(fields[0]))
```

### Explicit (full control)

```python
wtf = hw.WTF(dim=6, history_depth=8, T=64,
             readout_num_outputs=3, readout_task="classification")
wtf.collect_episodes(fields_train, labels_train)
wtf.train()
logits = wtf.predict(fields_test[0])
cls = wtf.predict_class(fields_test[0])
```

## Pipeline vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float32 vector on the cube (you pack domain data) |
| **Episode** | Reload frozen IC → drive field for T passes → pack end features |
| **Collect** | Run episode (optional train noise) → append features + label/target |
| **Train** | Batch-train HCNN on all collected episodes |
| **Predict** | Fresh episode (no train noise) + readout forward |

This is **not** HypercubeESN’s stream pipeline (no `reservoir_warmup` /
next-step `fit` on a 1D signal). Time here is synthetic and per-sample.

## API reference

### Constructor `WTF(dim, **kwargs)`

All knobs are fixed at construction (matches C++ `WTFConfig`).

| Group | Keyword | Default | Notes |
|-------|---------|---------|-------|
| Reservoir | `dim` | required | 5–16; N = 2<sup>dim</sup> |
| | `seed` | C++ default | Weight init |
| | `spectral_radius` | 0.999 | Target SR |
| | `input_scaling` | 0.02 | Drive strength |
| | `leak_rate` | 1.0 | |
| | `history_depth` | 16 | M |
| | `bias_scaling` | 0.003 | 0 = off |
| | `verbose` | False | Banner |
| Episode | `ic_seed` | 1 | Frozen s0 |
| | `T` | — | Drive passes; `0` expands to N at construct |
| | `readout_slices` | 1 | B (power of two ≤ M) |
| | `collect_threads` | 0 | 0 = auto |
| | `train_input_noise_sigma` | 0 | Collect only |
| | `bypass_reservoir` | False | Field → features (B=1) |
| Readout | `readout_num_outputs` | 1 | |
| | `readout_task` | `"regression"` | or `"classification"` |
| | `readout_epochs` | 200 | |
| | `readout_*` | (see `__init__` docstring) | Layers, LR, Adam, etc. |

### Methods

| Method | Role |
|--------|------|
| `run_episode(x)` | Drive one episode; updates `last_features()` |
| `last_features()` | B*N float32 from last serial episode path |
| `clear_collected()` | Drop training buffer |
| `collect_episode(x, target)` | Serial append one sample |
| `collect_episodes(fields, targets)` | Bulk parallel append |
| `fit(fields, targets, clear=True)` | clear + collect + train |
| `train()` | `TrainOnCollected` |
| `predict(x)` | `(num_outputs,)` float32 |
| `predict_class(x)` | int class index |
| `accuracy_on_collected()` | **Train-set** accuracy |
| `r2_on_collected()` | **Train-set** R² |
| `save` / `load` | Pickle config + readout weights |
| `save_readout_hcnn_model` / `load_readout_hcnn_model` | HCNW + arch JSON |

### Properties

`dim`, `N`, `T`, `B`, `M`, `feature_size`, `num_collected`, `num_outputs`,
`seed`, `ic_seed`, `spectral_radius`, `realized_spectral_radius`,
`input_scaling`, `leak_rate`, `history_depth`, `bias_scaling`,
`bypass_reservoir`, `collect_threads`, `train_input_noise_sigma`,
`readout_task`, `readout_best_epoch`, `verbose`.

## Input data layout

- **Fields** must be length **N** per sample. Shape `(count, N)` for bulk APIs.
- Host packing (MNIST → N, spectra → N, etc.) is **outside** this package.
- **Labels**: integer class indices in `[0, num_outputs)`.
- **Regression targets**: shape `(count, num_outputs)` float32.
- Exchange type: **float32** (auto-converted from other dtypes via NumPy).

## Model persistence

| Mechanism | What is stored | Collected episodes? |
|-----------|----------------|---------------------|
| `save` / pickle | Constructor config + readout weight blob | No |
| `save_readout_hcnn_model` | Portable HCNW + arch sidecar | No |

Pickle version is bumped when the serialized layout changes; newer libraries
reject unknown future versions with an upgrade message.

## Limitations

- One `WTF` instance is not thread-safe for concurrent public calls (bulk collect
  parallelism is internal only).
- `accuracy_on_collected` / `r2_on_collected` only look at samples you already
  trained on — a quick sanity check, not a test score. Hold fields out and call
  `predict` / `predict_class` for real evaluation.
- Early 0.1.x product — match C++ contracts in [CPP_SDK.md](CPP_SDK.md).

## Dependencies

Runtime: NumPy. Build: scikit-build-core, pybind11, C++23, CMake. HypercubeCNN
is vendored under `third_party/HypercubeCNN` and compiled into the extension.
