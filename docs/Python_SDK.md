# HypercubeWTF Python SDK

Static fields have no natural clock. HypercubeWTF invents a short stretch of
synthetic time: a **frozen** Boolean-hypercube reservoir drives each length-N
field for a short **episode**, then a small [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)
readout trains only on the **end state**. One class — `WTF` — owns collect →
train → predict.

This is the **episode API**, not a stream API. There is no per-tick input
sequence and no next-step `fit` on a 1D signal (that is
[HypercubeESN](https://github.com/dliptak001/HypercubeESN)). Time here is
synthetic and **per sample**.

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.  
PyPI-facing package story: **[python/README.md](../python/README.md)**.  
Package version: single source `python/hypercube_wtf/_version.py`
(`hypercube_wtf.__version__` and wheel metadata both read it).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What an episode is](#what-an-episode-is)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (preferred)

Pre-built **wheels** — no compiler required:

```bash
pip install hypercube-wtf
```

Import as `import hypercube_wtf as hw` (PyPI name `hypercube-wtf`). Wheels cover
Python 3.10–3.14 on common Windows (x64), Linux (x86_64, aarch64), and macOS
(x86_64, arm64) builds. NumPy is the only runtime dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeWTF. The extension links the C++
core and vendored HypercubeCNN that sit **outside** the `python/` package
directory; a `python/`-only tree is not enough.

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+),
CMake 3.20+, scikit-build-core, pybind11, NumPy.

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

From the `python/` directory after install:

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

Or from the repository root: `pytest python/tests/ -v --import-mode=importlib`.
Importlib mode avoids the source tree shadowing the installed `_core` extension.

### Examples

The [Quick start](#quick-start) below is enough after `pip install`. Longer demos
live in the **git tree** under
[`python/examples/`](../python/examples/README.md) — they are **not** part of the
wheel. From a clone, repository root:

```bash
pip install hypercube-wtf   # or: pip install ./python
python python/examples/synthetic_classification.py
```

## Quick start

```python
import numpy as np
import hypercube_wtf as hw

dim = 7
N = 1 << dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((128, N), dtype=np.float32)
labels = rng.integers(0, 4, size=128, dtype=np.int32)

wtf = hw.WTF(
    dim=dim,
    seed=1,
    ic_seed=2,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
wtf.fit(fields, labels)

print(wtf.accuracy_on_collected())  # train-set only — not a test score
print(wtf.predict_class(fields[0]))
```

### Explicit (full control)

```python
wtf = hw.WTF(
    dim=6,
    history_depth=8,
    T=64,
    readout_num_outputs=3,
    readout_task="classification",
)
wtf.collect_episodes(fields_train, labels_train)
wtf.train()
logits = wtf.predict(fields_test[0])   # shape (num_outputs,)
cls = wtf.predict_class(fields_test[0])
```

`fit` is `clear_collected` (optional) → `collect_episodes` → `train`. Prefer
`fit` for a first pass; use collect/train when you append batches or retrain
without re-driving every field.

## What an episode is

```text
x  (length-N field, host-packed)
    │
    ▼
 reload frozen IC  →  drive field for T passes  →  pack B end ages
    │
    ▼
 features (B×N)  →  HypercubeCNN  →  logits / values
```

- **N = 2^dim** vertices / field length (dim 5…16).
- Reservoir weights are **frozen** after construction; only the readout trains.
- **Predict** always runs a **fresh** episode (no collect-time train noise).
- Host packing (MNIST → N, spectra → N, …) is **your** problem — this package
  does not reshape domain data onto the cube.

The CNN head never sees the original field; it sees what the orbit leaves behind.

## Pipeline vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float32 vector on the cube (you pack domain data) |
| **Episode** | Reload frozen IC → drive field for T passes → pack end features |
| **Collect** | Run episode (optional train noise) → append features + label/target |
| **Train** | Batch-train HCNN on all collected episodes |
| **Predict** | Fresh episode (no train noise) + readout forward |
| **N** | Neurons / field length = 2^dim |
| **M** | `history_depth` — delay-line depth |
| **B** | `readout_slices` — ages packed into features (power of two, 1 ≤ B ≤ M) |
| **T** | Drive-pass count per episode (`T=0` expands to N at construction) |

Not HypercubeESN’s stream pipeline: no `reservoir_warmup`, no next-step `fit` on
a 1D signal.

## API reference

### Constructor `WTF(dim, **kwargs)`

All knobs are fixed at construction (same contract as C++ `WTFConfig`).

```python
import hypercube_wtf as hw

wtf = hw.WTF(
    dim=7,                         # required; 5–16
    seed=7934791766227647176,      # reservoir weight init
    spectral_radius=0.999,
    input_scaling=0.02,
    leak_rate=1.0,
    history_depth=16,              # M
    verbose=False,
    bias_scaling=0.003,
    ic_seed=1,                     # frozen episode IC (not weight seed)
    T=100,                         # drive passes; 0 → N
    readout_slices=1,              # B
    collect_threads=0,             # 0 = auto
    train_input_noise_sigma=0.0,   # collect only
    bypass_reservoir=False,        # field → features if True (needs B=1)
    readout_num_outputs=1,
    readout_task="regression",     # or "classification"
    # … readout_* kwargs below
)
```

#### Reservoir and episode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Hypercube dimension **[5, 16]**. N = 2^dim. |
| `seed` | `int` | `7934791766227647176` | Reservoir weight-init seed (matches C++). |
| `spectral_radius` | `float` | `0.999` | Target spectral radius for recurrent weights. |
| `input_scaling` | `float` | `0.02` | Input drive coefficient. |
| `leak_rate` | `float` | `1.0` | Leaky integrator; 1.0 = full replacement. |
| `history_depth` | `int` | `16` | Delay-line depth **M ∈ [1, 64]**. |
| `verbose` | `bool` | `False` | Reservoir construction banner. |
| `bias_scaling` | `float` | `0.003` | Per-neuron bias after tanh; 0 disables. |
| `ic_seed` | `int` | `1` | Frozen episode IC seed (separate from `seed`). |
| `T` | `int` | `100` | Drive-pass count; **`0` expands to N** after construction. |
| `readout_slices` | `int` | `1` | B ages packed into features (power of two, ≤ M). |
| `collect_threads` | `int` | `0` | Bulk collect workers: 0 = auto, 1 = serial, K = K workers. |
| `train_input_noise_sigma` | `float` | `0.0` | Gaussian σ on the field during **collect only** (not predict). |
| `bypass_reservoir` | `bool` | `False` | Skip orbit; features are the packed field (requires B = 1). |

#### Readout (HCNN)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `readout_num_outputs` | `int` | `1` | Classes (classification) or regression width. |
| `readout_task` | `str` | `"regression"` | `"regression"` or `"classification"`. |
| `readout_num_layers` | `int` | `1` | Conv(+Pool) stages. **`0` = auto** `min(dim−2, 2)`. |
| `readout_conv_channels` | `int` | `16` | Base channel count for the first conv. |
| `readout_epochs` | `int` | `200` | Batch-train epochs. |
| `readout_batch_size` | `int` | `32` | Mini-batch size. |
| `readout_lr_max` | `float` | `0.0015` | Cosine peak LR. Keep ≤ ~0.005 to avoid NaN. |
| `readout_lr_min_frac` | `float` | `0.01` | Floor = `lr_max * lr_min_frac`. |
| `readout_lr_decay_epochs` | `int` | `0` | Cosine horizon; `0` = use `readout_epochs`. |
| `readout_weight_decay` | `float` | `0.0` | L2 on CNN weights. |
| `readout_momentum` | `float` | `0.9` | SGD momentum; ignored under the default Adam optimizer. |
| `readout_activation` | `str` | `"tanh"` | `"tanh"`, `"relu"`, `"leaky_relu"`, or `"none"`. |
| `readout_seed` | `int` | `42` | CNN weight-init seed. |
| `readout_num_threads` | `int` | `0` | HCNN workers: 0 = auto, 1 = single-threaded. |
| `readout_restore_best_epoch` | `bool` | `True` | Restore best-epoch weights after batch train. |
| `readout_best_epoch_holdout_frac` | `float` | `0.0` | Tail hold-out for best-epoch scoring; 0 = full train set. |
| `readout_use_pooling` | `bool` | `True` | Antipodal pool after each conv. |

**Not bound in Python yet** (C++ `ReadoutConfig` only): optimizer choice (C++
default Adam), pool type, channel growth, batch-norm. C++ defaults apply.

### Methods

| Method | Role |
|--------|------|
| `run_episode(x)` | Drive one episode (or bypass copy). Updates `last_features()`. |
| `last_features()` | Length B×N float32 from the last path that writes the **primary** buffer: `run_episode`, serial `collect_episode`, `predict`, or `predict_class`. **Not** updated by bulk `collect_episodes` (those features go only into the training set). |
| `clear_collected()` | Drop the batch training buffer. |
| `collect_episode(x, target)` | Serial append one sample (label or regression vector). |
| `collect_episodes(fields, targets)` | Bulk parallel append. |
| `fit(fields, targets, *, clear=True)` | Optional clear → collect → train. Returns `self`. |
| `train()` | Batch-train HCNN on all collected episodes. Does not clear the set. |
| `predict(x)` | Fresh episode + forward → shape `(num_outputs,)` float32. |
| `predict_class(x)` | Fresh episode + argmax class (classification task only). |
| `accuracy_on_collected()` | Accuracy on the **collected training set** only. |
| `r2_on_collected()` | R² on the **collected training set** only. |
| `save(path)` / `load(path)` | Pickle constructor config + readout weights. |
| `save_readout_hcnn_model(path_stem)` | Portable `stem.hcnw` + `stem.arch.json`. |
| `load_readout_hcnn_model(path_stem, *, mode="eval")` | Load HCNW into this instance (`"eval"` or `"resume_train"`). |
| `readout_arch_summary()` | Human-readable HCNN architecture and parameter counts. |

### Properties

| Property | Meaning |
|----------|---------|
| `dim`, `N`, `T`, `B`, `M` | Geometry and episode knobs (N = 2^dim, B = readout slices, M = history depth) |
| `feature_size` | B × N floats per sample / `last_features` |
| `num_collected` | Episodes in the batch training buffer |
| `num_outputs` | Readout width |
| `seed`, `ic_seed` | Weight seed vs frozen IC seed |
| `spectral_radius`, `realized_spectral_radius` | Target vs post-rescale estimate |
| `input_scaling`, `leak_rate`, `history_depth`, `bias_scaling` | Reservoir config mirrors |
| `bypass_reservoir`, `collect_threads`, `train_input_noise_sigma` | Episode / collect mirrors |
| `readout_task` | `"regression"` or `"classification"` |
| `readout_best_epoch` | 1-based best epoch after restore; else 0 |
| `verbose` | Construction banner flag |

## Input data layout

- **Fields** must be length **N** per sample. Prefer shape `(count, N)` for bulk
  APIs; a flat length `count * N` vector is also accepted.
- **Host packing** (images, spectra, sensors → N) is outside this package.
- **Classification labels**: integer class indices in **`[0, num_outputs)`**
  (enforced at collect). Shape `(count,)` for bulk collect / fit.
- **Regression targets**: shape `(count, num_outputs)` float32 (or flat
  `count * num_outputs`).
- Single-sample methods accept any array that ravel-flattens to the right length.

## Data types

| Role | Preferred type | Notes |
|------|----------------|-------|
| Fields / features / predictions | `float32` | Other dtypes converted via NumPy to contiguous float32 |
| Class labels | `int32` (or Python `int`) | Must be in `[0, num_outputs)` at collect (C++ enforces) |
| Bool as a class label | rejected on serial collect | `collect_episode` raises `TypeError`; use an integer index. Bulk `collect_episodes` coerces via int32 (do not rely on bool labels). |

## Error handling

Python-side checks raise `ValueError` or `TypeError` with a short message (bad
`dim`, task string, activation, field shape, label count, …). Native
`std::invalid_argument` maps to `ValueError`; other C++ failures typically
surface as `RuntimeError` via pybind11.

Typical mistakes:

- Field length ≠ N
- Bulk `fields` / `targets` row counts disagree
- Class label outside `[0, num_outputs)`
- `predict_class` / `accuracy_on_collected` on a regression model
- Calling `train` or `accuracy_on_collected` with an empty collected set
- `bypass_reservoir=True` with B ≠ 1 (rejected at construction in C++)

## Model persistence

| Mechanism | What is stored | Collected episodes? |
|-----------|----------------|---------------------|
| `save` / `pickle` | Constructor config + readout weight blob | **No** (`num_collected` is 0 after load) |
| `save_readout_hcnn_model` | Portable HCNW + arch sidecar | **No** |

Pickle version is bumped when the serialized layout changes; newer libraries
reject unknown future versions with an upgrade message.

```python
wtf.save("model.pkl")
wtf2 = hw.WTF.load("model.pkl")   # same ctor knobs + weights; empty collect buffer

wtf.save_readout_hcnn_model("export/stem")   # stem.hcnw + stem.arch.json
# Target instance must build a matching HCNN input shape / task (same dim, B,
# M, and readout_* architecture knobs as the exporter — not only dim/outputs).
wtf3 = hw.WTF(
    dim=wtf.dim,
    history_depth=wtf.history_depth,
    readout_slices=wtf.B,
    readout_num_outputs=wtf.num_outputs,
    readout_task=wtf.readout_task,
    # plus any non-default readout_num_layers / channels / pooling / …
)
wtf3.load_readout_hcnn_model("export/stem", mode="eval")
```

Prefer `save` / `load` when you want a full Python round-trip of the product
config. Prefer HCNW when you need a portable HypercubeCNN weight export.

**Security:** `load` uses `pickle.load`. Never load untrusted files.

## Limitations

- One `WTF` instance is **not thread-safe** for concurrent public calls from
  multiple host threads. Bulk collect parallelism is internal only.
- `accuracy_on_collected` / `r2_on_collected` only score samples you already
  collected (and typically trained on). Hold fields out and call `predict` /
  `predict_class` for real evaluation.
- `last_features()` is not updated by bulk `collect_episodes` (see Methods).
- A few readout knobs remain C++-only (optimizer, pool type, channel growth,
  batch-norm); see constructor tables above.
- Native contracts, episode mechanics, and host integration detail:
  **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake ≥ 3.20, scikit-build-core, pybind11 |

The HypercubeCNN readout is built into the extension — no separate HCNN package.
