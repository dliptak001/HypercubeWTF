# HypercubeWTF

**HypercubeWTF** is for high-dimensional data that has no natural clock —
spectra, sensor frames, packed images, stills. Those are the same kinds of
static fields people usually feed a spatial CNN, an MLP, or a similar
feed-forward stack. Classical reservoir computing wants a stream: a new
low-dimensional sample each step, a state that evolves through real time. A
still field offers no such sequence — the pattern is already complete — so
HypercubeWTF repurposes the reservoir idea by inventing a short stretch of
synthetic time. It places your length-N field on a **frozen** hypercube
reservoir, drives a short orbit that encodes the field in the dynamics, and
trains a small [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) head
on the **end state only**. The CNN head never sees the original field — it sees
what the reservoir dynamics leave behind. Those dynamics are not a neutral pipe:
early work suggests they can filter, reshape, and otherwise transform the field
in ways a static pack-then-CNN path does not (see
[Early observations](#early-observations-exploratory) below).

That is the product idea: take a static field, encode it through a short stretch
of reservoir dynamics, and train a spatial readout on what remains.

This package is the **Python** surface for that product (`import hypercube_wtf`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)**.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
  <sub>One geometry. Topology-native intelligence.</sub>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
</p>

HypercubeWTF is an experiment in the **HypercubeAI** project — our quest to map
AI and ML strategies onto the hypercube as a computational substrate.

Why the hypercube? A few properties keep
showing up — and they explain why a frozen reservoir and a HypercubeCNN readout
fit together so cleanly:

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the reservoir’s output with
  zero geometric distortion, and the learned kernels exploit the same locality
  that generated the dynamics. The data never leaves the hypercube it was born
  on.

Each product in the family is a different architecture on that same foundation:

| Product | Natural data | Role of the hypercube |
|---------|--------------|------------------------|
| **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** | Low-dim **streams** over time | Frozen **reservoir** stepped each sample; multi-slice state → HypercubeCNN readout |
| **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** | Static patterns on the cube | Trainable **spatial** conv/pool on the cube (no recurrent reservoir) |
| **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** | Patterns / attractors | Associative memory dynamics on the cube |
| **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)** | Static high-dim fields (**no** intrinsic time) | Same **frozen hypercube reservoir** discipline as ESN, driven for a short orbit per sample, then HypercubeCNN on the **end state** |

---

## What does WTF stand for?

The design goal was simple: take the HypercubeESN idea — frozen reservoir,
trained head — and aim it at **data that has no time**. There was no lineage to
steal a name from, so the usual naming exercise followed. Nothing stuck. After a
few hours of “maybe this?” and “nah.”, the working monologue devolved to
**what the f\*\*\* do we call this project?**

So we called it that.

**HypercubeWTF**

The monologue won — and the brand gained a little personality :-)

---

## What is HypercubeWTF?

[HypercubeESN](https://github.com/dliptak001/HypercubeESN) processes
**temporal streams**. [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)
processes **spatial data** with a trainable conv stack on the cube. HypercubeWTF
also takes spatial data, but the conv stack is not first in line: each field
first passes through a **dynamical encoder** (the reservoir). The CNN head never
sees the original field; it sees an encoded end-of-orbit state produced by the
reservoir dynamics.

In classical reservoir computing (and in HypercubeESN):

- Recurrent weights are **frozen**
- Only a **readout** is trained
- Nonlinear dynamics expand and mix the drive into a rich state

WTF uses that same idea on a **still field**. There is no natural next sample,
so the library invents a short synthetic orbit: it re-addresses the same fixed
field over the cube for a number of passes, then samples **once** at the end.
Geometry and weights stay put; only the registration of the field moves.

Whether the dynamical encoding → CNN pipeline has **real product value** is
still an open question. Early studies suggest interesting transformational
behavior (see [Early observations](#early-observations-exploratory)).

---

## Pipeline

```text
x  (your length-N field — already on the cube, no natural time)
    │
    ▼
 frozen hypercube reservoir runs a short orbit
    │
    ▼
 end-of-orbit features → HypercubeCNN → logits / values
```

- Cube size from **dim** (N = 2<sup>dim</sup>; dim 5…16).
- Only the readout trains.
- In this package the everyday loop is:

  **`collect_episodes` → `train` → `predict` / `predict_class`**

  or the one-shot **`fit`** (collect + train).

Unlike HypercubeESN’s Python API, there is no stream of small samples over real
time and no next-step `fit` on a 1D signal. Each sample is one full field; the
“time” is the short synthetic orbit; the CNN only ever sees the state at the end.

Full method list and knobs:
**[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md)**.

---

## Early observations (exploratory)

The internal dynamics of this encoding appear to have some interesting
properties we have only lightly explored — for example filtering white noise
when present, acting closer to an identity map when noise is absent, and
reducing sensitivity to training-data quality when noise is present. Treat that
as early observation, not settled product behavior — the write-ups have the
details and how we ran them:

| Document | Question |
|----------|----------|
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/WhiteNoiseFilter.md) | Noisy test fields: does the reservoir orbit help vs pack-only → CNN? |
| [TrainingDataQualitySensitivity.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/TrainingDataQualitySensitivity.md) | Degraded training data: how much does each path lose? |

The studies use MNIST on small cubes because it is handy to pack and run, not
because we are chasing digit accuracy. A more rigorous study is still needed
before treating any of those results as settled. You can reproduce the same
ideas from Python with this package (pack fields yourself, then collect / train /
predict). The original write-ups and C++ demos that produced the numbers live
under
[`examples/mnist/`](https://github.com/dliptak001/HypercubeWTF/tree/main/examples/mnist).

---

## Installation

```bash
pip install hypercube-wtf
```

```python
import hypercube_wtf as hw
print(hw.__version__)
```

Package name on PyPI: **`hypercube-wtf`**. Import name: **`hypercube_wtf`**.
Main type: **`hw.WTF`**. Early **0.1.x** — APIs can still move.

When wheels are published: Python 3.10–3.14 on common Windows, Linux, and macOS
machines, no local compiler. Runtime dependency: NumPy only.

### From source

Building the extension yourself needs Python 3.10+, a C++23 compiler, and
CMake ≥ 3.20 (scikit-build + pybind11 pull the rest).

```bash
git clone https://github.com/dliptak001/HypercubeWTF.git
cd HypercubeWTF/python
pip install .
```

On Windows with CLion’s MinGW, put that compiler’s `bin` folder (and Ninja) on
your `PATH`, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.) Step-by-step toolchain notes:
[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md).

---

## Quick start

You bring each sample as a length-**N** float array (N = 2<sup>dim</sup>). How
you get there — pad an image, reshape a spectrum, invent a layout — is up to
you. This package does not pack 784 pixels or 300 bins for you.

Shapes that matter:

| Array | Shape | Notes |
|-------|-------|--------|
| `fields` | `(count, N)` | one length-N field per row |
| classification `labels` | `(count,)` | integer class indices |
| regression `targets` | `(count, num_outputs)` | float targets |

```python
import numpy as np
import hypercube_wtf as hw

dim = 7
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((200, N), dtype=np.float32)
labels = rng.integers(0, 4, size=200)

wtf = hw.WTF(
    dim=dim,
    history_depth=4,
    T=100,
    ic_seed=2,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
wtf.fit(fields, labels)  # collect_episodes + train

print(wtf.N, wtf.T, wtf.num_collected)
print(f"train sanity check: {wtf.accuracy_on_collected():.3f}")
print(wtf.predict_class(fields[0]), wtf.predict(fields[0]).shape)

wtf.save("model.pkl")
loaded = hw.WTF.load("model.pkl")
```

### Step by step (same loop, more control)

```python
wtf = hw.WTF(
    dim=7,
    readout_num_outputs=4,
    readout_task="classification",
)
wtf.collect_episodes(fields_train, labels_train)
wtf.train()
logits = wtf.predict(fields_test[0])       # (num_outputs,) float32
cls = wtf.predict_class(fields_test[0])    # int
```

For regression, set `readout_task="regression"` and pass float targets instead
of class labels. Then use `r2_on_collected()` the same way as the classification
sanity check.

`accuracy_on_collected` and `r2_on_collected` only look at the samples you
already trained on — they are a quick sanity check, not a test score. For real
evaluation, hold some fields out and call `predict` / `predict_class` yourself.

---

## Features

- **One class** — `hypercube_wtf.WTF` is the whole product surface
- **Episode loop** — `collect_episode` / `collect_episodes` → `train` →
  `predict` / `predict_class`
- **`fit`** — clear, collect, and train when your arrays are ready
- **dim 5–16** — field length N = 2<sup>dim</sup>; set orbit length with `T`,
  end-of-orbit ages with `readout_slices` (B)
- **Classification or regression** — `readout_task=...` fixed at construction
- **Bulk collect can parallelize** — `collect_threads` (0 = auto)
- **Train-only field noise** — `train_input_noise_sigma` on collect, never on
  predict
- **Skip-the-orbit path** — `bypass_reservoir=True` for pack-only comparisons
  (needs B = 1)
- **Inspect an episode** — `run_episode(x)` then `last_features()`
- **Save / load** — `save` / `load` (pickle: config + readout weights; collected
  samples are not stored). Optional `save_readout_hcnn_model` /
  `load_readout_hcnn_model` for portable HCNW + arch JSON
- **NumPy float32** — arrays converted for you; prefer contiguous float32

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
`pip install hypercube-wtf`. That is self-contained.

If you want a longer walk-through, the demo scripts on GitHub under
[`python/examples/`](https://github.com/dliptak001/HypercubeWTF/tree/main/python/examples)
are there to open or download — they are not added to your machine by pip.

| Script | What it is for |
|--------|----------------|
| [synthetic_classification.py](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/synthetic_classification.py) | Multi-class toy fields: `fit`, then train and test accuracy |

```bash
# from a clone of HypercubeWTF, after: pip install hypercube-wtf
python python/examples/synthetic_classification.py
```

These use easy made-up fields so the API is obvious — not scores to publish.
More notes:
[python/examples/README.md](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/README.md).

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md)** | Canonical Python API — every method, layout, pickle, limits |
| [python/examples/README.md](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/README.md) | Demo scripts on GitHub |
| [Project README](https://github.com/dliptak001/HypercubeWTF#readme) | Product story and C++ demos from the repo root |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/WhiteNoiseFilter.md) | Early white-noise study (MNIST as a test bed) |
| [TrainingDataQualitySensitivity.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/TrainingDataQualitySensitivity.md) | Early training-quality study (MNIST as a test bed) |

---

## Ecosystem

- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on streams; same cube + HCNN readout family.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** — cube-native conv stack; WTF’s trainable head.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](https://github.com/dliptak001/HypercubeWTF/blob/main/LICENSE).
