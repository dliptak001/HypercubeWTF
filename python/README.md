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
x  (length-N field, host-packed, no natural time)
    │
    ▼
 frozen hypercube reservoir runs a short orbit
    │
    ▼
 end-of-orbit features → HypercubeCNN → logits / values
```

- Cube size from **dim** (N = 2<sup>dim</sup>; dim 5…16).
- Only the readout trains.
- Product loop: **collect episodes → train readout → predict**.
- Full knobs and contracts: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md)**
  · **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/CPP_SDK.md)**.

This is **not** HypercubeESN’s stream pipeline (no warmup / next-step fit on a
1D signal). Time here is synthetic and per-sample: one full field in, one
end-state feature pack out.

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
before treating any of those results as settled. The campaigns themselves are
C++ demos under
[`examples/mnist/`](https://github.com/dliptak001/HypercubeWTF/tree/main/examples/mnist);
the Python package exposes the same product API for your own hosts.

---

## Installation

```bash
pip install hypercube-wtf
```

Import as `import hypercube_wtf as hw` (PyPI name `hypercube-wtf`). Pre-built
wheels for Python **3.10–3.14** on Windows (x64), Linux (x86_64, aarch64), and
macOS (x86_64, arm64) when published. No compiler required for wheels. NumPy is
the only runtime dependency.

### From source

Requires Python 3.10+, a C++23 compiler, and CMake ≥ 3.20.

```bash
git clone https://github.com/dliptak001/HypercubeWTF.git
cd HypercubeWTF/python
pip install .
```

On Windows with CLion’s bundled MinGW, put that toolchain’s `bin` (and Ninja) on
`PATH`, set `CMAKE_GENERATOR=Ninja`, and point `CC`/`CXX` at MinGW gcc/g++ —
exact install paths change with the CLion version. Then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

### Tests

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

---

## Quick start

Host packing is your job: every sample is a length-**N** float32 field
(`N = 2**dim`). The library does not invent a map from 784 pixels or 300 bins
onto the cube.

```python
import numpy as np
import hypercube_wtf as hw

# Host packs domain data to length N = 2**dim
dim = 7
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((200, N), dtype=np.float32)
labels = rng.integers(0, 4, size=200)

wtf = hw.WTF(
    dim=dim,
    history_depth=4,
    T=100,  # drive passes; 0 means T = N after construction
    ic_seed=2,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
# collect episodes → train readout → predict
wtf.fit(fields, labels)

print(f"train acc (collected set): {wtf.accuracy_on_collected():.3f}")
print(wtf.predict_class(fields[0]), wtf.predict(fields[0]).shape)
```

### Explicit lifecycle (same as C++ collect → train → predict)

```python
wtf = hw.WTF(
    dim=7,
    readout_num_outputs=4,
    readout_task="classification",
)
wtf.collect_episodes(fields_train, labels_train)
wtf.train()
logits = wtf.predict(fields_test[0])
cls = wtf.predict_class(fields_test[0])
```

`accuracy_on_collected` / `r2_on_collected` score the **training buffer**, not a
held-out test set. Evaluate test fields with `predict` / `predict_class` in the
host.

---

## Features

- **Episode API** — `collect_episode` / `collect_episodes` → `train` →
  `predict` / `predict_class` (mirrors C++ `WTF`)
- **One-shot `fit`** — clear + collect + train for static field batches
- **Hypercube dim 5–16** — N = 2<sup>dim</sup> field length; T drive passes
  (default T = N); optional multi-slice end features (B)
- **HCNN readout** — same family as ESN; classification or regression
- **Collect parallelism** — bulk collect uses internal worker reservoirs
- **Train-only field noise** — `train_input_noise_sigma` on collect, never on
  predict
- **Bypass path** — optional pack-only features (no orbit) for ablations
- **Persistence** — pickle (config + readout weights); optional HCNW readout
  export for C++ interop
- **NumPy float32** — contiguous arrays in and out

---

## Examples

Runnable hosts use only the public package — no CMake, no native example
binaries. They live in the **git tree** under
[`python/examples/`](https://github.com/dliptak001/HypercubeWTF/tree/main/python/examples)
and are **not** installed by the PyPI wheel.

| Script | What it is for |
|--------|----------------|
| [synthetic_classification.py](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/synthetic_classification.py) | Multi-class length-N fields → fit → train / test accuracy |

```bash
# clone HypercubeWTF, then from the repo root:
pip install hypercube-wtf   # or: pip install ./python
python python/examples/synthetic_classification.py
```

Onboarding demos only — easy synthetic fields, not storefront metrics. Index:
[python/examples/README.md](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/README.md).

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md)** | Python package — episode API, install, pickle, layout |
| **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/CPP_SDK.md)** | C++ product API — why explore, contracts, config, loop, pitfalls |
| [Project README](https://github.com/dliptak001/HypercubeWTF#readme) | Architecture framing (same story as this document’s intro) |
| [examples/README.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/README.md) | Demo map and MNIST data notes |
| [python/examples/README.md](https://github.com/dliptak001/HypercubeWTF/blob/main/python/examples/README.md) | Python host index |
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/WhiteNoiseFilter.md) | White-noise field study (MNIST test bed) |
| [TrainingDataQualitySensitivity.md](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/TrainingDataQualitySensitivity.md) | Training-set quality study (MNIST test bed) |
| [VENDORED.md](https://github.com/dliptak001/HypercubeWTF/blob/main/third_party/HypercubeCNN/VENDORED.md) | Which HypercubeCNN release is vendored |

---

## Ecosystem

- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on streams; same cube + HCNN readout family.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** — cube-native conv stack; WTF’s trainable head.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](https://github.com/dliptak001/HypercubeWTF/blob/main/LICENSE).
