# HypercubeWTF

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()
[![CMake](https://img.shields.io/badge/CMake-3.21+-blue.svg)]()

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

Full integration guide: **[docs/CPP_SDK.md](docs/CPP_SDK.md)**.

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

Why the hypercube rather than a random reservoir graph? A few properties keep
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
- Full knobs and contracts: **[docs/CPP_SDK.md](docs/CPP_SDK.md)**.

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
| [examples/mnist/WhiteNoiseFilter.md](examples/mnist/WhiteNoiseFilter.md) | Noisy test fields: does the reservoir orbit help vs pack-only → CNN? |
| [examples/mnist/TrainingDataQualitySensitivity.md](examples/mnist/TrainingDataQualitySensitivity.md) | Degraded training data: how much does each path lose? |

The studies use MNIST on small cubes because it is handy to pack and run, not
because we are chasing digit accuracy. A more rigorous study is still needed
before treating any of those results as settled.

---

## Quick start

**Needs:** C++23, CMake ≥ 3.21. Prefer **Release** when comparing numbers
(Debug/Release float behavior can differ with this project’s fast-math flags).

### CLion

Open the project, reload CMake, build **Release**.

### Command line

With a C++23 toolchain on `PATH` (or CLion’s bundled MinGW/CMake):

```bash
git clone https://github.com/dliptak001/HypercubeWTF.git
cd HypercubeWTF
cmake --build cmake-build-release
```

(If the build directory does not exist yet, configure once from CLion or with
your usual CMake generator — this repo is primarily developed under CLion.)

| Binary | Role |
|--------|------|
| `wtf_smoke` | Episode contracts + small train/predict smoke |
| `wtf_synth` | Multi-class synthetic fields (no data files) |
| `wtf_mnist` | MNIST pack → orbit → readout (needs IDX files) |

### Use as a dependency

```cmake
add_subdirectory(path/to/HypercubeWTF)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeWTFCore)
target_include_directories(my_app PRIVATE path/to/HypercubeWTF)
```

```cpp
#include "WTF.h"

WTFConfig cfg;
cfg.reservoir.dim = 7;                 // N = 128
cfg.reservoir.history_depth = 8;
cfg.ic_seed = 2;
cfg.episode.T = 0;                     // → T = N
cfg.readout.num_outputs = 10;
cfg.readout.task = ReadoutTask::Classification;

WTF wtf(cfg);
// collect episodes → TrainOnCollected → Predict / PredictClass
```

---

## Examples (recipes)

Product knobs live in each demo’s `MakeWTFConfig()`; demo-only constants sit
beside them. Details: **[examples/README.md](examples/README.md)**.

| Program | What it is for | Data files? |
|---------|----------------|-------------|
| `wtf_smoke` | Contract tests (sizes, determinism, parallel collect, noise) | No |
| `wtf_synth` | Fast multi-class stack check | No |
| `wtf_mnist` | Real packing + larger train loop | Yes — `C:\HypercubeWTF\data\` (see examples README) |

```bash
cmake --build cmake-build-release --target wtf_synth wtf_smoke
# MNIST: place uncompressed *-ubyte IDX files under C:\HypercubeWTF\data\
cmake --build cmake-build-release --target wtf_mnist
```

---

## Repository map

```text
WTF.h / WTF.cpp              Product façade (collect → train → predict)
Reservoir.h / Reservoir.cpp  Frozen hypercube reservoir (WTF-owned fork)
Readout.h / Readout.cpp      Thin HypercubeCNN façade
third_party/HypercubeCNN/    Vendored HCNN pin (see VENDORED.md)
tests/wtf_smoke.cpp          Public contracts
examples/
  common/                    Optional packing / data-path helpers (not the product)
  synth/                     Synthetic multi-class demo
  mnist/                     MNIST demo + study write-ups
docs/CPP_SDK.md              Canonical product API guide
```

CMake library target: **`HypercubeWTFCore`**. Optional top-level targets:
`wtf_smoke`, `wtf_synth`, `wtf_mnist`.

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/CPP_SDK.md](docs/CPP_SDK.md)** | Product API — why explore, contracts, config, loop, pitfalls |
| [examples/README.md](examples/README.md) | Demo map and MNIST data notes |
| [examples/mnist/WhiteNoiseFilter.md](examples/mnist/WhiteNoiseFilter.md) | White-noise field study (MNIST test bed) |
| [examples/mnist/TrainingDataQualitySensitivity.md](examples/mnist/TrainingDataQualitySensitivity.md) | Training-set quality study (MNIST test bed) |
| [third_party/HypercubeCNN/VENDORED.md](third_party/HypercubeCNN/VENDORED.md) | Which HypercubeCNN release is vendored |

---

## Ecosystem

- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on streams; same cube + HCNN readout family.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** — cube-native conv stack; WTF’s trainable head.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
