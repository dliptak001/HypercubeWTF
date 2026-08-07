# HypercubeWTF

Python bindings for **HypercubeWTF** — static high-dimensional fields with **no**
intrinsic time, driven through a **frozen hypercube reservoir** for a short
orbit, then a [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) head on
the **end state only**. Same reservoir discipline as HypercubeESN; different
drive (one length-N field per sample, synthetic time, not a live stream).

You pack domain data onto `N = 2^dim` vertices on the host. The product loop is:

**collect episodes → train readout → predict**

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

## Installation

```bash
pip install hypercube-wtf
```

Pre-built wheels for Python **3.10–3.14** on Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64) when published. No compiler required for
wheels.

### From source

```bash
git clone https://github.com/dliptak001/HypercubeWTF.git
cd HypercubeWTF/python
pip install .
```

Requires Python 3.10+, a C++23 compiler, and CMake ≥ 3.20. On Windows with
CLion’s bundled MinGW, put that toolchain’s `bin` (and Ninja) on `PATH`, set
`CMAKE_GENERATOR=Ninja`, and point `CC`/`CXX` at MinGW gcc/g++. Then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

## Quick start

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
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
wtf.fit(fields, labels)  # collect episodes → train

print(f"train acc (collected): {wtf.accuracy_on_collected():.3f}")
print(wtf.predict_class(fields[0]), wtf.predict(fields[0]).shape)
```

## Features

- **Episode API** — `collect_episode` / `collect_episodes` → `train` → `predict` / `predict_class`
- **One-shot `fit`** — collect + train for static field batches
- **Hypercube dim 5–16** — N = 2^dim field length; T drive passes (default T = N)
- **HCNN readout** — same family as ESN; classification or regression
- **Collect parallelism** — bulk collect uses internal worker reservoirs
- **Persistence** — pickle, optional HCNW readout export
- **NumPy float32** — contiguous arrays in and out

## Examples

Runnable Python hosts (public API only) live under
[`python/examples/`](https://github.com/dliptak001/HypercubeWTF/tree/main/python/examples)
in the **git tree**; they are **not** installed by the wheel.

```bash
python python/examples/synthetic_classification.py
```

## Documentation

| Doc | |
|-----|--|
| [Python SDK](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/Python_SDK.md) | API reference (episode lifecycle, config, pickle) |
| [C++ SDK](https://github.com/dliptak001/HypercubeWTF/blob/main/docs/CPP_SDK.md) | Native product guide |
| [Project README](https://github.com/dliptak001/HypercubeWTF#readme) | Architecture framing |

Repository: [github.com/dliptak001/HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)

## License

Apache-2.0
