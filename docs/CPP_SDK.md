# HypercubeWTF C++ SDK

You choose a cube dimension **dim** (an integer from 5 to 16). That fixes the
field length **N = 2<sup>dim</sup>** — for example dim 7 → N = 128, dim 10 →
N = 1024. You place a fixed pattern on those vertices (an image pack, a
spectrum, or any length-N floats you built yourself). HypercubeWTF **drives
that field through a frozen reservoir** for a short synthetic orbit, then
**trains a small CNN only on the state at the end**. One class does the whole
loop: collect episodes, train the head, predict.

You do not need to learn HypercubeESN or HypercubeCNN first. Link
**`HypercubeWTFCore`**, include **`WTF.h`** (or **`HypercubeWTF.h`**), and work
with **`WTF`**. Demos and packing helpers are optional recipes; they are not the
product.

This guide matches the public headers for **0.1.x** (early development — expect
change). For goals and locked design choices, see **[project.md](project.md)**.

**Who it is for:** anyone embedding WTF in a host (collect → train → predict),
and anyone learning the stack with the same API the demos use.

**What you get:** a C++23 static library. Headers sit at the repo root. A
vendored HypercubeCNN builds the trainable readout; hosts usually never call
HCNN themselves.

---

## 1. The big picture

Most learning systems either see a **stream** (one small input every step) or a
**static pattern** (classify an image once). WTF sits in between:

1. You give it one full-length field (which may be an image) on the cube.
2. It **re-addresses that same field** for `T` passes (a synthetic orbit).
3. It takes a **single snapshot at the end** and trains a CNN on those features.

The reservoir weights never train. Only the readout does. That is the whole
product idea.

| Library | You typically feed it… | What runs |
|---------|------------------------|-----------|
| HypercubeESN | a **stream** of small inputs over real time | step → state → HCNN |
| HypercubeCNN | a **static** pattern already on the cube | conv stack → labels |
| **HypercubeWTF** | a **static** length-N field with no real time | orbit → **end state** → HCNN |

### Your data does not have to be a power of two

**dim** is the only size knob for the cube: it is set on
`ReservoirConfig::dim` and yields **N = 2<sup>dim</sup>** vertices. The field
WTF accepts is always that length — geometry, not a style preference.

If your raw data is 784 pixels or 300 bins, **you** map it onto N floats first
(pad, resize, spatial embed, custom layout — your choice). WTF does not invent
that map. Demo helpers under `examples/common/` are one MNIST-oriented recipe;
skip them when you pack your own way.

### What freezes vs what learns

| Piece | Trains? |
|-------|---------|
| Reservoir weights and bias | No — drawn once at construct |
| Starting state **s0** (full delay line) | No — drawn once from `ic_seed`, reloaded every episode |
| How you pack domain data into the field | Your problem (outside WTF) |
| HCNN readout | **Yes** |

### Optional shortcut: skip the orbit

Set `bypass_reservoir` and the features are simply the field itself (requires
`B == 1`). Same collect / train / predict API — useful for A/B against “just
the pack.”

---

## 2. One episode, step by step

Think of an **episode** as: reset to a known start, drive for a while, read once.

That sentence is the same rhythm as classical reservoir computing (echo-state /
ESN style), just aimed at a **static** field instead of a live stream.

### Where the dynamical magic comes from

A reservoir is a **fixed nonlinear dynamical system**. Weights are drawn once
(here: recurrent cube edges, input gather `W_in`, optional bias), scaled so the
recurrent part sits near a chosen **spectral radius**, and then left alone.
You never backprop through that core. Training only fits a thin head on top of
the state — in WTF, a HypercubeCNN readout.

Each **step** does two things that matter:

1. **Drive** — the current field pattern is injected through `W_in` so every
   vertex feels a local mix of the input (strength set by `input_scaling`).
2. **Recur** — each vertex updates from itself and its cube neighbors (bit-flip
   edges), usually through a nonlinearity (tanh-family), with optional **leak**
   so the state blends old and new rather than replacing fully.

Do that many times and the high-dimensional state becomes a **nonlinear
trajectory** shaped by both the input history and the fixed graph. The “echo”
idea is that recent drive still rings in the state while older influence fades —
if the spectral radius and leak are in a sensible regime, trajectories from
different starts stay distinguishable without exploding. WTF reloads the **same
frozen s0** every episode so two runs of the same field match bit-for-bit
(modulo your packing).

### Making time when you only have a still picture

A classical ESN expects a **movie**: at time 0 a small input `u_0`, at time 1
`u_1`, and so on. Each tick, fresh information arrives from the outside world.
The reservoir’s job is to remember and mix that stream.

WTF usually has the opposite problem: one **still** — a full length-N field
(image pack, spectrum, whatever) with no natural “next frame.” If you only
shoved that field in once and stepped forever, most of the pattern would be a
one-shot kick; the dynamics would not systematically walk the spatial structure.

So WTF invents a clock from space. Call the pass counter `c` (starts at 0 each
episode). On pass `c`, every vertex `v` is driven by field sample

```text
x[(v XOR c) & (N − 1)]
```

Read that as: **the numbers in `x` never change**; you only change **which
number sits on which vertex**. XOR with `c` is a fixed, invertible shuffle of
addresses on the cube. Increment `c`, shuffle again. Geometry (who is neighbor
to whom) and all frozen weights stay put — they do not slide with `c`. What
moves is the **registration** of the field onto the graph.

Do that for `T` passes and the reservoir experiences a **synthetic time
series**: not new pixels from a camera, but the same global pattern seen under
`T` successive addressings. That orbit is what the echo digests.

You still follow RC discipline at the end: you do **not** train on every
intermediate state. After the last pass you read **once** — the delay line’s
newest ages (`B` of them, often just `B = 1`). That end pack is the feature
row: a nonlinear, dynamical summary of “this whole field, driven through this
orbit, starting from this s0.” Different fields (or different packs of the same
domain data) leave different end signatures; the CNN only has to separate those.

So the product hinge in one line: **spatial pattern in → fake time by
re-addressing → real reservoir dynamics → one end snapshot → trained reader.**
The magic is not a learned recurrent net. It is **rich fixed dynamics + a
trained head**. Bypass mode (`bypass_reservoir`) skips the orbit and feeds the
field straight to the CNN — the control question “did the reservoir actually
help?”

### Mechanics in order

1. Reload the frozen initial condition **s0** into the delay line.
2. For pass `c = 0, 1, …, T−1`:
   - Place the field on the cube with address offset `c`.
   - One reservoir step (field inject + `W_in` gather + recurrent update).
3. Build features from the **end** of the delay line only: ages `0 … B−1`
   concatenated → length `B × N`.
4. Hand those features to the readout (collect, train, or predict).

```text
x  (length N, fixed for this episode)
    │
    ▼
 Load s0 → drive T times → pack B end ages → features (B×N)
    │
    ▼
 HCNN readout → class logits or regression values
```

### Words you will see in the API

| Word | Plain meaning |
|------|----------------|
| **dim** | Cube dimension you choose (5…16); set as `reservoir.dim` |
| **N** | Field length = 2<sup>dim</sup> (also one reservoir state slice) |
| **T** | How many drive passes (`episode.T = 0` means “use N”) |
| **B** | How many end delay-line ages go into the feature vector (`readout_slices`) |
| **M** | Delay-line depth (`history_depth`) |
| **s0** | Frozen start state, length `N × M`, from `ic_seed` (not the weight seed) |
| **FeatureSize** | `B * N` — what the readout eats |

`B` must be a power of two and no larger than `M`. Default is `B = 1` (newest
slice only).

---

## 3. What is the product (and what is not)

Keep this map in mind when you open the tree:

| You care about… | Use… |
|-----------------|------|
| Integrating the library | **`WTF`** + **`WTFConfig`** |
| Logging realized spectral radius, saving readout weights | `wtf.reservoir()` / `wtf.readout()` |
| Learning by example | `wtf_smoke`, `wtf_synth`, `wtf_mnist` |
| MNIST paths / packing demos | `examples/common/` (optional) |
| Raw HypercubeCNN | Almost never — that lives under the readout |

```text
HypercubeWTF.h     umbrella → WTF.h
WTF.h              front door (WTF, WTFConfig, EpisodeConfig)
Reservoir.h        ReservoirConfig (+ Reservoir for inspection)
Readout.h          ReadoutConfig, enums, Readout
… .cpp files …
third_party/HypercubeCNN/    vendored; see VENDORED.md
examples/                    demos, not the SDK definition
docs/CPP_SDK.md              this guide
docs/project.md              charter
```

Link **`HypercubeWTFCore`** (it pulls **`HypercubeCNNCore`** for you).

### Rules that matter

These are product contracts, not “implementation details.”

- **Every field is length N.** Wrong size throws.
- **You pack; WTF drives.** No built-in image layout.
- **Reservoir and s0 freeze at construct.** Every episode reloads the same s0.
- **Only the end state** goes to the readout (optional multi-age pack `B`).
- **Train noise is collect-only.** `Predict` / `RunEpisode` stay clean.
- **`Predict` returns raw logits** (or regression values) — no softmax.
- **`AccuracyOnCollected` / `R2OnCollected` are training-set scores**, not held-out.
- **One `WTF` per thread of control.** Bulk collect parallelizes *inside* one call;
  do not call public methods concurrently on the same object.
- **`WTF` is not copyable.** Prefer exclusive ownership (move exists if you need it).

### The loop you will write

```text
fill WTFConfig
construct WTF once
collect many episodes   (optional train-input noise)
TrainOnCollected
Predict / PredictClass  (always a fresh clean episode)
```

Optional extras: `RunEpisode` + `LastFeatures`, train-set metrics, `ClearCollected`,
bypass A/B, `collect_threads` for faster bulk collect.

---

## 4. Build and consume

You need **C++23** and **CMake ≥ 3.21**. Prefer **Release** when you care about
study numbers (Debug and Release float behavior can differ with the project’s
fast-math flags).

In CLion: open the project, reload CMake, build. From a shell with the toolchain
available:

```bash
cmake --build cmake-build-release
```

When this repo is the top-level project you also get:

| Binary | Role |
|--------|------|
| `wtf_smoke` | Fast contract + train smoke |
| `wtf_synth` | Multi-class synthetic fields (no data files) |
| `wtf_mnist` | MNIST recipe (IDX files under `C:\HypercubeWTF\data`) |

If you pull HypercubeWTF in as a **subdirectory**, demos are skipped; you still
get the library.

```cmake
add_subdirectory(path/to/HypercubeWTF)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeWTFCore)
target_include_directories(my_app PRIVATE path/to/HypercubeWTF)
```

```cpp
#include "HypercubeWTF.h"   // or "WTF.h"
```

---

## 5. First program

A tiny two-class example — collect, train, predict:

```cpp
#include "HypercubeWTF.h"
#include <cstdio>
#include <vector>

int main() {
    WTFConfig cfg;
    cfg.reservoir.dim = 5;              // N = 32
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.seed = 1;
    cfg.ic_seed = 2;
    cfg.episode.T = 0;                  // → T = N
    cfg.episode.readout_slices = 1;     // B = 1
    cfg.readout.dim = 0;                // auto
    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 80;
    cfg.readout.num_threads = 1;
    cfg.readout.restore_best_epoch = false;

    WTF wtf(cfg);
    const size_t N = wtf.N();

    auto field = [&](int label) {
        std::vector<float> x(N, 0.f);
        const float s = (label == 0) ? 1.f : -1.f;
        for (size_t i = 0; i < N / 2; ++i)
            x[i] = s * (0.2f + 0.8f * float(i) / float(N));
        return x;
    };

    for (int i = 0; i < 24; ++i) {
        wtf.CollectEpisode(field(0), 0);
        wtf.CollectEpisode(field(1), 1);
    }
    wtf.TrainOnCollected();

    std::printf("train acc=%.3f  pred0=%d pred1=%d\n",
                wtf.AccuracyOnCollected(),
                wtf.PredictClass(field(0)),
                wtf.PredictClass(field(1)));
}
```

**Habits that save pain later**

1. Build the config, construct **one** `WTF` (weights and s0 freeze here).
2. Collect a dataset, then train. You can call `TrainOnCollected` again without
   clearing if you want another pass on the same features.
3. Treat `Predict` / `PredictClass` as clean inference — they never apply collect
   noise.
4. Keep packing in your code (or a demo helper). The core only accepts length-N
   fields.

---

## 6. The API you actually use

### Config at a glance

Everything interesting is set **before** `WTF` is constructed.

```cpp
struct EpisodeConfig {
    size_t T = 0;                         // 0 → use N
    size_t readout_slices = 1;            // B
    size_t collect_threads = 0;           // 0 = auto (leave OS/UI some cores)
    float  train_input_noise_sigma = 0.f; // collect only
    bool   bypass_reservoir = false;      // needs B == 1
};

struct WTFConfig {
    ReservoirConfig reservoir{};
    ReadoutConfig   readout{};
    EpisodeConfig   episode{};
    uint64_t        ic_seed = 1;          // s0 only
};
```

**Reservoir** (frozen dynamics) — common knobs:

| Field | Meaning | Typical demos |
|-------|---------|---------------|
| `dim` | Cube dimension; N = 2<sup>dim</sup> (5…16) | 7–10 |
| `seed` | Weight draws | fixed |
| `spectral_radius` | Target for recurrent rescale | ~0.999 |
| `leak_rate` | Mix each step; in (0, 1] | 1 |
| `input_scaling` | How hard the field drives | ~0.02–0.03 |
| `history_depth` | M (1…64) | 4–16 |
| `bias_scaling` | Bias strength; 0 = off | ~0.003 |
| `verbose` | Construction printout | false |

**Readout** (trainable head) — knobs most hosts touch:

| Field | Meaning |
|-------|---------|
| `dim` | Feature cube dim; **0 = auto** (`reservoir.dim` + log₂(B)) |
| `num_outputs` | Classes, or regression width |
| `task` | `Classification` or `Regression` |
| `epochs`, `batch_size` | Batch training |
| `lr_max`, `lr_min_frac`, `lr_decay_epochs` | Cosine learning-rate schedule |
| `num_threads` | HCNN workers — use **1** for simple determinism |
| `restore_best_epoch` | Keep best epoch weights (default true on Readout; demos often false) |
| `seed` | Readout weight init |

Deeper architecture fields (`num_layers`, pooling, activation, …) live on
`ReadoutConfig` in `Readout.h`. Classification hosts often leave them alone.

### After construct — sizes and inspection

```cpp
explicit WTF(const WTFConfig& cfg);

wtf.N();  wtf.T();  wtf.B();  wtf.M();
wtf.FeatureSize();      // B * N
wtf.NumCollected();
wtf.NumOutputs();
wtf.CollectThreads();   // what you configured (0 = auto)
wtf.BypassReservoir();

wtf.reservoir();        // const — e.g. realized spectral radius
wtf.readout();          // const — weights, save/load helpers
wtf.readout_config();   // resolved config (dim filled in if auto)
```

Construction checks the usual mistakes: bad T, B not a power of two, B > M,
bypass with B ≠ 1, invalid noise σ, wrong readout dim, `num_outputs < 1`.

### Run an episode (no training)

```cpp
wtf.RunEpisode(x);              // x.size() == N; x is not modified
auto feats = wtf.LastFeatures(); // length FeatureSize()
```

`LastFeatures` also updates after serial `CollectEpisode`, `Predict`, and
`PredictClass`. **Bulk `CollectEpisodes` does not touch it** — those features
only go into the training set.

### Collect, train, predict

**Classification**

```cpp
wtf.CollectEpisode(x, class_label);           // one sample
wtf.CollectEpisodes(fields_flat, labels);     // bulk, sample-major
wtf.CollectEpisodes(count, labels, fill_field); // fill_field(i, span) writes N floats
```

**Regression** — same idea with target vectors (`num_outputs` floats per sample).

```cpp
wtf.ClearCollected();           // drop training rows (keeps worker pool)
wtf.TrainOnCollected();         // needs at least one sample; does not clear

auto logits = wtf.Predict(x);   // num_outputs floats
int y = wtf.PredictClass(x);    // classification only

double acc = wtf.AccuracyOnCollected();  // train set
double r2  = wtf.R2OnCollected();        // train set, regression
```

Bulk layout notes:

- `fields_flat` is sample-major: sample `i` starts at `i * N`.
- `fill_field` may run on several samples at once — only touch index `i`’s buffer.
- Wrong task (class API on a regression net, etc.) throws.

### Train-input noise

Set `episode.train_input_noise_sigma > 0` to add Gaussian noise to the field
**only when collecting**. The draw is deterministic from `ic_seed` and the sample
index. Inference paths ignore σ. Name is deliberate: this is not eval noise.

### Faster bulk collect

`episode.collect_threads`:

- `0` — auto (leaves one or two cores free so the machine stays responsive)
- `1` — serial
- `K` — up to K workers

Worker 0 reuses the primary reservoir. Extra workers clone frozen weights once.
The internal thread pool **grows** for the life of the `WTF` and does not shrink.

### Bypass reservoir

`bypass_reservoir = true` (with `B == 1`): features = the field. Collect noise
still applies when σ > 0. Handy for “does the orbit help?” experiments.

---

## 7. Please do not

| Temptation | Better path |
|------------|-------------|
| Drive `Reservoir` yourself for product training | Use `WTF` episodes |
| Call vendored `hcnn::HCNN` from the app | Let `Readout` own it; export via `wtf.readout()` if needed |
| Depend on `examples/common` in production | Copy the idea; own your packing |
| Stream intermediate passes into the readout | Product samples **end of episode only** |
| Expect ESN-style external feedback | Not ported on purpose |

`Reservoir` and `Readout` headers are public so config and inspection work. The
happy path is still collect → train → predict on **`WTF`**.

---

## 8. Demos as recipes

Demos keep product knobs in `MakeWTFConfig()` and demo-only constants (`k*`)
beside them:

```text
config → WTF → collect → train → score → predict
```

| Demo | When to open it |
|------|-----------------|
| `tests/wtf_smoke.cpp` | Contracts: sizes, determinism, parallel collect, noise, bypass |
| `examples/synth/wtf_synth.cpp` | Fast multi-class without data files |
| `examples/mnist/wtf_mnist.cpp` | Real packing + larger train loop |

More context: [`examples/README.md`](../examples/README.md).

---

## 9. Threads, memory, and cost (short)

- Treat one `WTF` as **exclusive** for public calls.
- Bulk collect is where parallelism belongs; it clones reservoirs as needed.
- Orbit cost grows with `T × N`; features are `B × N`; the CNN scales with its
  own dim and channels.
- If you already run many `WTF` instances in parallel, set
  `readout.num_threads = 1` so HCNN does not oversubscribe the machine.
- Prefer Release when comparing accuracies across runs.

---

## 10. Common mistakes

| Symptom / assumption | Fix |
|----------------------|-----|
| Throw on collect / run | Field length must equal `wtf.N()` |
| “Why can’t I pass 784 floats?” | Pack to N first |
| Softmax inside `Predict` | You get logits; use `PredictClass` or argmax |
| Eval looks noisy after setting σ | σ is collect-only |
| `LastFeatures` empty after bulk collect | Expected — re-run `RunEpisode` if you need them |
| Great train accuracy, bad real test | `AccuracyOnCollected` is the training set |
| Construct fails on B | Power of two, and `B ≤ M` |
| Construct fails on bypass | Needs `B == 1` |
| Racey results with shared `WTF` | One instance, one host thread of control |
| Linked HypercubeCNN only | Link `HypercubeWTFCore`, include `WTF.h` |

---

## 11. Further reading

| Doc | What it is |
|-----|------------|
| [project.md](project.md) | Charter — why WTF exists, locked design |
| [examples/README.md](../examples/README.md) | Demo map and MNIST data notes |
| [VENDORED.md](../third_party/HypercubeCNN/VENDORED.md) | Which HypercubeCNN pin is in tree |
| HypercubeCNN’s own C++ SDK (upstream) | Deep CNN contracts if you dig into the readout |

---

## 12. Cheat sheet

```text
#include "HypercubeWTF.h"

WTFConfig cfg;
cfg.reservoir.dim = 7;                 // N = 128
cfg.reservoir.history_depth = 8;
cfg.reservoir.seed = 1;
cfg.ic_seed = 2;
cfg.episode.T = 0;                     // → N
cfg.episode.readout_slices = 1;        // B
cfg.episode.train_input_noise_sigma = 0.f;
cfg.episode.bypass_reservoir = false;
cfg.readout.dim = 0;                   // auto
cfg.readout.num_outputs = K;
cfg.readout.task = ReadoutTask::Classification;
cfg.readout.epochs = 100;
cfg.readout.num_threads = 1;

WTF wtf(cfg);

wtf.CollectEpisodes(fields_flat, labels);   // count * N floats, sample-major
wtf.TrainOnCollected();

int y = wtf.PredictClass(x);
auto logits = wtf.Predict(x);

wtf.RunEpisode(x);
auto feats = wtf.LastFeatures();            // B * N
float sr = wtf.reservoir().GetRealizedSpectralRadius();
```

**In one line:** pack a field → frozen orbit → end features → train the CNN head.
