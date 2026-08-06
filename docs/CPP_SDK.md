# HypercubeWTF C++ SDK

HypercubeWTF is a **C++23 episode product**: a static length-N field drives a
**frozen hypercube reservoir** for T passes, then a **HypercubeCNN readout**
learns from the **end-of-episode** feature pack. The public surface is **small
and contract-driven** so hosts can integrate one type (`WTF`) without learning
the full ESN or HCNN stacks.

**Examples stay examples. Helpers stay optional. Neither is the product’s reason
for existing** — the core (`WTF`) is.

Canonical API guide for **`HypercubeWTFCore` 0.1.x** (early development; APIs
may still move). Aligned with the public headers and in-tree demos. Design
charter (goals, locked decisions, workplan): **[project.md](project.md)**.

**Primary audience:** engineers embedding WTF in a host binary (offline collect →
train → predict).  
**Secondary:** engineers learning the HypercubeAI family — same API, progressive
disclosure, worked demos.

**Package:** C++23 static library **`HypercubeWTFCore`**. Headers live at the
repo root. Depends on vendored **`HypercubeCNNCore`** (used inside the readout;
hosts normally do not call HCNN directly).

---

## 1. What you are building

HypercubeWTF turns a **static** high-dimensional field into a **synthetic orbit**
on the Boolean hypercube, then samples **once at the end**.

| Product | Natural data | Pipeline |
|---------|--------------|----------|
| HypercubeESN | Low-dim **streams** | `u_t` → step → multi-slice → HCNN |
| HypercubeCNN | Static patterns on the cube | embed → conv+pool → labels |
| **HypercubeWTF** | Static high-dim, **no** intrinsic time | length-N `x` → drive orbit → **end state** → HCNN |

| Idea | Meaning in code |
|------|-----------------|
| Cube dimension `dim` | Integer in **[5, 16]**; `N = 2^dim` vertices |
| Field `x` | Caller-owned length-N floats (often in `[-1, 1]`); fixed for one episode |
| Episode | Load frozen IC `s0` → `T` drive passes → pack `B` end delay-line ages |
| Reservoir | **Frozen** recurrent dynamics (weights drawn once) |
| Readout | **Trained** HypercubeCNN head on end features (`FeatureSize() = B * N`) |

### Capacity is topological (power of two)

Per episode the field is **always** length `N = 2^dim`. That is the size of the
Boolean hypercube — not a free “input_size = 784” the library can relax.

| What the core does | What it does **not** do |
|--------------------|-------------------------|
| Own length-N episodes and end feature packs | Accept free-length raw vectors as the field |
| Affine re-address of fixed `x` each pass (`v XOR c`) | Learn packing of images/spectra into the cube |
| Train only the HCNN readout | Train reservoir weights or IC |
| Optional multi-age pack `B` at the **end** only | Stream intermediate passes into the readout |

**Non–power-of-two data is normal.** Pack outside WTF (pad, spatial embed,
custom maps). Optional demo helpers under `examples/common/` are one MNIST-oriented
recipe — not part of the product core.

**Pipeline:**

```text
x ∈ R^N   (caller-packed; fixed for the episode)
  → LoadIC(s0)
  → for pass = 0 .. T-1:
        drive[v] = x[(v XOR c) & (N-1)]
        InjectInputField + Step
        c += 1
  → pack B end delay-line ages  →  features[B*N]
  → HCNN readout  →  num_outputs
```

When `bypass_reservoir` is true, the orbit is skipped and features are the
length-N field itself (`B` must be 1). Same collect / train / predict surface.

---

## 2. Mental model of one episode

Each **pass** re-registers the **same** field under a new address offset `c`
(starts at 0 every episode). Geometry and frozen weights never move with `c`;
only the field→vertex map does. Default `T = N` is one full tour of offsets;
`T > N` is allowed (address wraps).

```text
LoadInitialCondition(s0)          // frozen; reloaded every episode
c = 0
for pass in 0 .. T-1:
    drive[v] = x[(v XOR c) & (N - 1)]   // length-N stage
    Step()                              // W_in gather + recurrent + delay line
    c += 1
features = concat( SliceAt(0), …, SliceAt(B-1) )   // end of delay line only
```

| Piece | Train? |
|-------|--------|
| Recurrent weights, `W_in`, bias | Frozen |
| Episode IC **s0** (`N × M`) | Frozen (drawn once from `ic_seed`) |
| Field packing | Caller-owned |
| HCNN readout | **Trained** |

**Vocabulary**

| Term | Meaning |
|------|---------|
| **N** | `2^dim`; field length and one state slice |
| **T** | Drive-pass count (config `0` → use N) |
| **B** | End ages packed into features (`readout_slices`; power of two, `1 ≤ B ≤ M`) |
| **M** | `history_depth` — delay-line depth |
| **s0** | Frozen IC, length `N × M`; separate from weight seed |
| **FeatureSize** | `B * N` floats into the readout |

---

## 3. SDK layers and source tree

| Layer | Role |
|-------|------|
| **Core (`WTF`)** | The product. Integration surface — document like a library another binary links. |
| **Config types** | `WTFConfig`, `EpisodeConfig`, `ReservoirConfig`, `ReadoutConfig` (+ readout enums). Set once at construct. |
| **Inspection** | `wtf.reservoir()` / `wtf.readout()` for live SR, weights, advanced readout I/O. |
| **Examples** | Proof of contracts + recipes (`wtf_smoke`, `wtf_synth`, `wtf_mnist`). Not the definition of the SDK. |
| **Vendored HCNN** | `third_party/HypercubeCNN` — linked by the readout; **not** the host integration surface for WTF apps. |

### Source tree (public headers)

```text
HypercubeWTF/
  HypercubeWTF.h       # umbrella — includes WTF.h
  WTF.h                # front door: WTF, WTFConfig, EpisodeConfig
  Reservoir.h          # ReservoirConfig (+ Reservoir type for inspection)
  Readout.h            # ReadoutConfig, enums, Readout type
  Reservoir.cpp / Readout.cpp / WTF.cpp
  third_party/HypercubeCNN/   # vendored; see VENDORED.md
  examples/                   # demos + examples/common helpers
  tests/wtf_smoke.cpp
  docs/CPP_SDK.md             # this file
  docs/project.md             # charter
```

| Product layer | Include | Required? |
|---------------|---------|-----------|
| Full product surface | `HypercubeWTF.h` or `WTF.h` | Yes (minimal integration) |
| Config field types only | pulled via `WTF.h` | Yes |
| MNIST pack helpers | `examples/common/*.h` | Demo-only |
| Raw HCNN API | `third_party/...` | No — prefer `WTF` / `readout()` |

Link target: **`HypercubeWTFCore`** (pulls **`HypercubeCNNCore`**).

### Host contracts (integrators)

These rules are intentional product contracts, not implementation accidents.

| Contract | Rule |
|----------|------|
| **Field length** | Every episode field is length **N**. Over/under size throws on `RunEpisode` / collect |
| **Packing** | Host maps domain data → `R^N`. Core never invents image/sequence layout |
| **Frozen dynamics** | Reservoir weights and `s0` fixed at construct; every episode reloads `s0` |
| **End sample only** | Readout sees the **end** feature pack (optional multi-age `B`), not intermediate passes |
| **Train noise** | `train_input_noise_sigma` applies on **collect** only — not `RunEpisode` / `Predict` / `PredictClass` |
| **Bypass** | `bypass_reservoir` requires `B == 1`; features = field (noise still applies on collect) |
| **Task / loss** | Classification → CE path; Regression → MSE path (fixed by `ReadoutTask`) |
| **Outputs** | `Predict` returns raw logits / preds — **never** softmax |
| **Collected metrics** | `AccuracyOnCollected` / `R2OnCollected` score the **training** set, not held-out data |
| **Concurrency** | One `WTF` instance is exclusive-use (no concurrent public calls). Bulk collect parallelizes **internally** |
| **Ownership** | `WTF` is non-copyable (move is available; prefer exclusive ownership) |

**Canonical train/infer surface for new hosts** (do not grow beyond this without
design review):

```text
fill WTFConfig → WTF wtf(cfg)
CollectEpisode / CollectEpisodes   (+ optional train_input_noise_sigma)
TrainOnCollected
Predict / PredictClass             (fresh clean episode each call)
optional: RunEpisode + LastFeatures, AccuracyOnCollected / R2OnCollected
optional: ClearCollected, bypass_reservoir A/B, collect_threads
```

Prefer **config structs** over long constructor argument lists. Prefer bulk
`CollectEpisodes` for large datasets.

### Integration contracts (quick table)

| Contract | Rule |
|----------|------|
| **Capacity** | Field = N; features = B×N |
| **T default** | `episode.T == 0` → T = N |
| **B** | Power of two, `1 ≤ B ≤ M` |
| **readout.dim** | `0` = auto `dim + log2(B)`; else must match |
| **Train noise** | Collect only; deterministic from `ic_seed` + sample index |
| **Predict** | Clean episode; no collect noise |
| **Threading** | Exclusive `WTF`; pin `readout.num_threads = 1` if host multiplies nets |
| **Ownership** | Non-copyable |

---

## 4. Build and consume

**Needs:** C++23, CMake ≥ 3.21. Prefer **Release** (Debug float paths differ under
`-ffast-math` recipes).

Open in CLion and reload CMake, or:

```bash
cmake --build cmake-build-release
```

| Binary (top-level only) | Role |
|-------------------------|------|
| `wtf_smoke` | Episode contract + small train/predict |
| `wtf_synth` | Synthetic multi-class (no data files) |
| `wtf_mnist` | MNIST recipe (IDX under `C:\HypercubeWTF\data`) |

When HypercubeWTF is **not** the top-level project, demos/smoke are skipped;
consumers still get **`HypercubeWTFCore`**.

### As a subdirectory

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

## 5. First program (collect → train → predict)

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
    cfg.episode.T = 0;                  // → N
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

**Integration habits:**

1. Fill **`WTFConfig`**, construct **`WTF` once** (weights + `s0` freeze here).
2. **Collect** many episodes, then **`TrainOnCollected`** once (or retrain without
   clearing if you want another pass on the same features).
3. **`Predict` / `PredictClass`** always run a **fresh clean** episode (no train noise).
4. Packing is yours — demos show patterns; the core only accepts length-N fields.

---

## 6. Core API (`WTF`)

### Config structs

```cpp
struct EpisodeConfig {
    size_t T = 0;                      // 0 → T = N after construct
    size_t readout_slices = 1;         // B; power of two, ≤ M
    size_t collect_threads = 0;        // 0 = auto (leave 1–2 cores free)
    float  train_input_noise_sigma = 0.f;  // collect-only AWGN
    bool   bypass_reservoir = false;   // field → features; requires B == 1
};

struct WTFConfig {
    ReservoirConfig reservoir{};
    ReadoutConfig   readout{};
    EpisodeConfig   episode{};
    uint64_t        ic_seed = 1;       // s0 only — not weight seed
};
```

#### ReservoirConfig (frozen dynamics)

| Field | Role | Typical |
|-------|------|---------|
| `dim` | Cube dim; N = 2^dim; **[5, 16]** | 7–10 in demos |
| `seed` | Weight / bias draws | fixed for reproducibility |
| `spectral_radius` | Target SR for recurrent block | ~0.999 |
| `leak_rate` | Leaky mix; (0, 1] | 1 |
| `input_scaling` | Drive strength | 0.02–0.03 |
| `history_depth` | M; **[1, 64]** | 4–16 |
| `bias_scaling` | Bias amplitude; 0 = off | ~0.003 |
| `verbose` | Construction banner | false |

#### ReadoutConfig (trainable head) — knobs hosts set most often

| Field | Role |
|-------|------|
| `dim` | Feature dim; **0 = auto** `reservoir.dim + log2(B)` |
| `num_outputs` | Classes or regression width |
| `task` | `ReadoutTask::Classification` or `Regression` |
| `epochs`, `batch_size` | Batch train loop |
| `lr_max`, `lr_min_frac`, `lr_decay_epochs` | Cosine LR schedule |
| `num_threads` | HCNN workers; **1** for deterministic / multi-net hosts |
| `restore_best_epoch` | Best-epoch restore (default true on Readout; demos often false) |
| `seed` | HCNN weight init |

Further architecture fields (`num_layers`, `use_pooling`, `activation`, …) live on
`ReadoutConfig` in `Readout.h` — same semantics as the HypercubeESN-style readout
façade. Hosts that only need classification rarely change them.

### Construct and sizing

```cpp
explicit WTF(const WTFConfig& cfg);

size_t N() const;            // 2^dim
size_t T() const;            // resolved drive passes
size_t B() const;            // readout_slices
size_t M() const;            // history_depth
size_t FeatureSize() const;  // B * N
size_t NumCollected() const;
size_t NumOutputs() const;
size_t CollectThreads() const;   // configured preference (0 = auto)
bool   BypassReservoir() const;

const Reservoir& reservoir() const;
const Readout&   readout() const;
const ReadoutConfig& readout_config() const;  // resolved (dim filled in)
```

Construct validates: T > 0 after resolve, B power of two and ≤ M, bypass ⇒ B == 1,
σ finite and ≥ 0, `readout.dim` auto or exact, `num_outputs ≥ 1`.

### Episode and features

```cpp
void RunEpisode(std::span<const float> x);   // size N; does not modify x
std::span<const float> LastFeatures() const; // length FeatureSize()
```

| | |
|--|--|
| `RunEpisode` | Drive (or bypass copy). Updates `LastFeatures` |
| `LastFeatures` | Also updated by serial `CollectEpisode`, `Predict`, `PredictClass` |
| Bulk collect | **Does not** update `LastFeatures` (rows go only into the train set) |

### Collect → train → predict

```cpp
// Classification
void CollectEpisode(std::span<const float> x, int class_label);
void CollectEpisodes(std::span<const float> fields_flat,
                     std::span<const int> labels);
void CollectEpisodes(size_t count, std::span<const int> labels,
                     const std::function<void(size_t, std::span<float>)>& fill_field);

// Regression
void CollectEpisode(std::span<const float> x, std::span<const float> target);
void CollectEpisodes(std::span<const float> fields_flat,
                     std::span<const float> targets_flat);
void CollectEpisodes(size_t count, std::span<const float> targets_flat,
                     const std::function<void(size_t, std::span<float>)>& fill_field);

void ClearCollected();
void TrainOnCollected();   // keeps samples; retrain OK

std::vector<float> Predict(std::span<const float> x);
int  PredictClass(std::span<const float> x);   // classification only

double AccuracyOnCollected() const;  // train-set accuracy
double R2OnCollected() const;        // train-set R²
```

| API | Notes |
|-----|--------|
| `fields_flat` | Sample-major; length `count * N` |
| `fill_field(i, field)` | Write N floats; must be safe for concurrent distinct `i` |
| `collect_threads` | 0 = auto; pool grows to high-water mark for the `WTF` lifetime |
| `TrainOnCollected` | Requires `NumCollected() > 0`; does not clear the set |
| Wrong task family | Throws (`invalid_argument` / task guards) |

### Train-input noise

`episode.train_input_noise_sigma > 0` adds i.i.d. Gaussian noise to the field
**before** the collect episode. Salt is deterministic from `ic_seed` and sample
index. **`RunEpisode` / `Predict` / `PredictClass` ignore σ.**

### Bypass reservoir

`episode.bypass_reservoir = true` (and `B == 1`): features = packed field. Use for
A/B vs orbit. Collect noise still applies when σ > 0.

---

## 7. What is *not* the host surface

| Item | Why |
|------|-----|
| Driving `Reservoir` yourself for product training | Episode contract lives on `WTF` |
| Calling vendored `hcnn::HCNN` from the app | Readout owns that stack |
| `examples/common/*` packing / data discovery | Demo recipes only |
| Streaming `Readout::TrainStep*` on intermediate states | Product samples **end-of-episode** only |
| External feedback / multi-channel ESN inject | Intentionally **not** ported |

**Advanced but still public types:** `Reservoir` and `Readout` headers exist so
`WTFConfig` and `wtf.readout()` / `wtf.reservoir()` work. Prefer WTF collect/train
for the product path. Use `readout().Weights()` / `SaveHcnnModel` when you need
export; use `reservoir().GetRealizedSpectralRadius()` for logging.

---

## 8. Training loop pattern (demos)

In-tree demos keep knobs in one place (`MakeWTFConfig` + demo-only `k*` constants):

```text
MakeWTFConfig()          // product WTFConfig
→ WTF wtf(cfg)
→ CollectEpisodes / CollectEpisode
→ TrainOnCollected
→ held-out or train-set metrics
→ PredictClass
```

| Demo | Role |
|------|------|
| `tests/wtf_smoke.cpp` | Contract gate (sizes, determinism, parallel collect, noise, bypass) |
| `examples/synth/wtf_synth.cpp` | Fast multi-class CI without data files |
| `examples/mnist/wtf_mnist.cpp` | MNIST pack → orbit → readout (+ optional A/B knobs) |

See [`examples/README.md`](../examples/README.md).

---

## 9. Memory, threading, performance

| Topic | Guidance |
|-------|----------|
| **Exclusive instance** | Do not share one `WTF` across host threads for concurrent public calls |
| **Bulk collect** | Internal fork-join; worker 0 reuses primary reservoir; others clone weights once |
| **Pool lifetime** | Collect thread pool grows; does not shrink until `WTF` destroys |
| **HCNN threads** | `readout.num_threads = 1` when you already parallelize across many `WTF`s |
| **Cost** | Orbit cost scales with `T * N`; feature size `B * N`; HCNN cost with dim/channels |
| **Release** | Prefer Release builds for study numbers (`-ffast-math` recipes) |

---

## 10. Pitfalls checklist

| Pitfall | Fix |
|---------|-----|
| Field length ≠ N | Pack to capacity; check `wtf.N()` |
| Expect free input size (not 2^dim) | Capacity is the cube; pack in the host |
| Softmax in `Predict` | Don’t; use logits + `PredictClass` / argmax |
| Train noise on eval | σ is collect-only; eval uses clean `Predict*` |
| `LastFeatures` after `CollectEpisodes` | Bulk path does not update it — use serial collect or re-`RunEpisode` |
| `AccuracyOnCollected` as held-out | It is train-set; score held-out yourself |
| `B` not power of two / `B > M` | Construct throws |
| Bypass with `B != 1` | Construct throws |
| Concurrent public use of one `WTF` | Exclusive instance; use bulk collect for parallelism |
| Treat packing as the product | Core is episode + readout; packing is host-owned |
| Link HCNN instead of WTF | Link `HypercubeWTFCore`; include `WTF.h` |

---

## 11. Private / non-product details

| Area | Role |
|------|------|
| `WTF::CollectPool` / workers | Internal parallel collect |
| Vendored `HCNNNetwork` / layers / `ThreadPool` | HCNN private (see HypercubeCNN docs) |
| Reservoir weight layout | Frozen draw ABI; not a host tuning surface mid-run |

**Boundary policy:**

1. **Hosts integrate `WTF`.** Config structs at construct; collect → train → predict.
2. **Examples are not the SDK.** Copy recipes, not dependency on `examples/common` in production if you can avoid it.
3. **Features land on `WTF` first.** Deep reservoir/readout knobs are pre-construct config, not a second runtime product.

---

## 12. Further reading in this repo

| Doc / path | Content |
|------------|---------|
| [`project.md`](project.md) | Charter, locked design, workplan |
| [`../examples/README.md`](../examples/README.md) | Demo map + MNIST data appendix |
| [`../third_party/HypercubeCNN/VENDORED.md`](../third_party/HypercubeCNN/VENDORED.md) | HCNN pin |
| HypercubeCNN `docs/CPP_SDK.md` (upstream) | Readout’s underlying CNN contracts |

---

## 13. One-page cheat sheet

```text
#include "HypercubeWTF.h"

WTFConfig cfg;
cfg.reservoir.dim = 7;              // N = 128
cfg.reservoir.history_depth = 8;
cfg.reservoir.seed = 1;
cfg.ic_seed = 2;
cfg.episode.T = 0;                  // → N
cfg.episode.readout_slices = 1;     // B
cfg.episode.train_input_noise_sigma = 0.f;
cfg.episode.bypass_reservoir = false;
cfg.readout.dim = 0;                // auto dim + log2(B)
cfg.readout.num_outputs = K;
cfg.readout.task = ReadoutTask::Classification;
cfg.readout.epochs = 100;
cfg.readout.num_threads = 1;

WTF wtf(cfg);

// Collect (sample-major fields_flat: count * N)
wtf.CollectEpisodes(fields_flat, labels);
wtf.TrainOnCollected();

// Infer (clean episode)
int y = wtf.PredictClass(x_span);
auto logits = wtf.Predict(x_span);

// Episode-only / inspection
wtf.RunEpisode(x_span);
auto feats = wtf.LastFeatures();     // B*N
float sr = wtf.reservoir().GetRealizedSpectralRadius();
```

**Dependencies:** C++23 + threads; vendored HypercubeCNN inside the readout.  
**Product hinge:** static field → frozen orbit → end features → trained HCNN.
