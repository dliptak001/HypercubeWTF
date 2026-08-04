# HypercubeWTF — project charter

> Living source of truth for goals, locked design, open questions, and workplan.
> Update when decisions land; do not leave resolved debates only in chat.
>
> **Status:** design / pre-implementation.

---

## 1. Goal

**HypercubeWTF** turns a **static length-N field** into a **driven hypercube
reservoir trajectory**, then trains a **HypercubeCNN readout** on that
trajectory.

- Same frozen-core / trained-head discipline as HypercubeESN.
- Same cube-native readout as HypercubeCNN.
- Product hinge: **spatial → temporal** via **affine field translation** of `x`
  on a **fixed** reservoir (geometry + weights never reindexed by the pass
  counter).

| Product | Natural data | Pipeline |
|---------|--------------|----------|
| HypercubeESN | Low-dim **streams** | `u_t` → step → multi-slice → HCNN |
| HypercubeCNN | Static patterns on the cube | embed → conv+pool → labels |
| **HypercubeWTF** | Static high-dim, **no** intrinsic time | length-N `x` → episode orbit → trajectory → HCNN |

**Mechanism:** values of `x` are fixed for an episode; each pass stages a
cube-translated registration of `x`, then a stock reservoir step. The sample is
**driven through an orbit**; the readout sees **one** reservoir sample at
**episode end** (the trajectory is encoded in that final state / delay line).

Packing raw data (spectra, tables, …) into `x ∈ R^N` is **caller-owned**.
Raman is a motivating load, not a product restriction.

---

## 2. Pipeline

```
  x ∈ R^N   (N = 2^dim; caller-packed; values in [-1, 1]; fixed for episode)
            │
            ▼
  Episode (WTF)   for c = 0 .. T-1:
                    vtx_input[v] = x[(v XOR c) & (N - 1)]
                    Reservoir::Step()          // stock W_in gather + recurrent
                  sample reservoir **once** at end → one feature row per sample
            │
            ▼
  HCNN readout (trained only) → y
```

| Piece | Train? |
|-------|--------|
| Recurrent weights, `W_in` (`N × dim`), bias | Frozen |
| Episode IC **s0** (length `N × M`) | Frozen (drawn once; reloaded each episode) |
| Field packing / normalization | Caller-owned |
| HCNN readout | **Trained** |

### Vocabulary

| Term | Meaning |
|------|---------|
| **N** | `2^dim`; length of `x` and one state slice |
| **x** | Caller field; values fixed for the episode |
| **Episode** | Load fixed IC → drive loop → end sample (one sample) |
| **Pass** | One stage + `Step` under counter `c` |
| **T** | Number of reservoir **drive** passes per episode. Readout is not involved during these passes |
| **B** | Multi-slice pack from the **end** delay line only (power of two; ESN seam); still one sample time |
| **vtx_input** | Staged length-N field for the input gather |
| **M** | `history_depth` — delay-line depth (same as hESN) |
| **s0** | Frozen IC buffer length **`N × M`**, drawn at construction; reloaded into the full delay line at every episode start |

---

## 3. Relation to HypercubeESN / HypercubeCNN

**Reuse components, not the ESN façade.**

| | ESN | WTF |
|---|-----|-----|
| Loop | Stream step ↔ one train row | Per-sample **episode** → **one** feature row (end state) |
| Input | Low-dim channels, block broadcast | Length-N **field** each pass |
| Time | Stream time | Synthetic orbit inside one sample |
| Product | Stream RC | Field orbit on fixed weights |

External feedback is **not** used and **not** ported for the orbit.

### Dependency model (locked)

Do **not** link ESN or HCNN as libraries. Vendor local copies:

| Piece | From |
|-------|------|
| `Reservoir`, `Readout` | **HypercubeESN** (pin commit/tag) |
| `third_party/HypercubeCNN` | **HypercubeCNN** upstream (pin commit/tag) — **not** via hESN’s tree |

Pin sources in `VENDORED.md`. Re-sync when upstream fixes matter.

---

## 4. Locked design

### 4.1 Field contract (S0)

| Rule | Behavior |
|------|----------|
| Length | `x.size() == N` required |
| Else | **Throw** (no pad / truncate / resample in core) |
| Range | Caller keeps **[-1, 1]**; library trusts caller |
| Packing | Caller-owned (typical: data in low addresses, zeros above) |
| Mutation | Do not modify caller’s `x` |

### 4.2 Affine field translation (S1) + input path (Q2)

**Fixed:** cube geometry and all reservoir weights.  
**Moving:** registration of `x` only.

```
// each pass
for v in 0 .. N-1:
    vtx_input[v] = x[(v XOR c) & (N - 1)]
Step()    // stock: s += sum_i vtx_input[v XOR (1<<i)] * W_in[v][i] + recurrent
c += 1
```

- Drive **is** `vtx_input`. Input path is stock hESN neighbor gather (`W_in` is
  `N × dim` — not “one weight per vertex”).
- Add length-N field inject on vendored Reservoir (`InjectInputField` or
  equivalent). Avoid `num_inputs = N` scalar hacks.
- Address period is **N** (mask with `N - 1` when `c` can exceed).
- Between samples: reload **s0** (not zero-clear) + `c = 0` (§4.3).

### 4.2a Episode initial state (**s0**, locked)

Do **not** start each episode from an all-zero reservoir. At construction, draw
a frozen IC buffer **s0** sized to the **full delay line** and keep it for the
life of the instance:

| Rule | Behavior |
|------|----------|
| Size | **`N × M`** with `M = history_depth` (one length-N slice per age) |
| When drawn | Construction only |
| Seed | **Same as the reservoir seed** (no separate IC seed knob in v1) |
| Distribution | i.i.d. uniform on **[-0.5, 0.5]** over the whole buffer |
| Lifetime | **Static** after construction — never trained, never redrawn per sample |
| Each episode start | **`memcpy`** the full **`N × M`** buffer into delay-line storage (and align current state / age-0 with that load) — **not** zeros, **not** residual state from the previous sample |
| `c` | Reset to `0` |

Implementation intent: one bulk copy of frozen `s0` into the contiguous
history block (same layout as hESN’s `N * history_depth` buffer), not a
per-vertex loop. Same **s0** every episode → no sample-order leakage;
full-depth nonzero start → recurrent gather does not see a zero-padded history
at episode 0.

**Out of scope for v1 product**

- Reindexing weights by `c` (map-side XOR).
- Diagonal-only input (`s += W[v] * x[v XOR c]` without gather).
- Extra length-N field-gain on top of `W_in` (optional later only if named).

### 4.3 Episode + readout sample (locked recipe)

**`T` is drive passes only.** The readout does not run during the loop. After
`T` steps, sample the reservoir **once** (optional end delay-line pack `B`) for
the HCNN. Washout of **s0** is just the early part of those same `T` orbit
passes — no separate warmup counter.

```
LoadDelayLine(s0)      // full N×M buffer; not zero Clear
c = 0
for t = 0 .. T-1:                          // drive only
    stage vtx_input[v] = x[(v XOR c) & (N - 1)]
    Step(); c += 1
// once, after the loop:
features = pack SliceAt(0 .. B-1)           // B=1 → newest slice only
// one training / predict row for this sample
```

| Knob | v1 |
|------|-----|
| Episode start | Reload frozen **s0** (`N × M` delay line); `c = 0` |
| `T` | Drive-pass count (orbit depth); open: how it scales with N |
| Readout | **Once after** the `T` drive passes — not interleaved |
| `B` | Multi-slice from **end** delay line only |
| Ext-fb | Off |

### 4.4 Decisions table

| Topic | Decision | Date |
|-------|----------|------|
| Name | HypercubeWTF | 2026-08-04 |
| S1 | `vtx_input[v] = x[v XOR c]`; weights fixed | 2026-08-04 |
| Input path | Stock `W_in` gather; drive = `vtx_input` | 2026-08-04 |
| Map-side XOR | Out of scope | 2026-08-04 |
| Field contract | size==N or throw; trust [-1,1]; packing caller-owned | 2026-08-04 |
| Dependencies | Vendor Reservoir+Readout from hESN; HCNN from HypercubeCNN upstream | 2026-08-04 |
| Wrap full ESN | No | 2026-08-04 |
| Readout sample (Q5) | **Once per episode at end** (optional end-only B-slices) | 2026-08-04 |
| Episode IC (**s0**) | Frozen U[-0.5, 0.5]^(N×M) from **reservoir seed**; `memcpy` full delay line every episode | 2026-08-04 |

---

## 5. Open questions

| ID | Question | Lean |
|----|----------|------|
| Q6 | How episode drive count `T` scales with N | open |
| Q8 | First demo | synthetic first; Raman only with clear data protocol |
| — | Vendoring pins (ESN + HCNN commits) | required before Phase 1 |
| — | Default `B` (1 vs multi-slice at end) | open; ESN-style knob |

---

## 6. API sketch (not frozen)

```text
WTFConfig
  reservoir: ReservoirConfig   // dim, seed, SR, leak, history_depth, input_scaling, …
  readout:   ReadoutConfig
  episode:   EpisodeConfig     // T
  readout_slices: size_t B     // end-of-episode delay-line pack only

class WTF
  explicit WTF(const WTFConfig&);
  void RunEpisode(span<const float> x);   // throws if x.size() != N
  void TrainOnCollected(...);
  vector<float> Predict(span<const float> x);
```

**v1 non-goals:** closed-loop free-run; replacing ESN for real streams.

---

## 7. Repo layout (intended)

```text
HypercubeWTF/
  CMakeLists.txt
  LICENSE
  README.md
  docs/project.md
  WTF.h / Reservoir.* / Readout.*   // flat core (ESN-style); Reservoir+Readout from hESN
  third_party/HypercubeCNN/         // from HypercubeCNN upstream
  examples/
  tests/
```

---

## 8. Validation

**Smoke:** fixed seed → bit-stable end features; reloading **s0** isolates
samples (order shuffle does not change per-sample features); `x.size() != N`
throws; train/predict parity after readout state get/set.

**Examples (when ready):** synthetic classification in R^N; optional spectral
toy with caller packing.

---

## 9. Workplan

Sequential gates. Docs may draft ahead; code waits on exit criteria.

| Phase | Work | Exit |
|-------|------|------|
| **0** Charter | This file; design lock | Pins chosen; optional B default |
| **1** Skeleton | CMake C++23; vendor HCNN + Reservoir/Readout; empty `WTF`; README | Release build |
| **2** Episode core | Field inject; field orbit; end sample; tests | Stable end features from synthetic `x` |
| **3** Readout | End pack (B); batch train; predict; metrics | E2E synthetic classification |
| **4** Demos | Example packing helpers; spectral toy; optional real data | Example binary + short doc |
| **5** Polish | Bindings / perf / richer packing as needed | — |

---

## 10. References

| Doc | Use |
|-----|-----|
| `HypercubeESN/docs/Reservoir.md` | Dynamics, inject, history |
| `HypercubeESN/docs/Readout.md` | HCNN head, multi-slice |

---

## 11. Next

1. Pin HypercubeESN + HypercubeCNN commits for vendoring.
2. Default **B** (1 vs multi-slice at end) if not deferred to config default.
3. Choose first demo (**Q8**); lean on episode length **T** (**Q6**).
4. Phase 1 skeleton.

---

## 12. Changelog

| Date | Change |
|------|--------|
| 2026-08-04 | Charter: goal, vendor-copy deps, S1 field orbit, workplan |
| 2026-08-04 | Locked: S1 = stage `x[v XOR c]` + stock `W_in`; S0 = length-N caller field; HCNN from upstream not via hESN |
| 2026-08-04 | Audit: cut fluff, closed debate residue, single locked-design section |
| 2026-08-04 | **Q5 locked:** readout samples reservoir **once at episode end** (optional end-only B); no multi-phase collect |
| 2026-08-04 | Removed bake-off / constant-drive control from charter, validation, and workplan |
| 2026-08-04 | Drop T_warmup: `T` is drive passes only; washout is implicit in the orbit |
| 2026-08-04 | Episode IC: frozen **s0** length **N×M** ~ U[-0.5,0.5]; reload full delay line each episode (not zero Clear) |
