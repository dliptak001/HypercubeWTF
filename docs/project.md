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
| Recurrent weights, `W_in` (`N × dim`), bias, delay line | Frozen |
| Field packing / normalization | Caller-owned |
| HCNN readout | **Trained** |

### Vocabulary

| Term | Meaning |
|------|---------|
| **N** | `2^dim`; length of `x` and one state slice |
| **x** | Caller field; values fixed for the episode |
| **Episode** | Clear → drive loop → feature extract (one sample) |
| **Pass** | One stage + `Step` under counter `c` |
| **T** | Total drive passes per episode (after clear); readout samples only after the last |
| **T_warmup** | Optional conceptual washout length inside T (no mid-episode collect) |
| **B** | Multi-slice pack from the **end** delay line only (power of two; ESN seam); still one sample time |
| **vtx_input** | Staged length-N field for the input gather |

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
- Between samples: full `Clear` + `c = 0`.

**Out of scope for v1 product**

- Reindexing weights by `c` (map-side XOR / RIMT default).
- Diagonal-only input (`s += W[v] * x[v XOR c]` without gather).
- Extra length-N field-gain on top of `W_in` (optional later only if named).

Background (map-side default, episode knobs):  
`HypercubeESN/docs/Rotating-input-map-temporalization.md` — not WTF’s S1 form;
old product name *HypercubeMLP* is obsolete.

### 4.3 Episode + readout sample (locked recipe)

**One readout sample per episode, after the final `Step`.** No multi-phase
collection over intermediate `c`. The orbit still runs for `T` passes; only the
**end** state (and optional end delay-line pack `B`) is presented to the HCNN.

```
Clear; c = 0
for t = 0 .. T-1:
    stage vtx_input[v] = x[(v XOR c) & (N - 1)]
    Step(); c += 1
// once:
features = pack SliceAt(0 .. B-1) at this end state   // B=1 → newest slice only
// one training / predict row for this sample
```

| Knob | v1 |
|------|-----|
| Reset | `Clear` + `c = 0` every sample |
| `T` | Episode length (orbit depth); tune so dynamics + registration have run (open: scale with N) |
| Readout sample | **Once at end** only |
| `B` | Multi-slice from **end** delay line only (not multi-phase over `c`) |
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

---

## 5. Open questions

| ID | Question | Lean |
|----|----------|------|
| Q6 | Episode length `T` vs N (and any washout split) | open |
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

**Smoke:** fixed seed → bit-stable end features; clear isolates samples;
`x.size() != N` throws; train/predict parity after readout state get/set.

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
| `HypercubeESN/docs/Rotating-input-map-temporalization.md` | Episode background; **not** WTF S1 form |

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
