# HypercubeWTF — project charter

> Living source of truth for goals, locked design, open questions, and workplan.
> Update when decisions land; do not leave resolved debates only in chat.
>
> **Status:** Phase 4 demos in tree (`wtf_synth`, `wtf_mnist`); polish / richer packs next.

---

## 1. Goal

**HypercubeWTF** drives a static length-N field through a fixed hypercube
reservoir for `T` passes (affine field translation), then trains a
**HypercubeCNN** on the **end-of-episode** reservoir state.

- Frozen reservoir / trained head (same RC discipline as HypercubeESN).
- Cube-native HCNN readout (same family as HypercubeCNN).
- Product hinge: **spatial → temporal** by re-registering fixed `x` each pass;
  geometry and weights never move with the pass counter.

| Product | Natural data | Pipeline |
|---------|--------------|----------|
| HypercubeESN | Low-dim **streams** | `u_t` → step → multi-slice → HCNN |
| HypercubeCNN | Static patterns on the cube | embed → conv+pool → labels |
| **HypercubeWTF** | Static high-dim, **no** intrinsic time | length-N `x` → drive orbit → **end state** → HCNN |

The orbit is **drive-only**. The readout is **not** fed intermediate passes.
Training/predict uses **one** sample per episode: the state after the last
drive step (plus optional multi-slice pack `B` from that same end delay line).

Packing raw data into `x ∈ R^N` is **caller-owned**. Raman is a motivating
load, not a product restriction.

---

## 2. Pipeline

```
  x ∈ R^N   (N = 2^dim; caller-packed; values in [-1, 1]; fixed for episode)
            │
            ▼
  Episode     LoadIC(s0)                 // intelligent full delay-line load
              c = 0
              for pass = 0 .. T-1:       // drive only; default T = N
                  vtx_input[v] = x[(v XOR c) & (N - 1)]
                  Step()                 // field inject + W_in gather + recurrent
                  c += 1                 // may exceed N; mask wraps address
              sample once at end         // default B = 1 → newest slice only
            │
            ▼
  HCNN readout (trained only) → y
```

| Piece | Train? |
|-------|--------|
| Recurrent weights, input weights `W_in` (`N × dim`), bias | Frozen |
| Episode IC **s0** (length `N × M`) | Frozen (drawn once; reloaded each episode) |
| Field packing / normalization | Caller-owned |
| HCNN readout | **Trained** |

### Vocabulary

| Term | Meaning |
|------|---------|
| **N** | `2^dim`; length of `x` and one state slice |
| **x** | Caller field; values fixed for the episode |
| **Episode** | Load IC → `T` drive passes → one end sample |
| **Pass** | One stage + `Step`; advance pass counter `c` once |
| **T** | Drive-pass count only (default **`T = N`**). May be **> N**; field address wraps |
| **B** | How many end delay-line ages to pack into the readout (default **`B = 1`**) |
| **M** | `history_depth` — delay-line depth |
| **vtx_input** | Staged length-N field for the input gather |
| **s0** | Frozen IC, length **`N × M`**, separate IC seed; reloaded every episode start |
| **c** | Pass counter; starts at 0 each episode; indexing uses `(… ) & (N - 1)` |

---

## 3. Relation to HypercubeESN / HypercubeCNN

**Start from hypercube RC physics, not from the ESN product façade.**

| | ESN | WTF |
|---|-----|-----|
| Loop | Stream step ↔ one train row | Per-sample **episode** → **one** end feature row |
| Input | Low-dim channels, block broadcast | Length-N **field** every pass |
| Time | Stream time | Synthetic orbit inside one sample |
| Feedback | Optional external feedback port | **None — do not port** |

### Dependency model (locked)

Do **not** link ESN or HCNN as libraries.

| Piece | From | Notes |
|-------|------|-------|
| Reservoir | **HypercubeESN** as a **starting copy** | Fork into a **WTF-specific** reservoir: keep cube dynamics, delay line, `W_in` gather, SR/leak. **Drop** stream-era baggage we do not need (`num_inputs` block inject, **entire external-feedback path**, ESN lifecycle hooks). First-class **length-N field inject**. Own delay-line layout so IC load is correct (§4.3). |
| Readout | **HypercubeESN** starting copy | Thin HCNN façade; adapt to end-of-episode packing |
| `third_party/HypercubeCNN` | **HypercubeCNN** upstream | Independent vendor; **not** via hESN’s tree |

**Source trees (locked for first copy):** use what’s on disk at Phase 1 — not
“latest main” as a slogan.

| Source | Branch | Commit (on disk at lock) |
|--------|--------|---------------------------|
| HypercubeESN | **`feedback`** (**not** `main`) | `ae3fb6430e557066a25d7678fdbfedad81093697` |
| HypercubeCNN | `main` | `20bbb23476342abfd957fdc0fabe88e4dad7ae00` |

Record the same in `VENDORED.md` when the files are copied. Re-sync later only
when useful — this is not “stay API-compatible with ESN forever.”

**External feedback:** not used, **not ported**, no stubs that invite misuse.
No feedback port.

---

## 4. Locked design

### 4.1 Field contract

| Rule | Behavior |
|------|----------|
| Length | `x.size() == N` required |
| Else | **Throw** (no pad / truncate / resample in core) |
| Range | Caller keeps **[-1, 1]**; library trusts caller |
| Packing | Caller-owned (typical: data in low addresses, zeros above) |
| Mutation | Do not modify caller’s `x` |

### 4.2 Affine field translation + input path

**Fixed:** cube geometry and all reservoir weights.  
**Moving:** registration of `x` only.

```
// each pass; c may be >= N — mask wraps the field orbit
for v in 0 .. N-1:
    vtx_input[v] = x[(v XOR c) & (N - 1)]
Step()    // field is vtx_input; s += sum_i vtx_input[v XOR e_i] * W_in[v][i] + recurrent
c += 1
```

- Drive **is** `vtx_input`. Input path is neighbor gather on that field
  (`W_in` is `N × dim`).
- Field inject is the native drive API — not channel/block hacks from ESN.
- Field alignment is **period N**. **`T` may exceed `N`**; `(v XOR c) & (N - 1)`
  wraps so the orbit continues without a separate wrap counter.

**Out of scope**

- Reindexing weights by `c` (map-side XOR).
- Diagonal-only input (no `W_in` gather).
- Extra length-N field-gain on top of `W_in` (unless named later).
- **Any external-feedback / closed-loop port.**

### 4.3 Episode initial state (**s0**)

Do **not** start each episode from all zeros.

| Rule | Behavior |
|------|----------|
| Size | **`N × M`** (`M = history_depth`) — full delay-line worth of IC |
| When drawn | Construction only |
| Seed | **Separate IC seed** (`ic_seed` / `s0_seed`) — not the reservoir weight seed |
| Distribution | i.i.d. uniform on **[-0.5, 0.5]** over the whole buffer |
| Lifetime | **Static** after construction |
| Each episode | Reload **s0** into the live delay line / state so the first `Step` sees that IC as logical ages — **not** residual state from the previous sample |
| `c` | `0` |

**Load must respect delay-line logic.** A blind `memcpy` into a mid-rotation
ring buffer (ESN-style rotating slice pointers) can scramble logical ages.
WTF owns the layout: either reset age base / rotation to a canonical position
then bulk-load `s0` in age order, or scatter-load by logical age. Prefer a
layout where episode start is **correct and simple** (canonical rest + bulk
copy of `N × M` is fine **after** rotation state is reset). Do not cargo-cult
a one-liner that ignores ring orientation.

### 4.4 Episode recipe

```
LoadIC(s0)                 // correct full-depth IC; see §4.3
c = 0
for pass = 0 .. T-1:       // drive only
    for v in 0 .. N-1:
        vtx_input[v] = x[(v XOR c) & (N - 1)]
    Step()
    c += 1
// once, after the loop:
features = pack end ages 0 .. B-1     // default B = 1
```

| Knob | v1 default |
|------|------------|
| Episode start | Load **s0**; `c = 0` |
| **T** | **`N`** (allowed to be set **> N**; field index wraps via mask) |
| Readout | **Once after** all `T` drive passes |
| **B** | **`1`** (newest slice only; config may raise later, power of two) |
| Ext-fb | **Does not exist** |

### 4.5 Decisions table

| Topic | Decision | Date |
|-------|----------|------|
| Name | HypercubeWTF | 2026-08-04 |
| S1 | `vtx_input[v] = x[(v XOR c) & (N-1)]`; weights fixed | 2026-08-04 |
| Input path | Field inject + stock-style `W_in` gather | 2026-08-04 |
| Map-side XOR | Out of scope | 2026-08-04 |
| Field contract | size==N or throw; trust [-1,1]; packing caller-owned | 2026-08-04 |
| Dependencies | Start-from hESN Reservoir/Readout + HCNN upstream; **WTF-specific** fork (drop ESN stream/fb baggage) | 2026-08-04 |
| First copy sources | hESN **`feedback`** @ `ae3fb64…`; HCNN `main` @ `20bbb23…` (on-disk lock) | 2026-08-04 |
| External feedback | **No port. No port. No port.** | 2026-08-04 |
| Readout sample | Once per episode at end | 2026-08-04 |
| **B** default | **1** | 2026-08-04 |
| **T** default | **N**; `T > N` allowed (wrap field index) | 2026-08-04 |
| **s0** | Frozen U[-0.5,0.5]^(N×M); **separate IC seed**; intelligent full-depth load each episode | 2026-08-04 |

---

## 5. Open questions

| ID | Question | Notes |
|----|----------|-------|
| Q8 | First demo | **Done** — `wtf_synth` + `wtf_mnist` (DualPlane / PadLow); Raman later |
| — | Train collect / batch API shape | revisit later (`RunEpisode` + collect vs other) |

---

## 6. API sketch (not frozen)

```text
WTFConfig
  reservoir: ReservoirConfig   // dim, seed, SR, leak, history_depth, input_scaling, …
  ic_seed:   uint64_t          // separate from reservoir weight seed
  readout:   ReadoutConfig
  episode_T: size_t            // default N
  readout_slices: size_t B     // default 1; end-of-episode pack only

class WTF
  explicit WTF(const WTFConfig&);
  void RunEpisode(span<const float> x);   // throws if x.size() != N
  // train / predict surface — revisit later
  vector<float> Predict(span<const float> x);
```

**v1 non-goals:** closed-loop free-run; replacing ESN for real streams; Python
bindings; any external-feedback API.

---

## 7. Repo layout (intended)

```text
HypercubeWTF/
  CMakeLists.txt
  LICENSE
  README.md
  docs/project.md
  WTF.h
  Reservoir.* / Readout.*     // WTF fork (started from hESN sources)
  third_party/HypercubeCNN/   // from HypercubeCNN upstream
  examples/
  tests/
```

---

## 8. Workplan

Sequential gates. Docs may draft ahead; code waits on exit criteria.

| Phase | Work | Exit |
|-------|------|------|
| **0** Charter | This file; design lock | **Done** — source trees recorded (§3) |
| **1** Skeleton | CMake C++23; vendor HCNN; fork Reservoir/Readout (strip fb / stream cruft); empty `WTF`; README | **Done** — Release build + `wtf_smoke` |
| **2** Episode core | Field inject; orbit; IC load; end sample; tests | **Done** — `RunEpisode` + smoke tests |
| **3** Readout | End pack (B); train/predict path | **Done** — CollectEpisode / TrainOnCollected / Predict* |
| **4** Demos | Example packing (PadLow + DualPlane); `wtf_synth`; `wtf_mnist` | **Done** — see `examples/` |
| **5** Polish | C++ perf, docs, richer example packing as needed | Solid C++ core |

**Python SDK / bindings:** not in this workplan. Only after the C++ core is
solid. Do not scaffold Python early.

---

## 9. References

| Doc | Use |
|-----|-----|
| `HypercubeESN/docs/Reservoir.md` | Starting physics (cube, gather, history) — not binding API |
| `HypercubeESN/docs/Readout.md` | HCNN head / multi-slice ideas |

---

## 10. Next

1. **Phase 5** — polish, docs, optional full-MNIST / larger-dim campaigns.
2. Keep `wtf_smoke` + `wtf_synth` as fast gates; `wtf_mnist` needs IDX data.
3. Python / bindings only after the C++ core is solid (not before).

---

## 11. Changelog

| Date | Change |
|------|--------|
| 2026-08-04 | Charter established; S1 field orbit; vendor model; end-only readout; no bake-off |
| 2026-08-04 | Defaults **B=1**, **T=N**; `T>N` wraps; separate **ic_seed**; intelligent **s0** load; WTF Reservoir fork (no ESN fb / stream baggage); end-state wording clarified |
| 2026-08-04 | First copy: HypercubeESN **`feedback`** (not main) @ ae3fb64…; HypercubeCNN main @ 20bbb23… |
| 2026-08-04 | Phase 3 collect/train/predict; smoke episode contract (`T>N`, `B=2`, IC seed split) |
| 2026-08-04 | Phase 4: `wtf_synth`, `wtf_mnist`; pack = PadLow + DualPlane only (no resize/letterbox/bit-map) |
