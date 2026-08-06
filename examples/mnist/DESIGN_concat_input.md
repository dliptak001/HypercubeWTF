# TEMP: Dual-block readout features (orbit + field)

**Status:** design notes only — not implemented.  
**Scope:** WTF episode feature packing for demos (MNIST first).  
**Folder:** `examples/mnist/` (temp; promote or delete after a decision).

---

## 1. Motivation

Today the readout sees only the **end-of-orbit** reservoir pack (length N when B=1).  
We want a residual-style second block: the **same drive field** that fed the reservoir, so the HCNN can mix:

1. Transformed / mixed state (orbit output)
2. Raw packed input (field)

Later we may replace block 2 with a **second orbit** of the same reservoir but with walk index `c` starting somewhere other than zero. Same width (2N); only the source of block 2 changes.

---

## 2. Current behavior (baseline)

```text
pack image → length-N field x
  → reservoir episode (T drive passes, c = 0,1,...,T-1)
  → features = end pack, size B*N
  → HCNN readout (features = 2^readout.dim)
```

Defaults of interest:

| Knob | Typical MNIST | Role |
|------|---------------|------|
| B (`readout_slices`) | 1 | Ages packed from delay line |
| N | 1024 (dim 10) | Field / orbit width |
| FeatureSize | B*N = N | Readout input |
| readout.dim auto | reservoir.dim + log2(B) | Must match FeatureSize |

`bypass_reservoir` skips the orbit and uses `x` alone as features (B=1).

---

## 3. Proposed near-term layout

**Constraint (agreed): B = 1 only.**  
HCNN feature count must be a power of two. With B=1:

```text
FeatureSize = 2N = 2^(dim + 1)
readout.dim auto = reservoir.dim + 1
```

Layout (sample-major, contiguous floats):

```text
features[0 .. N)     = orbit end pack (today's features)
features[N .. 2N)    = drive field x   (input to the reservoir)
```

Name sketch for the knob (not frozen):

```text
EpisodeConfig::concat_input   // default false for backward compatibility
```

When `concat_input == true`:

- Requires `readout_slices == 1`
- Requires `!bypass_reservoir`
- Collect / RunEpisode / Predict all produce length-2N features
- Readout constructed with dim = dim + 1 (auto) or explicit match

When `concat_input == false`: unchanged (FeatureSize = B*N).

---

## 4. Interaction with existing knobs

### 4.1 `bypass_reservoir`

Mutually exclusive with `concat_input`.  
Bypass already is “field only”; doubling as `[x|x]` is useless. Construct-time error if both true.

### 4.2 `train_input_noise_sigma` (train/collect-only)

Noise is applied to the field **before** the orbit.  
Block 2 should be that **same** post-noise field (the actual drive), not a clean copy.  
Inference (`RunEpisode` / `Predict`) has no collect noise → block 2 is clean `x`.

### 4.3 B > 1

Out of scope.  
`N*(B+1)` is not a power of two for power-of-two B except B=1.  
No padding plan in this temp doc; reject at construct if `concat_input && B != 1`.

### 4.4 Spatial pack / aug

Unaffected. Pack (and train-only SpatialAug) still produce length-N `x`; only the feature **assembly** after the episode changes.

---

## 5. Future: alt-c second orbit (not in v1)

Same 2N layout; reinterpret block 2:

| Mode | Block 1 | Block 2 |
|------|---------|---------|
| Orbit only (today) | orbit @ schedule c=0.. | — |
| Orbit + field (v1) | orbit @ c=0.. | raw field `x` |
| Orbit + alt-c (later) | orbit @ c=0.. | orbit @ c=c0.. (same weights, same `x`, different walk start) |

Possible later enum (illustrative only):

```text
enum class FeatureSource {
    OrbitOnly,       // B*N
    OrbitPlusField,  // 2N, B=1
    OrbitPlusAltC,   // 2N, B=1; second episode with non-zero c start
};
```

Alt-c details (open): choice of `c0`, whether T is the same, whether both orbits share s0, cost = ~2× episode FLOPs when both run.

v1 does **not** implement alt-c; only leave layout room (second block always length N).

---

## 6. Cost (T small / modest, T < 32)

| Piece | Change vs orbit-only |
|-------|----------------------|
| Episode (reservoir) | None for Orbit+Field |
| Feature pack | +N memcpy |
| HCNN | Sees 2N inputs → wider first stage / larger flatten head |
| Alt-c later | ~2× orbit cost if both orbits always run |

With T < 32, episode cost stays modest; the main new training cost is the larger readout (especially the linear head after flatten).

---

## 7. Implementation sketch (when approved)

Touch points (expected):

1. **`EpisodeConfig`** — `bool concat_input = false` (+ docs).  
2. **`WTF` construct** — validate B=1, !bypass; set `expected_readout_dim = dim + 1`.  
3. **`FeatureSize()`** — `concat_input ? 2*N : B*N`.  
4. **`PackEndFeaturesFrom` / `RunEpisodeOn`** — pack orbit into `[0,N)`, copy `x` into `[N,2N)`.  
5. **Bypass / parallel collect** — same rules; second half uses the field actually used as drive (post-noise on collect).  
6. **`print_config.h`** — banner flag.  
7. **`wtf_smoke`** — FeatureSize=2N, last half equals `x`, reject bad combos.  
8. **`wtf_mnist`** — opt-in knob only; do not rewrite unrelated experiment knobs.

No HypercubeCNN API change if dim auto already follows `2^dim` features.

---

## 8. Open questions

1. Default for MNIST experiments: leave off until A/B, or turn on once landed?  
2. Exact flag name: `concat_input` vs `feature_concat_field` vs enum from day one?  
3. Should `LastFeatures()` document layout explicitly for debug dumps?  
4. Alt-c: fixed `c0` knob vs sweep; one shared s0 or independent IC?

---

## 9. Non-goals (this temp doc)

- B > 1 dual-block packing  
- Changing reservoir dynamics or drive schedule for v1  
- Implementing alt-c  
- Python bindings / checkpoint format migration for the new feature width  
- Auto-enabling when bypass is on  

---

## 10. Decision log

| Date | Note |
|------|------|
| 2026-08-05 | Discussion: 2N = orbit + field; B=1; alt-c is later; no code yet. |
| 2026-08-05 | This temp design file created under `examples/mnist/`. |
