# Results: high test noise × bypass vs orbit

**Date:** 2026-08-05  
**Binary:** `cmake-build-release/wtf_mnist.exe`  
**Data:** `C:\CLion\HypercubeWTF\data` (full MNIST 60k / 10k)  
**Purpose:** Does a frozen reservoir orbit help when the **test** packed field is heavily corrupted, even if train is clean?

---

## 1. Protocol

| Stage | Corruption |
|-------|------------|
| Train collect | **Clean** pack (`aug=off`, `episode.input_noise_sigma=0`) |
| Train labels | Standard MNIST |
| Test | Pack → **i.i.d. Gaussian on packed length-N field** → `PredictClass` |

**Test noise (demo knob `kTestNoiseSigma`):**

- `N(0, 0.5)` added **after** `PackSample`, **before** `PredictClass`
- White in **vertex index** (flat spectrum on the field; on `[0,784)` that is spatially white on the 28×28 view; center-crop tail `[784,1024)` gets an independent white layer)
- Deterministic per sample: `seed_base=0x7E57` (`kTestNoiseSeedBase`)
- **Not** `episode.input_noise_sigma` (collect-only; left at 0)
- **Not** train spatial aug

**A/B factor (only intentional difference):**

| Run | `bypass_reservoir` | Features |
|-----|--------------------|----------|
| A | `true` | packed field → readout (no orbit) |
| B | `false` | packed field → T-step orbit → end pack → readout |

Same reservoir weights, IC, pack mode, readout recipe, seeds, train/test splits, and test noise stream.

---

## 2. Shared knobs (both runs)

From run headers (release):

```text
N=1024  T=32  B=1  M=1  ic_seed=12
reservoir.seed=13871537636959942979
SR_target=0.95  SR_realized≈0.950749
leak=1  in_scale=0.015  bias_scale=0.003
pack=PadLowCenter  full=28x28@[0,784)  center=15x16@(6,6)  pattern=N=1024
train=60000  test=10000  epochs=40
readout: layers=1  weights=82122  pooling=max  activation=leaky_relu  lr_max=0.0015
aug=off
test_noise=N(0,0.5) on packed field seed_base=0x7E57
input_noise_sigma=0
collect_threads=0 (auto)
```

Pack fills N exactly (PadLowCenter). Soft CI floor in demo is unrelated to this table.

---

## 3. Results

| Metric | bypass=`true` | orbit (bypass=`false`) | Notes |
|--------|---------------|------------------------|--------|
| **test_acc** | **0.847** (8467/10000) | **0.911** (9111/10000) | **+6.4 pp** with orbit |
| acc_on_collected | 0.999 | 0.991 | Clean train; bypass fits train slightly easier |
| collect+train | 307.3 s | 286.2 s | Wall clock; train path dominates |
| test | 2.7 s | 19.5 s | Orbit cost on 10k × T=32 |
| total | 310.0 s | 305.7 s | |

### Raw log excerpts

**Bypass on (A):**

```text
bypass_reservoir=true
test_noise=N(0,0.5) on packed field seed_base=0x7E57
acc_on_collected=0.999 test_acc=0.847 (8467/10000)
time 307.3+2.7=310.0s (collect+train|test|total)
```

**Orbit on (B):**

```text
bypass_reservoir=false
test_noise=N(0,0.5) on packed field seed_base=0x7E57
acc_on_collected=0.991 test_acc=0.911 (9111/10000)
time 286.2+19.5=305.7s (collect+train|test|total)
```

---

## 4. Interpretation

1. **Train clean / test noisy** is a domain-shift robustness test, not matched noisy training.
2. Under **σ = 0.5** white field noise, the frozen orbit is **not** a no-op: **+6.4 pp** test accuracy vs pack→readout alone.
3. Plausible mechanism: short stable dynamics act as a **mixing / contracting preprocessor**. The head trains on `F(clean x)` (or on `x` when bypassing). At test time, `F(x + noise)` can sit closer to the clean training manifold than raw `x + noise` does.
4. This does **not** contradict earlier clean-test experience that bypass ≈ orbit on clean MNIST. Clean task accuracy can be pack+HCNN dominated; **heavy test noise** is where the orbit earns its keep in this setup.
5. Cost: test wall time ~7× higher with orbit here (2.7 s → 19.5 s); collect+train similar in this pair (timing dominated by HCNN train / parallel collect noise).

### What this is not

- Not a claim that the reservoir **sharpens** digits.
- Not a claim of learned denoising (encoder is frozen; no noise in the train loss).
- Not validated for blur, occlusion, adversarial noise, or other σ values.
- Not a multi-seed / multi-readout sweep (single seed, 1-layer max-pool leaky-ReLU head as logged).

---

## 5. Related context

| Context | Takeaway |
|---------|----------|
| Clean best-run record in `wtf_mnist.cpp` header | ~0.975 test (orbit-era knobs; activation NONE in that record) — different from this noisy A/B head log |
| `mnist_scans.txt` | Informal clean-run scrapbook; not this protocol |
| `DESIGN_concat_input.md` | Separate idea: readout `[orbit \| field]`; not used in these runs |
| Demo knobs | `kTestNoiseSigma`, `episode.bypass_reservoir` |

---

## 6. Suggested follow-ups

1. **Clean test control** with the **same** readout/reservoir knobs as this A/B (σ=0) for both bypass on/off — quantify drop-from-clean.
2. **σ ladder** (e.g. 0.1, 0.25, 0.5, 0.75): where orbit starts to win.
3. **Matched train noise** (corrupt collect the same way, or use collect `input_noise_sigma`) vs train-clean/test-noisy.
4. **T sweep** at fixed σ=0.5 — is robustness from integration depth?
5. Optional: concat `[orbit \| field]` under the same test noise (see design note).

---

## 7. Decision log

| Date | Note |
|------|------|
| 2026-08-05 | High test noise σ=0.5 A/B: orbit 0.911 vs bypass 0.847; documented here. |
