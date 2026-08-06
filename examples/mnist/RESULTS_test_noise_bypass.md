# Results: test field noise × bypass vs orbit

**Source log:** operator dump `zzz.txt` (2026-08-05)  
**Binary:** `cmake-build-release/wtf_mnist.exe`  
**Data:** `C:\CLion\HypercubeWTF\data` (MNIST 60k train / 10k test)  
**Scope:** Train **clean**; optional white Gaussian noise on the **packed test field** only. Compare pack→readout (`bypass_reservoir=true`) vs pack→orbit→readout (`false`).

Live knobs in `wtf_mnist.cpp` are experiment-owned; this file records **logged runs only**.

---

## 1. Protocol

| Stage | Setting |
|-------|---------|
| Train | Clean pack (`aug=off`, `input_noise_sigma=0`) |
| Test | Pack → optional `N(0,σ)` on length-N field → `PredictClass` |
| Noise | Demo `kTestNoiseSigma` / `kTestNoiseSeedBase` (not collect noise) |
| Pack | PadLowCenter, 28×28 @ [0,784), center 15×16 @ (6,6) → N=1024 |

**Bypass:** features = packed field (reservoir dynamics unused).  
**Orbit:** features = end-of-episode pack after T drive steps.

Unless a row says otherwise, **primary recipe** is:

```text
N=1024  T=20  B=1  M=4  ic_seed=12
reservoir.seed=13871537636959942979
SR_target=0.4  SR_realized≈0.3988  leak=1  in_scale=0.005  bias_scale=0
readout: layers=1  weights=82122  pooling=max  activation=leaky_relu  lr_max=0.0015
epochs=40  train=60000  test=10000  pack=PadLowCenter  aug=off
```

---

## 2. Headline: σ = 0.5 (high noise), three noise seeds

Orbit recipe = primary (§1). Bypass uses the same pack/readout/noise; reservoir knobs do not change bypass features.

| seed_base | Bypass test_acc | Orbit test_acc | Δ (orbit − bypass) |
|-----------|-----------------|----------------|--------------------|
| `0x7E57` | 0.847† | **0.929** (9290/10000) | **+8.2 pp** |
| `0x3E57` | 0.846 | **0.931** (9312/10000) | **+8.5 pp** |
| `0x1E57` | 0.844‡ | **0.931** (9311/10000) | **+8.7 pp** |

† First bypass log used an older header (`T=32 M=1 SR=0.95 IS=0.015 bias=0.003`); with bypass on that does not change features.  
‡ Matched primary header; one of the epochs=20 checks (see §3) — test still ~0.84.

**Orbit stability across seeds (σ=0.5, epochs=40):**

| seed_base | collected | test_acc | gap | test/collected |
|-----------|-----------|----------|-----|----------------|
| `0x7E57` | 0.978 | 0.929 | 0.049 | 0.950 |
| `0x3E57` | 0.978 | 0.931 | 0.047 | 0.952 |
| `0x1E57` | 0.978 | 0.931 | 0.047 | 0.952 |

Orbit test_acc ≈ **0.930 ± 0.001** over three seeds — **not a fluke**.  
Bypass under the same σ stays ≈ **0.84–0.85**.

**Operator conclusion (log):** readout has significantly higher test scores with the orbit in a high-noise environment.

### Timing (representative)

| Arm | collect+train | test | total |
|-----|---------------|------|-------|
| Orbit σ=0.5 | ~112–116 s | ~8.6–8.9 s | ~120–125 s |
| Bypass σ=0.5 | varies with epochs | ~1.2–2.7 s | — |

---

## 3. Bypass checks: epochs and “overtraining”

Collected accuracy for bypass is ~0.999. Operator tried fewer epochs on the same noise seed.

| epochs | seed_base | collected | test_acc | Note |
|--------|-----------|-----------|----------|------|
| 40 | `0x3E57` | 0.999 | 0.846 | |
| 20 | `0x3E57` | 0.998 | 0.849 | |
| 20 | `0x1E57` | 0.998 | 0.844 | |

**Operator conclusion:** overtraining is **not** the issue — cutting epochs does not close the gap to orbit; noisy test remains ~0.84–0.85 for bypass.

---

## 4. Lower noise: σ = 0.1 (seed `0x1E57`)

Primary recipe, epochs=40.

| Arm | collected | test_acc | gap |
|-----|-----------|----------|-----|
| Orbit | 0.978 | 0.968 (9683/10000) | 0.010 |
| **Bypass** | 0.999 | **0.980** (9796/10000) | 0.019 |

At **mild** test noise, **bypass wins** (~+1.2 pp). Orbit no longer helps (slightly hurts vs pack-only).

---

## 5. Clean test: σ = 0 (`test_noise=off`)

Primary recipe, epochs=40.

| Arm | collected | test_acc | gap |
|-----|-----------|----------|-----|
| Orbit | 0.978 | 0.970 (9701/10000) | 0.008 |
| **Bypass** | 0.999 | **0.979** (9788/10000) | 0.020 |

On **clean** test, pack→readout is slightly better (~+0.9 pp). Matches the earlier “bypass ≈ orbit / bypass fine on clean MNIST” story under this readout.

---

## 6. Cross-σ summary (primary orbit recipe)

Same pack, readout, reservoir (when used), train clean. Prefer matched seed when both arms listed.

| Test noise | Bypass test_acc | Orbit test_acc | Who wins? |
|------------|-----------------|----------------|-----------|
| off | **0.979** | 0.970 | Bypass (~+0.9 pp) |
| N(0, **0.1**) seed `0x1E57` | **0.980** | 0.968 | Bypass (~+1.2 pp) |
| N(0, **0.5**) 3 seeds | ~**0.84–0.85** | ~**0.93** | **Orbit (~+8–9 pp)** |

```text
                    clean / low noise          high noise (σ=0.5)
bypass (pack only)  slightly better            collapses to ~0.85
orbit (frozen F)    slightly worse / similar   holds ~0.93  ← value here
```

---

## 7. Interpretation

1. **High white test noise (σ=0.5):** frozen orbit is a strong **robustness preprocessor** — ~+8–9 pp vs bypass, stable across three noise seeds.
2. **Clean and mild noise (σ=0.1):** pack+HCNN is enough; orbit is neutral-to-slightly worse. No free lunch.
3. **Collected vs test:** orbit collected stays ~0.978 with test ~0.93 under σ=0.5 (gap ~0.05). Bypass collected ~0.999 with test ~0.85 under σ=0.5 is **train/test domain shift**, not fixed by fewer epochs.
4. Mechanism (hypothesis): contracting dynamics (low SR, small IS, no bias, M=4, modest T) map noisy fields closer to the clean training manifold than the raw noisy pack does.
5. Not claimed: learned denoise, sharpening, non-white corruptions, or multi-readout sweeps.

---

## 8. Raw log anchors (from `zzz.txt`)

**Bypass σ=0.5 (illustrative):**

```text
bypass_reservoir=true  test_noise=N(0,0.5) seed_base=0x7E57
acc_on_collected=0.999 test_acc=0.847 (8467/10000)
```

**Orbit σ=0.5 (three seeds):**

```text
bypass_reservoir=false  M=4 SR=0.4 IS=0.005 bias=0 T=20
seed 0x7E57: collected=0.978 test_acc=0.929
seed 0x3E57: collected=0.978 test_acc=0.931
seed 0x1E57: collected=0.978 test_acc=0.931
```

**σ=0.1 seed 0x1E57:** orbit 0.968 / bypass 0.980  
**σ=off:** orbit 0.970 / bypass 0.979

---

## 9. Follow-ups

- denser σ ladder (e.g. 0.2, 0.3, 0.4) to locate the crossover  
- more noise seeds at σ=0.5 (already stable; optional)  
- matched train noise vs train-clean/test-noisy  
- T / M / SR ablations at fixed seed  
- optional `[orbit | field]` concat under high noise (`DESIGN_concat_input.md`)

---

## 10. Decision log

| Date | Note |
|------|------|
| 2026-08-05 | Full rewrite of this file from operator `zzz.txt`: multi-seed σ=0.5 orbit win; σ=0.1 and clean favor bypass; epochs cut does not fix bypass under high noise. |
