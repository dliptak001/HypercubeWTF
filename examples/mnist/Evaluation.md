# Evaluation: Hypercube Reservoir as a White-Noise Filter

## The HypercubeAI substrate

HypercubeWTF sits in the same family as **HypercubeESN** and **HypercubeCNN**:
computation lives on a **hypercube** of `N = 2^dim` vertices. Neighbors, gathers,
and spatial structure are cube-native — not a generic dense RNN with the graph
painted on afterward.

| Product | Natural data | Role of the hypercube |
|---------|--------------|------------------------|
| **HypercubeESN** | Low-dimensional **streams** over time | Frozen **reservoir** stepped each sample; multi-slice state → HypercubeCNN readout |
| **HypercubeCNN** | Static patterns already on the cube | Trainable **spatial** conv/pool on the cube (no recurrent reservoir) |
| **HypercubeWTF** | Static high-dimensional fields (**no** intrinsic time) | Same **frozen hypercube reservoir** discipline as ESN, driven for a short **episode** per sample, then HypercubeCNN on the **end state** |

Reservoir code in WTF started from **HypercubeESN** (cube dynamics, delay line,
input gather, spectral radius, leak). Readout is the same **HypercubeCNN** façade
family. The hypercube is not an implementation detail; it is the shared substrate.

---

## The preprocessor is a reservoir

In classical reservoir computing (and in HypercubeESN):

- Recurrent weights are **frozen**
- Only a **readout** is trained
- Nonlinear dynamics expand and mix the drive into a rich state

**HypercubeWTF uses that same idea.** The WTF preprocessor is a **hypercube
reservoir**: fixed `W`, fixed input scaling, leak, spectral radius, history depth
`M`, and a frozen initial condition reloaded each sample. **No reservoir weights
are learned.** Only the HCNN head trains.

So when this document says the pipeline “filters” noise, it is not introducing a
separate denoise network. It is asking what a **mild, contracting hypercube
reservoir episode** does to a static field before classification.

### How WTF uses the reservoir differently from HypercubeESN

| | HypercubeESN | HypercubeWTF (this evaluation) |
|---|--------------|--------------------------------|
| Input | Stream `u_t` (low-dim channels over real time) | One **length-N field** `x` packed onto the cube (e.g. MNIST) |
| Time | Stream time = model time | **Synthetic episode time** inside one static sample |
| Drive | New `u_t` each step | Same `x`, re-registered each pass (affine walk / XOR index) |
| Features | Often multi-slice along the stream | Default: **end-of-episode** state only (`B = 1`) |
| Product emphasis here | Temporal modeling of sequences | **Optional preprocessor** — under AWGN, a **white-noise filter** |

Same RC contract (frozen cube reservoir + trained head). Different **use**: ESN
follows a stream; WTF turns a static hypercube field into a short driven
**orbit** (trajectory through reservoir state) and reads the end.

### Bypass vs reservoir (the A/B)

```text
Pack        — map the input onto the length-N hypercube field
Bypass      — features = packed field (reservoir unused)
Reservoir   — features = end state after T drive steps on the frozen cube reservoir
```

In code, that switch is `episode.bypass_reservoir`. **Bypass** asks: is the
hypercube pack + HCNN enough? **Reservoir** asks: does a short frozen episode on
the same cube improve the features — especially when the pack is corrupted?

On clean MNIST the reservoir is easy to treat as optional: pack and readout
already do most of the work. Under strong **additive white Gaussian noise
(AWGN)** on the packed field, that same reservoir becomes the product story of
this document: a **white-noise filter** that holds accuracy the identity path
cannot.

Numbers: full MNIST (60k / 10k) via `wtf_mnist` (release). Train is always
**clean**; noise is **eval-only**. Live knobs may drift; claims pin to the logged
recipe in the appendix.

---

## The claim

**With the reservoir on, HypercubeWTF matches pack-only accuracy on clean data
and substantially outperforms pack-only under strong white test noise.**

| Condition | Bypass (pack → readout) | Reservoir (pack → episode → readout) |
|-----------|-------------------------|--------------------------------------|
| Clean or mild AWGN | Strong (~0.98) | Similar (~0.97–0.98) |
| Strong AWGN (σ = 0.5) | Collapses (~0.85) | Holds (~0.93) |

The gap at σ = 0.5 is about **eight to nine percentage points**, stable across
noise seeds. That is the product-relevant result.

---

## Protocol

The **only** intentional factor in the headline A/B is `bypass_reservoir` true
versus false. Within each noise condition, packing, train set, readout family,
and (when matched) noise seed are shared. The reservoir arm uses a fixed mild
contracting recipe (appendix). Bypass ignores reservoir dynamics; the hypercube
still appears in **packing** (PadLowCenter onto `N = 1024` vertices).

**Test noise** (`kTestNoiseSigma`) is i.i.d. Gaussian on every packed vertex after
pack and before predict — flat spectrum, Gaussian amplitudes, additive. Not
train/collect noise (`train_input_noise_sigma`), not geometric aug.

**Train** stays clean: no field noise, no spatial aug in these logs. The head
learns on clean features and is scored on clean or corrupted test fields — a
domain-shift stress test, not matched noisy training.

---

## Strong white noise — the reservoir earns its keep

At σ = 0.5, feeding the noisy pack straight into the readout fails hard: test
accuracy sits near **0.84–0.85**. The same class of readout, fed the **end state
of the hypercube reservoir episode**, sits near **0.93**.

That is not a one-draw fluke. Across multiple independent noise seeds, reservoir
test accuracy stays within a few tenths of a point of **0.93** (logged values
from **0.927** to **0.931**). Bypass remains in the mid-0.84s under the same
noise family.

Cutting readout epochs does not rescue bypass. Collected accuracy stays near
perfect while noisy test accuracy stays broken — classic train/test mismatch, not
“train longer.” The reservoir reduces that mismatch: it maps noisy fields into a
region the clean-trained head still understands.

**Functional reading:** under strong AWGN on the field, the frozen hypercube
reservoir **filters a large fraction of white noise** before classification. Same
module family as HypercubeESN; here the job is preprocessing a static cube field,
not following a stream.

---

## Clean and mild noise — nearly transparent

A filter that only works by destroying the signal is useless. On **clean** test
data, bypass is already excellent (~**0.979**). With the reservoir on, accuracy
reaches the same level (~**0.979** in the primary clean pair; other reservoir
logs sit within about a point).

At **mild** noise (σ = 0.1), bypass is slightly ahead (**0.980** vs **0.968**).
There is little white noise to remove, so the preprocessor is optional.

Operating picture:

- **Clean / light noise** — bypass and reservoir are comparable; the episode need
  not cost meaningful accuracy.
- **Heavy white noise** — the reservoir is the difference between mid-80s and
  low-90s.

---

## What has to stay true for the filter to work

| Change | Outcome |
|--------|---------|
| `tanh` → `\|tanh\|` on reservoir units | Clean: nearly tied; **high noise: fails badly** |
| Fewer readout epochs (bypass only) | Does **not** fix σ = 0.5 bypass |

Signed saturating dynamics matter under noise — the same odd nonlinearity
familiar from ESN-style reservoirs. Folding to non-negative magnitudes throws
away polarity the filter needs. Weights stay **frozen**; this is not a learned
denoiser with a noise loss. Results use a **mild contracting** recipe (low
spectral radius, small input scale, modest episode length) on the cube; see
appendix.

---

## How to read the numbers

**Test accuracy** is the decision metric (10k held-out MNIST images).  
**Accuracy on collected** is fit to the clean training feature buffer (including
any best-epoch holdout tail). Under strong test noise, a huge collected–test gap
on bypass means the head memorized clean packs and met a different distribution
at test. The reservoir shrinks that gap without matching train noise.

Representative timing (wall clock, one machine): reservoir test is several times
slower than bypass (short episode per image); collect+train is dominated by the
HCNN when epochs are high. Exact seconds vary; the accuracy story does not depend
on them.

---

## What this evaluation does not claim

- Superiority to classical image denoisers (Gaussian, Wiener, BM3D, …) — not
  measured here.
- Robustness to blur, occlusion, adversarial noise, or non-white corruptions.
- That every reservoir recipe is a white-noise filter — results use the mild
  hypercube recipe in the appendix.
- That the reservoir replaces good packing or a competent HypercubeCNN head on
  clean data.
- That HypercubeWTF replaces HypercubeESN for streams — different data modality;
  shared substrate and RC discipline.

What it **does** support: on the hypercube stack, a short frozen **reservoir
episode** can be offered as an **optional white-noise filter** for length-N
fields, with little clean-data tax in this regime, and a large multi-seed gain
under strong AWGN versus identity (bypass) features — same frozen-reservoir /
trained-head contract as HypercubeESN, applied to static cube fields.

---

## Reproduce

1. Build `wtf_mnist` (release recommended).
2. Align `MakeWTFConfig()` with the reservoir recipe in the appendix (or the
   current family in `wtf_mnist.cpp`).
3. Set `kTestNoiseSigma` and `kTestNoiseSeedBase`; set
   `episode.bypass_reservoir` **true** or **false** for the two arms.
4. Keep `train_input_noise_sigma = 0` and train aug off so train stays clean.

Primary demo path: `examples/mnist/wtf_mnist.cpp`.  
Charter / ecosystem notes: [`docs/project.md`](../../docs/project.md).

---

## Appendix A — Reservoir recipe

Logged configuration when the reservoir is **on**. Bypass uses the same
pack/readout/noise protocol; reservoir knobs do not affect bypass features.

```text
N=1024  T=20  B=1  M=4  ic_seed=12
reservoir.seed=13871537636959942979
SR_target=0.4  SR_realized≈0.3988  leak=0.5  in_scale=0.005  bias_scale=0
readout: layers=1  weights=82122  pooling=max  activation=NONE
         lr_max=0.0015  epochs=100
pack=PadLowCenter  train=60000  test=10000  aug=off
```

Some multi-seed σ = 0.5 reservoir rows used minor head/epoch variants of this
family; test accuracy remained ~0.93. The factor under study is always
**bypass vs reservoir**.

---

## Appendix B — Tabulated logs

### Strong AWGN (σ = 0.5)

**Bypass vs reservoir (three noise seeds):**

| seed_base | Bypass test_acc | Reservoir test_acc | Δ |
|-----------|-----------------|--------------------|---|
| `0x7E57` | 0.847 | 0.929 | +8.2 pp |
| `0x3E57` | 0.846 | 0.931 | +8.5 pp |
| `0x1E57` | 0.844 | 0.931 | +8.7 pp |

Reservoir collected ≈ 0.978 on those runs.

**Additional reservoir seeds (σ = 0.5):**

| seed_base | collected | test_acc |
|-----------|-----------|----------|
| `0x1E57` | 0.992 | 0.930 |
| `0x7E57` | 0.992 | 0.927 |

**Bypass epoch check (σ = 0.5):** epochs 40 → 0.846; epochs 20 → 0.849 / 0.844
(seed-dependent). No recovery.

### Mild AWGN (σ = 0.1, seed `0x1E57`)

| Arm | collected | test_acc |
|-----|-----------|----------|
| Reservoir | 0.978 | 0.968 |
| Bypass | 0.999 | 0.980 |

### Clean (σ = 0)

| Arm | collected | test_acc |
|-----|-----------|----------|
| Bypass | 0.999 | 0.979 |
| Reservoir | 0.992 | 0.979 |

(Other reservoir clean logs: 0.970–0.977 depending on minor head/epoch settings;
all near bypass.)

### Cross-σ snapshot

| Test noise | Bypass | Reservoir |
|------------|--------|-----------|
| off | ~0.979 | ~0.979 |
| N(0, 0.1) | **0.980** | 0.968 |
| N(0, 0.5) multi-seed | ~0.84–0.85 | **~0.93** |

---

## Appendix C — Scope notes

- Binary: `cmake-build-release/wtf_mnist.exe`; data under repo `data/`.
- One early bypass log used an older reservoir header in the printout; bypass
  ignores reservoir dynamics, so features remain pack-only.
- `|tanh|` high-noise failure is operator-confirmed; clean near-tie was
  instrumented (~0.975 vs ~0.977).
- Further work: denser σ ladder, classical denoise baselines, matched train
  noise, non-MNIST fields.
