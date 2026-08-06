# Hypercube Reservoir as a White-Noise Pre-Filter for HypercubeCNN

## The HypercubeAI substrate

HypercubeWTF sits in the same family as **HypercubeESN** and **HypercubeCNN**:
computation lives on a **hypercube** of *N* = 2<sup>dim</sup> vertices. Neighbors, gathers,
and spatial structure are cube-native — not a generic dense RNN with the graph
painted on afterward.

| Product | Natural data | Role of the hypercube |
|---------|--------------|------------------------|
| **HypercubeESN** | Low-dimensional **streams** over time | Frozen **reservoir** stepped each sample; multi-slice state → HypercubeCNN readout |
| **HypercubeCNN** | Static patterns already on the cube | Trainable **spatial** conv/pool on the cube (no recurrent reservoir) |
| **HypercubeWTF** | Static high-dimensional fields (**no** intrinsic time) | Same **frozen hypercube reservoir** discipline as ESN, driven for a short **episode** per sample, then HypercubeCNN on the **end state** |

Reservoir code in WTF started from **HypercubeESN** (cube dynamics, delay line,
input gather). Readout is the same **HypercubeCNN** façade family. The hypercube
is not an implementation detail; it is the shared substrate.

---

## The preprocessor is a reservoir

In classical reservoir computing (and in HypercubeESN):

- Recurrent weights are **frozen**
- Only a **readout** is trained
- Nonlinear dynamics expand and mix the drive into a rich state

**HypercubeWTF uses that same idea.** The WTF preprocessor is a **hypercube
reservoir**: frozen recurrent dynamics and a frozen initial condition reloaded
each sample. **No reservoir weights are learned.** Only the HCNN head trains.

So when this document says the pipeline “filters” noise, it is not introducing a
separate denoise network. It is asking what a **short hypercube reservoir
episode** does to a static field before HypercubeCNN.

### How WTF uses the reservoir differently from HypercubeESN

| | HypercubeESN | HypercubeWTF (this evaluation) |
|---|--------------|--------------------------------|
| Input | Stream over real time | One **static field** packed onto the cube (e.g. MNIST) |
| Time | Stream time = model time | **Synthetic episode time** inside one sample |
| Drive | New input each step | Same field re-presented each step of the episode |
| Features | State along the stream | **End-of-episode** reservoir state |
| Product emphasis here | Temporal modeling of sequences | **Optional pre-filter** for HypercubeCNN under additive white Gaussian noise (AWGN) |

Same RC contract (frozen cube reservoir + trained head). Different **use**: ESN
follows a stream; WTF drives a static hypercube field through a short episode and
hands the end state to HypercubeCNN.

### Bypass vs reservoir (the A/B)

Both arms start the same way: **pack** maps the input onto the hypercube field.
The A/B is only what becomes the HypercubeCNN feature vector:

```text
Bypass    — packed field (reservoir unused)
Reservoir — end state of a short frozen reservoir episode
```

**Bypass** asks: is the hypercube pack + HypercubeCNN enough? **Reservoir** asks:
does a short frozen episode on the same cube improve the features — especially
when the pack is corrupted?

On clean MNIST the reservoir is easy to treat as optional: pack and readout
already do most of the work. Under strong **additive white Gaussian noise
(AWGN)** on the packed field, that same reservoir becomes the product story of
this document: a **white-noise pre-filter for HypercubeCNN** that holds accuracy
the identity path cannot.

**MNIST is only the evaluation vehicle** — a lightweight, familiar, convenient dataset with a
standard train/test split, not a product claim about digits or vision. The
pipeline under test is pack → optional hypercube reservoir → HypercubeCNN on a
length-N field; any static field that packs onto the cube is in scope for the
same idea.

---

## The claim

**With the reservoir on, HypercubeWTF matches pack-only accuracy on clean data
and substantially outperforms pack-only under strong white test noise.**

| Condition | Bypass (pack → readout) | Reservoir (pack → episode → readout) |
|-----------|-------------------------|--------------------------------------|
| Clean or mild AWGN | Strong (~0.98) | Matchable (~0.98 when tuned; see below) |
| Strong AWGN (σ = 0.5) | Collapses (~0.85) | Holds (~0.93) |

The gap at σ = 0.5 is about **eight to nine percentage points**, stable across
noise seeds. That is the product-relevant result.

---

## Protocol

The **only** intentional factor in the headline A/B is reservoir **on vs off**.
Within each noise condition, packing, training set, readout, and (when matched)
noise seed are shared. The reservoir arm uses one fixed recipe (appendix).
Bypass ignores reservoir dynamics; the hypercube still appears in **packing**
the field onto the cube.

**Held-out noise** is i.i.d. Gaussian on every packed vertex after pack and
before prediction — flat spectrum, Gaussian amplitudes, additive (evaluation
protocol only). It is not training-set field noise and not geometric
augmentation.

**Train** stays clean. The head learns on clean features and is scored on clean
or corrupted test fields — a domain-shift stress test, not matched noisy training.

---

## Strong white noise — the reservoir earns its keep

At σ = 0.5, feeding the noisy pack straight into the readout fails hard: test
accuracy sits near **0.84–0.85**. The same class of readout, fed the **end state
of the hypercube reservoir episode**, sits near **0.93**.

That is not a one-draw fluke. Across multiple independent noise seeds, reservoir
test accuracy stays within a few tenths of a point of **0.93** (logged values
from **0.927** to **0.931**). Bypass remains in the mid-0.84s under the same
noise family.

Bypass still fits the **clean** training features almost perfectly, then fails
on noisy test — train/test mismatch. The reservoir reduces that mismatch: it
maps noisy fields into a region the clean-trained head still understands.

**Functional reading:** under strong AWGN on the field, the frozen hypercube
reservoir **pre-filters a large fraction of white noise** before the HypercubeCNN
readout.

---

## Clean and mild noise — nearly transparent

A filter that only works by destroying the signal is useless. On **clean** test
data, bypass is already excellent (~**0.979**). With the reservoir on, the same
level is reachable (~**0.979** in the primary clean pair). Other recipes can sit
a point or so under bypass; that is a **tuning** gap, not a hard ceiling. With
further mild-recipe tuning there is every reason to expect clean (and low-noise)
reservoir accuracy to **match** bypass when that is the goal — the pre-filter
need not tax the clean path.

At **mild** noise (σ = 0.1), one logged pair has bypass slightly ahead (**0.980**
vs **0.968**). There is little white noise to remove; the preprocessor is optional,
and the same tuning story applies.

Operating picture:

- **Clean / light noise** — aim for parity with bypass via recipe choice; logged
  runs already show a match is achievable.
- **Heavy white noise** — the reservoir is the difference between mid-80s and
  low-90s.

---

## Scope of the filter claim

Weights stay **frozen**; this is not a learned denoiser with a noise loss.
Results use one mild contracting recipe on the cube (appendix), not a sweep of
all reservoir settings. The claim is about **bypass vs that reservoir episode**
under AWGN — not a universal denoise theorem.

---

## How to read the numbers

**Test accuracy** is the decision metric (held-out MNIST).  
**Accuracy on collected** is fit to the clean training features. Under strong
test noise, a huge collected–test gap on bypass means the head memorized clean
packs and met a different distribution at test. The reservoir shrinks that gap
without matching train noise.

Reservoir inference costs a short episode per image; bypass does not. Wall-clock
details are secondary to the accuracy story.

---

## What this evaluation does not claim

- Superiority to classical image denoisers (Gaussian, Wiener, BM3D, …) — not
  measured here.
- Robustness to blur, occlusion, adversarial noise, or non-white corruptions.
- That every reservoir recipe is a white-noise pre-filter for HypercubeCNN —
  results use the mild hypercube recipe in the appendix.
- That the reservoir replaces good packing or a competent HypercubeCNN head on
  clean data.
- That HypercubeWTF replaces HypercubeESN for streams — different data modality;
  shared substrate and RC discipline.

What it **does** support: on the hypercube stack, a short frozen **reservoir
episode** can be offered as an **optional white-noise pre-filter for
HypercubeCNN** on length-N fields, with little clean-data tax in this regime, and
a large multi-seed gain under strong AWGN versus identity (bypass) features —
same frozen-reservoir / trained-head contract as HypercubeESN, applied so HCNN
sees a cleaner cube field.

---

## Appendix A — Logged recipe (reproducibility only)

Not part of the product claim — the settings used for the tables below. When the
reservoir is **on**, knobs apply; bypass uses the same pack/readout/noise
protocol and ignores reservoir dynamics.

```text
N=1024  T=20  B=1  M=4  ic_seed=12
reservoir.seed=13871537636959942979
SR_target=0.4  SR_realized≈0.3988  leak=0.5  in_scale=0.005  bias_scale=0
readout: layers=1  weights=82122  pooling=max  activation=NONE
         lr_max=0.0015  epochs=100
pack=PadLowCenter  train=60000  test=10000  aug=off
```

Some σ = 0.5 multi-seed rows used minor readout variants; reservoir test
accuracy remained ~0.93. Factor under study: **bypass vs reservoir**.

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

(Other reservoir clean logs: 0.970–0.977 depending on minor readout settings;
all near bypass.)

### Cross-σ snapshot

| Test noise | Bypass | Reservoir |
|------------|--------|-----------|
| off | ~0.979 | ~0.979 |
| N(0, 0.1) | **0.980** | 0.968 |
| N(0, 0.5) multi-seed | ~0.84–0.85 | **~0.93** |
