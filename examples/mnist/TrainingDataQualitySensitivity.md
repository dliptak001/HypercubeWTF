# Hypercube Reservoir and Training-Data Quality Sensitivity

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

So when this document compares feature paths under degraded **training data**,
it is not inventing a second architecture. It is asking how a **short hypercube
reservoir episode** changes sensitivity to training-set quality before
HypercubeCNN.

### How WTF uses the reservoir differently from HypercubeESN

| | HypercubeESN | HypercubeWTF (this evaluation) |
|---|--------------|--------------------------------|
| Input | Stream over real time | One **static field** packed onto the cube (e.g. MNIST) |
| Time | Stream time = model time | **Synthetic episode time** inside one sample |
| Drive | New input each step | Same field re-presented each step of the episode |
| Features | State along the stream | **End-of-episode** reservoir state |
| Product emphasis here | Temporal modeling of sequences | **Training-data quality sensitivity** of reservoir vs pack-only features |

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
does a short frozen episode on the same cube change how hard a **degraded
training data set** hits held-out accuracy?

This evaluation asks only:

**When the training data set is degraded, how much does held-out accuracy fall
on each path — and is the reservoir less sensitive than bypass?**

Sensitivity is measured under two **held-out** regimes (clean fields vs strong
**additive white Gaussian noise (AWGN)** on the packed field) so the
training-data effect is not confounded with a single evaluation condition.

**MNIST is only the evaluation vehicle** — a lightweight, familiar, convenient
data set with a standard training/held-out split, not a product claim about
digits or vision. The pipeline under study is pack → optional hypercube
reservoir → HypercubeCNN on a length-N field; any static field that packs onto
the cube is in scope for the same idea.

**Held-out accuracy here is not a ceiling on the substrate.** These studies use a
**dim = 10** hypercube (*N* = 1024) so campaigns stay fast to iterate. That is a
deliberate study choice, not a statement of product accuracy. **HypercubeCNN has
already demonstrated ~99.5% on MNIST**; the HypercubeWTF MNIST example does not
try to re-prove that number. The interesting deltas are **relative** (how each
path moves when the training data set is degraded), not absolute MNIST
leaderboard scores.

---

## The claim

**Hypothesis (confirmed on this survey):** under **strong AWGN on the held-out
packed field**, the reservoir path is **less sensitive** to training-data
quality than bypass; under a **clean held-out set**, both paths are **about
equally sensitive** (small, similar drops). Differences below are in
**percentage points (pp)**.

| Held-out condition | Reservoir vs degraded training set | Bypass vs degraded training set |
|--------------------|------------------------------------|---------------------------------|
| Strong AWGN (σ = 0.5) | Smaller drop (~8 pp) | Larger drop (~19 pp) |
| Clean (no field noise) | Small drop (~1 pp) | Small drop (~1 pp) |

“Sensitivity” here means how much **held-out accuracy falls** when the
**training data set** is degraded, holding the evaluation protocol fixed.

---

## What “training data quality” means in this study

**Spatial augmentation is not product-ready here.** Ideal aug should improve
generalization; however, the augmentation module **hurts** every arm relative to
a clean training set — at least at the dim-10 / MNIST packing used in this
survey. Until that is resolved, **do not treat spatial aug as a recommended
training path**, even though it remains available in the C++ / Python SDKs.

Serendipitously, for **this** study that same buggy / failing training
augmentation module can be used deliberately as a **training-set corruptor**:
systematic geometric and mild pixel noise on the image **before** pack, applied
only when building the training data set. **Held-out images** are never run
through that spatial-aug corruptor.

```text
Clean training set     — augmentation off; pack raw digits → features → train HCNN
Corrupted training set — augmentation on; pack corrupted digits → features → train HCNN
```

Crossing clean vs corrupted training sets with clean vs AWGN held-out fields is the rest of the design.

---

## Strong held-out AWGN — reservoir is more tolerant of a degraded training set

When the **held-out packed fields** carry heavy white noise (σ = 0.5), both arms
lose accuracy if the **training data set** was corrupted. They do not lose it
equally.

| Training data | Bypass held-out acc | Reservoir held-out acc | Reservoir − bypass |
|---------------|---------------------|------------------------|--------------------|
| Clean | 0.830 | **0.927** | +9.7 pp |
| Corrupted (aug) | 0.645 | **0.843** | +19.8 pp |

A corrupted training set costs bypass about **19 pp** and the reservoir about
**8 pp** — degraded training data hurts bypass more than twice as hard.

---

## Clean held-out set — similar sensitivity

When the **held-out data set** is noise-free, corrupting the **training data
set** costs both arms only about a point, and final accuracies stay close.

| Training data | Bypass held-out acc | Reservoir held-out acc |
|---------------|---------------------|------------------------|
| Clean | 0.978 | 0.979 |
| Corrupted (aug) | 0.971 | 0.969 |

About **1 pp** lost on each path — equally sensitive under a clean held-out set.

---

## Picture in one view

Rows = held-out condition; columns = training-data quality. Each cell:
**held-out accuracy** then **takeaway**.

| | **Clean training set** | **Corrupted training set** |
|:---|:---|:---|
| **Clean held-out** | ~0.98 / ~0.98 — **parity** | ~0.97 / ~0.97 — **parity** (small tax) |
| **Held-out AWGN (σ = 0.5)** | ~0.93 / ~0.83 — **reservoir ahead** | ~0.84 / ~0.65 — **reservoir much more tolerant** |

Within each cell, order is **reservoir / bypass**.

---

## What this evaluation does not claim

- That the logged spatial-aug recipe represents good augmentation or a recommended
  training path (it is used here only as a **corruptor** of the training data
  set; dim-10 results do not endorse using the SDK spatial-aug facility that way).
- Multi-seed coverage of every cell (this was a quick, one noise seed study).

What it **does** support: under strong AWGN on the held-out packed field,
degrading the **training data set** hurts **bypass more** than **reservoir**;
under a clean held-out set, both take a small, similar hit.

---

## Appendix A — Logged recipe (reproducibility only)

Not the product claim — settings for the tables below. When the reservoir is
**on**, reservoir and episode knobs apply; bypass uses the same pack and readout
and **ignores** reservoir dynamics.

| Meaning | Where in config / demo | Value used |
|---------|------------------------|------------|
| Hypercube dimension (field length *N* = 2<sup>dim</sup>) | `reservoir.dim` | 10 (*N* = 1024) |
| Episode length (drive steps) | `episode.T` | 20 |
| End-state slices into the readout | `episode.readout_slices` | 1 |
| Reservoir delay-line depth | `reservoir.history_depth` | 4 |
| Frozen episode initial-condition seed | `ic_seed` | 12 |
| Frozen reservoir weight seed | `reservoir.seed` | 13871537636959942979 |
| Target spectral radius (recurrent block) | `reservoir.spectral_radius` | 0.4 (realized ≈ 0.399) |
| Leak rate | `reservoir.leak_rate` | 0.5 |
| Input drive strength | `reservoir.input_scaling` | 0.005 |
| Per-vertex bias scale | `reservoir.bias_scaling` | 0 (off) |
| Train/collect field noise | `episode.train_input_noise_sigma` | 0 (off) |
| HCNN depth / channels / pool / activation | `readout.*` | 1 layer, 16 channels, max pool, none |
| Readout peak learning rate | `readout.lr_max` | 0.0015 |
| Readout training epochs | `readout.epochs` | **100** (reservoir on); **20** (bypass arms) |
| Readout weight count (result of that layout) | — | 82122 |
| MNIST packing mode | demo pack mode | PadLowCenter |
| Training / held-out set sizes | demo limits | 60000 / 10000 |

**Corrupted training set** (demo spatial aug, training only): rotation ±12°,
scale [0.9, 1.1], shift ±2 px, shear_x ±0.15, shear_y 0, elastic off, additive
image noise σ = 0.03 — applied on 28×28 **before** pack.

**Held-out AWGN** (evaluation only): Gaussian σ = 0.5 on the packed field; noise
seed base `0x7E57`.

Factor under study: training-data quality × held-out noise × bypass vs
reservoir.

---

## Appendix B — Tabulated logs

Column conventions: **Training data** = clean vs corrupted training set;
**Path** = Bypass or Reservoir; **collected** = accuracy on the clean training
feature buffer; **held-out acc** = accuracy on the held-out set. Drops are in
percentage points (pp).

### Held-out AWGN σ = 0.5 (noise seed `0x7E57`)

| Training data | Path | collected | held-out acc |
|---------------|------|-----------|--------------|
| Corrupted | Reservoir | 0.970 | 0.843 |
| Corrupted | Bypass | 0.995 | 0.645 |
| Clean | Reservoir | 0.992 | 0.927 |
| Clean | Bypass | 0.998 | 0.830 |

### Clean held-out set (no field noise)

| Training data | Path | collected | held-out acc |
|---------------|------|-----------|--------------|
| Corrupted | Reservoir | 0.970 | 0.969 |
| Corrupted | Bypass | 0.995 | 0.971 |
| Clean | Reservoir | 0.992 | 0.979 |
| Clean | Bypass | 0.999 | 0.978 |

### Sensitivity summary (held-out acc drop when the training data set is corrupted)

| Held-out condition | Reservoir drop | Bypass drop |
|--------------------|----------------|-------------|
| σ = 0.5 | −8.4 pp | −18.5 pp |
| clean | −1.0 pp | −0.7 pp |
