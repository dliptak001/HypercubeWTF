# Raman Baseline Extraction

A Raman spectrum is an array of intensity values: sharp molecular
peaks sitting on a slow fluorescence background. The task is to
characterize that background so that it can, in follow-on steps, be
subtracted from the original spectrum, leaving only Raman peaks and
random noise remaining. That extraction is the part conventional
methods fail, often miserably, at. Polynomials, asymmetric least
squares, and ordinary convolutional nets follow the empty stretches
and then ride up into the vibrational excitation bands or cut a hole
under them.

This example runs the WTF host on that task: a short reservoir orbit,
then a HypercubeCNN readout (one layer, one channel, no pooling). The
cube is the same length as the spectrum (`N = 2048`, dim 11). There
is nothing to pack, and there is no etalon: the normalized spectrum
drives the reservoir directly.

Train with `wtf_raman`, write selected spectra with
`wtf_raman_extract`, plot with `plot_extracted.py`.

---

## Results

Full LCOHard split — 10000 training spectra, 2000 held-out validation
spectra. Denormalized RMSE in raw counts (see [Error](#error)).

| Split | RMSE |
|-------|-----:|
| Training | 4.714 |
| Validation | 4.756 |

The validation score sits 0.04 counts above training. At this readout
capacity (one conv layer, one channel, no pooling) the fit does not
overfit the 10000-spectrum split.

Run details, from the run that produced those numbers:

- 60 epochs, batch 48, `lr_max` 0.003, cosine decay to
  `lr_min_frac` 0.04, `restore_best_epoch` on. Best epoch was **59
  of 60** (4.714, with epoch 60 at 4.717): the curve had flattened
  into its floor — the last ten epochs moved 0.06 counts in total.
- Readout training time 2974.9 s on a 32-hardware-thread box
  (`collect_threads` auto; the HCNN training pool used all 32).
- Feature-scale probe on the first training spectrum (mean |value|
  over N = 2048):

  | Stage | mean |
  |-------|-----:|
  | Reservoir drive (normalized spectrum) | 0.4774 |
  | Readout features (end-of-orbit state) | 0.0153 |

  `lr_max * N * mean|f|` = 0.094, the product this readout is known
  to train stably under.

---

## WTF vs the siblings

Three hosts have now run this task with the same readout shape, the
same 60-epoch budget, and the same split:

| Host | Preprocessor | Training | Validation |
|------|--------------|---------:|-----------:|
| HypercubeEtalon | etalon transit | 4.705 | 4.767 |
| HypercubeCascade | etalon transit → reservoir orbit | 4.779 | 4.819 |
| **HypercubeWTF** | reservoir orbit | **4.714** | **4.756** |

The reservoir here is the Cascade's second stage lifted out and run
alone: same seed (13871537636959942979), same spectral radius 0.95,
same history depth 8, same leak 1.0, same bias 0.001, same T = 60,
same frozen initial condition (`ic_seed` 1). The Cascade feeds it an
etalon transit scaled by 5.0 and scales its output by 0.84 before the
readout; WTF has neither gain stage, so the drive is the normalized
spectrum itself and `input_scaling` (0.045 here, 0.05 in the Cascade)
is the only knob in front of the orbit.

On validation WTF lands 0.011 counts below the etalon and 0.063 below
the Cascade. The three scores span 0.06 counts. At this noise level
that is the spread between hosts, not a ranking: the etalon and WTF
are a tie, and the Cascade sits a hundredth or so behind both.

What this run does say is that the orbit alone identifies the
baseline every bit as well as the transit alone does. The two
preprocessors are different mechanisms — a causal wave swept across
every cavity versus a frozen recurrent core re-addressed for sixty
passes — and each one, by itself, carries a one-layer, one-channel
readout to the same floor.

The overlay below is the same four held-out spectra the sibling
write-ups show, Validation/581–584. Grey is the raw spectrum, red is
the true baseline, blue is the extract.

![WTF extract, Validation 581 through 584](extracted_baselines_wtf.png)

Blue rides the fluorescence through the 480–530 peak stacks on 581,
holds the trough of 582 under the dense band, runs the straight
descent of 583, and follows the noisy low-count baseline of 584. The
peaks stay in the spectrum. Put this figure beside the etalon's and
there is nothing to point at: the traces sit in the same places and
show the same hairline of red in the trough of 582. The 0.011-count
validation gap is invisible in the extract.

### Training profile

The WTF fit started the way the Cascade's did, not the way the
etalon's did. The Cascade dropped 9.31, 6.90, then 5.87 by epoch 4;
WTF dropped 9.40, 6.87, 6.04, 5.81. The etalon, by contrast, spent
its first third oscillating between 6.8 and 7.6 and did not descend
cleanly until around epoch 14. The two hosts with a reservoir both
open with the same fast, clean drop; the host without one does not.

After that opening WTF settled into a long shelf. From epoch 5 to
epoch 31 the training RMSE bounced between 5.23 and 5.89 with no
sustained direction — 5.67, 5.89, 5.58, 5.73, 5.62, 5.53 over epochs
6 through 11 is typical. The steady descent began at epoch 32 (5.10)
and never broke: 5.0 by epoch 40, 4.9 by epoch 44, 4.8 by epoch 50,
and 4.714 at epoch 59. Epoch 60 ticked up three thousandths, so
`restore_best_epoch` handed back epoch 59.

The etalon finished its 60 epochs at 4.705 training; WTF at 4.714.
Both fits reached their floor in the last few epochs and neither was
still improving meaningfully when it stopped.

---

## Dataset on disk

Fixed root (read in place, not copied by the programs):

```text
C:\HypercubeWTF\RamanSpectraLCOHard\
  Training\
  Validation\
```

| Split | Patterns | Index range |
|-------|----------|-------------|
| Training | 10000 | `0` … `9999` |
| Validation | 2000 | `0` … `1999` |

Each pattern `X` is three files. Both splits also have one shared axis file.

```text
X.data.txt      input spectrum
X.label.txt     ground-truth baseline (train / score target)
X.peaks.txt     ignore for now
xaxis.txt       2048 wavenumbers; ignored by the programs, used by plot_extracted.py
```

Indices are contiguous. Training and validation reuse the same numeric names
in their own folders; they are different spectra.

---

## Host knobs

`MakeBaseConfig()` in `BaselineExtractor.h`. The values behind the
results above:

| Knob | Value |
|------|-------|
| Cube | dim 11, N = 2048 |
| Episode | T = 60, readout_slices 1, ic_seed 1, collect_threads auto |
| Reservoir | history depth 8, spectral radius 0.95 (realized 0.950), leak 1.0, input scale 0.045, bias 0.001, seed 13871537636959942979 |
| Readout | 1 layer, 1 channel, no pooling, no activation, epochs 60, batch 48, lr_max 0.003, lr_min_frac 0.04 |

The reservoir is a weak orbit on purpose: a mild preprocessor so the
readout still sees a field that looks like the spectrum. With no gain
stage behind the orbit, `input_scaling` sets the readout-feature
scale directly. At 0.05 (the Cascade's value) mean |f| came out at
0.0170, which puts `lr_max * N * mean|f|` at 0.105 — just past the
~0.1 where `lr_max` 0.003 stops being stable. At 0.045 it is 0.0153,
the same feature scale the Cascade reaches by scaling its reservoir
output by 0.84. `wtf_raman` prints that probe before it collects so
the scale can be watched when the knobs move.

---

## File format

- One line, no header.
- 2048 comma-separated ASCII floating-point amplitudes.
- Same count in `.data`, `.label`, `.peaks`, and `xaxis.txt`.

---

## Scale

Per-spectrum min/max from the **input**, never the label, mapped to
**[-1, 1]**:

```text
range = max - min
u     = (x - min) / range
norm  = 2 * u - 1
x     = (norm + 1) * 0.5 * range + min

If the spectrum is flat, range is 0, norm is 0, and denorm is min.
```

The label uses the **same** min/range as its matching `.data` spectrum, not
its own min/max. Predict only has the input, so the scale has to come from
there.

Collect and train see only these normalized values. `Predict` denormalizes
before it returns.

---

## Error

Score is **RMS** of the denormalized prediction vs raw `X.label.txt`.
No curve fit.

Per spectrum, over the 2048 bins:

```text
err[i]  = label[i] - predicted[i]
RMSE    = sqrt( mean( err[i]^2 ) )
```

On a split (train prefix or validation prefix), take the mean of those
per-spectrum MSEs, then sqrt — same as RMSE over every bin in the split.

That is the number this example reports. Each readout epoch prints
train `train_rmse`. After fit, the same score is printed on the train
prefix and the validation prefix. Peaks and percent-of-peak error are
out of scope.
