# HypercubeWTF examples

Phase 4 demos. Packing is **example-owned** (not core API).

## Layout

```text
examples/
  README.md
  common/           # shared helpers (e.g. pack_field.h)
  synth/            # wtf_synth — synthetic multi-class fields
  mnist/            # wtf_mnist — MNIST PadLowCenter / PadLow
```

Add a new demo as `examples/<name>/` and wire a target in the root `CMakeLists.txt`.

## Packing rules (v1)

| Packer | Use | Layout |
|--------|-----|--------|
| **PadLow** | Synth patterns; optional MNIST | HCNN `PadLow`: data in verts `[0, P)`, pad `[P, N)` |
| **PadLowCenter** | MNIST default (dim=10) | HCNN `PadLowCenter`: full 28×28 + centered crop in the tail |

MNIST demo maps `PackMode` → vendored `HCNNSpatialEmbedMode`. DualPlane remains on the embed enum for other hosts but is not a MNIST demo option.

**Train-only spatial aug** (vendored `HCNNSpatialAugmenter`): optional 2D geometry + noise on 28×28 **before** pack. Collect-once freezes one aug draw per sample.

**Test field noise** (`kTestNoiseSigma` in `wtf_mnist`): optional i.i.d. Gaussian on the **packed** length-N field after pack, before `PredictClass`. Eval protocol only (default off). Not `episode.train_input_noise_sigma` (train/collect-only).

**White-noise pre-filter (for HypercubeCNN):** see [`mnist/WhiteNoiseFilter.md`](mnist/WhiteNoiseFilter.md) — same frozen-reservoir discipline as HypercubeESN, pre-filtering static cube fields for the HCNN head; at σ=0.5 reservoir ~0.93 vs bypass ~0.85; clean accuracy comparable.

**Training-data quality sensitivity:** see [`mnist/TrainingDataQualitySensitivity.md`](mnist/TrainingDataQualitySensitivity.md) — under strong test AWGN, reservoir loses less than bypass when train is corrupted; on clean test both paths take a similar small hit.

## Binaries

| Target | Folder | Data |
|--------|--------|------|
| `wtf_synth` | `examples/synth/` | None (CI-friendly) |
| `wtf_mnist` | `examples/mnist/` | **`C:\HypercubeWTF\data` only** |

Edit knobs at the top of each demo: `MakeWTFConfig()` (product) and `k*`
demo/task constants — same pattern as HypercubeESN examples.

### MNIST data

`wtf_mnist` loads **only** from `C:\HypercubeWTF\data` (local deploy root).
It does not use a CLion/source-tree `data/`, cwd walks, or other projects.

```text
C:\HypercubeWTF\data\train-images-idx3-ubyte
C:\HypercubeWTF\data\train-labels-idx1-ubyte
C:\HypercubeWTF\data\t10k-images-idx3-ubyte
C:\HypercubeWTF\data\t10k-labels-idx1-ubyte
```

See [`data/README.md`](../data/README.md).

### Product note

These demos show **static field → reservoir orbit → end-state HCNN readout**.  
They are not a bake-off against HypercubeCNN `mnist_train` accuracy.
