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

**Test field noise** (`kTestNoiseSigma` in `wtf_mnist`): optional i.i.d. Gaussian on the **packed** length-N field after pack, before `PredictClass`. Eval protocol only (default off). Not `episode.input_noise_sigma` (collect-only).

**High-noise bypass A/B** (σ=0.5, train clean): see [`mnist/RESULTS_test_noise_bypass.md`](mnist/RESULTS_test_noise_bypass.md) — orbit test_acc 0.911 vs bypass 0.847.

## Binaries

| Target | Folder | Data |
|--------|--------|------|
| `wtf_synth` | `examples/synth/` | None (CI-friendly) |
| `wtf_mnist` | `examples/mnist/` | **`data/` at repo root only** |

Edit knobs at the top of each demo: `MakeWTFConfig()` (product) and `k*`
demo/task constants — same pattern as HypercubeESN examples.

### MNIST data

`wtf_mnist` searches for this repo’s `data/` (IDX present) under:

1. Current working directory and parents  
2. Executable directory and parents (CLion build dirs)  
3. Source tree next to `examples/`  

No other project path. No env override.

```text
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

See [`data/README.md`](../data/README.md). Default `kMaxTrain=2000` (raise for campaigns).

### Product note

These demos show **static field → reservoir orbit → end-state HCNN readout**.  
They are not a bake-off against HypercubeCNN `mnist_train` accuracy.
