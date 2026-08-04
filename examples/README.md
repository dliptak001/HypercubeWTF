# HypercubeWTF examples

Phase 4 demos. Packing is **example-owned** (not core API).

## Packing rules (v1)

| Packer | Use | Layout |
|--------|-----|--------|
| **Pad / low addresses** | Synth patterns; optional MNIST | data in verts `[0, P)`, pad `[P, N)` |
| **DualPlane** | MNIST default | ink \|\| \|grad\| via vendored `HCNNSpatialEmbed` DualPlaneResize; low addresses + pad tail |

**Not used:** standalone resize-to-N product path, letterbox/center-crop, bit-address `(row,col)→vertex` maps.

## Binaries

| Target | Role | Data |
|--------|------|------|
| `wtf_synth` | Synthetic multi-class fields → episode → train → test | None (CI-friendly) |
| `wtf_mnist` | MNIST pack → episode → train → test | IDX under `data/` or `WTF_MNIST_DATA` |

Edit knobs in each file’s `DemoConfig`.

### MNIST data

Not in git. Uncompressed IDX files:

```text
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

Or set `WTF_MNIST_DATA` to that directory.

Default demo uses a **subset** (2k train / 500 test), dim 10, DualPlane, so a run is feasible. Full 60k × dim 11 × `T=N` is a long campaign — raise `max_train` / `dim` in `DemoConfig` when you want that.

### Product note

These demos show **static field → reservoir orbit → end-state HCNN readout**.  
They are not a bake-off against HypercubeCNN `mnist_train` accuracy.
