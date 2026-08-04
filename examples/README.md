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
| `wtf_mnist` | MNIST pack → episode → train → test | **`data/` in this repo only** |

Edit knobs in each file’s `DemoConfig`.

### MNIST data

Always `HypercubeWTF/data/` (path next to `examples/`). No env override, no other project.

Uncompressed IDX (not in git):

```text
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

See [`data/README.md`](../data/README.md) for download steps.

Default demo uses a **subset** (see `DemoConfig::max_train` / `max_test`), dim 10, DualPlane. Raise those knobs for longer campaigns.

### Product note

These demos show **static field → reservoir orbit → end-state HCNN readout**.  
They are not a bake-off against HypercubeCNN `mnist_train` accuracy.
