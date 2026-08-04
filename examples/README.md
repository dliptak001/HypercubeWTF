# HypercubeWTF examples

Phase 4 demos. Packing is **example-owned** (not core API).

## Layout

```text
examples/
  README.md
  common/           # shared helpers (e.g. pack_field.h)
  synth/            # wtf_synth — synthetic multi-class fields
  mnist/            # wtf_mnist — MNIST DualPlane / PadLow
```

Add a new demo as `examples/<name>/` and wire a target in the root `CMakeLists.txt`.

## Packing rules (v1)

| Packer | Use | Layout |
|--------|-----|--------|
| **Pad / low addresses** | Synth patterns; optional MNIST | data in verts `[0, P)`, pad `[P, N)` |
| **DualPlane** | MNIST default | ink \|\| \|grad\| via vendored `HCNNSpatialEmbed` DualPlaneResize; low addresses + pad tail |

**Not used:** standalone resize-to-N product path, letterbox/center-crop, bit-address `(row,col)→vertex` maps.

## Binaries

| Target | Folder | Data |
|--------|--------|------|
| `wtf_synth` | `examples/synth/` | None (CI-friendly) |
| `wtf_mnist` | `examples/mnist/` | **`data/` at repo root only** |

Edit knobs at the top of each demo: `MakeWTFConfig()` (product) and `k*`
demo/task constants — same pattern as HypercubeESN examples.

### MNIST data

Always `HypercubeWTF/data/` (repo root). No env override, no other project.

```text
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

See [`data/README.md`](../data/README.md).

### Product note

These demos show **static field → reservoir orbit → end-state HCNN readout**.  
They are not a bake-off against HypercubeCNN `mnist_train` accuracy.
