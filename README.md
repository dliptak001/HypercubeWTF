# HypercubeWTF

Static length-N fields → driven hypercube reservoir orbit → HypercubeCNN readout
(end-of-episode sample).

Design charter: [`docs/project.md`](docs/project.md)  
Vendored / forked sources: [`VENDORED.md`](VENDORED.md)

**Status:** Phase 4 — demos (`wtf_synth`, `wtf_mnist`) + collect/train/predict core.

## Build (CLion)

Open the project in CLion and reload CMake. Prefer **Release**. Or, with the
bundled toolchain on PATH:

```text
cmake --build cmake-build-release
```

| Binary | Role |
|--------|------|
| `wtf_smoke` | Episode contract + small train/predict smoke |
| `wtf_synth` | Synthetic multi-class fields (no data files) |
| `wtf_mnist` | MNIST DualPlane/PadLow → orbit → readout (needs IDX under `data/`) |

Examples and packing rules: [`examples/README.md`](examples/README.md).

## License

See [LICENSE](LICENSE). Vendored HypercubeCNN is Apache-2.0 (see
`third_party/HypercubeCNN/LICENSE`).
