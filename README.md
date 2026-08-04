# HypercubeWTF

Static length-N fields → driven hypercube reservoir orbit → HypercubeCNN readout
(end-of-episode sample).

Design charter: [`docs/project.md`](docs/project.md)  
Vendored / forked sources: [`VENDORED.md`](VENDORED.md)

**Status:** Phase 3 — collect episodes, batch-train HCNN, predict.

## Build (CLion)

Open the project in CLion and reload CMake. Prefer **Release**. Or, with the
bundled toolchain on PATH:

```text
cmake --build cmake-build-release
```

Smoke: `wtf_smoke` (dim-5) checks the episode contract (`T=N` default, `T>N`
wrap, `B=2` pack, IC seed split, size throw) and a synthetic 2-class
train/predict path.

## License

See [LICENSE](LICENSE). Vendored HypercubeCNN is Apache-2.0 (see
`third_party/HypercubeCNN/LICENSE`).
