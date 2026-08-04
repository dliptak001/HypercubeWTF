# HypercubeWTF

Static length-N fields → driven hypercube reservoir orbit → HypercubeCNN readout
(end-of-episode sample).

Design charter: [`docs/project.md`](docs/project.md)  
Vendored / forked sources: [`VENDORED.md`](VENDORED.md)

**Status:** Phase 2 — `RunEpisode` drives the field orbit and packs end features;
readout train/predict is Phase 3.

## Build (CLion)

Open the project in CLion and reload CMake. Prefer **Release**. Or, with the
bundled toolchain on PATH:

```text
cmake --build cmake-build-release
```

Smoke: `wtf_smoke` constructs a dim-5 instance and checks N/T/B/M defaults.

## License

See [LICENSE](LICENSE). Vendored HypercubeCNN is Apache-2.0 (see
`third_party/HypercubeCNN/LICENSE`).
