# Vendored / forked sources — HypercubeWTF

Recorded at first skeleton (Phase 1). Paths are the on-disk trees used for the copy.

| Component | Source tree | Branch | Commit |
|-----------|-------------|--------|--------|
| `third_party/HypercubeCNN/*` | HypercubeCNN | `main` | Core mostly 1.0.3-aligned; **SpatialEmbed** re-copied at `a60b689` (see `third_party/HypercubeCNN/VENDORED.md`) |
| `Reservoir.*` (fork start) | HypercubeESN | **`feedback`** (not main) | `ae3fb6430e557066a25d7678fdbfedad81093697` |
| `Readout.*` (fork start) | HypercubeESN | **`feedback`** | `ae3fb6430e557066a25d7678fdbfedad81093697` |

## Fork notes (Reservoir)

Started from HypercubeESN `Reservoir` on `feedback`, then WTF-specific changes:

- **Removed** entire external-feedback port (config, weights, inject, gather).
- **Removed** multi-channel `num_inputs` / `InjectInput` block broadcast.
- **Added** `InjectInputField(field, N)` — full-field stage.
- **Added** `LoadInitialCondition(ic, N*M)` — canonical ring home + bulk IC load for episode **s0**.

Re-sync from upstream only deliberately; this is not API-locked to HypercubeESN.
