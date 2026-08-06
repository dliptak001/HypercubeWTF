# Vendored HypercubeCNN

**Upstream:** local tree `C:\CLion\HypercubeCNN` (GitHub: dliptak001/HypercubeCNN)  
**Branch:** `main`  
**Core snapshot base:** older pin; most library sources match upstream 1.0.3  
**SpatialEmbed re-sync:** `a60b6896e929eaae2429615692f5ce50109b3215`  
  (`chore(release): bump version to 1.0.3`, 2026-08-05)  
**License:** Apache-2.0 (see `LICENSE`)

Do not hand-edit snapshot sources; re-copy from upstream at a chosen commit.

`CMakeLists.txt` in this directory is HypercubeWTF build glue, not upstream.

### Overlay / re-sync notes

| File | Note |
|------|------|
| `HCNNSpatialEmbed.h` / `.cpp` | Re-copied from `C:\CLion\HypercubeCNN` at `a60b689` (unknown-mode throws, PadLowCenter row `memcpy`, `embed_batch(0)` no-op). SHA256 matches upstream tree. |
