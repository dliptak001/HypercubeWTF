# Python examples

Runnable **Python hosts** for the `hypercube-wtf` API. They use only the public
package surface — no CMake, no C++ example binaries.

| Script | What it shows |
|--------|----------------|
| [`synthetic_classification.py`](synthetic_classification.py) | Multi-class length-N fields → `fit` → train / test accuracy |

## How to run

Examples live in the **git tree**. They are **not** installed by the PyPI wheel
(`pip install hypercube-wtf` alone does not place these scripts on disk).

From the **repository root** (after a Release wheel or local build):

```bash
pip install hypercube-wtf
# or, from this tree:  pip install ./python
python python/examples/synthetic_classification.py
```

## What these are not

- **Not** the C++ demos (`wtf_synth`, `wtf_mnist`) or the MNIST study write-ups.
  Those live under [`examples/`](../../examples/).
- **Not** pytest. CI smoke is [`tests/test_basic.py`](../tests/test_basic.py).
- **Not** hard tasks. Synthetic multi-class fields are easy onboarding so the
  episode API is obvious; do not cite their metrics as storefront results.

## Going further

| Want… | See… |
|-------|------|
| Full Python API | [`docs/Python_SDK.md`](../../docs/Python_SDK.md) |
| C++ product guide | [`docs/CPP_SDK.md`](../../docs/CPP_SDK.md) |
| Package install | [`README.md`](../README.md) |
