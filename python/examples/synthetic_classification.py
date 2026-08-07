#!/usr/bin/env python3
"""Synthetic multi-class fields → WTF episode → train → test (Python host).

Mirrors the spirit of the C++ ``wtf_synth`` example using only the public
``hypercube_wtf`` API — no CMake, no native example binaries.

Onboarding demo (small dim, short train). Not a paper validator.

Requires a checkout of this file (examples are **not** installed by the
PyPI wheel). From the repo root, after ``pip install hypercube-wtf`` (or
``pip install ./python``)::

    python python/examples/synthetic_classification.py
"""

from __future__ import annotations

import numpy as np

import hypercube_wtf as hw


def make_patterns(dim: int, n_per_class: int, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    n = 1 << dim
    fields = []
    labels = []
    for c in range(n_classes):
        for rep in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            x = np.sin((c + 1) * t + 0.07 * rep).astype(np.float32)
            x += 0.12 * rng.standard_normal(n).astype(np.float32)
            x[n // 2 + (c % (n // 4))] += 1.2
            fields.append(x)
            labels.append(c)
    return np.stack(fields, axis=0), np.asarray(labels, dtype=np.int32)


def main() -> None:
    dim = 7
    n_classes = 6
    fields_tr, labels_tr = make_patterns(dim, 64, n_classes, seed=1)
    fields_te, labels_te = make_patterns(dim, 32, n_classes, seed=10_000)

    wtf = hw.WTF(
        dim=dim,
        seed=1,
        ic_seed=2,
        history_depth=8,
        T=0,  # → N
        input_scaling=0.03,
        readout_num_outputs=n_classes,
        readout_task="classification",
        readout_epochs=100,
        readout_batch_size=32,
        readout_num_threads=1,
        readout_restore_best_epoch=False,
        collect_threads=0,
        verbose=True,
    )
    # collect → train (episode API)
    wtf.fit(fields_tr, labels_tr)

    train_acc = wtf.accuracy_on_collected()
    correct = sum(
        wtf.predict_class(fields_te[i]) == int(labels_te[i])
        for i in range(len(labels_te))
    )
    test_acc = correct / len(labels_te)

    print(f"N={wtf.N}  T={wtf.T}  B={wtf.B}  M={wtf.M}")
    print(f"collected: {wtf.num_collected}")
    print(f"train accuracy (collected set): {train_acc:.4f}")
    print(f"test accuracy (held-out fields): {test_acc:.4f}")
    print(f"live predict() shape: {wtf.predict(fields_te[0]).shape}")


if __name__ == "__main__":
    main()
