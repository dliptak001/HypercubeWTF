"""Essential smoke tests for the hypercube_wtf wheel.

Kept lean so cibuildwheel stays short. Prove the compiled ``_core`` loads and
the episode pipeline (collect → train → predict) produces sane results on the
target platform — not exhaustive façade coverage.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import hypercube_wtf
from hypercube_wtf import WTF


def _make_patterns(dim: int, n_per_class: int, n_classes: int, seed: int = 0):
    """Simple multi-class length-N fields (tone-ish + noise)."""
    rng = np.random.default_rng(seed)
    n = 1 << dim
    fields = []
    labels = []
    for c in range(n_classes):
        for rep in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            x = np.sin((c + 1) * t + 0.1 * rep).astype(np.float32)
            x += 0.05 * rng.standard_normal(n).astype(np.float32)
            # Class-dependent peak in the high half
            x[n // 2 + c * 2] += 1.5
            fields.append(x)
            labels.append(c)
    return np.stack(fields, axis=0), np.asarray(labels, dtype=np.int32)


@pytest.fixture(scope="module")
def cls_data():
    return _make_patterns(dim=5, n_per_class=24, n_classes=3, seed=1)


@pytest.fixture(scope="module")
def trained_cls(cls_data):
    fields, labels = cls_data
    wtf = WTF(
        dim=5,
        seed=1,
        ic_seed=2,
        history_depth=4,
        T=16,
        readout_num_outputs=3,
        readout_task="classification",
        readout_epochs=40,
        readout_batch_size=16,
        readout_num_threads=1,
        readout_restore_best_epoch=False,
        collect_threads=1,
    )
    wtf.fit(fields, labels)
    return wtf, fields, labels


# ── Construction ──

class TestVersion:
    def test_version_string(self):
        v = hypercube_wtf.__version__
        assert isinstance(v, str) and len(v) > 0
        assert v[0].isdigit()
        # Same string as the extension module (CMake baked from _version.py)
        from hypercube_wtf import _core
        assert _core.__version__ == v


class TestConstruction:
    @pytest.mark.parametrize("dim", [5, 7])
    def test_construct(self, dim):
        wtf = WTF(dim=dim, history_depth=4, T=8)
        assert wtf.dim == dim
        assert wtf.N == 2**dim
        assert wtf.num_collected == 0
        assert wtf.T == 8
        assert wtf.B == 1
        assert wtf.M == 4

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be"):
            WTF(dim=4)
        with pytest.raises(ValueError, match="dim must be"):
            WTF(dim=17)

    def test_defaults(self):
        wtf = WTF(dim=5, history_depth=4)
        assert wtf.T == 100
        assert wtf.bypass_reservoir is False
        assert wtf.readout_task == "regression"
        assert wtf.num_outputs == 1

    def test_t_zero_means_n(self):
        # Explicit T=0 still expands to N (full-cube orbit override)
        wtf = WTF(dim=6, T=0, history_depth=4)
        assert wtf.T == 64

    def test_repr(self):
        wtf = WTF(dim=5, history_depth=4, T=8)
        r = repr(wtf)
        assert "dim=5" in r
        assert "N=32" in r


# ── Classification pipeline ──

class TestClassification:
    def test_fit_train_acc(self, trained_cls):
        wtf, _, _ = trained_cls
        acc = wtf.accuracy_on_collected()
        assert acc > 0.85, f"train accuracy too low: {acc}"

    def test_predict_class_shape(self, trained_cls):
        wtf, fields, labels = trained_cls
        pred = wtf.predict_class(fields[0])
        assert isinstance(pred, int)
        assert 0 <= pred < wtf.num_outputs
        logits = wtf.predict(fields[0])
        assert logits.shape == (wtf.num_outputs,)
        assert logits.dtype == np.float32
        assert int(np.argmax(logits)) == pred

    def test_heldout_sane(self, trained_cls):
        wtf, _, _ = trained_cls
        # Fresh draws with different seed — should still beat chance
        fields_te, labels_te = _make_patterns(5, 16, 3, seed=99)
        correct = sum(
            wtf.predict_class(fields_te[i]) == int(labels_te[i])
            for i in range(len(labels_te))
        )
        acc = correct / len(labels_te)
        assert acc > 0.5, f"test accuracy too low: {acc}"


# ── Regression ──

class TestRegression:
    def test_r2_on_collected(self):
        dim = 5
        n = 1 << dim
        rng = np.random.default_rng(3)
        fields = rng.standard_normal((80, n), dtype=np.float32)
        # Strong scalar signal in the field so a short train can move R²
        targets = (2.0 * fields[:, 0:1] + 0.05 * rng.standard_normal((80, 1))).astype(
            np.float32
        )
        wtf = WTF(
            dim=dim,
            history_depth=4,
            T=8,
            input_scaling=0.5,
            readout_num_outputs=1,
            readout_task="regression",
            readout_epochs=80,
            readout_num_threads=1,
            collect_threads=1,
            readout_restore_best_epoch=False,
        )
        wtf.fit(fields, targets)
        r2 = wtf.r2_on_collected()
        assert np.isfinite(r2), f"R² not finite: {r2}"
        # Smoke: should beat "always predict mean" by a bit on this easy signal
        assert r2 > 0.0, f"R² too low: {r2}"
        y = wtf.predict(fields[0])
        assert y.shape == (1,)
        assert y.dtype == np.float32


# ── Episode helpers ──

class TestEpisode:
    def test_run_episode_last_features(self):
        wtf = WTF(dim=5, history_depth=4, T=8, readout_slices=1)
        x = np.zeros(wtf.N, dtype=np.float32)
        x[0] = 1.0
        wtf.run_episode(x)
        feat = wtf.last_features()
        assert feat.shape == (wtf.feature_size,)
        assert feat.dtype == np.float32

    def test_field_size_check(self):
        wtf = WTF(dim=5, history_depth=4, T=8)
        with pytest.raises(Exception, match="must equal N"):
            wtf.run_episode(np.zeros(16, dtype=np.float32))

    def test_bypass(self):
        wtf = WTF(dim=5, history_depth=4, T=8, bypass_reservoir=True)
        assert wtf.bypass_reservoir is True
        x = np.arange(wtf.N, dtype=np.float32)
        wtf.run_episode(x)
        np.testing.assert_allclose(wtf.last_features(), x, atol=1e-6)


# ── Persistence ──

class TestPersistence:
    def test_pickle_roundtrip(self, trained_cls):
        wtf, fields, labels = trained_cls
        acc_before = wtf.accuracy_on_collected()
        loaded = pickle.loads(pickle.dumps(wtf))
        assert loaded.num_collected == 0
        assert loaded.dim == wtf.dim
        # Same weights → same predictions
        for i in range(8):
            assert loaded.predict_class(fields[i]) == wtf.predict_class(fields[i])
        # Retrain not required for infer
        assert acc_before > 0.85

    def test_save_load(self, trained_cls, tmp_path):
        wtf, fields, _ = trained_cls
        path = tmp_path / "model.pkl"
        wtf.save(path)
        loaded = WTF.load(path)
        assert loaded.predict_class(fields[0]) == wtf.predict_class(fields[0])


# ── Surface ──

class TestSurface:
    EXPECTED = [
        "run_episode",
        "last_features",
        "clear_collected",
        "collect_episode",
        "collect_episodes",
        "fit",
        "train",
        "predict",
        "predict_class",
        "accuracy_on_collected",
        "r2_on_collected",
        "save",
        "load",
        "save_readout_hcnn_model",
        "load_readout_hcnn_model",
        "readout_arch_summary",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_method_present(self, name):
        assert callable(getattr(WTF, name, None)), f"WTF.{name} missing"
