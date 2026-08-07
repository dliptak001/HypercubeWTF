"""HypercubeWTF: frozen hypercube reservoir orbit + HypercubeCNN on end state.

Static length-N fields (no intrinsic time): drive a short reservoir orbit per
sample, then train a HypercubeCNN readout on the end-state features only.

Quick start::

    import numpy as np
    import hypercube_wtf as hw

    # fields: (num_samples, N) float32, labels: (num_samples,) int
    wtf = hw.WTF(dim=7, readout_num_outputs=6, readout_task="classification",
                 readout_epochs=80)
    wtf.fit(fields_train, labels_train)
    pred = wtf.predict_class(fields_test[0])
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np

from ._core import _WTF, __version__ as _core_version

__version__ = _core_version
__all__ = ["WTF", "__version__"]

# Valid hypercube dimensions (matches C++ Reservoir::Create [5, 16] check).
_DIM_MIN = 5
_DIM_MAX = 16


def _to_float32(arr):
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_int32(arr):
    """Ensure array is C-contiguous int32 (labels)."""
    return np.ascontiguousarray(arr, dtype=np.int32)


class WTF:
    """HypercubeWTF product: static field → reservoir orbit → HCNN on end state.

    The reservoir has N = 2^dim neurons on a Boolean hypercube. Each sample is
    a full length-N field (you pack domain data on the host). An **episode**
    reloads a frozen IC, drives the field for T synthetic passes, and packs
    B end-of-orbit delay-line ages into the readout input (default B=1 → N).

    Typical lifecycle: :meth:`collect_episodes` → :meth:`train` →
    :meth:`predict` / :meth:`predict_class`, or the one-shot :meth:`fit`.

    Parameters
    ----------
    dim : int
        Hypercube dimension (5–16). N = 2^dim neurons / field length.
    seed : int
        Reservoir weight-init seed. Default matches C++ ``ReservoirConfig``.
    spectral_radius : float
        Target spectral radius for recurrent weights. Default: 0.999.
    input_scaling : float
        Input drive coefficient. Default: 0.02 (C++ product default).
    leak_rate : float
        Leaky integrator; 1.0 = full replacement. Default: 1.0.
    history_depth : int
        Delay-line depth M ∈ [1, 64]. Default: 16.
    verbose : bool
        Print the reservoir construction banner. Default: False.
    bias_scaling : float
        Per-neuron bias scale after tanh; 0 disables. Default: 0.003.
    ic_seed : int
        Seed for frozen episode initial condition (separate from weight seed).
        Default: 1.
    T : int
        Drive-pass count. 0 means use N after construction.
    readout_slices : int
        B end-of-episode ages packed into features (power of two, ≤ M).
        Default: 1.
    collect_threads : int
        Workers for bulk collect: 0 = auto, 1 = serial, K = K workers.
    train_input_noise_sigma : float
        Gaussian σ on the field during collect only (not predict). 0 = off.
    bypass_reservoir : bool
        If True, skip the orbit; features are the packed field (requires B=1).
    readout_num_outputs : int
        Classes (classification) or regression width. Default: 1.
    readout_task : str
        ``"regression"`` (default) or ``"classification"``.
    readout_num_layers : int
        Conv(+Pool) pairs. Default: 1. 0 = auto min(dim-2, 2).
    readout_conv_channels : int
        Base channel count. Default: 16.
    readout_epochs : int
        Batch train epochs. Default: 200.
    readout_batch_size : int
        Mini-batch size. Default: 32.
    readout_lr_max : float
        Cosine peak LR. Default: 0.0015.
    readout_lr_min_frac : float
        Floor as fraction of lr_max. Default: 0.01.
    readout_lr_decay_epochs : int
        Cosine horizon; 0 = use epochs. Default: 0.
    readout_weight_decay : float
        L2 on CNN weights. Default: 0.0.
    readout_momentum : float
        SGD momentum (ignored by Adam). Default: 0.9.
    readout_activation : str
        ``"tanh"`` (default), ``"relu"``, ``"leaky_relu"``, or ``"none"``.
    readout_seed : int
        CNN weight-init seed. Default: 42.
    readout_num_threads : int
        HCNN pool: 0 = auto, 1 = single-threaded. Default: 0.
    readout_restore_best_epoch : bool
        Restore best-epoch weights after batch train. Default: True.
    readout_best_epoch_holdout_frac : float
        Tail hold-out for best-epoch scoring. Default: 0.0.
    readout_use_pooling : bool
        Antipodal pool after each conv. Default: True.

    Notes
    -----
    This class is **not thread-safe** for concurrent public calls from multiple
    host threads. Bulk collect parallelism is internal.

    Examples
    --------
    >>> import numpy as np
    >>> import hypercube_wtf as hw
    >>> rng = np.random.default_rng(0)
    >>> N = 32  # dim=5
    >>> fields = rng.standard_normal((64, N), dtype=np.float32)
    >>> labels = rng.integers(0, 3, size=64)
    >>> wtf = hw.WTF(dim=5, readout_num_outputs=3, readout_task="classification",
    ...              readout_epochs=40, history_depth=4, T=32)
    >>> wtf.fit(fields, labels)
    WTF(dim=5, N=32, ...)
    """

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 7934791766227647176,
        spectral_radius: float = 0.999,
        input_scaling: float = 0.02,
        leak_rate: float = 1.0,
        history_depth: int = 16,
        verbose: bool = False,
        bias_scaling: float = 0.003,
        ic_seed: int = 1,
        T: int = 100,
        readout_slices: int = 1,
        collect_threads: int = 0,
        train_input_noise_sigma: float = 0.0,
        bypass_reservoir: bool = False,
        readout_num_outputs: int = 1,
        readout_task: str = "regression",
        readout_num_layers: int = 1,
        readout_conv_channels: int = 16,
        readout_epochs: int = 200,
        readout_batch_size: int = 32,
        readout_lr_max: float = 0.0015,
        readout_lr_min_frac: float = 0.01,
        readout_lr_decay_epochs: int = 0,
        readout_weight_decay: float = 0.0,
        readout_momentum: float = 0.9,
        readout_activation: str = "tanh",
        readout_seed: int = 42,
        readout_num_threads: int = 0,
        readout_restore_best_epoch: bool = True,
        readout_best_epoch_holdout_frac: float = 0.0,
        readout_use_pooling: bool = True,
    ):
        if not isinstance(dim, int) or not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(
                f"dim must be an integer in [{_DIM_MIN}, {_DIM_MAX}], got {dim!r}"
            )
        if readout_task not in ("regression", "classification"):
            raise ValueError(
                f"readout_task must be 'regression' or 'classification', "
                f"got {readout_task!r}"
            )
        if readout_activation not in ("tanh", "relu", "leaky_relu", "none"):
            raise ValueError(
                "readout_activation must be one of "
                "'tanh', 'relu', 'leaky_relu', 'none' "
                f"(got {readout_activation!r})"
            )
        self._verbose = verbose
        self._ctor = {
            "dim": dim,
            "seed": seed,
            "spectral_radius": spectral_radius,
            "input_scaling": input_scaling,
            "leak_rate": leak_rate,
            "history_depth": history_depth,
            "verbose": verbose,
            "bias_scaling": bias_scaling,
            "ic_seed": ic_seed,
            "T": T,
            "readout_slices": readout_slices,
            "collect_threads": collect_threads,
            "train_input_noise_sigma": train_input_noise_sigma,
            "bypass_reservoir": bypass_reservoir,
            "readout_num_outputs": readout_num_outputs,
            "readout_task": readout_task,
            "readout_num_layers": readout_num_layers,
            "readout_conv_channels": readout_conv_channels,
            "readout_epochs": readout_epochs,
            "readout_batch_size": readout_batch_size,
            "readout_lr_max": readout_lr_max,
            "readout_lr_min_frac": readout_lr_min_frac,
            "readout_lr_decay_epochs": readout_lr_decay_epochs,
            "readout_weight_decay": readout_weight_decay,
            "readout_momentum": readout_momentum,
            "readout_activation": readout_activation,
            "readout_seed": readout_seed,
            "readout_num_threads": readout_num_threads,
            "readout_restore_best_epoch": readout_restore_best_epoch,
            "readout_best_epoch_holdout_frac": readout_best_epoch_holdout_frac,
            "readout_use_pooling": readout_use_pooling,
        }
        self._impl = _WTF(
            dim=dim,
            seed=seed,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            history_depth=history_depth,
            verbose=verbose,
            bias_scaling=bias_scaling,
            ic_seed=ic_seed,
            episode_T=T,
            readout_slices=readout_slices,
            collect_threads=collect_threads,
            train_input_noise_sigma=train_input_noise_sigma,
            bypass_reservoir=bypass_reservoir,
            readout_num_outputs=readout_num_outputs,
            readout_task=readout_task,
            readout_num_layers=readout_num_layers,
            readout_conv_channels=readout_conv_channels,
            readout_epochs=readout_epochs,
            readout_batch_size=readout_batch_size,
            readout_lr_max=readout_lr_max,
            readout_lr_min_frac=readout_lr_min_frac,
            readout_lr_decay_epochs=readout_lr_decay_epochs,
            readout_weight_decay=readout_weight_decay,
            readout_momentum=readout_momentum,
            readout_activation=readout_activation,
            readout_seed=readout_seed,
            readout_num_threads=readout_num_threads,
            readout_restore_best_epoch=readout_restore_best_epoch,
            readout_best_epoch_holdout_frac=readout_best_epoch_holdout_frac,
            readout_use_pooling=readout_use_pooling,
        )

    # ── Episode ──

    def run_episode(self, x: np.ndarray) -> None:
        """Drive one episode (or bypass copy). Updates :meth:`last_features`.

        Parameters
        ----------
        x : ndarray
            Length-N field (or shape that ravel-flattens to N). Converted to float32.
        """
        self._impl.run_episode(_to_float32(np.ravel(x)))

    def last_features(self) -> np.ndarray:
        """Feature pack from the last serial episode path (length B*N).

        Not updated by bulk :meth:`collect_episodes`.
        """
        return self._impl.last_features()

    def clear_collected(self) -> None:
        """Drop all samples collected for batch training."""
        self._impl.clear_collected()

    # ── Collect ──

    def collect_episode(
        self,
        x: np.ndarray,
        target,
    ) -> None:
        """Serial collect one episode (classification label or regression target).

        Parameters
        ----------
        x : ndarray
            Length-N field.
        target : int or ndarray
            Class index (classification) or length-``num_outputs`` floats
            (regression).
        """
        field = _to_float32(np.ravel(x))
        if self._ctor["readout_task"] == "classification":
            if isinstance(target, (bool, np.bool_)):
                raise TypeError("class label must be an integer")
            if isinstance(target, (int, np.integer)):
                label = int(target)
            else:
                arr = np.asarray(target).ravel()
                if arr.size != 1:
                    raise ValueError("classification target must be a single class index")
                label = int(arr[0])
            self._impl.collect_episode_class(field, label)
        else:
            t = _to_float32(np.ravel(target))
            self._impl.collect_episode_reg(field, t)

    def collect_episodes(
        self,
        fields: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Bulk parallel collect (appends to the training set).

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` preferred, or flat length ``count * N``.
        targets : ndarray
            Classification: shape ``(count,)`` integer labels.
            Regression: shape ``(count, num_outputs)`` or flat ``count * num_outputs``.
        """
        fields = _to_float32(fields)
        n = self.N
        if fields.ndim == 2:
            if fields.shape[1] != n:
                raise ValueError(
                    f"fields.shape[1] ({fields.shape[1]}) must equal N ({n})"
                )
            count = int(fields.shape[0])
            flat = np.ascontiguousarray(fields.reshape(-1))
        elif fields.ndim == 1:
            if fields.size % n != 0:
                raise ValueError(
                    f"flat fields size ({fields.size}) must be a multiple of N ({n})"
                )
            count = int(fields.size // n)
            flat = fields
        else:
            raise ValueError(
                f"fields must be 1D or 2D, got ndim={fields.ndim}"
            )

        if self._ctor["readout_task"] == "classification":
            labels = _to_int32(np.ravel(targets))
            if labels.size != count:
                raise ValueError(
                    f"labels length ({labels.size}) must equal sample count ({count})"
                )
            self._impl.collect_episodes_class(flat, labels)
        else:
            t = _to_float32(targets)
            k = self.num_outputs
            if t.ndim == 2:
                if t.shape[0] != count or t.shape[1] != k:
                    raise ValueError(
                        f"targets shape must be ({count}, {k}), got {t.shape}"
                    )
                t = np.ascontiguousarray(t.reshape(-1))
            elif t.ndim == 1:
                if t.size != count * k:
                    raise ValueError(
                        f"flat targets size ({t.size}) must equal "
                        f"count * num_outputs ({count * k})"
                    )
            else:
                raise ValueError(f"targets must be 1D or 2D, got ndim={t.ndim}")
            self._impl.collect_episodes_reg(flat, t)

    def fit(
        self,
        fields: np.ndarray,
        targets: np.ndarray,
        *,
        clear: bool = True,
    ) -> "WTF":
        """Collect episodes then train the readout (episode-shaped one-shot).

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` length-N fields (host packing is your problem).
        targets : ndarray
            Labels or regression targets (see :meth:`collect_episodes`).
        clear : bool
            If True (default), :meth:`clear_collected` first.

        Returns
        -------
        WTF
            Self, for method chaining.
        """
        if clear:
            self.clear_collected()
        self.collect_episodes(fields, targets)
        self.train()
        return self

    def train(self) -> None:
        """Batch-train the HCNN on all collected episodes.

        Does not clear the collected set — call again to retrain, or
        :meth:`clear_collected` first to start over.
        """
        self._impl.train()

    # ── Inference ──

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fresh episode + readout forward (no train-input noise).

        Parameters
        ----------
        x : ndarray
            Length-N field.

        Returns
        -------
        ndarray
            Shape ``(num_outputs,)`` float32 logits / regression values.
        """
        return self._impl.predict(_to_float32(np.ravel(x)))

    def predict_class(self, x: np.ndarray) -> int:
        """Fresh episode + argmax class (classification task only)."""
        return int(self._impl.predict_class(_to_float32(np.ravel(x))))

    def accuracy_on_collected(self) -> float:
        """Accuracy on the **collected training set** — not a test-set metric."""
        return float(self._impl.accuracy_on_collected())

    def r2_on_collected(self) -> float:
        """R² on the **collected training set** — not a test-set metric."""
        return float(self._impl.r2_on_collected())

    # ── Properties ──

    @property
    def dim(self) -> int:
        """Hypercube dimension."""
        return int(self._impl.dim)

    @property
    def N(self) -> int:
        """Neuron / field length: 2^dim."""
        return int(self._impl.N)

    @property
    def T(self) -> int:
        """Drive-pass count for each episode."""
        return int(self._impl.T)

    @property
    def B(self) -> int:
        """Readout slices (end-of-episode ages packed)."""
        return int(self._impl.B)

    @property
    def M(self) -> int:
        """Delay-line depth (history_depth)."""
        return int(self._impl.M)

    @property
    def feature_size(self) -> int:
        """B * N floats per collected sample / last_features."""
        return int(self._impl.feature_size)

    @property
    def num_collected(self) -> int:
        """Number of episodes in the batch training set."""
        return int(self._impl.num_collected)

    @property
    def num_outputs(self) -> int:
        """Readout output width."""
        return int(self._impl.num_outputs)

    @property
    def collect_threads(self) -> int:
        """Configured collect-thread preference (0 = auto)."""
        return int(self._impl.collect_threads)

    @property
    def bypass_reservoir(self) -> bool:
        """True when the orbit is skipped (field → features)."""
        return bool(self._impl.bypass_reservoir)

    @property
    def ic_seed(self) -> int:
        """Frozen episode IC seed."""
        return int(self._impl.ic_seed)

    @property
    def train_input_noise_sigma(self) -> float:
        """Collect-only field noise σ (0 = off)."""
        return float(self._impl.train_input_noise_sigma)

    @property
    def seed(self) -> int:
        """Reservoir weight seed."""
        return int(self._impl.seed)

    @property
    def spectral_radius(self) -> float:
        """Target spectral radius (config)."""
        return float(self._impl.spectral_radius)

    @property
    def realized_spectral_radius(self) -> float:
        """Post-rescale realized spectral-radius estimate."""
        return float(self._impl.realized_spectral_radius)

    @property
    def input_scaling(self) -> float:
        return float(self._impl.input_scaling)

    @property
    def leak_rate(self) -> float:
        return float(self._impl.leak_rate)

    @property
    def history_depth(self) -> int:
        return int(self._impl.history_depth)

    @property
    def bias_scaling(self) -> float:
        return float(self._impl.bias_scaling)

    @property
    def verbose(self) -> bool:
        return bool(self._verbose)

    @property
    def readout_task(self) -> str:
        return str(self._impl.readout_task)

    @property
    def readout_best_epoch(self) -> int:
        """1-based best epoch after restore_best_epoch train, else 0."""
        return int(self._impl.readout_best_epoch)

    def __repr__(self) -> str:
        return (
            f"WTF(dim={self.dim}, N={self.N}, T={self.T}, B={self.B}, "
            f"collected={self.num_collected}, outputs={self.num_outputs}, "
            f"task={self.readout_task})"
        )

    # ── Persistence ──

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        """Serialize constructor config + trained readout weights.

        Collected episodes are **not** saved.
        """
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "readout_state": self._impl._get_readout_state(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-wtf."
            )
        ctor = dict(state["ctor"])
        self.__init__(**ctor)
        self._impl._set_readout_state(state["readout_state"])

    def save(self, path) -> None:
        """Save config + trained readout to a pickle file.

        Collected episodes are not saved. Prefer
        :meth:`save_readout_hcnn_model` for portable HCNW + arch JSON.
        """
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "WTF":
        """Load a model saved by :meth:`save`.

        .. warning::

            Uses ``pickle.load``. Never load untrusted files.
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected WTF, got {type(obj).__name__}")
        return obj

    def save_readout_hcnn_model(self, path_stem) -> None:
        """Export the HCNN readout as portable ``stem.hcnw`` + ``stem.arch.json``."""
        self._impl.save_readout_hcnn_model(str(path_stem))

    def load_readout_hcnn_model(self, path_stem, *, mode: str = "eval") -> None:
        """Load ``stem.hcnw`` (+ arch sidecar) into this instance's readout.

        Parameters
        ----------
        path_stem : str or Path
            Path without extension.
        mode : str
            ``"eval"`` (default) or ``"resume_train"``.
        """
        self._impl.load_readout_hcnn_model(str(path_stem), mode)

    def readout_arch_summary(self) -> str:
        """Human-readable HCNN readout architecture and parameter counts."""
        return self._impl.readout_arch_summary()
