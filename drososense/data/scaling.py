"""Train-only standardisation.

Implements the protocol rule ``x' = (x - mu_train) / (sigma_train + eps)``.

The scaler is fitted exclusively on rows owned by the training specimens and
is frozen thereafter: it is applied unchanged to validation and test rows, and
the same scaler is reused for every window of every model within a fold. The
fitted artifact is persisted as ``scaler.pkl`` next to the split so that a
result can be reproduced without re-running the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

DEFAULT_EPS = 1.0e-8
SCALER_FILENAME = "scaler.pkl"


@dataclass(frozen=True)
class Standardizer:
    """A frozen per-channel z-score transform.

    Attributes:
        mean: Per-channel training means, shape ``(n_channels,)``.
        std: Per-channel training standard deviations already including eps.
        eps: The epsilon added to sigma at fit time.
        n_fit_rows: Number of training rows the statistics came from.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = DEFAULT_EPS
    n_fit_rows: int = 0

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("mean and std must be 1-D per-channel vectors")
        if mean.shape != std.shape:
            raise ValueError(f"mean shape {mean.shape} != std shape {std.shape}")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean contains non-finite values")
        if not np.all(np.isfinite(std)):
            raise ValueError("std contains non-finite values")
        if np.any(std <= 0):
            raise ValueError("std must be strictly positive (eps must be added at fit time)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def n_channels(self) -> int:
        """Number of standardised channels."""
        return int(self.mean.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the frozen transform to any array whose last axis is channels.

        Args:
            x: Array of shape ``(..., n_channels)``.

        Returns:
            The standardised array, same shape.

        Raises:
            ValueError: If the trailing dimension does not match the scaler.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels on the last axis, got {arr.shape[-1]}"
            )
        return (arr - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Undo the transform, e.g. to report errors in physical units.

        Args:
            x: Standardised array of shape ``(..., n_channels)``.

        Returns:
            The array in original units.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels on the last axis, got {arr.shape[-1]}"
            )
        return arr * self.std + self.mean

    def to_dict(self) -> dict[str, object]:
        """Serialise to plain Python types for run records.

        Returns:
            Mapping with lists rather than arrays.
        """
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
            "n_fit_rows": self.n_fit_rows,
            "n_channels": self.n_channels,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Standardizer":
        """Rebuild a scaler from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed standardizer.
        """
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            eps=float(payload["eps"]),
            n_fit_rows=int(payload["n_fit_rows"]),
        )

    def save(self, directory: str | Path) -> Path:
        """Persist the scaler as ``scaler.pkl`` inside ``directory``.

        Args:
            directory: Destination directory; created if missing.

        Returns:
            Path to the written file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / SCALER_FILENAME
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "Standardizer":
        """Load a scaler previously written by :meth:`save`.

        Args:
            directory: Directory containing ``scaler.pkl``.

        Returns:
            The loaded standardizer.

        Raises:
            FileNotFoundError: If ``scaler.pkl`` is absent.
        """
        path = Path(directory) / SCALER_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"no scaler artifact at {path}")
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"artifact at {path} is {type(loaded).__name__}, expected Standardizer")
        return loaded


def fit_standardizer(x_train: np.ndarray, eps: float = DEFAULT_EPS) -> Standardizer:
    """Fit a standardizer on training rows only.

    Args:
        x_train: Training rows of shape ``(n_rows, n_channels)``.
        eps: Constant added to sigma to keep the transform finite on
            zero-variance channels.

    Returns:
        A frozen :class:`Standardizer`.

    Raises:
        ValueError: If ``x_train`` is not 2-D or has fewer than 2 rows.
    """
    arr = np.asarray(x_train, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"x_train must be 2-D (n_rows, n_channels), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 training rows to fit a scaler, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x_train contains non-finite values; impute or drop them before scaling")
    return Standardizer(
        mean=np.mean(arr, axis=0),
        std=np.std(arr, axis=0) + eps,
        eps=eps,
        n_fit_rows=int(arr.shape[0]),
    )
