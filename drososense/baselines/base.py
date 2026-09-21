"""The single interface every baseline implements.

A model receives ``X`` of shape ``(n_windows, window_length, n_channels)`` and
returns predictions of shape ``(n_windows,)``. Classical estimators flatten the
tensor internally with :meth:`WindowSet.flat`, so the flattening order is shared
by every model and cannot become a hidden source of difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

TaskType = Literal["classification", "regression"]


class ModelUnavailableError(RuntimeError):
    """Raised when a model's backend is not importable in this environment."""


class BaseModel(ABC):
    """Common interface for every baseline and, later, the connectome reservoir.

    Attributes:
        model_id: Stable identifier used in configs and result records.
        task: ``classification`` or ``regression``.
        seed: Seed the model was constructed with.
        params: Hyperparameters, recorded verbatim in the run record.
    """

    model_id: str = "base"
    supports_classification: bool = True
    supports_regression: bool = True

    def __init__(self, task: TaskType, seed: int, params: dict[str, Any] | None = None) -> None:
        """Initialise the model.

        Args:
            task: ``classification`` or ``regression``.
            seed: Seed for every stochastic component.
            params: Hyperparameters.

        Raises:
            ValueError: If the task is not supported by this model.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        if task == "classification" and not self.supports_classification:
            raise ValueError(f"{self.model_id} does not support classification")
        if task == "regression" and not self.supports_regression:
            raise ValueError(f"{self.model_id} does not support regression")
        self.task: TaskType = task
        self.seed = int(seed)
        self.params: dict[str, Any] = dict(params or {})
        self._fitted = False

    @abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit on already-flattened 2-D input.

        Args:
            x: Design matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)``.
        """

    @abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from already-flattened 2-D input.

        Args:
            x: Design matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)``.
        """

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, or ``None`` when unsupported.

        Args:
            x: Input tensor of shape ``(n_samples, window_length, n_channels)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` for classifiers that can
            produce probabilities, otherwise ``None``.
        """
        return None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model on window tensors.

        Args:
            x: Tensor of shape ``(n_samples, window_length, n_channels)``.
            y: Targets of shape ``(n_samples,)``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the input is not 3-D or the sample counts disagree.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 3:
            raise ValueError(f"X must be 3-D (n, L, C), got shape {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"X has {x.shape[0]} samples but y has {y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on an empty window set")
        self._fit(self.flatten(x), y)
        self._fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from window tensors.

        Args:
            x: Tensor of shape ``(n_samples, window_length, n_channels)``.

        Returns:
            Predictions of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.model_id}: predict() called before fit()")
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError(f"X must be 3-D (n, L, C), got shape {x.shape}")
        return np.asarray(self._predict(self.flatten(x)))

    @staticmethod
    def flatten(x: np.ndarray) -> np.ndarray:
        """Flatten ``(n, L, C)`` to ``(n, L*C)`` in a fixed order.

        Args:
            x: Window tensor.

        Returns:
            The 2-D design matrix.
        """
        x = np.asarray(x)
        return x.reshape(x.shape[0], -1)

    def n_trainable_parameters(self) -> int | None:
        """Report the number of trained parameters, when meaningful.

        This is the quantity E5/E7 report for TAFE's Circuits/Systems framing —
        trainable parameters, not total reservoir size.

        Returns:
            Parameter count, or ``None`` when the model has no natural count.
        """
        return None

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the run record.

        Returns:
            Mapping with model id, task, seed, params and parameter count.
        """
        return {
            "model_id": self.model_id,
            "task": self.task,
            "seed": self.seed,
            "params": self.params,
            "n_trainable_parameters": self.n_trainable_parameters(),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r}, task={self.task!r}, seed={self.seed})"
