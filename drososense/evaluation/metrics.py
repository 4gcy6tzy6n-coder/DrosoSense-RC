"""Metrics fixed by the frozen protocol.

Classification primary is macro-F1 (secondary: balanced accuracy, macro one-vs-
rest AUROC, accuracy). Regression primary is MAE (secondary: RMSE, R2).

Two honesty guards live here:

* AUROC is skipped, not faked, for a class absent from the test split, and the
  number of classes actually scored is reported alongside the value.
* Samples whose probability row is entirely NaN are excluded from AUROC rather
  than being scored as 0.5, because a fabricated 0.5 would move the metric
  without any evidence behind it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict[str, Any]:
    """Compute the protocol's classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_score: Optional ``(n_samples, n_classes)`` scores for AUROC. When
            omitted, ``auroc`` is ``None``.
        n_classes: Total class count; inferred from the data if omitted.

    Returns:
        Mapping with ``macro_f1``, ``balanced_accuracy``, ``auroc``, ``accuracy``
        and ``auroc_n_classes_scored``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if n_classes is None:
        observed = set(y_true.tolist()) | set(y_pred.tolist())
        n_classes = max(observed) + 1 if observed else 1
    # Score over the FULL label set, not just the labels that happen to appear.
    # A fold whose test split lacks a class would otherwise be scored over three
    # classes instead of four, and its macro-F1 would not be comparable with a
    # fold that has all four — which would quietly break the paired tests that
    # the protocol's whole comparison rests on.
    labels = list(range(int(n_classes)))

    metrics: dict[str, Any] = {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": None,
        "auroc_n_classes_scored": 0,
    }

    if y_score is None:
        return metrics

    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.ndim != 2 or y_score.shape[1] < n_classes:
        return metrics

    # A class with no test samples has an undefined one-vs-rest AUROC. Dropping
    # it and reporting how many were scored is honest; inventing a value is not.
    per_class: list[float] = []
    for class_index in range(n_classes):
        binary_truth = (y_true == class_index).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        scores = y_score[:, class_index]
        finite = np.isfinite(scores)
        if finite.sum() < 2 or binary_truth[finite].min() == binary_truth[finite].max():
            continue
        per_class.append(float(roc_auc_score(binary_truth[finite], scores[finite])))

    if per_class:
        metrics["auroc"] = float(np.mean(per_class))
        metrics["auroc_n_classes_scored"] = len(per_class)
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the protocol's regression metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Mapping with ``mae``, ``rmse`` and ``r2``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.shape[0] > 1 else float("nan"),
    }


def primary_metric(task: str) -> str:
    """Return the frozen primary metric name for a task.

    Args:
        task: ``classification`` or ``regression``.

    Returns:
        ``macro_f1`` or ``mae``.

    Raises:
        ValueError: If the task is unknown.
    """
    if task == "classification":
        return "macro_f1"
    if task == "regression":
        return "mae"
    raise ValueError(f"unknown task {task!r}")
