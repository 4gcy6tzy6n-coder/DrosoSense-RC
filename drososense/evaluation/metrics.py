"""Metrics fixed by the frozen protocol.

Classification primary is macro-F1 (secondary: balanced accuracy, macro one-vs-
rest AUROC, accuracy). Regression primary is MAE (secondary: RMSE, R2).

The empty-class policy is declared here rather than left to whatever the metric
implementation happens to do, because the two halves must agree — R0 audit item
X5 found the protocol claiming a four-class macro-over-one-vs-rest AUROC while
the implementation averaged over whichever two or three classes happened to be
scorable, making the numbers incomparable between folds:

``macro_f1``
    Scored over the FIXED label set ``0..n_classes-1`` with ``zero_division=0``.
    A class absent from a fold's test split contributes F1 = 0 rather than
    removing itself from the average. The value is therefore always defined and
    is always a four-class macro-F1.

``auroc``
    Scored ONLY when every one of the ``n_classes`` classes is present in the
    test split and has at least one finite score. Otherwise AUROC for that fold
    is ``None`` — never a partial average over a subset of classes, which would
    be a different quantity wearing the same name. When it is not ``None``,
    ``auroc_n_classes_scored`` is always exactly ``n_classes``. Folds with an
    undefined AUROC are counted in the summary so the loss of coverage is
    visible instead of silent.

Samples whose probability row is entirely non-finite are excluded from the
per-class AUROC rather than scored as 0.5, because a fabricated 0.5 would move
the metric without any evidence behind it.
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

# The declared empty-class policy. Recorded in every run record so a reader can
# tell which convention produced a number without reading this file.
EMPTY_CLASS_POLICY = "require_all_classes"
MACRO_F1_ZERO_DIVISION = 0


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
        Mapping with ``macro_f1``, ``balanced_accuracy``, ``auroc``, ``accuracy``,
        ``auroc_n_classes_scored`` and ``auroc_defined``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if n_classes is None:
        observed = set(y_true.tolist()) | set(y_pred.tolist())
        n_classes = max(observed) + 1 if observed else 1
    n_classes = int(n_classes)
    # Score over the FULL label set, not just the labels that happen to appear.
    # A fold whose test split lacks a class would otherwise be scored over three
    # classes instead of four, and its macro-F1 would not be comparable with a
    # fold that has all four — which would quietly break the paired tests that
    # the protocol's whole comparison rests on.
    labels = list(range(n_classes))

    metrics: dict[str, Any] = {
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", labels=labels,
                zero_division=MACRO_F1_ZERO_DIVISION,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": None,
        "auroc_n_classes_scored": 0,
        "auroc_defined": False,
        "empty_class_policy": EMPTY_CLASS_POLICY,
    }

    if y_score is None:
        return metrics

    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.ndim != 2 or y_score.shape[1] < n_classes:
        return metrics

    # A class with no test samples has an undefined one-vs-rest AUROC. Averaging
    # over only the scorable classes would produce a 2- or 3-class number under a
    # 4-class name; the declared policy is to report it as undefined instead.
    per_class: list[float] = []
    for class_index in range(n_classes):
        binary_truth = (y_true == class_index).astype(int)
        if binary_truth.min() == binary_truth.max():
            return metrics
        scores = y_score[:, class_index]
        finite = np.isfinite(scores)
        if finite.sum() < 2 or binary_truth[finite].min() == binary_truth[finite].max():
            return metrics
        per_class.append(float(roc_auc_score(binary_truth[finite], scores[finite])))

    metrics["auroc"] = float(np.mean(per_class))
    metrics["auroc_n_classes_scored"] = len(per_class)
    metrics["auroc_defined"] = True
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
