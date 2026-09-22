"""Active leakage audits.

These functions do not merely assert that the code *intends* to be leak-free —
they re-derive the invariant from the data that was actually produced and raise
:class:`LeakageError` when it does not hold. Each audit is paired with a
positive-control test that feeds it a deliberately leaky input and asserts that
it fires (see ``tests/test_leakage.py``); an audit that cannot fail is not an
audit.

Three independent leakage channels are covered:

1. **Split leakage** — a specimen reaching more than one split.
2. **Preprocessing leakage** — scaler statistics influenced by non-train rows.
3. **Window leakage** — a window spanning two specimens, or a specimen's
   windows landing in a split that does not own that specimen.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from drososense.data.splits import Fold

# Absolute tolerance when re-deriving scaler statistics. Generous enough for
# float32 round-trips, far tighter than any real leakage effect.
_STAT_TOLERANCE = 1e-6


class LeakageError(Exception):
    """Raised when an audit detects information moving across a split boundary."""


def _specimen_set(values: Sequence[str]) -> set[str]:
    """Return a set of specimen identifiers as strings."""
    return {str(v) for v in values}


def audit_fold(fold: Fold, all_specimens: Sequence[str] | None = None) -> None:
    """Verify a fold is a disjoint partition of the available specimens.

    Args:
        fold: The fold to audit.
        all_specimens: If given, the fold's union must equal this set exactly.

    Raises:
        LeakageError: If any two splits share a specimen, or if the fold's
            union does not match ``all_specimens``.
    """
    train = _specimen_set(fold.train)
    val = _specimen_set(fold.val)
    test = _specimen_set(fold.test)

    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = a & b
        if overlap:
            raise LeakageError(
                f"fold {fold.fold_id} ({fold.strategy}, seed {fold.seed}): "
                f"{sorted(overlap)} appear in both {a_name} and {b_name}"
            )

    if all_specimens is not None:
        expected = _specimen_set(all_specimens)
        union = train | val | test
        missing = expected - union
        extra = union - expected
        if missing or extra:
            raise LeakageError(
                f"fold {fold.fold_id}: partition does not cover the dataset "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
            )


def audit_scaler(
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    X_train: np.ndarray,
    eps: float,
    X_contaminated: np.ndarray | None = None,
) -> None:
    """Verify scaler statistics were computed from the training rows alone.

    The scaler's stored mean/std are recomputed from ``X_train`` and compared.
    When ``X_contaminated`` is supplied (train rows concatenated with rows that
    should never have been seen), the audit additionally checks that the stored
    statistics do NOT match the contaminated recomputation — this is the
    positive control that proves the audit can detect contamination.

    Args:
        scaler_mean: Mean stored on the scaler.
        scaler_std: Standard deviation stored on the scaler.
        X_train: Rows the scaler was legitimately fitted on, shape ``(n, c)``.
        eps: The eps added to sigma by the standardizer.
        X_contaminated: Optional larger row set that must not reproduce the
            stored statistics.

    Raises:
        LeakageError: If the stored statistics do not match the train-only
            recomputation, or if they *do* match a contaminated recomputation.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2-D, got shape {X_train.shape}")
    if X_train.shape[0] < 2:
        raise ValueError("need at least 2 training rows to audit a scaler")

    expected_mean = np.nanmean(X_train, axis=0)
    expected_std = np.nanstd(X_train, axis=0) + eps

    if not np.allclose(np.asarray(scaler_mean, dtype=np.float64), expected_mean, atol=_STAT_TOLERANCE):
        raise LeakageError(
            "scaler mean does not match train-only statistics "
            f"(max |delta| = {np.max(np.abs(np.asarray(scaler_mean) - expected_mean)):.3e})"
        )
    if not np.allclose(np.asarray(scaler_std, dtype=np.float64), expected_std, atol=_STAT_TOLERANCE):
        raise LeakageError(
            "scaler std does not match train-only statistics "
            f"(max |delta| = {np.max(np.abs(np.asarray(scaler_std) - expected_std)):.3e})"
        )

    if X_contaminated is not None:
        contaminated = np.asarray(X_contaminated, dtype=np.float64)
        contam_mean = np.nanmean(contaminated, axis=0)
        contam_std = np.nanstd(contaminated, axis=0) + eps
        if np.allclose(contam_mean, expected_mean, atol=_STAT_TOLERANCE) and np.allclose(
            contam_std, expected_std, atol=_STAT_TOLERANCE
        ):
            raise LeakageError(
                "contaminated rows are indistinguishable from training rows; "
                "the audit has no power on this input and the split must be inspected"
            )


def audit_windows(
    window_specimens: Sequence[str],
    row_specimens: np.ndarray,
    window_length: int,
) -> None:
    """Verify every window lies entirely inside one specimen.

    Args:
        window_specimens: Specimen owning each window, length ``n_windows``.
        row_specimens: Per-row specimen labels, shape ``(n_windows, L)``.
        window_length: Expected window length ``L``.

    Raises:
        LeakageError: If a window mixes specimens or its rows disagree with its
            declared owner.
    """
    row_specimens = np.asarray(row_specimens)
    if row_specimens.ndim != 2:
        raise ValueError(f"row_specimens must be 2-D, got shape {row_specimens.shape}")
    if row_specimens.shape[1] != window_length:
        raise ValueError(
            f"row_specimens width {row_specimens.shape[1]} != window_length {window_length}"
        )
    if row_specimens.shape[0] != len(window_specimens):
        raise ValueError("window_specimens and row_specimens disagree on the number of windows")

    for i, owner in enumerate(window_specimens):
        distinct = set(row_specimens[i].tolist())
        if len(distinct) != 1:
            raise LeakageError(
                f"window {i} spans multiple specimens: {sorted(distinct)}"
            )
        if distinct.pop() != str(owner):
            raise LeakageError(
                f"window {i} declares specimen {owner!r} but its rows come from another specimen"
            )


def audit_window_split_membership(
    window_specimens: Sequence[str],
    fold: Fold,
    split: str,
) -> None:
    """Verify all windows handed to a split belong to that split's specimens.

    Args:
        window_specimens: Specimen owning each window.
        fold: The fold the windows were built for.
        split: One of ``train``, ``val``, ``test``.

    Raises:
        ValueError: If ``split`` is not a known split name.
        LeakageError: If any window belongs to a specimen outside the split.
    """
    allowed = {
        "train": _specimen_set(fold.train),
        "val": _specimen_set(fold.val),
        "test": _specimen_set(fold.test),
    }
    if split not in allowed:
        raise ValueError(f"unknown split {split!r}; expected {sorted(allowed)}")
    stray = sorted({str(s) for s in window_specimens} - allowed[split])
    if stray:
        raise LeakageError(
            f"fold {fold.fold_id}: windows for the {split} split reference specimens "
            f"that do not belong to it: {stray}"
        )


def audit_no_row_reuse(
    train_row_index: Sequence[int],
    test_row_index: Sequence[int],
) -> None:
    """Verify no underlying row is used on both sides of a split.

    This is the guard against a row-level ``train_test_split``: it catches the
    case where specimen bookkeeping looks correct but the same physical
    measurement was emitted twice.

    Args:
        train_row_index: Row identifiers used for training.
        test_row_index: Row identifiers used for testing.

    Raises:
        LeakageError: If any row identifier appears in both collections.
    """
    overlap = {str(i) for i in train_row_index} & {str(i) for i in test_row_index}
    if overlap:
        preview = sorted(overlap)[:10]
        raise LeakageError(
            f"{len(overlap)} rows are used in both train and test (e.g. {preview})"
        )


def audit_class_coverage(
    y_train: Sequence[int],
    y_test: Sequence[int],
    n_classes: int,
) -> dict[str, list[int]]:
    """Report class coverage per split without altering it.

    The protocol forbids moving specimens to fix class balance, so this returns
    a diagnostic instead of raising. It exists so that a split which happens to
    leave a class absent from training is visible in the run record rather than
    silently producing a misleading macro-F1.

    Args:
        y_train: Training labels.
        y_test: Test labels.
        n_classes: Total number of classes in the task.

    Returns:
        Mapping with ``missing_in_train`` and ``missing_in_test`` class indices.
    """
    train_present = set(int(v) for v in np.unique(np.asarray(y_train)))
    test_present = set(int(v) for v in np.unique(np.asarray(y_test)))
    all_classes = set(range(int(n_classes)))
    return {
        "missing_in_train": sorted(all_classes - train_present),
        "missing_in_test": sorted(all_classes - test_present),
    }


def leakage_report(
    frame: pd.DataFrame,
    fold: Fold,
    specimen_column: str = "specimen_id",
) -> dict[str, object]:
    """Summarise the split's actual specimen membership for the run record.

    Args:
        frame: The tidy dataset frame.
        fold: The fold being reported on.
        specimen_column: Column holding specimen identifiers.

    Returns:
        Mapping with per-split specimen lists and row counts.
    """
    present = set(frame[specimen_column].astype(str).unique())
    report: dict[str, object] = {}
    for name, members in (("train", fold.train), ("val", fold.val), ("test", fold.test)):
        resolve = [s for s in members if s in present]
        report[f"{name}_specimens"] = sorted(members)
        report[f"{name}_rows"] = int(frame[specimen_column].astype(str).isin(resolve).sum())
    return report
