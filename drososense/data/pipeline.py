"""Assemble leakage-audited fold tensors from a dataset and a fold.

This is the only path from a tidy frame to model-ready arrays. It performs, in
order: fold audit, train-only scaler fit, scaler audit, scaler persistence,
per-split standardisation, per-specimen windowing, window audit, split
membership audit and row-reuse audit. Any violation raises before a model is
constructed, so a leaky configuration cannot reach a results file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drososense.data.leakage import (
    audit_class_coverage,
    audit_fold,
    audit_no_row_reuse,
    audit_scaler,
    audit_window_split_membership,
    audit_windows,
    leakage_report,
)
from drososense.data.scaling import DEFAULT_EPS, Standardizer, fit_standardizer
from drososense.data.schema import CLASS_COLUMN, SPECIMEN_COLUMN, Dataset
from drososense.data.splits import Fold
from drososense.data.windowing import WindowSet, make_windows


@dataclass(frozen=True)
class FoldTensors:
    """Everything one model needs to run on one fold.

    Attributes:
        fold: The specimen partition these tensors were built from.
        scaler: The train-only standardizer, frozen.
        train: Training windows.
        val: Validation windows.
        test: Test windows.
        scaler_path: Location of the persisted ``scaler.pkl``.
        class_coverage: Diagnostic of classes absent from train or test.
        split_report: Specimen/row counts per split.
    """

    fold: Fold
    scaler: Standardizer
    train: WindowSet
    val: WindowSet
    test: WindowSet
    scaler_path: Path
    class_coverage: dict[str, list[int]]
    split_report: dict[str, object]

    def split(self, name: str) -> WindowSet:
        """Return one split's windows.

        Args:
            name: ``train``, ``val`` or ``test``.

        Returns:
            The requested :class:`WindowSet`.

        Raises:
            ValueError: If ``name`` is not a known split.
        """
        mapping = {"train": self.train, "val": self.val, "test": self.test}
        if name not in mapping:
            raise ValueError(f"unknown split {name!r}; expected {sorted(mapping)}")
        return mapping[name]

    def summarise(self) -> dict[str, object]:
        """Return a compact, JSON-serialisable description for the run record.

        Returns:
            Mapping with window counts, split membership and class coverage.
        """
        return {
            "fold_id": self.fold.fold_id,
            "strategy": self.fold.strategy,
            "fold_fingerprint": self.fold.fingerprint,
            "window_length": self.train.window_length,
            "n_train_windows": len(self.train),
            "n_val_windows": len(self.val),
            "n_test_windows": len(self.test),
            "n_train_specimens": len(self.fold.train),
            "n_val_specimens": len(self.fold.val),
            "n_test_specimens": len(self.fold.test),
            "n_train_sessions": len(set(self.train.session_ids.tolist())),
            "n_val_sessions": len(set(self.val.session_ids.tolist())),
            "n_test_sessions": len(set(self.test.session_ids.tolist())),
            "train_specimens": list(self.fold.train),
            "val_specimens": list(self.fold.val),
            "test_specimens": list(self.fold.test),
            "scaler_n_fit_rows": self.scaler.n_fit_rows,
            "class_coverage": self.class_coverage,
        }


def usable_specimens(dataset: Dataset, window_length: int) -> tuple[str, ...]:
    """Return specimens long enough to contribute at least one window.

    Specimens shorter than ``window_length`` are excluded *before* the split is
    built, so the partition is still a clean disjoint cover of what is actually
    used. The exclusion is recorded by the caller.

    Args:
        dataset: The normalised dataset.
        window_length: Window length ``L``.

    Returns:
        Specimens with at least ``window_length`` rows, sorted.
    """
    counts = dataset.frame.groupby(SPECIMEN_COLUMN, observed=True).size()
    return tuple(sorted(str(s) for s, n in counts.items() if int(n) >= window_length))


def build_fold_tensors(
    dataset: Dataset,
    fold: Fold,
    window_length: int,
    artifact_dir: str | Path,
    stride: int = 1,
    eps: float = DEFAULT_EPS,
    label_rule: str = "last",
) -> FoldTensors:
    """Build audited train/val/test windows for one fold.

    Args:
        dataset: Normalised dataset.
        fold: Specimen partition to realise.
        window_length: Window length ``L``.
        artifact_dir: Directory to write ``scaler.pkl`` into.
        stride: Step between window starts.
        eps: Epsilon added to sigma when fitting the standardizer.
        label_rule: ``last`` or ``majority``.

    Returns:
        The assembled :class:`FoldTensors`.

    Raises:
        ValueError: If a fold specimen is too short to window.
        LeakageError: If any audit fails.
    """
    all_specimens = dataset.specimens()
    audit_fold(fold, all_specimens)

    short = set(fold.specimens) - set(usable_specimens(dataset, window_length))
    if short:
        raise ValueError(
            f"fold {fold.fold_id}: specimens too short for window_length={window_length}: "
            f"{sorted(short)}. Exclude them before building the fold via usable_specimens()."
        )

    frame = dataset.frame
    membership = frame[SPECIMEN_COLUMN].astype(str)
    feature_columns = list(dataset.schema.feature_columns)

    train_mask = membership.isin(set(fold.train)).to_numpy()
    val_mask = membership.isin(set(fold.val)).to_numpy()
    test_mask = membership.isin(set(fold.test)).to_numpy()

    # 1. Fit the scaler on training rows only, then prove it.
    x_train = frame.loc[train_mask, feature_columns].to_numpy(dtype=np.float64)
    scaler = fit_standardizer(x_train, eps=eps)
    x_contaminated = frame.loc[train_mask | val_mask | test_mask, feature_columns].to_numpy(
        dtype=np.float64
    )
    audit_scaler(scaler.mean, scaler.std, x_train, eps, X_contaminated=x_contaminated)

    # 2. Persist the frozen scaler next to the split it belongs to.
    artifact_dir = Path(artifact_dir)
    scaler_path = scaler.save(artifact_dir)

    # 3. Standardise every split with the SAME frozen scaler — never refit.
    scaled = frame.copy()
    scaled.loc[:, feature_columns] = scaler.transform(
        frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    )

    windows = {
        "train": make_windows(scaled.loc[train_mask], feature_columns, window_length, stride, label_rule),
        "val": make_windows(scaled.loc[val_mask], feature_columns, window_length, stride, label_rule),
        "test": make_windows(scaled.loc[test_mask], feature_columns, window_length, stride, label_rule),
    }

    # 4. Verify the produced windows, not the intention behind them.
    for name, window_set in windows.items():
        audit_windows(window_set.specimen_ids, window_set.row_specimens, window_length)
        # The same audit, run on the session key: a window that stays inside one
        # specimen can still straddle two acquisition occasions, and that is a
        # defect the specimen audit alone cannot see.
        audit_windows(window_set.session_ids, window_set.row_sessions, window_length)
        audit_window_split_membership(window_set.specimen_ids, fold, name)
    audit_no_row_reuse(
        windows["train"].row_index.ravel().tolist(),
        windows["val"].row_index.ravel().tolist(),
    )
    audit_no_row_reuse(
        windows["train"].row_index.ravel().tolist(),
        windows["test"].row_index.ravel().tolist(),
    )
    audit_no_row_reuse(
        windows["val"].row_index.ravel().tolist(),
        windows["test"].row_index.ravel().tolist(),
    )

    coverage = audit_class_coverage(
        windows["train"].y_class,
        windows["test"].y_class,
        dataset.schema.n_classes,
    )

    return FoldTensors(
        fold=fold,
        scaler=scaler,
        train=windows["train"],
        val=windows["val"],
        test=windows["test"],
        scaler_path=scaler_path,
        class_coverage=coverage,
        split_report=leakage_report(frame, fold, SPECIMEN_COLUMN),
    )
