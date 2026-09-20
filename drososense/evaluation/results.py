"""Run records and aggregation.

Raw and summary results are kept apart by construction:

* ``results/raw/<experiment>/<dataset>/<model>/seed<seed>_fold<fold>.json`` holds
  one record per (model, seed, fold) with full provenance.
* ``results/tables/<experiment>_summary.csv`` holds the aggregated mean ± SD
  across seeds.

A summary is always regenerable from the raw records; the reverse is not true,
which is why the raw records are the source of truth.

Every record carries an ``evidence_class``. It is ``real`` only for observed
data; the synthetic fixture is ``synthetic_fixture``, and a split that does not
satisfy ``split_unit: specimen`` is flagged ``protocol_compliant: false``. Both
flags travel with the number all the way into the summary table, so a
non-compliant or synthetic number cannot be mistaken for a protocol result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir

# Fields the frozen protocol requires on every record (section 14).
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "run_id",
    "experiment",
    "dataset",
    "model",
    "task",
    "seed",
    "fold_id",
    "protocol_version",
    "window_length",
    "metrics",
    "n_train_windows",
    "n_test_windows",
    "train_specimens",
    "test_specimens",
    "duration_s",
    "environment",
    "timestamp_utc",
)


@dataclass(frozen=True)
class RunRecord:
    """One model's result on one fold of one dataset at one seed.

    Attributes:
        run_id: Deterministic identifier of this exact evaluation.
        experiment: Experiment the run belongs to, e.g. ``smoke``.
        dataset: Dataset identifier.
        model: Model identifier.
        task: ``classification`` or ``regression``.
        seed: Seed for the split and the model.
        fold_id: Fold index within the seed's split.
        protocol_version: Protocol the run was produced under.
        window_length: Window length used.
        metrics: Computed metrics.
        n_train_windows: Training window count.
        n_test_windows: Test window count.
        train_specimens: Specimens used for training.
        test_specimens: Specimens held out for testing.
        duration_s: Wall-clock fitting plus prediction time.
        environment: Interpreter and library versions.
        timestamp_utc: When the run finished.
        evidence_class: ``real`` or ``synthetic_fixture``.
        protocol_compliant: Whether the split satisfied ``split_unit: specimen``.
        model_description: The model's own description of itself.
        fold_fingerprint: Identifier of the exact partition used.
        class_coverage: Classes absent from train or test.
        config_hash: Fingerprint of the run configuration.
        test_fingerprint: Identifier of the (test split, window length, model,
            task) evaluation this run represents. Protocol v1.1 requires each
            such evaluation to happen exactly once; a second record with the
            same fingerprint and a different configuration is a violation.
        empty_class_policy: The declared AUROC empty-class policy in force.
        n_train_sessions: Acquisition sessions in the training split.
        n_test_sessions: Acquisition sessions in the test split.
        status: ``ok``, ``failed``, ``skipped``, ``oom`` or ``timeout``.
        failure_reason: Populated whenever ``status`` is not ``ok``.
        notes: Free-text caveats attached to this run.
    """

    run_id: str
    experiment: str
    dataset: str
    model: str
    task: str
    seed: int
    fold_id: int
    protocol_version: str
    window_length: int
    metrics: dict[str, Any]
    n_train_windows: int
    n_test_windows: int
    train_specimens: list[str]
    test_specimens: list[str]
    duration_s: float
    environment: dict[str, Any]
    timestamp_utc: str
    evidence_class: str = "real"
    protocol_compliant: bool = True
    model_description: dict[str, Any] = field(default_factory=dict)
    fold_fingerprint: str = ""
    class_coverage: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    test_fingerprint: str = ""
    empty_class_policy: str = ""
    n_train_sessions: int = 0
    n_test_sessions: int = 0
    status: str = "ok"
    failure_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.evidence_class not in ("real", "synthetic_fixture"):
            raise ValueError(
                f"evidence_class must be 'real' or 'synthetic_fixture', got {self.evidence_class!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to plain JSON-compatible types.

        Returns:
            Mapping of every field.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Mapping containing at least the required fields.

        Returns:
            The reconstructed record.

        Raises:
            ValueError: If a required field is missing.
        """
        missing = [name for name in REQUIRED_RECORD_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"run record is missing required fields: {missing}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def capture_environment() -> dict[str, Any]:
    """Capture interpreter and library versions for reproducibility.

    Returns:
        Mapping with the Python version, platform and key package versions.
    """
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost"):
        try:
            module = __import__(module_name)
            environment[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            environment[module_name] = None
    return environment


def record_path(record: RunRecord, base_dir: Path | None = None) -> Path:
    """Return the canonical location of a record's JSON file.

    The task is part of the filename because one model on one fold produces a
    classification record AND a regression record, and they must not collide.

    Args:
        record: The record.
        base_dir: Override for ``results/raw``.

    Returns:
        Path to the JSON file.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_RAW_DIR
    return (
        base
        / record.experiment
        / record.dataset
        / record.model
        / f"{record.task}_seed{record.seed:02d}_fold{record.fold_id:02d}.json"
    )


def write_record(record: RunRecord, base_dir: Path | None = None) -> Path:
    """Write a record to its canonical path.

    Args:
        record: The record to persist.
        base_dir: Override for ``results/raw``.

    Returns:
        Path written.
    """
    path = record_path(record, base_dir)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True, default=str)
    return path


def load_records(base_dir: Path | None = None) -> list[RunRecord]:
    """Load every run record under ``results/raw``.

    Args:
        base_dir: Override for ``results/raw``.

    Returns:
        Records sorted by experiment, dataset, model, seed and fold.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_RAW_DIR
    records: list[RunRecord] = []
    if not base.is_dir():
        return records
    for path in sorted(base.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(RunRecord.from_dict(json.load(handle)))
    records.sort(key=lambda r: (r.experiment, r.dataset, r.model, r.task, r.seed, r.fold_id))
    return records


def _metric_names(records: list[RunRecord]) -> list[str]:
    """Collect the union of metric names across records.

    Args:
        records: Records to scan.

    Returns:
        Sorted metric names.
    """
    names: set[str] = set()
    for record in records:
        names.update(record.metrics)
    return sorted(names)


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Flatten records into a tidy per-run frame.

    Args:
        records: Records to flatten.

    Returns:
        Frame with one row per run and one column per metric.
    """
    if not records:
        return pd.DataFrame()
    metric_names = _metric_names(records)
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "run_id": record.run_id,
            "experiment": record.experiment,
            "dataset": record.dataset,
            "model": record.model,
            "task": record.task,
            "seed": record.seed,
            "fold_id": record.fold_id,
            "window_length": record.window_length,
            "protocol_version": record.protocol_version,
            "evidence_class": record.evidence_class,
            "protocol_compliant": record.protocol_compliant,
            "status": record.status,
            "n_train_windows": record.n_train_windows,
            "n_test_windows": record.n_test_windows,
            "n_train_specimens": len(record.train_specimens),
            "n_test_specimens": len(record.test_specimens),
            "n_train_sessions": record.n_train_sessions,
            "n_test_sessions": record.n_test_sessions,
            "duration_s": record.duration_s,
        }
        for name in metric_names:
            row[name] = record.metrics.get(name)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_records(
    records: list[RunRecord],
    group_by: tuple[str, ...] = (
        "experiment",
        "dataset",
        "model",
        "task",
        "window_length",
        "protocol_version",
        "evidence_class",
        "protocol_compliant",
        "status",
    ),
) -> pd.DataFrame:
    """Aggregate per-run metrics into mean and SD across seeds.

    The grouping deliberately includes ``evidence_class``, ``protocol_compliant``
    and ``status``: aggregating a synthetic run together with real ones, a
    non-compliant split together with a compliant one, or a failed run together
    with a successful one would produce a number that describes none of them.

    Args:
        records: Records to aggregate.
        group_by: Columns forming an aggregate group.

    Returns:
        Frame with ``<metric>_mean``, ``<metric>_sd`` and ``n_seeds`` columns.
    """
    frame = records_to_frame(records)
    if frame.empty:
        return frame

    # Failed runs stay in the frame under their own status group so the summary
    # can state how many runs were lost, rather than averaging over survivors and
    # implying complete coverage.
    metric_columns = [
        c
        for c in frame.columns
        if c
        not in {
            "run_id",
            "seed",
            "fold_id",
            "protocol_version",
            "evidence_class",
            "protocol_compliant",
            "status",
            "n_train_windows",
            "n_test_windows",
            "n_train_specimens",
            "n_test_specimens",
            "n_train_sessions",
            "n_test_sessions",
            "duration_s",
            *group_by,
        }
    ]

    # Only numeric metrics are averaged. A record's `metrics` mapping also
    # carries descriptive entries (the empty-class policy in force, for example);
    # taking a mean of those is meaningless, and silently coercing them would be
    # worse than leaving them out.
    metric_columns = [
        c for c in metric_columns if pd.api.types.is_numeric_dtype(frame[c])
    ]
    metric_columns = [c for c in metric_columns if c != "auroc_defined"]

    grouped = frame.groupby(list(group_by), dropna=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped["seed"].nunique()
    summary["n_runs"] = grouped.size()
    summary["n_specimens_tested"] = grouped["n_test_specimens"].sum()
    summary["mean_duration_s"] = grouped["duration_s"].mean()
    # AUROC is defined only on folds where every class is present, so the number
    # of runs that contributed to `auroc_mean` is reported next to it rather than
    # left for a reader to assume equals n_runs.
    if "auroc_defined" in frame.columns:
        summary["n_auroc_defined"] = grouped["auroc_defined"].sum()
    summary = summary.reset_index()

    # Pin the summary schema explicitly rather than relying on how the installed
    # pandas version happens to handle an all-NaN aggregation. A group with a
    # single run has an undefined SD; if that column were ever dropped, a
    # downstream table would silently imply that uncertainty was never computed
    # rather than showing it as undefined. Re-adding the pairs also guarantees
    # that a summary built from one seed and one built from ten have identical
    # columns, so they can be concatenated.
    expected = [f"{metric}_{stat}" for metric in metric_columns for stat in ("mean", "std")]
    ordered = [
        *group_by,
        *expected,
        "n_seeds",
        "n_runs",
        "n_specimens_tested",
        "n_auroc_defined",
        "mean_duration_s",
    ]
    summary = summary.reindex(columns=ordered)
    for column in expected:
        summary[column] = summary[column].astype(float)
    return summary


def write_summary_csv(summary: pd.DataFrame, experiment: str, base_dir: Path | None = None) -> Path:
    """Write an aggregated summary to ``results/tables``.

    Args:
        summary: Frame produced by :func:`aggregate_records`.
        experiment: Experiment name used in the filename.
        base_dir: Override for ``results/tables``.

    Returns:
        Path written.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_TABLES_DIR
    ensure_dir(base)
    path = base / f"{experiment}_summary.csv"
    summary.to_csv(path, index=False)
    return path


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timestamp, e.g. ``2026-09-20T12:00:00+00:00``.
    """
    return datetime.now(timezone.utc).isoformat()


def make_run_id(dataset: str, model: str, task: str, seed: int, fold_id: int) -> str:
    """Build the deterministic identifier of a run.

    Args:
        dataset: Dataset identifier.
        model: Model identifier.
        task: Task name.
        seed: Seed.
        fold_id: Fold index.

    Returns:
        Identifier string.
    """
    return f"{dataset}|{model}|{task}|seed{seed:02d}|fold{fold_id:02d}"


def make_test_fingerprint(
    fold_fingerprint: str, window_length: int, model: str, task: str
) -> str:
    """Build the identifier of one test-set evaluation.

    The fold fingerprint already identifies the exact partition, so combining it
    with the window length, model and task identifies a single scored evaluation.
    Two runs sharing this value have scored the same held-out data with the same
    model and task, which protocol v1.1 §17 permits only once.

    Args:
        fold_fingerprint: The fold's partition fingerprint.
        window_length: Window length used.
        model: Model identifier.
        task: Task name.

    Returns:
        Short hex identifier.
    """
    payload = f"{fold_fingerprint}|w{window_length}|{model}|{task}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def metric_float(record: RunRecord, name: str) -> float | None:
    """Read a metric as a float, tolerating nulls and non-finite values.

    Args:
        record: The record to read.
        name: Metric name.

    Returns:
        The value, or ``None`` when absent or not finite.
    """
    value = record.metrics.get(name)
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if np.isfinite(as_float) else None
