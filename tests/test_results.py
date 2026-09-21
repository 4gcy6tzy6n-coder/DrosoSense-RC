"""Run records, metrics and aggregation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drososense.evaluation.metrics import (
    EMPTY_CLASS_POLICY,
    classification_metrics,
    regression_metrics,
)
from drososense.evaluation.results import (
    REQUIRED_RECORD_FIELDS,
    RunRecord,
    aggregate_records,
    capture_environment,
    load_records,
    make_run_id,
    metric_float,
    records_to_frame,
    write_record,
)


def _record(**overrides) -> RunRecord:
    """Build a minimal valid record for testing.

    Args:
        **overrides: Field values to override.

    Returns:
        The record.
    """
    payload = {
        "run_id": "ds|svm_rbf|classification|seed00|fold00",
        "experiment": "unit",
        "dataset": "ds",
        "model": "svm_rbf",
        "task": "classification",
        "seed": 0,
        "fold_id": 0,
        "protocol_version": "1.0.0",
        "window_length": 16,
        "metrics": {"macro_f1": 0.5, "auroc": None},
        "n_train_windows": 100,
        "n_test_windows": 50,
        "train_specimens": ["a", "b"],
        "test_specimens": ["c"],
        "duration_s": 1.25,
        "environment": {"python": "3.12"},
        "timestamp_utc": "2026-09-20T00:00:00+00:00",
    }
    payload.update(overrides)
    return RunRecord(**payload)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_perfect_classification_scores_one():
    """A perfect classifier scores 1.0 on every primary metric."""
    y = np.array([0, 1, 2, 3])
    scores = np.eye(4)
    metrics = classification_metrics(y, y, scores, n_classes=4)
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auroc_n_classes_scored"] == 4


@pytest.mark.unit
def test_auroc_is_absent_rather_than_invented_without_scores():
    """With no scores, AUROC is None — not 0.5, not omitted silently."""
    metrics = classification_metrics(np.array([0, 1]), np.array([0, 1]))
    assert metrics["auroc"] is None
    assert metrics["auroc_n_classes_scored"] == 0


@pytest.mark.unit
def test_auroc_is_null_unless_every_class_is_present():
    """The declared policy: four classes or nothing, never a partial average.

    R0 audit item X5: the protocol named a four-class macro-over-one-vs-rest
    AUROC while the implementation averaged over whichever two or three classes
    happened to be scorable, so a two-class number was being reported under a
    four-class name and folds were not comparable with each other.
    """
    y_true = np.array([0, 1, 0, 1])
    scores = np.random.default_rng(0).random((4, 4))
    metrics = classification_metrics(y_true, y_true, scores, n_classes=4)
    assert metrics["auroc"] is None
    assert metrics["auroc_n_classes_scored"] == 0
    assert metrics["auroc_defined"] is False
    assert metrics["empty_class_policy"] == EMPTY_CLASS_POLICY


@pytest.mark.unit
def test_auroc_n_classes_scored_is_either_four_or_zero():
    """There is no third value: the policy admits no partial average."""
    rng = np.random.default_rng(3)
    for n_present in (2, 3, 4):
        y_true = np.repeat(np.arange(n_present), 3)
        scores = rng.random((y_true.size, 4))
        metrics = classification_metrics(y_true, y_true, scores, n_classes=4)
        assert metrics["auroc_n_classes_scored"] in (0, 4)
        assert (metrics["auroc"] is None) == (metrics["auroc_n_classes_scored"] == 0)


@pytest.mark.unit
def test_macro_f1_stays_defined_when_a_class_is_absent():
    """macro-F1 uses the fixed label set, so AUROC's dropped coverage is not shared."""
    y_true = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y_true, y_true, None, n_classes=4)
    assert metrics["macro_f1"] is not None
    assert metrics["auroc"] is None


@pytest.mark.unit
def test_macro_f1_spans_the_declared_label_set_not_just_the_observed_one():
    """Macro-F1 is averaged over all four classes even when a split lacks some.

    Otherwise a fold whose test split happens to omit a class would be scored
    over three classes, and its macro-F1 could not be paired with a fold that
    has all four.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    over_four = classification_metrics(y_true, y_pred, n_classes=4)
    over_two = classification_metrics(y_true, y_pred, n_classes=2)
    assert over_two["macro_f1"] == 1.0
    assert over_four["macro_f1"] == pytest.approx(0.5)  # classes 2 and 3 score 0


@pytest.mark.unit
def test_macro_f1_ignores_majority_class_dominance():
    """Macro-F1 punishes a model that only predicts the majority class."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.9
    assert metrics["macro_f1"] < 0.5


@pytest.mark.unit
def test_regression_metrics_are_exact_for_a_known_case():
    """MAE, RMSE and R2 match hand-computed values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["r2"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_record_round_trips_through_json(tmp_path):
    """A record written to disk reloads identically."""
    record = _record()
    path = write_record(record, tmp_path)
    assert path.is_file()

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert set(REQUIRED_RECORD_FIELDS).issubset(payload)

    reloaded = load_records(tmp_path)
    assert len(reloaded) == 1
    assert reloaded[0] == record


@pytest.mark.unit
def test_record_requires_every_protocol_field():
    """A record missing a protocol-mandated field is rejected on load."""
    payload = _record().to_dict()
    del payload["window_length"]
    with pytest.raises(ValueError, match="missing required fields"):
        RunRecord.from_dict(payload)


@pytest.mark.unit
def test_evidence_class_is_restricted():
    """A record cannot claim an evidence class outside the two allowed values."""
    with pytest.raises(ValueError, match="evidence_class"):
        _record(evidence_class="probably_real")


@pytest.mark.unit
def test_record_path_keeps_raw_results_separated(tmp_path):
    """Raw records land under experiment/dataset/model."""
    path = write_record(_record(), tmp_path)
    assert path.parent.name == "svm_rbf"
    assert path.parent.parent.name == "ds"
    assert path.parent.parent.parent.name == "unit"
    assert path.name == "classification_seed00_fold00.json"


@pytest.mark.unit
def test_classification_and_regression_records_do_not_collide(tmp_path):
    """One model on one fold writes two records, not one overwriting the other."""
    write_record(_record(task="classification"), tmp_path)
    write_record(_record(task="regression", metrics={"mae": 0.5}), tmp_path)

    records = load_records(tmp_path)
    assert len(records) == 2
    assert {record.task for record in records} == {"classification", "regression"}


@pytest.mark.unit
def test_run_id_is_deterministic():
    """The run id is a pure function of its inputs."""
    first = make_run_id("ds", "gru", "classification", 3, 2)
    second = make_run_id("ds", "gru", "classification", 3, 2)
    assert first == second
    assert first != make_run_id("ds", "gru", "classification", 3, 1)


@pytest.mark.unit
def test_capture_environment_reports_core_libraries():
    """The environment block records the interpreter and key packages."""
    environment = capture_environment()
    assert environment["python"]
    assert environment["platform"]
    assert "numpy" in environment


@pytest.mark.unit
def test_metric_float_tolerates_nulls_and_non_finite_values():
    """Reading a metric never raises on a missing or NaN value."""
    record = _record(metrics={"macro_f1": None, "mae": float("nan"), "r2": 0.5})
    assert metric_float(record, "macro_f1") is None
    assert metric_float(record, "mae") is None
    assert metric_float(record, "r2") == 0.5
    assert metric_float(record, "absent") is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_aggregate_computes_mean_and_sd_across_seeds():
    """Mean and SD are computed per model across seeds."""
    records = [
        _record(seed=s, run_id=f"r{s}", metrics={"macro_f1": 0.4 + 0.1 * s}) for s in range(3)
    ]
    summary = aggregate_records(records)
    assert len(summary) == 1
    assert summary.iloc[0]["macro_f1_mean"] == pytest.approx(0.5)
    assert summary.iloc[0]["n_seeds"] == 3


@pytest.mark.unit
def test_aggregate_keeps_a_stable_column_schema_for_one_seed():
    """A single-seed summary still has an SD column, holding NaN."""
    summary = aggregate_records([_record()])
    assert "macro_f1_mean" in summary.columns
    assert "macro_f1_std" in summary.columns
    assert np.isnan(summary.iloc[0]["macro_f1_std"])


@pytest.mark.unit
def test_aggregate_never_mixes_evidence_classes():
    """Synthetic and real runs never land in the same aggregate row."""
    records = [
        _record(seed=0, evidence_class="real", metrics={"macro_f1": 0.9}),
        _record(seed=0, evidence_class="synthetic_fixture", metrics={"macro_f1": 0.1}),
    ]
    summary = aggregate_records(records)
    assert len(summary) == 2
    assert set(summary["evidence_class"]) == {"real", "synthetic_fixture"}


@pytest.mark.unit
def test_aggregate_never_mixes_compliant_and_non_compliant_splits():
    """A protocol-compliant and a non-compliant run are never averaged together."""
    records = [
        _record(seed=0, protocol_compliant=True, metrics={"macro_f1": 0.9}),
        _record(seed=0, protocol_compliant=False, metrics={"macro_f1": 0.1}),
    ]
    summary = aggregate_records(records)
    assert len(summary) == 2
    assert set(summary["protocol_compliant"]) == {True, False}


@pytest.mark.unit
def test_records_to_frame_keeps_one_row_per_run():
    """The tidy frame has a row per record and a column per metric."""
    records = [_record(seed=s, run_id=f"r{s}") for s in range(4)]
    frame = records_to_frame(records)
    assert len(frame) == 4
    assert "macro_f1" in frame.columns
    assert "auroc" in frame.columns
    assert frame["auroc"].isna().all()


@pytest.mark.unit
def test_empty_input_aggregates_to_an_empty_frame():
    """Aggregating nothing yields an empty frame rather than raising."""
    assert aggregate_records([]).empty
