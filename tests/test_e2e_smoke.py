"""End-to-end pipeline test.

Runs the real driver over a real (temporary) dataset with real models, and
checks the artifacts it leaves behind. The point is to catch integration breaks
that unit tests cannot see: a config field the loader ignores, a metric the
aggregator drops, a record field the writer forgets.

The dataset is generated in ``tmp_path`` and the loader/registry lookups are
redirected there, so the suite stays hermetic and leaves no repository state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from drososense.baselines.registry import MODEL_IDS
from drososense.evaluation.results import load_records
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

TINY_PARAMS: dict[str, dict] = {
    "svm_rbf": {"C": 1.0},
    "random_forest": {"n_estimators": 15},
    "xgboost": {"n_estimators": 15, "max_depth": 3},
    "pca_svm": {"n_components": 4},
    "gru": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "lstm": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "cnn1d": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "tcn": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "esn": {"reservoir_size": 40, "density": 0.1, "washout": 1},
}


@pytest.mark.integration
def test_full_pipeline_writes_traceable_records(temporary_dataset, tmp_path):
    """A benchmark run produces raw records and a summary that match each other."""
    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "results_raw"
    tables_dir = tmp_path / "results_tables"

    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e2e",
        models=("svm_rbf", "random_forest", "esn"),
        tasks=("classification", "regression"),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params=TINY_PARAMS,
    )
    summary = run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)

    records = load_records(raw_dir)
    assert len(records) == 3 * 2  # three models x two tasks
    assert not summary.empty

    for record in records:
        assert record.protocol_compliant is True
        assert record.evidence_class == "real"
        assert record.window_length == 8
        assert record.n_train_windows > 0
        assert record.n_test_windows > 0
        assert not set(record.train_specimens) & set(record.test_specimens)
        assert record.metrics
        assert record.duration_s >= 0.0
        assert record.environment["python"]

    assert (tables_dir / "e2e_summary.csv").is_file()
    assert set(summary["dataset"]) == {dataset_id}
    assert set(summary["task"]) == {"classification", "regression"}


@pytest.mark.integration
def test_every_available_model_runs_end_to_end(temporary_dataset, tmp_path):
    """Every baseline whose backend is installed integrates with the pipeline.

    The assertion is deliberately against the models available in THIS
    environment, not against the full registry. A machine without xgboost cannot
    run an xgboost baseline, and a suite that fails there — or a claim that
    "159 passed" made without saying on which machine — is the reporting defect
    the R0 audit flagged (item C7). Which models were skipped, and why, is
    asserted separately below so the gap is recorded rather than hidden.
    """
    from drososense.baselines.registry import model_availability

    dataset_id, _ = temporary_dataset
    availability = model_availability()
    available = [m for m in MODEL_IDS if availability[m]["available"]]
    assert available, "no models are installed; this environment cannot exercise the pipeline"

    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e2e_all",
        models=MODEL_IDS,
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params=TINY_PARAMS,
    )
    summary = run_benchmark(config, raw_dir=tmp_path / "raw", tables_dir=tmp_path / "tables")

    ran = set(summary["model"])
    assert ran == set(available), f"models that did not produce a result: {set(available) - ran}"
    assert (summary["macro_f1_mean"].notna()).all()

    # The gap is asserted, not tolerated: a model that is unavailable must appear
    # in the recorded skipped list with a reason, so a comparison table built
    # from this run can state what is missing from it.
    skipped = {e["model"] for e in summary.attrs.get("skipped_models", [])}
    assert skipped == set(MODEL_IDS) - set(available), (
        f"skipped={sorted(skipped)} but unavailable={sorted(set(MODEL_IDS) - set(available))}"
    )
    for entry in summary.attrs.get("skipped_models", []):
        assert entry["reason"] and entry["reason"] != "None"


@pytest.mark.integration
def test_an_unavailable_model_is_reported_not_silently_dropped(temporary_dataset, tmp_path, monkeypatch):
    """A skipped model leaves a trace, so the comparison table can state the gap."""
    import drososense.evaluation.runner as runner_module

    dataset_id, _ = temporary_dataset
    monkeypatch.setattr(
        runner_module,
        "model_availability",
        lambda: {
            model_id: {
                "available": model_id != "gru",
                "reason": None if model_id != "gru" else "forced unavailable for the test",
            }
            for model_id in MODEL_IDS
        },
    )

    summary = run_benchmark(
        BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="e2e_skip",
            models=("svm_rbf", "gru"),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            n_splits=3,
            max_folds=1,
            model_params=TINY_PARAMS,
        ),
        raw_dir=tmp_path / "raw",
        tables_dir=tmp_path / "tables",
    )

    assert set(summary["model"]) == {"svm_rbf"}
    assert summary.attrs.get("skipped_models"), "the skipped model must be recorded"


@pytest.mark.integration
def test_non_compliant_dataset_is_flagged_in_every_record(temporary_dataset, tmp_path, monkeypatch):
    """A time-block dataset produces records marked non-compliant."""
    import drososense.evaluation.runner as runner_module

    dataset_id, config_path = temporary_dataset
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["specimen"] = {
        "source": "time_block",
        "block_by": "rows",
        "block_size": 20,
        "note": "assumed for the test",
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # The loader is already redirected at this dataset by the fixture.
    monkeypatch.setattr(runner_module, "dataset_config_path", lambda _: config_path)

    summary = run_benchmark(
        BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="e2e_noncompliant",
            models=("svm_rbf",),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            n_splits=3,
            max_folds=1,
            model_params=TINY_PARAMS,
        ),
        raw_dir=tmp_path / "raw",
        tables_dir=tmp_path / "tables",
    )

    assert summary["protocol_compliant"].all() is False or not summary["protocol_compliant"].any()
    records = load_records(tmp_path / "raw")
    assert all(record.protocol_compliant is False for record in records)
    assert all("NOT satisfied" in record.notes for record in records)
