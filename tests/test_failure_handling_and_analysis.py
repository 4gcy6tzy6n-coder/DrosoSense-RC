"""Failure handling in the runner, and the analysis that consumes its records.

Protocol v1.1 §20 says a run that cannot be fitted is recorded with ``status:
"failed"`` and a reason, counted in the table, and never silently dropped or
retried with different parameters. The alternative — letting one degenerate fold
abort a benchmark — is worse, because it hides how many runs were affected.

The second half exercises ``scripts/analyze.py``'s contrast builder, which is
where the declared resampling unit, the Holm family and the exploratory label
either hold or quietly stop meaning anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from drososense.evaluation.results import load_records  # noqa: E402
from drososense.utils.config import load_protocol  # noqa: E402


def _load_analyze():
    """Import ``scripts/analyze.py`` as a module.

    Returns:
        The imported module.
    """
    path = PROJECT_ROOT / "scripts" / "analyze.py"
    spec = importlib.util.spec_from_file_location("analyze_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_script"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A failing run is recorded, not fatal
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_a_failing_model_is_recorded_and_does_not_abort_the_benchmark(
    temporary_dataset, tmp_path, monkeypatch
):
    """One model blowing up must cost one model, not the whole experiment."""
    import drososense.evaluation.runner as runner_module
    from drososense.baselines.registry import ModelSpec

    dataset_id, _ = temporary_dataset

    def exploding_factory(*args, **kwargs):
        """Build a model whose fit always raises.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always.
        """

        class Exploding:
            """A model that cannot be fitted."""

            def fit(self, X, y):
                """Raise.

                Args:
                    X: Ignored.
                    y: Ignored.

                Raises:
                    RuntimeError: Always.
                """
                raise RuntimeError("The number of classes has to be greater than one")

            def describe(self):
                """Return an empty description.

                Returns:
                    An empty mapping.
                """
                return {}

        return Exploding()

    monkeypatch.setattr(runner_module, "build_model", exploding_factory)
    monkeypatch.setattr(
        runner_module,
        "model_spec",
        lambda model_id: ModelSpec(model_id, "classical", model_id, note=""),
    )

    summary = runner_module.run_benchmark(
        runner_module.BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="failure_probe",
            models=("svm_rbf", "random_forest"),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            n_splits=3,
            max_folds=1,
        ),
        raw_dir=tmp_path / "raw",
        tables_dir=tmp_path / "tables",
    )

    records = load_records(tmp_path / "raw")
    assert records, "the benchmark must still produce records"
    assert all(record.status == "failed" for record in records)
    for record in records:
        assert "greater than one" in record.failure_reason
        assert record.metrics["macro_f1"] is None

    # The failure is visible in the summary under its own status group, not
    # averaged together with any successful run.
    assert set(summary["status"]) == {"failed"}
    assert int(summary["n_runs"].sum()) == len(records)


@pytest.mark.integration
def test_a_successful_run_is_still_marked_ok(temporary_dataset, tmp_path):
    """The status field defaults to ok, so a healthy run is not accidentally muted."""
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    run_benchmark(
        BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="ok_probe",
            models=("svm_rbf",),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            n_splits=3,
            max_folds=1,
        ),
        raw_dir=tmp_path / "raw",
        tables_dir=tmp_path / "tables",
    )
    records = load_records(tmp_path / "raw")
    assert records and all(record.status == "ok" for record in records)
    assert all(record.failure_reason == "" for record in records)


# ---------------------------------------------------------------------------
# The contrast builder
# ---------------------------------------------------------------------------
def _frame(rows: list[dict]) -> "object":
    """Build a per-run frame from explicit rows.

    Args:
        rows: Row mappings.

    Returns:
        The frame.
    """
    import pandas as pd

    return pd.DataFrame(rows)


@pytest.mark.unit
def test_paired_observations_require_both_models_on_the_same_fold():
    """A contrast is paired; an unpaired model pairing must produce fewer rows."""
    analyze = _load_analyze()
    rows = []
    for seed in (0, 1):
        for fold in (0, 1):
            rows.append(
                {
                    "dataset": "D", "task": "classification", "seed": seed, "fold_id": fold,
                    "window_length": 16, "model": "a", "macro_f1": 0.5 + 0.1 * fold,
                }
            )
            if (seed, fold) != (1, 1):
                rows.append(
                    {
                        "dataset": "D", "task": "classification", "seed": seed, "fold_id": fold,
                        "window_length": 16, "model": "b", "macro_f1": 0.4 + 0.1 * fold,
                    }
                )
    paired = analyze.observations(_frame(rows), "a", "b", "macro_f1")
    assert len(paired) == 3, "the unpaired (seed 1, fold 1) evaluation must drop out"
    assert np.allclose(paired["delta"], 0.1)


@pytest.mark.unit
def test_declared_contrasts_are_corrected_and_exploratory_ones_are_not():
    """An exploratory p-value must not look pre-registered."""
    analyze = _load_analyze()
    protocol = load_protocol()

    rows = []
    for model, offset in (("esn", 0.0), ("gru", -0.05), ("svm_rbf", 0.03)):
        for seed in range(4):
            for fold in range(5):
                rows.append(
                    {
                        "dataset": "D", "task": "classification", "seed": seed, "fold_id": fold,
                        "window_length": 16, "model": model,
                        "macro_f1": 0.8 + offset + 0.001 * folder_noise(seed, fold),
                    }
                )
    table = analyze.build_contrast_table(
        _frame(rows),
        protocol,
        {"classification": "macro_f1"},
        exploratory=[("esn", "gru")],
    )
    assert not table.empty
    exploratory = table[table["classification"] == "exploratory"]
    assert len(exploratory) == 1
    assert exploratory["p_holm"].isna().all(), (
        "an exploratory contrast must not receive a family-corrected p-value"
    )
    assert np.isfinite(exploratory["p_value"]).all()
    # The resampling unit reaches the table, which is where a reader checks it.
    assert set(exploratory["n_clusters"]) == {5}
    assert exploratory["n_pairs"].iloc[0] == 20


def folder_noise(seed: int, fold: int) -> float:
    """Deterministic small jitter, so the fixture has non-degenerate differences.

    Args:
        seed: Seed index.
        fold: Fold index.

    Returns:
        A small deterministic value.
    """
    return float((seed * 7 + fold * 3) % 5) / 1000.0


@pytest.mark.unit
def test_a_declared_contrast_with_no_records_is_absent_not_fabricated():
    """The builder only emits rows it could actually compute."""
    analyze = _load_analyze()
    protocol = load_protocol()
    rows = [
        {
            "dataset": "D", "task": "classification", "seed": seed, "fold_id": fold,
            "window_length": 16, "model": "esn", "macro_f1": 0.8,
        }
        for seed in range(3)
        for fold in range(5)
    ]
    table = analyze.build_contrast_table(_frame(rows), protocol, {"classification": "macro_f1"})
    assert table.empty, "no declared contrast has both of its models present"
