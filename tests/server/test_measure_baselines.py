"""Unit tests for :mod:`scripts.server.measure_baselines`.

These tests are intentionally narrow: they exercise the row-shape dataclasses,
the summariser, the device selector and the public CLI's argument parser
without running a real model. The end-to-end measurement is smoke-tested in
``test_measure_baselines_e2e.py`` and gated behind ``--run-e2e`` so a
``pytest -q`` on the laptop stays fast.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest

from scripts.server import measure_baselines


def test_per_rep_row_round_trips_through_asdict() -> None:
    """A ``PerRepRow`` survives a dict round-trip with all fields preserved."""
    row = measure_baselines.PerRepRow(
        model="svm_rbf", repetition=2, seed=7, task="classification",
        n_train_windows=10, n_val_windows=5, n_channels=4,
        train_time_s=1.5, predict_time_s=0.5, ms_per_sample=100.0,
        peak_rss_kb=2048, peak_gpu_mb=0.0,
        n_trainable_parameters=None, device="cpu", torch_num_threads=1,
    )
    dumped = measure_baselines.asdict(row)
    rebuilt = measure_baselines.PerRepRow(**dumped)
    assert rebuilt == row


def test_summary_row_defaults_failure_reason_to_empty() -> None:
    """A bare ``SummaryRow`` defaults ``failure_reason`` to the empty string."""
    row = measure_baselines.SummaryRow(
        model="x", task="classification", repetitions=1,
        n_train_windows=0, n_val_windows=0, n_channels=0,
        train_time_s_mean=0.0, train_time_s_std=0.0,
        predict_time_s_mean=0.0, predict_time_s_std=0.0,
        ms_per_sample_mean=0.0, ms_per_sample_std=0.0,
        peak_rss_kb_mean=0.0, peak_gpu_mb_mean=0.0,
        n_trainable_parameters=0.0,
        device="cpu", torch_num_threads=1,
    )
    assert row.failure_reason == ""


def test_summarise_empty_rows_carry_failure_reason() -> None:
    """When every repetition failed, the summary still records the reason."""
    summary = measure_baselines._summarise([], failure_reason="boom")
    assert summary.repetitions == 0
    assert summary.failure_reason == "boom"
    assert summary.model == ""


def test_summarise_aggregates_means_and_stds() -> None:
    """Mean/std across repetitions match ``statistics`` for a known set."""
    rows = [
        measure_baselines.PerRepRow(
            model="svm_rbf", repetition=i, seed=i, task="classification",
            n_train_windows=100, n_val_windows=50, n_channels=8,
            train_time_s=float(i + 1), predict_time_s=0.1 * (i + 1),
            ms_per_sample=2.0 * (i + 1),
            peak_rss_kb=1000 + i, peak_gpu_mb=0.0,
            n_trainable_parameters=42, device="cpu", torch_num_threads=1,
        )
        for i in range(3)
    ]
    summary = measure_baselines._summarise(rows, failure_reason="")
    assert summary.repetitions == 3
    assert math.isclose(summary.train_time_s_mean, 2.0)
    assert summary.train_time_s_std > 0.0
    assert summary.predict_time_s_std > 0.0
    assert summary.failure_reason == ""


def test_select_device_classical_models_always_cpu() -> None:
    """Classical estimators are CPU-only regardless of the ``--use-gpu`` flag."""
    label, torch_device = measure_baselines._select_device("svm_rbf", use_gpu=True)
    assert label == "cpu"
    assert torch_device == ""


def test_select_device_torch_models_respect_flag() -> None:
    """Torch models pick CPU unless ``--use-gpu`` was passed AND CUDA is available."""
    label, _ = measure_baselines._select_device("gru", use_gpu=False)
    assert label == "cpu"
    label, _ = measure_baselines._select_device("gru", use_gpu=True)
    assert label in {"cpu", "cuda:0"}


def test_parse_args_defaults_to_all_models() -> None:
    """The default model set covers every registered baseline."""
    args = measure_baselines.parse_args([])
    assert tuple(args.models) == measure_baselines.MODEL_IDS
    assert args.repetitions == measure_baselines.DEFAULT_REPETITIONS
    assert args.task == "classification"
    assert args.fold_index == 0


def test_parse_args_honours_limit() -> None:
    """``--limit`` caps the model list, used by the developer smoke harness."""
    args = measure_baselines.parse_args(["--models", "svm_rbf", "gru", "esn", "--limit", "2"])
    assert args.limit == 2


def test_ensure_fixture_writes_to_data_raw(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ensure_fixture`` materialises the CSV at the expected path."""
    # The fixture uses DATA_RAW_DIR which is the project-root relative path;
    # isolate it to a tmp dir so we do not pollute the real data tree.
    monkeypatch.setattr(measure_baselines, "DATA_RAW_DIR", tmp_path / "data" / "raw")
    target = measure_baselines._ensure_fixture()
    assert target.exists()
    assert target.suffix == ".csv"
    # Header must contain the canonical schema columns.
    with target.open(encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    for required in ("specimen_id", "time_index", "freshness_class", "tvc"):
        assert required in first_line


def test_ensure_fixture_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call leaves the file untouched (same hash)."""
    monkeypatch.setattr(measure_baselines, "DATA_RAW_DIR", tmp_path / "data" / "raw")
    target = measure_baselines._ensure_fixture()
    first_hash = target.read_bytes()
    measure_baselines._ensure_fixture()
    second_hash = target.read_bytes()
    assert first_hash == second_hash


def test_build_fold_train_and_val_arrays_differ(project_root: Path) -> None:
    """The fold tensors expose distinct train, val and test arrays."""
    # Wipe any cached fixture first so _build_fold does not rely on disk state.
    fixture_dir = measure_baselines.DATA_RAW_DIR / "synthetic_enose"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    if not (fixture_dir / measure_baselines.FIXTURE_FILENAME).exists():
        measure_baselines._ensure_fixture()
    fold_tensors = measure_baselines._build_fold(fold_index=0, seed=0)
    assert fold_tensors.train.X is not fold_tensors.val.X
    assert fold_tensors.train.X is not fold_tensors.test.X
    assert fold_tensors.val.X.shape[0] > 0