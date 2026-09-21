"""Unit tests for :mod:`scripts.server.concurrency_sweep`.

The tests cover the summariser and the CLI argument parser; the full pool
machinery is exercised by the ``test_concurrency_sweep_e2e`` test, which is
slow and skipped unless ``--run-e2e`` is given to pytest.
"""

from __future__ import annotations

import pytest

from scripts.server import concurrency_sweep


def test_child_result_round_trip() -> None:
    """``ChildResult`` is a frozen dataclass and round-trips through asdict."""
    result = concurrency_sweep.ChildResult(worker_id=0, latency_s=1.5, exit_code=0)
    dumped = concurrency_sweep.asdict(result)
    rebuilt = concurrency_sweep.ChildResult(**dumped)
    assert rebuilt == result


def test_summarise_child_latencies_handles_empty() -> None:
    """With no successful children the helper returns zeros and the failure count."""
    mean, std, failures = concurrency_sweep._summarise_child_latencies([])
    assert mean == 0.0
    assert std == 0.0
    assert failures == 0


def test_summarise_child_latencies_counts_failures() -> None:
    """Non-zero exit codes contribute to the failure tally, not the latency stats."""
    results = [
        concurrency_sweep.ChildResult(worker_id=0, latency_s=2.0, exit_code=0),
        concurrency_sweep.ChildResult(worker_id=1, latency_s=1.0, exit_code=2),
    ]
    mean, std, failures = concurrency_sweep._summarise_child_latencies(results)
    assert mean == pytest.approx(2.0)
    assert std == 0.0  # only one successful sample
    assert failures == 1


def test_summarise_child_latencies_std_for_two() -> None:
    """With two successful children the standard deviation is the half-range."""
    results = [
        concurrency_sweep.ChildResult(worker_id=0, latency_s=2.0, exit_code=0),
        concurrency_sweep.ChildResult(worker_id=1, latency_s=4.0, exit_code=0),
    ]
    mean, std, failures = concurrency_sweep._summarise_child_latencies(results)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)
    assert failures == 0


def test_parse_args_defaults() -> None:
    """The default sweep covers the published worker counts and a small model subset."""
    args = concurrency_sweep.parse_args([])
    assert tuple(args.workers) == concurrency_sweep.DEFAULT_WORKER_COUNTS
    assert args.runs_per_worker == 2
    assert args.models == ["svm_rbf", "random_forest", "gru", "esn"]


def test_sweep_row_round_trip() -> None:
    """``SweepRow`` survives an asdict round-trip."""
    row = concurrency_sweep.SweepRow(
        workers=4, runs_per_worker=2, total_runs=8,
        wall_clock_s=10.0, runs_per_minute=48.0,
        per_run_latency_s_mean=10.0, per_run_latency_s_std=0.0,
        n_failures=0, torch_num_threads=1, model_subset="svm_rbf",
    )
    rebuilt = concurrency_sweep.SweepRow(**concurrency_sweep.asdict(row))
    assert rebuilt == row