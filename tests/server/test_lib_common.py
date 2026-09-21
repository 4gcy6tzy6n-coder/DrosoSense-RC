"""Unit tests for :mod:`scripts.server.lib_common`.

These tests cover the small bits of plumbing the harness depends on: path
bootstrap, RSS sampling, GPU memory helpers and the env fingerprint. The
``measure_baselines`` and ``concurrency_sweep`` tests live next to this one
in their own files because they exercise different code paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.server import lib_common


def test_bootstrap_project_root_returns_repo_root(project_root: Path) -> None:
    """``bootstrap_project_root`` resolves to the directory with drosophila + scripts."""
    resolved = lib_common.bootstrap_project_root(Path(__file__).resolve().parent)
    assert resolved == project_root
    assert (resolved / "drososense").is_dir()
    assert (resolved / "scripts").is_dir()
    # Must also have side-effected sys.path so subsequent imports work.
    assert str(project_root) in lib_common.sys.path


def test_bootstrap_project_root_rejects_orphan(tmp_path: Path) -> None:
    """Walking up from an orphan directory raises with a clear message."""
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    with pytest.raises(RuntimeError, match="could not locate a project root"):
        lib_common.bootstrap_project_root(orphan)


def test_read_self_rss_kb_is_nonnegative() -> None:
    """``read_self_rss_kb`` returns either ``0`` (kernel hides it) or a positive int."""
    rss = lib_common.read_self_rss_kb()
    assert isinstance(rss, int)
    assert rss >= 0


def test_rss_poller_records_peak() -> None:
    """The poller reports a peak across its active window."""
    with lib_common.RssPoller(interval_s=0.01) as poller:
        peak_during_run = lib_common.read_self_rss_kb()
        # Sleep at least one poll interval so the thread has a chance to record.
        import time

        time.sleep(0.05)
    assert poller._peak_kb >= 0
    # We cannot guarantee the poller observed the same peak we read, only that
    # the peak it reports is at least the seed reading taken at entry.
    assert poller._peak_kb >= peak_during_run


def test_rss_poller_stop_is_idempotent() -> None:
    """Calling ``stop`` twice does not raise or wedge the thread."""
    poller = lib_common.RssPoller(interval_s=0.01)
    poller.__enter__()
    poller.stop()
    poller.stop()


def test_gpu_available_returns_bool() -> None:
    """``gpu_available`` is a boolean that depends on torch + CUDA state."""
    result = lib_common.gpu_available()
    assert isinstance(result, bool)


def test_reset_gpu_memory_stats_is_safe_without_cuda() -> None:
    """The reset call is a no-op when CUDA is missing or torch is absent."""
    lib_common.reset_gpu_memory_stats()  # must not raise


def test_peak_gpu_memory_mb_returns_float() -> None:
    """``peak_gpu_memory_mb`` is a float; ``0.0`` when CUDA is unavailable."""
    lib_common.reset_gpu_memory_stats()
    peak = lib_common.peak_gpu_memory_mb()
    assert isinstance(peak, float)
    assert peak >= 0.0


def test_sample_nvidia_smi_returns_dict_or_empty() -> None:
    """``sample_nvidia_smi`` returns either the three keys or an empty mapping."""
    sample = lib_common.sample_nvidia_smi()
    assert isinstance(sample, dict)
    if sample:
        assert set(sample) == {"util_pct", "mem_used_mb", "mem_total_mb"}


def test_env_fingerprint_contains_required_keys() -> None:
    """The fingerprint carries platform, python, pid and the library versions."""
    fingerprint = lib_common.env_fingerprint()
    for key in ("platform", "python", "pid", "numpy"):
        assert key in fingerprint


def test_write_csv_round_trip(tmp_path: Path) -> None:
    """``write_csv`` writes the requested header and rows in order."""
    target = tmp_path / "out.csv"
    rows = [
        {"a": "1", "b": "x"},
        {"a": "2", "b": "y"},
    ]
    lib_common.write_csv(target, ["a", "b"], rows)
    with target.open(encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    assert reader == rows