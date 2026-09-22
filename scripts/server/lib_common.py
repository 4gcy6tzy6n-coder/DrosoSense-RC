"""Shared helpers for the DATA-28 server harness.

The two scripts that drive the harness — :mod:`measure_baselines` and
:mod:`concurrency_sweep` — share three concerns: importing the M1 pipeline
without poisoning ``sys.path``, polling process RSS for peak-memory
measurement, and sampling GPU utilisation. Centralising them keeps the
measurement loop short and the per-script CLI uniform.

Why this exists
---------------
The harness runs on a remote GPU box where the only Python on PATH is
``/root/miniconda3/bin/python`` and the project is imported through a path
relative to the checkout. ``bootstrap_project_root`` makes that import safe
whether the script is invoked from the project root, from ``scripts/server/``,
or from a remote ``bash`` command that has been piped over SSH.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


def bootstrap_project_root(start: Path | None = None) -> Path:
    """Insert the project root onto ``sys.path`` and return it.

    The script can be invoked from the project root, from inside
    ``scripts/server/``, or from a CI runner that has only the script path.
    All three need the same behaviour: locate the directory that contains
    ``drososense/`` and ``scripts/`` and prepend it to ``sys.path``.

    Args:
        start: Directory to walk up from. Defaults to this file's parent.

    Returns:
        The resolved project root.

    Raises:
        RuntimeError: If no directory containing ``drososense/`` and ``scripts/``
            can be located.
    """
    cursor = (start or Path(__file__).resolve()).parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "drososense").is_dir() and (candidate / "scripts").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        f"could not locate a project root containing drososense/ and scripts/ "
        f"walking up from {cursor}"
    )


@dataclass(frozen=True)
class RssSample:
    """A single resident-memory observation in kilobytes.

    Attributes:
        rss_kb: Resident set size in kilobytes, as reported by
            ``/proc/self/status``.
        timestamp: Wall-clock seconds since the epoch, captured when the sample
            was taken. Used to compute sampling rate, not the metric.
    """

    rss_kb: int
    timestamp: float


def read_self_rss_kb() -> int:
    """Return the current process RSS in kilobytes.

    Returns:
        The ``VmRSS`` field of ``/proc/self/status`` as an integer, or ``0``
        when the kernel hides it (macOS, locked-down containers). Measurement
        code must treat ``0`` as "unavailable", never as "zero memory".
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except OSError:
        return 0
    return 0


@dataclass
class RssPoller:
    """Background RSS sampler that keeps the peak across a callback window.

    The poll runs in a daemon thread so the measurement loop is not slowed
    down by reading ``/proc/self/status`` between steps. ``stop()`` joins the
    thread and returns the peak sample observed while ``start()`` was active.

    Attributes:
        interval_s: Seconds between samples; ``0.05`` (50 ms) is a good
            compromise — fine enough to catch a transient spike, cheap enough
            not to perturb the measurement itself.
    """

    interval_s: float = 0.05

    def __post_init__(self) -> None:
        self._stop = False
        self._peak_kb = 0
        self._thread: _PollThread | None = None

    def __enter__(self) -> "RssPoller":
        """Start the polling thread and return ``self``.

        Returns:
            The poller, for ``with``-statement use.
        """
        import threading

        self._stop = False
        self._peak_kb = read_self_rss_kb()
        self._thread = _PollThread(self)
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop polling on scope exit.

        Args:
            exc_type: Exception class, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """
        self.stop()

    def _record(self, sample: int) -> None:
        """Update the peak RSS observation.

        Args:
            sample: Latest RSS in kilobytes.
        """
        if sample > self._peak_kb:
            self._peak_kb = sample

    def stop(self) -> int:
        """Stop polling and return the peak RSS observed.

        Returns:
            Peak RSS in kilobytes across the active window. ``0`` when RSS
            could not be read on this platform.
        """
        self._stop = True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self.interval_s * 4, 0.5))
        return self._peak_kb


class _PollThread:
    """Internal helper that owns the polling loop.

    A class instead of a ``threading.Thread`` subclass so :class:`RssPoller`
    stays a frozen dataclass and the threading detail is hidden.
    """

    def __init__(self, owner: RssPoller) -> None:
        self._owner = owner
        self._t: "object | None" = None  # populated in start()

    def start(self) -> None:
        import threading

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def join(self, timeout: float) -> None:
        if self._t is not None:
            self._t.join(timeout=timeout)

    def _run(self) -> None:
        interval = self._owner.interval_s
        while not self._owner._stop:
            sample = read_self_rss_kb()
            self._owner._record(sample)
            time.sleep(interval)


def gpu_available() -> bool:
    """Return ``True`` if torch was built with CUDA and sees at least one device.

    Returns:
        Whether a CUDA device is reachable from this Python process.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def reset_gpu_memory_stats() -> None:
    """Reset torch's per-device CUDA memory stats.

    Safe to call when CUDA is unavailable — it is a no-op in that case.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_gpu_memory_mb(device_index: int = 0) -> float:
    """Return torch's peak CUDA memory for a device, in megabytes.

    Args:
        device_index: CUDA device to query.

    Returns:
        Peak megabytes observed since the last ``reset_gpu_memory_stats``.
        ``0.0`` when CUDA is unavailable.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024))


def sample_nvidia_smi(device_index: int = 0) -> dict[str, str]:
    """Sample ``nvidia-smi`` for util and memory of one device.

    The sample is informational — the measurement record carries it so the
    report can answer "did this workload actually touch the GPU?". Returns an
    empty mapping when ``nvidia-smi`` is not on PATH (e.g. on the macOS
    developer machine).

    Args:
        device_index: CUDA device index to query.

    Returns:
        Mapping with ``util_pct``, ``mem_used_mb`` and ``mem_total_mb`` keys.
    """
    fields = ("utilization.gpu", "memory.used", "memory.total")
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    line = completed.stdout.strip().splitlines()
    if not line:
        return {}
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < len(fields):
        return {}
    return {
        "util_pct": parts[0],
        "mem_used_mb": parts[1],
        "mem_total_mb": parts[2],
    }


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Mapping[str, object]]) -> Path:
    """Write a CSV file with the given header and ordered row dicts.

    Args:
        path: Destination path; parents are created.
        header: Column names in the desired order.
        rows: Iterable of mappings whose keys cover the header.

    Returns:
        The path written.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(header)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def env_fingerprint() -> dict[str, str]:
    """Collect a small fingerprint of the current environment.

    Returns:
        Mapping with platform, python and the relevant library versions. The
        shape is deliberately narrow — broad fingerprints leak hostnames into
        result files.
    """
    import platform

    fingerprint: dict[str, str] = {
        "platform": platform.platform(terse=True),
        "python": platform.python_version(),
        "pid": str(os.getpid()),
    }
    for name in ("numpy", "scipy", "sklearn", "xgboost", "torch", "pandas"):
        try:
            module = __import__(name)
        except ImportError:
            fingerprint[name] = "missing"
            continue
        fingerprint[name] = getattr(module, "__version__", "unknown")
    return fingerprint