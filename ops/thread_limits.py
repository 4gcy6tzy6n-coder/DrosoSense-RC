"""DATA-61 defect 3 — BLAS/threading governance for parallel E3 batches.

The v6 E3 batch (``stage_d60/e3_d2_smoke.sh``, 2026-09-22 11:46:53Z) hung
for >26 min with zero output: 5 concurrent subprocesses, each spinning
128 BLAS threads on an 80-core box, with **no** ``OMP_NUM_THREADS`` /
``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` set in the environment —
thread oversubscription with no evidence left anywhere (defect 3).

This module makes the governance machine-enforceable at two levels:

* :func:`set_thread_limits` is the entry every launcher (server batch
  scripts, smoke gates, CI) should call or export from, so the limits
  land in the process environment **before** numpy/OpenBLAS initialize.
* :func:`validate_thread_limits` is the gate a launcher runs *at start*
  (and the test below pins): the limits must be set in this process and
  must hold a sane per-process ceiling (``1 <= limit <= max_threads``),
  because an unlimited per-process pool is exactly what oversubscribes
  under fan-out.

It is deliberately stdlib-only: it may be imported from a shell-driven
launcher before heavy dependencies are loaded.
"""

from __future__ import annotations

import os

#: The four knobs a BLAS/OpenMP fan-out must pin. Numexpr (used by
#: pandas ``.to_frame()`` selection paths) respects the fourth.
THREAD_LIMIT_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: The ceiling this project's launcher enforces per process. 80-core box,
#: 5-process E3 fan-out ⇒ 5 × 4 = 20 in-flight threads: no oversubscription,
#: no under-utilisation of any one core.
DEFAULT_LIMIT: int = 4


def set_thread_limits(limit: int = DEFAULT_LIMIT) -> dict[str, str]:
    """Pin every BLAS/threading knob in ``os.environ`` to ``limit``.

    Args:
        limit: Threads per library per process; must satisfy ``1 <= limit``.

    Returns:
        The mapping ``{var: str(limit)}`` that was set, so a launcher can
        log exactly what the batch runs under (the record-level evidence
        for defect 3 lives in ``capture_environment()``'s ``ENV_*`` keys).

    Raises:
        ValueError: If ``limit`` is not a positive integer.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"thread limit must be a positive integer, got {limit!r}")
    value = str(limit)
    for var in THREAD_LIMIT_VARS:
        os.environ[var] = value
    return {var: value for var in THREAD_LIMIT_VARS}


def _read_limit(var: str) -> int | None:
    """Parse one threading var from the environment.

    Returns:
        The integer limit, ``None`` when the var is unset or unparsable.
    """
    raw = os.environ.get(var)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def validate_thread_limits(max_threads: int | None = None) -> dict[str, int | None]:
    """Gate the start of a batch: every threading knob must be pinned.

    A batch that forgets to set the knobs is the v6 hang again — so the
    check is not advisory: a launcher is expected to call this **before**
    spawning workers, and fail fast when any knob is unset or above
    ``max_threads`` (the per-process ceiling the fan-out budget allows).

    Args:
        max_threads: Per-process ceiling; when given, any limit above it
            (or a value the kernel would silently clamp, i.e. <= 0) fails.
            ``None`` only requires the vars to be set at all.

    Returns:
        ``{var: limit}`` — one entry per knob, the parsed value.

    Raises:
        RuntimeError: When any knob is unset, unparsable, < 1, or
            (when ``max_threads`` is given) above the ceiling. The message
            names every offending var so the operator fixes them in one
            pass.
    """
    if max_threads is not None and max_threads < 1:
        raise ValueError(f"max_threads must satisfy 1 <= n, got {max_threads!r}")
    limits: dict[str, int | None] = {}
    offenders: list[str] = []
    for var in THREAD_LIMIT_VARS:
        limit = _read_limit(var)
        limits[var] = limit
        if limit is None:
            offenders.append(f"{var} (unset or unparsable)")
        elif max_threads is not None and limit > max_threads:
            offenders.append(f"{var}={limit} exceeds ceiling {max_threads}")
    if offenders:
        raise RuntimeError(
            "thread limits not pinned for this batch — the v6 oversubscription "
            f"condition is back: {offenders}. Export {', '.join(THREAD_LIMIT_VARS)}"
            f"=4 (see ops/thread_limits.py::set_thread_limits) before launching."
        )
    return limits


if __name__ == "__main__":  # pragma: no cover - operator CLI
    import sys

    set_thread_limits()
    print(validate_thread_limits(), file=sys.stderr)
