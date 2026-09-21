#!/usr/bin/env python
"""Concurrency scaling sweep for the baseline measurement workload.

This driver answers the budget question directly: "if I dispatch N independent
measurement workers, what total throughput do I get, and at what point does
adding more workers stop paying for itself?". It does NOT measure scientific
metrics and it does NOT touch the test split — every worker runs the same
:mod:`measure_baselines` workload on a per-worker copy of the synthetic
fixture, then exits.

How the sweep works
-------------------
1. Pick a list of worker counts ``N`` (default ``[1, 4, 8, 16, 32]``).
2. For each ``N``, spawn ``N`` child processes via ``multiprocessing.Pool``.
3. Each child runs ``--runs-per-worker`` independent invocations of
   ``scripts/server/measure_baselines.py`` as a subprocess, on its own
   ``--output-dir`` (so they do not race on the CSV files), and reports back
   its wall-clock seconds.
4. The driver records ``wall_clock_s``, ``runs_per_minute`` (total throughput)
   and ``per_run_latency_s_mean`` (per-worker run latency) for each ``N``.

The sweet spot is the smallest ``N`` whose ``runs_per_minute`` has plateaued
— adding more workers beyond that point buys latency on the parent but no
extra measured throughput, which is exactly what the issue asks us to flag.

Notes
-----
- The harness shells out to ``measure_baselines.py`` rather than importing it,
  so a hung worker cannot deadlock the parent and a worker crash shows up as a
  ``nonzero_exit`` in the output.
- OMP / MKL threads are pinned per child to keep the comparison honest: if the
  parent process accidentally inherited an over-broad thread cap, every child
  would slow down equally and the curve would look flat.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

from scripts.server.lib_common import (  # noqa: E402  (sys.path mutation above)
    bootstrap_project_root,
    sample_nvidia_smi,
    write_csv,
)

bootstrap_project_root(Path(__file__).resolve().parent)


DEFAULT_WORKER_COUNTS: tuple[int, ...] = (1, 4, 8, 16, 32)


@dataclass(frozen=True)
class SweepRow:
    """One row of the concurrency sweep.

    Attributes:
        workers: Number of parallel workers used.
        runs_per_worker: Independent measurement runs each worker performed.
        total_runs: ``workers * runs_per_worker``.
        wall_clock_s: Wall-clock seconds for the pool to finish.
        runs_per_minute: ``total_runs / wall_clock_s * 60``.
        per_run_latency_s_mean: Mean per-worker wall-clock across children.
        per_run_latency_s_std: Standard deviation across children; ``0`` for ``N=1``.
        n_failures: Children that exited non-zero.
        torch_num_threads: torch thread count pinned in every child.
        model_subset: Models the children measured.
    """

    workers: int
    runs_per_worker: int
    total_runs: int
    wall_clock_s: float
    runs_per_minute: float
    per_run_latency_s_mean: float
    per_run_latency_s_std: float
    n_failures: int
    torch_num_threads: int
    model_subset: str


@dataclass(frozen=True)
class ChildResult:
    """What a single child process reports back to the parent.

    Attributes:
        worker_id: Zero-based worker index within the pool.
        latency_s: Wall-clock seconds this child spent, end-to-end.
        exit_code: ``0`` on success; non-zero propagates to ``n_failures``.
    """

    worker_id: int
    latency_s: float
    exit_code: int


def _run_child(args: tuple[int, int, Path, Path, list[str], int]) -> ChildResult:
    """Run one worker's measurement workload and report back.

    Args:
        args: ``(worker_id, runs_per_worker, output_dir, project_root,
        model_subset, torch_num_threads)``.

    Returns:
        A :class:`ChildResult`.
    """
    worker_id, runs_per_worker, output_dir, project_root, model_subset, torch_num_threads = args

    started = time.perf_counter()
    exit_code = 0
    for run_index in range(runs_per_worker):
        per_worker_dir = output_dir / f"worker{worker_id:02d}" / f"run{run_index:02d}"
        command = [
            sys.executable,
            str(project_root / "scripts" / "server" / "measure_baselines.py"),
            "--models",
            *model_subset,
            "--repetitions",
            "1",
            "--torch-num-threads",
            str(torch_num_threads),
            "--output-dir",
            str(per_worker_dir),
        ]
        env = {
            "OMP_NUM_THREADS": str(torch_num_threads),
            "MKL_NUM_THREADS": str(torch_num_threads),
            "OPENBLAS_NUM_THREADS": str(torch_num_threads),
            "NUMEXPR_NUM_THREADS": str(torch_num_threads),
            "PYTHONPATH": str(project_root),
        }
        completed = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            exit_code = completed.returncode
            print(
                f"worker {worker_id} run {run_index}: nonzero exit {completed.returncode}\n"
                f"stderr tail: {completed.stderr.splitlines()[-5:]}",
                file=sys.stderr,
            )
            break
    elapsed = time.perf_counter() - started
    return ChildResult(worker_id=worker_id, latency_s=elapsed, exit_code=exit_code)


def _summarise_child_latencies(results: Sequence[ChildResult]) -> tuple[float, float, int]:
    """Summarise latency across children.

    Args:
        results: Per-child results from the pool.

    Returns:
        ``(mean_seconds, std_seconds, n_failures)``.
    """
    successes = [r.latency_s for r in results if r.exit_code == 0]
    failures = sum(1 for r in results if r.exit_code != 0)
    if not successes:
        return 0.0, 0.0, failures
    mean_s = float(statistics.fmean(successes))
    std_s = float(statistics.pstdev(successes)) if len(successes) > 1 else 0.0
    return mean_s, std_s, failures


def _drive_one_pool(
    workers: int,
    runs_per_worker: int,
    model_subset: list[str],
    torch_num_threads: int,
    output_root: Path,
) -> SweepRow:
    """Run one ``N`` of the sweep.

    Args:
        workers: Pool size.
        runs_per_worker: Sequential runs each child performs.
        model_subset: Models the children will measure.
        torch_num_threads: torch thread count pinned per child.
        output_root: Where this pool writes per-child artefacts.

    Returns:
        A populated :class:`SweepRow`.
    """
    project_root = PROJECT_ROOT_HINT
    pool_dir = output_root / f"workers{workers:02d}"
    pool_dir.mkdir(parents=True, exist_ok=True)

    args_iterable = [
        (
            worker_id,
            runs_per_worker,
            pool_dir,
            project_root,
            model_subset,
            torch_num_threads,
        )
        for worker_id in range(workers)
    ]

    started = time.perf_counter()
    if workers == 1:
        results = [_run_child(args_iterable[0])]
    else:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_run_child, args_iterable)
    wall_clock = time.perf_counter() - started

    mean_s, std_s, failures = _summarise_child_latencies(results)
    total_runs = workers * runs_per_worker
    runs_per_minute = total_runs / max(wall_clock, 1e-9) * 60.0

    return SweepRow(
        workers=workers,
        runs_per_worker=runs_per_worker,
        total_runs=total_runs,
        wall_clock_s=float(wall_clock),
        runs_per_minute=float(runs_per_minute),
        per_run_latency_s_mean=mean_s,
        per_run_latency_s_std=std_s,
        n_failures=failures,
        torch_num_threads=torch_num_threads,
        model_subset=",".join(model_subset),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=list(DEFAULT_WORKER_COUNTS),
        help="worker counts to sweep; default = 1 4 8 16 32",
    )
    parser.add_argument(
        "--runs-per-worker",
        type=int,
        default=2,
        help="sequential runs each worker performs; >=2 gives per-worker latency stats",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svm_rbf", "random_forest", "gru", "esn"],
        help="model subset every worker measures; keep this list small so each run fits in seconds",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="torch thread count pinned in each child; 1 keeps the comparison fair",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/concurrency_sweep"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_sample = sample_nvidia_smi()
    rows: list[SweepRow] = []
    for workers in args.workers:
        print(f"-- workers={workers} --", file=sys.stderr)
        row = _drive_one_pool(
            workers=workers,
            runs_per_worker=args.runs_per_worker,
            model_subset=args.models,
            torch_num_threads=args.torch_num_threads,
            output_root=output_dir,
        )
        rows.append(row)
        print(
            f"   wall_clock_s={row.wall_clock_s:.2f} "
            f"runs_per_minute={row.runs_per_minute:.2f} "
            f"per_run_latency_s_mean={row.per_run_latency_s_mean:.2f} "
            f"n_failures={row.n_failures}",
            file=sys.stderr,
        )

    csv_path = output_dir / "concurrency_sweep.csv"
    columns = list(asdict(rows[0]).keys()) if rows else list(asdict(SweepRow(
        workers=0, runs_per_worker=0, total_runs=0, wall_clock_s=0.0,
        runs_per_minute=0.0, per_run_latency_s_mean=0.0, per_run_latency_s_std=0.0,
        n_failures=0, torch_num_threads=0, model_subset="",
    )).keys())
    write_csv(csv_path, columns, [asdict(r) for r in rows])

    env_path = output_dir / "concurrency_environment.json"
    env_path.write_text(json.dumps({
        "torch_num_threads": args.torch_num_threads,
        "workers": args.workers,
        "runs_per_worker": args.runs_per_worker,
        "models": args.models,
        "gpu_sample": gpu_sample,
    }, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {env_path}")
    return 0


if __name__ == "__main__":
    # The driver spawns subprocesses; using the default 'fork' start method on
    # Linux is fine, but 'spawn' is the safe default on macOS where fork-after-
    # thread can hang. Set it explicitly so the behaviour does not depend on the
    # platform.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    raise SystemExit(main())