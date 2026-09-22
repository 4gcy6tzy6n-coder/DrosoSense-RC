#!/usr/bin/env python
"""Run the R0–R6 reservoir family over a dataset's specimen-level folds.

The protocol-compliant reservoir runner (DATA-50). It is the missing
"E2/E1 reservoir half" of the experiment matrix: it drives the frozen
connectome reservoir over the SAME specimen-level folds the E1 baselines use
(``make_folds(specimens, split_strategy, seed, n_splits)`` +
``build_fold_tensors``), so a reservoir record and an E1 record for the same
``(dataset, seed, fold)`` carry the same ``fold_fingerprint`` and the paired
statistical tests stay legitimate.

Every run trains ONLY the readout: the reservoir matrix ``A`` and the input
projection ``W_in`` are frozen (M3 semantics), and ``W_in`` / ``b`` are
shared identically across R0–R6 for one seed.

Records land in ``results/raw/<experiment>/<dataset>/<model>/
<task>_seed<NN>_fold<NN>.json`` with the same fields as an E1 record, and the
§17 semantics are: only ``status == "ok"`` counts as a touch, a failed run is
recorded with its reason and never dropped, and a unit that is already ok
under the same config is skipped (a ``status == "skipped"`` record is
written), not re-computed and not refused.

Examples
--------
Smoke on the synthetic fixture with a small NPZ (no 739 MB artifact needed)::

    python scripts/run_reservoir_e2.py --dataset synthetic_enose \\
        --experiment e2_smoke --seeds 0 --window-lengths 16 \\
        --max-folds 1 --reservoir-size 80

D2, one seed, all folds, R0–R6, both tasks — the acceptance shape::

    DROSOSENSE_DATA=/root/autodl-tmp/drososense/data \\
    python scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled \\
        --experiment e2_topology --seeds 0 --window-lengths 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    ALLOWED_NORMALIZATIONS,
    TOPOLOGY_FAMILY_IDS,
)
from drososense.reservoir.runner import (  # noqa: E402
    PINNED_KNOBS,
    PROTOCOL_ID_BY_FAMILY,
    ReservoirConfig,
    run_reservoir_benchmark,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. d2_beef_uncontrolled")
    parser.add_argument("--experiment", default="e2_topology", help="experiment label")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--window-lengths", nargs="+", type=int, default=[16])
    parser.add_argument(
        "--tasks", nargs="+", default=["classification", "regression"],
        choices=["classification", "regression"],
    )
    parser.add_argument(
        "--split-strategy",
        default="auto",
        choices=["auto", "group_kfold", "loso", "time_block_holdout"],
        help="'auto' reads split.strategy from the dataset config",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--label-rule", default="last", choices=["last", "majority"]
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help=(
            "E3 low-data: fraction of each fold's TRAIN-side specimen pool admitted. "
            "Test and validation specimens are untouched, so the per-fold TEST "
            "partition (and its fingerprint, which the E1-vs-E2 pairing keys on) is "
            "identical across 10/25/50/75/100%. Sampling is nested within each fold "
            "(for a fixed seed, the 10% pool is contained in the 25% pool, up to 100% "
            "which is byte-for-byte the full E2 train set). The pool is always a "
            "subset of the fold's own TRAIN side, so no fold is ever admitted its "
            "test/val specimen and no train side is emptied. Must satisfy 0 < f <= 1; "
            "omit for full-pool E2."
        ),
    )
    parser.add_argument(
        "--reservoir-size", type=int, default=None,
        help="N; the DATA-3 selection runs when N < the full graph",
    )
    parser.add_argument(
        "--normalization", default="n1_pre_l1", choices=list(ALLOWED_NORMALIZATIONS),
    )
    parser.add_argument(
        "--spectral-radius", type=float, default=PINNED_KNOBS.get("spectral_radius", 0.9)
        if "spectral_radius" in PINNED_KNOBS else 0.9,
        help="target spectral radius (must be a declared grid value)",
    )
    parser.add_argument(
        "--families", nargs="+", default=None,
        help=f"subset of {list(TOPOLOGY_FAMILY_IDS)}; the whole family when omitted",
    )
    parser.add_argument("--npz-path", default=None, help="explicit olfactory NPZ")
    parser.add_argument(
        "--select-hyperparameters", action="store_true",
        help="turn on the protocol §17 per-fold grid selection on the "
             "validation split (spectral scaling shared across R0–R6)",
    )
    parser.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true", default=True,
        help="skip units that are already ok under the same config (default)",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="re-run even when an ok record already exists (writes a new ok "
             "record; the same-config re-touch is allowed, a different one is "
             "still refused by §17)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="override for results/raw (records are written under "
             "<output-dir>/<experiment>/...)",
    )
    parser.add_argument(
        "--tables-dir", default=None,
        help="override for results/tables; mirrors --output-dir for the "
             "aggregated summary CSV (defaults to <output-dir>/tables when "
             "--output-dir is set)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    # DATA-61 defect 3: validate the BLAS threading knobs at start (the v6
    # hang was 5 procs × 128 threads, no limits set, nothing in the record
    # to evidence it). Unset knobs are pinned in-process so the record's
    # ENV_* keys state the limits the batch ran under.
    from ops import thread_limits

    try:
        thread_limits.validate_thread_limits(max_threads=8)
    except RuntimeError:
        thread_limits.set_thread_limits(thread_limits.DEFAULT_LIMIT)
        family_ids = TOPOLOGY_FAMILY_IDS
    else:
        # E3 (DATA-60): protocol shorthand aliases — R0/R1/…/R6 map to the
        # registry family ids (R0_real_fly, R1_weight_shuffled, …). The
        # smoke script's "--families R0 R2" shorthand is the protocol's own
        # id; without the alias the runner rejects unknown family ids and
        # the reservoir half silently runs zero families (exit 0).
        alias_by_protocol = {v: k for k, v in PROTOCOL_ID_BY_FAMILY.items()}
        family_ids_resolved = tuple(
            alias_by_protocol[f] if f in alias_by_protocol else f for f in args.families
        )
        unknown = [f for f in family_ids_resolved if f not in TOPOLOGY_FAMILY_IDS]
        if unknown:
            print(
                f"unknown families {unknown}; declared: {list(TOPOLOGY_FAMILY_IDS)}; "
                f"shorthand aliases: {list(PROTOCOL_ID_BY_FAMILY.values())}",
                file=sys.stderr,
            )
            return 2
        family_ids = family_ids_resolved

    if args.train_fraction is not None and not 0.0 < args.train_fraction <= 1.0:
        print(f"--train-fraction must satisfy 0 < f <= 1, got {args.train_fraction}", file=sys.stderr)
        return 2
    train_fraction: float | None = None
    if args.train_fraction is not None and args.train_fraction < 1.0:
        train_fraction = args.train_fraction

    config = ReservoirConfig(
        dataset_id=args.dataset,
        experiment=args.experiment,
        tasks=tuple(args.tasks),
        seeds=tuple(args.seeds),
        window_lengths=tuple(args.window_lengths),
        split_strategy=args.split_strategy,
        n_splits=args.n_splits,
        stride=args.stride,
        label_rule=args.label_rule,
        max_folds=args.max_folds,
        npz_path=Path(args.npz_path) if args.npz_path else None,
        reservoir_size=args.reservoir_size,
        normalization=args.normalization,
        spectral_radius=args.spectral_radius,
        family_ids=family_ids,
        select_hyperparameters=args.select_hyperparameters,
        train_fraction=train_fraction,
    )

    started = time.time()
    tables_dir = Path(args.tables_dir) if args.tables_dir else (
        Path(args.output_dir) / "tables" if args.output_dir else None
    )
    report = run_reservoir_benchmark(
        config,
        raw_dir=Path(args.output_dir) if args.output_dir else None,
        tables_dir=tables_dir,
    )
    wallclock = time.time() - started

    print(f"experiment: {config.experiment} | dataset: {config.dataset_id} | "
          f"families: {len(config.family_ids)} | seeds: {len(config.seeds)} | "
          f"wallclock: {wallclock:.1f}s")
    print(f"config_hash: {report.config_hash}")
    print(f"runs ok: {report.ok_count} | skipped: {len(report.skipped_units)} | "
          f"failed: {len(report.failed_units)}")
    for unit in report.skipped_units:
        print(f"  skipped (already ok): {unit}")
    for unit in report.failed_units:
        print(f"  FAILED: {unit}", file=sys.stderr)
    if report.summary_frame is not None and not report.summary_frame.empty:
        columns = [
            c for c in ("model", "task", "macro_f1_mean", "macro_f1_std",
                        "mae_mean", "mae_std", "rmse_mean", "n_runs")
            if c in report.summary_frame.columns
        ]
        print(report.summary_frame[columns].to_string(index=False))
        summary_path = (
            Path(args.tables_dir) if args.tables_dir
            else (Path(args.output_dir) / "tables" if args.output_dir else None)
        )
        print("\nraw records: results/raw/" + config.experiment + "/")
        print(f"recompute: python scripts/summarize.py --experiment {config.experiment}")

    if report.failed_units:
        print(f"\n{len(report.failed_units)} failed run(s) recorded with reasons; "
              "see the failure_reason field in results/raw/" + config.experiment + "/",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
