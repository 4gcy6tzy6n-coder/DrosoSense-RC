#!/usr/bin/env python
"""M4/E1-E2 — drive the R0-R6 reservoir family through the official runner.

Protocol v1.3 §17 forbids re-scoring a (dataset, seed, fold, model, task)
unit under a changed configuration, and the runner is the ONLY driver that
honours that rule: it fingerprints every test evaluation, enforces
``enforce_test_touched_once`` and writes one auditable JSON record per
evaluation under ``results/raw``.

``scripts/run_reservoir.py`` (M3) predates that machinery and scores a
75/25 window split with no split protocol and no fingerprints, so its
records cannot support the E1/E2 contrasts.  This script therefore

* builds the seven families on the runner's own specimen-disjoint folds
  (LOSO for D2/D3, GroupKFold for D3 under v1.2),
* runs them through :func:`drososense.evaluation.runner.run_benchmark`
  via :mod:`drososense.evaluation.reservoir_bridge`, and
* then feeds the raw records to the pre-registered contrast machinery
  (:mod:`drososense.evaluation.e2_stats`) and the gate evaluator
  (:mod:`drososense.evaluation.gates`), which is how the protocol's own
  decisions get read from this experiment.

Usage
-----
.. code-block:: bash

    # one dataset, all families, protocol seed set, window 16, reservoir 250
    python scripts/run_reservoir_family.py --dataset d2_beef_uncontrolled \
        --window-lengths 16 --reservoir-size 250 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --experiment e2_d2

    # the E1/E2 matrix, several window lengths
    python scripts/run_reservoir_family.py --dataset d3_rainbow_trout \
        --window-lengths 16 32 --reservoir-size 250 --seeds 0 1 2 \
        --experiment e2_d3 --split-strategy loso --n-splits 62

The script refuses to start when the NPZ is not provisioned, when no fold
can be built, or when a §17 fingerprint collision is detected (the ok-only
guard in ``run_benchmark``); in that case it stops BEFORE writing any
record so the existing raw tree is left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation.reservoir_bridge import (  # noqa: E402
    FAMILY_MODEL_IDS,
    ReservoirFingerprintRegistry,
    make_family_constructor,
    run_family_benchmark,
)
from drososense.evaluation.runner import BenchmarkConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4/E1-E2 reservoir family run through the official runner."
    )
    parser.add_argument("--dataset", required=True,
                        help="dataset config id, e.g. d2_beef_uncontrolled")
    parser.add_argument("--experiment", default="e2_reservoir",
                        help="experiment label used in the raw-record paths")
    parser.add_argument("--window-lengths", nargs="+", type=int, default=[16],
                        help="window lengths to evaluate (protocol grid: 8 16 32 64)")
    parser.add_argument("--reservoir-size", type=int, default=250,
                        help="number of reservoir nodes N")
    parser.add_argument("--normalization", default="n0_raw",
                        help="R0 normalization (DATA-3 scheme)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="root seed set (protocol: 0..9, extension to 20)")
    parser.add_argument("--split-strategy", default="auto",
                        choices=["auto", "loso", "group_kfold"],
                        help="fold strategy; 'auto' reads the dataset config")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="splits for the strategy (62 for D3 LOSO under v1.3)")
    parser.add_argument("--tasks", nargs="+", default=["classification"],
                        choices=["classification", "regression"])
    parser.add_argument("--max-folds", type=int, default=None,
                        help="cap folds per seed (smoke runs only)")
    parser.add_argument("--npz-path", default=None,
                        help="explicit olfactory NPZ; derived otherwise")
    parser.add_argument("--raw-dir", default=None,
                        help="override results/raw (default: the repo tree)")
    parser.add_argument("--tables-dir", default=None,
                        help="override results/tables (default: the repo tree)")
    parser.add_argument("--skip-statistics", action="store_true",
                        help="write only the raw records, do not run e2_stats")
    parser.add_argument("--print-statistics", dest="do_print", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset_id=args.dataset,
        experiment=args.experiment,
        models=tuple(FAMILY_MODEL_IDS),
        tasks=tuple(args.tasks),
        seeds=tuple(args.seeds),
        window_lengths=tuple(args.window_lengths),
        split_strategy=args.split_strategy,
        n_splits=args.n_splits,
        max_folds=args.max_folds,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    # Build the family factory first: this fails fast on a missing NPZ or a
    # reservoir size the NPZ cannot support, BEFORE the runner touches any
    # test fold (so §17's "one touch" is not spent on a run that would abort).
    constructor = make_family_constructor(
        npz_path=args.npz_path,
        reservoir_size=args.reservoir_size,
        normalization=args.normalization,
    )

    registry = ReservoirFingerprintRegistry()
    summary = run_family_benchmark(
        config,
        family_constructor=constructor,
        fingerprints=registry,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        tables_dir=Path(args.tables_dir) if args.tables_dir else None,
    )

    out_dir = Path(args.tables_dir) if args.tables_dir else PROJECT_ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "models": list(FAMILY_MODEL_IDS),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "window_lengths": args.window_lengths,
        "reservoir_size": args.reservoir_size,
        "normalization": args.normalization,
        "family_meta": getattr(constructor, "family_meta", {}),
        "n_summary_rows": int(len(summary)),
        "skipped_models": list(summary.attrs.get("skipped_models", [])),
        "split_strategy": str(summary.attrs.get("split_strategy", config.split_strategy)),
        "note": (
            "One record per (model, task, seed, fold) evaluation under the runner; "
            "§17 ok-only fingerprints recorded; feed results/raw/"
            + args.experiment + "/ to drososense.evaluation.e2_stats."
        ),
    }
    side = out_dir / f"{args.experiment}_reservoir_run.json"
    side.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {side}")
    print(summary[["model", "task", "seed", "fold_id", "macro_f1", "mae"]].to_string(index=False))

    if args.skip_statistics:
        return 0

    # The pre-registered contrasts for whatever records this run produced.
    from drososense.data.loaders import dataset_config_path, load_dataset
    from drososense.evaluation.e2_stats import build_e2_table, index_runs, load_reservoir_records
    from drososense.evaluation.runner import load_records, RESULTS_RAW_DIR
    from drososense.utils.config import load_protocol

    protocol = load_protocol()
    records_dir = Path(args.raw_dir) if args.raw_dir else RESULTS_RAW_DIR
    run_csv = records_dir / args.experiment / f"{args.experiment}_per_run.csv"
    if run_csv.is_file():
        frame = index_runs(load_reservoir_records([run_csv]))
        table = build_e2_table(protocol, frame)
        stats_out = out_dir / f"{args.experiment}_e2_statistics.csv"
        table.to_csv(stats_out, index=False)
        audit = {
            "protocol_file": str(PROJECT_ROOT / "configs" / "protocol_v1.3.yaml"),
            "alpha": protocol["statistical_tests"]["alpha"],
            "bootstrap_b": protocol["statistical_tests"]["bootstrap"]["b"],
            "bootstrap_seed": protocol["statistical_tests"]["bootstrap"]["seed"],
            "primary_test": protocol["statistical_tests"]["primary_test"]["name"],
            "source_records": str(run_csv),
            "n_records": int(len(frame)),
        }
        (out_dir / f"{args.experiment}_e2_statistics.audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {stats_out} ({len(table)} rows)")
        if args.do_print:
            cols = [c for c in ("contrast_id", "family", "metric", "dataset", "task",
                                "n_pairs", "n_clusters", "delta", "delta_ci_low",
                                "delta_ci_high", "effect_size", "p_value", "p_holm",
                                "p_paired_wilcoxon", "minimum_achievable_p_over_clusters",
                                "status") if c in table.columns]
            print(table[cols].to_string(index=False))
    else:
        print(f"\nno per-run CSV at {run_csv}; skipped the pre-registered contrast step")

    # Gate evaluation: run_benchmark already wrote the aggregated summary
    # under results/tables via write_summary_csv.  Read it back and feed it
    # to the gate evaluator so Gate_A/B/C and the narrative rules get the
    # same verdicts as any other M4 run.
    from drososense.evaluation.gates import GateEvaluator
    from drososense.utils.config import (
        protocol_dataset_symbols,
        protocol_family_membership,
        protocol_metric_properties,
        protocol_model_symbols,
    )

    summary_csv = out_dir / f"{args.experiment}_summary.csv"
    if summary_csv.is_file():
        per_run = load_records(Path(args.raw_dir) if args.raw_dir else RESULTS_RAW_DIR)
        family_rows = [r for r in per_run if r.experiment == args.experiment]
        contrasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        if run_csv.is_file():
            frame2 = index_runs(load_reservoir_records([run_csv]))
            table2 = build_e2_table(protocol, frame2)
            for _, row in table2[table2["status"] == "ok"].iterrows():
                contrasts[(row["contrast_id"], row["metric"], row["dataset"],
                           str(row.get("condition", "full")))] = {
                    "delta": float(row["delta"]),
                    "ci_low": float(row["delta_ci_low"]),
                    "ci_high": float(row["delta_ci_high"]),
                    "p_holm": float(row.get("p_holm", float("nan"))),
                    "n_pairs": int(row["n_pairs"]),
                    "n_clusters": int(row["n_clusters"]),
                    "n_clusters_nonzero": int(row.get("n_clusters_nonzero", row["n_clusters"])),
                }
        evaluator = GateEvaluator(
            contrasts=contrasts,
            metrics=protocol_metric_properties(protocol),
            model_params={},
            symbols={
                **{str(s): str(s) for s in protocol_model_symbols(protocol)},
                **protocol_dataset_symbols(protocol),
            },
            available_datasets=[args.dataset],
        )
        gate_out: dict[str, dict[str, Any]] = {}
        for rule_id, definition in protocol["gates"].items():
            expression = definition.get("expression")
            if not expression:
                continue
            try:
                result = evaluator.evaluate(rule_id, str(expression))
                gate_out[rule_id] = {
                    "expression": result.expression,
                    "result": result.result,
                    "detail": result.detail,
                }
            except Exception as exc:  # noqa: BLE001 — a gate that cannot be
                # evaluated is a reported gap, not a silent False.
                gate_out[rule_id] = {
                    "expression": str(expression),
                    "result": "UNEVALUABLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        gates_path = out_dir / f"{args.experiment}_gates.json"
        gates_path.write_text(json.dumps(gate_out, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {gates_path}")
        for rule_id, verdict in gate_out.items():
            print(f"  {rule_id}: {verdict['result']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
