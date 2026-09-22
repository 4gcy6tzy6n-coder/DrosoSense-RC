"""Assemble the four-file M4 evidence bundles from raw run records.

Reads ``results/raw/<experiment>/**.json`` (or a local mirror tree passed via
``--raw-root``) and writes, per experiment:

* ``<experiment>_summary.csv``      (aggregate_records + write_summary_csv)
* ``<experiment>_per_run.csv``      (records_to_frame)
* ``<experiment>_fingerprints.csv`` (fingerprint_rows)
* ``<experiment>_test_touched_once.json`` (test_touched_once_report)

into ``--out`` (default ``results/tables``). Never edits the raw records.
This is the M4 re-aggregation step for the clobbered E2-D3 summary: it is the
same code path ``scripts/summarize.py`` uses, pointed at the raw records
only.

    python scripts/m4/assemble_bundles.py \
        --raw-root m4_pull/bundle/raw \
        --out results/tables \
        --experiment e2_main_d3 --experiment e2_main_d2 \
        --experiment e1_main_d2 --experiment e1_main_d3 --check

``--check`` verifies row counts / specimen counts against expected values and
exits non-zero on mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from drososense.evaluation.results import (
    aggregate_records,
    fingerprint_rows,
    load_records,
    records_to_frame,
    test_touched_once_report,
    write_summary_csv,
)

#: (experiment, expected per-run rows, expected distinct specimens)
EXPECTED = {
    "e1_main_d2": (900, 5),
    "e1_main_d3": (11160, 62),
    "e2_main_d2": (700, 5),
    "e2_main_d3": (8680, 62),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=None,
                        help="raw-record tree to read (default: results/raw)")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--check", action="store_true",
                        help="verify EXPECTED row/specimen counts, exit 1 on mismatch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_root = Path(args.raw_root) if args.raw_root else PROJECT_ROOT / "results" / "raw"
    records = load_records(raw_root)
    experiments = sorted({r.experiment for r in records})
    selected = experiments if not args.experiment else args.experiment
    missing = [e for e in selected if e not in experiments]
    if missing:
        print(f"no records for experiment(s) {missing}; available: {experiments}", file=sys.stderr)
        return 2

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for experiment in selected:
        subset = [r for r in records if r.experiment == experiment]
        summary = aggregate_records(subset)
        summary_path = write_summary_csv(summary, experiment, out_dir)
        per_run = records_to_frame(subset)
        per_run_path = out_dir / f"{experiment}_per_run.csv"
        per_run.to_csv(per_run_path, index=False)
        fingerprints = pd.DataFrame(fingerprint_rows(subset))
        fingerprints_path = out_dir / f"{experiment}_fingerprints.csv"
        fingerprints.to_csv(fingerprints_path, index=False)
        touched = test_touched_once_report(subset)
        touched_path = out_dir / f"{experiment}_test_touched_once.json"
        touched_path.write_text(json.dumps(touched, indent=2, sort_keys=True), encoding="utf-8")
        n_distinct_specimens = per_run["test_specimens_joined"].dropna().nunique()
        n_summary_rows = len(summary)
        print(
            f"{experiment}: n_runs={len(subset)} n_summary_rows={n_summary_rows} "
            f"n_distinct_specimens={n_distinct_specimens} n_violations={touched['n_violations']} "
            f"-> {summary_path.name} / {per_run_path.name} / "
            f"{fingerprints_path.name} / {touched_path.name}"
        )
        if args.check:
            expected = EXPECTED.get(experiment)
            if expected is None:
                failures.append(f"{experiment}: no expected count declared")
            else:
                exp_rows, exp_specimens = expected
                if len(subset) != exp_rows:
                    failures.append(f"{experiment}: n_runs={len(subset)} != {exp_rows}")
                # summary rows = n_models * n_tasks; for the four batches
                # (9 models, or 7 reservoir models) x 2 tasks that is 18/14.
                if n_summary_rows not in (14, 18):
                    failures.append(f"{experiment}: n_summary_rows={n_summary_rows} unexpected")
                if n_distinct_specimens != exp_specimens:
                    failures.append(f"{experiment}: n_distinct_specimens={n_distinct_specimens} != {exp_specimens}")
                # per (model, task) coverage
                ok = per_run["status"] == "ok"
                bad = per_run[~ok]
                if len(bad):
                    failures.append(f"{experiment}: {len(bad)} non-ok record(s): {bad['run_id'].tolist()[:5]}")
    if failures:
        for line in failures:
            print(f"CHECK FAIL: {line}", file=sys.stderr)
        return 1
    print("check: all experiments match EXPECTED counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
