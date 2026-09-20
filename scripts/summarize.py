#!/usr/bin/env python
"""Aggregate raw run records into summary tables and record the environment.

Raw records and summaries stay separate: this script reads
``results/raw/**/*.json`` and writes ``results/tables/*.csv``. It never edits a
raw record.

The summary carries ``evidence_class``, ``protocol_compliant`` and ``status``
through from the records, so a synthetic or non-compliant or failed number cannot
be averaged into a real one and cannot lose its label on the way to a table.

It also writes two side files that qualify every number in the tables:
``environment_report.json`` compares the declared environment with the local one,
and ``data_contact_log.json`` (written by the runner) records when a test split
was first evaluated.

Examples
--------
    python scripts/summarize.py --experiment smoke
    python scripts/summarize.py --all --print
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation.contact_log import load_contact_log  # noqa: E402
import pandas as pd  # noqa: E402

from drososense.evaluation.results import (  # noqa: E402
    aggregate_records,
    capture_environment,
    fingerprint_rows,
    load_records,
    records_to_frame,
    test_touched_once_report,
    write_summary_csv,
)
from drososense.utils.env_report import compare_environments  # noqa: E402
from drososense.utils.paths import RESULTS_TABLES_DIR, ensure_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="aggregate every experiment")
    parser.add_argument("--print", dest="do_print", action="store_true", help="print the summary")
    parser.add_argument(
        "--per-run", action="store_true",
        help="also write the tidy per-run frame, not just the aggregate",
    )
    parser.add_argument(
        "--fingerprints",
        action="store_true",
        help=(
            "also write the de-identified per-run fingerprint table and the "
            "test-touched-once audit. results/raw/** is gitignored, so without these the "
            "§17 fingerprint chain is auditable only from the machine that produced it"
        ),
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
    records = load_records()

    if not records:
        print("no run records under results/raw", file=sys.stderr)
        return 1

    experiments = sorted({r.experiment for r in records})
    selected = experiments if (args.all or not args.experiment) else args.experiment
    missing = [e for e in selected if e not in experiments]
    if missing:
        print(f"no records for experiment(s) {missing}; available: {experiments}", file=sys.stderr)
        return 2

    ensure_dir(RESULTS_TABLES_DIR)
    for experiment in selected:
        subset = [r for r in records if r.experiment == experiment]
        summary = aggregate_records(subset)
        if summary.empty:
            print(f"{experiment}: nothing to aggregate", file=sys.stderr)
            continue

        path = write_summary_csv(summary, experiment)
        print(f"{path}  ({len(summary)} rows from {len(subset)} runs)")

        if args.per_run:
            per_run_path = RESULTS_TABLES_DIR / f"{experiment}_per_run.csv"
            records_to_frame(subset).to_csv(per_run_path, index=False)
            print(f"{per_run_path}")

        if args.fingerprints:
            fingerprint_path = RESULTS_TABLES_DIR / f"{experiment}_fingerprints.csv"
            pd.DataFrame(fingerprint_rows(subset)).to_csv(fingerprint_path, index=False)
            print(f"{fingerprint_path}")
            audit = test_touched_once_report(subset)
            audit_path = RESULTS_TABLES_DIR / f"{experiment}_test_touched_once.json"
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            print(
                f"{audit_path}  ({audit['n_distinct_test_fingerprints']} distinct test "
                f"fingerprints, {audit['n_violations']} violation(s))"
            )

        if args.do_print:
            print(summary.to_string(index=False))
            print()

    environment_path = RESULTS_TABLES_DIR / "environment.json"
    environment_path.write_text(
        json.dumps(capture_environment(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"{environment_path}")

    # The comparison against environment.yml, so a number in the tables can be
    # attributed to the environment it was produced in. `capture_environment`
    # above records what IS installed; this records what the project DECLARED and
    # which parts of that are absent here.
    report = compare_environments()
    report_path = RESULTS_TABLES_DIR / "environment_report.json"
    report_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"{report_path}")
    print(f"  {report.summary()}")

    contact = load_contact_log()
    print(
        f"data contact: first test evaluation "
        f"{contact.first_test_evaluation_at or 'not started'}"
        + (f", datasets {list(contact.datasets_touched)}" if contact.datasets_touched else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
