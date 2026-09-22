#!/usr/bin/env python
"""Build the M4 evidence audit table from the pipeline's output artefacts.

Maps every paper-claim candidate that the frozen protocol declares
(hypotheses, gates, narrative rules) to the evidence that supports or fails
to support it: the exact artefact files, row counts and the command that
reproduces the numbers. The table is a report, not a verdict — the
verdicts live in ``gates.json`` / ``narrative.json``, and this table
points at them cell by cell so a reviewer can trace every number.

Run after the pipeline:

    python scripts/evidence_audit.py --experiment e1_main_d2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.utils.config import load_protocol  # noqa: E402
from drososense.utils.paths import RESULTS_TABLES_DIR  # noqa: E402


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def build_audit(
    experiment: str, out_path: Path | None = None, tables_dir: str | None = None
) -> pd.DataFrame:
    """Assemble the evidence-audit rows for one bundle.

    Args:
        experiment: Bundle label whose pipeline output directory to read.
        out_path: Where to write the audit CSV (default
            ``results/tables/<experiment>/evidence_audit.csv``).
        tables_dir: Override the input tables directory.

    Returns:
        The audit frame.
    """
    protocol = load_protocol()
    base = Path(tables_dir) if tables_dir else RESULTS_TABLES_DIR
    out_dir = base / experiment
    gates_path = out_dir / "gates.json"
    narrative_path = out_dir / "narrative.json"
    audit_path = out_dir / "audit.json"
    contrast_path = out_dir / "contrast_statistics.csv"
    descriptive_path = out_dir / "descriptive.csv"

    rows: list[dict[str, Any]] = []
    add = lambda **kwargs: rows.append(kwargs)  # noqa: E731

    # --- hypothesis rows -------------------------------------------------
    primary = protocol.get("primary_hypothesis")
    if primary:
        add(
            claim_type="hypothesis",
            claim_id=str(primary.get("id")),
            claim_text=str(primary.get("statement", ""))[:200],
            contrast_id=str(primary.get("contrast", "")),
            metric=str(primary.get("metric", "")),
            status="PENDING (E2/E1-reservoir bundle)",
            supporting_evidence="",
            reproduction_command=(
                f"python -m drososense.evaluation.evidence_stats "
                f"--experiment e2_topology"
            ),
        )
    for hypothesis in protocol.get("secondary_hypotheses", []):
        add(
            claim_type="hypothesis",
            claim_id=str(hypothesis.get("id")),
            claim_text=str(hypothesis.get("statement", ""))[:200],
            contrast_id=str(hypothesis.get("contrast", "")),
            metric=str(hypothesis.get("metric", "")),
            status="PENDING (E2/E1-reservoir bundle)",
            supporting_evidence="",
            unresolved_terms="",
            reproduction_command=(
                f"python -m drososense.evaluation.evidence_stats "
                f"--experiment e2_topology"
            ),
        )

    # --- gate rows -------------------------------------------------------
    if gates_path.is_file():
        payload = json.loads(gates_path.read_text(encoding="utf-8"))
        for gate_id, record in (payload.get("gates") or {}).items():
            result = record.get("result")
            status = {
                True: "SUPPORTED",
                False: "NOT SUPPORTED",
                "UNEVALUABLE": "UNEVALUABLE",
            }.get(result, "UNKNOWN")
            add(
                claim_type="gate",
                claim_id=gate_id,
                claim_text=str(record.get("expression", ""))[:200],
                contrast_id="",
                metric="",
                status=status,
                supporting_evidence=str(record.get("reason", ""))[:200],
                reproduction_command=(
                    f"python -m drososense.evaluation.evidence_stats "
                    f"--experiment {experiment}"
                ),
            )

    # --- narrative-rule rows --------------------------------------------
    if narrative_path.is_file():
        payload = json.loads(narrative_path.read_text(encoding="utf-8"))
        for rule_id, record in (payload.get("rules") or {}).items():
            fired = bool(record.get("fired"))
            reason = str(record.get("reason", ""))
            if fired:
                status = "FIRED (action applies)"
            elif reason:
                status = "NOT FIRED — " + reason[:80]
            else:
                status = "NOT FIRED"
            add(
                claim_type="narrative_rule",
                claim_id=rule_id,
                claim_text=str(record.get("trigger", ""))[:200],
                contrast_id="",
                metric="",
                status=status,
                supporting_evidence=str(record.get("action", ""))[:200],
                unresolved_terms=str(record.get("reason", ""))[:200],
                reproduction_command=(
                    f"python -m drososense.evaluation.evidence_stats "
                    f"--experiment {experiment}"
                ),
            )

    # --- contrast-status rows (from the contrast table) ------------------
    if contrast_path.is_file():
        table = pd.read_csv(contrast_path)
        for _, row in table.iterrows():
            if str(row.get("status")) in ("unpairable", "insufficient_data"):
                add(
                    claim_type="contrast_status",
                    claim_id=str(row.get("contrast_id")),
                    claim_text=(
                        f"{row.get('contrast_id')}/{row.get('metric')}/"
                        f"{row.get('dataset')}/{row.get('task')}"
                    ),
                    contrast_id=str(row.get("contrast_id")),
                    metric=str(row.get("metric")),
                    status=str(row.get("status")),
                    supporting_evidence=str(row.get("note", ""))[:200],
                    unresolved_terms="",
                    reproduction_command=(
                        f"python -m drososense.evaluation.evidence_stats "
                        f"--experiment {experiment}"
                    ),
                )

    frame = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4 evidence audit table from pipeline artefacts."
    )
    parser.add_argument(
        "--experiment",
        default="e1_main_d2",
        help="bundle label whose pipeline output to read",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output CSV path (default results/tables/<experiment>/evidence_audit.csv)",
    )
    parser.add_argument(
        "--tables-dir",
        default=None,
        help="override the input tables directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = Path(args.tables_dir) if args.tables_dir else RESULTS_TABLES_DIR
    default_out = base / args.experiment / "evidence_audit.csv"
    out = Path(args.output) if args.output else default_out
    frame = build_audit(args.experiment, out_path=out, tables_dir=args.tables_dir)
    print(f"{out}  ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
