#!/usr/bin/env python
"""ResAudit-Food Stage 1 -- the A1-A5 audit of the declared families.

READ-ONLY with respect to every dataset: this script accepts no data path, imports no
loader, and cannot see a label, a split or a task metric. It reads family DEFINITIONS
and writes one JSON report plus the Stage 1 admissibility table.

Because the real F1-F5 substrates are not built yet (pre-registration Section 8, step 3
is "build the audit runner + F2/F3/F4 generators"), this entry point runs the battery on
the CALIBRATION families in ``resaudit.toys``. That is deliberate: it proves the gate
engine end to end, including the three-valued outcomes and the Stage 1 table, before any
real reservoir exists to be judged. Replace ``_families()`` with the real generators; the
audit path below does not change.

Usage:
    PYTHONPATH=. python ops/audit/resaudit_stage1.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resaudit.battery import audit_all, stage1_json, stage1_table  # noqa: E402
from resaudit.family import FamilySpec  # noqa: E402
from resaudit import toys  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "audit" / "resaudit_stage1" / "ResAudit_stage1.json"


def _spec(family_id: str, label: str, role: str, A, B, parent=None, construction=None):
    A = sp.csr_matrix(A)
    B = np.asarray(B, dtype=np.float64)
    return FamilySpec(
        family_id=family_id,
        label=label,
        role=role,
        A=A,
        B=B,
        n_nodes=A.shape[0],
        n_edges=int(A.nnz),
        din=B.shape[1],
        counterfactual_parent=parent,
        seed=20260923,
        construction=construction or {"kind": "calibration_toy"},
    )


def _families() -> list[FamilySpec]:
    """The calibration families. These are NOT F1-F5; they are the battery's own tests."""
    out: list[FamilySpec] = []

    # A calibrated substrate that clears BOTH A1 and A3 -- the qualification contrast.
    A, B, meta = toys.battery_ready_ring()
    out.append(_spec("C1", "ring+chords, rho-normalised", "calibration_qualified", A, B, construction=meta))

    # A substrate that clears A3 but fails A1 on mean out-degree: admission vs primary gate.
    A, B, meta = toys.two_coprime_cycles()
    out.append(_spec("C2", "coprime cycles", "calibration_a3_only", A, B, construction=meta))

    # An A3 failure: the hairline 2-cycle.
    A, B, meta = toys.hairline_fail_cycle()
    out.append(_spec("C3", "single 2-cycle (hairline)", "calibration_a3_fail", A, B, construction=meta))

    # An A3 failure that is loud rather than marginal.
    A, B, meta = toys.zero_input_graph()
    out.append(_spec("C4", "recurrent graph, zero input", "calibration_a3_fail_loud", A, B, construction=meta))

    # A3 passes despite being feed-forward -- the documented surprise.
    A, B, meta = toys.nilpotent_chain()
    out.append(_spec("C5", "nilpotent chain", "calibration_a3_surprise", A, B, construction=meta))

    # An A2-applicable pair: parent and its reversal permutation (degrees and weights
    # preserved, every edge remixed -> A2 PASS).
    A0, B0, meta0 = toys.k_out_ring(n=20, k=1)
    perm = (-np.arange(A0.shape[0])) % A0.shape[0]
    out.append(_spec("C6", "20-cycle (A2 parent)", "calibration_a2_parent", A0, B0, construction=meta0))
    out.append(
        _spec(
            "C7",
            "20-cycle, reversal permutation",
            "calibration_a2_child",
            sp.csr_matrix(A0[perm][:, perm]),
            B0,
            parent="C6",
            construction={"kind": "reversal_permutation", "parent": "C6"},
        )
    )
    return out


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-downstream", action="store_true",
                    help="skip A4/A5 even where A3 passes")
    args = ap.parse_args(argv)

    specs = _families()
    audits = audit_all(specs, measure_downstream=not args.no_downstream)
    report = stage1_json(audits, git_head=git_head(REPO))
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    report["note"] = (
        "Calibration families from resaudit.toys, not the real F1-F5 substrates. "
        "A3 on these toys is measured WITHOUT spectral-radius normalisation except where "
        "the builder normalises explicitly."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(stage1_table(audits))
    print()
    print(f"written: {args.out}")
    for a in audits:
        fam = a.family["family_id"]
        a3 = a.results["A3"]
        a4 = a.results.get("A4")
        # The two eligibility fields are printed side by side and NEVER merged into one
        # "Stage 1 passed" value: food_eligible is about the food stage, and
        # construct_qualified is about the audit.
        print(
            f"  {fam}: A3={a3.state.value:14s} "
            f"D_eff={a3.value!s:>9.9s} gate={a3.threshold} "
            f"A4={a4.state.value if a4 else '-':14s}(descriptive only) "
            f"food_eligible={a.food_eligible} "
            f"construct_qualified={a.construct_qualified}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
