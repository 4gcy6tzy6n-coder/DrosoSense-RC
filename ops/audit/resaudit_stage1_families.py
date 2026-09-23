#!/usr/bin/env python
"""ResAudit-Food Stage 1 -- the real F1/F2/F3/F5 x Din{6,8} admissibility audit.

Authority: docs/resaudit_food_preregistration.md and amendments 1-6.

READ-ONLY with respect to every dataset: this script accepts no data path, imports no food
loader, and cannot see a label, a split or a task metric. It reads F1, which was
materialized and provenance-verified by ``resaudit_materialize_f1.py``
(status VERIFIED_FOR_STAGE1), generates F2/F3/F5 against it, and writes the Stage 1
admissibility table and the per-criterion JSON reports.

Two Din widths are audited, because Din varies by dataset (amendment 2 section 4):
FD1/FD3 use 6 and FD2 uses 8. Each width is audited in its OWN ``audit_all`` call so that
"F1"'s family_id still resolves as F2/F3/F5's declared ``counterfactual_parent`` -- the
generators hardcode that parent id. Running both widths in one call would either collide on
family_id or silently drop A2 to NOT_APPLICABLE, so the widths are kept separate and the
results are concatenated with a Din column.

F4 is reported as BLOCKED / NOT EVALUATED. Its varied factor is named (cell-type
connectivity) but the construction-feasibility gate does not pass, and amendment 1's design
commitment forbids substituting a stand-in for the sake of a complete table.

Usage:
    PYTHONPATH=. python3 ops/audit/resaudit_stage1_families.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resaudit.battery import (  # noqa: E402
    FROZEN_RHO_TARGET,
    audit_all,
    scale_to_spectral_radius,
    spectral_radius_of,
    stage1_table,
)
from resaudit.family import FamilySpec  # noqa: E402
from resaudit.families import generate_f2, generate_f3, generate_f5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
F1_DIR = REPO / "results/audit/resaudit_stage1/f1"

#: The two preregistered widths (amendment 2 section 4; amendment 6).
WIDTHS = (6, 8)

#: What FD1/FD3/FD2 map to, for the verdict's data-role column.
WIDTH_TO_DATASETS = {6: "FD1, FD3", 8: "FD2"}

#: F4's status, carried explicitly rather than dropped from the table.
F4_STATUS = {
    "family": "F4",
    "status": "BLOCKED / NOT EVALUATED",
    "varied_factor": "cell-type connectivity (named)",
    "reason": (
        "the construction-feasibility gate does not pass; no stand-in is substituted, "
        "because amendment 1's design commitment forbids it and the pre-registration "
        "already contained F4"
    ),
}


def load_f1():
    """Load the provenance-verified F1 wiring and its dataset-conditioned inputs."""
    meta_path = F1_DIR / "F1_materialization.json"
    if not meta_path.exists():
        raise SystemExit(
            f"F1 is not materialized ({meta_path} missing). Run "
            "ops/audit/resaudit_materialize_f1.py first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "VERIFIED_FOR_STAGE1":
        raise SystemExit(f"F1 status is {meta.get('status')!r}; Stage 1 requires VERIFIED_FOR_STAGE1")

    # The RAW induced adjacency -- the substrate's A. NOT F1_A.npz, which is a legacy
    # 0.9-targeted scoring copy. The audit has its own declared operating radius
    # (FROZEN_RHO_TARGET), and A3's verdict depends on it (amendment 1 section 1), so the
    # normalisation is applied HERE, once, by the battery's own declared-seed routine.
    raw_path = F1_DIR / "F1_A_raw.npz"
    if not raw_path.exists():
        raise SystemExit(
            f"{raw_path} missing: the audit must run on the raw substrate, not on a legacy "
            "scoring copy. Re-run ops/audit/resaudit_materialize_f1.py."
        )
    z = np.load(raw_path, allow_pickle=True)
    A_raw = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])
    )
    A = scale_to_spectral_radius(A_raw, FROZEN_RHO_TARGET)
    rho = spectral_radius_of(A)
    if abs(rho - FROZEN_RHO_TARGET) > 1e-9:
        raise SystemExit(
            f"normalisation failed: rho={rho!r} != FROZEN_RHO_TARGET={FROZEN_RHO_TARGET}"
        )
    print(f"A normalised to rho={rho:.12f} (FROZEN_RHO_TARGET, declared-seed routine)")
    Bs = {}
    for din in WIDTHS:
        bz = np.load(F1_DIR / f"F1_B_Din{din}.npz", allow_pickle=True)
        Bs[din] = np.asarray(bz["w_in"], dtype=np.float64)
    return meta, A, Bs


def f1_spec(A, B, din: int, meta: dict) -> FamilySpec:
    """F1's family spec. Construction record stays task-blind."""
    return FamilySpec(
        family_id="F1",
        label="Drosophila olfactory connectome (frozen S0)",
        role="negative_biological_control",
        A=sp.csr_matrix(A),
        B=np.asarray(B, dtype=np.float64),
        n_nodes=int(A.shape[0]),
        n_edges=int(A.nnz),
        din=int(din),
        counterfactual_parent=None,
        seed=int(meta["selection_settings"]["seed"]),
        construction={
            "kind": "frozen_substrate",
            "origin": "frozen S0 induced subgraph of the olfactory connectome",
            "selection_rule": meta.get("selection_rule_version"),
            "node_list_sha256": meta.get("node_list_sha256"),
            "edge_list_sha256": meta.get("edge_list_sha256"),
            "din": int(din),
            "allocation": meta.get("allocations", {}).get(str(din)),
        },
    )


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def crit(audit, name):
    return audit.results.get(name)


def row_for(audit, din: int, gen_report) -> dict:
    """One matrix row in the operator's declared field set."""
    fam = audit.family
    a1, a2, a3, a4, a5 = (crit(audit, c) for c in ("A1", "A2", "A3", "A4", "A5"))

    def cell(c):
        if c is None:
            return "", ""
        return ("" if c.value is None else c.value), c.state.value

    return {
        "family": fam["family_id"],
        "din": din,
        "seed": fam.get("seed"),
        "N": fam.get("n_nodes"),
        "M": fam.get("n_edges"),
        "input_mapping": fam.get("construction", {}).get("kind", ""),
        "rho_operating": round(float(FROZEN_RHO_TARGET), 6),
        "A1_state": a1.state.value if a1 else "",
        "A2_state": a2.state.value if a2 else "",
        "A3_value": cell(a3)[0],
        "A3_threshold": "" if a3 is None or a3.threshold is None else a3.threshold,
        "A3_pass": cell(a3)[1],
        "A4_value": cell(a4)[0],
        "A4_pass": cell(a4)[1],
        "A5_value": cell(a5)[0],
        "A5_pass": cell(a5)[1],
        "eligible_for_food": bool(audit.food_eligible),
        "construct_qualified": bool(audit.construct_qualified),
        "datasets": WIDTH_TO_DATASETS.get(din, ""),
        "wiring_hash": fam.get("wiring_hash", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO / "results/audit/resaudit_stage1"))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, A, Bs = load_f1()
    head = git_head()
    print(f"F1 status: {meta['status']}  | HEAD {head}")

    rows: list[dict] = []
    provenance: dict = {}
    a3_json: dict = {}
    a4_json: dict = {}
    a5_json: dict = {}
    tables: list[str] = []

    for din in WIDTHS:
        B = Bs[din]
        specs = [f1_spec(A, B, din, meta)]
        gen: dict = {}
        for fn, name, key in ((generate_f2, "F2", "F2"), (generate_f3, "F3", "F3"),
                              (generate_f5, "F5", "F5")):
            spec, report = fn(A, B)
            specs.append(spec)
            gen[name] = report.as_dict() if hasattr(report, "as_dict") else report
        print(f"\n--- Din={din}: auditing {[s.family_id for s in specs]} ---")
        audits = audit_all(specs)
        for audit in audits:
            rows.append(row_for(audit, din, gen.get(audit.family["family_id"])))
            key = f"{audit.family['family_id']}@Din{din}"
            provenance[key] = {
                "family": audit.family,
                "generation": gen.get(audit.family["family_id"]),
                "food_eligible": audit.food_eligible,
                "construct_qualified": audit.construct_qualified,
                "downstream_evaluated": audit.downstream_evaluated,
                "downstream_not_evaluated_reason": audit.downstream_not_evaluated_reason,
            }
            for name, store in (("A3", a3_json), ("A4", a4_json), ("A5", a5_json)):
                c = crit(audit, name)
                if c is not None:
                    store[key] = c.as_dict() if hasattr(c, "as_dict") else {
                        "state": c.state.value, "value": c.value,
                        "threshold": c.threshold, "expression": c.expression,
                        "detail": dict(c.detail),
                    }
        tables.append(stage1_table(audits))
        # A4 is descriptive only (amendment 2; BLOCKING_CRITERIA excludes it).
        for audit in audits:
            print(f"  {audit.family['family_id']}: food_eligible={audit.food_eligible} "
                  f"construct_qualified={audit.construct_qualified}")

    # ---- F4 stays in the table as an explicit blocked row ---------------------------
    rows.append({
        "family": "F4", "din": "", "seed": "", "N": "", "M": "", "input_mapping": "",
        "rho_operating": "",
        "A1_state": "", "A2_state": "", "A3_value": "", "A3_threshold": "", "A3_pass": "",
        "A4_value": "", "A4_pass": "", "A5_value": "", "A5_pass": "",
        "eligible_for_food": "", "construct_qualified": "",
        "datasets": "", "wiring_hash": "", "status": "BLOCKED / NOT EVALUATED",
    })

    fields = ["family", "din", "seed", "N", "M", "input_mapping", "rho_operating",
              "A1_state", "A2_state",
              "A3_value", "A3_threshold", "A3_pass", "A4_value", "A4_pass", "A5_value",
              "A5_pass", "eligible_for_food", "construct_qualified", "datasets",
              "wiring_hash", "status"]
    with (out_dir / "STAGE1_AUDIT_MATRIX.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    (out_dir / "FAMILY_PROVENANCE.json").write_text(json.dumps(
        {"git_head": head, "f1_status": meta["status"], "widths": list(WIDTHS),
         "width_to_datasets": WIDTH_TO_DATASETS, "f4": F4_STATUS,
         "families": provenance, "generated_utc": datetime.now(timezone.utc).isoformat()},
        indent=2), encoding="utf-8")
    (out_dir / "A3_KRYLOV.json").write_text(json.dumps(
        {"criterion": "A3", "expression": "D_eff([B, A*B, ..., A^16*B]) >= 2*Din",
         "caveat": "A3 PASS is not evidence of recurrence; a nilpotent chain passes it",
         "git_head": head, "cells": a3_json}, indent=2), encoding="utf-8")
    (out_dir / "A4_MEMORY.json").write_text(json.dumps(
        {"criterion": "A4",
         "status": "DESCRIPTIVE ONLY / RETIRED AS A QUALIFICATION GATE (amendment 2)",
         "note": "reported and explained, never blocking; BLOCKING_CRITERIA = A1,A2,A3,A5",
         "git_head": head, "cells": a4_json}, indent=2), encoding="utf-8")
    (out_dir / "A5_STATE_EXPANSION.json").write_text(json.dumps(
        {"criterion": "A5", "expression": "D_eff(state) >= 1.5*Din",
         "git_head": head, "cells": a5_json}, indent=2), encoding="utf-8")

    # ---- verdict document -----------------------------------------------------------
    def tbl(din):
        return tables[WIDTHS.index(din)]

    lines = [
        "# ResAudit-Food Stage 1 verdict — the family x dataset admissibility matrix",
        "",
        f"Status: generated {datetime.now(timezone.utc).isoformat()} at HEAD `{head}`. "
        "**No food data was read, no label was seen, and no task metric is computed anywhere "
        "in this stage.**",
        "",
        "## What was measured",
        "",
        f"F1's wiring is the provenance-verified frozen S0 substrate "
        f"(node_list_sha256 `{meta['node_list_sha256'][:16]}…`, edge_list_sha256 "
        f"`{meta['edge_list_sha256'][:16]}…`, N={meta['N']}, M={meta['M']}), status "
        f"`{meta['status']}`. F2/F3/F5 were generated against it at each width, and each "
        "width was audited in its own pass so the declared `counterfactual_parent='F1'` "
        "resolves.",
        "",
        "Two input widths are audited because `Din` varies by dataset (amendment 1 §4 of "
        "amendment 2; amendment 6): **Din=6 for FD1/FD3** and **Din=8 for FD2**. The input "
        "support is 473 nodes at both widths, so `Din` is the only thing that varies.",
        "",
    ]
    for din in WIDTHS:
        lines += [f"### Din = {din}  (used by {WIDTH_TO_DATASETS[din]})", "", tbl(din), ""]
    lines += [
        "## F4",
        "",
        f"**{F4_STATUS['status']}** — varied factor *{F4_STATUS['varied_factor']}*. "
        f"{F4_STATUS['reason']}",
        "",
        "F4 is kept in the matrix above as an explicit blocked row rather than deleted: the "
        "pre-registration contained F4, so a silent omission would misrepresent the design.",
        "",
        "## The two verdicts are separate and must stay separate",
        "",
        "```",
        "food_eligible        = A3 PASS                    (pre-registration section 6.1)",
        "construct_qualified  = A1 and A2 and A3 and A5    (BLOCKING_CRITERIA)",
        "```",
        "",
        "A4 is **not** in `BLOCKING_CRITERIA`: amendment 2 retired it to descriptive status "
        "because the observable does not track recurrence strength in the direction its own "
        "definition requires (more recurrent coupling scores *lower*). A4 values are reported "
        "in `A4_MEMORY.json` and in the matrix, and they never gate anything.",
        "",
        "A family can therefore be `food_eligible = True` while `construct_qualified = False` "
        "— it cleared the primary gate but failed an admission or downstream criterion. The "
        "two columns are emitted separately and must never be collapsed into one "
        "\"Stage 1 passed\" flag.",
        "",
        "## Caveats carried with these numbers",
        "",
        "- **A3 PASS is not evidence of recurrence.** Its name was narrowed to *finite-horizon "
        "input-reachable state diversity* because calibration falsified the broader reading: a "
        "feed-forward nilpotent chain passes it.",
        "- **F5's A3 status is a construction property, not a result.** F5 is built to satisfy "
        "A3, so \"F5 passes A3\" cannot be a finding.",
        "- **A3 FAIL does not predict poor task performance** in this design, and that claim is "
        "deliberately untestable here: A3-FAIL families are never run on food data.",
        "- **A4/A5 were measured on the frozen white-noise probe** (amendment 1 §0.1, "
        "amendment 3), whose spectral character differs from the autocorrelated food drives, "
        "so neither may be presented as a predictor of task performance.",
        "",
    ]
    (out_dir / "STAGE1_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote STAGE1_AUDIT_MATRIX.csv, FAMILY_PROVENANCE.json, A3_KRYLOV.json, "
          f"A4_MEMORY.json, A5_STATE_EXPANSION.json, STAGE1_VERDICT.md in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
