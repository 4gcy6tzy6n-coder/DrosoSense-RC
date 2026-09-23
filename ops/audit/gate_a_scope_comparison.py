#!/usr/bin/env python3
"""Gate A: pre-v1.5.1 vs post-v1.5.1 parameter evidence scope.

Runs the SAME contrast table through the SAME gate expression twice, changing only
how `params(...)` is resolved:

  PRE   the pre-v1.5.1 path: scan every record under results/raw/** and take the
        maximum n_trainable_parameters per model (`scripts.analyze.
        model_parameter_counts`, retained for exactly this comparison);
  POST  the v1.5.1 path: matched result rows inside the declared scope
        (`drososense.evaluation.parameter_scope.resolve_gate_parameter_scope`),
        which fails closed.

The contrast table is built once and shared, so any difference in the verdict is
attributable to the parameter path and nothing else.

Read-only: reads run records, evaluates expressions, writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drososense.evaluation.gates import (  # noqa: E402
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.evaluation.parameter_scope import (  # noqa: E402
    load_parameter_scope_declaration,
    resolve_gate_parameter_scope,
)
from drososense.evaluation.results import load_records, records_to_frame  # noqa: E402
from drososense.utils.config import (  # noqa: E402
    load_protocol,
    protocol_condition_symbols,
    protocol_dataset_symbols,
    protocol_metric_properties,
    protocol_model_symbols,
)
from drososense.utils.paths import PROTOCOL_PATH, RESULTS_RAW_DIR  # noqa: E402


def _evaluate(gate_id, protocol, table, model_params, provenance=None,
              counts_by_dataset=None, expression=None):
    """Evaluate an expression through the DELIVERED pipeline (analyze.evaluate_rules).

    Using the production entry point rather than a bespoke evaluator is the point
    of the comparison: it shows what the shipped analysis reports under each
    resolution, not what a hand-built evaluator would. `expression` overrides the
    gate's own text so a single term can be reported on its own, still evaluated
    by the frozen-expression engine.
    """
    from scripts.analyze import evaluate_rules

    # The aggregate shape the production pipeline passes, not a bare scope, so
    # the comparison exercises the same code path analyze.py does.
    audit = (
        {"declaration": "configs/protocol_v1.5.2.yaml", "scopes": {gate_id: provenance}}
        if provenance
        else None
    )
    if expression is not None:
        protocol = dict(protocol)
        protocol["gates"] = dict(protocol["gates"])
        protocol["gates"][gate_id] = {
            **protocol["gates"][gate_id], "expression": expression,
        }
    rules = evaluate_rules(
        table, protocol, None, model_params, audit,
        parameter_counts_by_dataset=counts_by_dataset,
    )
    entry = rules.get(gate_id, {})
    return {
        "result": entry.get("result"),
        "reason": entry.get("reason", ""),
        "detail": entry.get("detail", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=None,
                    help="restrict the frame to one experiment label (default: all)")
    args = ap.parse_args()

    protocol = load_protocol(PROTOCOL_PATH)
    records = load_records(RESULTS_RAW_DIR)
    print(f"records under results/raw: {len(records)}")
    per_exp: dict[str, int] = {}
    for r in records:
        per_exp[r.experiment] = per_exp.get(r.experiment, 0) + 1
    print("  by experiment:", dict(sorted(per_exp.items())))

    frame = records_to_frame(records)
    if args.experiment:
        frame = frame[frame["experiment"] == args.experiment]
    frame = frame[frame["status"] == "ok"]

    from scripts.analyze import build_contrast_table

    # The task -> primary-metric map comes from the protocol, exactly as
    # analyze.py's own entry point builds it.
    metric_by_task = {
        task: str(definition["metrics"]["primary"])
        for task, definition in protocol["tasks"].items()
        if isinstance(definition, dict) and "metrics" in definition
    }
    table = build_contrast_table(frame, protocol, metric_by_task, [("R0", "R4")])
    rows = table[["dataset", "contrast_id", "metric", "status"]].to_dict("records")
    print(f"\ncontrast table rows for Gate_A's performance contrast R0_vs_R4: {len(rows)}")
    for row in rows[:6]:
        print("   ", row)

    expression = protocol["gates"]["Gate_A"]["expression"]
    print(f"\nGate_A expression:\n  {' '.join(expression.split())}")

    # ---------------- PRE: the unscoped whole-tree scan ----------------
    from scripts.analyze import model_parameter_counts

    pre_params = model_parameter_counts(records, protocol)
    pre = _evaluate("Gate_A", protocol, table, pre_params)

    # ---------------- POST: the declared, scoped resolution ----------------
    declaration = load_parameter_scope_declaration()
    scope = resolve_gate_parameter_scope(
        "Gate_A", records, protocol=protocol, declaration=declaration
    )
    # v1.5.2: a scope conditioned on several datasets hands the evaluator counts
    # per dataset; an unevaluable scope hands it nothing at all.
    post_by_dataset = {"Gate_A": scope.counts_by_dataset} if scope.evaluable else {}
    post = _evaluate(
        "Gate_A", protocol, table, {}, scope.provenance(), post_by_dataset
    )

    # The two terms of the frozen expression, evaluated separately so the report
    # can say WHICH one decides the gate. The engine computes both; nothing here
    # is hand-written.
    a1_expression = (
        "sum([ci_contains_zero(R0, R4, macro_f1, D2) or noninferior(R0, R4, macro_f1, D2), "
        "ci_contains_zero(R0, R4, macro_f1, D3) or noninferior(R0, R4, macro_f1, D3)]) == 2"
    )
    a2_expression = "params(R0) < params(GRU)"
    a1 = _evaluate("Gate_A", protocol, table, {}, None, None, expression=a1_expression)
    a2_pre = _evaluate("Gate_A", protocol, table, pre_params, None, None, expression=a2_expression)
    a2_post = _evaluate(
        "Gate_A", protocol, table, {}, scope.provenance(), post_by_dataset,
        expression=a2_expression,
    )

    print("\n" + "=" * 78)
    print("PARAMETER VALUES")
    print("=" * 78)
    for model in ("R0", "GRU"):
        print(f"  {model:4s} prev1.5.1 (unscoped scan) = {pre_params.get(model)}")
        ev = scope.evidence.get(model)
        print(f"  {model:4s} post-v1.5.1 (scoped)      = "
              f"{ev.parameter_count if ev and ev.resolved else 'UNRESOLVED'}  "
              f"[{ev.status if ev else 'no term'}]")

    print("\n" + "=" * 78)
    print("GATE A TERMS  (the engine evaluates the frozen expression; nothing here is hand-written)")
    print("=" * 78)
    print(f"  A1  R0 not meaningfully worse than R4 on D2 and D3   = {a1['result']}")
    print(f"  A2  pre-v1.5.1   params(R0) < params(GRU)            = {a2_pre['result']}")
    print(f"  A2  post-v1.5.2  forall d in {{D2,D3}}: params(R0,d) < params(GRU,d) "
          f"= {a2_post['result']}")
    if scope.evaluable:
        for dataset, counts in sorted(scope.counts_by_dataset.items()):
            r0, gru = counts.get("R0"), counts.get("GRU")
            verdict = "PASS" if (r0 is not None and gru is not None and r0 < gru) else "not PASS"
            print(f"        {dataset}: params(R0)={r0} params(GRU)={gru}  -> {verdict}")

    print("\n" + "=" * 78)
    print("GATE A  (A1 and A2 combined by the frozen expression)")
    print("=" * 78)
    for label, out in (("pre-v1.5.1 ", pre), ("post-v1.5.2", post)):
        print(f"  {label:12s} result = {out['result']}")
        if out["reason"]:
            print(f"               reason = {out['reason'][:220]}")
    if pre["result"] != post["result"]:
        print("\n  -> THE VERDICT CHANGED between the two resolutions.")
    else:
        print("\n  -> same verdict; the difference is in the evidence behind it.")

    print("\n" + "=" * 78)
    print("POST-v1.5.1 PARAMETER PROVENANCE")
    print("=" * 78)
    print(json.dumps(scope.provenance(), indent=2, sort_keys=True)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
