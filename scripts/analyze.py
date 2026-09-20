#!/usr/bin/env python
"""Paired contrasts, multiplicity correction, and gate evaluation.

This is where the protocol stops being a document. It reads the raw run records,
builds the paired differences the protocol declares, runs the pre-registered test
with the pre-registered parameters, corrects inside the pre-registered families,
and evaluates each gate and narrative rule as a boolean expression.

Three properties are worth stating plainly, because they are the difference
between this and a script that prints p-values:

* The resampling unit is the **fold** (protocol v1.1 §8). Seeds are a
  stratification, never a resampling unit; ``PairedSpec`` refuses the latter.
* A contrast with too few observations to test raises rather than reporting
  ``p = 1``, and a gate that references an unavailable contrast is reported as
  ``UNEVALUABLE`` rather than as failed.
* Every line carries ``n_pairs``, ``n_clusters`` and the effect size beside the
  p-value, so a reader can see how much independent evidence is behind it.

Examples
--------
    python scripts/analyze.py --experiment m1_benchmark
    python scripts/analyze.py --experiment m1_benchmark --print
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

import pandas as pd  # noqa: E402

from drososense.evaluation.gates import (  # noqa: E402
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.evaluation.results import load_records  # noqa: E402
from drososense.evaluation.stats import (  # noqa: E402
    InsufficientDataError,
    holm_correction,
    minimum_achievable_p,
    paired_test,
)
from drososense.utils.config import (  # noqa: E402
    load_protocol,
    protocol_metric_properties,
    protocol_paired_spec,
)
from drososense.utils.paths import RESULTS_TABLES_DIR, ensure_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="m1_benchmark")
    parser.add_argument(
        "--include-non-compliant",
        action="store_true",
        help=(
            "include runs flagged protocol_compliant: false. Off by default: protocol v1.1 "
            "excludes them from every gate, and a contrast built from them is exploratory"
        ),
    )
    parser.add_argument(
        "--exploratory",
        action="append",
        default=[],
        metavar="A,B",
        help=(
            "add an exploratory baseline pair, e.g. --exploratory esn,gru. Exploratory "
            "contrasts are written to the table marked as such, are NOT corrected inside any "
            "protocol family, and can never make a gate evaluate to true (v1.1 §9)"
        ),
    )
    parser.add_argument("--print", dest="do_print", action="store_true")
    return parser.parse_args(argv)


def observations(
    frame: pd.DataFrame, first: str, second: str, metric: str
) -> pd.DataFrame:
    """Build the paired differences for one contrast and metric.

    Both models must appear on the same (dataset, seed, fold) with the same
    window length — that pairing is what makes the difference a paired one.

    Args:
        frame: Per-run frame with one row per (dataset, model, task, seed, fold).
        first: The first model of the contrast.
        second: The second model.
        metric: Metric column to difference.

    Returns:
        Frame with columns ``dataset``, ``seed``, ``fold_id``, ``delta``.
    """
    key = ["dataset", "task", "seed", "fold_id", "window_length"]
    left = frame[frame["model"] == first][[*key, metric]].rename(columns={metric: "a"})
    right = frame[frame["model"] == second][[*key, metric]].rename(columns={metric: "b"})
    joined = left.merge(right, on=key, how="inner")
    joined = joined.dropna(subset=["a", "b"])
    joined["delta"] = joined["a"] - joined["b"]
    return joined


def declared_contrasts(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the protocol's contrast pairs as ``(first, second)``.

    Args:
        protocol: The parsed protocol.

    Returns:
        One tuple per declared contrast.
    """
    return [
        (str(c["first"]), str(c["second"])) for c in protocol["contrasts"]["list"]
    ]


def build_contrast_table(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    metric_by_task: dict[str, str],
    exploratory: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run every requested contrast on every dataset present, with Holm correction.

    Declared contrasts are corrected inside the family the protocol enumerates;
    exploratory pairs are marked ``exploratory`` and are deliberately left
    uncorrected and unfit to support a gate.

    Args:
        frame: Per-run frame.
        protocol: The parsed protocol.
        metric_by_task: Task name to the metric the contrast is evaluated on.
        exploratory: Extra baseline pairs, labelled exploratory.

    Returns:
        One row per (contrast, metric, dataset, condition), with raw and corrected
        p-values, the interval, the effect size and the counts behind them.
    """
    requested: list[tuple[str, str, str]] = [
        (first, second, "declared") for first, second in declared_contrasts(protocol)
    ]
    requested += [
        (first, second, "exploratory") for first, second in (exploratory or [])
    ]

    rows: list[dict[str, Any]] = []
    for first, second, kind in requested:
        contrast_id = f"{first}_vs_{second}"
        for task, metric in metric_by_task.items():
            if metric not in frame.columns:
                continue
            spec = protocol_paired_spec(protocol, metric)
            paired_all = observations(frame, first, second, metric)
            if paired_all.empty:
                continue
            for dataset in sorted(paired_all["dataset"].unique()):
                paired = paired_all[paired_all["dataset"] == dataset]
                try:
                    result = paired_test(
                        paired["delta"].to_numpy(),
                        paired["seed"].to_numpy(),
                        paired["fold_id"].to_numpy(),
                        spec,
                        contrast_id=contrast_id,
                        metric=metric,
                        dataset=dataset,
                        condition="full",
                    )
                except InsufficientDataError as exc:
                    rows.append(
                        {
                            "contrast_id": contrast_id,
                            "metric": metric,
                            "dataset": dataset,
                            "condition": "full",
                            "status": "insufficient_data",
                            "classification": kind,
                            "note": str(exc),
                        }
                    )
                    continue
                row = result.as_dict()
                row["task"] = task
                row["status"] = "ok"
                row["classification"] = kind
                row["p_holm"] = float("nan")
                row[
                    "minimum_achievable_p_over_clusters"
                ] = minimum_achievable_p(result.n_clusters)
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Holm is applied inside one family: the declared contrasts of a single
    # dataset and metric. Exploratory pairs are excluded — they are not part of
    # any pre-registered family, so a corrected p-value for them would imply a
    # pre-registration that does not exist.
    declared = table[(table["status"] == "ok") & (table["classification"] == "declared")]
    for (metric, dataset), group in declared.groupby(["metric", "dataset"], observed=True):
        adjusted = holm_correction(group["p_value"].tolist())
        table.loc[group.index, "p_holm"] = adjusted
    return table


def evaluate_rules(
    table: pd.DataFrame, protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Evaluate every gate and narrative rule against the contrast table.

    Args:
        table: The contrast table from :func:`build_contrast_table`.
        protocol: The parsed protocol.

    Returns:
        Mapping of rule id to ``{"expression", "result", "detail"}`` where
        ``result`` is ``True``, ``False`` or ``"UNEVALUABLE"``.
    """
    contrasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, row in table[table["status"] == "ok"].iterrows():
        contrasts[(row["contrast_id"], row["metric"], row["dataset"], row["condition"])] = {
            "delta": float(row["delta"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
            "p_holm": float(row["p_holm"]),
            "n_pairs": int(row["n_pairs"]),
        }

    zoo = protocol["model_zoo"]
    models = [
        entry["id"]
        for family in zoo.values()
        if isinstance(family, list)
        for entry in family
    ]
    models += [entry["first"] for entry in protocol["contrasts"]["list"]]
    models += [entry["second"] for entry in protocol["contrasts"]["list"]]
    metrics = sorted(protocol_metric_properties(protocol))
    datasets = sorted({key[2] for key in contrasts})
    conditions = sorted({key[3] for key in contrasts}) + [
        "dropout_p0.3",
        "noise_s0.1",
        "train10pct",
        "train25pct",
    ]

    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={},
        symbols=build_symbols(models, metrics, datasets, conditions),
        available_datasets=datasets,
    )

    out: dict[str, dict[str, Any]] = {}
    for rule_id, definition in protocol["gates"].items():
        if not isinstance(definition, dict) or "expression" not in definition:
            continue
        out[rule_id] = _safe_evaluate(evaluator, rule_id, definition["expression"])
    for rule in protocol["narrative_adjustment_rules"]["list"]:
        out[rule["id"]] = _safe_evaluate(evaluator, rule["id"], rule["trigger_expression"])
    return out


def _safe_evaluate(evaluator: GateEvaluator, rule_id: str, expression: str) -> dict[str, Any]:
    """Evaluate one rule, turning an unevaluable expression into a stated outcome.

    An expression that references a contrast with no result is NOT a failure of
    the rule; reporting it as ``False`` would read as "the gate did not pass",
    which is a different and much stronger statement.

    Args:
        evaluator: The gate evaluator.
        rule_id: Rule identifier.
        expression: The declared expression.

    Returns:
        Mapping with the result and, when applicable, the reason.
    """
    try:
        evaluation = evaluator.evaluate(rule_id, expression)
    except GateExpressionError as exc:
        return {
            "expression": " ".join(str(expression).split()),
            "result": "UNEVALUABLE",
            "reason": str(exc),
            "detail": {},
        }
    return {
        "expression": evaluation.expression,
        "result": evaluation.result,
        "reason": "",
        "detail": evaluation.detail,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    protocol = load_protocol()

    records = [r for r in load_records() if r.experiment == args.experiment]
    if not args.include_non_compliant:
        records = [r for r in records if r.protocol_compliant]
    if not records:
        print(
            f"no protocol-compliant records for experiment {args.experiment!r}; "
            f"pass --include-non-compliant to analyse them as exploratory",
            file=sys.stderr,
        )
        return 1

    from drososense.evaluation.results import records_to_frame

    frame = records_to_frame(records)
    frame = frame[frame["status"] == "ok"]

    metric_by_task = {
        "classification": protocol["tasks"]["classification"]["metrics"]["primary"],
        "regression": protocol["tasks"]["regression"]["metrics"]["primary"],
    }
    exploratory = []
    for pair in args.exploratory:
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) != 2 or not all(parts):
            print(f"--exploratory expects A,B; got {pair!r}", file=sys.stderr)
            return 2
        exploratory.append((parts[0], parts[1]))

    table = build_contrast_table(frame, protocol, metric_by_task, exploratory)
    ensure_dir(RESULTS_TABLES_DIR)

    if table.empty:
        print(
            "no contrast could be built from the available models.\n"
            "The protocol's declared contrasts are connectome contrasts (R0..R5), and this "
            "experiment contains only baselines — so there is genuinely nothing to test here, "
            "which is the correct answer rather than a missing one. Pass --exploratory A,B to "
            "compare two baselines, clearly labelled as exploratory.",
            file=sys.stderr,
        )
        return 1

    stats_path = RESULTS_TABLES_DIR / f"{args.experiment}_statistics.csv"
    table.to_csv(stats_path, index=False)
    print(f"{stats_path}  ({len(table)} rows)")

    rules = evaluate_rules(table, protocol)
    rules_path = RESULTS_TABLES_DIR / f"{args.experiment}_gates.json"
    rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"{rules_path}")

    for rule_id, outcome in sorted(rules.items()):
        print(f"  {rule_id}: {outcome['result']}")

    if args.do_print:
        columns = [
            c
            for c in (
                "contrast_id", "classification", "metric", "dataset", "task", "n_pairs",
                "n_clusters", "n_nonzero", "delta", "delta_ci_low", "delta_ci_high",
                "effect_size", "test", "p_value", "p_holm", "equivalent",
                "minimum_achievable_p_over_clusters",
            )
            if c in table.columns
        ]
        print(table[columns].to_string(index=False))

    considered = len([r for r in rules.values() if r["result"] is not False])
    if considered == 0:
        print(
            "\nNOTE: no gate could be evaluated — the connectome reservoirs (R0..R5) do not "
            "exist yet, so the topology contrasts have no results. This is reported as "
            "UNEVALUABLE, not as a failed gate.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
