#!/usr/bin/env python
"""Paired contrasts, multiplicity correction, and gate evaluation.

This is where the protocol stops being a document. It reads the raw run records,
builds the paired differences the protocol declares, runs the pre-registered test
with the pre-registered parameters, corrects inside the pre-registered families,
and evaluates each gate and narrative rule as a boolean expression.

Three properties are worth stating plainly, because they are the difference
between this and a script that prints p-values:

* The resampling unit is the **fold** (protocol v1.2 §8). Seeds are a
  stratification, never a resampling unit; ``PairedSpec`` refuses the latter.
* The DECISIVE p-value is the cluster-level exact sign test (protocol v1.2 §10),
  not the pair-level Wilcoxon. The pair-level test is still computed and
  published beside it, marked descriptive, because a reader must be able to see
  the difference between a pseudoreplicated p and a real one.
* A contrast with too few observations to test raises rather than reporting
  ``p = 1``, and a gate that references an unavailable contrast is reported as
  ``UNEVALUABLE`` rather than as failed.
* Every line carries ``n_pairs``, ``n_clusters`` and the effect size beside the
  p-value, so a reader can see how much independent evidence is behind it.
* The symbol table a gate resolves against is built from the protocol's own
  ``datasets``/``model_zoo``/``multiplicity`` declarations, never from whichever
  values happen to appear in this experiment's results. Deriving declared names
  from observed data is what made every gate report UNEVALUABLE in the first
  delivery (R0.1 review item C1).

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
    paired_test,
)
from drososense.utils.config import (  # noqa: E402
    load_protocol,
    protocol_condition_symbols,
    protocol_dataset_symbols,
    protocol_metric_properties,
    protocol_model_symbols,
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
            spec = protocol_paired_spec(protocol, metric, task)
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
                # `p_holm` is filled in below, inside the family; NaN here means
                # "not corrected" rather than "corrected to NaN".
                row["p_holm"] = float("nan")
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


def dataset_availability(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Report which declared datasets have their data on disk.

    Availability is read from the dataset's own manifest and its files, NOT from
    whether this experiment happened to produce a contrast for it. The first
    delivery derived it from the contrast table, which made ``unavailable(D1)``
    true for a dataset that was acquired and merely excluded for a different
    reason — and N5 would have fired, instructing a reader to re-acquire data
    that is already present.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of dataset id to a report with ``available``, ``status``,
        ``missing`` and ``corrupt``.
    """
    from drososense.data.manifest import load_manifest, manifest_path, verify_manifest
    from drososense.utils.config import protocol_dataset_entries

    report: dict[str, dict[str, Any]] = {}
    for short_name, entry in protocol_dataset_entries(protocol).items():
        dataset_id = str(entry["id"])
        path = manifest_path(dataset_id)
        if not path.is_file():
            report[dataset_id] = {
                "short_name": short_name,
                "available": False,
                "status": "no_manifest",
                "missing": [],
                "corrupt": [],
                "reason": f"no manifest at {path}",
            }
            continue
        manifest = load_manifest(path)
        verification = verify_manifest(manifest)
        available = bool(manifest.is_usable and verification["ok"])
        reason = ""
        if not manifest.is_usable:
            reason = f"manifest status is {manifest.status.value!r}"
        elif verification["missing"]:
            reason = f"{len(verification['missing'])} required file(s) not on disk"
        elif verification["corrupt"]:
            reason = f"{len(verification['corrupt'])} file(s) failed their recorded sha256"
        report[dataset_id] = {
            "short_name": short_name,
            "available": available,
            "status": manifest.status.value,
            "missing": list(verification["missing"]),
            "corrupt": list(verification["corrupt"]),
            "reason": reason,
        }
    return report


def model_parameter_counts(records: list[Any]) -> dict[str, int]:
    """Collect each model's trainable-parameter count from the run records.

    ``Gate_A`` ends in ``params(R0) < params(GRU)``, and the evaluator used to be
    handed an empty mapping — so even with every contrast present, Gate_A could
    never be evaluated. The count is a property of the MODEL, so it is read from
    every record available, including non-compliant ones: excluding a run for a
    non-compliant split must not also erase the fact that the model has that many
    parameters (review item C1, second occurrence — the geometry was wired, the
    numbers behind it were not).

    Args:
        records: Run records to scan.

    Returns:
        Mapping of model id to its recorded trainable-parameter count.
    """
    counts: dict[str, int] = {}
    for record in records:
        description = getattr(record, "model_description", None) or {}
        reported = description.get("n_trainable_parameters")
        if reported is None:
            continue
        counts.setdefault(str(record.model), int(reported))
    return counts


def evaluate_rules(
    table: pd.DataFrame,
    protocol: dict[str, Any],
    availability: dict[str, dict[str, Any]] | None = None,
    model_params: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate every gate and narrative rule against the contrast table.

    The symbol table is built from what the PROTOCOL declares — its datasets,
    model zoo, metrics and multiplicity conditions — not from what this
    experiment happens to contain. A name that is declared but has no result
    produces "no result for contrast"; a name that is not declared at all
    produces a namespace error. Those are different failures and the reviewer
    must be able to tell them apart (R0.1 review item C1).

    Args:
        table: The contrast table from :func:`build_contrast_table`.
        protocol: The parsed protocol.
        availability: Output of :func:`dataset_availability`; computed here when
            omitted.
        model_params: Trainable-parameter counts by model id, from
            :func:`model_parameter_counts`. Left empty, every ``params(...)``
            term is unevaluable and Gate_A can never pass.

    Returns:
        Mapping of rule id to ``{"expression", "result", "detail"}`` where
        ``result`` is ``True``, ``False`` or ``"UNEVALUABLE"``.
    """
    contrasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    # An empty table is a legitimate state — it means no declared contrast is
    # present in this experiment — and it must produce a stated UNEVALUABLE for
    # every rule rather than a KeyError on a column that no longer exists.
    ok_rows = table[table["status"] == "ok"] if "status" in table.columns else table.iloc[0:0]
    for _, row in ok_rows.iterrows():
        contrasts[(row["contrast_id"], row["metric"], row["dataset"], row["condition"])] = {
            "delta": float(row["delta"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
            "p_holm": float(row["p_holm"]),
            "n_pairs": int(row["n_pairs"]),
            "n_clusters": int(row["n_clusters"]),
            "n_clusters_nonzero": int(row["n_clusters_nonzero"]),
        }

    availability = availability if availability is not None else dataset_availability(protocol)
    available_datasets = sorted(
        dataset_id for dataset_id, entry in availability.items() if entry["available"]
    )

    metrics = sorted(protocol_metric_properties(protocol))
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params=model_params or {},
        symbols=build_symbols(
            protocol_model_symbols(protocol),
            metrics,
            [],
            protocol_condition_symbols(protocol),
            aliases=protocol_dataset_symbols(protocol),
        ),
        available_datasets=available_datasets,
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

    # Parameter counts come from every record, not only this experiment's: a
    # model's size does not depend on which split produced the run.
    model_params = model_parameter_counts(load_records())

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

    availability = dataset_availability(protocol)
    availability_path = RESULTS_TABLES_DIR / f"{args.experiment}_dataset_availability.json"
    availability_path.write_text(
        json.dumps(availability, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    rules = evaluate_rules(table, protocol, availability, model_params)
    rules_path = RESULTS_TABLES_DIR / f"{args.experiment}_gates.json"
    rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"{rules_path}")
    print(f"{availability_path}")

    for dataset_id, entry in sorted(availability.items()):
        state = "available" if entry["available"] else f"NOT available ({entry['reason']})"
        print(f"  {entry['short_name']} -> {dataset_id}: {state}")

    for rule_id, outcome in sorted(rules.items()):
        print(f"  {rule_id}: {outcome['result']}")

    if args.do_print:
        columns = [
            c
            for c in (
                "contrast_id", "classification", "metric", "dataset", "task", "n_pairs",
                "n_clusters", "n_clusters_nonzero", "n_nonzero", "delta", "delta_ci_low",
                "delta_ci_high", "effect_size", "effect_size_name", "test", "statistic",
                "p_value_reported", "p_holm", "paired_test",
                "p_paired_wilcoxon_reported", "equivalent", "tost_proxy_p",
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
