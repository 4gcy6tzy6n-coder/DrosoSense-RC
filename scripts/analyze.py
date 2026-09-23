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
    protocol_family_membership,
    protocol_metric_properties,
    protocol_model_id_bindings,
    protocol_model_symbols,
    protocol_paired_spec,
)
from drososense.utils.paths import (  # noqa: E402
    MODEL_PARAMETERS_PATH,
    RESULTS_TABLES_DIR,
    ensure_dir,
)


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

    Declared contrasts are corrected inside the family the protocol enumerates —
    ``multiplicity.families``, keyed on ``(contrast, condition)`` and stratified
    by dataset and metric. Exploratory pairs are marked ``exploratory`` and are
    deliberately left uncorrected and unfit to support a gate.

    The family is read from the protocol and NOT re-derived from the rows. v1.2
    declared eight families and nothing read them: the correction ran inside a
    ``(metric, dataset)`` group, which put the primary comparison, the topology
    controls and the baseline comparison in one family and would have merged the
    five robustness families into a single 21-test family (review item N1). The
    membership rule is enforced by
    :func:`~drososense.utils.config.protocol_family_membership`, which raises if
    a pair is declared in two families — the protocol says the analysis stops
    rather than picks one.

    Args:
        frame: Per-run frame.
        protocol: The parsed protocol.
        metric_by_task: Task name to the metric the contrast is evaluated on.
        exploratory: Extra baseline pairs, labelled exploratory.

    Returns:
        One row per (contrast, metric, dataset, condition), with raw and corrected
        p-values, the owning family, the interval, the effect size and the counts
        behind them.

    Raises:
        ValueError: If the protocol's family declarations overlap.
    """
    # Fail on an overlapping family declaration before doing any of the work.
    protocol_family_membership(protocol)
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
                # `p_holm` is filled in below, inside the owning family; NaN here
                # means "not corrected" rather than "corrected to NaN".
                row["p_holm"] = float("nan")
                row["family"] = ""
                row["family_note"] = ""
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return apply_family_correction(table, protocol)


def apply_family_correction(
    table: pd.DataFrame,
    protocol: dict[str, Any],
    membership: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Assign every row its owning multiplicity family, then Holm-correct inside it.

    Kept separate from :func:`build_contrast_table` because the family rule is a
    property of the protocol and must be testable on a table that carries
    conditions the current pipeline does not yet build (the robustness families
    are all keyed on a condition other than ``full``).

    Three rules, all from ``multiplicity.family_rule``:

    * a declared ``(contrast, condition)`` pair is corrected once, inside the one
      family that enumerates it;
    * a pair in two families is a protocol error and the analysis stops —
      enforced by :func:`~drososense.utils.config.protocol_family_membership`,
      which raises before any correction is computed;
    * a pair in no family is reported without correction AND labelled
      exploratory.

    Holm is applied per family per ``(metric, dataset)``: the protocol says the
    topology family is "within one dataset and condition", and the co-primary
    metrics are "each evaluated inside its own multiplicity family at alpha =
    0.05 with no alpha split". Pooling datasets or metrics would correct one
    experiment's test against another's.

    Args:
        table: A contrast table with ``status``, ``classification``,
            ``contrast_id``, ``condition``, ``metric``, ``dataset`` and
            ``p_value`` columns.
        protocol: The parsed protocol.
        membership: Precomputed membership; computed here when omitted.

    Returns:
        The table with ``family``, ``family_note`` and ``p_holm`` filled in.
    """
    membership = membership if membership is not None else protocol_family_membership(protocol)

    # Assign each row to the family that owns its (contrast, condition) pair.
    ok = table["status"] == "ok"
    for index in table[ok].index:
        if table.at[index, "classification"] != "declared":
            continue
        pair = (table.at[index, "contrast_id"], table.at[index, "condition"])
        family = membership.get(pair)
        if family is None:
            table.at[index, "classification"] = "exploratory"
            table.at[index, "family_note"] = (
                f"({pair[0]}, {pair[1]}) is enumerated by no multiplicity family, so it is "
                f"reported without correction and labelled exploratory (protocol §12 "
                f"family_rule)"
            )
            continue
        table.at[index, "family"] = family

    corrected = table[ok & (table["family"] != "")]
    for _, group in corrected.groupby(["family", "metric", "dataset"], observed=True):
        table.loc[group.index, "p_holm"] = holm_correction(group["p_value"].tolist())
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


def load_committed_model_parameters(path: str | Path | None = None) -> dict[str, int]:
    """Read the committed model-parameter table.

    Args:
        path: Override for ``results/tables/model_parameters.json``.

    Returns:
        Mapping of model id to its decisive trainable-parameter count. Empty when
        the file does not exist, which is a legitimate state for a repository
        that has never run anything.
    """
    target = Path(path) if path is not None else MODEL_PARAMETERS_PATH
    if not target.is_file():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for model_id, entry in (payload.get("models") or {}).items():
        value = entry.get("n_trainable_parameters")
        if value is not None:
            counts[str(model_id)] = int(value)
        protocol_id = entry.get("protocol_id")
        if protocol_id and value is not None:
            counts[str(protocol_id)] = int(value)
    return counts


def model_parameter_counts(
    records: list[Any] | None = None,
    protocol: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, int]:
    """Collect each model's trainable-parameter count. **UNSCOPED — DO NOT USE FOR GATES.**

    .. deprecated:: protocol v1.5.1
       This scans every record under ``results/raw/**`` and takes the maximum
       ``n_trainable_parameters`` per model, so its answer depends on which
       unrelated experiments happen to be on disk: with the E9 size study present
       it reports ``params(R0) = 16004`` (the N=4000 readout) instead of the
       N=250 count the topology contrast used. That is the defect v1.5.1 fixes.
       Gate evaluation now reads :func:`drososense.evaluation.parameter_scope.
       resolve_gate_parameter_scope`, which resolves from matched result rows
       inside a declared scope and fails closed. This function is retained only
       because the audit and its regression tests need the old behaviour to
       compare against, and because ``model_parameter_audit`` reports it.


    ``Gate_A`` ends in ``params(R0) < params(GRU)``, and the evaluator used to be
    handed an empty mapping — so even with every contrast present, Gate_A could
    never be evaluated (review item C1, second occurrence: the geometry was
    wired, the numbers behind it were not).

    Two sources, because neither is sufficient:

    * ``results/tables/model_parameters.json`` — committed, so a clean clone can
      evaluate the gates at all. Without it ``params(...)`` resolves only on the
      machine that owns the git-ignored ``results/raw/**`` (review item N2).
    * the run records — measured, and the source the committed file is generated
      from.

    Records win when both are present: they are the direct measurement. Each
    source also declares the model under a protocol id (``esn`` is written ``R4``
    in every frozen expression, ``gru`` is written ``GRU``), and both spellings
    are registered so a gate resolves whichever id the runs carry.

    Args:
        records: Run records to scan; may be empty or ``None``.
        protocol: The parsed protocol, for the registry-id to protocol-id binding.
        path: Override for the committed table.

    Returns:
        Mapping of model id to its trainable-parameter count.
    """
    counts = dict(load_committed_model_parameters(path))
    measured: dict[str, int] = {}
    for record in records or []:
        description = getattr(record, "model_description", None) or {}
        reported = description.get("n_trainable_parameters")
        if reported is None:
            continue
        model = str(record.model)
        # `max` over the runs, matching the committed table's selection rule: the
        # largest configuration actually used represents the model.
        measured[model] = max(int(reported), measured.get(model, int(reported)))
    counts.update(measured)

    if protocol is not None:
        for record_id, protocol_id in protocol_model_id_bindings(protocol).items():
            if record_id in counts:
                counts.setdefault(protocol_id, counts[record_id])
            if protocol_id in counts:
                counts.setdefault(record_id, counts[protocol_id])
    return counts


def model_parameter_audit(
    counts: dict[str, int],
    records: list[Any] | None,
    protocol: dict[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Record which source supplied each parameter count, and whether they agree.

    A gate whose ``params(...)`` term resolved from a different source than the
    reviewer's is a silent difference between two runs of the same analysis, so
    the provenance is written out rather than assumed.

    Args:
        counts: The merged counts actually used.
        records: Run records the counts were merged from.
        protocol: The parsed protocol.
        path: Override for the committed table.

    Returns:
        Mapping with the committed path, the per-model value and source, and any
        model named by a ``params(...)`` term that no source could resolve.
    """
    committed = load_committed_model_parameters(path)
    bindings = protocol_model_id_bindings(protocol)
    entries: dict[str, Any] = {}
    for model_id in sorted(counts):
        if model_id in committed:
            source = "committed results/tables/model_parameters.json"
        else:
            source = "run records (this working copy)"
        entries[model_id] = {"n_trainable_parameters": counts[model_id], "source": source}
    unresolved = sorted(
        name for name in _params_term_models(protocol) if name not in counts
    )
    return {
        "committed_table": str(Path(path) if path is not None else MODEL_PARAMETERS_PATH),
        "n_committed_models": len(committed),
        "protocol_id_bindings": bindings,
        "models": entries,
        "unresolved_params_terms": unresolved,
        "unresolved_note": (
            "Each unresolved name is a model a gate names in params(...) that has no run "
            "record and no committed entry. On the delivered data this is R0..R5: the "
            "connectome reservoirs have never been run, so their parameter count does not "
            "exist yet. That is different from 'the data is git-ignored' and the reason "
            "string in the gate output now says which one it is."
        ),
    }


def _params_term_models(protocol: dict[str, Any]) -> list[str]:
    """Return every model name a gate passes to ``params(...)``.

    Args:
        protocol: The parsed protocol.

    Returns:
        Sorted distinct model names.
    """
    names: set[str] = set()
    expressions: list[str] = []
    for definition in protocol.get("gates", {}).values():
        if isinstance(definition, dict) and definition.get("expression"):
            expressions.append(str(definition["expression"]))
    for rule in protocol.get("narrative_adjustment_rules", {}).get("list", ()):
        expressions.append(str(rule.get("trigger_expression", "")))
    for expression in expressions:
        for chunk in expression.split("params(")[1:]:
            name = chunk.split(")")[0].strip().strip("'\"")
            if name:
                names.add(name)
    return sorted(names)


def evaluate_rules(
    table: pd.DataFrame,
    protocol: dict[str, Any],
    availability: dict[str, dict[str, Any]] | None = None,
    model_params: dict[str, int] | None = None,
    parameter_provenance: dict[str, Any] | None = None,
    parameter_counts_by_dataset: dict[str, dict[str, dict[str, int]]] | None = None,
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
        parameter_provenance=parameter_provenance,
        parameter_counts_by_dataset=parameter_counts_by_dataset,
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
    from drososense.evaluation.parameter_scope import (
        load_parameter_scope_declaration,
        resolve_gate_parameter_scope,
    )

    parameter_declaration = load_parameter_scope_declaration()

    # Protocol v1.5.1: a gate's parameter terms are read from MATCHED RESULT ROWS
    # inside the scope the amendment declares for that gate — never from a
    # whole-tree scan. The previous behaviour took the maximum
    # n_trainable_parameters over every record under results/raw, so an E9
    # size-study record (R0 at N=4000 -> 16004) settled Gate_A's parameter term
    # and the gate outcome depended on the presence of unrelated evidence.
    all_records = load_records()
    parameter_scopes = {
        gate_id: resolve_gate_parameter_scope(
            gate_id, all_records, protocol=protocol, declaration=parameter_declaration
        )
        for gate_id in protocol.get("gates", {})
    }
    # A gate is handed ONLY its own scope's counts. An unevaluable scope yields an
    # empty mapping, so `params(...)` raises with the scope's reason and the gate
    # is reported UNEVALUABLE rather than resolved from somewhere else.
    # Protocol v1.5.2: a gate whose evaluated_on spans several registered datasets
    # gets the counts DATASET BY DATASET, and the evaluator folds its parameter
    # comparison over the datasets with AND. The flat mapping is kept for gates
    # that are not dataset-conditioned, so a single-dataset scope behaves exactly
    # as v1.5.1 left it.
    parameter_counts_by_dataset = {
        gate_id: scope.counts_by_dataset
        for gate_id, scope in parameter_scopes.items()
        if scope.evaluable and len(scope.datasets) > 1
    }
    model_params = {
        model: count
        for scope in parameter_scopes.values()
        if scope.evaluable and len(scope.datasets) <= 1
        for model, count in (scope.counts_by_dataset[scope.datasets[0]].items()
                             if scope.datasets else [])
    }
    parameter_audit = {
        "declaration": "configs/protocol_v1.5.1.yaml",
        "scopes": {g: s.provenance() for g, s in sorted(parameter_scopes.items())},
        "unevaluable": sorted(g for g, s in parameter_scopes.items() if not s.evaluable),
        "note": (
            "params(model) is read from matched result rows inside the declared scope. "
            "No fallback, no cross-experiment substitution, no max/min/first/last "
            "selection. An unevaluable scope makes the gates that read it UNEVALUABLE."
        ),
    }

    records = [r for r in all_records if r.experiment == args.experiment]
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

    parameter_path = RESULTS_TABLES_DIR / f"{args.experiment}_model_parameters.json"
    parameter_path.write_text(
        json.dumps(parameter_audit, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    rules = evaluate_rules(
        table,
        protocol,
        availability,
        model_params,
        parameter_audit,
        parameter_counts_by_dataset=parameter_counts_by_dataset,
    )
    rules_path = RESULTS_TABLES_DIR / f"{args.experiment}_gates.json"
    rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"{rules_path}")
    print(f"{availability_path}")
    print(f"{parameter_path}")

    for dataset_id, entry in sorted(availability.items()):
        state = "available" if entry["available"] else f"NOT available ({entry['reason']})"
        print(f"  {entry['short_name']} -> {dataset_id}: {state}")

    print(
        f"  trainable parameters: {len(model_params)} name(s) resolved"
        + (
            f"; params(...) terms with no value: {parameter_audit['unresolved_params_terms']}"
            if parameter_audit["unresolved_params_terms"]
            else ""
        )
    )

    for rule_id, outcome in sorted(rules.items()):
        print(f"  {rule_id}: {outcome['result']}")

    if args.do_print:
        columns = [
            c
            for c in (
                "contrast_id", "classification", "family", "metric", "dataset", "task", "n_pairs",
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
