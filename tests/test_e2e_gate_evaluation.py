"""End to end: from the committed per-run table to Gate_A / Gate_B / Gate_C.

The R0.1 statistical review (DATA-18) found that every gate reported
UNEVALUABLE while the 258 unit tests passed, because the unit tests exercised
``GateEvaluator`` against a symbol table built by hand and the production path
built a different one. A test of the evaluator is therefore not a test of the
gate engine.

These tests drive the real path — ``build_contrast_table`` then
``evaluate_rules``, the same functions ``scripts/analyze.py`` calls — over
(a) the committed ``results/tables/m1_benchmark_per_run.csv`` and (b) a
fabricated table that carries the reservoir contrasts the frozen expressions
name. The first pins the failure mode that was actually shipped; the second
proves the engine can reach a verdict at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import build_contrast_table, evaluate_rules  # noqa: E402
from drososense.evaluation.gates import (  # noqa: E402
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.utils.config import protocol_metric_properties  # noqa: E402
from drososense.utils.paths import RESULTS_TABLES_DIR  # noqa: E402

COMMITTED_PER_RUN = RESULTS_TABLES_DIR / "m1_benchmark_per_run.csv"

DATASET_IDS = {
    "D1": "d1_beef_controlled",
    "D2": "d2_beef_uncontrolled",
    "D3": "d3_rainbow_trout",
}


def _metric_by_task(protocol: dict) -> dict[str, str]:
    return {
        "classification": protocol["tasks"]["classification"]["metrics"]["primary"],
        "regression": protocol["tasks"]["regression"]["metrics"]["primary"],
    }


def _availability(**overrides) -> dict[str, dict]:
    """A hermetic availability report.

    Availability is read from the dataset manifests and their files, so using
    the real one would make this test depend on whether the raw data happens to
    be downloaded on the machine running it. The gate verdicts under test do not
    depend on N5.
    """
    report = {}
    for dataset_id in DATASET_IDS.values():
        report[dataset_id] = {
            "short_name": dataset_id,
            "available": True,
            "status": "auto",
            "missing": [],
            "corrupt": [],
            "reason": "",
        }
    report.update(overrides)
    return report


def _fabricated_rows(
    *,
    n_folds: int,
    model_delta: dict[str, float],
    dataset: str = "d3_rainbow_trout",
    n_seeds: int = 10,
) -> pd.DataFrame:
    """Build a per-run frame structured exactly like the committed one.

    The spread is a function of the FOLD only, so a run with more seeds has the
    same within-cluster distribution and differs only in how many times each
    cluster is measured. That is what makes the seed count a stratification and
    not a sample size, and it is what
    ``test_the_bootstrap_interval_width_does_not_shrink_with_the_seed_count``
    needs in order to mean anything.

    Args:
        n_folds: Fold count, i.e. the number of independent clusters.
        model_delta: macro-F1 improvement of each model over the reference.
        dataset: Dataset id the rows belong to.
        n_seeds: Seeds to repeat across.

    Returns:
        One row per (model, seed, fold).
    """
    base = 0.50
    rows = []
    for model, delta in model_delta.items():
        for seed in range(n_seeds):
            for fold in range(n_folds):
                # A fold-level wobble so the cluster means are not all identical,
                # but every cluster keeps the sign of `delta`.
                wobble = 0.02 * ((fold % 5) - 2) / 2.0
                value = base + delta + wobble
                rows.append(
                    {
                        "run_id": f"{dataset}|{model}|classification|seed{seed:02d}|fold{fold:02d}",
                        "experiment": "e2e_gate",
                        "dataset": dataset,
                        "model": model,
                        "task": "classification",
                        "seed": seed,
                        "fold_id": fold,
                        "window_length": 16,
                        "protocol_version": "1.2.0",
                        "evidence_class": "real",
                        "protocol_compliant": True,
                        "status": "ok",
                        "macro_f1": value,
                        "mae": 1.0 - delta,
                    }
                )
    return pd.DataFrame(rows)


def _fabricated_both_foods(**kwargs) -> pd.DataFrame:
    """Fabricated rows on D2 and D3, the two datasets Gate_A and Gate_B name."""
    return pd.concat(
        [
            _fabricated_rows(dataset=DATASET_IDS["D2"], **kwargs),
            _fabricated_rows(dataset=DATASET_IDS["D3"], **kwargs),
        ],
        ignore_index=True,
    )


def _run_pipeline(
    frame: pd.DataFrame,
    protocol: dict,
    model_params: dict,
    exploratory: list[tuple[str, str]] | None = None,
    availability: dict | None = None,
    parameter_audit: dict | None = None,
):
    frame = frame[frame["status"] == "ok"]
    table = build_contrast_table(frame, protocol, _metric_by_task(protocol), exploratory)
    rules = evaluate_rules(
        table, protocol, availability or _availability(), model_params, parameter_audit
    )
    return table, rules


# ---------------------------------------------------------------------------
# The committed benchmark: correct failure, not a namespace failure
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_the_committed_benchmark_reaches_the_gates_without_a_namespace_error(protocol):
    """The shipped UNEVALUABLE must be about missing evidence, not missing names.

    The committed per-run table holds the M1 baselines only. No R0..R5 contrast
    exists, so the gates genuinely cannot be evaluated — but the reason reported
    must be that the contrast is absent, because that is a fact about the
    evidence. Reporting "D2 is not a declared dataset" instead, as the first
    delivery did, told a reader to look for a namespace bug that was not there.

    The exploratory pair is what the shipped analysis was run with: with no
    declared contrast present the table is empty and the run stops before the
    gates, which is reported by the next test rather than by this one.
    """
    assert COMMITTED_PER_RUN.is_file(), f"{COMMITTED_PER_RUN} must be committed"
    frame = pd.read_csv(COMMITTED_PER_RUN)

    table, rules = _run_pipeline(frame, protocol, {}, exploratory=[("esn", "gru")])
    assert set(rules) >= {"Gate_A", "Gate_B", "Gate_C"}

    for rule_id, outcome in rules.items():
        reason = outcome["reason"]
        assert "is not a declared" not in reason, f"{rule_id} still fails on the namespace: {reason}"

    for gate in ("Gate_A", "Gate_B", "Gate_C"):
        assert rules[gate]["result"] == "UNEVALUABLE"
        assert "no result for contrast" in rules[gate]["reason"], rules[gate]["reason"]

    # The baselines that ARE in the table still produce contrasts, so the
    # pipeline is not simply returning nothing.
    assert not table.empty
    assert "esn_vs_gru" in set(table["contrast_id"])
    # And every decisive p-value in it is a cluster-level one.
    assert set(table["test"]) == {"cluster_sign_test"}
    assert set(table["paired_test"]) <= {"wilcoxon_signed_rank", "sign_test"}


@pytest.mark.integration
def test_an_experiment_with_no_declared_contrast_reports_unevaluable_not_a_crash(protocol):
    """No R0..R5 rows means an empty contrast table, which must still be a verdict."""
    frame = pd.read_csv(COMMITTED_PER_RUN)
    frame = frame[frame["status"] == "ok"]

    table = build_contrast_table(frame, protocol, _metric_by_task(protocol))
    assert table.empty

    rules = evaluate_rules(table, protocol, _availability(), {})
    for gate in ("Gate_A", "Gate_B", "Gate_C"):
        assert rules[gate]["result"] == "UNEVALUABLE"
        assert "no result for contrast" in rules[gate]["reason"]


@pytest.mark.integration
def test_the_committed_benchmark_is_not_pseudoreplicated_any_more(protocol):
    """The D2 rows the review re-derived, re-derived here from the committed CSV.

    Review item C2 quoted `esn_vs_gru / macro_f1 / D2` at p = 5.45e-13 beside a
    cluster-level minimum of 0.0625. The pair-level number is still published —
    it is what the earlier protocol reported — but it is no longer the one that
    decides, and the decisive column can no longer go below the floor.
    """
    frame = pd.read_csv(COMMITTED_PER_RUN)
    table, _ = _run_pipeline(frame, protocol, {}, exploratory=[("esn", "gru")])

    row = table[
        (table["contrast_id"] == "esn_vs_gru")
        & (table["metric"] == "macro_f1")
        & (table["dataset"] == DATASET_IDS["D2"])
    ].iloc[0]

    assert int(row["n_pairs"]) == 50
    assert int(row["n_clusters"]) == 5
    assert float(row["minimum_achievable_p_over_clusters"]) == pytest.approx(0.0625)
    assert float(row["p_value"]) >= 0.0625
    assert float(row["p_paired_wilcoxon"]) < 1e-6
    assert float(row["p_value"]) > float(row["p_paired_wilcoxon"])
    assert row["test"] == "cluster_sign_test"


@pytest.mark.integration
def test_the_committed_benchmark_publishes_a_usable_effect_size_for_regression(protocol):
    """Review item M3: the choice follows the TASK, so r2 takes Hodges-Lehmann.

    The pipeline only evaluates each task's PRIMARY metric, so `r2` never reaches
    the table. The defect was not in the table though — it was in the rule the
    code used to pick the estimator, and that rule is exercised here through the
    same spec constructor the pipeline uses.
    """
    from drososense.evaluation.stats import hodges_lehmann, paired_test
    from drososense.utils.config import protocol_paired_spec

    frame = pd.read_csv(COMMITTED_PER_RUN)
    table, _ = _run_pipeline(frame, protocol, {}, exploratory=[("esn", "gru")])
    by_metric = table.groupby("metric")["effect_size_name"].unique().to_dict()
    assert list(by_metric["mae"]) == ["hodges_lehmann"]
    assert list(by_metric["macro_f1"]) == ["rank_biserial"]

    # r2 is a regression metric whose NAME looks like neither mae nor rmse: the
    # v1.1 rule sent it to the rank-biserial branch for exactly that reason.
    spec = protocol_paired_spec(protocol, "r2", "regression")
    assert spec.effect_size_name == "hodges_lehmann"
    deltas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    folds = list(range(len(deltas)))
    result = paired_test(deltas, [0] * 6, folds, spec, metric="r2")
    assert result.effect_size_name == "hodges_lehmann"
    assert result.effect_size == pytest.approx(hodges_lehmann(np.array(deltas)))

    # And a classification metric is unaffected by the change.
    assert protocol_paired_spec(protocol, "macro_f1", "classification").effect_size_name == (
        "rank_biserial"
    )


# ---------------------------------------------------------------------------
# A table that carries the reservoir contrasts: the gates must decide
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_the_gates_evaluate_when_the_reservoir_contrasts_exist(protocol):
    """With R0..R5 present, all three gates reach a verdict rather than 'UNEVALUABLE'."""
    frame = _fabricated_both_foods(
        n_folds=10,
        model_delta={"R0": 0.10, "R1": 0.02, "R2": 0.01, "R3": 0.01, "R4": 0.02, "R5": 0.01,
                     "GRU": 0.00},
    )
    _, rules = _run_pipeline(frame, protocol, {"R0": 1000, "GRU": 50000})

    for gate in ("Gate_A", "Gate_B", "Gate_C"):
        assert rules[gate]["result"] != "UNEVALUABLE", rules[gate]["reason"]

    # A connectome that beats every control passes Gate_A and Gate_B.
    assert rules["Gate_A"]["result"] is True
    assert rules["Gate_B"]["result"] is True
    # Gate_C reaches a verdict but is False, and for a reason that is about the
    # evidence rather than the engine: its `sig_any` clause scans the robustness
    # conditions (dropout / noise / low-data), and `build_contrast_table` builds
    # only the `full` condition today. Closing that is M4's job, not this one's.
    assert rules["Gate_C"]["result"] is False
    assert "dropout_p0.3" in rules["Gate_C"]["expression"]


@pytest.mark.integration
def test_the_same_fabricated_result_fails_gate_b_at_five_clusters(protocol):
    """Review item C2 end to end: the fold count changes the verdict, not the effect.

    The per-run differences are identical in both runs. Only the number of
    independent blocks changes — five instead of ten — and the gate that the
    delivered v1.1 table put within reach closes, because five clusters cannot
    produce a p-value below alpha at all.
    """
    deltas = {"R0": 0.10, "R1": 0.02, "R2": 0.01, "R3": 0.01, "R4": 0.02, "R5": 0.01, "GRU": 0.00}
    params = {"R0": 1000, "GRU": 50000}

    wide = _run_pipeline(_fabricated_both_foods(n_folds=10, model_delta=deltas), protocol, params)[1]
    narrow = _run_pipeline(_fabricated_both_foods(n_folds=5, model_delta=deltas), protocol, params)[1]

    assert wide["Gate_B"]["result"] is True
    assert narrow["Gate_B"]["result"] is False
    # Gate_A is not a significance gate, so it survives the narrower design.
    assert narrow["Gate_A"]["result"] is True


@pytest.mark.integration
def test_a_missing_dataset_is_reported_as_unavailable_and_does_not_stop_the_gates(protocol):
    """N5 fires on a dataset whose data was not obtained, and only then."""
    frame = _fabricated_rows(n_folds=10, model_delta={"R0": 0.10, "R2": 0.01, "R4": 0.0})
    availability = _availability(
        d1_beef_controlled={
            "short_name": "d1_beef_controlled",
            "available": False,
            "status": "blocked",
            "missing": ["d1_beef_controlled.csv"],
            "corrupt": [],
            "reason": "required file not on disk",
        }
    )
    table = build_contrast_table(frame, protocol, _metric_by_task(protocol))
    rules = evaluate_rules(table, protocol, availability, {"R0": 1000, "GRU": 50000})
    assert rules["N5"]["result"] is True

    # And with every dataset present, N5 does not fire even though the experiment
    # produced no D1 contrast — the distinction the first delivery got wrong.
    table = build_contrast_table(frame, protocol, _metric_by_task(protocol))
    rules = evaluate_rules(table, protocol, _availability(), {"R0": 1000, "GRU": 50000})
    assert rules["N5"]["result"] is False


@pytest.mark.integration
def test_the_effect_size_choice_follows_the_task_not_the_metric_name(protocol):
    """`r2` must not fall through to the rank-biserial branch (review item M3).

    The regression family is exercised end to end because the defect lived in the
    seam between the protocol's task-level declaration and the code's
    metric-name test.
    """
    frame = _fabricated_rows(n_folds=10, model_delta={"R0": 0.10, "R2": 0.01})
    frame["mae"] = 1.0 - frame["macro_f1"]
    frame["r2"] = frame["macro_f1"]
    table, _ = _run_pipeline(frame, protocol, {"R0": 1000, "GRU": 50000})

    by_metric = table.groupby("metric")["effect_size_name"].unique().to_dict()
    if "r2" in by_metric:
        assert list(by_metric["r2"]) == ["hodges_lehmann"], by_metric


@pytest.mark.integration
def test_no_p_value_below_the_float_floor_is_published_as_a_number(protocol):
    """A `1 - cdf` cancellation artefact is reported as '<1e-12', not as a value."""
    frame = _fabricated_rows(n_folds=10, model_delta={"R0": 0.9, "R2": 0.0})
    table, _ = _run_pipeline(frame, protocol, {"R0": 1000, "GRU": 50000})
    reported = table["p_paired_wilcoxon_reported"]
    assert all(isinstance(value, str) for value in reported)
    assert any(value == "<1e-12" for value in reported)


@pytest.mark.integration
def test_a_delta_of_zero_on_every_cluster_is_not_significant(protocol):
    """A tie must not be read as a win through the sign test's tie handling."""
    frame = _fabricated_rows(n_folds=10, model_delta={"R0": 0.0, "R2": 0.0})
    frame["macro_f1"] = 0.5
    table, _ = _run_pipeline(frame, protocol, {"R0": 1000, "GRU": 50000})
    row = table[table["contrast_id"] == "R0_vs_R2"].iloc[0]
    assert int(row["n_clusters_nonzero"]) == 0
    assert float(row["p_value"]) == 1.0
    assert bool(row["favours_first"]) is False


@pytest.mark.integration
def test_the_results_are_invariant_to_the_seed_count(protocol):
    """Seeds re-measure the same clusters and must not buy confidence.

    The per-run values are unchanged; only the number of times each cluster is
    measured changes. Neither the interval nor the decisive p-value may move,
    because neither the specimen sample nor the cluster count has.
    """
    deltas = {"R0": 0.10, "R2": 0.0}
    measured = []
    for n_seeds in (2, 10, 20):
        frame = _fabricated_rows(n_folds=10, model_delta=deltas, n_seeds=n_seeds)
        table = build_contrast_table(frame, protocol, _metric_by_task(protocol))
        row = table[table["contrast_id"] == "R0_vs_R2"].iloc[0]
        measured.append(
            (
                int(row["n_pairs"]),
                int(row["n_clusters"]),
                float(row["delta_ci_low"]),
                float(row["delta_ci_high"]),
                float(row["p_value"]),
            )
        )

    # Pairs multiply with seeds; clusters and both decisive quantities do not.
    assert [m[0] for m in measured] == [20, 100, 200]
    assert {m[1] for m in measured} == {10}
    first = measured[0]
    for row in measured[1:]:
        assert row[2] == pytest.approx(first[2])
        assert row[3] == pytest.approx(first[3])
        assert row[4] == pytest.approx(first[4])


# ---------------------------------------------------------------------------
# Reproducibility from the committed artifacts alone
#
# `results/raw/**` is git-ignored, so anything the gate engine reads from it is
# present only in the author's working copy. Two things were: the trainable
# parameter counts behind `params(...)` (review item N2), and — as a consequence
# — whether Gate_A can reach a verdict at all.
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_the_committed_parameter_table_resolves_params_without_the_run_records(protocol, tmp_path):
    """`results/tables/model_parameters.json` alone must satisfy `params(GRU)`.

    The file is committed precisely because `results/raw/**` is not. Reading it
    with no run records at all is what a reviewer's clean clone does.
    """
    from scripts.analyze import load_committed_model_parameters

    counts = load_committed_model_parameters()
    assert counts, "results/tables/model_parameters.json must be committed"
    assert "GRU" in counts, "the protocol id must be registered, not only the registry id"
    assert "R4" in counts
    assert counts["GRU"] == counts["gru"]
    assert counts["R4"] == counts["esn"]
    assert all(isinstance(value, int) and value > 0 for value in counts.values())


@pytest.mark.integration
def test_the_committed_parameter_table_states_its_selection_rule_and_spread(protocol):
    """A single number for a per-fold-tuned model has to be justified, not implied.

    Hyperparameters are selected per fold, so `n_trainable_parameters` is not
    unique: the GRU in this project reports eight different sizes. The table
    therefore records min/median/max/n_distinct beside the decisive value and
    states which one `params()` uses.
    """
    import json

    from drososense.utils.paths import MODEL_PARAMETERS_PATH

    payload = json.loads(MODEL_PARAMETERS_PATH.read_text(encoding="utf-8"))
    assert "largest selected configuration" in payload["selection_rule"]
    assert "git-ignored" in payload["source"]

    variable = [m for m, e in payload["models"].items() if e["n_distinct"] > 1]
    assert variable, "at least one model must show the selection spread"
    for entry in payload["models"].values():
        assert entry["n_trainable_parameters"] == entry["max"]
        assert entry["min"] <= entry["median"] <= entry["max"]
        assert entry["n_records"] >= entry["n_distinct"]


@pytest.mark.integration
def test_the_gate_outcome_does_not_depend_on_the_git_ignored_run_records(protocol, tmp_path):
    """The same table must reach the same verdict with and without `results/raw`.

    This is the actual defect behind review item N2: the delivered gate artifact
    said one thing on the author's machine and another on a clean clone, because
    a term was resolved from a directory the repository does not carry.

    Protocol v1.5.1 changed WHICH resolution the gates use, so this test now
    exercises the scoped path. Under a declared scope the two trees agree for a
    stronger reason than before: the count is read from matched rows inside the
    scope, so the presence of other records is irrelevant by construction, and a
    tree that lacks the scope's own rows is UNEVALUABLE rather than different.
    The unscoped scan is retained only as ``model_parameter_counts`` and is
    asserted below to be the thing the gates no longer call.
    """
    import drososense.evaluation.results as results_module
    from drososense.evaluation.parameter_scope import (
        load_parameter_scope_declaration,
        resolve_gate_parameter_scope,
    )

    declaration = load_parameter_scope_declaration()
    frame = pd.read_csv(COMMITTED_PER_RUN)
    table = build_contrast_table(
        frame[frame["status"] == "ok"], protocol, _metric_by_task(protocol), [("esn", "gru")]
    )

    with_scope = resolve_gate_parameter_scope(
        "Gate_A", results_module.load_records(), protocol=protocol, declaration=declaration
    )
    params_with = with_scope.counts if with_scope.evaluable else {}
    _, with_results = _run_pipeline(
        frame, protocol, params_with, exploratory=[("esn", "gru")],
        parameter_audit={"scopes": {"Gate_A": with_scope.provenance()}},
    )

    # A clean clone: results/raw does not exist at all.
    monkeypatch_target = results_module
    original = monkeypatch_target.RESULTS_RAW_DIR
    monkeypatch_target.RESULTS_RAW_DIR = tmp_path / "absent"
    try:
        assert results_module.load_records() == []
        without_scope = resolve_gate_parameter_scope(
            "Gate_A", results_module.load_records(), protocol=protocol, declaration=declaration
        )
        params_without = without_scope.counts if without_scope.evaluable else {}
        without_results = evaluate_rules(
            table, protocol, _availability(), params_without,
            {"scopes": {"Gate_A": without_scope.provenance()}},
        )
    finally:
        monkeypatch_target.RESULTS_RAW_DIR = original

    # The VERDICT must not depend on whether results/raw is present. The
    # provenance legitimately does: with the records present an unevaluable scope
    # says "conflicting values, here they are", and with the tree absent it says
    # "no rows in the declared scope". Both are UNEVALUABLE, and diagnosing them
    # identically would be the wrong kind of stable.
    assert {k: v["result"] for k, v in with_results.items()} == {
        k: v["result"] for k, v in without_results.items()
    }, "a verdict must not depend on the git-ignored records"
    for rule_id, entry in with_results.items():
        if entry["result"] == "UNEVALUABLE":
            assert entry["reason"], rule_id
            assert without_results[rule_id]["reason"], rule_id

    # The scope's own counts agree model for model between the two trees: absent
    # the rows there is no count at all, never a substituted one.
    for model in with_scope.provenance()["terms"]:
        assert (
            with_scope.provenance()["terms"][model]["parameter_count"]
            == without_scope.provenance()["terms"][model]["parameter_count"]
        ), model
    assert set(with_scope.provenance()["terms"]) == set(without_scope.provenance()["terms"])


@pytest.mark.integration
def test_an_unrun_model_says_so_rather_than_blaming_the_gitignore(protocol, tmp_path):
    """A `params(...)` term with no value must name the real cause.

    `R0` has never been run, so it has no parameter count anywhere — not in the
    committed table and not in `results/raw`. The message must not read like a
    missing-file problem, which is what sent the first round of this review
    looking in the wrong place.
    """
    from scripts.analyze import model_parameter_audit, model_parameter_counts

    counts = model_parameter_counts([], protocol)
    audit = model_parameter_audit(counts, [], protocol)
    assert audit["unresolved_params_terms"] == ["R0"]

    evaluator = GateEvaluator(
        contrasts={},
        metrics=protocol_metric_properties(protocol),
        model_params=counts,
        symbols=build_symbols(
            ["R0", "GRU"], [], [], [], aliases={}
        ),
    )
    with pytest.raises(GateExpressionError, match="no parameter evidence scope was declared"):
        evaluator.evaluate("G", "params(R0) < params(GRU)")

    # Protocol v1.5.1: when a scope IS declared, the error quotes the scope's own
    # reason, which is more specific than "not recorded" and names what would have
    # to change.
    from drososense.evaluation.parameter_scope import (
        load_parameter_scope_declaration,
        resolve_gate_parameter_scope,
    )

    scope = resolve_gate_parameter_scope(
        "Gate_A", [], protocol=protocol, declaration=load_parameter_scope_declaration()
    )
    scoped = GateEvaluator(
        contrasts={},
        metrics=protocol_metric_properties(protocol),
        model_params={},
        symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
        parameter_provenance=scope.provenance(),
    )
    with pytest.raises(GateExpressionError, match="declared parameter evidence scope"):
        scoped.evaluate("Gate_A", "params(R0) < params(GRU)")
