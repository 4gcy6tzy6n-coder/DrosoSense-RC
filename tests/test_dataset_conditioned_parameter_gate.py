"""Protocol v1.5.2 — dataset-conditioned parameter gates: regression tests.

THE DECISION v1.5.2 FIXES. v1.5.1 made `params(model)` read matched rows inside a
declared scope and fail closed, and then reported what the delivered evidence
says: within Gate_A's scope `params(R0)` is 1004 on both datasets but
`params(GRU)` is 4452 on D2 and 4068 on D3, because the GRU's width follows its
per-dataset registered configuration. Gate_A's expression is one dataset-agnostic
`params(R0) < params(GRU)` term, so v1.5.1 correctly called it UNEVALUABLE rather
than picking a maximum. v1.5.2 defines the semantics instead:

    A2 = PASS          iff for every d in evaluated_on: params(R0,d) < params(GRU,d)
         FAIL          iff for some d: params(R0,d) >= params(GRU,d)
         UNEVALUABLE   iff for some d the dataset-specific scope is unresolved

with UNEVALUABLE taking precedence over FAIL, and no averaging, extrema
selection, cross-dataset fallback or unscoped search anywhere.
"""

from __future__ import annotations

import pytest

from drososense.evaluation.gates import (
    DatasetVector,
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.evaluation.parameter_scope import (
    CONFLICTING_VALUES,
    NO_RECORDS,
    load_parameter_scope_declaration,
    resolve_gate_parameter_scope,
)
from drososense.evaluation.results import RunRecord
from drososense.utils.config import load_protocol
from drososense.utils.paths import PROTOCOL_PATH

D2 = "d2_beef_uncontrolled"
D3 = "d3_rainbow_trout"
EXPRESSION = "params(R0) < params(GRU)"

#: The delivered evidence, as the amendment's own table states it.
DELIVERED = {"D2": {"R0": 1004, "GRU": 4452}, "D3": {"R0": 1004, "GRU": 4068}}


def record(*, experiment, model, dataset, n_params, n_nodes=None,
           task="classification", config_hash="c0ffee", run_id=None):
    description = {"n_trainable_parameters": n_params}
    if n_nodes is not None:
        description["topology"] = {"n_nodes": n_nodes}
    return RunRecord(
        run_id=run_id or f"{dataset}|{model}|{task}|seed00|fold00",
        experiment=experiment, dataset=dataset, model=model, task=task,
        seed=0, fold_id=0, protocol_version="1.5.0", window_length=16,
        metrics={"macro_f1": 0.5}, n_train_windows=10, n_test_windows=5,
        train_specimens=["TS1"], test_specimens=["TS2"], duration_s=1.0,
        environment={}, timestamp_utc="2026-09-23T00:00:00+00:00",
        config_hash=config_hash, model_description=description, status="ok",
    )


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


@pytest.fixture()
def declaration():
    return load_parameter_scope_declaration()


def scope_of(records, protocol, declaration):
    return resolve_gate_parameter_scope(
        "Gate_A", records, protocol=protocol, declaration=declaration
    )


def evaluate(per_dataset_counts, provenance=None, protocol=None):
    """Evaluate the frozen expression with the given per-dataset counts."""
    evaluator = GateEvaluator(
        contrasts={},
        metrics={},
        model_params={},
        symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
        parameter_provenance=provenance or {},
        parameter_counts_by_dataset={"Gate_A": per_dataset_counts},
    )
    try:
        return evaluator.evaluate("Gate_A", EXPRESSION).result
    except GateExpressionError as exc:
        return f"UNEVALUABLE: {exc}"


# ---------------------------------------------------------------------------
# 1. every dataset holds -> PASS
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_every_dataset_holding_makes_the_predicate_pass():
    assert evaluate(DELIVERED) is True


@pytest.mark.unit
def test_the_resolved_scope_reports_one_count_per_dataset(protocol, declaration):
    records = [
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", dataset=D3, n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", dataset=D2, n_params=4452),
        record(experiment="e1_main_d3", model="gru", dataset=D3, n_params=4068),
    ]
    scope = scope_of(records, protocol, declaration)
    assert scope.evaluable
    assert scope.counts_by_dataset == DELIVERED
    # v1.5.2's rejected alternative would call this a conflict. It is not one:
    # the count follows the registered per-dataset configuration.
    assert scope.counts_by_dataset["D2"]["GRU"] != scope.counts_by_dataset["D3"]["GRU"]


# ---------------------------------------------------------------------------
# 2. any dataset failing -> FAIL
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset,counts",
    [
        ("D3", {"D2": {"R0": 1004, "GRU": 4452}, "D3": {"R0": 5000, "GRU": 4068}}),
        ("D2", {"D2": {"R0": 9000, "GRU": 4452}, "D3": {"R0": 1004, "GRU": 4068}}),
        ("D3 tie", {"D2": {"R0": 1004, "GRU": 4452}, "D3": {"R0": 4068, "GRU": 4068}}),
    ],
)
def test_any_dataset_failing_makes_the_predicate_fail(dataset, counts):
    assert evaluate(counts) is False, dataset


# ---------------------------------------------------------------------------
# 3. any dataset missing -> UNEVALUABLE
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_missing_dataset_scope_is_unevaluable(protocol, declaration):
    """No rows on D3 means D3 is unresolved, and the aggregate cannot PASS."""
    records = [
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", dataset=D2, n_params=4452),
    ]
    scope = scope_of(records, protocol, declaration)
    assert not scope.evaluable
    assert scope.per_dataset["D3"]["R0"].status == NO_RECORDS
    assert "on D3" in scope.reason

    # ... and the evaluator reports UNEVALUABLE, NOT PASS, even though D2 holds.
    outcome = evaluate(
        {"D2": {"R0": 1004, "GRU": 4452}, "D3": {"R0": 1004}},
        {"scopes": {"Gate_A": scope.provenance()}},
    )
    assert isinstance(outcome, str) and outcome.startswith("UNEVALUABLE")


# ---------------------------------------------------------------------------
# 4. a conflict INSIDE one dataset -> UNEVALUABLE
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_conflict_inside_one_dataset_is_unevaluable(protocol, declaration):
    """The v1.5.1 rule still holds within a dataset: two counts, no silent pick."""
    records = [
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1004, n_nodes=250),
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1005, n_nodes=250,
               run_id="dup", config_hash="beef"),
        record(experiment="e2_main_d3", model="R0", dataset=D3, n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", dataset=D2, n_params=4452),
        record(experiment="e1_main_d3", model="gru", dataset=D3, n_params=4068),
    ]
    scope = scope_of(records, protocol, declaration)
    assert not scope.evaluable
    assert scope.per_dataset["D2"]["R0"].status == CONFLICTING_VALUES
    assert scope.per_dataset["D2"]["R0"].conflicting_values == (1004, 1005)
    assert scope.per_dataset["D3"]["R0"].status == "resolved", (
        "the conflict is confined to the dataset that has it"
    )
    # The production wiring, mirrored: analyze.py passes counts ONLY for an
    # evaluable scope, so an unevaluable scope hands the evaluator no counts and
    # no per-dataset vector at all — `params(...)` then raises with the scope's
    # own reason. Handing it the resolved D3 numbers instead would let the
    # comparison return True while D2 was never established, which is exactly the
    # failure mode v1.5.2 forbids.
    evaluator = GateEvaluator(
        contrasts={}, metrics={}, model_params={},
        symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
        parameter_provenance={"scopes": {"Gate_A": scope.provenance()}},
    )
    with pytest.raises(GateExpressionError, match="conflicting|unevaluable"):
        evaluator.evaluate("Gate_A", EXPRESSION)


# ---------------------------------------------------------------------------
# 5. unrelated evidence changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unrelated_records_do_not_change_the_result(protocol, declaration):
    """E9 at N=4000, another experiment, a different condition: all inert."""
    base = [
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", dataset=D3, n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", dataset=D2, n_params=4452),
        record(experiment="e1_main_d3", model="gru", dataset=D3, n_params=4068),
    ]
    unrelated = []
    for n in (250, 500, 1000, 2000, 4000):
        for dataset in (D2, D3):
            unrelated.append(record(
                experiment=f"e9_size_d2_n{n}", model="R0", dataset=dataset,
                n_params=(n + 1) * 4, n_nodes=n,
                run_id=f"{dataset}|R0|e9n{n}", config_hash=f"e9{n}",
            ))
    unrelated += [
        record(experiment="e7_efficiency_d2", model="R0", dataset=D2, n_params=99999, n_nodes=250),
        record(experiment="e12_graph_d2", model="gru", dataset=D2, n_params=88888),
        record(experiment="m1_benchmark", model="R0", dataset=D2, n_params=77777, n_nodes=250),
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=12345, n_nodes=250,
               task="regression", run_id="reg", config_hash="reg"),
    ]

    before = scope_of(base, protocol, declaration)
    after = scope_of(base + unrelated, protocol, declaration)
    assert before.provenance()["per_dataset_counts"] == \
           after.provenance()["per_dataset_counts"] == DELIVERED
    assert evaluate(before.counts_by_dataset) == evaluate(after.counts_by_dataset) is True
    # and the identifiers behind the counts are unchanged too
    for dataset in ("D2", "D3"):
        for model in ("R0", "GRU"):
            assert (before.per_dataset[dataset][model].n_source_records
                    == after.per_dataset[dataset][model].n_source_records), (dataset, model)


# ---------------------------------------------------------------------------
# precedence, and the rejected alternative
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unevaluable_takes_precedence_over_fail():
    """D2 failing AND D3 unresolved is UNEVALUABLE, not FAIL.

    §13 already requires a gate that cannot be evaluated to be reported
    UNEVALUABLE and never as passed or failed; reporting FAIL would assert an
    inequality about a dataset whose count was never established.
    """
    outcome = evaluate(
        {"D2": {"R0": 9000, "GRU": 4452}, "D3": {"R0": 1004}},
        {"scopes": {"Gate_A": {"per_dataset": {"D3": {"GRU": {"detail": "conflict on D3"}}}}}},
    )
    assert isinstance(outcome, str) and outcome.startswith("UNEVALUABLE")
    assert "conflict on D3" in outcome


@pytest.mark.unit
def test_a_dataset_dependent_count_is_not_by_itself_a_conflict(protocol, declaration):
    """The rejected alternative, stated as a test.

    Requiring params(GRU, D2) == params(GRU, D3) would impose a cross-dataset
    invariance on a count that legitimately follows the registered configuration.
    v1.5.2 records that rejection, so this behaviour must not be 'fixed' later by
    accident.
    """
    import yaml

    from drososense.utils.paths import PROTOCOL_V1_5_2_PATH

    parsed = yaml.safe_load(PROTOCOL_V1_5_2_PATH.read_text(encoding="utf-8"))
    block = parsed["dataset_conditioned_parameter_predicates"]
    assert "rejected_alternative" in block
    assert "be equal" in block["rejected_alternative"]["rule"]
    assert block["conjunction"] == "all_of_evaluated_on"
    for forbidden in ("fallback", "averaging", "extrema_selection",
                      "cross_dataset_fallback", "unscoped_search"):
        assert block[forbidden] == "forbidden", forbidden

    records = [
        record(experiment="e2_main_d2", model="R0", dataset=D2, n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", dataset=D3, n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", dataset=D2, n_params=4452),
        record(experiment="e1_main_d3", model="gru", dataset=D3, n_params=4068),
    ]
    scope = scope_of(records, protocol, declaration)
    assert scope.evaluable, "a per-dataset difference must not make the scope unevaluable"


@pytest.mark.unit
def test_the_frozen_expression_text_is_unchanged(protocol):
    """v1.5.2 changes semantics, not text: editing a frozen expression is not an amendment."""
    expression = " ".join(protocol["gates"]["Gate_A"]["expression"].split())
    assert "params(R0) < params(GRU)" in expression
    assert "forall" not in expression and "for all" not in expression.lower()


@pytest.mark.unit
def test_v1_5_2_is_frozen_with_a_matching_sidecar_and_earlier_files_untouched():
    import yaml

    from drososense.utils.config import protocol_sha256, recorded_protocol_sha256
    from drososense.utils.paths import (
        CONFIGS_DIR,
        PROTOCOL_V1_5_2_PATH,
        PROTOCOL_V1_5_2_SHA256_PATH,
    )

    assert PROTOCOL_V1_5_2_PATH.is_file()
    assert PROTOCOL_V1_5_2_SHA256_PATH.is_file()
    assert protocol_sha256(PROTOCOL_V1_5_2_PATH) == recorded_protocol_sha256(
        PROTOCOL_V1_5_2_SHA256_PATH
    )
    parsed = yaml.safe_load(PROTOCOL_V1_5_2_PATH.read_text(encoding="utf-8"))
    assert parsed["protocol_version"] == "1.5.2"
    assert parsed["frozen"] is True
    assert parsed["supersedes"] == "1.5.1"

    for name in ("protocol_v1.yaml", "protocol_v1.1.yaml", "protocol_v1.2.yaml",
                 "protocol_v1.3.yaml", "protocol_v1.4.yaml", "protocol_v1.5.yaml",
                 "protocol_v1.5.1.yaml"):
        path = CONFIGS_DIR / name
        sidecar = CONFIGS_DIR / (name.replace(".yaml", "") + ".sha256")
        if not sidecar.is_file():
            continue
        assert protocol_sha256(path) == recorded_protocol_sha256(sidecar), name


@pytest.mark.unit
def test_gate_b_and_c_carry_no_parameter_term_and_are_untouched(protocol):
    """v1.5.2's effect on B and C is nil: they have no params(...) term at all."""
    for gate_id in ("Gate_B", "Gate_C"):
        expression = protocol["gates"][gate_id]["expression"]
        assert "params(" not in expression, gate_id
        assert "Gate_A" not in expression, gate_id
