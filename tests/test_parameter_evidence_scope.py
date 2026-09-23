"""Protocol v1.5.1 — parameter evidence scope: regression tests.

WHAT THIS PINS. `params(model)` used to be resolved by scanning every record
under ``results/raw/**`` and taking the maximum ``n_trainable_parameters``. So
``params(R0)`` was "whichever R0 count currently sits in the tree": with the E9
size study on disk that is the N=4000 readout (16004), and Gate_A's
``params(R0) < params(GRU)`` then evaluated **false** for a reason unrelated to
the comparison, while a tree without ``results/raw`` reported the committed table
and gave another answer again.

The five regressions the amendment promises, each one a test:

    R1  E9 records present or absent      -> Gate_A's parameter scope is identical
    R2  N=250 and N=4000 coexist          -> only the designated size is read
    R3  an unrelated experiment appears   -> the scope's counts do not change
    R4  the designated source is missing  -> FAIL CLOSED, never substituted
    R5  two counts inside one scope       -> UNEVALUABLE, never max/min/first/last

plus a test that the declaration is anchored to the protocol rather than tuned
to a wanted number.
"""

from __future__ import annotations

import pytest

from drososense.evaluation.gates import (
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.evaluation.parameter_scope import (
    CONFLICTING_VALUES,
    NO_RECORDS,
    NO_RESERVOIR_SIZE,
    RESOLVED,
    load_parameter_scope_declaration,
    resolve_gate_parameter_scope,
)
from drososense.evaluation.results import RunRecord
from drososense.utils.config import load_protocol
from drososense.utils.paths import PROTOCOL_PATH

D2 = "d2_beef_uncontrolled"
D3 = "d3_rainbow_trout"


def record(
    *,
    experiment: str,
    model: str,
    dataset: str = D2,
    task: str = "classification",
    n_params: int | None = 1004,
    n_nodes: int | None = 250,
    config_hash: str = "c0ffee",
    run_id: str | None = None,
) -> RunRecord:
    """A minimal ok RunRecord carrying the fields the scope reads."""
    description: dict = {}
    if n_params is not None:
        description["n_trainable_parameters"] = n_params
    if n_nodes is not None:
        description["topology"] = {"n_nodes": n_nodes}
    return RunRecord(
        run_id=run_id or f"{dataset}|{model}|{task}|seed00|fold00",
        experiment=experiment,
        dataset=dataset,
        model=model,
        task=task,
        seed=0,
        fold_id=0,
        protocol_version="1.5.0",
        window_length=16,
        metrics={"macro_f1": 0.5},
        n_train_windows=10,
        n_test_windows=5,
        train_specimens=["TS1"],
        test_specimens=["TS2"],
        duration_s=1.0,
        environment={},
        timestamp_utc="2026-09-23T00:00:00+00:00",
        config_hash=config_hash,
        model_description=description,
        status="ok",
    )


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


@pytest.fixture()
def declaration():
    return load_parameter_scope_declaration()


def scope_for(records, protocol, declaration):
    return resolve_gate_parameter_scope(
        "Gate_A", records, protocol=protocol, declaration=declaration
    )


# ---------------------------------------------------------------------------
# R1 — E9 (an unrelated size study) present or absent changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_R1_unrelated_e9_records_do_not_change_the_scope(protocol, declaration):
    """The E9 size study is a different register entry; it must not be read.

    This is the regression that the pre-v1.5.1 code failed: its whole-tree scan
    let an E9 N=4000 record set `params(R0)` to 16004.
    """
    scoped = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
    ]
    e9 = [
        record(experiment=f"e9_size_d2_n{n}", model="R0", n_params=(n + 1) * 4, n_nodes=n)
        for n in (250, 500, 1000, 2000, 4000)
    ]

    without = scope_for(scoped, protocol, declaration)
    with_e9 = scope_for(scoped + e9, protocol, declaration)

    assert without.provenance() == with_e9.provenance(), (
        "adding the E9 size study must not move Gate_A's parameter scope"
    )
    assert with_e9.counts == {"R0": 1004, "GRU": 4452}


# ---------------------------------------------------------------------------
# R2 — coexisting sizes: only the designated size is read
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_R2_only_the_designated_reservoir_size_is_read(protocol, declaration):
    records = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", n_params=1004, n_nodes=250, dataset=D3),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
        record(experiment="e1_main_d3", model="gru", n_params=4353, n_nodes=None, dataset=D3),
    ]
    scope = scope_for(records, protocol, declaration)
    r0 = scope.evidence["R0"]
    assert r0.status == RESOLVED
    assert r0.parameter_count == 1004
    assert r0.reservoir_size == 250, "the scope reports the size it read"
    assert r0.n_source_records == 2


@pytest.mark.unit
def test_R2b_a_scope_mixing_two_reservoir_sizes_is_unevaluable(protocol, declaration):
    """Two substrates in one scope is not a number, it is an ambiguity."""
    records = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e2_main_d2", model="R0", n_params=16004, n_nodes=4000,
               run_id="dup", config_hash="beef"),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
    ]
    scope = scope_for(records, protocol, declaration)
    assert not scope.evaluable
    assert "different trainable-parameter counts" in scope.reason or \
           "mixes reservoir sizes" in scope.reason


# ---------------------------------------------------------------------------
# R3 — an unrelated experiment does not touch the scope
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_R3_an_unrelated_experiment_does_not_change_the_scope(protocol, declaration):
    base = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
    ]
    unrelated = [
        record(experiment="e7_efficiency_d2", model="R0", n_params=99999, n_nodes=250),
        record(experiment="e12_graph_d2", model="gru", n_params=88888, n_nodes=None),
        record(experiment="m1_benchmark", model="R0", n_params=77777, n_nodes=250),
    ]
    before = scope_for(base, protocol, declaration).provenance()
    after = scope_for(base + unrelated, protocol, declaration).provenance()
    assert before == after
    assert scope_for(base + unrelated, protocol, declaration).counts == {
        "R0": 1004, "GRU": 4452,
    }


# ---------------------------------------------------------------------------
# R4 — the designated source missing means FAIL CLOSED
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_R4_missing_designated_source_fails_closed(protocol, declaration):
    """No rows in the scope is UNEVALUABLE — never a substituted count."""
    only_r0 = [record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250)]
    scope = scope_for(only_r0, protocol, declaration)
    assert not scope.evaluable
    assert scope.evidence["GRU"].status == NO_RECORDS
    assert "unevaluable rather than substituted" in scope.reason

    # an empty tree, and a tree with only the committed benchmar experiment
    assert not scope_for([], protocol, declaration).evaluable
    assert not scope_for(
        [record(experiment="m1_benchmark", model="gru", n_params=4452, n_nodes=None)],
        protocol, declaration,
    ).evaluable


@pytest.mark.unit
def test_R4b_a_gate_reading_an_unevaluable_scope_reports_unevaluable(protocol, declaration):
    """End to end through the evaluator: the gate must not resolve a number."""
    scope = scope_for([], protocol, declaration)
    evaluator = GateEvaluator(
        contrasts={}, metrics={}, model_params={},
        symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
        parameter_provenance=scope.provenance(),
    )
    with pytest.raises(GateExpressionError) as excinfo:
        evaluator.evaluate("Gate_A", "params(R0) < params(GRU)")
    assert "declared parameter evidence scope" in str(excinfo.value)
    assert "no ok classification row" in str(excinfo.value)


# ---------------------------------------------------------------------------
# R5 — two different counts inside one scope: UNEVALUABLE, never a silent pick
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_R5_conflicting_counts_in_one_scope_are_unevaluable(protocol, declaration):
    """`params(GRU)` is 4452 on D2 and 4068 on D3 in the delivered evidence.

    The term is dataset-agnostic, so there is no unique value to report. The
    pre-v1.5.1 code took the maximum and published it as the model's size.
    """
    records = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", n_params=1004, n_nodes=250, dataset=D3),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
        record(experiment="e1_main_d3", model="gru", n_params=4068, n_nodes=None, dataset=D3),
    ]
    scope = scope_for(records, protocol, declaration)
    assert not scope.evaluable
    gru = scope.evidence["GRU"]
    assert gru.status == CONFLICTING_VALUES
    assert gru.conflicting_values == (4068, 4452), "both values are named, neither is chosen"
    assert "taking max/min/first/last would be a silent choice" in gru.detail
    assert "no fallback" not in gru.detail  # the reason is specific, not boilerplate
    with pytest.raises(Exception):
        scope.counts  # no lazy resolution either

    # and the gate says so rather than resolving
    # both shapes must surface the conflict, not just the bare scope
    for provenance in (
        scope.provenance(),
        {"declaration": "configs/protocol_v1.5.1.yaml",
         "scopes": {"Gate_A": scope.provenance()}},
    ):
        evaluator = GateEvaluator(
            contrasts={}, metrics={}, model_params={},
            symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
            parameter_provenance=provenance,
        )
        with pytest.raises(GateExpressionError, match="different trainable-parameter counts"):
            evaluator.evaluate("Gate_A", "params(R0) < params(GRU)")


@pytest.mark.unit
def test_R5b_the_conflict_is_reported_for_every_conflicting_model(protocol, declaration):
    """More than one conflict must all appear, not just the first."""
    records = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
        record(experiment="e2_main_d3", model="R0", n_params=1005, n_nodes=250, dataset=D3),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
        record(experiment="e1_main_d3", model="gru", n_params=4068, n_nodes=None, dataset=D3),
    ]
    scope = scope_for(records, protocol, declaration)
    assert set(scope.provenance()["terms"]) == {"R0", "GRU"}
    assert scope.reason.count("different trainable-parameter counts") == 2
    assert scope.evidence["R0"].conflicting_values == (1004, 1005)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_resolved_count_carries_its_provenance(protocol, declaration):
    """A published parameter number must be traceable to the rows behind it."""
    records = [
        record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250,
               config_hash="07cd93e03295", run_id="d2_beef_uncontrolled|R0|classification|seed00|fold00"),
        record(experiment="e2_main_d3", model="R0", n_params=1004, n_nodes=250,
               dataset=D3, config_hash="199df0376461", run_id="d3_rainbow_trout|R0|classification|seed00|fold00"),
        record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None,
               config_hash="c84d501b1a72", run_id="d2_beef_uncontrolled|gru|classification|seed00|fold00"),
    ]
    scope = scope_for(records, protocol, declaration)
    prov = scope.provenance()["terms"]["R0"]
    for key in ("model", "parameter_count", "experiment", "condition", "task",
                "datasets", "reservoir_size", "n_source_records", "source_run_ids",
                "config_hashes"):
        assert key in prov, key
    assert prov["parameter_count"] == 1004
    assert prov["reservoir_size"] == 250
    assert prov["condition"] == "full"
    assert prov["datasets"] == ["D2", "D3"]
    assert prov["n_source_records"] == 2
    assert set(prov["config_hashes"]) == {"07cd93e03295", "199df0376461"}


@pytest.mark.unit
def test_the_gate_result_carries_the_scope_it_used(protocol, declaration):
    scope = scope_for(
        [
            record(experiment="e2_main_d2", model="R0", n_params=1004, n_nodes=250),
            record(experiment="e1_main_d2", model="gru", n_params=4452, n_nodes=None),
        ],
        protocol, declaration,
    )
    evaluator = GateEvaluator(
        contrasts={}, metrics={}, model_params=scope.counts,
        symbols=build_symbols(["R0", "GRU"], [], [], [], aliases={}),
        parameter_provenance=scope.provenance(),
    )
    evaluation = evaluator.evaluate("Gate_A", "params(R0) < params(GRU)")
    assert evaluation.result is True
    assert evaluation.detail["parameter_scope"]["terms"]["R0"]["parameter_count"] == 1004


# ---------------------------------------------------------------------------
# the declaration is anchored to the protocol, not tuned to a wanted number
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_declaration_is_anchored_to_the_frozen_protocol(protocol, declaration):
    """Every declared value must be copied from the protocol's own blocks.

    The point is that the scope cannot be quietly re-pointed at whatever
    experiment yields a preferred number: the datasets, the task, the register
    entries and the contrasts all have to agree with the frozen text.
    """
    from drososense.utils.config import (
        protocol_dataset_symbols,
        protocol_model_symbols,
    )

    assert declaration["fallback"] == "forbidden"
    assert declaration["on_conflicting_values"] == "UNEVALUABLE"
    assert declaration["on_no_records"] == "UNEVALUABLE"

    spec = declaration["gates"]["Gate_A"]
    # datasets == gates.Gate_A.evaluated_on
    assert spec["datasets"] == list(protocol["gates"]["Gate_A"]["evaluated_on"])
    # task == the task whose primary metric the gate's terms read
    assert spec["task"] == "classification"
    assert "macro_f1" in protocol["gates"]["Gate_A"]["expression"]

    contrasts = {c["id"]: c for c in protocol["contrasts"]["list"]}
    models = protocol_model_symbols(protocol)
    for term in spec["terms"]:
        assert term["register_entry"] in protocol["experiments"], term
        contrast = contrasts[term["via_contrast"]]
        side = contrast["first"] if term["side"] == "first" else contrast["second"]
        assert side == term["model"], f"{term['model']} is not {term['side']} of {term['via_contrast']}"
        assert term["model"] in models
        # a connectome family requires the connectome; a baseline does not, and the
        # register entry must agree with which one this is
        is_connectome = term["model"] in {f"R{i}" for i in range(7)}
        assert bool(protocol["experiments"][term["register_entry"]]["needs_connectome"]) is is_connectome
        assert term["condition_bindings"]["full"], term

    # the register entries named must be the ones carrying the connectome
    # topology experiment and the baseline zoo, not the size study
    labels = {t["model"]: set(t["condition_bindings"]["full"]) for t in spec["terms"]}
    assert all(not any("e9_size" in x for x in v) for v in labels.values()), (
        "the E9 size study is a different register entry and must not be in scope"
    )
    assert any("e2_main" in x for x in labels["R0"]), labels["R0"]
    assert any("e1_main" in x for x in labels["GRU"]), labels["GRU"]
    # and the reference in the declaration to a dataset short name must be a
    # declared symbol
    assert set(spec["datasets"]) <= set(protocol_dataset_symbols(protocol))


@pytest.mark.unit
def test_v1_5_1_is_frozen_with_a_matching_sidecar_and_earlier_files_untouched():
    from drososense.utils.config import protocol_sha256, recorded_protocol_sha256
    from drososense.utils.paths import (
        CONFIGS_DIR,
        PROTOCOL_V1_5_1_PATH,
        PROTOCOL_V1_5_1_SHA256_PATH,
    )

    assert PROTOCOL_V1_5_1_PATH.is_file()
    assert PROTOCOL_V1_5_1_SHA256_PATH.is_file()
    assert protocol_sha256(PROTOCOL_V1_5_1_PATH) == recorded_protocol_sha256(
        PROTOCOL_V1_5_1_SHA256_PATH
    )
    import yaml

    parsed = yaml.safe_load(PROTOCOL_V1_5_1_PATH.read_text(encoding="utf-8"))
    assert parsed["protocol_version"] == "1.5.1"
    assert parsed["frozen"] is True
    assert parsed["supersedes"] == "1.5.0"
    assert "params(model)" in parsed["scope_statement"]

    for name in ("protocol_v1.yaml", "protocol_v1.1.yaml", "protocol_v1.2.yaml",
                 "protocol_v1.3.yaml", "protocol_v1.4.yaml", "protocol_v1.5.yaml"):
        path = CONFIGS_DIR / name
        sidecar = CONFIGS_DIR / (name.replace(".yaml", "") + ".sha256")
        if not sidecar.is_file():
            continue
        assert protocol_sha256(path) == recorded_protocol_sha256(sidecar), name
