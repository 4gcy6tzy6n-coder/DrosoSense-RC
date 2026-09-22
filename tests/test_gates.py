"""Gate and narrative-rule evaluation, and the safety of the evaluator itself.

Protocol v1 stated its gates as prose. Protocol v1.1 stated them as boolean
expressions over a fixed predicate vocabulary, and these tests hold the evaluator
to two things: that it computes what the protocol says, and that it cannot be
talked into executing anything else.

Protocol v1.2 adds a third: that the expressions can actually be evaluated on a
real contrast table. The R0.1 statistical review (DATA-18) found that they could
not. The gate text names datasets ``D1`` / ``D2`` / ``D3`` while the contrast
table is keyed by config ids, so every rule reported UNEVALUABLE and the reason
printed was the one the protocol reserves for a missing contrast. The dataset
constants below are therefore the PROTOCOL's short names mapped through the
protocol's own aliases, and the contrast keys are the config ids — the same
split of vocabularies production has, because a test that lives in a namespace
production never produces is exactly what let this through.
"""

from __future__ import annotations

import pytest

from drososense.evaluation.gates import (
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.utils.config import load_protocol, protocol_metric_properties

MODELS = ["R0", "R1", "R2", "R3", "R4", "R5", "GRU", "ESN"]
METRICS = ["macro_f1", "mae", "balanced_accuracy"]
CONDITIONS = ["dropout_p0.3", "noise_s0.1", "train10pct", "train25pct"]

# The protocol's own vocabulary, and the ids its result tables actually carry.
DATASET_ALIASES = {
    "D1": "d1_beef_controlled",
    "D2": "d2_beef_uncontrolled",
    "D3": "d3_rainbow_trout",
}
DATASETS = sorted(DATASET_ALIASES)
DATASET_IDS = sorted(DATASET_ALIASES.values())

# The config ids, used as contrast-table keys. Gate expressions keep the
# protocol's short names; the alias is what connects the two.
D1, D2, D3 = (DATASET_ALIASES[name] for name in ("D1", "D2", "D3"))

# Six clusters is the fewest for which a cluster-level exact two-sided p can
# reach alpha = 0.05 (2/2**5 = 0.0625 > 0.05, 2/2**6 = 0.03125 <= 0.05). Most
# tests use ten so that `sig` is decided by its p-value and not by its floor.
REACHABLE_CLUSTERS = 10
UNREACHABLE_CLUSTERS = 5


def _dataset_symbols() -> dict:
    """Both spellings of every dataset, exactly as production registers them."""
    symbols = dict(DATASET_ALIASES)
    for dataset_id in DATASET_ALIASES.values():
        symbols[dataset_id] = dataset_id
    return symbols


def _symbols() -> dict:
    return build_symbols(MODELS, METRICS, [], CONDITIONS, aliases=_dataset_symbols())


def _metrics() -> dict:
    return {
        "macro_f1": {"direction": "maximize", "margin": 0.02, "alpha": 0.05},
        "balanced_accuracy": {"direction": "maximize", "margin": 0.02, "alpha": 0.05},
        "mae": {"direction": "minimize", "margin": 0.05, "alpha": 0.05},
    }


def _contrast(
    delta: float,
    ci_low: float,
    ci_high: float,
    p_holm: float,
    n_pairs: int = 50,
    n_clusters: int = REACHABLE_CLUSTERS,
) -> dict:
    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_holm": p_holm,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "n_clusters_nonzero": n_clusters,
    }


def _evaluator(contrasts: dict, model_params: dict | None = None, available=None) -> GateEvaluator:
    return GateEvaluator(
        contrasts=contrasts,
        metrics=_metrics(),
        model_params=model_params or {},
        symbols=_symbols(),
        available_datasets=available if available is not None else DATASET_IDS,
    )


# ---------------------------------------------------------------------------
# The evaluator cannot be used as a general expression runner
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_evaluator_refuses_attribute_access():
    """No attribute access means no reachable object graph."""
    with pytest.raises(GateExpressionError, match="not allowed"):
        _evaluator({}).evaluate("G", "__import__('os').system('true') == None")


@pytest.mark.unit
def test_the_evaluator_refuses_unknown_names():
    """A bare name must be a declared model, metric, dataset or condition."""
    with pytest.raises(GateExpressionError, match="not a declared model"):
        _evaluator({}).evaluate("G", "R9 == 1")


@pytest.mark.unit
def test_the_evaluator_refuses_unknown_functions():
    """Only the declared predicate vocabulary is callable."""
    with pytest.raises(GateExpressionError, match="not a known gate predicate"):
        _evaluator({}).evaluate("G", "open('x') == 1")


@pytest.mark.unit
def test_the_evaluator_requires_a_boolean_result():
    """An expression that yields a number is a protocol mistake, not a truthy gate."""
    with pytest.raises(GateExpressionError, match="not a boolean"):
        _evaluator({}).evaluate("G", "1 + 1")


@pytest.mark.unit
def test_a_missing_contrast_makes_the_gate_unevaluable_not_false():
    """Silently returning False would read as 'the gate failed'."""
    with pytest.raises(GateExpressionError, match="no result for contrast"):
        _evaluator({}).evaluate("G", "sig(R0, R2, macro_f1, D3)")


# ---------------------------------------------------------------------------
# The namespace: declared names resolve, observed values do not decide
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_protocols_dataset_short_name_resolves_to_its_config_id():
    """`D3` in a gate expression must reach a row keyed `d3_rainbow_trout`."""
    contrasts = {
        ("R0_vs_R2", "macro_f1", DATASET_ALIASES["D3"], "full"): _contrast(0.05, 0.01, 0.09, 0.004)
    }
    evaluation = _evaluator(contrasts).evaluate("G", "delta(R0, R2, macro_f1, D3) > 0")
    assert evaluation.result is True
    # The trace is keyed by the resolved id, so the row that decided the gate is
    # identifiable without re-deriving the alias.
    assert any(DATASET_ALIASES["D3"] in key for key in evaluation.detail)


@pytest.mark.unit
def test_both_spellings_of_a_dataset_resolve():
    """The config id is usable in an expression too; either spelling is the name."""
    contrasts = {
        ("R0_vs_R2", "macro_f1", DATASET_ALIASES["D2"], "full"): _contrast(0.05, 0.01, 0.09, 0.004)
    }
    evaluator = _evaluator(contrasts)
    assert evaluator.evaluate("G", "delta(R0, R2, macro_f1, D2) > 0").result is True
    assert (
        evaluator.evaluate("G", "delta(R0, R2, macro_f1, d2_beef_uncontrolled) > 0").result is True
    )


@pytest.mark.unit
def test_an_unknown_name_and_a_missing_contrast_are_different_failures():
    """The R0.1 review's C1 was hidden by these two reading the same.

    An unknown name is a namespace defect the code must fix. A missing contrast
    is a gap in the evidence that an experiment must fill. Reporting the first as
    the second sent a reader looking for results that were never the problem.
    """
    evaluator = _evaluator({})
    with pytest.raises(GateExpressionError, match="namespace mismatch"):
        evaluator.evaluate("G", "delta(R0, R2, macro_f1, D9) > 0")
    with pytest.raises(GateExpressionError, match="no result for contrast") as excinfo:
        evaluator.evaluate("G", "delta(R0, R2, macro_f1, D3) > 0")
    assert "namespace" not in str(excinfo.value)


@pytest.mark.unit
def test_an_unknown_name_reports_what_is_declared():
    """The message must be actionable without reading the symbol table source."""
    with pytest.raises(GateExpressionError, match="Declared names: .*D2"):
        _evaluator({}).evaluate("G", "delta(R0, R2, macro_f1, D9) > 0")


@pytest.mark.unit
def test_the_test_aliases_match_what_the_protocol_declares(protocol):
    """The constants above must be the protocol's own binding, not a copy.

    If the protocol renamed a dataset and this file did not, the suite would go
    on passing in a namespace that no longer exists — which is precisely the
    failure mode it is here to prevent.
    """
    from drososense.utils.config import protocol_dataset_symbols

    assert protocol_dataset_symbols(protocol) == _dataset_symbols()


@pytest.mark.unit
def test_the_protocols_declared_names_all_resolve(protocol):
    """Nothing the frozen gates name may be missing from the symbol table."""
    from drososense.utils.config import (
        protocol_condition_symbols,
        protocol_dataset_symbols,
        protocol_model_symbols,
    )

    symbols = build_symbols(
        protocol_model_symbols(protocol),
        sorted(protocol_metric_properties(protocol)),
        [],
        protocol_condition_symbols(protocol),
        aliases=protocol_dataset_symbols(protocol),
    )
    evaluator = GateEvaluator(contrasts={}, metrics=protocol_metric_properties(protocol), symbols=symbols)
    for gate_id, definition in protocol["gates"].items():
        expression = definition.get("expression") if isinstance(definition, dict) else None
        if expression is None:
            continue
        # A missing contrast is fine here; an unknown NAME is not.
        try:
            evaluator.evaluate(gate_id, str(expression))
        except GateExpressionError as exc:
            assert "is not a declared" not in str(exc), f"{gate_id}: {exc}"


@pytest.mark.unit
def test_build_symbols_refuses_an_alias_that_shadows_a_declared_name():
    """An alias that means something else makes every expression ambiguous."""
    with pytest.raises(ValueError, match="shadows the declared name"):
        build_symbols(["R0"], [], ["D2"], [], aliases={"D2": "d2_beef_uncontrolled"})


@pytest.mark.unit
def test_build_symbols_without_aliases_is_unchanged():
    """The alias parameter is additive; the plain call still maps name to itself."""
    assert build_symbols(["R0"], ["macro_f1"], ["D2"], ["full"]) == {
        "R0": "R0",
        "macro_f1": "macro_f1",
        "D2": "D2",
        "full": "full",
    }


# ---------------------------------------------------------------------------
# The predicates compute what the protocol says
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sig_needs_both_a_corrected_p_and_an_interval_off_zero():
    """A p-value alone is not enough; the effect has to point the right way."""
    improving = {("R0_vs_R2", "macro_f1", D3, "full"): _contrast(0.05, 0.01, 0.09, 0.004)}
    assert _evaluator(improving).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is True

    not_significant = {("R0_vs_R2", "macro_f1", D3, "full"): _contrast(0.05, 0.01, 0.09, 0.4)}
    assert _evaluator(not_significant).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is False

    interval_spans_zero = {
        ("R0_vs_R2", "macro_f1", D3, "full"): _contrast(0.05, -0.01, 0.09, 0.004)
    }
    assert _evaluator(interval_spans_zero).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is False


@pytest.mark.unit
def test_sig_refuses_a_pseudoreplicated_p_on_too_few_clusters():
    """R0.1 review item C2, as a test.

    Five clusters have an exact two-sided floor of 0.0625. A p-value of 5e-13 on
    such a dataset is not evidence of anything: it can only have come from
    counting the same five units many times over. `sig` must return False.
    """
    pseudoreplicated = {
        ("R0_vs_R2", "macro_f1", D2, "full"): _contrast(
            0.05, 0.01, 0.09, 5.453415496958769e-13, n_clusters=UNREACHABLE_CLUSTERS
        )
    }
    evaluation = _evaluator(pseudoreplicated).evaluate("G", "sig(R0, R2, macro_f1, D2)")
    assert evaluation.result is False
    trace = next(iter(evaluation.detail.values()))
    assert trace["minimum_achievable_p_over_clusters"] == pytest.approx(0.0625)


@pytest.mark.unit
def test_sig_is_reachable_once_there_are_enough_clusters():
    """The same numbers with ten clusters are decided by the p-value again."""
    reachable = {
        ("R0_vs_R2", "macro_f1", D3, "full"): _contrast(
            0.05, 0.01, 0.09, 5.453415496958769e-13, n_clusters=REACHABLE_CLUSTERS
        )
    }
    assert _evaluator(reachable).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is True


@pytest.mark.unit
def test_sig_refuses_to_decide_without_a_cluster_count():
    """A p-value with no independent-unit count is not evaluable."""
    row = _contrast(0.05, 0.01, 0.09, 0.001)
    del row["n_clusters"]
    del row["n_clusters_nonzero"]
    with pytest.raises(GateExpressionError, match="carries no cluster count"):
        _evaluator({("R0_vs_R2", "macro_f1", D3, "full"): row}).evaluate(
            "G", "sig(R0, R2, macro_f1, D3)"
        )


@pytest.mark.unit
def test_minimising_metrics_are_favoured_by_lower_values():
    """MAE improves downward, so the favourable interval is the negative one."""
    improving = {("R0_vs_GRU", "mae", D3, "full"): _contrast(-0.3, -0.5, -0.1, 0.001)}
    assert _evaluator(improving).evaluate("G", "sig(R0, GRU, mae, D3)").result is True
    worsening = {("R0_vs_GRU", "mae", D3, "full"): _contrast(0.3, 0.1, 0.5, 0.001)}
    assert _evaluator(worsening).evaluate("G", "sig(R0, GRU, mae, D3)").result is False


@pytest.mark.unit
def test_equiv_is_tost_not_a_failure_to_reject():
    """A wide interval spanning zero proves nothing about equivalence."""
    wide = {("R0_vs_R2", "macro_f1", D3, "full"): _contrast(0.0, -0.5, 0.5, 0.9)}
    assert _evaluator(wide).evaluate("G", "equiv(R0, R2, macro_f1, D3)").result is False

    tight = {("R0_vs_R2", "macro_f1", D3, "full"): _contrast(0.0, -0.005, 0.005, 0.9)}
    assert _evaluator(tight).evaluate("G", "equiv(R0, R2, macro_f1, D3)").result is True


@pytest.mark.unit
def test_noninferiority_uses_the_declared_margin():
    """Gate_A's tolerance is the protocol's margin, not a number in the gate text."""
    inside = {("R0_vs_R4", "macro_f1", D3, "full"): _contrast(-0.01, -0.015, 0.0, 0.2)}
    assert _evaluator(inside).evaluate("G", "noninferior(R0, R4, macro_f1, D3)").result is True

    outside = {("R0_vs_R4", "macro_f1", D3, "full"): _contrast(-0.3, -0.4, -0.2, 0.001)}
    assert _evaluator(outside).evaluate("G", "noninferior(R0, R4, macro_f1, D3)").result is False


@pytest.mark.unit
def test_params_and_margin_come_from_the_declared_tables():
    """A gate may not invent a parameter count or an equivalence margin."""
    evaluator = _evaluator({}, model_params={"R0": 1000, "GRU": 50000})
    assert evaluator.evaluate("G", "params(R0) < params(GRU)").result is True
    assert evaluator.evaluate("G", "margin(macro_f1) == 0.02").result is True
    with pytest.raises(GateExpressionError, match="no trainable-parameter count"):
        evaluator.evaluate("G", "params(ESN) > 0")


@pytest.mark.unit
def test_sig_any_scans_datasets_and_conditions():
    """Gate_B's robustness clause is a scan, and it skips absent combinations."""
    contrasts = {
        ("R0_vs_R4", "macro_f1", D3, "dropout_p0.3"): _contrast(0.06, 0.02, 0.1, 0.002),
    }
    evaluator = _evaluator(contrasts)
    # Conditions are quoted because `dropout_p0.3` is not a Python identifier.
    expression = 'sig_any(R0, R4, macro_f1, [D2, D3], ["noise_s0.1", "dropout_p0.3"])'
    assert evaluator.evaluate("G", expression).result is True


@pytest.mark.unit
def test_equivalent_all_requires_every_dataset():
    """equiv_all is a conjunction, so one wide interval breaks it."""
    partial = {
        ("R0_vs_R4", "macro_f1", D2, "full"): _contrast(0.0, -0.005, 0.005, 0.9),
        ("R0_vs_R4", "macro_f1", D3, "full"): _contrast(0.0, -0.4, 0.4, 0.9),
    }
    assert _evaluator(partial).evaluate("G", "equiv_all(R0, R4, macro_f1, [D2, D3])").result is False


@pytest.mark.unit
def test_unavailable_tracks_the_datasets_that_were_obtained():
    """N5 fires on a missing dataset, and only on a missing one.

    Availability is expressed in the resolved id namespace, because that is what
    the manifest and the run records are keyed by.
    """
    evaluator = _evaluator({}, available=DATASET_IDS[1:])
    assert evaluator.evaluate("G", "unavailable(D1)").result is True
    assert evaluator.evaluate("G", "unavailable(D2)").result is False


# ---------------------------------------------------------------------------
# The protocol's own gates
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_protocols_gates_evaluate_in_a_contrived_strong_result(protocol):
    """A fabricated result that satisfies every clause must pass all three gates."""
    name = "macro_f1"
    contrasts = {}
    for dataset in (DATASET_ALIASES["D2"], DATASET_ALIASES["D3"]):
        contrasts[("R0_vs_R4", name, dataset, "full")] = _contrast(0.10, 0.05, 0.15, 0.001)
        contrasts[("R0_vs_R2", name, dataset, "full")] = _contrast(0.08, 0.03, 0.13, 0.002)
        contrasts[("R0_vs_R3", name, dataset, "full")] = _contrast(0.12, 0.06, 0.18, 0.001)
        contrasts[("R0_vs_R4", name, dataset, "dropout_p0.3")] = _contrast(
            0.07, 0.02, 0.12, 0.003
        )
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASET_IDS,
    )
    evaluations = evaluator.evaluate_all(protocol["gates"])
    assert set(evaluations) == {"Gate_A", "Gate_B", "Gate_C"}
    for gate in evaluations.values():
        assert gate.result is True, f"{gate.gate_id} did not pass on a strong fabricated result"
        assert gate.detail, "a gate evaluation must carry the per-term values behind it"


@pytest.mark.unit
def test_the_protocols_gates_fail_on_a_negative_result(protocol):
    """A fabricated result where the connectome loses must not pass Gate_B."""
    name = "macro_f1"
    contrasts = {
        ("R0_vs_R4", name, "d2_beef_uncontrolled", "full"): _contrast(-0.10, -0.15, -0.05, 0.001),
        ("R0_vs_R4", name, "d3_rainbow_trout", "full"): _contrast(-0.12, -0.18, -0.06, 0.001),
        ("R0_vs_R2", name, "d2_beef_uncontrolled", "full"): _contrast(-0.05, -0.09, -0.01, 0.01),
        ("R0_vs_R2", name, "d3_rainbow_trout", "full"): _contrast(-0.09, -0.14, -0.04, 0.002),
        ("R0_vs_R3", name, "d3_rainbow_trout", "full"): _contrast(0.01, -0.02, 0.04, 0.6),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASET_IDS,
    )
    evaluations = evaluator.evaluate_all(protocol["gates"])
    assert evaluations["Gate_A"].result is False
    assert evaluations["Gate_B"].result is False
    assert evaluations["Gate_C"].result is False


@pytest.mark.unit
def test_gate_b_is_unreachable_at_five_clusters_even_with_a_tiny_p(protocol):
    """The pre-registered consequence, held as a test rather than as a caveat.

    D2 has five clusters and always will: it has five published groups. Gate_B's
    D2 term therefore cannot fire, whatever p-value is put in the table. v1.2
    states this in §13 `gate_reachability`; this test is what stops it being
    quietly forgotten the first time a tiny p-value appears.
    """
    name = "macro_f1"
    contrasts = {
        ("R0_vs_R2", name, "d2_beef_uncontrolled", "full"): _contrast(
            0.20, 0.15, 0.25, 5.45e-13, n_clusters=UNREACHABLE_CLUSTERS
        ),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASET_IDS,
    )
    assert evaluator.evaluate("G", "sig(R0, R2, macro_f1, D2)").result is False


@pytest.mark.unit
def test_the_protocols_narrative_rules_select_the_negative_branch(protocol):
    """N3 fires when the connectome loses, which is the branch most likely to be avoided."""
    name = "macro_f1"
    contrasts = {
        ("R0_vs_R4", name, "d3_rainbow_trout", "full"): _contrast(-0.12, -0.18, -0.06, 0.001),
        ("R0_vs_R2", name, "d2_beef_uncontrolled", "full"): _contrast(-0.05, -0.09, -0.01, 0.01),
        ("R0_vs_R2", name, "d3_rainbow_trout", "full"): _contrast(-0.09, -0.14, -0.04, 0.002),
        ("R0_vs_R3", name, "d3_rainbow_trout", "full"): _contrast(0.01, -0.02, 0.04, 0.6),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASET_IDS,
    )
    fired = [
        rule["id"]
        for rule in protocol["narrative_adjustment_rules"]["list"]
        if evaluator.evaluate(rule["id"], rule["trigger_expression"]).result
    ]
    assert fired == ["N3"]


@pytest.mark.unit
def test_the_protocols_equivalence_branch_fires_on_a_tie(protocol):
    """N1 is what protects against over-claiming when the rewired control matches."""
    name = "macro_f1"
    contrasts = {
        ("R0_vs_R2", name, "d2_beef_uncontrolled", "full"): _contrast(0.04, 0.01, 0.08, 0.01),
        ("R0_vs_R2", name, "d3_rainbow_trout", "full"): _contrast(0.001, -0.005, 0.006, 0.9),
        ("R0_vs_R3", name, "d3_rainbow_trout", "full"): _contrast(0.13, 0.07, 0.19, 0.0005),
        ("R0_vs_R4", name, "d3_rainbow_trout", "full"): _contrast(0.10, 0.05, 0.15, 0.001),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASET_IDS,
    )
    fired = [
        rule["id"]
        for rule in protocol["narrative_adjustment_rules"]["list"]
        if evaluator.evaluate(rule["id"], rule["trigger_expression"]).result
    ]
    assert "N1" in fired
    assert "N3" not in fired
