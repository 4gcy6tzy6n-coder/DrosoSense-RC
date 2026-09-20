"""Gate and narrative-rule evaluation, and the safety of the evaluator itself.

Protocol v1 stated its gates as prose. Protocol v1.1 states them as boolean
expressions over a fixed predicate vocabulary, and these tests hold the
evaluator to two things: that it computes what the protocol says, and that it
cannot be talked into executing anything else.
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
DATASETS = ["D1", "D2", "D3"]
CONDITIONS = ["dropout_p0.3", "noise_s0.1", "train10pct", "train25pct"]


def _symbols() -> dict:
    return build_symbols(MODELS, METRICS, DATASETS, CONDITIONS)


def _metrics() -> dict:
    return {
        "macro_f1": {"direction": "maximize", "margin": 0.02, "alpha": 0.05},
        "balanced_accuracy": {"direction": "maximize", "margin": 0.02, "alpha": 0.05},
        "mae": {"direction": "minimize", "margin": 0.05, "alpha": 0.05},
    }


def _contrast(delta: float, ci_low: float, ci_high: float, p_holm: float, n_pairs: int = 50) -> dict:
    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_holm": p_holm,
        "n_pairs": n_pairs,
    }


def _evaluator(contrasts: dict, model_params: dict | None = None, available=None) -> GateEvaluator:
    return GateEvaluator(
        contrasts=contrasts,
        metrics=_metrics(),
        model_params=model_params or {},
        symbols=_symbols(),
        available_datasets=available if available is not None else DATASETS,
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
# The predicates compute what the protocol says
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sig_needs_both_a_corrected_p_and_an_interval_off_zero():
    """A p-value alone is not enough; the effect has to point the right way."""
    improving = {("R0_vs_R2", "macro_f1", "D3", "full"): _contrast(0.05, 0.01, 0.09, 0.004)}
    assert _evaluator(improving).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is True

    not_significant = {("R0_vs_R2", "macro_f1", "D3", "full"): _contrast(0.05, 0.01, 0.09, 0.4)}
    assert _evaluator(not_significant).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is False

    interval_spans_zero = {
        ("R0_vs_R2", "macro_f1", "D3", "full"): _contrast(0.05, -0.01, 0.09, 0.004)
    }
    assert _evaluator(interval_spans_zero).evaluate("G", "sig(R0, R2, macro_f1, D3)").result is False


@pytest.mark.unit
def test_minimising_metrics_are_favoured_by_lower_values():
    """MAE improves downward, so the favourable interval is the negative one."""
    improving = {("R0_vs_GRU", "mae", "D3", "full"): _contrast(-0.3, -0.5, -0.1, 0.001)}
    assert _evaluator(improving).evaluate("G", "sig(R0, GRU, mae, D3)").result is True
    worsening = {("R0_vs_GRU", "mae", "D3", "full"): _contrast(0.3, 0.1, 0.5, 0.001)}
    assert _evaluator(worsening).evaluate("G", "sig(R0, GRU, mae, D3)").result is False


@pytest.mark.unit
def test_equiv_is_tost_not_a_failure_to_reject():
    """A wide interval spanning zero proves nothing about equivalence."""
    wide = {("R0_vs_R2", "macro_f1", "D3", "full"): _contrast(0.0, -0.5, 0.5, 0.9)}
    assert _evaluator(wide).evaluate("G", "equiv(R0, R2, macro_f1, D3)").result is False

    tight = {("R0_vs_R2", "macro_f1", "D3", "full"): _contrast(0.0, -0.005, 0.005, 0.9)}
    assert _evaluator(tight).evaluate("G", "equiv(R0, R2, macro_f1, D3)").result is True


@pytest.mark.unit
def test_noninferiority_uses_the_declared_margin():
    """Gate_A's tolerance is the protocol's margin, not a number in the gate text."""
    inside = {("R0_vs_R4", "macro_f1", "D3", "full"): _contrast(-0.01, -0.015, 0.0, 0.2)}
    assert _evaluator(inside).evaluate("G", "noninferior(R0, R4, macro_f1, D3)").result is True

    outside = {("R0_vs_R4", "macro_f1", "D3", "full"): _contrast(-0.3, -0.4, -0.2, 0.001)}
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
        ("R0_vs_R4", "macro_f1", "D3", "dropout_p0.3"): _contrast(0.06, 0.02, 0.1, 0.002),
    }
    evaluator = _evaluator(contrasts)
    # Conditions are quoted because `dropout_p0.3` is not a Python identifier.
    expression = 'sig_any(R0, R4, macro_f1, [D2, D3], ["noise_s0.1", "dropout_p0.3"])'
    assert evaluator.evaluate("G", expression).result is True


@pytest.mark.unit
def test_equivalent_all_requires_every_dataset():
    """equiv_all is a conjunction, so one wide interval breaks it."""
    partial = {
        ("R0_vs_R4", "macro_f1", "D2", "full"): _contrast(0.0, -0.005, 0.005, 0.9),
        ("R0_vs_R4", "macro_f1", "D3", "full"): _contrast(0.0, -0.4, 0.4, 0.9),
    }
    assert _evaluator(partial).evaluate("G", "equiv_all(R0, R4, macro_f1, [D2, D3])").result is False


@pytest.mark.unit
def test_unavailable_tracks_the_datasets_that_were_obtained():
    """N5 fires on a missing dataset, and only on a missing one."""
    evaluator = _evaluator({}, available=["D2", "D3"])
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
    for dataset in ("D2", "D3"):
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
        available_datasets=DATASETS,
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
        ("R0_vs_R4", name, "D2", "full"): _contrast(-0.10, -0.15, -0.05, 0.001),
        ("R0_vs_R4", name, "D3", "full"): _contrast(-0.12, -0.18, -0.06, 0.001),
        ("R0_vs_R2", name, "D2", "full"): _contrast(-0.05, -0.09, -0.01, 0.01),
        ("R0_vs_R2", name, "D3", "full"): _contrast(-0.09, -0.14, -0.04, 0.002),
        ("R0_vs_R3", name, "D3", "full"): _contrast(0.01, -0.02, 0.04, 0.6),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params={"R0": 1000, "GRU": 50000},
        symbols=_symbols(),
        available_datasets=DATASETS,
    )
    evaluations = evaluator.evaluate_all(protocol["gates"])
    assert evaluations["Gate_A"].result is False
    assert evaluations["Gate_B"].result is False
    assert evaluations["Gate_C"].result is False


@pytest.mark.unit
def test_the_protocols_narrative_rules_select_the_negative_branch(protocol):
    """N3 fires when the connectome loses, which is the branch most likely to be avoided."""
    name = "macro_f1"
    contrasts = {
        ("R0_vs_R4", name, "D3", "full"): _contrast(-0.12, -0.18, -0.06, 0.001),
        ("R0_vs_R2", name, "D2", "full"): _contrast(-0.05, -0.09, -0.01, 0.01),
        ("R0_vs_R2", name, "D3", "full"): _contrast(-0.09, -0.14, -0.04, 0.002),
        ("R0_vs_R3", name, "D3", "full"): _contrast(0.01, -0.02, 0.04, 0.6),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        symbols=_symbols(),
        available_datasets=DATASETS,
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
        ("R0_vs_R2", name, "D2", "full"): _contrast(0.04, 0.01, 0.08, 0.01),
        ("R0_vs_R2", name, "D3", "full"): _contrast(0.001, -0.005, 0.006, 0.9),
        ("R0_vs_R3", name, "D3", "full"): _contrast(0.13, 0.07, 0.19, 0.0005),
        ("R0_vs_R4", name, "D3", "full"): _contrast(0.10, 0.05, 0.15, 0.001),
    }
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        symbols=_symbols(),
        available_datasets=DATASETS,
    )
    fired = [
        rule["id"]
        for rule in protocol["narrative_adjustment_rules"]["list"]
        if evaluator.evaluate(rule["id"], rule["trigger_expression"]).result
    ]
    assert "N1" in fired
    assert "N3" not in fired
