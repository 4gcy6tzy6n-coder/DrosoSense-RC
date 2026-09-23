"""Invariant tests for the ResAudit-Food gate engine.

These are the properties the audit runner must hold BEFORE any family is generated. They
are properties of the ENGINE, not of any reservoir, which is why the runner is built
first:

1. each of A1-A5 can PASS and FAIL;
2. ``not_applicable`` is never folded into ``FAIL``;
3. renaming or reordering families does not change a verdict;
4. the construction path cannot see a task/food metric;
5. the A3 threshold is always the frozen ``2 * Din``;
6. A4/A5 are evaluated only after A3 passes;
7. the no-food-data rule is machine-enforced, not a documented promise.

Where a criterion turns out NOT to be independently reachable, the test says so and
asserts the reachable behaviour instead of manufacturing a case. Two such facts were
measured while calibrating the toys (see ``resaudit/toys.py``): A1's SCC term is nearly
implied by its isolated term, and A4's sign is unstable in its own probe. Both are
asserted below as documented behaviour.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from resaudit import battery, criteria, toys
from resaudit.battery import (
    A2_NO_PARENT_REASON,
    A4_A5_SKIPPED_REASON,
    audit_a1,
    audit_a2,
    audit_a3,
    audit_all,
    audit_family,
    stage1_table,
)
from resaudit.criteria import (
    A3_FACTOR,
    A3_KRYLOV_K,
    A4_MIN,
    A5_FACTOR,
    ALL_CRITERIA,
    CriterionResult,
    CriterionState,
    evaluate_ge,
    not_applicable,
)
from resaudit.family import FamilySpec, TaskBlindnessError

pytestmark = pytest.mark.unit

_PACKAGE = Path(battery.__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spec(
    family_id: str,
    A: sp.spmatrix,
    B: np.ndarray,
    *,
    parent: str | None = None,
    label: str | None = None,
    role: str = "test",
    construction: dict | None = None,
) -> FamilySpec:
    A = sp.csr_matrix(A)
    B = np.asarray(B, dtype=np.float64)
    return FamilySpec(
        family_id=family_id,
        label=label or f"toy {family_id}",
        role=role,
        A=A,
        B=B,
        n_nodes=A.shape[0],
        n_edges=int(A.nnz),
        din=B.shape[1],
        counterfactual_parent=parent,
        seed=1,
        construction=construction or {"kind": "toy"},
    )


def _a3_pass() -> FamilySpec:
    A, B, _ = toys.two_coprime_cycles()
    return _spec("a3pass", A, B)


def _a3_fail() -> FamilySpec:
    A, B, _ = toys.hairline_fail_cycle()
    return _spec("a3fail", A, B)


# ---------------------------------------------------------------------------
# Invariant 1 -- A1-A5 each reach PASS and FAIL
# ---------------------------------------------------------------------------


def test_the_battery_is_exactly_a1_to_a5_with_the_signed_thresholds():
    assert ALL_CRITERIA == ("A1", "A2", "A3", "A4", "A5")
    assert criteria.PRIMARY_GATE == "A3"
    assert criteria.ADMISSION_CRITERIA == ("A1", "A2")
    # A4 was retired as a gate by pre-registration decision rule 6.2; A5 is the only
    # remaining downstream gate. See docs/resaudit_a4_construct_validity.md.
    assert criteria.DOWNSTREAM_CRITERIA == ("A5",)
    assert criteria.DESCRIPTIVE_ONLY_CRITERIA == ("A4",)
    assert "A4" not in criteria.BLOCKING_CRITERIA
    assert set(criteria.BLOCKING_CRITERIA) | {"A4"} == set(ALL_CRITERIA)
    # verbatim transcription of docs/resaudit_food_preregistration.md Section 3
    assert criteria.A1_SCC_FRACTION_MIN == 0.90
    assert criteria.A1_ISOLATED_FRACTION_MAX == 0.02
    assert criteria.A1_MEAN_OUT_DEGREE_MIN == 2.0
    assert criteria.A2_OVERLAP_MAX == 0.20
    assert criteria.A3_FACTOR == 2.0
    assert criteria.A3_KRYLOV_K == 16
    assert criteria.A4_MIN == 0.20
    assert criteria.A5_FACTOR == 1.5


def test_a1_passes_on_a_dense_enough_ring():
    A, B, _ = toys.k_out_ring(n=40, k=2)
    r = audit_a1(_spec("a1pass", A, B))
    assert r.state is CriterionState.PASS, r.reason
    assert r.detail["terms_met"] == {
        "scc_largest_fraction": True,
        "isolated_fraction": True,
        "mean_out_degree": True,
    }


def test_a1_fails_on_mean_out_degree_alone():
    """A plain directed ring: SCC and isolated terms pass, mean out-degree does not."""
    A, B, _ = toys.k_out_ring(n=40, k=1)
    r = audit_a1(_spec("a1deg", A, B))
    assert r.state is CriterionState.FAIL
    assert r.detail["terms_met"] == {
        "scc_largest_fraction": True,
        "isolated_fraction": True,
        "mean_out_degree": False,
    }
    assert "mean_out_degree" in r.reason


def test_a1_fails_on_largest_scc_alone():
    A, B, meta = toys.two_component_graph(n_each=20, k=2)
    r = audit_a1(_spec("a1scc", A, B))
    assert r.state is CriterionState.FAIL
    assert r.detail["terms_met"] == {
        "scc_largest_fraction": False,
        "isolated_fraction": True,
        "mean_out_degree": True,
    }
    assert r.detail["scc_largest_fraction"] == pytest.approx(0.50)
    assert meta["fails_term"] == "scc_largest_fraction"


def test_a1_isolated_term_drives_the_verdict_across_its_boundary():
    """1 % isolated passes; 3.9 % fails, with the other two terms still passing."""
    A_ok, B_ok, _ = toys.isolated_boundary_graph(n_ring=99, k=3, n_isolated=1)
    r_ok = audit_a1(_spec("iso_ok", A_ok, B_ok))
    assert r_ok.state is CriterionState.PASS
    assert r_ok.detail["isolated_fraction"] == pytest.approx(0.01)
    assert all(r_ok.detail["terms_met"].values())

    A_bad, B_bad, _ = toys.isolated_boundary_graph(n_ring=99, k=3, n_isolated=4)
    r_bad = audit_a1(_spec("iso_bad", A_bad, B_bad))
    assert r_bad.state is CriterionState.FAIL
    assert r_bad.detail["terms_met"] == {
        "scc_largest_fraction": True,
        "isolated_fraction": False,
        "mean_out_degree": True,
    }
    assert r_bad.detail["isolated_fraction"] > criteria.A1_ISOLATED_FRACTION_MAX


def test_a1_scc_term_is_nearly_implied_by_the_isolated_term():
    """Documented, not a defect: the two A1 terms are not independently reachable.

    Out-degree-0 nodes are singleton SCCs, so they depress ``largest_SCC/N``. Clearing
    the isolated ceiling caps that depression below the SCC floor.
    """
    assert 1.0 - criteria.A1_ISOLATED_FRACTION_MAX > criteria.A1_SCC_FRACTION_MIN
    # raising the isolated count depresses mean out-degree too, at small scale
    A_small, B_small, _ = toys.isolated_boundary_graph(n_ring=40, k=2, n_isolated=4)
    r_small = audit_a1(_spec("iso_small", A_small, B_small))
    assert r_small.detail["terms_met"]["mean_out_degree"] is False
    assert r_small.detail["terms_met"]["isolated_fraction"] is False


def test_a3_passes_and_fails_on_designed_toys():
    A_pass, B_pass, meta = toys.two_coprime_cycles()
    r_pass, score = audit_a3(_spec("a3pass", A_pass, B_pass))
    assert r_pass.state is CriterionState.PASS, r_pass.as_dict()
    assert score["effective_rank"] == pytest.approx(meta["measured"]["D_eff"], abs=5e-4)

    A_fail, B_fail, meta_fail = toys.hairline_fail_cycle()
    r_fail, score_fail = audit_a3(_spec("a3fail", A_fail, B_fail))
    assert r_fail.state is CriterionState.FAIL, r_fail.as_dict()
    assert score_fail["effective_rank"] == pytest.approx(meta_fail["measured"]["D_eff"], abs=5e-4)


def test_a3_hairline_case_is_still_a_failure():
    """`close` is not a pass: 1.9983 against 2.0 must be FAIL."""
    r, _ = audit_a3(_a3_fail())
    assert r.state is CriterionState.FAIL
    assert r.value / r.threshold > 0.99


def test_a3_fails_loudly_with_no_input_at_all():
    A, B, _ = toys.zero_input_graph()
    r, score = audit_a3(_spec("a3zero", A, B))
    assert r.state is CriterionState.FAIL
    assert score["effective_rank"] == pytest.approx(0.0, abs=1e-12)


def test_a3_passes_on_a_nilpotent_chain_documented_surprise():
    """A feedforward chain is NOT uncontrollable under this metric. Recorded, not hidden."""
    A, B, meta = toys.nilpotent_chain()
    r, score = audit_a3(_spec("nilp", A, B))
    assert r.state is CriterionState.PASS
    assert score["effective_rank"] == pytest.approx(meta["measured"]["D_eff"], abs=5e-4)
    assert score["effective_rank"] > 8 * r.threshold


def test_a4_and_a5_values_are_well_defined_when_reached():
    spec = _a3_pass()
    a4, a5, diag = battery.audit_a4_a5(spec)
    assert a4.criterion == "A4" and a5.criterion == "A5"
    assert a4.state in (CriterionState.PASS, CriterionState.FAIL)
    assert a5.state in (CriterionState.PASS, CriterionState.FAIL)
    assert a4.detail["gate_status"] == "descriptive_only"
    assert a4.detail["caveat"] == criteria.A4_DESCRIPTIVE_CAVEAT
    m_recur, m_ff = diag["M_recurrent"], diag["M_feedforward"]
    expected = (m_recur - m_ff) / m_recur if m_recur > 0 else 0.0
    assert a4.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert diag["D_eff_state"] == pytest.approx(a5.value)


def test_a5_can_pass_and_fail():
    A_ok, B_ok, _ = toys.two_coprime_cycles()
    _, a5_ok, diag_ok = battery.audit_a4_a5(_spec("a5ok", A_ok, B_ok))
    assert a5_ok.state is CriterionState.PASS
    assert diag_ok["D_eff_state"] > a5_ok.threshold

    A_bad, B_bad, meta = toys.zero_input_graph()
    _, a5_bad, diag_bad = battery.audit_a4_a5(_spec("a5bad", A_bad, B_bad))
    assert a5_bad.state is CriterionState.FAIL
    assert diag_bad["D_eff_state"] == pytest.approx(0.0, abs=1e-12)
    assert meta["designed_verdicts"]["A5"] == "FAIL"


def test_a4_sign_is_unstable_in_its_own_probe_documented_finding():
    """A4's contribution is small, sign-unstable, and non-monotone in recurrent coupling.

    Measured while calibrating: on bounded random drive the metric is O(0.01) at both
    N=40 and N=1000, and the signed contribution is NEGATIVE for graphs with genuine
    recurrence. The engine must report that as FAIL rather than rescaling it into a pass,
    so this test pins the behaviour and the magnitude.
    """
    A, B, _ = toys.two_coprime_cycles()
    a4, _, diag = battery.audit_a4_a5(_spec("a4", A, B))
    assert abs(diag["M_recurrent"]) < 0.1
    assert abs(diag["M_feedforward"]) < 0.1
    assert abs(a4.value) < 10.0
    # the recurrent graph does not beat its own feed-forward control here
    assert a4.value < A4_MIN
    assert a4.state is CriterionState.FAIL
    # and a graph with MORE recurrent coupling scores LOWER, not higher
    A_more, B_more, _ = toys.k_out_ring(n=40, k=2)
    a4_more, _, diag_more = battery.audit_a4_a5(_spec("a4more", A_more, B_more))
    assert a4_more.value < a4.value


def test_a3_verdict_depends_on_the_spectral_radius_normalization():
    """Documented battery property: A3 as written is normalized-scale dependent.

    ``krylov_score`` does not rescale ``A``. The Frobenius norms of ``A^k B`` therefore
    grow like ``rho^k``, so a substrate with ``rho`` appreciably above 1 has ONE dominant
    singular value in the Krylov block and measures ``D_eff ~ 1`` -- an effective "FAIL"
    that reflects numerical scale, not controllability. Rescaling the same wiring to
    ``rho ~ 1`` restores a well-conditioned block and a PASS. Two families with identical
    wiring and different scaling therefore receive different A3 verdicts, so the audit
    protocol must declare a single normalization rule and apply it to every family.
    """
    rng = np.random.default_rng(1)
    n, m, din = 100, 400, 4
    rows = rng.integers(0, n, m)
    cols = rng.integers(0, n, m)
    keep = rows != cols
    A = sp.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    B = np.zeros((n, din))
    for c in range(din):
        B[rng.integers(0, n), c] = 1.0

    rho_raw = float(np.abs(np.linalg.eigvals(A.toarray())).max())
    assert rho_raw > 3.0
    r_raw, score_raw = audit_a3(_spec("raw", A, B))
    assert r_raw.state is CriterionState.FAIL
    assert score_raw["effective_rank"] < 1.05  # collapsed, not merely low

    A_scaled = A * (1.0 / rho_raw)
    r_scaled, score_scaled = audit_a3(_spec("scaled", A_scaled, B))
    assert r_scaled.state is CriterionState.PASS
    assert score_scaled["effective_rank"] > r_scaled.threshold

    # the power norms are the mechanism, and they are visible in the report
    assert score_raw["power_norms"][3] > 10.0 * score_raw["power_norms"][0]
    assert score_scaled["power_norms"][3] < score_scaled["power_norms"][0]


def test_memory_contribution_definition_matches_the_signed_expression():
    assert battery._memory_contribution(0.5, 0.25) == pytest.approx(0.5)
    assert battery._memory_contribution(0.0, 0.0) == 0.0
    assert battery._memory_contribution(0.4, 0.4) == pytest.approx(0.0)


def test_threshold_comparisons_are_inclusive():
    assert evaluate_ge("A3", 2.0, 2.0, expression="x").state is CriterionState.PASS
    assert evaluate_ge("A3", 1.999999, 2.0, expression="x").state is CriterionState.FAIL
    assert criteria.evaluate_le("A1", 0.02, 0.02, expression="x").state is CriterionState.PASS
    assert criteria.evaluate_le("A1", 0.020001, 0.02, expression="x").state is CriterionState.FAIL


def test_unknown_criterion_id_is_rejected():
    with pytest.raises(ValueError, match="unknown criterion"):
        CriterionResult(
            criterion="A6",
            state=CriterionState.PASS,
            applicable=True,
            value=1.0,
            threshold=0.5,
            expression="x",
        )


# ---------------------------------------------------------------------------
# Invariant 2 -- not_applicable never becomes FAIL
# ---------------------------------------------------------------------------


def test_a2_is_not_applicable_without_a_counterfactual_parent():
    A, B, _ = toys.two_coprime_cycles()
    r = audit_a2(_spec("noparent", A, B), None)
    assert r.state is CriterionState.NOT_APPLICABLE
    assert r.applicable is False
    assert r.failed is False
    assert r.reason == A2_NO_PARENT_REASON
    assert "never a counterfactual" in r.reason
    assert r.value is None and r.threshold is None


def test_a2_applicable_and_passing_when_the_edge_set_is_fully_remixed():
    """A reversal permutation preserves degrees and weights and changes every edge."""
    A0, B0, _ = toys.k_out_ring(n=20, k=1)
    perm = (-np.arange(20)) % 20
    A_child = sp.csr_matrix(A0[perm][:, perm])
    parent = _spec("P", A0, B0)
    child = _spec("C", A_child, B0, parent="P")
    r = audit_a2(child, parent)
    assert r.applicable is True
    assert r.state is CriterionState.PASS, r.as_dict()
    assert r.detail["edge_overlap"] == pytest.approx(0.0)
    assert all(r.detail["exact_terms"].values())


def test_a2_fails_on_overlap_while_exactness_holds():
    """A shift permutation preserves degrees and weights but keeps every edge: FAIL."""
    A0, B0, _ = toys.k_out_ring(n=20, k=1)
    perm = np.roll(np.arange(20), 1)
    A_child = sp.csr_matrix(A0[perm][:, perm])
    parent = _spec("P", A0, B0)
    child = _spec("C", A_child, B0, parent="P")
    r = audit_a2(child, parent)
    assert r.applicable is True
    assert r.state is CriterionState.FAIL
    assert r.detail["edge_overlap"] == pytest.approx(1.0)
    assert all(r.detail["exact_terms"].values())
    assert "edge_overlap" in r.reason


def test_a2_fails_when_the_pair_is_identical():
    A, B, _ = toys.two_coprime_cycles()
    parent = _spec("P_same", A, B)
    child = _spec("C_same", A, B, parent="P_same")
    r = audit_a2(child, parent)
    assert r.state is CriterionState.FAIL
    assert r.detail["edge_overlap"] == pytest.approx(1.0)


def test_a2_named_parent_without_a_supplied_spec_is_not_applicable():
    A, B, _ = toys.two_coprime_cycles()
    r = audit_a2(_spec("orphan", A, B, parent="missing"), None)
    assert r.state is CriterionState.NOT_APPLICABLE
    assert "was not supplied" in r.reason
    assert r.detail["parent_supplied"] is False


def test_a2_mismatched_parent_is_a_hard_error():
    A, B, _ = toys.two_coprime_cycles()
    spec = _spec("child", A, B, parent="declared")
    wrong = _spec("other", A, B)
    with pytest.raises(ValueError, match="declared parent"):
        audit_a2(spec, wrong)


def test_not_applicable_requires_a_reason():
    with pytest.raises(ValueError, match=r"A2: NOT_APPLICABLE requires a reason"):
        not_applicable("A2", reason="")


def test_applicable_flag_cannot_contradict_state():
    with pytest.raises(ValueError, match="contradicts"):
        CriterionResult(
            criterion="A2",
            state=CriterionState.NOT_APPLICABLE,
            applicable=True,
            value=None,
            threshold=None,
            expression="x",
            reason="r",
        )
    with pytest.raises(ValueError, match="contradicts"):
        CriterionResult(
            criterion="A2",
            state=CriterionState.PASS,
            applicable=False,
            value=1.0,
            threshold=0.5,
            expression="x",
        )


def test_a2_not_applicable_does_not_block_qualification():
    """With A2 the ONLY not-applicable criterion, it must not manufacture a failure.

    A4/A5 are measured here so that they are real verdicts rather than skip-stubs; the
    property under test is that ``NOT_APPLICABLE`` is not ``FAIL``.
    """
    audit = audit_family(_a3_pass(), parent=None, measure_downstream=True)
    assert audit.results["A2"].state is CriterionState.NOT_APPLICABLE
    assert audit.results["A2"].state.blocks_qualification is False
    assert audit.food_eligible is True
    real = {k: v for k, v in audit.results.items() if k != "A2"}
    assert all(v.state is not CriterionState.NOT_APPLICABLE for v in real.values())
    # qualified is decided by the EVALUATED criteria only, and A2 does not fail
    assert audit.construct_qualified == all(v.state is not CriterionState.FAIL for v in real.values())


def test_a_fully_construct_qualified_toy_exists():
    """The battery can return a qualified family: A1+A3 pass, A2/A4/A5 not applicable.

    Note the construction requirement this pins down: a substrate must be scaled to a
    sane spectral radius before A3 means anything (see
    ``test_a3_verdict_depends_on_the_spectral_radius_normalization``). The
    ``two_coprime_cycles`` toy clears A3 but FAILS A1 on mean out-degree (1.0), so it
    cannot be the qualified example.
    """
    A, B, meta = toys.battery_ready_ring()
    assert meta["rho_before"] > 2.0
    spec = _spec("qualified", A, B)
    audit = audit_family(spec, parent=None, measure_downstream=False)
    states = {k: v.state for k, v in audit.results.items()}
    assert states["A1"] is CriterionState.PASS
    assert states["A3"] is CriterionState.PASS
    assert states["A2"] is CriterionState.NOT_APPLICABLE
    assert states["A4"] is CriterionState.NOT_APPLICABLE
    assert states["A5"] is CriterionState.NOT_APPLICABLE
    assert audit.food_eligible is True
    assert audit.construct_qualified is True


def test_not_construct_qualified_when_an_admission_gate_fails():
    """A3 passing is necessary but not sufficient: a failed A1 blocks qualification."""
    spec = _spec("a3_only", *toys.two_coprime_cycles()[:2])
    audit = audit_family(spec, parent=None, measure_downstream=False)
    assert audit.results["A3"].state is CriterionState.PASS
    assert audit.results["A1"].state is CriterionState.FAIL
    assert audit.food_eligible is True   # A3 is what gates the food stage
    assert audit.construct_qualified is False      # but the family is not qualified


def test_a4_a5_skipped_is_not_applicable_and_states_the_observability_limit():
    audit = audit_family(_a3_fail(), measure_downstream=True)
    assert audit.downstream_evaluated is False
    for cid in ("A4", "A5"):
        r = audit.results[cid]
        assert r.state is CriterionState.NOT_APPLICABLE
        assert r.failed is False
        assert r.reason == A4_A5_SKIPPED_REASON
    assert "UNOBSERVABLE" in A4_A5_SKIPPED_REASON


def test_stage1_table_distinguishes_not_applicable_from_not_evaluated():
    """`N/A` (never applicable) and `—` (not evaluated) must NOT render the same."""
    parent = _a3_pass()
    child = _a3_fail()
    # give the parent an A2 so both markers appear: N/A from no parent, — from A3 failure
    audits = audit_all([parent, child], measure_downstream=False)
    table = stage1_table(audits)
    rows = [ln for ln in table.splitlines() if ln.startswith("| a3")]
    assert len(rows) == 2
    # both rows show a real verdict, and A2 is N/A on both (no parent declared)
    for row in rows:
        assert "N/A" in row
        assert "FAIL" in row
    # the em dash appears for a criterion that exists but was never evaluated: render
    # an audit whose A4 is missing entirely, which is what a partial report looks like
    audit = audits[0]
    partial = battery.FamilyAudit(
        family=audit.family,
        results={k: v for k, v in audit.results.items() if k != "A4"},
        downstream_evaluated=False,
        downstream_not_evaluated_reason="stripped for rendering test",
    )
    data_row = [
        ln for ln in stage1_table([partial]).splitlines() if ln.startswith("| a3pass")
    ][0]
    assert "—" in data_row
    assert "N/A" in data_row
    assert "N/A" != "—"


# ---------------------------------------------------------------------------
# Invariant 3 -- names and order do not change verdicts
# ---------------------------------------------------------------------------


def test_verdict_is_invariant_to_family_name_and_label():
    A, B, _ = toys.two_coprime_cycles()
    a = audit_family(_spec("F1", A, B, label="connectome"), measure_downstream=False)
    b = audit_family(_spec("ZZZ", A, B, label="something else"), measure_downstream=False)
    for cid in ALL_CRITERIA:
        ra, rb = a.results[cid], b.results[cid]
        assert ra.state is rb.state, cid
        assert ra.value == rb.value, cid
        assert ra.threshold == rb.threshold, cid
    assert a.food_eligible == b.food_eligible
    assert a.construct_qualified == b.construct_qualified


def test_verdict_is_invariant_to_audit_order():
    specs = [_spec("F5", *toys.two_coprime_cycles()[:2]), _a3_fail(), _spec("F2", *toys.nilpotent_chain()[:2])]
    forward = {a.family["family_id"]: a for a in audit_all(specs, measure_downstream=False)}
    reverse = {
        a.family["family_id"]: a for a in audit_all(list(reversed(specs)), measure_downstream=False)
    }
    assert set(forward) == set(reverse)
    for fid in forward:
        for cid in ALL_CRITERIA:
            assert forward[fid].results[cid].state is reverse[fid].results[cid].state
            assert forward[fid].results[cid].value == reverse[fid].results[cid].value


def test_wiring_hash_depends_on_matrices_and_not_on_identity():
    A, B, _ = toys.two_coprime_cycles()
    assert _spec("F1", A, B).wiring_hash() == _spec("F1", A, B).wiring_hash()
    assert _spec("F1", A, B).wiring_hash() != _spec("F1", A, B * 2.0).wiring_hash()
    assert _spec("F1", A, B).wiring_hash() == _spec("ZZZ", A, B).wiring_hash()


def test_duplicate_family_ids_are_rejected():
    A, B, _ = toys.two_coprime_cycles()
    with pytest.raises(ValueError, match="duplicate family_id"):
        audit_all([_spec("F1", A, B), _spec("F1", A, B)], measure_downstream=False)


# ---------------------------------------------------------------------------
# Invariant 4 -- the construction path cannot see a task/food metric
# ---------------------------------------------------------------------------


def test_family_spec_has_no_field_for_task_information():
    assert FamilySpec.__slots__ is not None
    for forbidden in ("macro_f1", "mae", "labels", "test_split", "food", "target"):
        assert not hasattr(FamilySpec, forbidden)
    spec = _spec("F5", *toys.two_coprime_cycles()[:2])
    with pytest.raises(AttributeError):
        spec.macro_f1 = 0.9  # type: ignore[attr-defined]


def test_construction_record_rejects_task_metric_keys():
    A, B, _ = toys.two_coprime_cycles()
    for bad_key in ("macro_f1", "mae", "mean_absolute_error", "n_labels", "food_split"):
        with pytest.raises(TaskBlindnessError):
            _spec("F5", A, B, construction={"kind": "x", bad_key: 1.0})


def test_construction_record_accepts_task_blind_keys():
    A, B, _ = toys.two_coprime_cycles()
    spec = _spec(
        "F5",
        A,
        B,
        construction={"kind": "coprime_blocks", "lengths": [2, 3, 5]},
    )
    assert spec.construction["kind"] == "coprime_blocks"


def test_battery_does_not_import_food_or_task_machinery():
    banned = ("herbdry", "sklearn", "pandas", "torch")
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0].lower()
                assert root not in banned, f"{path.name} imports {name!r}"


def test_battery_entry_points_take_only_sealed_specs():
    for fn in (audit_a1, audit_a3, audit_family):
        params = inspect.signature(fn).parameters
        assert "spec" in params
        assert "FamilySpec" in str(params["spec"].annotation)
    extra = set(inspect.signature(audit_family).parameters) - {
        "spec", "parent", "measure_downstream",
    }
    assert not extra


def test_probe_protocol_is_constant_and_task_free():
    proto = battery.probe_protocol()
    assert set(proto) == {
        "leak", "gain", "windows", "length", "seed", "washout",
        "input_scale", "krylov_K", "memory_lags",
    }
    assert proto["krylov_K"] == A3_KRYLOV_K
    assert proto["seed"] == battery.PROBE_SEED
    # the same protocol is reported on every run, so no family sees a private rule
    assert battery.probe_protocol() == proto


def test_probe_input_is_deterministic_and_family_independent():
    x1 = battery.probe_input(3)
    assert x1.shape == (battery.PROBE_WINDOWS, battery.PROBE_LENGTH, 3)
    assert np.array_equal(x1, battery.probe_input(3))
    assert np.array_equal(battery.probe_input(4)[:, :, :3], x1)


# ---------------------------------------------------------------------------
# Invariant 5 -- the A3 threshold is always the frozen 2 * Din
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("din", [1, 2, 3, 5])
def test_a3_threshold_is_exactly_two_times_din(din):
    n = 64
    A, _starts, n_nodes = toys._cycles((32, 32))
    assert n_nodes == n
    B = np.zeros((n, din))
    for c in range(din):
        B[c, c] = 1.0
    r, score = audit_a3(_spec(f"din{din}", A, B))
    assert r.threshold == pytest.approx(A3_FACTOR * din)
    assert r.threshold == pytest.approx(2.0 * din)
    assert score["Din"] == din
    assert score["K"] == A3_KRYLOV_K


def test_a3_expression_names_the_frozen_form():
    r, _ = audit_a3(_a3_pass())
    assert "A^16*B" in r.expression
    assert "2.0 * Din" in r.expression


def test_threshold_record_matches_the_frozen_constants():
    record = battery._threshold_record()
    assert record == {
        "A1_scc_fraction_min": 0.90,
        "A1_isolated_fraction_max": 0.02,
        "A1_mean_out_degree_min": 2.0,
        "A2_overlap_max": 0.20,
        "A3_factor": 2.0,
        "A3_K": 16,
        "A4_min": 0.20,
        "A5_factor": 1.5,
    }


def test_a3_factor_is_referenced_and_never_re_typed():
    """The gate literal appears once, in criteria.py, and nowhere as a raw product."""
    src = (_PACKAGE / "battery.py").read_text(encoding="utf-8")
    assert "A3_FACTOR" in src
    for bad in ("2.0 * din", "2 * din", "2.0*din", "2*din"):
        assert bad not in src, f"battery.py re-types the A3 gate as {bad!r}"


# ---------------------------------------------------------------------------
# Invariant 6 -- A4/A5 run only after A3 passes
# ---------------------------------------------------------------------------


def test_a4_a5_are_not_measured_when_a3_fails(monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("A4/A5 were evaluated despite A3 failing")

    monkeypatch.setattr(battery, "audit_a4_a5", _boom)
    audit = audit_family(_a3_fail(), measure_downstream=True)
    assert called["n"] == 0
    assert audit.downstream_evaluated is False
    assert audit.food_eligible is False
    assert audit.construct_qualified is False


def test_a4_a5_are_measured_when_a3_passes():
    audit = audit_family(_a3_pass(), measure_downstream=True)
    assert audit.results["A3"].state is CriterionState.PASS
    assert audit.downstream_evaluated is True
    for cid in ("A4", "A5"):
        assert audit.results[cid].applicable is True
        assert audit.results[cid].state in (CriterionState.PASS, CriterionState.FAIL)


def test_measure_downstream_false_skips_even_when_a3_passes(monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("downstream measured despite measure_downstream=False")

    monkeypatch.setattr(battery, "audit_a4_a5", _boom)
    audit = audit_family(_a3_pass(), measure_downstream=False)
    assert called["n"] == 0
    assert audit.results["A4"].state is CriterionState.NOT_APPLICABLE
    assert audit.results["A5"].state is CriterionState.NOT_APPLICABLE


def test_food_eligibility_tracks_a3_exactly():
    assert audit_family(_a3_pass(), measure_downstream=False).food_eligible is True
    assert audit_family(_a3_fail(), measure_downstream=False).food_eligible is False


def test_a3_clear_with_a5_failure_is_not_construct_qualified():
    """A5 is the remaining downstream gate: failing it blocks qualification."""
    audit = battery.FamilyAudit(
        family={"family_id": "stub"},
        results={
            "A3": CriterionResult(
                criterion="A3", state=CriterionState.PASS, applicable=True,
                value=10.0, threshold=10.0, expression="x",
            ),
            "A5": CriterionResult(
                criterion="A5", state=CriterionState.FAIL, applicable=True,
                value=0.1, threshold=A5_FACTOR, expression="x",
            ),
        },
        downstream_evaluated=True,
        downstream_not_evaluated_reason="",
    )
    assert audit.food_eligible is True
    assert audit.construct_qualified is False


def test_an_a4_failure_does_not_block_qualification():
    """A4 is DESCRIPTIVE ONLY: its number is reported, its verdict does not disqualify.

    This is the code-level consequence of the construct-validity audit. A test asserts it
    so that re-promoting A4 to a gate would be a visible change, not a silent one.
    """
    audit = battery.FamilyAudit(
        family={"family_id": "stub"},
        results={
            "A3": CriterionResult(
                criterion="A3", state=CriterionState.PASS, applicable=True,
                value=10.0, threshold=10.0, expression="x",
            ),
            "A4": CriterionResult(
                criterion="A4", state=CriterionState.FAIL, applicable=True,
                value=-1.02, threshold=A4_MIN, expression="x",
            ),
            "A5": CriterionResult(
                criterion="A5", state=CriterionState.PASS, applicable=True,
                value=20.0, threshold=A5_FACTOR, expression="x",
            ),
        },
        downstream_evaluated=True,
        downstream_not_evaluated_reason="",
    )
    assert audit.results["A4"].failed is True
    assert audit.food_eligible is True
    assert audit.construct_qualified is True
    # and the caveat travels with the number
    assert criteria.A4_DESCRIPTIVE_CAVEAT
    assert "decision rule 6.2" in criteria.A4_DESCRIPTIVE_CAVEAT


# ---------------------------------------------------------------------------
# Invariant 7 -- no-food-data is a machine guard
# ---------------------------------------------------------------------------

#: Named data artefacts and task metrics that must not appear in the package at all.
#: Matched with word boundaries so that legitimate prose ("feed-forward") is not a hit.
_FORBIDDEN_IN_SOURCE = (
    r"\bdata1\b", r"\bmacro_?f1\b", r"\bread_csv\b", r"\bload_data\b",
    r"\btest_split\b", r"\bsklearn\b", r"\bpandas\b", r"\bherbdry\b",
)


#: The guard declaration itself necessarily CONTAINS the forbidden names. The scan below
#: removes ``TaskBlindnessError.FORBIDDEN_KEY_PARTS`` before searching, so it measures
#: whether a food/task quantity is REFERENCED anywhere else -- not whether the guard
#: lists it. The guard's behaviour is checked separately by
#: ``test_construction_record_rejects_task_metric_keys``.
def _source_without_the_guard_list(text: str) -> str:
    """Remove the guard DECLARATION and any comment explaining it.

    ``family.py`` necessarily contains the forbidden names -- they ARE its rules -- so the
    scan removes the guard block plus contiguous comment lines that discuss it, and then
    looks for a genuine reference elsewhere.
    """
    out = text
    for name in ("FORBIDDEN_KEY_PARTS", "FORBIDDEN_KEY_WORDS"):
        block = re.search(rf"{name}[^\n]*=\s*\((?:[^)]*\))+", out, re.S)
        if block:
            out = out.replace(block.group(0), "")
    # drop comment lines that name the rules, since they are documentation of the guard
    out = "\n".join(
        line for line in out.split("\n") if not line.lstrip().startswith("#")
    )
    return out


def test_no_food_dataset_or_task_metric_is_referenced_in_the_package():
    for path in sorted(_PACKAGE.glob("*.py")):
        text = _source_without_the_guard_list(path.read_text(encoding="utf-8"))
        for pattern in _FORBIDDEN_IN_SOURCE:
            assert not re.search(pattern, text), f"{path.name} matches {pattern!r}"


def test_package_dependency_surface_is_numpy_scipy_and_the_frozen_package():
    allowed_roots = {
        "__future__", "sys", "hashlib", "json", "dataclasses", "enum", "pathlib",
        "typing", "ast", "re", "math", "argparse", "numpy", "scipy", "drososense",
        "resaudit",
    }
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for m in mods:
                root = m.split(".")[0]
                assert root in allowed_roots, f"{path.name} imports disallowed root {root!r}"


# ---------------------------------------------------------------------------
# Stage 1 table and report
# ---------------------------------------------------------------------------


def test_stage1_table_has_the_declared_columns():
    audits = audit_all([_a3_pass()], measure_downstream=False)
    header = stage1_table(audits).splitlines()[0]
    assert header == (
        "| Family | A1 | A2 | A3 Krylov | A4 Memory | A5 State | Food eligible "
        "| Construct qualified |"
    )


def test_stage1_table_shows_verdicts_without_misleading_rounding():
    """A hairline FAIL must not render as `FAIL (2.00>=2.00)`.

    The value is carried at full precision in ``as_dict``; the table carries only the
    verdict, so a strict comparison can never look self-contradictory.
    """
    audit = audit_family(_a3_fail(), measure_downstream=False)
    table = stage1_table([audit])
    assert "FAIL (2.00>=2.00)" not in table
    assert "(" not in table.split("\n")[-1]
    # the exact value survives in the machine-readable report
    a3 = audit.as_dict()["criteria"]["A3"]
    assert a3["value"] < a3["threshold"]
    assert a3["value"] == pytest.approx(1.9982683, abs=1e-6)


def test_stage1_json_is_serialisable_and_carries_provenance():
    audits = audit_all([_spec("F5", *toys.two_coprime_cycles()[:2]), _a3_fail()], measure_downstream=False)
    report = battery.stage1_json(audits, git_head="deadbeef")
    assert report["provenance"]["primary_gate"] == "A3"
    assert report["provenance"]["git_head"] == "deadbeef"
    assert report["provenance"]["probe_protocol"]["krylov_K"] == A3_KRYLOV_K
    assert report["provenance"]["thresholds"]["A3_factor"] == 2.0
    assert len(report["families"]) == 2
    assert "| Food eligible |" in report["stage1_table_markdown"]
    json.dumps(report)  # must be writable as-is

# ---------------------------------------------------------------------------
# The provenance chain: the original pre-registration is immutable
# ---------------------------------------------------------------------------

#: The pre-registration's blob hash as committed. Amendment 1 and amendment 2 record
#: findings WITHOUT editing this file, so that a later reader can see the original
#: commitment, the falsification and its timing, rather than a document quietly brought
#: into line with its results, A changed value here must be a deliberate, reviewed act.
PREREGISTRATION_BLOB_SHA1 = "350edc2b8cf88dbbc77b162ccd4a40a56ff5c71d"

_AMENDMENTS = (
    "resaudit_food_preregistration_amendment_1.md",
    "resaudit_food_preregistration_amendment_2.md",
)

_AUDIT_DOCS = ("resaudit_a4_construct_validity.md",)


def _repo_root() -> Path:
    return Path(battery.__file__).resolve().parent.parent


def test_original_preregistration_is_byte_identical_to_its_committed_form():
    """Amendments record; they never rewrite. The chain must stay checkable."""
    import hashlib
    import subprocess

    path = _repo_root() / "docs" / "resaudit_food_preregistration.md"
    assert path.exists(), "the pre-registration must exist"
    raw = path.read_bytes()
    on_disk = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert on_disk == PREREGISTRATION_BLOB_SHA1, (
        "docs/resaudit_food_preregistration.md has been modified. The pre-registration is "
        "immutable: record new findings in a further amendment instead of editing it."
    )
    # and it must be committed in that exact form
    out = subprocess.run(
        ["git", "-C", str(_repo_root()), "rev-parse", "HEAD:docs/resaudit_food_preregistration.md"],
        capture_output=True, text=True,
    )
    if out.returncode == 0 and out.stdout.strip():
        assert out.stdout.strip() == PREREGISTRATION_BLOB_SHA1


def test_the_amendment_chain_exists_and_is_append_only():
    """Amendment 1 and 2 are separate files; neither replaces the pre-registration."""
    docs = _repo_root() / "docs"
    for name in _AMENDMENTS:
        assert (docs / name).exists(), f"missing amendment {name}"
    numbers = sorted(
        int(m.group(1))
        for m in (
            re.search(r"preregistration_amendment_(\d+)\.md$", f.name)
            for f in docs.glob("resaudit_food_preregistration_amendment_*.md")
        )
        if m
    )
    assert numbers == list(range(1, len(numbers) + 1)), f"amendment numbering has a gap: {numbers}"


def test_amendment_2_records_the_retirement_and_the_no_substitution_rule():
    """The four commitments amendment 2 must carry, checked as text so they cannot drift."""
    text = (_repo_root() / "docs" / "resaudit_food_preregistration_amendment_2.md").read_text(
        encoding="utf-8"
    )
    # markdown emphasis, hard wraps and blockquote continuation markers must not defeat
    # a content check: `> ` at a line start joins into the sentence once flattened
    flat = " ".join(re.sub(r"[*`>]", "", text).split())
    assert "retired as a qualification gate" in flat
    assert "no longer used for qualification" in flat
    # the replacement framing, tolerant of markdown line wrapping
    assert "finite-horizon input-reachable state diversity and state expansion" in flat
    assert "post-calibration construct substitution" in flat
    assert "Blocking = {A1, A2, A3, A5}" in flat
    # timing must be stated, because it is what makes the withdrawal credible
    assert "before F1–F5" in flat or "before F1-F5" in flat


def test_a4_retirement_is_consistent_across_docs_and_code():
    """The docs, the constants, and the runner must agree that A4 does not gate."""
    text = (_repo_root() / "docs" / "resaudit_food_preregistration_amendment_2.md").read_text(
        encoding="utf-8"
    )
    flat = " ".join(re.sub(r"[*`>]", "", text).split())
    assert "A4_MIN = 0.20" in flat          # threshold retained in the record
    assert "not moved" in flat              # and not tuned
    assert criteria.A4_MIN == 0.20          # code agrees
    assert "A4" not in criteria.BLOCKING_CRITERIA
    assert "A4" in criteria.DESCRIPTIVE_ONLY_CRITERIA
    assert criteria.BLOCKING_CRITERIA == ("A1", "A2", "A3", "A5")


def test_stage1_report_carries_no_merged_pass_flag():
    """`Stage 1 passed` may never appear as a machine field."""
    audits = audit_all([_a3_pass()], measure_downstream=False)
    report = battery.stage1_json(audits)
    blob = json.dumps(report)
    for banned in ("stage1_passed", "stage_1_passed", "\"passed\"", "all_passed"):
        assert banned not in blob
    for fam in report["families"]:
        assert "food_eligible" in fam
        assert "construct_qualified" in fam

def test_contrast_matrix_doc_matches_the_enforced_matrix():
    """The published matrix must agree with the enforced one, or one of them is stale."""
    from resaudit.contrast import CONTRAST_MATRIX, blocked_families

    text = (_repo_root() / "docs" / "resaudit_contrast_matrix.md").read_text(encoding="utf-8")
    flat = " ".join(re.sub(r"[*`>]", "", text).split())
    for d in CONTRAST_MATRIX:
        assert d.family_id in flat, f"{d.family_id} missing from the published matrix"
        # role labels are written prose-style in the doc ("biological / reference"),
        # so compare on a normalization that ignores separators and punctuation
        norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
        assert norm(d.role) in norm(flat), (d.family_id, d.role)
    # the blocked family must be shown as blocked, not quietly omitted
    for fid in blocked_families():
        assert "NOT YET NAMED" in flat, f"{fid} is blocked but the doc does not say so"
    assert "instances_generated = 0" in flat
    assert "food_evaluation = HOLD" in flat
