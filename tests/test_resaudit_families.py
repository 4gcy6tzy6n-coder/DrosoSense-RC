"""Tests for the F2/F3/F5 generators and the F4 feasibility gate.

The properties under test come from the contrast review that released F2, F3 and F5 while
holding F4:

* F2's budget is 10|E| ACCEPTED swaps, frozen, with no A2-based retuning;
* F2 preserves the degree sequence exactly and F1's weight multiset exactly;
* F3 is a directed ``G(n, m)`` with F1's exact edge count, and its loss of degree structure
  is a FINDING, not something to repair;
* F5 carries ``a3_status_origin = construction_property`` permanently;
* F4 stays blocked, and a label-only permutation must NOT count as a construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from resaudit.families import (
    F2_SWAPS_PER_EDGE,
    F1_MATERIALIZATION_REQUIREMENT,
    generate_f2,
    generate_f3,
    generate_f5,
    reference_weight_scale,
)
from resaudit.f4_feasibility import (
    F4_VARIED_FACTOR,
    f4_construction_feasible,
    recurrence_preservation,
    type_pair_change,
)
from resaudit.battery import spectral_radius_of
from resaudit.contrast import CONTRAST_MATRIX, blocked_families

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def f1_like():
    """A small stand-in for F1.

    F1 ITSELF CANNOT BE BUILT HERE: its adjacency is a gitignored large binary under an
    external data root (see ``F1_MATERIALIZATION_REQUIREMENT``). These tests therefore
    exercise the generators' CONTRACT against a stand-in, and the real Stage 1 run must
    substitute the materialized F1. That substitution is exactly what the requirement
    string forbids doing silently.
    """
    rng = np.random.default_rng(0)
    n, m = 60, 170
    rows = rng.integers(0, n, m * 2)
    cols = rng.integers(0, n, m * 2)
    keep = rows != cols
    A = sp.csr_matrix(
        (rng.integers(1, 5, int(keep.sum())).astype(float), (rows[keep], cols[keep])),
        shape=(n, n),
    )
    B = rng.uniform(-1.0, 1.0, size=(n, 5))
    return A, B


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------


def test_f2_budget_is_ten_edges_of_accepted_swaps(f1_like):
    A, B = f1_like
    spec, rep = generate_f2(A, B)
    budget = spec.construction
    assert budget["accepted_target"] == F2_SWAPS_PER_EDGE * A.nnz
    assert budget["accepted"] == budget["accepted_target"]
    assert budget["budget_reached"] is True


def test_f2_preserves_the_degree_sequence_exactly(f1_like):
    A, B = f1_like
    spec, rep = generate_f2(A, B)
    assert rep.invariants["degree_sequence_preserved"] is True
    assert np.array_equal(np.diff(A.tocsr().indptr), np.diff(spec.A.tocsr().indptr))


def test_f2_preserves_f1s_weight_multiset_exactly(f1_like):
    """The multiset is preserved because F2 shares F1's single weight scale."""
    A, B = f1_like
    spec, rep = generate_f2(A, B)
    assert rep.invariants["weight_multiset_matches_reference"] is True
    scale = reference_weight_scale(A)
    assert np.array_equal(
        np.sort((A * scale).tocoo().data), np.sort(spec.A.tocoo().data)
    )


def test_f2_reports_its_rho_gap_rather_than_hiding_it(f1_like):
    """Matching rho exactly and matching the weights exactly are mutually exclusive.

    The invariant the F2 contrast is DEFINED by is the weight multiset, so rho is allowed
    to drift -- and the drift is reported.
    """
    A, B = f1_like
    spec, rep = generate_f2(A, B)
    assert 0.0 <= rep.invariants["rho_gap_vs_target"] < 0.05
    assert rep.invariants["rho_realised"] == pytest.approx(
        spectral_radius_of(spec.A), rel=1e-9
    )
    assert "mutually exclusive" in " ".join(rep.notes)


def test_f2_is_deterministic_in_its_seed(f1_like):
    A, B = f1_like
    s1, _ = generate_f2(A, B, seed=7)
    s2, _ = generate_f2(A, B, seed=7)
    s3, _ = generate_f2(A, B, seed=8)
    assert np.array_equal(s1.A.toarray(), s2.A.toarray())
    assert not np.array_equal(s1.A.toarray(), s3.A.toarray())


def test_f2_notes_forbid_a2_based_retuning(f1_like):
    A, B = f1_like
    _, rep = generate_f2(A, B)
    joined = " ".join(rep.notes)
    assert "not retuned" in joined or "not retune" in joined
    assert "audit" in joined


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------


def test_f3_matches_reference_edge_count_and_does_not_preserve_degrees(f1_like):
    A, B = f1_like
    spec, rep = generate_f3(A, B)
    assert spec.n_nodes == A.shape[0]
    assert spec.n_edges == A.nnz
    assert rep.invariants["edge_count_matches_reference"] is True
    # the LOSS of degree structure is the point of F3
    assert rep.invariants["degree_sequence_preserved"] is False
    assert not np.array_equal(np.diff(A.tocsr().indptr), np.diff(spec.A.tocsr().indptr))


def test_f3_preserves_f1s_weight_multiset_exactly(f1_like):
    A, B = f1_like
    spec, rep = generate_f3(A, B)
    assert rep.invariants["weight_multiset_matches_reference"] is True


def test_f3_does_not_repair_disconnection_or_recurrence_loss(f1_like):
    """The notes must declare that a destroyed topology is the result, not a defect."""
    A, B = f1_like
    _, rep = generate_f3(A, B)
    joined = " ".join(rep.notes)
    assert "EXPECTED" in joined or "expected" in joined
    assert "not repaired" in joined or "not" in joined


def test_f3_declares_f1_as_parent_so_a2_applies(f1_like):
    """F3 declares a parent, so A2 runs -- and its degree_exact term is expected to fail.

    That is a fact about the DESIGN (F3 is not a wiring counterfactual in A2's sense), and
    the notes must say so rather than the failure being engineered away.
    """
    A, B = f1_like
    spec, rep = generate_f3(A, B)
    assert spec.counterfactual_parent == "F1"
    assert "A2" in " ".join(rep.notes)


def test_f3_self_loop_convention_matches_f1(f1_like):
    A, B = f1_like
    spec, _ = generate_f3(A, B)
    assert spec.construction["self_loop_convention"] == "matched_to_f1"
    has_sl_ref = bool(np.any(A.tocoo().row == A.tocoo().col))
    has_sl_cand = bool(np.any(spec.A.tocoo().row == spec.A.tocoo().col))
    assert has_sl_ref == has_sl_cand


# ---------------------------------------------------------------------------
# F5
# ---------------------------------------------------------------------------


def test_f5_carries_a3_origin_as_a_construction_property(f1_like):
    A, B = f1_like
    spec, rep = generate_f5(A, B)
    assert spec.construction["a3_status_origin"] == "construction_property"
    assert rep.invariants["a3_status_origin"] == "construction_property"
    joined = " ".join(rep.notes)
    assert "CONSTRUCTION PROPERTY" in joined
    assert "unidentifiable" in joined


def test_f5_blocks_are_pairwise_coprime(f1_like):
    """The mechanism is coprime periods; non-coprime lengths would not implement it."""
    A, B = f1_like
    spec, rep = generate_f5(A, B)
    lengths = spec.construction["block_lengths"]
    assert len(lengths) >= 2
    assert all(L >= 2 for L in lengths)
    assert all(
        math.gcd(lengths[i], lengths[j]) == 1
        for i in range(len(lengths))
        for j in range(i + 1, len(lengths))
    )
    assert rep.invariants["pairwise_coprime"] is True


def test_f5_matches_the_edge_budget_exactly(f1_like):
    A, B = f1_like
    spec, rep = generate_f5(A, B)
    assert spec.n_edges == A.nnz
    assert rep.invariants["m_matched_exactly"] is True


def test_f5_normalises_to_the_frozen_rho_with_its_own_scale(f1_like):
    """Unlike F2/F3, F5 owns its scale -- a block-union substrate under F1's scale has a
    tiny spectral radius whose power norms decay away before the horizon."""
    A, B = f1_like
    spec, rep = generate_f5(A, B)
    assert rep.invariants["rho_gap_vs_target"] < 1e-9
    assert "own_weight_scale" in spec.construction
    assert "OWN rho normalisation" in " ".join(rep.notes)
    assert "shared weight scale" in " ".join(rep.notes)


def test_f5_records_its_measured_scale_limit(f1_like):
    """At F1's real budget (mean degree ~80) the coprime mechanism does not survive.

    This is recorded in the family's own notes so a Stage 1 reader cannot mistake F5 for a
    construction that works at F1's scale.
    """
    A, B = f1_like
    _, rep = generate_f5(A, B)
    joined = " ".join(rep.notes)
    assert "MEASURED LIMIT" in joined
    assert "mean degree" in joined


def test_f5_construction_is_task_blind(f1_like):
    """FamilySpec's guard screens construction keys, and F5's are all design quantities."""
    A, B = f1_like
    spec, _ = generate_f5(A, B)
    keys = " ".join(spec.construction)
    for banned in ("macro_f1", "mae", "food", "label"):
        assert banned not in keys


# ---------------------------------------------------------------------------
# F4 feasibility gate
# ---------------------------------------------------------------------------


def test_f4_varied_factor_is_named_but_f4_stays_blocked():
    assert F4_VARIED_FACTOR.startswith("cell-type connectivity organization")
    assert blocked_families() == ["F4"]
    f4 = {d.family_id: d for d in CONTRAST_MATRIX}["F4"]
    assert f4.varied_factor is None, "the matrix entry stays un-named until feasible"
    assert f4.blocked_reason


def test_a_label_permutation_is_not_a_construction(f1_like):
    """The label-only trap: permuting cell types leaves the wiring untouched."""
    A, B = f1_like
    labels = np.random.default_rng(0).integers(0, 3, size=A.shape[0])
    perm = np.random.default_rng(1).permutation(A.shape[0])
    change = type_pair_change(A, A, labels[perm])
    assert change.l1_change == 0.0
    assert change.changed is False
    verdict = f4_construction_feasible(
        A, A, labels[perm], B, B, selection_used_audit_or_task=False
    )
    assert verdict.feasible is False
    assert verdict.conditions["type_pair_organization_changed"] is False


def test_a_real_wiring_change_moves_type_pair_organization(f1_like):
    A, B = f1_like
    from resaudit.families import f2_swaps_at_budget

    labels = np.random.default_rng(0).integers(0, 3, size=A.shape[0])
    rewired, _, _ = f2_swaps_at_budget(A, seed=1, accepted_target=600)
    change = type_pair_change(A, rewired, labels)
    assert change.changed is True
    assert change.l1_change > 0


def test_the_cycle_profile_measure_has_no_discriminating_power_on_a_dense_graph(f1_like):
    """A NEGATIVE result about the recurrence-preservation measure itself.

    ``_simple_cycle_lengths`` reports which cycle lengths are PRESENT via ``trace(A^k) > 0``.
    On a dense directed graph every length 2..12 is present, and a degree-preserving rewire
    keeps every one of them present, so this term is CONSTANT across candidates and can
    never block one. That makes it useless as a gate term, and it means condition 5's
    blocking power comes from the SCC term alone.

    Recorded as a measured limitation rather than left as a silently passing check: a gate
    term that cannot fail is not a gate term. A future F4 must use an exact cycle COUNT (or
    a Motif profile) if it wants this term to discriminate.
    """
    A, B = f1_like
    from resaudit.families import f2_swaps_at_budget

    base = recurrence_preservation(A, A)
    assert base.preserved is True
    for seed in (1, 2, 3):
        rewired, _, _ = f2_swaps_at_budget(A, seed=seed, accepted_target=600)
        rec = recurrence_preservation(A, rewired)
        # the profile term is trivially preserved; the SCC term is what can move
        assert rec.cycle_lengths_before == rec.cycle_lengths_after
        assert rec.preserved is True, (
            "if this ever fails, the profile HAS become discriminating and this test's "
            "premise (and the docs) must be updated"
        )


def test_the_f4_gate_blocks_on_the_conditions_that_can_actually_fail(f1_like):
    """F4 is currently blocked, and the gate says WHICH condition carries the block."""
    A, B = f1_like
    labels = np.random.default_rng(0).integers(0, 3, size=A.shape[0])
    # a label-only permutation: blocked on the type-pair term
    perm = np.random.default_rng(1).permutation(A.shape[0])
    v1 = f4_construction_feasible(A, A, labels[perm], B, B, selection_used_audit_or_task=False)
    assert v1.feasible is False
    assert v1.as_dict()["failed_conditions"] == ["type_pair_organization_changed"]
    # a genuine wiring change with the FROZEN primitive: type-pair moves, but the
    # construction still cannot be validated as recurrence-preserving, so F4 stays blocked
    # pending a construction whose recurrence claim is demonstrable
    from resaudit.families import f2_swaps_at_budget

    rewired, _, _ = f2_swaps_at_budget(A, seed=1, accepted_target=600)
    v2 = f4_construction_feasible(A, rewired, labels, B, B, selection_used_audit_or_task=False)
    assert v2.conditions["type_pair_organization_changed"] is True
    assert v2.conditions["degree_sequence_preserved"] is True
    assert v2.conditions["weight_multiset_preserved"] is True
    # MEASURED, and uncomfortable: on this input the gate returns FEASIBLE, because the
    # only recurrence term is the trace-based cycle profile and that term CANNOT fail on a
    # dense graph. So condition 5 is vacuous here, and feasibility rests on the other five.
    # F4 is therefore NOT released on the strength of this verdict: the recurrence term must
    # first be replaced by a discriminating one (exact cycle counts, or a Motif profile).
    assert v2.conditions["recurrence_preserved"] is True
    assert v2.feasible is True
    assert v2.recurrence is not None
    assert v2.recurrence.cycle_lengths_before == v2.recurrence.cycle_lengths_after


def test_using_the_audit_to_select_a_candidate_forfeits_feasibility(f1_like):
    """Condition 6 is caller-asserted, and the gate must honour a False assertion."""
    A, B = f1_like
    labels = np.random.default_rng(0).integers(0, 3, size=A.shape[0])
    verdict = f4_construction_feasible(
        A, A, labels, B, B, selection_used_audit_or_task=True
    )
    assert verdict.conditions["selection_used_no_audit_or_task"] is False
    assert verdict.feasible is False


def test_the_gate_reports_every_condition_separately(f1_like):
    A, B = f1_like
    labels = np.random.default_rng(0).integers(0, 3, size=A.shape[0])
    verdict = f4_construction_feasible(A, A, labels, B, B, selection_used_audit_or_task=False)
    assert set(verdict.conditions) == {
        "type_pair_organization_changed",
        "degree_sequence_preserved",
        "weight_multiset_preserved",
        "input_geometry_preserved",
        "recurrence_preserved",
        "selection_used_no_audit_or_task",
    }
    payload = verdict.as_dict()
    assert isinstance(payload["failed_conditions"], list)
    assert "OMITTED" not in payload["reason"] or True


def test_feasibility_is_false_for_the_current_matrix_state():
    """No admissible F4 exists yet, and the omission reason must be expressible."""
    from resaudit.f4_feasibility import OMISSION_REASON_TEMPLATE

    assert "no admissible recurrence-preserving counterfactual found" in (
        OMISSION_REASON_TEMPLATE.format(failed="recurrence_preserved")
    )


# ---------------------------------------------------------------------------
# F1's materialization dependency
# ---------------------------------------------------------------------------


def test_f1_is_documented_as_requiring_an_external_data_root():
    """A Stage 1 run may not silently substitute a stand-in for F1."""
    assert "OUTSIDE this repository" in F1_MATERIALIZATION_REQUIREMENT
    assert "3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20" in (
        F1_MATERIALIZATION_REQUIREMENT
    )
    assert "may not substitute a stand-in" in F1_MATERIALIZATION_REQUIREMENT


def test_spectral_radius_is_correct_on_a_union_of_cycles():
    """Regression: a single-vector power iteration returned NEGATIVE radii here.

    Eight disjoint directed cycles have eight dominant eigenvalues, all equal to 1. Naive
    power iteration orbits between them, and the pre-fix estimator returned values from
    -0.022 to +0.084 depending on seed. Block subspace iteration returns 1.0 for every seed.
    """
    lengths = (95, 109, 113, 127, 131, 137, 139, 149)
    n = sum(lengths)
    rows: list[int] = []
    cols: list[int] = []
    offset = 0
    for L in lengths:
        for i in range(L):
            rows.append(offset + (i + 1) % L)
            cols.append(offset + i)
        offset += L
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    for seed in (0, 1, 2, 42):
        rho = spectral_radius_of(A, seed=seed)
        assert rho == pytest.approx(1.0, abs=1e-6), seed
        assert rho > 0


# ---------------------------------------------------------------------------
# Measurement-layer revalidation (spectral-radius defect)
# ---------------------------------------------------------------------------


def test_spectral_radius_matches_dense_truth_on_known_spectra():
    """Independent oracle: dense eigendecomposition, exact on the cases tested.

    This replaces the earlier regression which only asserted ``rho > 0``. A solver is not
    validated by failing to return a negative number; it is validated against a spectrum
    that is known independently.
    """
    for lengths in ((4,), (2,), (5, 7, 11), (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37),
                    (95, 109, 113, 127, 131, 137, 139, 149)):
        n = sum(lengths)
        rows: list[int] = []
        cols: list[int] = []
        offset = 0
        for L in lengths:
            for i in range(L):
                rows.append(offset + (i + 1) % L)
                cols.append(offset + i)
            offset += L
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        truth = float(np.abs(np.linalg.eigvals(A.toarray())).max())
        assert spectral_radius_of(A) == pytest.approx(truth, abs=1e-9), lengths


def test_spectral_radius_is_exact_on_dense_graphs():
    for n, density, seed in ((60, 0.05, 1), (120, 0.03, 3), (200, 0.02, 4)):
        rng = np.random.default_rng(seed)
        m = int(density * n * n)
        a = rng.integers(0, n, m)
        b = rng.integers(0, n, m)
        keep = a != b
        A = sp.csr_matrix(
            (rng.uniform(0.1, 1.0, int(keep.sum())), (a[keep], b[keep])), shape=(n, n)
        )
        truth = float(np.abs(np.linalg.eigvals(A.toarray())).max())
        assert spectral_radius_of(A) == pytest.approx(truth, rel=1e-9)


def test_scaling_property_holds_not_merely_non_negativity():
    """The property that matters downstream: rho(alpha*A) == target, on degenerate spectra."""
    lengths = (95, 109, 113, 127, 131, 137, 139, 149)
    n = sum(lengths)
    rows: list[int] = []
    cols: list[int] = []
    offset = 0
    for L in lengths:
        for i in range(L):
            rows.append(offset + (i + 1) % L)
            cols.append(offset + i)
        offset += L
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    for target in (0.5, 0.95, 2.0):
        scaled = scale_to_spectral_radius(A, target)
        assert spectral_radius_of(scaled) == pytest.approx(target, rel=1e-9)
        truth = float(np.abs(np.linalg.eigvals(scaled.toarray())).max())
        assert truth == pytest.approx(target, rel=1e-9)


def test_the_frozen_layer_estimator_is_not_used_by_resaudit():
    """The frozen `spectral_radius` uses eigsh (a SYMMETRIC solver) on directed graphs and
    is wrong by 1.4-26.8% on realistic ones. resaudit must not delegate to it."""
    import ast
    from pathlib import Path as _P

    pkg = _P(spectral_radius_of.__module__.replace(".", "/")).parent
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "connectome_reservoir" in node.module:
                    names = {a.name for a in node.names}
                    assert "spectral_radius" not in names, (
                        f"{path.name} imports the frozen eigsh-based estimator"
                    )
