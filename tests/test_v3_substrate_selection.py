"""v3 substrate selection — the declared rules and the one invariant that must hold.

Pre-registration: `docs/v3_preregistration.md`. The tests here pin the *mechanical* parts:

* the selection rule reads only ``S1``, ``S2_participation_ratio`` and ``n_nodes``;
* **changing S3 cannot change the selected candidate** (the owner's constraint 4, so a
  diagnostic can never leak into the ranking later);
* the stop-loss fires exactly when no candidate reaches the gate, and the gate is not
  lowered to admit the best available candidate;
* the tie-break order is higher S2 participation ratio, then smaller N, with **no
  tolerance-based "approximately tied" judgement**;
* S2 uses LEFT eigenvectors (or left Schur vectors) and reports the conditioning;
* S1 is measured with one fixed rule: same Din, same K, same tolerance, same effective-rank
  definition, and ``B`` never rescaled per candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.substrate_scores import (  # noqa: E402
    KRYLOV_K,
    NUMERICAL_TOLERANCE,
    eigenmode_score,
    henrici_departure,
    krylov_score,
    spectral_score,
)
from ops.audit.v3_substrate_selection import SELECTION_GATE_FACTOR, select_candidate  # noqa: E402


def _row(cid: str, s1: float, s2: float = 1.0, n: int = 1000, gate: float = 10.0) -> dict:
    return {
        "candidate_id": cid, "S1": s1, "S2_participation_ratio": s2,
        "n_nodes": n, "gate": gate,
    }


# ---------------------------------------------------------------------------
# The selection rule
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_stop_loss_fires_when_no_candidate_clears_the_gate():
    rows = [_row("S0", 5.7), _row("S1", 8.1), _row("S2", 9.7)]
    out = select_candidate(rows)
    assert out["stop_loss"] is True
    assert out["selected"] is None
    assert out["eligible"] == []
    assert "STOPS" in out["reason"] or "stops" in out["reason"]


@pytest.mark.unit
def test_the_gate_is_not_lowered_to_admit_the_best_available_candidate():
    """9.999 against a gate of 10 is a STOP, not a selection."""
    out = select_candidate([_row("S0", 9.999), _row("S1", 9.5)])
    assert out["stop_loss"] is True
    assert out["selected"] is None


@pytest.mark.unit
def test_selection_is_max_s1_among_the_eligible():
    rows = [_row("S0", 5.7), _row("S1", 8.1), _row("S2", 9.7), _row("S3", 12.4), _row("S4", 14.0)]
    out = select_candidate(rows)
    assert out["selected"] == "S4"
    assert out["eligible"] == ["S4", "S3"]
    assert out["stop_loss"] is False


@pytest.mark.unit
def test_tie_breaks_by_s2_then_by_smaller_n_and_uses_exact_ordering():
    # identical S1 -> higher S2 participation ratio wins
    out = select_candidate([_row("A", 12.0, s2=2.0), _row("B", 12.0, s2=5.0)])
    assert out["selected"] == "B"
    # identical S1 and S2 -> smaller N wins
    out = select_candidate([_row("A", 12.0, s2=3.0, n=900), _row("B", 12.0, s2=3.0, n=800)])
    assert out["selected"] == "B"
    # a difference far below any plausible "approximately equal" threshold still orders
    out = select_candidate([_row("A", 12.0), _row("B", 12.0 + 1e-12)])
    assert out["selected"] == "B", "exact float ordering; no tolerance is introduced"


# ---------------------------------------------------------------------------
# The invariant: S3 cannot influence selection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_changing_s3_while_s1_s2_and_n_are_fixed_cannot_change_the_selection():
    """The owner's constraint 4, as an executable test.

    The selection engine's inputs are dicts with only the fields it reads; a S3 field on the
    same dict must be invisible. This test constructs two candidate sets that differ ONLY in
    their S3 payloads and asserts the selection is identical -- so a future code path that
    accidentally reads a diagnostic would fail here.
    """
    base = [_row("S0", 5.7), _row("S1", 12.4), _row("S2", 12.4, s2=1.0)]
    with_s3_a = [dict(r, S3={"S3_spectral_radius": 0.9}) for r in base]
    with_s3_b = [dict(r, S3={"S3_spectral_radius": 99.0}) for r in base]
    assert select_candidate(with_s3_a) == select_candidate(with_s3_b)

    from ops.audit.v3_substrate_selection import select_candidate as engine
    import inspect

    source = inspect.getsource(engine)
    assert "S3" not in source, (
        "the selection engine's body must not mention any S3 field: a diagnostic that the "
        "engine can name is a diagnostic that can leak into the ranking"
    )


@pytest.mark.unit
def test_the_gate_factor_is_the_pre_registered_two_times_din():
    assert SELECTION_GATE_FACTOR == 2.0
    # and the gate scales with Din rather than being a hard-coded number
    assert select_candidate([_row("A", 9.9, gate=10.0)])["stop_loss"] is True
    assert select_candidate([_row("A", 9.9, gate=8.0)])["stop_loss"] is False


# ---------------------------------------------------------------------------
# S1's measurement discipline
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_s1_is_effective_rank_of_the_krylov_block_with_the_declared_depth():
    """``K`` is fixed at 16; the block has ``(K+1)*Din`` columns.

    The numerical rank is monotone in K (more powers can only add directions). The
    participation ratio is NOT, and this test records that as a measured property rather
    than assuming otherwise: the singular values of the concatenated block span a dynamic
    range of ``~rho^K``, so the pre-registered effective rank at large K is dominated by
    whichever power has the largest norm. Both quantities are reported for every candidate
    so the sensitivity is visible.
    """
    rng = np.random.default_rng(0)
    n = 60
    a = sp.csr_matrix((rng.random((n, n)) < 0.1).astype(float))
    b = rng.standard_normal((n, 5))
    short = krylov_score(a, b, K=2)
    long = krylov_score(a, b, K=KRYLOV_K)
    assert short.K == 2 and long.K == KRYLOV_K
    assert long.theoretical_columns == (KRYLOV_K + 1) * 5
    assert long.numerical_rank_relative >= short.numerical_rank_relative
    assert long.numerical_rank <= min(n, (KRYLOV_K + 1) * 5)
    assert len(long.power_norms) == KRYLOV_K + 1
    # the power norms grow like rho^k on an unnormalised A, which is exactly why the
    # participation ratio of the raw block is scale-dominated
    assert long.power_norms[-1] > long.power_norms[0]
    assert NUMERICAL_TOLERANCE == 1e-10


@pytest.mark.unit
def test_s1_does_not_rescale_b():
    """A per-candidate rescale of B would compare input scales instead of substrates."""
    a = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    b = np.array([[1.0], [0.0]])
    base = krylov_score(a, b, K=4).S1
    scaled = krylov_score(a, b * 1000.0, K=4).S1
    assert base == pytest.approx(scaled, rel=1e-9)


@pytest.mark.unit
def test_s1_reports_the_capacity_diagnostics():
    rng = np.random.default_rng(1)
    n = 100
    a = sp.csr_matrix((rng.random((n, n)) < 0.2).astype(float))
    b = rng.standard_normal((n, 5))
    payload = krylov_score(a, b, K=8).as_dict()
    assert payload["theoretical_columns"] == 9 * 5
    assert 0.0 <= payload["krylov_capacity_used"] <= 1.0
    assert payload["S1_per_node"] == pytest.approx(payload["effective_rank"] / n)


# ---------------------------------------------------------------------------
# S2's left-eigenvector basis
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_s2_uses_a_left_basis_and_reports_its_conditioning():
    rng = np.random.default_rng(2)
    n = 40
    a = sp.csr_matrix((rng.random((n, n)) < 0.15).astype(float) * rng.integers(1, 9, (n, n)))
    b = rng.standard_normal((n, 3))
    score = eigenmode_score(a, b)
    assert score.basis in ("left_eigenvectors", "left_schur_vectors")
    assert score.mode_count > 0
    assert score.participation_ratio >= 1.0
    assert score.mode_energy_entropy >= 0.0
    assert np.isfinite(score.eigenvector_condition_number)
    payload = score.as_dict()
    assert payload["S2_basis"] == score.basis


@pytest.mark.unit
def test_non_normality_is_zero_for_a_normal_matrix_and_positive_otherwise():
    normal = np.array([[0.0, 1.0], [-1.0, 0.0]])  # rotation: normal
    assert henrici_departure(normal) == pytest.approx(0.0, abs=1e-12)
    non_normal = np.array([[1.0, 5.0], [0.0, 1.0]])
    assert henrici_departure(non_normal) > 0.0


# ---------------------------------------------------------------------------
# S3's diagnostics
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_s3_reports_the_declared_diagnostics_and_is_labelled_report_only():
    rng = np.random.default_rng(3)
    n = 30
    a = sp.csr_matrix((rng.random((n, n)) < 0.2).astype(float))
    s3 = spectral_score(a)
    for key in (
        "S3_spectral_radius",
        "S3_eigenvalue_modulus_entropy",
        "S3_singular_value_spread",
        "S3_henrici_non_normality",
        "S3_transient_amplification_proxy",
    ):
        assert key in s3, key
    assert "REPORT-ONLY" in s3["S3_note"]
    assert s3["S3_spectral_radius"] > 0.0


# ---------------------------------------------------------------------------
# Candidate provenance
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_candidate_ids_and_the_provenance_contract():
    from drososense.connectome_candidates import CANDIDATE_IDS, CandidateError, generate_candidate

    assert CANDIDATE_IDS == ("S0", "S1", "S2", "S3", "S4")
    with pytest.raises(CandidateError, match="unknown candidate"):
        generate_candidate("S9", raw_graph=sp.csr_matrix((4, 4)), root_ids=np.arange(4),
                           annotation=None, din=2, target_n=4)


@pytest.mark.unit
def test_the_provenance_records_answer_why_a_node_was_admitted():
    """Constraint 5: the record must let a machine ask 'why was this node added?'."""
    from drososense.connectome_candidates import CandidateSubstrate

    provenance_fields = set(CandidateSubstrate.__dataclass_fields__)
    assert {
        "candidate_id", "node_indices", "root_ids", "adjacency", "input_rows", "provenance"
    } == provenance_fields

    # the declared provenance keys the generators must produce (checked structurally here so
    # the contract is pinned without needing the delivered graph)
    required = {
        "selection_rule", "seed_populations", "added_populations", "frontier_rule",
        "cap_rule", "typed_composition", "admission_reason",
        "node_list_sha256", "edge_list_sha256",
    }
    import inspect

    from drososense.connectome_candidates import _provenance

    body = inspect.getsource(_provenance)
    for key in required:
        assert f'"{key}"' in body, f"provenance must record {key}"