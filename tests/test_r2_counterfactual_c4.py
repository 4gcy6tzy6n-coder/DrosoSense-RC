"""C4 — R2 as a TRUE wiring-only counterfactual (v2 D6, criteria C4.1-C4.8).

The v1 defect these tests are written against: ``make_degree_rewired`` rebuilds R2
from ``np.ones(n_edges)`` and rescales it, so R2 is a **uniform-weight graph**. R0-vs-R2
was therefore a JOINT contrast — wiring *and* weight structure — and the v1 conclusion
"the wiring does not matter" is not what that comparison measured.

Every criterion here is an exact equality or a pre-registered tolerance, and the
judgement head is flat on purpose:

```
degree_exact / global_weights_exact / per_source_weights_exact / input_population_same
median_in_strength_err <= 0.05 | edge_overlap <= 0.20 | build_time <= 600 s
mixing_declared
C4 = PASS iff all pass
```

The conservation claims are published as **hashes over sorted values**, so they cannot
be satisfied by a tolerance. ``build_time`` is compared against the signed 600 s
threshold, never against the 1 h hard cap: a counterfactual that is fair but too slow
to generate per seed is a construct failure, not a success with a caveat.
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

from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    ReservoirTopology,
    make_degree_rewired,
    make_degree_rewired_weight_preserving,
    rescale_to_spectral_radius,
    spectral_radius,
)
from drososense.reservoir.r2_counterfactual import (  # noqa: E402
    BUILD_TIME_BUDGET_S,
    IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT,
    MIXING_OVERLAP_LIMIT,
    RewireError,
    cell_type_pair_matrix,
    counterfactual_quality,
    degree_sequence_hash,
    multiset_hash,
    per_source_multiset_hash,
    verdict_header,
    weight_preserving_degree_rewire,
)


def _quantized_graph(n: int = 220, *, seed: int = 3, density: float = 0.07) -> sp.csr_matrix:
    """Integer weights, like the delivered substrate's synapse-count weights."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    weights = rng.integers(1, 21, size=(n, n)).astype(float) * mask
    np.fill_diagonal(weights, 0.0)
    return sp.csr_matrix(weights)


@pytest.fixture()
def r0_topology() -> ReservoirTopology:
    matrix = rescale_to_spectral_radius(_quantized_graph(), 0.9)
    n = int(matrix.shape[0])
    return ReservoirTopology(
        matrix=matrix,
        n_nodes=n,
        n_edges=int(matrix.nnz),
        spectral_radius=spectral_radius(matrix, seed=0),
        density=matrix.nnz / (n * n),
        kind="R0_real_fly",
        normalization="synthetic_c4",
    )


# ---------------------------------------------------------------------------
# C4.1 / C4.2 / C4.3 — the exact conservations, as hashes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c4_1_c4_2_c4_3_the_three_exact_conservations(r0_topology):
    rewired, report = weight_preserving_degree_rewire(r0_topology.matrix, seed=7)
    quality = counterfactual_quality(r0_topology.matrix, rewired, report)

    assert quality["C4.1_degree_sequences"]["pass"]
    observed = quality["C4.1_degree_sequences"]["observed"]
    assert observed["max_abs_in_degree_delta"] == 0
    assert observed["max_abs_out_degree_delta"] == 0
    assert observed["in_degree_hash_R0"] == observed["in_degree_hash_R2"]
    assert observed["out_degree_hash_R0"] == observed["out_degree_hash_R2"]

    assert quality["C4.2_global_weight_multiset"]["pass"]
    weights = quality["C4.2_global_weight_multiset"]["observed"]
    assert weights["multiset_hash_R0"] == weights["multiset_hash_R2"]
    assert weights["n_edges_R0"] == weights["n_edges_R2"]

    assert quality["C4.3_per_source_out_weight_multiset"]["pass"]
    per_source = quality["C4.3_per_source_out_weight_multiset"]["observed"]
    assert per_source["per_source_hash_R0"] == per_source["per_source_hash_R2"]


@pytest.mark.unit
def test_c4_2_the_v1_counterfactual_fails_the_weight_criterion(r0_topology):
    """The defect D6 repairs, pinned as a test rather than left in prose."""
    v1 = make_degree_rewired(r0_topology, seed=7)
    v2 = make_degree_rewired_weight_preserving(r0_topology, seed=7)

    r0_weights = np.sort(r0_topology.matrix.tocoo().data.astype(np.float64))
    v1_weights = np.sort(v1.matrix.tocoo().data.astype(np.float64))
    v2_weights = np.sort(v2.matrix.tocoo().data.astype(np.float64))

    assert not np.array_equal(v1_weights, r0_weights), "v1 R2 is a uniform-weight graph"
    assert len(np.unique(v1_weights)) == 1, "v1 writes a constant weight"
    assert np.array_equal(v2_weights, r0_weights), "v2 keeps the multiset exactly"
    assert multiset_hash(v1_weights) != multiset_hash(r0_weights)
    assert multiset_hash(v2_weights) == multiset_hash(r0_weights)
    # v1 also matches the spectral radius by rescaling, which is exactly what destroys
    # the multiset; v2 reports the difference instead of hiding it
    assert v1.spectral_radius == pytest.approx(r0_topology.spectral_radius)
    assert v2.counterfactual["quality"]["spectral_radius"]["ratio"] != pytest.approx(1.0)
    assert v2.counterfactual["quality"]["verdict"]["C4"] is True


@pytest.mark.unit
def test_the_hashes_are_multiset_hashes_not_order_hashes():
    a = np.array([3.0, 1.0, 2.0])
    b = np.array([2.0, 3.0, 1.0])
    assert multiset_hash(a) == multiset_hash(b)
    assert multiset_hash(a) != multiset_hash(np.array([3.0, 1.0, 1.0]))
    rows = np.array([0, 0, 1])
    assert per_source_multiset_hash(rows, np.array([2.0, 1.0, 5.0])) == (
        per_source_multiset_hash(rows, np.array([1.0, 2.0, 5.0]))
    )
    assert degree_sequence_hash(np.array([1, 2, 3])) != degree_sequence_hash(np.array([3, 2, 1]))


# ---------------------------------------------------------------------------
# C4.4 — the in-strength distribution, and why the second edge is weight-matched
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c4_4_weight_matching_makes_the_in_strength_error_exactly_zero(r0_topology):
    """An exchange of EQUAL weights moves no in-strength at all."""
    rewired, report = weight_preserving_degree_rewire(r0_topology.matrix, seed=11)
    quality = counterfactual_quality(r0_topology.matrix, rewired, report)
    observed = quality["C4.4_in_strength_median_relative_error"]["observed"]
    assert observed["median"] == pytest.approx(0.0)
    assert observed["p90"] == pytest.approx(0.0)
    assert observed["max"] == pytest.approx(0.0)
    assert quality["C4.4_in_strength_median_relative_error"]["pass"]
    # the tail is reported as a diagnostic beside the gate
    assert set(observed) == {"median", "p90", "max"}
    assert report.swaps_weight_matched_nearest == 0
    assert report.swaps_rejected_no_weight_partner == 0


@pytest.mark.unit
def test_c4_4_continuous_weights_still_pass_the_median_tolerance():
    """A graph with NO duplicate weights falls back to near weights, and reports it."""
    rng = np.random.default_rng(5)
    n = 200
    mask = rng.random((n, n)) < 0.10
    weights = rng.random((n, n)) * mask  # all distinct
    np.fill_diagonal(weights, 0.0)
    matrix = sp.csr_matrix(weights)
    rewired, report = weight_preserving_degree_rewire(matrix, seed=2, target_overlap=0.4)
    quality = counterfactual_quality(matrix, rewired, report)
    observed = quality["C4.4_in_strength_median_relative_error"]["observed"]
    # the fallback is used, and its size is visible rather than assumed away
    assert report.swaps_weight_matched_exact == 0
    assert report.swaps_weight_matched_nearest > 0
    assert np.isfinite(observed["median"])
    assert observed["p90"] >= observed["median"]


# ---------------------------------------------------------------------------
# C4.6 / C4.7 / C4.8 — mixing, time, and what must be reported
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c4_6_mixing_overlap_is_reached_and_the_curve_is_reported(r0_topology):
    rewired, report = weight_preserving_degree_rewire(r0_topology.matrix, seed=13)
    assert report.overlap_final <= MIXING_OVERLAP_LIMIT
    assert report.mixing_reached
    assert report.stopped_because == "target_overlap"
    # the curve, not just its endpoint: a reader must be able to see whether 20 % is a
    # plateau or an arbitrary stopping point
    curve = report.as_dict()["mixing_curve"]
    assert curve[0] == {"accepted_swaps": 0, "edge_overlap": 1.0}
    assert len(curve) >= 2
    assert all(
        later["edge_overlap"] <= earlier["edge_overlap"]
        for earlier, later in zip(curve, curve[1:])
    )
    assert curve[-1]["edge_overlap"] <= MIXING_OVERLAP_LIMIT


@pytest.mark.unit
def test_c4_7_the_budget_is_the_signed_600s_not_the_hard_cap():
    """600 s is the criterion; 1 h is only a cap. 'Fair but slow' is a FAIL."""
    assert BUILD_TIME_BUDGET_S == 600.0
    fast = verdict_header(
        degree_ok=True, weights_ok=True, per_source_ok=True, population_ok=True,
        median_relative_error=0.0, overlap=0.10, seconds=100.0, mixing_reached=True,
    )
    assert fast["C4"] is True
    slow = verdict_header(
        degree_ok=True, weights_ok=True, per_source_ok=True, population_ok=True,
        median_relative_error=0.0, overlap=0.10, seconds=1800.0, mixing_reached=True,
    )
    assert slow["checks"]["build_time"] is False
    assert slow["C4"] is False, "under the hard cap but over the criterion is a FAIL"


@pytest.mark.unit
def test_c4_7_the_chain_stops_at_its_time_budget_and_says_so(r0_topology):
    rewired, report = weight_preserving_degree_rewire(
        r0_topology.matrix, seed=17, time_budget_s=0.0, target_overlap=0.0
    )
    assert report.stopped_because == "time_budget"
    assert not report.mixing_reached
    # and the matrix it returns is still a valid counterfactual: same degrees, same
    # weights, just less mixed -- the failure is reported, not concealed
    quality = counterfactual_quality(r0_topology.matrix, rewired, report)
    assert quality["C4.1_degree_sequences"]["pass"]
    assert quality["C4.2_global_weight_multiset"]["pass"]
    assert quality["C4.6_mixing_overlap"]["pass"] is False
    assert quality["verdict"]["C4"] is False


@pytest.mark.unit
def test_c4_8_the_report_carries_swaps_retention_and_the_mixing_figure(r0_topology):
    _, report = weight_preserving_degree_rewire(r0_topology.matrix, seed=19)
    payload = report.as_dict()
    for field in (
        "swaps_attempted",
        "swaps_accepted",
        "acceptance_rate",
        "overlap_final",
        "original_edge_retention",
        "mixing_reached",
        "mixing_curve",
        "seconds",
        "stopped_because",
    ):
        assert field in payload, field
    assert 0.0 <= payload["acceptance_rate"] <= 1.0
    assert payload["original_edge_retention"] == pytest.approx(payload["overlap_final"])


# ---------------------------------------------------------------------------
# The verdict head, and the refusals
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "degree_exact",
        "global_weights_exact",
        "per_source_weights_exact",
        "input_population_same",
        "median_in_strength_err",
        "edge_overlap",
        "build_time",
        "mixing_declared",
    ],
)
def test_one_failed_line_fails_c4(field):
    passed = dict(
        degree_ok=True, weights_ok=True, per_source_ok=True, population_ok=True,
        median_relative_error=0.0, overlap=0.10, seconds=10.0, mixing_reached=True,
    )
    assert verdict_header(**passed)["C4"] is True
    broken = dict(passed)
    overrides = {
        "degree_exact": {"degree_ok": False},
        "global_weights_exact": {"weights_ok": False},
        "per_source_weights_exact": {"per_source_ok": False},
        "input_population_same": {"population_ok": False},
        "median_in_strength_err": {"median_relative_error": 0.06},
        "edge_overlap": {"overlap": 0.21},
        "build_time": {"seconds": 601.0},
        "mixing_declared": {"mixing_reached": False},
    }
    broken.update(overrides[field])
    verdict = verdict_header(**broken)
    assert verdict["checks"][field] is False
    assert verdict["C4"] is False
    assert IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT == 0.05
    assert MIXING_OVERLAP_LIMIT == 0.20


@pytest.mark.unit
def test_duplicate_edges_are_refused():
    """A duplicate would silently change the weight multiset the counterfactual keeps.

    It is refused on the INPUT, before ``tocsr()``: CSR sums duplicate entries, so by
    the time the chain sees the matrix the information is already gone.
    """
    duplicated = sp.coo_matrix(
        (np.ones(4), ([0, 0, 1, 2], [1, 1, 2, 0])), shape=(3, 3)
    )
    assert duplicated.nnz == 4, "the fixture really does carry the pair (0, 1) twice"
    with pytest.raises(RewireError, match="duplicate"):
        weight_preserving_degree_rewire(duplicated, seed=0)
    # the summed CSR form has no duplicates left to detect, and is accepted
    _, report = weight_preserving_degree_rewire(duplicated.tocsr(), seed=0)
    assert report.n_edges == 3


@pytest.mark.unit
def test_a_graph_with_nothing_to_swap_is_refused():
    matrix = sp.csr_matrix(np.eye(4) * 2.0)  # self-loops only
    with pytest.raises(RewireError, match="two"):
        weight_preserving_degree_rewire(matrix, seed=0)


@pytest.mark.unit
def test_the_chain_is_deterministic():
    matrix = _quantized_graph(seed=4)
    first, first_report = weight_preserving_degree_rewire(matrix, seed=23)
    second, second_report = weight_preserving_degree_rewire(matrix, seed=23)
    assert np.array_equal(first.toarray(), second.toarray())
    assert first_report.as_dict()["swaps_accepted"] == second_report.as_dict()["swaps_accepted"]
    assert first_report.overlap_final == second_report.overlap_final
    third, _ = weight_preserving_degree_rewire(matrix, seed=24)
    assert not np.array_equal(first.toarray(), third.toarray())


# ---------------------------------------------------------------------------
# The cell-type pair matrix: a diagnostic, deliberately not a gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_type_pair_matrix_is_a_diagnostic_and_is_not_a_gate(r0_topology):
    rewired, report = weight_preserving_degree_rewire(r0_topology.matrix, seed=29)
    quality = counterfactual_quality(r0_topology.matrix, rewired, report)
    assert "type_pair_matrix" not in quality["verdict"]["lines"]
    assert quality["verdict"]["C4"] is True, (
        "type-level connectivity is NOT preserved and must not gate C4"
    )

    classes = np.array(["ORN", "PN", "KC", "MBON"] * (r0_topology.n_nodes // 4), dtype=object)
    coo = rewired.tocoo()
    matrix = cell_type_pair_matrix(
        coo.row.astype(np.int64), coo.col.astype(np.int64), classes
    )
    assert set(matrix) <= {"ORN", "PN", "KC", "MBON"}
    assert sum(sum(targets.values()) for targets in matrix.values()) == rewired.nnz
