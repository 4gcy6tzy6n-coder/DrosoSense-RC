"""Tests for the three contracts frozen by amendment 6.

These are the rules that had to hold before a real Stage-1 run: deterministic Din
apportionment, a content-based F1 identity, and weight matching up to one global scalar with
EXACT rho. Each is asserted against values stated in the amendment, so a drift in the code
or in the amendment is caught either way.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from resaudit.contracts import (
    FROZEN_RHO_EXACT,
    TYPED_ALIGNMENT_ORDER,
    TYPED_ALIGNMENT_WEIGHTS,
    apportion_din,
    canonical_adjacency_bytes,
    canonical_adjacency_hash,
    global_normalisation_scalar,
    weights_up_to_global_scalar,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Din apportionment
# ---------------------------------------------------------------------------


def test_the_frozen_ratio_is_the_declared_one():
    assert TYPED_ALIGNMENT_ORDER == ("ORN", "PN", "KC")
    assert TYPED_ALIGNMENT_WEIGHTS == (2, 2, 1)
    assert sum(TYPED_ALIGNMENT_WEIGHTS) == 5


def test_din5_reproduces_the_frozen_allocation_exactly():
    """The consistency requirement: the rule must agree with the signed Din=5 allocation."""
    assert apportion_din(5) == {"ORN": 2, "PN": 2, "KC": 1}


@pytest.mark.parametrize(
    "din,expected",
    [
        (6, {"ORN": 3, "PN": 2, "KC": 1}),
        (8, {"ORN": 3, "PN": 3, "KC": 2}),
    ],
)
def test_the_widths_that_blocked_F1_materialization_now_resolve(din, expected):
    """Din=6 and Din=8 could not be built because the rule declared no allocation."""
    assert apportion_din(din) == expected


@pytest.mark.parametrize("din", list(range(1, 41)))
def test_every_width_sums_exactly(din):
    alloc = apportion_din(din)
    assert set(alloc) == set(TYPED_ALIGNMENT_ORDER)
    assert sum(alloc.values()) == din
    assert all(v >= 0 for v in alloc.values())


@pytest.mark.parametrize("din", list(range(1, 41)))
def test_apportionment_is_deterministic(din):
    assert apportion_din(din) == apportion_din(din)


def test_apportionment_is_monotone_in_each_layer():
    """A wider input never removes a channel from a layer: the rule extends, it does not
    reshuffle."""
    for layer in TYPED_ALIGNMENT_ORDER:
        series = [apportion_din(d)[layer] for d in range(1, 41)]
        assert all(series[i + 1] >= series[i] for i in range(len(series) - 1)), layer


def test_the_largest_layers_keep_the_frozen_ratio():
    """ORN:PN:KC stays at or near 2:2:1, i.e. the ratio is extended not replaced."""
    for din in (5, 10, 20, 40):
        a = apportion_din(din)
        orn, pn, kc = a["ORN"], a["PN"], a["KC"]
        assert abs(orn - 2 * din / 5) <= 1
        assert abs(pn - 2 * din / 5) <= 1
        assert abs(kc - din / 5) <= 1


def test_din_below_one_is_refused():
    with pytest.raises(ValueError, match="must be >= 1"):
        apportion_din(0)


# ---------------------------------------------------------------------------
# canonical adjacency identity
# ---------------------------------------------------------------------------


def _example():
    return sp.csr_matrix(np.array([[0, 2, 0], [3, 0, 1], [0, 0, 0]], dtype=float))


def test_canonical_hash_is_format_independent():
    A = _example()
    h = canonical_adjacency_hash(A)
    for variant in (A.tocsc(), A.tocoo(), A.tolil().tocsr(), A.todense()):
        assert canonical_adjacency_hash(variant) == h


def test_canonical_hash_ignores_explicit_zeros_and_duplicates():
    """Both are representation artefacts, not graph differences."""
    A = _example()
    h = canonical_adjacency_hash(A)
    with_zero = sp.csr_matrix(
        (np.array([0.0, 2.0, 3.0, 1.0]), ([0, 0, 1, 1], [2, 1, 0, 2])), shape=(3, 3)
    )
    with_dup = sp.csr_matrix(
        (np.array([1.0, 1.0, 3.0, 1.0]), ([0, 0, 1, 1], [1, 1, 0, 2])), shape=(3, 3)
    )
    assert canonical_adjacency_hash(with_zero) == h
    assert canonical_adjacency_hash(with_dup) == h


def test_canonical_hash_separates_different_graphs():
    A = _example()
    B = sp.csr_matrix(np.array([[0, 2, 1], [3, 0, 0], [0, 0, 0]], dtype=float))
    assert canonical_adjacency_hash(A) != canonical_adjacency_hash(B)


def test_canonical_hash_uses_no_estimator_and_is_stable_across_calls():
    """The identity must not depend on any spectral-radius path; repeated calls agree."""
    A = _example()
    assert canonical_adjacency_bytes(A) == canonical_adjacency_bytes(A)
    assert canonical_adjacency_hash(A) == canonical_adjacency_hash(A)
    assert len(canonical_adjacency_hash(A)) == 64


def test_canonical_hash_refuses_non_finite_weights():
    A = sp.csr_matrix((np.array([np.nan, 1.0]), ([0, 1], [1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="non-finite"):
        canonical_adjacency_hash(A)


def test_canonical_serialisation_layout_is_declared():
    """The header carries shape, nnz and dtype, so two layouts cannot collide silently."""
    raw = canonical_adjacency_bytes(_example())
    assert len(raw) > 24
    assert b"<f8" in raw[:32]


# ---------------------------------------------------------------------------
# exact rho, weights up to one global scalar
# ---------------------------------------------------------------------------


def test_the_frozen_scale_is_exact_095():
    assert FROZEN_RHO_EXACT == 0.95


def test_global_scalar_is_target_over_raw_radius():
    assert global_normalisation_scalar(1.9) == pytest.approx(0.5)
    assert global_normalisation_scalar(0.95) == pytest.approx(1.0)


def test_a_zero_raw_radius_is_refused():
    with pytest.raises(ValueError, match="must be > 0"):
        global_normalisation_scalar(0.0)


def test_weight_matching_preserves_shape_but_not_absolute_values():
    """The amended rule's whole content: ranking and ratios in, absolute scale out."""
    W = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    Wn, alpha = weights_up_to_global_scalar(W, rho_raw=7.4)
    assert alpha == pytest.approx(0.95 / 7.4)
    # ranking preserved
    assert list(np.argsort(W)) == list(np.argsort(Wn))
    # pairwise ratios preserved
    assert np.allclose(Wn / Wn[0], W / W[0])
    # distribution shape preserved (same normalised histogram)
    assert np.allclose(np.sort(Wn) / Wn.sum(), np.sort(W) / W.sum())
    # absolute equality deliberately NOT claimed
    assert not np.allclose(Wn, W)


def test_applying_the_scalar_makes_the_family_reach_the_target_exactly():
    """The property that motivated the amendment: rho lands ON target, not near it."""
    from resaudit.battery import spectral_radius_of

    lengths = (5, 7, 11)
    n = sum(lengths)
    rows: list[int] = []
    cols: list[int] = []
    off = 0
    for L in lengths:
        for i in range(L):
            rows.append(off + (i + 1) % L)
            cols.append(off + i)
        off += L
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    rho_raw = spectral_radius_of(A)
    alpha = global_normalisation_scalar(rho_raw, FROZEN_RHO_EXACT)
    assert spectral_radius_of(A * alpha) == pytest.approx(FROZEN_RHO_EXACT, rel=1e-9)


def test_scaling_is_the_only_difference_between_reference_and_family_weights():
    """Two families with the same reference W differ only by their own scalar."""
    W = np.array([2.0, 2.0, 4.0, 6.0])
    a, alpha_a = weights_up_to_global_scalar(W, rho_raw=5.0)
    b, alpha_b = weights_up_to_global_scalar(W, rho_raw=9.5)
    assert alpha_a != alpha_b
    # the ratio a/b is a single constant across every weight
    ratios = a / b
    assert np.allclose(ratios, ratios[0])
