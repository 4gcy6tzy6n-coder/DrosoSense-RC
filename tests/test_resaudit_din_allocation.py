"""Tests for amendment 6's per-layer Din apportionment rule.

These pin the RULE, not just its outputs: the three allocations are checked as consequences
of the frozen 2:2:1 weights, largest-remainder apportionment and the fixed ORN > PN > KC tie
order, so that a later edit which hardcodes (3,2,1)/(3,3,2) or quietly changes the tie-break
fails here.
"""

from __future__ import annotations

import pytest

from resaudit.din_allocation import (
    FROZEN_DEFAULT_ALLOCATION,
    LAYER_WEIGHTS,
    TIE_ORDER,
    DinAllocationError,
    allocation_for_frozen_default,
    apportion_din,
)

#: The preregistered native widths (DATA_PREREGISTRATION.md; amendment 4).
PREREGISTERED_WIDTHS = (6, 8)


def test_din5_reproduces_the_frozen_default_exactly():
    """Amendment 6 section 1: failure to reproduce Din=5 bit-for-bit blocks Stage 1."""
    assert allocation_for_frozen_default() == dict(FROZEN_DEFAULT_ALLOCATION)
    assert allocation_for_frozen_default() == {"ORN": 2, "PN": 2, "KC": 1}


def test_preregistered_widths_get_the_declared_allocations():
    assert apportion_din(6) == {"ORN": 3, "PN": 2, "KC": 1}
    assert apportion_din(8) == {"ORN": 3, "PN": 3, "KC": 2}


@pytest.mark.parametrize("din", [5, *PREREGISTERED_WIDTHS])
def test_allocation_sums_to_din(din):
    alloc = apportion_din(din)
    assert sum(alloc.values()) == din, "the mapping builder rejects a non-summing allocation"
    assert list(alloc) == [name for name, _ in LAYER_WEIGHTS], "declared layer order preserved"


@pytest.mark.parametrize("din", [5, *PREREGISTERED_WIDTHS])
def test_rule_is_deterministic_across_calls(din):
    """No randomness, no state: repeated calls must be identical."""
    assert apportion_din(din) == apportion_din(din)
    assert [apportion_din(din) for _ in range(5)].count(apportion_din(din)) == 5


def test_largest_remainder_is_what_assigns_the_surplus_not_the_tie_order():
    """Din=7 has quotients 2.8/2.8/1.4 -> surplus to ORN and PN, i.e. remainders decide.

    If a later edit switched to assigning the surplus by tie order alone, this would give
    (3,2,2) instead of (3,3,1), so the check distinguishes the two mechanisms.
    """
    assert apportion_din(7) == {"ORN": 3, "PN": 3, "KC": 1}


def test_tie_order_is_orn_then_pn_then_kc():
    assert TIE_ORDER == ("ORN", "PN", "KC")
    # Din=25: quotas 10.0/10.0/5.0 exactly -- no surplus, so no tie is exercised.
    assert apportion_din(25) == {"ORN": 10, "PN": 10, "KC": 5}
    # Din=10: quotas 4.0/4.0/2.0 exactly.
    assert apportion_din(10) == {"ORN": 4, "PN": 4, "KC": 2}
    # With an exact three-way tie in remainder, ORN before PN before KC.
    alloc = apportion_din(5, weights=(("ORN", 1), ("PN", 1), ("KC", 1)))
    assert alloc == {"ORN": 2, "PN": 2, "KC": 1}


def test_weights_are_the_frozen_two_two_one():
    assert dict(LAYER_WEIGHTS) == {"ORN": 2, "PN": 2, "KC": 1}
    assert sum(w for _, w in LAYER_WEIGHTS) == 5


def test_allocations_do_not_depend_on_anything_but_the_arguments():
    """Amendment 6: no score, label or audit outcome may influence the allocation.

    The function's signature is the enforcement: it accepts a width and a weight table and
    nothing else, so there is no channel through which a family score could enter.
    """
    import inspect

    params = set(inspect.signature(apportion_din).parameters)
    assert params == {"din", "weights", "tie_order"}


def test_rejects_widths_that_cannot_cover_the_declared_layers():
    with pytest.raises(DinAllocationError):
        apportion_din(2)
    with pytest.raises(DinAllocationError):
        apportion_din(0)


def test_rejects_a_tie_order_naming_undeclared_layers():
    with pytest.raises(DinAllocationError):
        apportion_din(6, tie_order=("ORN", "PN", "MBON"))
