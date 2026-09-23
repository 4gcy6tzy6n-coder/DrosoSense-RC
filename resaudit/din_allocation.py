"""Deterministic per-layer Din allocation for the typed-aligned input mapping.

Authority: docs/resaudit_food_preregistration_amendment_6.md.

The frozen typed-aligned mapping declared its allocation only at ``Din=5``
(``DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER = {ORN:2, PN:2, KC:1}``). For the preregistered
native widths ``Din in {6, 8}`` the allocation is extended deterministically by
preserving the frozen 2:2:1 layer ratio and applying Hamilton / largest-remainder
apportionment with a FIXED tie order ``ORN > PN > KC``.

The RULE is frozen, not its three outputs: ``(2,2,1)``, ``(3,2,1)`` and ``(3,3,2)`` are
consequences of the rule, so a reviewer can verify that the wider allocations were not
chosen to make a family clear A3. No score, label, performance metric or audit outcome may
influence the result -- this function reads nothing but its arguments.

At ``Din=5`` the rule reproduces the frozen default exactly, which is what makes it a
generalisation of the existing rule rather than a replacement.
"""

from __future__ import annotations

from typing import Mapping, Sequence

#: The frozen layer weights, carried over from the ``Din=5`` default allocation.
LAYER_WEIGHTS: tuple[tuple[str, int], ...] = (("ORN", 2), ("PN", 2), ("KC", 1))

#: The frozen tie order for equal fractional remainders. NOT by layer size or node count.
TIE_ORDER: tuple[str, ...] = ("ORN", "PN", "KC")

#: The allocation the frozen mapping used before this rule existed. ``Din=5`` must match it.
FROZEN_DEFAULT_ALLOCATION: Mapping[str, int] = {"ORN": 2, "PN": 2, "KC": 1}


class DinAllocationError(ValueError):
    """Raised when the rule cannot produce an allocation for the requested width."""


def apportion_din(
    din: int,
    *,
    weights: Sequence[tuple[str, int]] = LAYER_WEIGHTS,
    tie_order: Sequence[str] = TIE_ORDER,
) -> dict[str, int]:
    """Allocate ``din`` input channels over the declared layers.

    Hamilton / largest-remainder: floor each layer's quota, then hand the surplus channels
    out in descending fractional-remainder order, breaking ties by the frozen ``tie_order``.

    Args:
        din: The total input width. Must be >= the number of declared layers.
        weights: ``(layer, weight)`` pairs, in the declared layer order.
        tie_order: Layer preference used only when remainders are exactly equal.

    Returns:
        ``{layer: channels}`` in the declared layer order, summing to ``din``.

    Raises:
        DinAllocationError: If ``din`` is too small to give every layer a channel, or the
            layer set is empty.
    """
    layers = [name for name, _ in weights]
    if not layers:
        raise DinAllocationError("no layers declared")
    total_din = int(din)
    if total_din < len(layers):
        raise DinAllocationError(
            f"din={total_din} cannot cover {len(layers)} declared layers {layers}"
        )
    unknown = [n for n in tie_order if n not in layers]
    if unknown:
        raise DinAllocationError(f"tie_order names undeclared layers: {unknown}")

    weight_total = sum(w for _, w in weights)
    if weight_total <= 0:
        raise DinAllocationError("layer weights must sum to a positive value")

    quotas = {name: total_din * w / weight_total for name, w in weights}
    allocation = {name: int(quotas[name]) for name in layers}
    surplus = total_din - sum(allocation.values())

    if surplus:
        rank = {name: i for i, name in enumerate(tie_order)}
        # descending remainder, then frozen tie order; a layer that already received a
        # surplus channel is not eligible again, so each gets at most one here.
        order = sorted(
            layers,
            key=lambda n: (-(quotas[n] - int(quotas[n])), rank[n]),
        )
        for name in order[:surplus]:
            allocation[name] += 1

    assert sum(allocation.values()) == total_din, "allocation must sum to Din"
    return {name: allocation[name] for name in layers}


def allocation_for_frozen_default() -> dict[str, int]:
    """The rule's ``Din=5`` result, which must equal the frozen default bit-for-bit."""
    return apportion_din(5)
