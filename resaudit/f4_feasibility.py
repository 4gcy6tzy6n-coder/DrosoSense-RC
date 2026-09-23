"""F4's construction-feasibility gate.

The contrast review fixed F4's single varied factor as **cell-type connectivity
organization** -- specifically the ``source-type -> target-type`` connection organization,
not the cell-type label itself -- while keeping F4 blocked until an admissible construction
exists. This module is that admissibility check, as a machine-checkable predicate.

**The label-only trap, recorded first because it is the obvious wrong answer.** Relabelling
nodes' cell types does NOT constitute F4. If cell type does not enter the reservoir
dynamics, permuting labels leaves ``A``, ``B``, the state trajectory and every downstream
quantity bit-identical -- an *annotation* counterfactual, not a *structural* one. F4 must
change the actual wiring so that the type-pair organization changes, and
:func:`type_pair_change` measures that on the released matrices rather than trusting the
constructor's description.

The six conditions, all of which must hold:

1. the type-pair organization changes by a DECLARED amount;
2. the degree sequence is preserved;
3. the weight multiset is preserved;
4. the input geometry is preserved;
5. every preregistered recurrence-preservation condition holds;
6. no A1-A5 verdict and no food score was used to select the candidate.

Condition 6 is checked structurally: this module imports no criterion, no battery and no
task metric, and :func:`f4_construction_feasible` accepts only matrices plus declared
invariants -- so a candidate cannot be chosen by consulting its own audit.

If no admissible construction is found, the outcome is ``F4 = OMITTED`` with the reason
recorded, which the review states is preferable to weakening F4's held-fixed constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

__all__ = [
    "F4_VARIED_FACTOR",
    "TypePairChange",
    "RecurrencePreservation",
    "F4Feasibility",
    "type_pair_change",
    "recurrence_summary",
    "f4_construction_feasible",
    "OMISSION_REASON_TEMPLATE",
]

#: F4's single varied factor, fixed by the contrast review.
F4_VARIED_FACTOR = "cell-type connectivity organization (source-type -> target-type)"

OMISSION_REASON_TEMPLATE = (
    "no admissible recurrence-preserving counterfactual found: {failed}"
)


@dataclass(frozen=True)
class TypePairChange:
    """Condition 1: did the type-pair organization actually change, and by how much?"""

    total_edges_moved: int
    l1_change: float
    relative_l1_change: float
    changed: bool
    n_type_pairs: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_edges_moved": int(self.total_edges_moved),
            "l1_change": float(self.l1_change),
            "relative_l1_change": float(self.relative_l1_change),
            "changed": bool(self.changed),
            "n_type_pairs": int(self.n_type_pairs),
        }


@dataclass(frozen=True)
class RecurrencePreservation:
    """Condition 5: the preregistered recurrence summaries, before and after."""

    scc_largest_fraction_before: float
    scc_largest_fraction_after: float
    n_cycles_before: int
    n_cycles_after: int
    cycle_lengths_before: tuple[int, ...]
    cycle_lengths_after: tuple[int, ...]
    preserved: bool
    tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scc_largest_fraction_before": float(self.scc_largest_fraction_before),
            "scc_largest_fraction_after": float(self.scc_largest_fraction_after),
            "n_cycles_before": int(self.n_cycles_before),
            "n_cycles_after": int(self.n_cycles_after),
            "cycle_lengths_before": list(self.cycle_lengths_before),
            "cycle_lengths_after": list(self.cycle_lengths_after),
            "preserved": bool(self.preserved),
            "tolerance": float(self.tolerance),
        }


@dataclass(frozen=True)
class F4Feasibility:
    """The gate's verdict, with every condition's outcome recorded separately."""

    feasible: bool
    conditions: Mapping[str, bool]
    reason: str
    type_pair_change: TypePairChange | None = None
    recurrence: RecurrencePreservation | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "conditions": dict(self.conditions),
            "reason": self.reason,
            "failed_conditions": sorted(k for k, v in self.conditions.items() if not v),
            "type_pair_change": self.type_pair_change.as_dict() if self.type_pair_change else None,
            "recurrence": self.recurrence.as_dict() if self.recurrence else None,
        }


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def type_pair_change(
    A_reference: sp.spmatrix,
    A_candidate: sp.spmatrix,
    labels: np.ndarray,
    *,
    min_relative_change: float = 0.05,
) -> TypePairChange:
    """Condition 1, measured on the WIRING, not on the labels.

    Args:
        A_reference: F1's adjacency.
        A_candidate: the proposed F4 wiring.
        labels: integer cell-type label per node, same indexing for both.
        min_relative_change: the DECLARED change threshold. A relabelling produces exactly
            zero here, which is how the label-only trap is caught.

    Returns:
        The measured change. ``changed`` is False for any candidate whose type-pair
        organization is untouched, including every pure permutation of labels.
    """
    labels = np.asarray(labels).astype(np.int64).ravel()

    def matrix(A: sp.spmatrix) -> np.ndarray:
        coo = sp.csr_matrix(A).tocoo()
        src = labels[coo.row]
        dst = labels[coo.col]
        u = int(labels.max()) + 1 if labels.size else 0
        m = np.zeros((u, u), dtype=np.float64)
        np.add.at(m, (src, dst), 1.0)
        return m

    before, after = matrix(A_reference), matrix(A_candidate)
    if before.shape != after.shape:
        raise ValueError("the two graphs carry different cell-type label spaces")
    delta = np.abs(after - before)
    l1 = float(delta.sum())
    denom = float(before.sum())
    rel = l1 / denom if denom else float("nan")
    return TypePairChange(
        total_edges_moved=int(np.count_nonzero(delta)),
        l1_change=l1,
        relative_l1_change=rel,
        changed=bool(not math.isnan(rel) and rel >= float(min_relative_change)),
        n_type_pairs=int(np.count_nonzero(before) + np.count_nonzero(after)),
    )


def _scc_largest_fraction(A: sp.spmatrix) -> float:
    from resaudit.battery import structural_diagnostics

    return float(structural_diagnostics(A).scc_largest_fraction)


def _simple_cycle_lengths(A: sp.spmatrix, *, max_len: int = 12) -> tuple[int, ...]:
    """Cycle counts by length via ``trace(A^k) > 0``, capped at ``max_len``.

    A diagnostic on the trace, deliberately cheap: the exact cycle basis is not needed to
    ask "did the cycle profile move", and paying for one would make the gate too expensive
    to run on every candidate.
    """
    A = sp.csr_matrix(A).tocsr()
    lengths: list[int] = []
    power = A.copy()
    for k in range(2, int(max_len) + 1):
        power = (power @ A).tocsr()
        if power.diagonal().sum() > 0:
            lengths.append(k)
    return tuple(lengths)


def recurrence_summary(A: sp.spmatrix, *, max_len: int = 12) -> dict[str, Any]:
    """The preregistered recurrence summaries: largest SCC fraction and cycle profile."""
    return {
        "scc_largest_fraction": _scc_largest_fraction(A),
        "cycle_lengths": _simple_cycle_lengths(A, max_len=max_len),
    }


def recurrence_preservation(
    A_reference: sp.spmatrix,
    A_candidate: sp.spmatrix,
    *,
    scc_tolerance: float = 0.02,
    max_len: int = 12,
) -> RecurrencePreservation:
    """Condition 5: largest-SCC fraction within tolerance and an identical cycle profile."""
    before = recurrence_summary(A_reference, max_len=max_len)
    after = recurrence_summary(A_candidate, max_len=max_len)
    scc_ok = abs(before["scc_largest_fraction"] - after["scc_largest_fraction"]) <= scc_tolerance
    cycles_ok = tuple(before["cycle_lengths"]) == tuple(after["cycle_lengths"])
    return RecurrencePreservation(
        scc_largest_fraction_before=before["scc_largest_fraction"],
        scc_largest_fraction_after=after["scc_largest_fraction"],
        n_cycles_before=len(before["cycle_lengths"]),
        n_cycles_after=len(after["cycle_lengths"]),
        cycle_lengths_before=tuple(before["cycle_lengths"]),
        cycle_lengths_after=tuple(after["cycle_lengths"]),
        preserved=bool(scc_ok and cycles_ok),
        tolerance=float(scc_tolerance),
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _degree_sequence(A: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    A = sp.csr_matrix(A)
    out = np.diff(A.indptr).astype(np.int64)
    inn = np.bincount(A.tocoo().col, minlength=A.shape[0]).astype(np.int64)
    return out, inn


def f4_construction_feasible(
    A_reference: sp.spmatrix,
    A_candidate: sp.spmatrix,
    labels: np.ndarray,
    B_reference: np.ndarray,
    B_candidate: np.ndarray,
    *,
    selection_used_audit_or_task: bool,
    min_relative_type_pair_change: float = 0.05,
    scc_tolerance: float = 0.02,
    max_cycle_len: int = 12,
) -> F4Feasibility:
    """The machine-checkable F4 gate. All six conditions must hold.

    Args:
        A_reference: F1's adjacency.
        A_candidate: the proposed F4 wiring.
        labels: cell-type label per node.
        B_reference / B_candidate: the input mappings.
        selection_used_audit_or_task: whether the candidate was chosen by consulting an
            A1-A5 verdict or any task metric. MUST be False for feasibility; the caller
            asserts this from its own process, and this module cannot check it because it
            deliberately cannot see the battery.

    Returns:
        :class:`F4Feasibility`. ``feasible`` False means F4 is omitted, not that it should
        be retried with weakened constraints.
    """
    A_reference = sp.csr_matrix(A_reference)
    A_candidate = sp.csr_matrix(A_candidate)
    A_ref_scaled = _match_scale(A_reference, A_candidate)

    tp = type_pair_change(
        A_ref_scaled, A_candidate, labels, min_relative_change=min_relative_type_pair_change
    )
    rec = recurrence_preservation(
        A_ref_scaled, A_candidate, scc_tolerance=scc_tolerance, max_len=max_cycle_len
    )

    out_ref, in_ref = _degree_sequence(A_ref_scaled)
    out_cand, in_cand = _degree_sequence(A_candidate)
    deg_ok = bool(np.array_equal(out_ref, out_cand) and np.array_equal(in_ref, in_cand))
    w_ref = np.sort(A_ref_scaled.tocoo().data)
    w_cand = np.sort(A_candidate.tocoo().data)
    weight_ok = bool(w_ref.size == w_cand.size and np.allclose(w_ref, w_cand, rtol=1e-9))
    B_reference = np.asarray(B_reference, dtype=np.float64)
    B_candidate = np.asarray(B_candidate, dtype=np.float64)
    input_ok = bool(
        B_reference.shape == B_candidate.shape
        and np.array_equal(B_reference != 0.0, B_candidate != 0.0)
    )

    conditions = {
        "type_pair_organization_changed": bool(tp.changed),
        "degree_sequence_preserved": deg_ok,
        "weight_multiset_preserved": weight_ok,
        "input_geometry_preserved": input_ok,
        "recurrence_preserved": bool(rec.preserved),
        "selection_used_no_audit_or_task": bool(not selection_used_audit_or_task),
    }
    feasible = all(conditions.values())
    failed = sorted(k for k, v in conditions.items() if not v)
    reason = (
        "all six conditions hold"
        if feasible
        else OMISSION_REASON_TEMPLATE.format(failed=", ".join(failed))
    )
    return F4Feasibility(
        feasible=feasible, conditions=conditions, reason=reason, type_pair_change=tp, recurrence=rec
    )


def _match_scale(A_reference: sp.spmatrix, A_candidate: sp.spmatrix) -> sp.csr_matrix:
    """Put the reference on the candidate's weight scale, so the 1:1 comparison is meaningful.

    Both matrices are compared as RELEASED, i.e. after their respective normalisations. The
    candidate is the object under test, so the reference is rescaled to it rather than the
    other way round.
    """
    A_reference = sp.csr_matrix(A_reference)
    if A_reference.nnz == 0 or A_candidate.nnz == 0:
        return A_reference
    r_ref = float(np.median(A_reference.tocoo().data))
    r_cand = float(np.median(A_candidate.tocoo().data))
    if r_ref <= 0:
        return A_reference
    return (A_reference * (r_cand / r_ref)).tocsr()
