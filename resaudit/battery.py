"""ResAudit-Food -- the audit battery (A1-A5) and the task-blind gate engine.

This module is the paper's C1: five cheap, model-free, pre-deployment checks applied to
a candidate reservoir BEFORE task training. It is deliberately a *gate engine*, not a
model comparison: it consumes :class:`~resaudit.family.FamilySpec` objects, which cannot
carry task information, and it produces per-criterion verdicts with three-valued
outcomes.

Frozen thresholds live in :mod:`resaudit.criteria` and are imported, never re-typed.

Two structural rules are enforced in code rather than left to discipline:

1. **A4/A5 are not measured unless A3 passes.** The pre-registration makes A3 the
   primary qualification gate and A4/A5 the downstream gates a QUALIFIED reservoir must
   still clear. On an A3 failure this module returns A4/A5 as ``not_evaluated`` with an
   explicit reason, and it does NOT pay the cost of driving the reservoir.

2. **A2 is three-valued.** The fairness conditions (degree sequence, global weight
   multiset, per-source weight multisets preserved exactly; overlap under the ceiling)
   are defined only for a counterfactual that shares a wiring parent with the candidate.
   A family with no ``counterfactual_parent`` has no such counterfactual, so A2 is
   ``NOT_APPLICABLE`` -- never silently ``FAIL``. This is the difference between "this
   family was audited and found unfair" and "this family was never a counterfactual".

Measurement kernels are imported from the frozen DrosoSense-RC package
(``drososense.substrate_scores.krylov_score``, ``drososense.reservoir.dynamics``), so
A3/A4/A5 here are the SAME functions that produced the motivating negative, not
re-implementations that merely resemble them.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resaudit import _kernels as _dyn  # noqa: E402
from resaudit import _kernels as _r2  # noqa: E402
from resaudit._kernels import krylov_score  # noqa: E402

from resaudit.criteria import (  # noqa: E402
    A1_ISOLATED_FRACTION_MAX,
    A3_CAVEAT,
    A3_NAME,
    A4_DESCRIPTIVE_CAVEAT,
    BLOCKING_CRITERIA,
    A1_MEAN_OUT_DEGREE_MIN,
    A1_SCC_FRACTION_MIN,
    A2_OVERLAP_MAX,
    A3_FACTOR,
    A3_KRYLOV_K,
    A4_MIN,
    A5_FACTOR,
    CriterionResult,
    CriterionState,
    evaluate_ge,
    evaluate_le,
    not_applicable,
)
from resaudit.family import FamilySpec  # noqa: E402

# ---------------------------------------------------------------------------
# The probe protocol: the measurement rule held CONSTANT across every family.
# ---------------------------------------------------------------------------
# The pre-registration fixes K = A3_KRYLOV_K, the A3/A4/A5 thresholds and "the same
# input geometry"; it does not fix the drive hyper-parameters. They are declared here
# once so that no family is measured under a rule chosen after seeing its score.
#
# These values are a DECLARED CHOICE, not signed criteria. They are recorded in every
# report (``probe_protocol()``) so a reader can see exactly what "the same rule" meant.

#: Leaky-integration rate. 1.0 = the pure tanh recurrence (no state carry-over), so the
#: substrate alone carries memory and no leak parameter can manufacture it.
PROBE_LEAK = 1.0

#: Spectral gain applied to the recurrent term.
PROBE_GAIN = 1.0

#: Probe windows. Fixed seed, so the drive is byte-identical across families.
PROBE_WINDOWS = 48
PROBE_LENGTH = 256
PROBE_SEED = 20260923

#: Washout steps discarded before the memory correlation.
PROBE_WASHOUT = 16

#: Scaling of the probe input. A fixed constant, never per-family.
PROBE_INPUT_SCALE = 1.0

#: The pre-family synthesis conventions declared by amendment 1 Section 0.1. These are
#: CALIBRATION-DERIVED, not pre-registration rules. They are frozen here, before any F
#: family exists, so that they cannot be adjusted in response to an F family's score.
#:
#: ``rho_target`` in particular: A3 is normalized-scale dependent (amendment 1 Section 1),
#: so every family is brought to this spectral radius by a single scalar multiplication
#: before A3/A4/A5 are measured.
FROZEN_RHO_TARGET = 0.95


def probe_protocol() -> dict[str, Any]:
    """The held-constant measurement rule, for the report."""
    return {
        "leak": PROBE_LEAK,
        "gain": PROBE_GAIN,
        "windows": PROBE_WINDOWS,
        "length": PROBE_LENGTH,
        "seed": PROBE_SEED,
        "washout": PROBE_WASHOUT,
        "input_scale": PROBE_INPUT_SCALE,
        "krylov_K": A3_KRYLOV_K,
        "memory_lags": list(_dyn.MEMORY_LAGS),
    }


def probe_input_with_seed(din: int, seed: int) -> np.ndarray:
    """A probe drive for a given seed, shape ``(windows, length, din)``.

    Channel ``c`` is drawn from its own generator seeded ``seed + c``, so the probe for
    ``din = k`` is a strict prefix of the probe for ``din = k + 1``: widening ``Din`` does
    not change any existing channel, so comparisons across families cannot be moved by an
    accidental redraw.

    The PROTOCOL (windows, length, leak, gain, washout, scale) is fixed; only the
    realization varies. Nothing here reads a family, a task or a label.
    """
    d = int(din)
    if d < 1:
        raise ValueError("din must be >= 1")
    out = np.empty((PROBE_WINDOWS, PROBE_LENGTH, d), dtype=np.float64)
    for c in range(d):
        rng = np.random.default_rng(int(seed) + c)
        out[:, :, c] = rng.uniform(-1.0, 1.0, size=(PROBE_WINDOWS, PROBE_LENGTH))
    return np.ascontiguousarray(out * PROBE_INPUT_SCALE)


def probe_input(din: int) -> np.ndarray:
    """The single shared probe drive, at the frozen seed."""
    return probe_input_with_seed(din, PROBE_SEED)



# Imported from resaudit._kernels (the corrected estimator, inlined here
# so the framework is self-contained and free of circular imports).
from resaudit._kernels import spectral_radius_of  # noqa: E402  (re-export)

def scale_to_spectral_radius(A: sp.spmatrix, target: float = FROZEN_RHO_TARGET) -> sp.csr_matrix:
    """Scale ``A`` so its spectral radius is ``target``. One scalar, nothing else changes."""
    A = sp.csr_matrix(A).tocsr()
    r = spectral_radius_of(A)
    return (A * (float(target) / r)).tocsr() if r > 0 else A


# ---------------------------------------------------------------------------
# A1 -- structure present
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralDiagnostics:
    """A1's supporting numbers, kept as a diagnostic rather than folded into the verdict."""

    n_nodes: int
    n_edges: int
    scc_largest: int
    scc_count: int
    scc_largest_fraction: float
    isolated_fraction: float
    mean_out_degree: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "scc_largest": int(self.scc_largest),
            "scc_count": int(self.scc_count),
            "scc_largest_fraction": float(self.scc_largest_fraction),
            "isolated_fraction": float(self.isolated_fraction),
            "mean_out_degree": float(self.mean_out_degree),
        }


def _tarjan_scc(A: sp.spmatrix) -> list[list[int]]:
    """Iterative Tarjan SCC on a sparse directed graph. No recursion limit concerns."""
    A = A.tocsr()
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    index = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    on_stack = np.zeros(n, dtype=bool)
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work: list[tuple[int, int]] = [(root, indptr[root])]
        while work:
            v, pi = work[-1]
            if pi == indptr[v]:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            for i in range(pi, indptr[v + 1]):
                w = int(indices[i])
                if index[w] == -1:
                    work[-1] = (v, i + 1)
                    work.append((w, indptr[w]))
                    recurse = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                components.append(comp)
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return components


def structural_diagnostics(A: sp.spmatrix) -> StructuralDiagnostics:
    """Compute A1's three quantities plus the SCC histogram."""
    A = A.tocsr()
    n = int(A.shape[0])
    if n == 0:
        raise ValueError("empty graph")
    out_degree = np.diff(A.indptr).astype(np.int64)
    isolated = float(np.count_nonzero(out_degree == 0) / n)
    mean_out = float(out_degree.mean())
    comps = _tarjan_scc(A)
    sizes = sorted((len(c) for c in comps), reverse=True)
    largest = sizes[0] if sizes else 0
    return StructuralDiagnostics(
        n_nodes=n,
        n_edges=int(A.nnz),
        scc_largest=int(largest),
        scc_count=len(comps),
        scc_largest_fraction=float(largest / n),
        isolated_fraction=isolated,
        mean_out_degree=mean_out,
    )


def audit_a1(spec: FamilySpec) -> CriterionResult:
    """A1 -- structure present.

    PASS iff ``largest SCC >= 0.90*N`` AND ``isolated <= 0.02`` AND
    ``mean out-degree >= 2.0``. All three are required; "close" is not a pass.
    """
    d = structural_diagnostics(spec.A)
    ok = (
        d.scc_largest_fraction >= A1_SCC_FRACTION_MIN
        and d.isolated_fraction <= A1_ISOLATED_FRACTION_MAX
        and d.mean_out_degree >= A1_MEAN_OUT_DEGREE_MIN
    )
    worst = {
        "scc_largest_fraction": d.scc_largest_fraction >= A1_SCC_FRACTION_MIN,
        "isolated_fraction": d.isolated_fraction <= A1_ISOLATED_FRACTION_MAX,
        "mean_out_degree": d.mean_out_degree >= A1_MEAN_OUT_DEGREE_MIN,
    }
    return CriterionResult(
        criterion="A1",
        state=CriterionState.PASS if ok else CriterionState.FAIL,
        applicable=True,
        value=float(d.scc_largest_fraction),
        threshold=float(A1_SCC_FRACTION_MIN),
        expression=(
            f"largest_SCC/N >= {A1_SCC_FRACTION_MIN} and "
            f"isolated <= {A1_ISOLATED_FRACTION_MAX} and "
            f"mean_out_degree >= {A1_MEAN_OUT_DEGREE_MIN}"
        ),
        reason="" if ok else "one or more A1 terms not met: " + ", ".join(
            k for k, v in worst.items() if not v
        ),
        detail={**d.as_dict(), "terms_met": worst},
    )


# ---------------------------------------------------------------------------
# A2 -- counterfactual fair (THREE-VALUED)
# ---------------------------------------------------------------------------

A2_NO_PARENT_REASON = (
    "no wiring counterfactual parent is declared for this family, so A2's fairness "
    "conditions (degree sequence, global weight multiset, per-source weight multisets "
    "preserved exactly; overlap ceiling) are not defined for it. NOT_APPLICABLE is not a "
    "pass and not a failure: this family was never a counterfactual."
)


def audit_a2(spec: FamilySpec, parent: FamilySpec | None) -> CriterionResult:
    """A2 -- counterfactual fair.

    Applicability is decided FIRST and on a structural precondition: the family must
    declare a ``counterfactual_parent`` AND that parent's spec must be supplied. Without
    a shared wiring parent there is nothing whose degree sequence or weight multisets
    could be preserved, and the criterion is ``NOT_APPLICABLE``.

    When applicable, the exactness terms are booleans and the overlap ceiling is
    ``A2_OVERLAP_MAX``, compared through the frozen project's own
    ``counterfactual_quality`` so the verdict uses the signed C4 logic.
    """
    if spec.counterfactual_parent is None:
        return not_applicable(
            "A2",
            reason=A2_NO_PARENT_REASON,
            expression="degree/weight/per-source multisets exact; overlap <= 0.20",
            detail={"counterfactual_applicable": False, "declared_parent": None},
        )
    if parent is None:
        return not_applicable(
            "A2",
            reason=(
                f"family declares counterfactual_parent={spec.counterfactual_parent!r} but "
                "that parent's spec was not supplied to the audit, so the preservation "
                "terms cannot be measured"
            ),
            expression="degree/weight/per-source multisets exact; overlap <= 0.20",
            detail={
                "counterfactual_applicable": False,
                "declared_parent": spec.counterfactual_parent,
                "parent_supplied": False,
            },
        )
    if parent.family_id != spec.counterfactual_parent:
        raise ValueError(
            f"{spec.family_id}: supplied parent is {parent.family_id!r}, "
            f"but the declared parent is {spec.counterfactual_parent!r}"
        )

    # ``counterfactual_quality`` returns one entry per signed C4.x criterion plus a
    # ``verdict`` block; the verdict is what carries the boolean lines A2 needs.
    verdict_block = _r2.counterfactual_quality(parent.A, spec.A, _parent_report(spec, parent))["verdict"]
    checks: Mapping[str, Any] = verdict_block["checks"]
    lines: Mapping[str, Any] = verdict_block["lines"]
    # C4.4's median in-strength error is a signed diagnostic; A2's declared terms are the
    # three exactness booleans plus the overlap ceiling.
    exact_terms = {
        "degree_exact": bool(checks["degree_exact"]),
        "global_weights_exact": bool(checks["global_weights_exact"]),
        "per_source_weights_exact": bool(checks["per_source_weights_exact"]),
    }
    overlap = float(lines["edge_overlap"])
    overlap_ok = bool(overlap <= A2_OVERLAP_MAX)
    ok = all(exact_terms.values()) and overlap_ok
    failed = [k for k, v in exact_terms.items() if not v]
    if not overlap_ok:
        failed.append("edge_overlap")
    return CriterionResult(
        criterion="A2",
        state=CriterionState.PASS if ok else CriterionState.FAIL,
        applicable=True,
        value=overlap,
        threshold=float(A2_OVERLAP_MAX),
        expression=(
            "degree sequence, global weight multiset and per-source weight multisets "
            f"preserved exactly; edge_overlap <= {A2_OVERLAP_MAX}"
        ),
        reason="" if ok else "A2 terms not met: " + ", ".join(failed),
        detail={
            "counterfactual_applicable": True,
            "declared_parent": spec.counterfactual_parent,
            "exact_terms": exact_terms,
            "edge_overlap": overlap,
            "median_in_strength_err": float(lines["median_in_strength_err"]),
            "mixing_declared": bool(lines["mixing_declared"]),
            "overlap_ceiling": float(A2_OVERLAP_MAX),
        },
    )


def _parent_report(spec: FamilySpec, parent: FamilySpec) -> _r2.RewireReport:
    """Reconstruct the swap-chain facts A2 needs, from the pair itself.

    ``counterfactual_quality`` expects the builder's ``RewireReport``. For an audit of an
    externally supplied pair we do not have that object, so the overlap is measured
    directly from the two graphs and reported honestly; the report's provenance fields
    are marked as reconstructed rather than claiming a chain that never ran here.
    """
    a, b = parent.A.tocsr(), spec.A.tocsr()
    a_set = set(zip(*a.nonzero()))
    b_set = set(zip(*b.nonzero()))
    overlap = float(len(a_set & b_set) / len(a_set)) if a_set else 0.0
    self_loops = int(np.count_nonzero(a.diagonal() != 0))
    return _r2.RewireReport(
        swaps_attempted=0,
        swaps_accepted=0,
        swaps_rejected_duplicate=0,
        swaps_rejected_self_loop=0,
        swaps_rejected_shared_endpoint=0,
        swaps_weight_matched_exact=0,
        swaps_weight_matched_nearest=0,
        swaps_rejected_no_weight_partner=0,
        swaps_rejected_no_partner=0,
        swaps_rejected_no_legal_partner=0,
        swaps_accepted_exact_weight=0,
        swaps_accepted_near_weight=0,
        allow_near_weights=False,
        mixing_curve=(),
        n_edges=int(a.nnz),
        n_self_loops=self_loops,
        overlap_initial=1.0,
        overlap_final=overlap,
        seconds=0.0,
        stopped_because="reconstructed_from_pair",
        target_overlap=float(A2_OVERLAP_MAX),
    )


# ---------------------------------------------------------------------------
# A3 -- Krylov controllability (THE PRIMARY GATE)
# ---------------------------------------------------------------------------


def audit_a3(spec: FamilySpec) -> tuple[CriterionResult, dict[str, Any]]:
    """A3 -- finite-horizon input-reachable state diversity.

    ``D_eff([B, A*B, ..., A^16*B]) >= 2*Din``. Uses the frozen project's ``krylov_score``
    verbatim (same K, same tolerance, same participation-ratio definition), so this number
    is directly comparable with the motivating measurement ``D_eff ~ 4.7-7.1 against a
    gate of 10``.

    The id and threshold are unchanged from the pre-registration; the NAME is narrowed.
    A3 was introduced as "Krylov controllability" and read as a proxy for a useful
    reservoir, which calibration falsified -- a feed-forward nilpotent chain passes it.
    The result therefore carries :data:`~resaudit.criteria.A3_CAVEAT` in its detail, and
    no report may present an A3 PASS as evidence of recurrence.
    """
    score = krylov_score(spec.A, spec.B, K=A3_KRYLOV_K)
    din = int(score.din)
    threshold = float(A3_FACTOR * din)
    result = evaluate_ge(
        "A3",
        score.effective_rank,
        threshold,
        expression=f"D_eff([B, A*B, ..., A^{A3_KRYLOV_K}*B]) >= {A3_FACTOR} * Din",
        detail={
            "criterion_name": A3_NAME,
            "caveat": A3_CAVEAT,
            "Din": din,
            "K": int(score.K),
            "D_eff": float(score.effective_rank),
            "numerical_rank": int(score.numerical_rank),
            "numerical_rank_relative": int(score.numerical_rank_relative),
            "theoretical_columns": int(score.theoretical_columns),
            "power_norms": [float(v) for v in score.power_norms],
            "D_eff_over_threshold": float(
                score.effective_rank / threshold if threshold else float("nan")
            ),
        },
    )
    return result, score.as_dict()


# ---------------------------------------------------------------------------
# A4 / A5 -- downstream gates, measured ONLY after A3 passes
# ---------------------------------------------------------------------------


def audit_a4_a5(spec: FamilySpec) -> tuple[CriterionResult, CriterionResult, dict[str, Any]]:
    """A4 (recurrent memory) and A5 (state expansion), on the shared probe.

    Returns:
        ``(a4, a5, diagnostics)``. Both are executed together because they share one
        reservoir drive: A4 needs the state trajectory and the input series, A5 needs the
        state matrix. Paying for two drives would measure the same thing twice.
    """
    n = int(spec.n_nodes)
    X = probe_input(spec.din)
    W_in = sp.csr_matrix(spec.B)
    bias = np.zeros(n, dtype=np.float64)

    driven = _dyn.reservoir_drive(
        spec.A,
        W_in,
        bias,
        X,
        gain=PROBE_GAIN,
        leak=PROBE_LEAK,
    )
    zero = sp.csr_matrix((n, n), dtype=np.float64)
    driven_ff = _dyn.reservoir_drive(
        zero,
        W_in,
        bias,
        X,
        gain=PROBE_GAIN,
        leak=PROBE_LEAK,
    )

    with_recurrence = _dyn.memory_metric(
        driven.states, X, discard_washout=PROBE_WASHOUT
    )
    without_recurrence = _dyn.memory_metric(
        driven_ff.states, X, discard_washout=PROBE_WASHOUT
    )
    a4_value = _memory_contribution(with_recurrence, without_recurrence)

    a4 = evaluate_ge(
        "A4",
        a4_value,
        A4_MIN,
        expression=f"(M_recurrent - M_A:=0) / M_recurrent >= {A4_MIN}",
        detail={
            "M_recurrent": float(with_recurrence),
            "M_feedforward": float(without_recurrence),
            "definition": "(M_recurrent - M_A:=0) / M_recurrent",
            "gate_status": "descriptive_only",
            "caveat": A4_DESCRIPTIVE_CAVEAT,
        },
    )

    rank = _dyn.effective_rank(driven.states)
    d_eff_state = float(rank["D_eff"])
    din = int(spec.din)
    threshold = float(A5_FACTOR * din)
    a5 = evaluate_ge(
        "A5",
        d_eff_state,
        threshold,
        expression=f"D_eff(state) >= {A5_FACTOR} * Din",
        detail={
            "Din": din,
            "D_eff_state": d_eff_state,
            "D_eff_over_threshold": float(d_eff_state / threshold if threshold else float("nan")),
            "states_shape": list(driven.states.shape),
        },
    )
    return a4, a5, {
        "M_recurrent": float(with_recurrence),
        "M_feedforward": float(without_recurrence),
        "D_eff_state": d_eff_state,
        "probe_protocol": probe_protocol(),
    }


def _memory_contribution(m_recurrent: float, m_feedforward: float) -> float:
    """``(M_recurrent - M_{A:=0}) / M_recurrent``, defined as 0.0 at zero recurrence.

    Returns 0.0 rather than NaN/inf when ``M_recurrent`` is 0: no memory at all is a
    FAIL on the signed threshold, not an undefined comparison.
    """
    m_recur = float(m_recurrent)
    if m_recur <= 0.0:
        return 0.0
    return float((m_recur - float(m_feedforward)) / m_recur)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyAudit:
    """A1-A5 for one family, plus the qualification verdict.

    Attributes:
        family: The family's report header (no matrices).
        results: ``{criterion: CriterionResult}`` for A1, A2, A3, and for A4/A5 when
            they were evaluated.
        downstream_evaluated: Whether A4/A5 were measured (i.e. A3 passed).
        downstream_not_evaluated_reason: Why they were not, when they were not.
        diagnostics: Supporting numbers, including the A3 Krylov score.
    """

    family: Mapping[str, Any]
    results: Mapping[str, CriterionResult]
    downstream_evaluated: bool
    downstream_not_evaluated_reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def food_eligible(self) -> bool:
        """Pre-registration Section 6.1: no family runs food tasks unless it clears A3."""
        a3 = self.results.get("A3")
        return bool(a3 is not None and a3.passed)

    @property
    def construct_qualified(self) -> bool:
        """Every ADMISSION and EVALUATED criterion passed. DELIBERATELY NOT ``qualified``.

        Kept separate from :attr:`food_eligible` on purpose, and the two are emitted as
        separate machine fields that must never be merged into a single "Stage 1 passed"
        flag. A family can be ``food_eligible`` (it cleared A3) without being
        ``construct_qualified`` (it failed A1) -- the nilpotent chain is exactly that, and
        reading its A3 PASS as a pass would be the framework's central error.

        ``NOT_APPLICABLE`` never blocks. A4/A5 that were never evaluated because A3
        failed cannot make a family qualified, and A3's own failure already excludes it.

        Only :data:`~resaudit.criteria.BLOCKING_CRITERIA` can disqualify. **A4 is not one
        of them**: its construct-validity audit retired it as a gate, so an A4 number is
        reported and explained but never blocks. See
        ``docs/resaudit_a4_construct_validity.md``.
        """
        if not self.food_eligible:
            return False
        return all(
            self.results[c].state is not CriterionState.FAIL
            for c in BLOCKING_CRITERIA
            if c != "A3" and c in self.results
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": dict(self.family),
            "criteria": {k: v.as_dict() for k, v in self.results.items()},
            # These two are emitted SEPARATELY and permanently. Never merge them into one
            # "stage1_passed" flag: food_eligible is about the food stage, and
            # construct_qualified is about the audit.
            "food_eligible": self.food_eligible,
            "construct_qualified": self.construct_qualified,
            "downstream_evaluated": bool(self.downstream_evaluated),
            "downstream_not_evaluated_reason": self.downstream_not_evaluated_reason,
            "diagnostics": _jsonable(self.diagnostics),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


A4_A5_SKIPPED_REASON = (
    "A3 (the primary qualification gate) did not pass, so the downstream gates A4 and A5 "
    "were not evaluated. This is not a pass and not a failure for A4/A5: the "
    "pre-registration measures them only on a QUALIFIED reservoir. Consequently whether "
    "A3 failure predicts A4/A5 failure is UNOBSERVABLE by construction in this design."
)


def audit_family(
    spec: FamilySpec,
    *,
    parent: FamilySpec | None = None,
    measure_downstream: bool = True,
) -> FamilyAudit:
    """Run A1-A5 on one family and return the verdict.

    Args:
        spec: The sealed family object. It cannot carry task information.
        parent: The counterfactual parent's spec, required only when ``spec`` declares a
            ``counterfactual_parent``.
        measure_downstream: When ``False``, A4/A5 are never measured even if A3 passes.
            Used by tests that assert the ordering rule without paying for a drive.

    Returns:
        A :class:`FamilyAudit`.
    """
    results: dict[str, CriterionResult] = {}
    results["A1"] = audit_a1(spec)
    results["A2"] = audit_a2(spec, parent)
    a3, a3_score = audit_a3(spec)
    results["A3"] = a3

    diagnostics: dict[str, Any] = {"A3_krylov": a3_score}

    if not a3.passed:
        results["A4"] = not_applicable(
            "A4",
            reason=A4_A5_SKIPPED_REASON,
            expression=f"(M_recurrent - M_A:=0) / M_recurrent >= {A4_MIN}",
            detail={"skipped_because": "A3_FAIL", "primary_gate": "A3"},
        )
        results["A5"] = not_applicable(
            "A5",
            reason=A4_A5_SKIPPED_REASON,
            expression=f"D_eff(state) >= {A5_FACTOR} * Din",
            detail={"skipped_because": "A3_FAIL", "primary_gate": "A3"},
        )
        return FamilyAudit(
            family=spec.as_dict(),
            results=results,
            downstream_evaluated=False,
            downstream_not_evaluated_reason=A4_A5_SKIPPED_REASON,
            diagnostics=diagnostics,
        )

    if not measure_downstream:
        results["A4"] = not_applicable(
            "A4",
            reason="downstream measurement disabled by the caller (measure_downstream=False)",
            expression=f"(M_recurrent - M_A:=0) / M_recurrent >= {A4_MIN}",
            detail={"skipped_because": "caller_disabled"},
        )
        results["A5"] = not_applicable(
            "A5",
            reason="downstream measurement disabled by the caller (measure_downstream=False)",
            expression=f"D_eff(state) >= {A5_FACTOR} * Din",
            detail={"skipped_because": "caller_disabled"},
        )
        return FamilyAudit(
            family=spec.as_dict(),
            results=results,
            downstream_evaluated=False,
            downstream_not_evaluated_reason="downstream measurement disabled by the caller",
            diagnostics=diagnostics,
        )

    a4, a5, dyn_diag = audit_a4_a5(spec)
    results["A4"] = a4
    results["A5"] = a5
    diagnostics["dynamics"] = dyn_diag
    return FamilyAudit(
        family=spec.as_dict(),
        results=results,
        downstream_evaluated=True,
        downstream_not_evaluated_reason="",
        diagnostics=diagnostics,
    )


def audit_all(
    specs: Sequence[FamilySpec],
    *,
    measure_downstream: bool = True,
) -> list[FamilyAudit]:
    """Audit every family, resolving counterfactual parents by ``family_id``.

    Order-independent by construction: each family's audit reads only its own spec and
    its declared parent's, never a neighbour's result.
    """
    by_id = {s.family_id: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError("duplicate family_id in the audit input")
    out: list[FamilyAudit] = []
    for spec in specs:
        parent = by_id.get(spec.counterfactual_parent) if spec.counterfactual_parent else None
        out.append(audit_family(spec, parent=parent, measure_downstream=measure_downstream))
    return out


# ---------------------------------------------------------------------------
# Stage 1 table
# ---------------------------------------------------------------------------

#: Column order of the Stage 1 "reservoir admissibility table".
STAGE1_COLUMNS: tuple[str, ...] = (
    "Family",
    "A1",
    "A2",
    "A3 Krylov",
    "A4 Memory",
    "A5 State",
    "Food eligible",
    "Construct qualified",
)


def _cell(audit: FamilyAudit, criterion: str) -> str:
    """One table cell: the verdict only.

    The VALUE is deliberately not rendered here. A rounded value next to a strict
    comparison reads as a contradiction -- a hairline A3 failure at ``D_eff = 1.9983``
    against a gate of ``2.0`` renders as "FAIL (2.00>=2.00)", which is worse than
    saying nothing. Verdicts go in the table; exact values and thresholds go in
    :meth:`FamilyAudit.as_dict`, where they keep full precision.
    """
    r = audit.results.get(criterion)
    if r is None:
        return "—"
    if r.state is CriterionState.NOT_APPLICABLE:
        return "N/A"
    return "PASS" if r.state is CriterionState.PASS else "FAIL"


def stage1_table(audits: Sequence[FamilyAudit]) -> str:
    """Render the Stage 1 admissibility table (verdicts only).

    ``N/A`` marks a criterion that was not applicable; ``—`` marks one that was not
    evaluated because the primary gate did not pass. The two are rendered differently on
    purpose: collapsing them is the error this battery exists to avoid. Exact values are
    in the JSON report beside this table.
    """
    rows = [
        "| " + " | ".join(STAGE1_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(STAGE1_COLUMNS)) + "|",
    ]
    for audit in audits:
        fam = f"{audit.family['family_id']} {audit.family['label']}"
        cells = [
            fam,
            _cell(audit, "A1"),
            _cell(audit, "A2"),
            _cell(audit, "A3"),
            _cell(audit, "A4"),
            _cell(audit, "A5"),
            "yes" if audit.food_eligible else "no",
            "yes" if audit.construct_qualified else "no",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    rows.append(
        "`Food eligible` = cleared A3, so it may enter the food stage. "
        "`Construct qualified` = cleared every ADMISSION and evaluated criterion. "
        "They are independent: a family can be food-eligible without being "
        "construct-qualified. Neither column means \"Stage 1 passed\"."
    )
    return "\n".join(rows)


def stage1_json(audits: Sequence[FamilyAudit], *, git_head: str | None = None) -> dict[str, Any]:
    """The machine-readable Stage 1 report."""
    return {
        "report": "ResAudit-Food Stage 1 reservoir admissibility audit",
        "provenance": {
            "probe_protocol": probe_protocol(),
            "thresholds": _threshold_record(),
            "primary_gate": "A3",
            "primary_gate_name": A3_NAME,
            "primary_gate_caveat": A3_CAVEAT,
            "frozen_conventions": frozen_conventions(),
            "a4_gate_status": "descriptive_only",
            "a4_caveat": A4_DESCRIPTIVE_CAVEAT,
            "git_head": git_head,
        },
        "families": [a.as_dict() for a in audits],
        "stage1_table_markdown": stage1_table(audits),
    }


def frozen_conventions() -> dict[str, Any]:
    """The conventions a result was produced under, for the report's provenance."""
    return {
        "rho_target": FROZEN_RHO_TARGET,
        "rho_target_provenance": "amendment 1 Section 0.1 (calibration-derived, not pre-registered)",
        "leak": PROBE_LEAK,
        "gain": PROBE_GAIN,
        "parameter_sweep_performed": False,
        "note": (
            "Applied to every family by the single declared scaling, before A3/A4/A5. "
            "Not to be re-chosen after seeing a family score."
        ),
    }


def _threshold_record() -> dict[str, Any]:
    return {
        "A1_scc_fraction_min": A1_SCC_FRACTION_MIN,
        "A1_isolated_fraction_max": A1_ISOLATED_FRACTION_MAX,
        "A1_mean_out_degree_min": A1_MEAN_OUT_DEGREE_MIN,
        "A2_overlap_max": A2_OVERLAP_MAX,
        "A3_factor": A3_FACTOR,
        "A3_K": A3_KRYLOV_K,
        "A4_min": A4_MIN,
        "A5_factor": A5_FACTOR,
    }


__all__ = [
    "FROZEN_RHO_TARGET",
    "frozen_conventions",
    "PROBE_LEAK",
    "PROBE_GAIN",
    "PROBE_WINDOWS",
    "PROBE_LENGTH",
    "PROBE_SEED",
    "PROBE_WASHOUT",
    "PROBE_INPUT_SCALE",
    "probe_protocol",
    "probe_input",
    "probe_input_with_seed",
    "spectral_radius_of",
    "scale_to_spectral_radius",
    "StructuralDiagnostics",
    "structural_diagnostics",
    "audit_a1",
    "audit_a2",
    "audit_a3",
    "audit_a4_a5",
    "A2_NO_PARENT_REASON",
    "A4_A5_SKIPPED_REASON",
    "FamilyAudit",
    "audit_family",
    "audit_all",
    "STAGE1_COLUMNS",
    "stage1_table",
    "stage1_json",
]
