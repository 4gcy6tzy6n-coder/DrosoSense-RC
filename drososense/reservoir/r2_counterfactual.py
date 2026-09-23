"""R2 as a TRUE wiring-only counterfactual (v2 D6 / criteria C4.1-C4.8).

THE DEFECT THIS REPAIRS. ``make_degree_rewired`` rebuilds R2 from a Maslov-Sneppen
double-edge swap and then writes ``np.ones(n_edges)`` into the matrix -- so R2 is a
**uniform-weight graph**: it destroys the synapse-count weight structure *as well as*
the wiring, and R0-vs-R2 was a JOINT contrast. The v1 conclusion "the wiring does not
matter" is therefore not what that comparison measured. Two further v1 properties are
worth stating because v2 changes them deliberately:

* the v1 chain rescales R2 to R0's spectral radius, which multiplies every weight by a
  constant and so changes the weight multiset (C4.2 forbids that);
* the v1 duplicate check is ``np.any`` over the whole edge array per attempt, which is
  the projection that reached **71 days** at M = 14.8 M (C4.7 caps the build at 10 min).

WHAT v2 PRESERVES, AND WHAT IT DOES NOT. Signed (D6), verbatim in substance: same
nodes, the same input-population membership, the same directed degree sequence, the
same GLOBAL weight multiset, and the same PER-SOURCE outgoing weight multiset /
out-strength; only the wiring changes. Each swap below rewrites

    (a -> b, w1), (c -> d, w2)   ==>   (a -> d, w1), (c -> b, w2)

so both sources keep their own weight, out-degree and out-strength; both targets keep
their in-degree; and the global weight multiset is untouched. What is NOT preserved is
exact per-node weighted **in-strength** -- the two targets exchange weights -- which is
why C4.4 is a pre-registered distribution tolerance (median relative error <= 5 %)
rather than an equality, exactly as the pre-registration says.

The spectral radius is therefore NOT matched either, and that is a consequence of the
signed list rather than an oversight: preserving the weight multiset and rescaling the
graph to a target radius are mutually exclusive. Both radii are reported so the reader
can see the size of the difference instead of having it hidden by a rescale.

C4.6 exists because a swap chain that has not mixed can be degree-preserving and still
be almost R0; the overlap ``|E_R0 ∩ E_R2| / |E_R0|`` is what makes "rewired" checkable
rather than asserted (v1 measured 0.058 at N=1000, with a fully-mixed expectation of
0.055).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

#: C4.6's ceiling on the R0/R2 edge overlap.
MIXING_OVERLAP_LIMIT = 0.20

#: C4.4's tolerance on the weighted in-strength distribution.
IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT = 0.05

#: C4.7's wall-time budget for one graph at N=1000, and the hard cap.
BUILD_TIME_BUDGET_S = 600.0
BUILD_TIME_HARD_CAP_S = 3600.0

#: How often the (cheap but not free) overlap check runs inside the chain.
_OVERLAP_CHECK_EVERY = 2_000

#: The declared weight-matching rule for the SECOND edge of a swap.
#:
#: Because the weight stays with its source (C4.3), a swap makes the two targets
#: EXCHANGE weights: target ``b`` loses ``w1`` and gains ``w2``. If ``w2 == w1`` the
#: exchange moves nothing, so in-strength is preserved exactly and C4.4 is satisfied
#: with room to spare; if the weights differ, the in-strength distribution drifts.
#:
#: The delivered substrate is quantized (80,443 edges over 363 distinct
#: synapse-count weights, 26,035 of them exactly 1), so exact-weight partners are
#: plentiful: measured, 99.87 % of edges have at least one, and the median weight
#: class holds 6,802 edges. The chain therefore draws the second edge from the same
#: weight class, and only when a class is a singleton falls back to the nearest
#: weights within :data:`WEIGHT_MATCH_FALLBACK_TOLERANCE` (rejecting the proposal if
#: even that finds nobody). This is a declared rule about the GRAPH, not a knob fitted
#: to any model result.
WEIGHT_MATCH_FALLBACK_TOLERANCE = 0.25

#: How many candidates from the weight class are examined before a proposal is
#: abandoned. The acceptance conditions are unchanged; this only avoids paying for
#: candidates that are illegal by construction (shared endpoint, self-loop, duplicate).
PARTNER_SEARCH_TRIES = 16

#: Accepted-swap counts (as multiples of |E|) at which the mixing curve is sampled.
MIXING_CURVE_MULTIPLES: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)


class RewireError(ValueError):
    """Raised when a counterfactual cannot be built as declared."""


@dataclass(frozen=True)
class RewireReport:
    """What the swap chain actually did (C4.8 requires this on every R2 result).

    Attributes:
        swaps_attempted: Swap proposals drawn.
        swaps_accepted: Proposals applied.
        swaps_rejected_duplicate: Rejected because a new edge already exists.
        swaps_rejected_self_loop: Rejected because a new edge would be a self-loop.
        swaps_rejected_shared_endpoint: Rejected because the two edges shared an
            endpoint and the swap would be a no-op.
        n_edges: Edges in the graph (including any self-loops, which are fixed).
        n_self_loops: Self-loops, which the chain never moves.
        overlap_initial: ``|E_R0 ∩ E_R2| / |E_R0|`` before any swap (1.0).
        overlap_final: The same after the chain stopped -- C4.6's measurement.
        original_edges_retained_fraction: ``1 - overlap_final_removed``, i.e. how much
            of R0's wiring survives (C4.8's "original-edge retention").
        seconds: Wall time of the chain.
        stopped_because: ``"target_overlap"``, ``"attempts_exhausted"`` or
            ``"time_budget"`` -- a chain that ran out of budget is reported as such.
    """

    swaps_attempted: int
    swaps_accepted: int
    swaps_rejected_duplicate: int
    swaps_rejected_self_loop: int
    swaps_rejected_shared_endpoint: int
    swaps_weight_matched_exact: int
    swaps_weight_matched_nearest: int
    swaps_rejected_no_weight_partner: int
    swaps_rejected_no_legal_partner: int
    swaps_accepted_exact_weight: int
    swaps_accepted_near_weight: int
    allow_near_weights: bool
    mixing_curve: tuple[tuple[int, float], ...]
    n_edges: int
    n_self_loops: int
    overlap_initial: float
    overlap_final: float
    seconds: float
    stopped_because: str
    target_overlap: float

    @property
    def original_edges_retained_fraction(self) -> float:
        return float(self.overlap_final)

    @property
    def mixing_reached(self) -> bool:
        return bool(self.overlap_final <= self.target_overlap + 1e-12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "swaps_attempted": int(self.swaps_attempted),
            "swaps_accepted": int(self.swaps_accepted),
            "swaps_rejected_duplicate": int(self.swaps_rejected_duplicate),
            "swaps_rejected_self_loop": int(self.swaps_rejected_self_loop),
            "swaps_rejected_shared_endpoint": int(self.swaps_rejected_shared_endpoint),
            "swaps_weight_matched_exact": int(self.swaps_weight_matched_exact),
            "swaps_weight_matched_nearest": int(self.swaps_weight_matched_nearest),
            "swaps_rejected_no_weight_partner": int(
                self.swaps_rejected_no_weight_partner
            ),
            "swaps_rejected_no_legal_partner": int(self.swaps_rejected_no_legal_partner),
            "swaps_accepted_exact_weight": int(self.swaps_accepted_exact_weight),
            "swaps_accepted_near_weight": int(self.swaps_accepted_near_weight),
            "allow_near_weights": bool(self.allow_near_weights),
            "fraction_of_accepted_swaps_that_moved_a_weight": (
                self.swaps_accepted_near_weight / self.swaps_accepted
                if self.swaps_accepted
                else 0.0
            ),
            # The mixing CURVE, not just its endpoint: without it a reader cannot tell
            # whether 20 % overlap is a plateau or an arbitrary stopping point.
            "mixing_curve": [
                {"accepted_swaps": int(swaps), "edge_overlap": float(value)}
                for swaps, value in self.mixing_curve
            ],
            "acceptance_rate": (
                self.swaps_accepted / self.swaps_attempted if self.swaps_attempted else 0.0
            ),
            "n_edges": int(self.n_edges),
            "n_self_loops": int(self.n_self_loops),
            "overlap_initial": float(self.overlap_initial),
            "overlap_final": float(self.overlap_final),
            "mixing_overlap_limit": MIXING_OVERLAP_LIMIT,
            "mixing_reached": self.mixing_reached,
            "original_edge_retention": self.original_edges_retained_fraction,
            "seconds": float(self.seconds),
            "stopped_because": self.stopped_because,
        }


def weight_preserving_degree_rewire(
    matrix: sp.spmatrix,
    *,
    seed: int,
    target_overlap: float = MIXING_OVERLAP_LIMIT,
    max_attempts: int | None = None,
    time_budget_s: float = BUILD_TIME_BUDGET_S,
    allow_near_weights: bool = False,
) -> tuple[sp.csr_matrix, RewireReport]:
    """Rewire the graph, preserving degree AND the weight structure (v2 R2).

    Args:
        matrix: The R0 matrix (any sparse format).
        seed: Seed of the swap sequence. The chain is reproducible from it alone.
        target_overlap: Stop once the R0/R2 edge overlap reaches this (C4.6).
        max_attempts: Cap on swap proposals; defaults to 200 per edge, which is far
            above what a graph of this density needs to mix.
        time_budget_s: Wall-time budget (C4.7). The chain stops and says so when it
            expires, rather than being silently truncated.
        allow_near_weights: Whether a proposal whose edge has no EXACT-weight partner
            may fall back to the nearest weights within
            :data:`WEIGHT_MATCH_FALLBACK_TOLERANCE`. Default ``False``, and that
            default is load-bearing: a near-weight exchange moves the two targets'
            in-strength, and the drift accumulates over millions of swaps (measured:
            median relative error 0.32 with the fallback on, exactly 0 with it off).

    Returns:
        ``(rewired, report)``.

    Raises:
        RewireError: If the graph has duplicate edges (a CSR after a graph build
            should not -- a duplicate would break the degree accounting the whole
            counterfactual rests on) or fewer than two swappable edges.
    """
    if sp.issparse(matrix) and matrix.format == "coo":
        # A COO can carry the same pair twice; ``tocsr()`` SUMS those entries, which
        # would silently change the weight multiset the counterfactual is supposed to
        # preserve. The caller's graph is what it says it is, so this is refused before
        # any conversion.
        pairs = list(zip(matrix.row.tolist(), matrix.col.tolist()))
        if len(pairs) != len(set(pairs)):
            raise RewireError(
                f"the input has duplicate (row, column) entries "
                f"({len(pairs)} entries, {len(set(pairs))} distinct); CSR would sum "
                f"them and the weight multiset would not be the caller's"
            )
    csr = matrix.tocsr()
    coo = csr.tocoo()
    rows = coo.row.astype(np.int64).copy()
    cols = coo.col.astype(np.int64).copy()
    weights = coo.data.astype(np.float64).copy()
    n_edges = int(rows.size)
    n_nodes = int(csr.shape[0])

    keys = list(zip(rows.tolist(), cols.tolist()))
    if len(set(keys)) != n_edges:
        raise RewireError(
            f"the graph has duplicate (row, column) pairs ({n_edges} entries, "
            f"{len(set(keys))} distinct): the degree accounting assumes one entry per edge"
        )
    self_loops = int((rows == cols).sum())
    swappable = np.flatnonzero(rows != cols)
    if swappable.size < 2:
        raise RewireError(
            f"{swappable.size} non-self-loop edge(s): a double-edge swap needs two"
        )

    weights_before = np.sort(weights)
    out_strength_before = _per_source_multisets(rows, weights)
    out_degree_before = np.bincount(rows, minlength=n_nodes)
    in_degree_before = np.bincount(cols, minlength=n_nodes)

    original = set(keys)
    present = set(keys)
    overlap = len(original)

    # The weight classes are FIXED for the whole chain (only rows/cols move), so one
    # sorted index serves every proposal.
    order = np.argsort(weights, kind="stable")
    sorted_weights = weights[order]

    def weight_class_window(value: float, tolerance: float = 0.0) -> tuple[int, int]:
        low = int(np.searchsorted(sorted_weights, value * (1.0 - tolerance), side="left"))
        high = int(np.searchsorted(sorted_weights, value * (1.0 + tolerance), side="right"))
        return low, high

    rng = np.random.default_rng(int(seed))
    attempts = int(max_attempts) if max_attempts is not None else 200 * n_edges
    started = time.monotonic()
    accepted = rejected_dup = rejected_loop = rejected_shared = 0
    matched_exact = matched_nearest = rejected_no_partner = 0
    rejected_no_legal = 0
    accepted_exact = accepted_near = 0
    stopped = "attempts_exhausted"
    overlap_final = 1.0
    curve_targets = sorted({int(round(m * n_edges)) for m in MIXING_CURVE_MULTIPLES})
    curve: list[tuple[int, float]] = [(0, 1.0)]
    curve_next = 0

    proposals = 0
    for proposal in range(attempts):
        proposals = proposal + 1
        if proposal % _OVERLAP_CHECK_EVERY == 0:
            overlap_final = overlap / n_edges if n_edges else 1.0
            if overlap_final <= target_overlap + 1e-12:
                stopped = "target_overlap"
                break
            if time.monotonic() - started > time_budget_s:
                stopped = "time_budget"
                break
        first, second = rng.integers(0, swappable.size, size=2)
        if first == second:
            continue
        i = int(swappable[int(rng.integers(0, swappable.size))])
        w1 = float(weights[i])
        # the partner comes from the same weight class: exchanging equal weights moves
        # no in-strength at all, which is what C4.4 asks for
        lo, hi = weight_class_window(w1)
        near_weight = False
        if hi - lo < 2:
            if not allow_near_weights:
                rejected_no_partner += 1
                continue
            lo, hi = weight_class_window(w1, tolerance=WEIGHT_MATCH_FALLBACK_TOLERANCE)
            if hi - lo < 2:
                rejected_no_partner += 1
                continue
            matched_nearest += 1
            near_weight = True
        else:
            matched_exact += 1

        # Search the weight class for a LEGAL partner instead of drawing one and
        # rejecting: measured on the delivered substrate, 61 % of all proposals were
        # thrown away because the two edges shared an endpoint, which is what made the
        # chain slow rather than stuck. The acceptance conditions are unchanged -- this
        # only stops paying for candidates that cannot be used.
        a, b = int(rows[i]), int(cols[i])
        j = -1
        for _ in range(PARTNER_SEARCH_TRIES):
            if hi - lo < 2:
                break
            candidate = int(order[int(rng.integers(lo, hi))])
            if candidate == i:
                continue
            c, d = int(rows[candidate]), int(cols[candidate])
            if a == c or b == d:
                rejected_shared += 1
                continue
            if a == d or c == b:
                rejected_loop += 1
                continue
            if (a, d) in present or (c, b) in present:
                rejected_dup += 1
                continue
            j = candidate
            break
        if j < 0:
            rejected_no_legal += 1
            continue
        c, d = int(rows[j]), int(cols[j])

        # apply: both sources keep their own weight
        present.discard((a, b))
        present.discard((c, d))
        if (a, b) in original:
            overlap -= 1
        if (c, d) in original:
            overlap -= 1
        rows[i], cols[i] = a, d
        rows[j], cols[j] = c, b
        present.add((a, d))
        present.add((c, b))
        if (a, d) in original:
            overlap += 1
        if (c, b) in original:
            overlap += 1
        accepted += 1
        if near_weight:
            accepted_near += 1
        else:
            accepted_exact += 1
        while curve_next < len(curve_targets) and accepted >= curve_targets[curve_next]:
            curve.append((int(accepted), overlap / n_edges if n_edges else 1.0))
            curve_next += 1

    seconds = time.monotonic() - started
    overlap_final = overlap / n_edges if n_edges else 1.0
    if overlap_final <= target_overlap + 1e-12:
        stopped = "target_overlap"
    # the curve must END at the point the chain actually stopped, or its last entry
    # would be the last milestone rather than the answer
    if not curve or curve[-1][0] != accepted or curve[-1][1] != overlap_final:
        curve.append((int(accepted), float(overlap_final)))
    rewired = sp.csr_matrix(
        (weights, (rows, cols)), shape=(n_nodes, n_nodes)
    )

    # The preservation claims are MEASURED, not asserted from the construction.
    weights_after = np.sort(rewired.tocoo().data.astype(np.float64))
    if not np.array_equal(weights_before, weights_after):
        raise RewireError("the weight multiset changed during the swap chain")
    if not np.array_equal(np.bincount(rewired.tocoo().row, minlength=n_nodes), out_degree_before):
        raise RewireError("an out-degree changed during the swap chain")
    if not np.array_equal(np.bincount(rewired.tocoo().col, minlength=n_nodes), in_degree_before):
        raise RewireError("an in-degree changed during the swap chain")
    after_multisets = _per_source_multisets(rewired.tocoo().row.astype(np.int64), rewired.tocoo().data)
    if after_multisets != out_strength_before:
        raise RewireError("a per-source outgoing weight multiset changed during the chain")

    report = RewireReport(
        swaps_attempted=int(proposals),
        swaps_accepted=int(accepted),
        swaps_rejected_duplicate=int(rejected_dup),
        swaps_rejected_self_loop=int(rejected_loop),
        swaps_rejected_shared_endpoint=int(rejected_shared),
        swaps_weight_matched_exact=int(matched_exact),
        swaps_weight_matched_nearest=int(matched_nearest),
        swaps_rejected_no_weight_partner=int(rejected_no_partner),
        swaps_rejected_no_legal_partner=int(rejected_no_legal),
        swaps_accepted_exact_weight=int(accepted_exact),
        swaps_accepted_near_weight=int(accepted_near),
        allow_near_weights=bool(allow_near_weights),
        mixing_curve=tuple(curve),
        n_edges=n_edges,
        n_self_loops=self_loops,
        overlap_initial=1.0,
        overlap_final=float(overlap_final),
        seconds=float(seconds),
        stopped_because=stopped,
        target_overlap=float(target_overlap),
    )
    if seconds > BUILD_TIME_HARD_CAP_S:
        raise RewireError(
            f"the swap chain took {seconds:.1f}s, past the {BUILD_TIME_HARD_CAP_S:.0f}s "
            f"hard cap (C4.7)"
        )
    return rewired, report


def _hash_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def multiset_hash(values: np.ndarray) -> str:
    """SHA-256 of a CANONICALLY SORTED value array.

    The conservation claims are exact equalities, so they are published as digests
    over sorted values rather than as "max difference == 0": a digest is checkable by
    a reader who has neither the graph nor the code, and it cannot be satisfied by a
    tolerance. Sorting first makes it a MULTISET hash, which is what C4.2 and C4.3
    ask about -- the weights may move, their multiset may not.
    """
    array = np.sort(np.asarray(values, dtype=np.float64))
    return _hash_bytes(np.ascontiguousarray(array).tobytes())


def degree_sequence_hash(counts: np.ndarray) -> str:
    """SHA-256 of a degree sequence (the array is ordered by node, so not sorted)."""
    return _hash_bytes(np.ascontiguousarray(np.asarray(counts, dtype=np.int64)).tobytes())


def per_source_multiset_hash(rows: np.ndarray, weights: np.ndarray) -> str:
    """SHA-256 of the per-source outgoing weight multisets (C4.3)."""
    import json as _json

    payload = _json.dumps(
        [[int(source), [float(w) for w in values]]
         for source, values in _per_source_multisets(rows, weights)],
        separators=(",", ":"),
    )
    return _hash_bytes(payload.encode("utf-8"))


def _per_source_multisets(rows: np.ndarray, weights: np.ndarray) -> list[tuple]:
    """``[(source, sorted tuple of its outgoing weights), ...]`` -- C4.3's object."""
    buckets: dict[int, list[float]] = {}
    for row, weight in zip(rows.tolist(), weights.tolist()):
        buckets.setdefault(int(row), []).append(float(weight))
    return [(source, tuple(sorted(values))) for source, values in sorted(buckets.items())]


# ---------------------------------------------------------------------------
# C4.1 - C4.6: the criteria, measured
# ---------------------------------------------------------------------------
def counterfactual_quality(
    reference: sp.spmatrix,
    rewired: sp.spmatrix,
    report: RewireReport,
    *,
    population_rows_r0: np.ndarray | None = None,
    population_rows_r2: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure C4.1-C4.6 on a built pair, so the claims are facts not assertions.

    Args:
        reference: R0.
        rewired: R2.
        report: The swap chain's own report.
        population_rows_r0: The input population's rows in R0, for C4.5.
        population_rows_r2: The same in R2, derived independently from the same
            annotation. The criterion is that these two agree -- comparing an array
            with itself would measure nothing.

    Returns:
        A JSON-serialisable mapping: each criterion's observed value, its threshold
        and its verdict, plus the two spectral radii and the in-strength summary.
    """
    from drososense.reservoir.connectome_reservoir import spectral_radius

    a = reference.tocsr()
    b = rewired.tocsr()
    a_coo, b_coo = a.tocoo(), b.tocoo()
    a_rows, a_cols = a_coo.row.astype(np.int64), a_coo.col.astype(np.int64)

    b_rows = b_coo.row.astype(np.int64)
    b_cols = b_coo.col.astype(np.int64)
    out_a = np.bincount(a_rows, minlength=a.shape[0])
    out_b = np.bincount(b_rows, minlength=b.shape[0])
    in_a = np.bincount(a_cols, minlength=a.shape[0])
    in_b = np.bincount(b_cols, minlength=b.shape[0])
    degree_ok = bool(np.array_equal(out_a, out_b) and np.array_equal(in_a, in_b))
    max_out_delta = int(np.max(np.abs(out_a - out_b))) if out_a.size else 0
    max_in_delta = int(np.max(np.abs(in_a - in_b))) if in_a.size else 0

    weights_a = a_coo.data.astype(np.float64)
    weights_b = b_coo.data.astype(np.float64)
    weights_ok = bool(np.array_equal(np.sort(weights_a), np.sort(weights_b)))
    global_hash_a = multiset_hash(weights_a)
    global_hash_b = multiset_hash(weights_b)

    per_source_hash_a = per_source_multiset_hash(a_rows, weights_a)
    per_source_hash_b = per_source_multiset_hash(b_rows, weights_b)
    per_source_ok = bool(per_source_hash_a == per_source_hash_b)

    in_strength_a = np.asarray(a.sum(axis=0)).ravel().astype(np.float64)
    in_strength_b = np.asarray(b.sum(axis=0)).ravel().astype(np.float64)
    positive = in_strength_a > 0
    relative_error = (
        np.abs(in_strength_b[positive] - in_strength_a[positive])
        / np.maximum(np.abs(in_strength_a[positive]), 1e-12)
    )
    median_relative_error = float(np.median(relative_error)) if relative_error.size else float("nan")
    # median alone can hide a few badly moved targets, so the tail is reported beside
    # it -- as a DIAGNOSTIC, with no new gate (the signed criterion is the median)
    p90_relative_error = (
        float(np.quantile(relative_error, 0.9)) if relative_error.size else float("nan")
    )
    max_relative_error = float(relative_error.max()) if relative_error.size else float("nan")

    population_ok: bool | None = None
    if population_rows_r0 is not None and population_rows_r2 is not None:
        population_ok = bool(
            np.array_equal(
                np.sort(np.asarray(population_rows_r0, dtype=np.int64)),
                np.sort(np.asarray(population_rows_r2, dtype=np.int64)),
            )
        )

    rho_a = float(spectral_radius(a, seed=0))
    rho_b = float(spectral_radius(b, seed=0))
    target = mixing_target(reference)
    mixing = {
        "measured_overlap": float(report.overlap_final),
        **target,
        "pass": bool(report.overlap_final <= target["threshold"] + 1e-12),
    }

    return {
        "C4.1_degree_sequences": {
            "observed": {
                "max_abs_in_degree_delta": max_in_delta,
                "max_abs_out_degree_delta": max_out_delta,
                "in_degree_hash_R0": degree_sequence_hash(in_a),
                "in_degree_hash_R2": degree_sequence_hash(in_b),
                "out_degree_hash_R0": degree_sequence_hash(out_a),
                "out_degree_hash_R2": degree_sequence_hash(out_b),
            },
            "threshold": "exact: max|delta| == 0 and equal hashes, in and out, unweighted",
            "pass": degree_ok,
        },
        "C4.2_global_weight_multiset": {
            "observed": {
                "multiset_hash_R0": global_hash_a,
                "multiset_hash_R2": global_hash_b,
                "n_edges_R0": int(weights_a.size),
                "n_edges_R2": int(weights_b.size),
            },
            "threshold": "exact: equal multiset hashes",
            "pass": weights_ok,
        },
        "C4.3_per_source_out_weight_multiset": {
            "observed": {
                "per_source_hash_R0": per_source_hash_a,
                "per_source_hash_R2": per_source_hash_b,
            },
            "threshold": "exact: equal per-source multiset hashes",
            "pass": per_source_ok,
        },
        "C4.4_in_strength_median_relative_error": {
            "observed": {
                "median": median_relative_error,
                "p90": p90_relative_error,
                "max": max_relative_error,
            },
            "threshold": f"median <= {IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT}",
            "pass": bool(median_relative_error <= IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT),
            "note": (
                "PRESERVED in distribution, not per node: a swap exchanges weights "
                "between two targets, so a target's in-strength moves while its "
                "in-degree does not. p90 and max are reported as DIAGNOSTICS and are "
                "not gates -- the signed criterion is the median. The chain draws the "
                "second edge from the same weight class, which makes the exchange move "
                "nothing at all whenever the class has a partner."
            ),
        },
        "C4.5_input_population_identical": {
            "observed": population_ok,
            "threshold": "exact equality",
            "pass": population_ok,
            "note": (
                "the counterfactual changes the WIRING only: the node set is identical, "
                "so the ORN population and every class label are identical by construction"
            ),
        },
        "C4.6_mixing_overlap": {
            "observed": report.overlap_final,
            "threshold": target["threshold"],
            "pass": bool(report.overlap_final <= target["threshold"] + 1e-12),
            "note": (
                f"stopped_because={report.stopped_because!r}, "
                f"{report.swaps_accepted} swap(s) accepted; threshold form "
                f"{target['form']}"
            ),
        },
        "C4.7_build_seconds": {
            "observed": report.seconds,
            "threshold": f"<= {BUILD_TIME_BUDGET_S}",
            "pass": bool(report.seconds <= BUILD_TIME_BUDGET_S),
        },
        "C4.8_report": report.as_dict(),
        "mixing_target": mixing,
        "verdict": verdict_header(
            degree_ok=degree_ok,
            weights_ok=weights_ok,
            per_source_ok=per_source_ok,
            population_ok=population_ok,
            median_relative_error=median_relative_error,
            overlap=report.overlap_final,
            seconds=report.seconds,
            mixing_reached=report.overlap_final <= target["threshold"] + 1e-12,
        ),
        "spectral_radius": {
            "R0": rho_a,
            "R2": rho_b,
            "ratio": (rho_b / rho_a) if rho_a else float("nan"),
            "note": (
                "NOT matched: preserving the global weight multiset (C4.2) and rescaling "
                "to a target radius are mutually exclusive. v1's rescale is exactly what "
                "made R2 a uniform-weight graph. Reported so the size of the difference "
                "is visible rather than hidden."
            ),
        },
        "in_strength": {
            "R0_total": float(in_strength_a.sum()),
            "R2_total": float(in_strength_b.sum()),
            "median_relative_error": median_relative_error,
            "p90_relative_error": (
                float(np.quantile(relative_error, 0.9)) if relative_error.size else float("nan")
            ),
        },
    }


def verdict_header(
    *,
    degree_ok: bool,
    weights_ok: bool,
    per_source_ok: bool,
    population_ok: bool | None,
    median_relative_error: float,
    overlap: float,
    seconds: float,
    mixing_reached: bool,
) -> dict[str, Any]:
    """C4's judgement head: eight lines, and C4 passes iff every line passes.

    Kept deliberately flat so the verdict cannot hide inside a nested report:

    ```
    degree_exact              PASS/FAIL
    global_weights_exact      PASS/FAIL
    per_source_weights_exact  PASS/FAIL
    input_population_same     PASS/FAIL
    median_in_strength_err    value <= 0.05
    edge_overlap              value <= max(0.20, 1.10 * E[overlap])   [amendment 1]
    build_time                value <= 600 s
    mixing_declared           PASS/FAIL
    C4                        PASS iff all pass
    ```

    ``build_time`` compares against the SIGNED 600 s threshold, not the 1 h hard cap:
    a counterfactual that is fair but too slow to generate per seed is a construct
    failure, not a success with a caveat. Equally, reaching only a high overlap inside
    the budget is a failure -- "close" is not a pass.
    """
    lines = {
        "degree_exact": bool(degree_ok),
        "global_weights_exact": bool(weights_ok),
        "per_source_weights_exact": bool(per_source_ok),
        "input_population_same": None if population_ok is None else bool(population_ok),
        "median_in_strength_err": float(median_relative_error),
        "edge_overlap": float(overlap),
        "build_time": float(seconds),
        "mixing_declared": bool(mixing_reached),
    }
    checks = {
        "degree_exact": bool(degree_ok),
        "global_weights_exact": bool(weights_ok),
        "per_source_weights_exact": bool(per_source_ok),
        "input_population_same": True if population_ok is None else bool(population_ok),
        "median_in_strength_err": bool(
            median_relative_error <= IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT
        ),
        # amendment 1: the caller passes the overlap ALREADY compared against the
        # floor-relative threshold through ``mixing_reached``; the absolute form is kept
        # as a belt-and-braces check so a caller that forgets cannot pass a 0.9 overlap
        "edge_overlap": bool(
            mixing_reached and overlap <= max(MIXING_OVERLAP_LIMIT, 1e-12) * 1.0
            if overlap > MIXING_OVERLAP_LIMIT
            else True
        ),
        "build_time": bool(seconds <= BUILD_TIME_BUDGET_S),
        "mixing_declared": bool(mixing_reached),
    }
    return {
        "lines": lines,
        "thresholds": {
            "median_in_strength_err": IN_STRENGTH_MEDIAN_RELATIVE_ERROR_LIMIT,
            "edge_overlap": MIXING_OVERLAP_LIMIT,
            "build_time": BUILD_TIME_BUDGET_S,
        },
        "checks": checks,
        "C4": bool(all(checks.values())),
    }


def cell_type_pair_matrix(
    rows: np.ndarray, cols: np.ndarray, classes: np.ndarray
) -> dict[str, dict[str, int]]:
    """``M[a][b] = #{u -> v : type(u) = a, type(v) = b}`` for an edge list.

    A DIAGNOSTIC, deliberately not a gate. R2 preserves node identities, the directed
    degree sequence and the per-source weight multisets, but it does NOT preserve
    ORN->PN, PN->KC and the rest of the type-level connectivity -- which is exactly
    what "wiring-only counterfactual" means. Reporting both matrices means that if R0
    ever beats R2, the next question ("is the advantage the specific neuron-to-neuron
    wiring, or the cell-type-level grammar?") can be asked with the numbers already on
    the table, instead of re-running the phase.

    **It must not be turned into a constraint on R2.** Making R2 type-pair preserving
    now would change the frozen primary counterfactual; that belongs to a separately
    pre-registered secondary analysis if it is ever needed.
    """
    out: dict[str, dict[str, int]] = {}
    for source, target in zip(classes[rows].tolist(), classes[cols].tolist()):
        out.setdefault(str(source), {})
        out[str(source)][str(target)] = out[str(source)].get(str(target), 0) + 1
    return {source: dict(sorted(targets.items())) for source, targets in sorted(out.items())}


# ---------------------------------------------------------------------------
# v2 amendment 1 (signed): the measurement object of C4.2/C4.3/C4.4
# ---------------------------------------------------------------------------
#: The declared expectation estimator behind C4.6's floor-relative form.
#:
#: Measured on the C2 substrate: 0.2022, i.e. AT the absolute 0.20 threshold, so the
#: absolute form alone cannot discriminate a mixed graph from an unmixed one there. The
#: criterion is therefore ``overlap <= max(0.20, 1.10 * E[overlap])``: the absolute 0.20
#: REMAINS the operative threshold wherever the substrate's floor is comfortably below it
#: (a sparse substrate must still mix to 0.20, and this amendment must not make that
#: easier), and the floor-relative form takes over only when the floor reaches it.
#:
#: THE OWNER APPROVED THE DESCRIBED SEMANTICS, NOT THE LITERAL ``min``. The option text put
#: to them wrote ``min`` while describing ``max`` ("keep the absolute threshold where it is
#: above the floor; use the relative form where the floor reaches it"). ``min`` would
#: TIGHTEN the criterion on sparse substrates -- v1's floor of 0.0570 would have demanded
#: 0.0627 instead of 0.20 -- which is the opposite of the intent. The implementation
#: follows the described semantics, ``max``, and this note records the discrepancy rather
#: than leaving two versions to be confused later. Both forms accept v1's measured 0.058
#: against its floor of 0.0570, so no past verdict changes; what changes is which
#: substrates the criterion is strict on.
FLOOR_RELATIVE_FACTOR = 1.10


def expected_overlap_configuration_model(matrix: sp.spmatrix) -> float:
    """``E[overlap]`` for a uniformly random graph with the same degree sequences.

    For each original edge ``(u -> v)`` the probability that a random out-stub of ``u``
    lands on ``v`` is ``out_u * in_v / M`` (configuration model), so

        ``E[overlap] = (1/M) * sum over R0's edges of (out_u * in_v / M)``.

    This is the floor that makes C4.6 substrate-dependent, and it is computed from R0
    alone -- no result enters it.
    """
    csr = matrix.tocsr()
    coo = csr.tocoo()
    out_degree = np.bincount(coo.row.astype(np.int64), minlength=csr.shape[0]).astype(float)
    in_degree = np.bincount(coo.col.astype(np.int64), minlength=csr.shape[0]).astype(float)
    m = float(csr.nnz)
    if m == 0:
        return float("nan")
    per_edge = out_degree[coo.row.astype(np.int64)] * in_degree[coo.col.astype(np.int64)] / m
    return float(per_edge.sum() / m)


def mixing_target(matrix: sp.spmatrix, *, absolute: float = MIXING_OVERLAP_LIMIT) -> dict[str, Any]:
    """C4.6's threshold, absolute and floor-relative (amendment 1)."""
    expected = expected_overlap_configuration_model(matrix)
    relative = FLOOR_RELATIVE_FACTOR * expected
    return {
        "absolute": float(absolute),
        "expected_overlap_configuration_model": expected,
        "floor_relative": float(relative),
        "threshold": float(max(absolute, relative)),
        "form": "max(absolute 0.20, 1.10 * E[overlap])",
        "factor": FLOOR_RELATIVE_FACTOR,
    }


@dataclass(frozen=True)
class WiringCounterfactual:
    """R0 and R2 built under the signed measurement object (amendment 1).

    Attributes:
        raw_r0: The selected substrate as the connectome stores it (synapse-count
            weights), before any normalization.
        raw_r2: The same graph, rewired, with every raw weight in place.
        r0: R0 as the reservoir sees it: declared normalization + spectral rescale.
        r2: R2 preprocessed IDENTICALLY -- same normalization, and R0's own scale
            factor -- so the two differ in wiring only and share one weight scale.
        report: The swap chain's report (C4.8).
        conservation_raw: The raw-object conservations, measured.
        composition_scale: R0's rescale factor, applied to both.
        rho_r0, rho_r2: The two spectral radii after pre-processing. They differ,
            and that difference is disclosed rather than hidden by a second rescale.
    """

    raw_r0: sp.csr_matrix
    raw_r2: sp.csr_matrix
    r0: sp.csr_matrix
    r2: sp.csr_matrix
    report: RewireReport
    conservation_raw: dict[str, Any]
    composition_scale: float
    rho_r0: float
    rho_r2: float
    #: The same R2 pre-processed the OTHER way -- each graph rescaled to the target radius,
    #: which is the family convention -- kept so the cost of that choice is measurable
    #: rather than asserted.
    r2_rho_matched: Any = None
    rho_matched_scale: float = float("nan")


def normalize_with_scale(matrix: sp.spmatrix, normalization: str, scale: float) -> sp.csr_matrix:
    """Apply a declared DATA-3 normalization, then ONE explicit scale factor.

    ``load_reservoir_topology_from_npz`` rescales to a target radius, which multiplies
    each graph by its own constant. Under amendment 1 the two graphs must share ONE scale,
    so the normalization is applied here and the caller passes R0's factor for both.
    """
    from drososense.reservoir.connectome_reservoir import ALLOWED_NORMALIZATIONS

    if normalization not in ALLOWED_NORMALIZATIONS:
        raise RewireError(
            f"normalization {normalization!r} is not one of {list(ALLOWED_NORMALIZATIONS)}"
        )
    csr = matrix.tocsr()
    if normalization == "n1_pre_l1":
        # VERIFIED against the delivered block: ``norm_n1_pre_l1`` has ROW sums of exactly
        # 1 (median 1.0, min 0.9999997) and column sums that vary (median 0.562), i.e. it
        # is the PRESYNAPTIC (source) L1 normalization. Scaling the wrong axis would have
        # silently produced a different matrix from the one the pipeline uses.
        row = np.asarray(np.abs(csr).sum(axis=1)).ravel()
        row[row == 0] = 1.0
        normalized = sp.diags(1.0 / row) @ csr
    else:  # pragma: no cover - the other DATA-3 schemes are precomputed in the NPZ
        raise RewireError(
            f"amendment 1 requires a normalization that can be applied to a REWIRED graph; "
            f"{normalization!r} is only available as a stored block of the delivered matrix"
        )
    return sp.csr_matrix(normalized * float(scale))


def build_wiring_counterfactual(
    raw_block: sp.spmatrix,
    *,
    normalization: str,
    target_spectral_radius: float,
    seed: int,
    target_overlap: float = MIXING_OVERLAP_LIMIT,
    time_budget_s: float = BUILD_TIME_BUDGET_S,
    max_attempts: int | None = None,
) -> WiringCounterfactual:
    """Build R0 and R2 under the signed measurement object (v2 amendment 1).

    The counterfactual is built on the **biological graph** -- the selected substrate with
    its raw synapse-count weights -- because that is the object whose weights, degrees and
    per-source weight structure the counterfactual is supposed to keep. The declared
    normalization is then applied to BOTH graphs, with R0's scale factor, so the pair
    differs in wiring only and shares one weight scale.

    Why not rewire the normalized matrix (the first, failing design): normalizing changes a
    column's weights by a function of that column's wiring, so preserving the *normalized*
    multiset pins the wiring and collapses the swap space to ~5 edges per weight class.
    Measured: overlap stalls at 0.858 over 4.3 M swaps. On the raw graph the classes hold
    221.6 edges on average, the same chain reaches 0.200 in 14.4 s, and in-strength is
    preserved exactly.

    Args:
        raw_block: The selected substrate's raw (un-normalized) matrix.
        normalization: The declared DATA-3 normalization.
        target_spectral_radius: R0's target ``rho``.
        seed: Swap-chain seed.
        target_overlap: C4.6's absolute ceiling.
        time_budget_s: C4.7's budget.
        max_attempts: Optional cap on swap proposals.

    Returns:
        A :class:`WiringCounterfactual`.
    """
    raw_r2, report = weight_preserving_degree_rewire(
        raw_block,
        seed=seed,
        target_overlap=target_overlap,
        time_budget_s=time_budget_s,
        max_attempts=max_attempts,
    )
    raw_r0 = raw_block.tocsr()

    # R0's scale factor, from R0 alone
    unit = normalize_with_scale(raw_r0, normalization, 1.0)
    rho_unit = _spectral_radius(unit)
    scale = float(target_spectral_radius) / rho_unit if rho_unit > 0 else 1.0
    r0 = normalize_with_scale(raw_r0, normalization, scale)
    r2 = normalize_with_scale(raw_r2, normalization, scale)
    # The family convention (protocol §17, and what every v1 control does) is to rescale a
    # control to R0's spectral radius. That and a shared weight scale are mutually
    # exclusive, so the signed amendment fixes the scale and this reports what the
    # alternative would cost -- as a number the owner can decide on, not a hidden choice.
    rho_r2_own = _spectral_radius(normalize_with_scale(raw_r2, normalization, 1.0))
    rho_matched_scale = (
        float(target_spectral_radius) / rho_r2_own if rho_r2_own > 0 else 1.0
    )
    r2_rho_matched = normalize_with_scale(raw_r2, normalization, rho_matched_scale)

    raw_coo, rew_coo = raw_r0.tocoo(), raw_r2.tocoo()
    out_a = np.bincount(raw_coo.row.astype(np.int64), minlength=raw_r0.shape[0])
    out_b = np.bincount(rew_coo.row.astype(np.int64), minlength=raw_r0.shape[0])
    in_a = np.bincount(raw_coo.col.astype(np.int64), minlength=raw_r0.shape[0])
    in_b = np.bincount(rew_coo.col.astype(np.int64), minlength=raw_r0.shape[0])
    conservation_raw = {
        "degrees_exact": bool(np.array_equal(out_a, out_b) and np.array_equal(in_a, in_b)),
        "max_abs_out_degree_delta": int(np.max(np.abs(out_a - out_b))) if out_a.size else 0,
        "max_abs_in_degree_delta": int(np.max(np.abs(in_a - in_b))) if in_a.size else 0,
        "out_degree_hash_R0": degree_sequence_hash(out_a),
        "out_degree_hash_R2": degree_sequence_hash(out_b),
        "in_degree_hash_R0": degree_sequence_hash(in_a),
        "in_degree_hash_R2": degree_sequence_hash(in_b),
        "weight_multiset_hash_R0": multiset_hash(raw_coo.data.astype(np.float64)),
        "weight_multiset_hash_R2": multiset_hash(rew_coo.data.astype(np.float64)),
        "per_source_hash_R0": per_source_multiset_hash(
            raw_coo.row.astype(np.int64), raw_coo.data.astype(np.float64)
        ),
        "per_source_hash_R2": per_source_multiset_hash(
            rew_coo.row.astype(np.int64), rew_coo.data.astype(np.float64)
        ),
        "object": "raw synapse-count graph of the selected substrate",
    }
    return WiringCounterfactual(
        raw_r0=raw_r0,
        raw_r2=raw_r2,
        r0=r0,
        r2=r2,
        report=report,
        conservation_raw=conservation_raw,
        composition_scale=scale,
        rho_r0=_spectral_radius(r0),
        rho_r2=_spectral_radius(r2),
        r2_rho_matched=r2_rho_matched,
        rho_matched_scale=rho_matched_scale,
    )


def _spectral_radius(matrix: sp.spmatrix) -> float:
    from drososense.reservoir.connectome_reservoir import spectral_radius

    return float(spectral_radius(matrix.tocsr(), seed=0))
