"""F1-F5 family generators, under the frozen contrast declarations.

**Pre-family conventions (amendment 1 Section 0.1 and the contrast review).** No constant
in this module may be re-chosen after seeing a family's score, an A1-A5 verdict, or any food
result. The numeric choices and their provenance:

| constant | value | provenance |
|---|---|---|
| ``RHO_TARGET`` | 0.95 | amendment 1 Section 0.1 |
| ``PROBE_*`` | see ``battery`` | amendment 1 Section 0.1 |
| ``F2_SWAPS_PER_EDGE`` | **10** | contrast review: frozen randomization strength |
| ``F3`` self-loop convention | copied from F1 | contrast review |
| F1 identity | N=1000, Din=5, K=16 | frozen ``V3_substrate_selection.json`` settings |

**F2's budget is 10|E| ACCEPTED swaps, not attempts.** The distinction is load-bearing:
the frozen ``weight_preserving_degree_rewire`` stops on an *overlap* target, which would
make A2 the thing that decides the randomization strength -- exactly the
``construction <-> A2`` coupling the review forbids. So the accept ceiling is implemented
here as a port of that chain's swap semantics, with its self-checks reused verbatim, and
``target_overlap=0.0`` is passed to the frozen primitive so that comparison runs are not
truncated by the overlap rule.

**Absolute rule: a generator may not read its own audit.** Every function here takes
``(A, B)`` and returns a *candidate*. None of them may consult A1-A5, Page, D_eff, or a food
metric. F3's disconnectedness and F5's A3 status are CONSEQUENCES to be measured
afterwards, never objectives to be optimised.
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

from drososense.reservoir import r2_counterfactual as _r2  # noqa: E402
from drososense.reservoir.input_mapping import dense_random_mapping  # noqa: E402

from resaudit.battery import (  # noqa: E402
    FROZEN_RHO_TARGET,
    scale_to_spectral_radius,
    spectral_radius_of,
)
from resaudit.family import FamilySpec  # noqa: E402

__all__ = [
    "FROZEN_H1_SETTINGS",
    "F2_SWAPS_PER_EDGE",
    "F2_SEED",
    "F3_SEED",
    "F5_SEED",
    "GenerationReport",
    "f2_swaps_at_budget",
    "generate_f2",
    "generate_f3",
    "generate_f5",
    "F1_MATERIALIZATION_REQUIREMENT",
]

#: The frozen F1 settings, transcribed from
#: ``results/audit/v3_selection/V3_substrate_selection.json`` (``settings`` block).
#: These are the values F2/F3/F5 must match; they are NOT re-derived here.
FROZEN_H1_SETTINGS: dict[str, Any] = {
    "target_n": 1000,
    "din": 5,
    "krylov_K": 16,
    "rho_target": FROZEN_RHO_TARGET,
    "input_rule": "amendment 2/3 typed-aligned mapping, receiving_fraction 0.80, density_limit 0.16",
    "source": "results/audit/v3_selection/V3_substrate_selection.json",
}

#: F2's randomization strength. ACCEPTED swaps, i.e. ``10 * |E|`` applied double-edge
#: swaps -- deliberately not "attempts", and deliberately not chosen by looking at A2.
F2_SWAPS_PER_EDGE = 10

F2_SEED = 20260924
F3_SEED = 20260925
F5_SEED = 20260926

#: What must exist before F1 itself can be instantiated. Recorded as a constant because a
#: Stage 1 run that silently substitutes a stand-in for F1 would invalidate every contrast.
F1_MATERIALIZATION_REQUIREMENT = (
    "F1 is the frozen project's S0 Drosophila olfactory connectome substrate (N=1000, "
    "M=80443, S1=7.129) plus its typed-aligned input mapping. Both live OUTSIDE this "
    "repository: the raw connectome and induced adjacency are gitignored large binaries "
    "resolved by connectome/paths.py under the external data root ($DROSOSENSE_DATA, "
    "default ../data-root). Neither exists on a machine without that root, and the "
    "substrate's A/B are recorded in Git only as content hashes "
    "(A_hash 3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20, "
    "B_hash 3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968). "
    "Therefore F2/F3/F5 generate against an F1 that must be materialized first, and a "
    "Stage 1 report may not substitute a stand-in without saying so."
)


def reference_weight_scale(A_f1: sp.spmatrix, rho_target: float = FROZEN_RHO_TARGET) -> float:
    """The ONE scalar that brings F1 to ``rho_target``, shared by every family.

    **Why one shared scale rather than one per family.** Matching ``rho = 0.95`` exactly and
    preserving F1's weight multiset exactly are mutually exclusive: the normalisation
    multiplies every weight by a graph-dependent factor. The frozen project already recorded
    this incompatibility (``R2.spectral.rho.note``: *"preserving the global weight multiset
    (C4.2) and rescaling to a target radius are mutually exclusive"*).

    Faced with the choice, the weight multiset WINS, because the F2 contrast is defined as
    "first-order statistics and weight distribution held fixed, wiring arrangement varied".
    Sharing F1's scale makes the weight multisets bit-identical; the cost is that a family's
    spectral radius lands slightly off the target, and that residual is REPORTED rather than
    hidden. Measured on a 60-node stand-in at the frozen ``10|E|`` budget: F1 rho = 0.950000,
    F2 rho = 0.943874, a 0.64 % gap.
    """
    r = spectral_radius_of(A_f1)
    return float(rho_target) / r if r > 0 else 1.0


@dataclass(frozen=True)
class GenerationReport:
    """What a generator did, carried into the family's construction record."""

    family_id: str
    generator: str
    seed: int
    invariants: Mapping[str, Any]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "generator": self.generator,
            "seed": int(self.seed),
            "invariants": dict(self.invariants),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# F2 -- acceptance-budgeted degree-preserving rewire
# ---------------------------------------------------------------------------


def f2_swaps_at_budget(
    matrix: sp.spmatrix,
    *,
    seed: int,
    accepted_target: int,
    max_attempts: int | None = None,
) -> tuple[sp.csr_matrix, int, dict[str, Any]]:
    """Degree- and weight-preserving double-edge swaps, stopped at an ACCEPTED count.

    A port of ``drososense.reservoir.r2_counterfactual.weight_preserving_degree_rewire``
    with one addition: an *accepted*-swap ceiling. The port is necessary because the frozen
    chain's only stopping rules are an overlap target and an attempt budget, and an overlap
    target would let A2 decide F2's randomization strength.

    The swap semantics are the frozen ones, unchanged: a swap partner is drawn from the
    **same weight class** (so exchanging equal weights moves no in-strength at all), only
    exact weight matches are accepted, and self-loops are never moved.

    The frozen chain's preservation self-checks are reproduced at the end and raise rather
    than warn, so a silent degree or weight violation cannot reach a family.

    Returns:
        ``(rewired, accepted, diagnostics)``.
    """
    csr = sp.csr_matrix(matrix)
    coo = csr.tocoo()
    rows = coo.row.astype(np.int64).copy()
    cols = coo.col.astype(np.int64).copy()
    weights = coo.data.astype(np.float64).copy()
    n_edges = int(rows.size)
    n_nodes = int(csr.shape[0])
    keys = list(zip(rows.tolist(), cols.tolist()))
    if len(set(keys)) != n_edges:
        raise _r2.RewireError("the graph has duplicate (row, column) pairs")
    swappable = np.flatnonzero(rows != cols)
    if swappable.size < 2:
        raise _r2.RewireError(f"{swappable.size} non-self-loop edge(s): a swap needs two")

    original = set(keys)
    present = set(keys)
    overlap = len(original)
    out_degree_before = np.bincount(rows, minlength=n_nodes)
    in_degree_before = np.bincount(cols, minlength=n_nodes)
    weights_before = np.sort(weights.copy())
    per_source_before = _r2._per_source_multisets(rows, weights)

    order = np.argsort(weights, kind="stable")
    sorted_weights = weights[order]

    def class_window(value: float) -> tuple[int, int]:
        lo = int(np.searchsorted(sorted_weights, value, side="left"))
        hi = int(np.searchsorted(sorted_weights, value, side="right"))
        return lo, hi

    rng = np.random.default_rng(int(seed))
    attempts_cap = int(max_attempts) if max_attempts is not None else 400 * max(n_edges, 1)
    accepted = 0
    rejected = {"shared_endpoint": 0, "self_loop": 0, "duplicate": 0,
                "no_partner": 0, "no_legal_partner": 0}
    proposals = 0
    for proposal in range(attempts_cap):
        proposals = proposal + 1
        if accepted >= int(accepted_target):
            break
        i = int(swappable[int(rng.integers(0, swappable.size))])
        w1 = float(weights[i])
        lo, hi = class_window(w1)
        if hi - lo < 2:
            rejected["no_partner"] += 1
            continue
        a, b = int(rows[i]), int(cols[i])
        j = -1
        for _ in range(_r2.PARTNER_SEARCH_TRIES):
            if hi - lo < 2:
                break
            cand = int(order[int(rng.integers(lo, hi))])
            if cand == i:
                continue
            c, d = int(rows[cand]), int(cols[cand])
            if a == c or b == d:
                rejected["shared_endpoint"] += 1
                continue
            if a == d or c == b:
                rejected["self_loop"] += 1
                continue
            if (a, d) in present or (c, b) in present:
                rejected["duplicate"] += 1
                continue
            j = cand
            break
        if j < 0:
            rejected["no_legal_partner"] += 1
            continue
        c, d = int(rows[j]), int(cols[j])
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

    rewired = sp.csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    # the frozen chain's own preservation checks, reproduced and fatal
    if not np.array_equal(weights_before, np.sort(rewired.tocoo().data.astype(np.float64))):
        raise _r2.RewireError("the weight multiset changed during the swap chain")
    if not np.array_equal(np.bincount(rewired.tocoo().row, minlength=n_nodes), out_degree_before):
        raise _r2.RewireError("an out-degree changed during the swap chain")
    if not np.array_equal(np.bincount(rewired.tocoo().col, minlength=n_nodes), in_degree_before):
        raise _r2.RewireError("an in-degree changed during the swap chain")
    if _r2._per_source_multisets(rewired.tocoo().row.astype(np.int64), rewired.tocoo().data) != per_source_before:
        raise _r2.RewireError("a per-source outgoing weight multiset changed during the chain")

    diag = {
        "accepted_target": int(accepted_target),
        "accepted": int(accepted),
        "attempts": int(proposals),
        "acceptance_rate": (accepted / proposals) if proposals else float("nan"),
        "edge_overlap": (overlap / n_edges) if n_edges else float("nan"),
        "rejected": dict(rejected),
        "budget_reached": bool(accepted >= int(accepted_target)),
    }
    return rewired, accepted, diag


def generate_f2(
    A_f1: sp.spmatrix,
    B_f1: np.ndarray,
    *,
    family_id: str = "F2",
    seed: int = F2_SEED,
) -> tuple[FamilySpec, GenerationReport]:
    """F2 -- statistics-matched random control.

    A degree-preserving, weight-preserving rewire of F1 at a FROZEN accepted-swap budget of
    ``10|E|``. The budget is not adjusted for any reason after generation, and A2 is run
    afterwards as an audit, never as a tuning signal. If F2 fails A2 at this budget, that is
    the result.
    """
    A_f1 = sp.csr_matrix(A_f1)
    B = np.asarray(B_f1, dtype=np.float64)
    n_edges = int(A_f1.nnz)
    target = F2_SWAPS_PER_EDGE * n_edges
    rewired, accepted, diag = f2_swaps_at_budget(A_f1, seed=seed, accepted_target=target)
    scale = reference_weight_scale(A_f1)
    A = (rewired * scale).tocsr()
    spec = FamilySpec(
        family_id=family_id,
        label="statistics-matched random control (10|E| accepted swaps)",
        role="statistics_matched_random_control",
        A=A,
        B=B,
        n_nodes=A.shape[0],
        n_edges=int(A.nnz),
        din=B.shape[1],
        counterfactual_parent="F1",
        seed=seed,
        construction={
            "generator": "f2_swaps_at_budget",
            "swaps_per_edge": F2_SWAPS_PER_EDGE,
            **diag,
        },
    )
    report = GenerationReport(
        family_id=family_id,
        generator="f2_swaps_at_budget",
        seed=seed,
        invariants={
            "n_nodes": int(A.shape[0]),
            "n_edges": int(A.nnz),
            "degree_sequence_preserved": bool(
                np.array_equal(np.diff(A_f1.indptr), np.diff(A.indptr))
            ),
            "weight_multiset_preserved_exactly": bool(
                np.array_equal(np.sort(A_f1.tocoo().data), np.sort(rewired.tocoo().data))
            ),
            "weight_multiset_preserved_up_to_shared_scale": bool(
                np.allclose(np.sort(A_f1.tocoo().data), np.sort(A.tocoo().data), rtol=1e-9)
            ),
            "shared_weight_scale": float(scale),
            "weight_multiset_matches_reference": bool(
                np.array_equal(
                    np.sort((A_f1 * scale).tocoo().data), np.sort(A.tocoo().data)
                )
            ),
            "rho_target": FROZEN_RHO_TARGET,
            "rho_realised": spectral_radius_of(A),
            "rho_gap_vs_target": float(abs(spectral_radius_of(A) / FROZEN_RHO_TARGET - 1.0)),
        },
        notes=(
            "F2 shares F1's single weight scale, so the released matrices have IDENTICAL "
            "weight multisets. The cost is that F2's spectral radius lands slightly off "
            "rho_target (reported as rho_gap_vs_target); matching rho exactly and matching "
            "the weights exactly are mutually exclusive, and the weight invariant is the "
            "one the F2 contrast is defined by.",
            "A2 is applicable to F2 (counterfactual_parent='F1') and is run AFTER "
            "generation as an audit. F2's budget is not retuned if A2 fails: the review "
            "fixed 10|E| accepted swaps as a randomization strength independent of A2.",
        ),
    )
    return spec, report


# ---------------------------------------------------------------------------
# F3 -- topology-destroyed directed G(n, m) control
# ---------------------------------------------------------------------------


def generate_f3(
    A_f1: sp.spmatrix,
    B_f1: np.ndarray,
    *,
    family_id: str = "F3",
    seed: int = F3_SEED,
) -> tuple[FamilySpec, GenerationReport]:
    """F3 -- topology-destroyed control: directed ``G(n, m)`` with F1's exact edge count.

    Held from F1: ``n``, exact ``m`` (written as ``mean_degree`` in the contrast matrix
    because for fixed ``n`` the two are the same quantity), the weight multiset, the input
    geometry, the self-loop convention, and the ``rho`` normalisation.

    Deliberately NOT retained, and NOT repaired: the degree distribution, weak or strong
    connectivity, and the cycle profile. If the random control is disconnected or
    cycle-poor, that IS the finding -- repairing it would turn "topology destroyed" into
    "topology destroyed except the parts we disliked".
    """
    A_f1 = sp.csr_matrix(A_f1)
    B = np.asarray(B_f1, dtype=np.float64)
    n = int(A_f1.shape[0])
    m = int(A_f1.nnz)

    # self-loop convention MUST match F1's, since it changes what "m edges" means
    f1_coo = A_f1.tocoo()
    f1_has_self_loops = bool(np.any(f1_coo.row == f1_coo.col))
    f1_self_loop_count = int(np.sum(f1_coo.row == f1_coo.col))

    rng = np.random.default_rng(int(seed))
    chosen: set[tuple[int, int]] = set()
    # preserve F1's self-loop COUNT exactly where F1 has them, so the edge budget means
    # the same thing in both graphs
    if f1_has_self_loops:
        for node in rng.choice(n, size=min(f1_self_loop_count, n), replace=False):
            chosen.add((int(node), int(node)))
    guard = 0
    while len(chosen) < m and guard < 200 * m + 1000:
        guard += 1
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if not f1_has_self_loops and a == b:
            continue
        chosen.add((a, b))
    if len(chosen) < m:
        raise RuntimeError(
            f"could not draw {m} distinct directed edges over {n} nodes; "
            f"got {len(chosen)} after {guard} draws"
        )

    rows = np.fromiter((e[0] for e in chosen), dtype=np.int64, count=len(chosen))
    cols = np.fromiter((e[1] for e in chosen), dtype=np.int64, count=len(chosen))
    # F1's weight multiset, permuted onto the new edges. The permutation is random and
    # independent of everything else, so the multiset is preserved and nothing else is.
    weights = rng.permutation(f1_coo.data.astype(np.float64))
    raw = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    # F5 normalises to rho_target with its OWN scale, unlike F2 and F3.
    #
    # F2/F3 share F1's weight scale because their contrast is DEFINED by an identical
    # weight multiset. F5 has no such claim -- it must match the edge BUDGET and the scale
    # convention, and its construction is already unit-weight. What F5 does need is to
    # actually reach rho_target, because A3's measurement is normalisation-dependent
    # (amendment 1 Section 1) and a block-union substrate scaled by F1's much smaller
    # factor has a tiny spectral radius whose power norms decay away before the horizon.
    # Measured: at rho=0.95 the coprime construction returns D_eff ~ 80 against a gate of
    # 10; at F1's 0.128 scale the same graph returns D_eff ~ 6.5.
    scale = float(FROZEN_RHO_TARGET) / spectral_radius_of(raw) if spectral_radius_of(raw) > 0 else 1.0
    A = (raw * scale).tocsr()

    spec = FamilySpec(
        family_id=family_id,
        label="topology-destroyed directed G(n, m) control",
        role="topology_destroyed_control",
        A=A,
        B=B,
        n_nodes=n,
        n_edges=int(A.nnz),
        din=B.shape[1],
        counterfactual_parent="F1",
        seed=seed,
        construction={
            "generator": "directed_G_n_m",
            "n": n,
            "m_target": m,
            "m_realised": int(A.nnz),
            "self_loops_in_reference": f1_self_loop_count,
            "self_loop_convention": "matched_to_f1",
            "weights": "F1 multiset, randomly permuted onto new edges",
            "rho_target": FROZEN_RHO_TARGET,
        },
    )
    report = GenerationReport(
        family_id=family_id,
        generator="directed_G_n_m",
        seed=seed,
        invariants={
            "n_nodes": n,
            "n_edges": int(A.nnz),
            "mean_degree": float(A.nnz) / n,
            "edge_count_matches_reference": bool(int(A.nnz) == m),
            "degree_sequence_preserved": bool(
                np.array_equal(np.diff(A_f1.indptr), np.diff(A.indptr))
            ),
            "weight_multiset_preserved_exactly": bool(
                np.array_equal(np.sort(A_f1.tocoo().data), np.sort(raw.tocoo().data))
            ),
            "weight_multiset_matches_reference": bool(
                np.array_equal(np.sort((A_f1 * scale).tocoo().data), np.sort(A.tocoo().data))
            ),
            "shared_weight_scale": float(scale),
            "rho_target": FROZEN_RHO_TARGET,
            "rho_realised": spectral_radius_of(A),
            "rho_gap_vs_target": float(abs(spectral_radius_of(A) / FROZEN_RHO_TARGET - 1.0)),
        },
        notes=(
            "Disconnectedness, SCC collapse and cycle-poorness are EXPECTED and are not "
            "repaired. They are the mechanism by which F3 differs from F2, and reporting "
            "them is the result rather than a defect to fix.",
            "F3 declares counterfactual_parent='F1', so A2 applies. A directed G(n,m) does "
            "not preserve F1's degree sequence, so A2's degree_exact term is expected to "
            "FAIL -- that is a fact about the design (F3 is not a wiring counterfactual in "
            "A2's sense) and must be reported, not engineered away.",
        ),
    )
    return spec, report


# ---------------------------------------------------------------------------
# F5 -- A3-qualified construction
# ---------------------------------------------------------------------------


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            return False
        f += 2
    return True


def _coprime_block_lengths(n: int, blocks: int) -> tuple[int, ...]:
    """``blocks`` pairwise-coprime lengths, each >= 2, summing to exactly ``n``.

    Coprime cycle lengths are the whole mechanism: a directed rotation of period ``L``
    contributes ``L`` independent directions under propagation, so decoupled blocks of
    PAIRWISE COPRIME period give a Krylov block whose columns do not collapse into a shared
    period. Coprimality here is not decoration -- lengths that are merely "spread out"
    would not implement the declared construction.

    The sum is exact, because ``m`` (and therefore the edge budget) is a quantity F5 must
    match. Construction: the first ``blocks - 1`` lengths are DISTINCT PRIMES near
    ``n / blocks``, which are pairwise coprime by definition; the final length absorbs the
    remainder and is searched over coprimality with those primes plus the >= 2 floor. No
    exponential search is involved.

    Raises:
        ValueError: if no admissible partition exists, rather than silently returning
            lengths that are not pairwise coprime.
    """
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    if blocks == 1:
        if n < 2:
            raise ValueError(f"cannot make a cycle of length {n}")
        return (n,)

    def partition_from(start_value: int) -> tuple[int, ...] | None:
        """Distinct primes from ``start_value`` upward, plus a remainder that copes."""
        primes: list[int] = []
        c = max(2, start_value)
        while len(primes) < blocks - 1:
            if _is_prime(c):
                primes.append(c)
            c += 1
            if c > n:
                return None
        rem = n - sum(primes)
        if rem >= 2 and all(math.gcd(rem, prime) == 1 for prime in primes):
            return tuple(sorted(primes + [rem]))
        return None

    # Search bands around n/blocks and keep the MOST BALANCED valid partition. An
    # unbalanced partition (one tiny cycle plus one huge one) would still be pairwise
    # coprime, but it would not be the declared construction -- the point is many
    # comparable periods, not one dominant block.
    best: tuple[int, ...] | None = None
    best_spread: float = float("inf")
    band = max(2, n // blocks)
    for shift in range(0, 400):
        got = partition_from(band - shift) if shift <= band else None
        if got is None:
            got = partition_from(band + shift)
        if got is None:
            continue
        spread = max(got) - min(got)
        if spread < best_spread:
            best, best_spread = got, spread
            if spread <= 2:      # cannot do better than near-equal
                break
    if best is None:
        raise ValueError(
            f"no pairwise-coprime {blocks}-block partition of n={n} found; refusing to "
            "return lengths that are not pairwise coprime"
        )
    return tuple(int(x) for x in best)


def generate_f5(
    A_f1: sp.spmatrix,
    B_f1: np.ndarray,
    *,
    family_id: str = "F5",
    seed: int = F5_SEED,
    blocks: int | None = None,
) -> tuple[FamilySpec, GenerationReport]:
    """F5 -- deliberately A3-qualified construction, within F1's edge budget.

    A3 PASS is a **construction property**, carried as ``a3_status_origin`` in the family's
    construction record so a reader cannot mistake it for an independent experimental
    result. F5 may answer "given that reachable diversity is guaranteed, how does the
    system behave?"; it may NOT answer "does A3 cause better performance".

    The construction is task-blind: it composes directed cycles of pairwise-coprime length
    and places the input across those blocks. It consults no task metric, no food label and
    no A3 verdict -- the verdict is measured afterwards.
    """
    A_f1 = sp.csr_matrix(A_f1)
    B = np.asarray(B_f1, dtype=np.float64)
    n = int(A_f1.shape[0])
    m = int(A_f1.nnz)
    din = int(B.shape[1])

    # Block count is DERIVED from n, not fixed: a partition into k pairwise-coprime parts
    # each >= 2 needs n >= 2+3+...+... , so a fixed count would fail on small substrates.
    # The rule below is declared, not tuned: aim for blocks averaging ~125 nodes, cap at 8.
    if blocks is None:
        blocks = max(2, min(8, n // 125))
    lengths = _coprime_block_lengths(n, blocks)
    rows: list[int] = []
    cols: list[int] = []
    offsets: list[int] = []
    offset = 0
    for L in lengths:
        offsets.append(offset)
        for i in range(L):
            rows.append(offset + (i + 1) % L)
            cols.append(offset + i)
        offset += L
    n_cycle_edges = len(rows)

    # Spend the remaining budget on EXTRA chords inside each block, which preserves the
    # block structure (and therefore the coprime-period mechanism) rather than adding
    # cross-block edges that would couple the periods.
    rng = np.random.default_rng(int(seed))
    extra = max(0, m - n_cycle_edges)
    existing = set(zip(rows, cols))
    added = 0
    guard = 0
    while added < extra and guard < 100 * extra + 1000:
        guard += 1
        # locate a block with this length (lengths may repeat only by fallback)
        L = int(lengths[int(rng.integers(0, len(lengths)))])
        blk = next(b for b, x in enumerate(lengths) if x == L)
        a = offsets[blk] + int(rng.integers(0, L))
        step = int(rng.integers(2, max(3, L)))
        b = offsets[blk] + (a - offsets[blk] + step) % L
        if a != b and (a, b) not in existing:
            existing.add((a, b))
            rows.append(a)
            cols.append(b)
            added += 1

    weights = np.ones(len(rows), dtype=np.float64)
    raw = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    # F5 normalises to rho_target with its OWN scale, unlike F2 and F3.
    #
    # F2/F3 share F1's weight scale because their contrast is DEFINED by an identical
    # weight multiset. F5 has no such claim -- it must match the edge BUDGET and the scale
    # convention, and its construction is already unit-weight. What F5 does need is to
    # actually reach rho_target, because A3's measurement is normalisation-dependent
    # (amendment 1 Section 1) and a block-union substrate scaled by F1's much smaller
    # factor has a tiny spectral radius whose power norms decay away before the horizon.
    # Measured: at rho=0.95 the coprime construction returns D_eff ~ 80 against a gate of
    # 10; at F1's 0.128 scale the same graph returns D_eff ~ 6.5.
    scale = float(FROZEN_RHO_TARGET) / spectral_radius_of(raw) if spectral_radius_of(raw) > 0 else 1.0
    A = (raw * scale).tocsr()

    # The input is placed across the blocks, one channel per block where possible, so the
    # input reaches several coprime periods rather than one.
    B5 = np.zeros((n, din), dtype=np.float64)
    chosen = [offsets[b] for b in range(min(din, len(offsets)))]
    for c, node in enumerate(chosen):
        B5[int(node), c] = 1.0
    spec = FamilySpec(
        family_id=family_id,
        label="audit-qualified sparse construction (coprime-period blocks)",
        role="deliberately_a3_qualified_construction",
        A=A,
        B=B5,
        n_nodes=n,
        n_edges=int(A.nnz),
        din=din,
        counterfactual_parent="F1",
        seed=seed,
        construction={
            "generator": "coprime_period_blocks",
            "blocks": len(lengths),
            "block_lengths": list(lengths),
            "cycle_edges": n_cycle_edges,
            "extra_chords": added,
            "m_target": m,
            "m_realised": int(A.nnz),
            "rho_target": FROZEN_RHO_TARGET,
            "own_weight_scale": float(scale),
            # the field the review asked for, permanently attached
            "a3_status_origin": "construction_property",
            "task_blind": True,
        },
    )
    report = GenerationReport(
        family_id=family_id,
        generator="coprime_period_blocks",
        seed=seed,
        invariants={
            "n_nodes": n,
            "n_edges": int(A.nnz),
            "m_target": m,
            "m_matched_exactly": bool(int(A.nnz) == m),
            "block_lengths": list(lengths),
            "pairwise_coprime": all(
                math.gcd(lengths[i], lengths[j]) == 1
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ),
            "a3_status_origin": "construction_property",
            "rho_target": FROZEN_RHO_TARGET,
            "rho_raw": spectral_radius_of(raw),
            "rho_realised": spectral_radius_of(A),
            "rho_gap_vs_target": float(abs(spectral_radius_of(A) / FROZEN_RHO_TARGET - 1.0)),
        },
        notes=(
            "F5 uses its OWN rho normalisation, not F1's shared weight scale; see the "
            "comment on the scaling line. F2/F3 share F1's scale because their contrast is "
            "defined by an identical weight multiset, which F5 does not claim.",
            "MEASURED LIMIT: at F1's edge budget (mean degree ~80) the coprime-block "
            "mechanism does NOT survive. The blocks supply only ~N cycle edges, so the "
            "remaining ~79k edges must be in-block chords, which raise rho to ~82; "
            "normalising rho back to 0.95 then collapses D_eff to ~7.9 against a gate of "
            "10. The construction reaches D_eff ~ 80 at mean degree ~1. See F5's status in "
            "docs/resaudit_contrast_matrix.md.",
            "A3 PASS is a CONSTRUCTION PROPERTY. The report field a3_status_origin exists "
            "so a reader cannot read it as an independent A3 experiment.",
            "F5 may not be used to infer that A3 causes food performance: A3-FAIL families "
            "are never run on food data (rule 6.1), so that causal question is "
            "unidentifiable by design.",
            "The block mechanism does not read A3. It reads a declared structural rule "
            "(coprime cycle lengths) and the verdict is measured afterwards.",
        ),
    )
    return spec, report
