"""Synthetic toy graphs with KNOWN verdicts, for invariant tests of the battery.

Every expectation below was CALIBRATED against the frozen metric
(``drososense.substrate_scores.krylov_score`` /
``drososense.reservoir.dynamics``) before it was written down. The calibration produced
three facts that matter for reading the battery, and they are recorded here rather than
hidden:

**Fact 1 -- A3's gate sits low relative to what a Krylov block can offer.**
``D_eff = (sum s)^2 / sum s^2`` over the singular values of ``[B, AB, ..., A^16 B]``. For
a block of ``n`` equal singular values this is exactly ``n``, and the block has
``(K+1)*Din = 17*Din`` columns. So a substrate reaching only ``Din`` independent
propagated directions measures ``~Din``, while a dense block measures far more. The gate
``2*Din`` therefore asks "does the input's reach survive propagation with at least twice
its own dimension" -- it is not a demanding ceiling. Consequences measured here:

* a single directed 2-cycle with one input channel measures ``D_eff = 1.9983`` against a
  gate of ``2.0``: a hairline FAIL;
* a feedforward **nilpotent chain** -- the textbook uncontrollable example -- measures
  ``D_eff = 17.0`` against a gate of ``2.0`` and **PASSES**, because each ``A^k B`` is a
  fresh standard basis vector and the Krylov columns are orthogonal.

**Fact 2 -- A1's SCC term is nearly implied by its isolated term.**
Nodes with out-degree 0 form singleton SCCs, so they depress ``largest_SCC/N``. With
``N = 1000`` and the signed thresholds, at most 2 % of nodes may be isolated, which caps
the SCC depression at 0.98 -- and ``0.98 > 0.90``. So for any graph that clears the
isolated term, the SCC term is satisfied UNLESS the graph has several large nontrivial
components. The battery's two A1 terms are therefore not independently reachable, and the
tests assert the reachable combinations rather than pretending otherwise.

**Fact 3 -- A4 (recurrent memory) is protocol-dependent and unstable.**
``memory_metric`` correlates a state channel against lagged inputs; the feedforward
control is the same quantity with ``A := 0``. On bounded random drive at any scale tested
(``N = 40`` and ``N = 1000``, ``Din = 16``, leak in {0.1, 0.3, 0.6, 1.0}), the metric
sits at ``0.005-0.018`` and the signed contribution ``(M - M_A:=0)/M`` is NEGATIVE for
most graph/leak combinations, including graphs with genuine recurrence. It is also
non-monotone in recurrent coupling. This is reported as a battery finding, not tuned
away; the toys below therefore carry A4 expectations that were MEASURED, and the tests
assert the definition rather than engineering an A4 pass.

Design rationale, per toy
-------------------------

``two_coprime_cycles`` / ``hairline_fail_cycle`` / ``zero_input_graph`` / ``nilpotent_chain``
    The A3 contrast family: comfortable pass, hairline fail, loud fail, and the
    documented surprise.

``k_out_ring``
    Ring with out-degree ``k``. Also fails A1 on the mean-out-degree term when ``k = 1``
    (the SCC and isolated terms both pass), which is the cleanest single-term A1 failure.

``two_component_graph``
    Two disjoint rings. Fails A1 on the largest-SCC term while the isolated term passes,
    because the rings have no out-degree-0 nodes.

``isolated_boundary_graph``
    A ``k``-out ring plus ``n_iso`` nodes of out-degree 0, used to drive the isolated
    fraction across its 0.02 boundary. Deliberately kept at a scale where the other two
    A1 terms still pass, so the boundary is what is being tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def _cycles(lengths: tuple[int, ...]) -> tuple[sp.csr_matrix, list[int], int]:
    """Concatenate decoupled directed cycles; return ``(A, block_starts, N)``."""
    lengths = tuple(int(L) for L in lengths)
    if any(L < 2 for L in lengths):
        raise ValueError("every cycle length must be >= 2")
    n = int(sum(lengths))
    rows: list[int] = []
    cols: list[int] = []
    starts: list[int] = []
    offset = 0
    for L in lengths:
        starts.append(offset)
        for i in range(L):
            rows.append(offset + (i + 1) % L)
            cols.append(offset + i)
        offset += L
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A, starts, n


def two_coprime_cycles(
    lengths: tuple[int, ...] = (2, 3, 5, 7, 11),
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Coprime-length cycles, one input channel each. A3 PASSES with margin.

    Measured: ``D_eff = 25.4360`` against a gate of ``2*Din = 10``.
    A1 PASSES at ``k=1`` rings? No -- mean out-degree is 1.0, so A1 FAILS. This toy is
    for A3; the tests read A3 only, and A1 tests use ``k_out_ring``.
    """
    A, starts, n = _cycles(lengths)
    din = len(lengths)
    B = np.zeros((n, din))
    for c, s in enumerate(starts):
        B[s, c] = 1.0
    return A, B, {
        "designed_verdicts": {"A3": "PASS"},
        "measured": {"D_eff": 25.4360, "gate": 2 * din, "theoretical_columns": 17 * din},
        "why": "coprime rotation periods keep the Krylov columns independent",
        "lengths": lengths,
    }


def hairline_fail_cycle() -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """One directed 2-cycle, one input channel. A3 FAILS by a hair.

    Measured: ``D_eff = 1.9983`` against a gate of ``2.0``. A hostile boundary case: the
    verdict must still be FAIL, because "close" is not a pass.
    """
    A, starts, n = _cycles((2,))
    B = np.zeros((n, 1))
    B[starts[0], 0] = 1.0
    return A, B, {
        "designed_verdicts": {"A3": "FAIL"},
        "measured": {"D_eff": 1.9983, "gate": 2.0},
        "why": "a 2-cycle offers only Din independent propagated directions",
    }


def zero_input_graph(n: int = 12, din: int = 2) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """A recurrent graph with ``B = 0``. A3 FAILS loudly (``D_eff = 0``)."""
    A, _, _n = _cycles((n,))
    B = np.zeros((n, din))
    return A, B, {
        "designed_verdicts": {"A3": "FAIL", "A5": "FAIL"},
        "measured": {"D_eff": 0.0, "gate": 2 * din, "D_eff_state": 0.0},
        "why": "no input mapping at all: controllability rank is zero by construction",
    }


def nilpotent_chain(n: int = 40, din: int = 1) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Lower shift with inputs on the first ``din`` nodes. A3 PASSES -- documented surprise.

    Each ``A^k B`` is a shifted standard basis vector, so the Krylov columns are
    orthogonal and ``D_eff`` counts surviving powers instead of collapsing to ``Din``.
    """
    rows = np.arange(1, n)
    cols = np.arange(0, n - 1)
    A = sp.csr_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n))
    B = np.zeros((n, din))
    for c in range(din):
        B[min(c, n - 1), c] = 1.0
    return A, B, {
        "designed_verdicts": {"A3": "PASS"},
        "measured": {"D_eff": 17.0, "gate": 2 * din},
        "why": (
            "A^k B are distinct standard basis vectors, so the Krylov block is "
            "orthonormal and D_eff = number of surviving powers"
        ),
    }


def k_out_ring(n: int = 40, k: int = 2) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Directed ring, each node wired to its ``k`` successors. Mean out-degree is ``k``.

    ``k = 1`` fails A1's mean-out-degree term with the SCC and isolated terms passing.
    """
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for j in range(1, k + 1):
            rows.append((i + j) % n)
            cols.append(i)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    return A, B, {"mean_out_degree": float(k), "n": n}


def two_component_graph(
    n_each: int = 20, k: int = 2
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Two disjoint ``k``-out rings. Fails A1 on the largest-SCC term only.

    Measured: ``scc_largest_fraction = 0.50`` FAIL, ``isolated_fraction = 0.0`` PASS,
    ``mean_out_degree = 2.0`` PASS.
    """
    A_one, _, _n = k_out_ring(n_each, k)
    n = 2 * n_each
    A = sp.block_diag([A_one, A_one], format="csr")
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    return A, B, {
        "designed_verdicts": {"A1": "FAIL"},
        "fails_term": "scc_largest_fraction",
        "measured": {"scc_largest_fraction": 0.50, "isolated_fraction": 0.0, "mean_out_degree": 2.0},
        "n": n,
    }


def isolated_boundary_graph(
    n_ring: int = 99, k: int = 2, n_isolated: int = 1
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """``k``-out ring plus ``n_isolated`` out-degree-0 nodes.

    Drives the isolated fraction across its boundary. At the default (``1/100 = 0.01``)
    all three A1 terms pass; larger ``n_isolated`` pushes the fraction over 0.02.

    Note the interaction this toy documents: raising ``n_isolated`` to exceed the ceiling
    ALSO depresses the mean out-degree, so "isolated fails alone" is not reachable at
    small scale. It becomes reachable once ``n_ring`` is large enough that the isolated
    nodes are a negligible share of the out-degree sum.
    """
    n = n_ring + n_isolated
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n_ring):
        for j in range(1, k + 1):
            rows.append((i + j) % n_ring)
            cols.append(i)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    return A, B, {
        "n_ring": n_ring,
        "n_isolated": n_isolated,
        "expected_isolated_fraction": n_isolated / n,
        "expected_mean_out_degree": k * n_ring / n,
        "n": n,
    }


def ring_with_chords(
    n: int = 60, k: int = 2, n_chords: int = 30, din: int = 4, seed: int = 5
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """``k``-out ring plus random chords. One strongly connected component, tunable degree.

    Raw, its spectral radius is well above 1, so A3 collapses to ``D_eff ~ 1`` (see
    ``normalize_to_spectral_radius``). The :func:`battery_ready_ring` wrapper scales it
    and is what a family should actually use.
    """
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for j in range(1, k + 1):
            rows.append((i + j) % n)
            cols.append(i)
    for _ in range(n_chords):
        a, b = (int(x) for x in rng.integers(0, n, 2))
        if a != b:
            rows.append(b)
            cols.append(a)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    B = np.zeros((n, din))
    for c in range(din):
        B[int(rng.integers(0, n)), c] = 1.0
    return A, B, {"n": n, "k": k, "n_chords": n_chords, "seed": seed}


def spectral_radius(A: sp.spmatrix, *, power_iters: int = 300, seed: int = 0) -> float:
    """Dominant ``|lambda|``. Thin alias for :func:`resaudit.battery.spectral_radius_of`."""
    from resaudit.battery import spectral_radius_of

    return spectral_radius_of(A, power_iters=power_iters, seed=seed)


def normalize_to_spectral_radius(A: sp.spmatrix, target: float = 0.95) -> sp.csr_matrix:
    """Scale ``A`` so its spectral radius is ``target``.

    Required before A3 is meaningful: ``krylov_score`` does not rescale, and the
    Frobenius norms of ``A^k B`` grow like ``rho^k``, so an unnormalized substrate with
    ``rho`` above 1 measures ``D_eff ~ 1`` regardless of its controllability.
    """
    from resaudit.battery import scale_to_spectral_radius

    return scale_to_spectral_radius(A, target)


def battery_ready_ring(
    n: int = 60, k: int = 2, n_chords: int = 30, din: int = 4, seed: int = 5,
    target_radius: float = 0.95,
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """A ring+chords substrate scaled to ``rho = target_radius``.

    Measured at the defaults: A1 all three terms PASS, and A3 returns
    ``D_eff = 13.87`` against a gate of ``8``. This is the smallest toy that clears BOTH
    A1 and A3, which is what a "qualified" family requires.
    """
    A, B, meta = ring_with_chords(n=n, k=k, n_chords=n_chords, din=din, seed=seed)
    A_scaled = normalize_to_spectral_radius(A, target_radius)
    meta = {**meta, "normalized_to": target_radius, "rho_before": spectral_radius(A)}
    return A_scaled, B, meta


def permuted_graph(A: sp.csr_matrix, seed: int = 7) -> tuple[sp.csr_matrix, np.ndarray]:
    """Relabel a graph under a node permutation, returning ``(A_perm, perm)``.

    Preserves the degree sequence exactly while changing the edge set -- exactly the
    distinction A2's exactness terms are supposed to make.
    """
    A = A.tocsr()
    n = A.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    return A[perm][:, perm].tocsr(), perm


__all__ = [
    "ring_with_chords",
    "spectral_radius",
    "normalize_to_spectral_radius",
    "battery_ready_ring",
    "two_coprime_cycles",
    "hairline_fail_cycle",
    "zero_input_graph",
    "nilpotent_chain",
    "k_out_ring",
    "two_component_graph",
    "isolated_boundary_graph",
    "permuted_graph",
]
