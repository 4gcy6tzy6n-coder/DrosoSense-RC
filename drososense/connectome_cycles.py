"""Structural-dynamics metrics for the M5 audit (independent of any reservoir).

What this measures, on a directed adjacency ``A``:

* WCC vs SCC: the graph may be globally connected under undirected walk (WCC = 100 %)
  but still be near-feedforward under directed walk. SCCs are what the reservoir's
  echo actually has to cycle through, so they matter and WCC alone does not.
* 2-core participation: a directed edge is in some cycle iff both endpoints have
  in-degree >= 1 AND out-degree >= 1 within the SAME SCC. That is a necessary
  condition; the fraction of edges in any SCC's 2-core is a tight LOWER BOUND on the
  fraction of edges that participate in a directed cycle, and it is cheap (one pass
  over the SCCs).
* Cycle-length distribution (sampled): a length-1 walk from i to itself is (A^k)_{ii}.
  The count of ``(A^k)_{ii} > 0`` for small k gives a cycle-length distribution up to
  length k, with one ``A^k`` matmul per k (sparse, truncated to keep memory bounded).
* ORN-reachable recurrent core: nodes that (a) are reachable from at least one ORN,
  and (b) lie in a non-trivial SCC (|SCC| >= 2). For a feedforward pathway this is ~0
  even though WCC = 100 % and ORN-reachability is full -- the two halves together
  separate "the input can get there" from "the substrate has cycles to amplify".

Nothing is written anywhere. The module is pure-numpy / pure-scipy / scipy.sparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


#: A "non-trivial" SCC has at least one directed cycle (|SCC| >= 2). The singleton SCCs
#: (one node, no self-loop) and the one-off self-loop SCCs are excluded from the
#: recurrent-core counts because they do not provide echo.
MIN_SCC_SIZE_FOR_RECURRENT = 2


@dataclass(frozen=True)
class StructuralMetrics:
    """One substrate's structural-dynamics audit.

    Attributes:
        n_nodes: ``A.shape[0]``.
        n_edges: ``A.nnz``.
        wcc_largest_fraction: ``|largest WCC| / N``.
        wcc_count: Number of weakly connected components.
        scc_count: Number of strongly connected components.
        scc_largest_fraction: ``|largest SCC| / N``.
        nodes_in_nontrivial_scc_fraction: Nodes in an SCC of size >= 2 / N.
        edges_in_nontrivial_scc_fraction: Edges inside an SCC of size >= 2 / ``n_edges``.
        scc_sizes: SCC size histogram (sorted descending).
        cycle_edge_fraction: Edges in the 2-core of any SCC (necessary for being in a
            cycle) / ``n_edges``. A tight LOWER BOUND on the fraction of edges that
            participate in any directed cycle.
        cycle_length_counts: ``{k: number of i with (A^k)_{ii} > 0}`` for the ks measured.
        orn_reachable_nodes: Fraction of nodes reachable from at least one ORN.
        orn_reachable_in_nontrivial_scc_fraction: Fraction of ORN-reachable nodes that
            also lie in a non-trivial SCC -- the ``ORN-reachable recurrent core``.
        spectrum_top: Top-|lambda| values of A (sparse eigsh, k top eigenvalues).
        spectrum_gap: Ratio between the two top |lambda| values; ``inf`` when there
            is exactly one dominant mode.
    """

    n_nodes: int
    n_edges: int
    wcc_largest_fraction: float
    wcc_count: int
    scc_count: int
    scc_largest_fraction: float
    nodes_in_nontrivial_scc_fraction: float
    edges_in_nontrivial_scc_fraction: float
    scc_sizes: tuple[int, ...]
    cycle_edge_fraction: float
    cycle_length_counts: dict[int, int]
    orn_reachable_nodes: float
    orn_reachable_in_nontrivial_scc_fraction: float
    spectrum_top: tuple[float, ...]
    spectrum_gap: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "wcc_largest_fraction": self.wcc_largest_fraction,
            "wcc_count": self.wcc_count,
            "scc_count": self.scc_count,
            "scc_largest_fraction": self.scc_largest_fraction,
            "nodes_in_nontrivial_scc_fraction": self.nodes_in_nontrivial_scc_fraction,
            "edges_in_nontrivial_scc_fraction": self.edges_in_nontrivial_scc_fraction,
            "scc_sizes_top10": list(self.scc_sizes[:10]),
            "cycle_edge_fraction": self.cycle_edge_fraction,
            "cycle_length_counts": {int(k): int(v) for k, v in self.cycle_length_counts.items()},
            "orn_reachable_nodes": self.orn_reachable_nodes,
            "orn_reachable_in_nontrivial_scc_fraction": self.orn_reachable_in_nontrivial_scc_fraction,
            "spectrum_top": list(self.spectrum_top),
            "spectrum_gap": self.spectrum_gap,
        }


def _tarjan_scc(A: sp.spmatrix) -> list[list[int]]:
    """Tarjan's SCC algorithm. scipy's csgraph.connected_components(directed=True)
    collapses all strongly-connected graphs to one SCC on some versions, so we
    implement it directly. Linear in N+M; ~10 ms on a 1000-node 80k-edge graph."""
    n = int(A.shape[0])
    coo = A.tocoo()
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(coo.row.tolist(), coo.col.tolist()):
        if u != v:
            adj[u].append(v)
    index_counter = [0]
    stack: list[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlinks = [0] * n
    sccs: list[list[int]] = []
    def strongconnect(v: int) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            scc: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)
    return sccs


def wcc_scc_decomposition(
    A: sp.spmatrix,
    *,
    min_scc_for_nontrivial: int = MIN_SCC_SIZE_FOR_RECURRENT,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_nodes, wcc_labels, scc_labels, in_deg_per_scc, out_deg_per_scc)``.

    ``in_deg_per_scc[i]`` is the in-degree WITHIN the node's SCC for node ``i`` (NOT the
    global in-degree). Same for ``out_deg_per_scc``.
    """
    n = int(A.shape[0])
    coo = A.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    # WCC on the symmetrised graph
    import scipy.sparse as _sp
    sym = (coo + coo.T).astype(bool)
    n_wcc, wcc_labels = _sp.csgraph.connected_components(sym, directed=False, return_labels=True)
    # SCC via Tarjan (see _tarjan_scc)
    scc_components = _tarjan_scc(A)
    n_scc = len(scc_components)
    scc_labels = np.zeros(n, dtype=np.int64)
    for label, comp in enumerate(scc_components):
        for v in comp:
            scc_labels[v] = label
    # in-degree WITHIN SCC: count (u -> v) where scc[u] == scc[v]
    same_scc = scc_labels[rows] == scc_labels[cols]
    in_scc_rows = rows[same_scc]
    out_scc_rows = cols[same_scc]
    in_deg = np.bincount(in_scc_rows, minlength=n).astype(np.int64)
    out_deg = np.bincount(out_scc_rows, minlength=n).astype(np.int64)
    return n, np.asarray(wcc_labels, dtype=np.int64), scc_labels, in_deg, out_deg


def cycle_edge_fraction(
    scc_labels: np.ndarray, in_deg: np.ndarray, out_deg: np.ndarray
) -> float:
    """Lower bound on the fraction of edges in any directed cycle.

    An edge is in some cycle iff BOTH endpoints have in-degree >= 1 AND out-degree
    >= 1 within the same SCC (excluding self-loops on the singleton SCCs whose in/out
    are 1 if a self-loop is present).
    """
    in_deg_arr = np.asarray(in_deg, dtype=np.int64)
    out_deg_arr = np.asarray(out_deg, dtype=np.int64)
    # count "good" edges: both endpoints have in >= 1 AND out >= 1
    good_nodes = (in_deg_arr >= 1) & (out_deg_arr >= 1)
    # a strict lower bound uses >=1 on both: a self-loop on a singleton node also has
    # in=out=1, so we exclude self-loops by requiring >=2 in/out (== SCC size >= 2). The
    # strict version is good_nodes & scc_size >= 2, but that requires scc sizes here. We
    # do the inclusive lower bound (>=1) for transparency; the inclusive bound is a
    # *lower bound* on cycle participation regardless.
    mask = good_nodes[scc_labels]
    # we don't have edge counts here; the caller passes the denominator.
    return float(mask.sum())  # node count for the in/out >= 1 condition


def cycle_length_distribution(
    A: sp.spmatrix, *, ks: tuple[int, ...] = (2, 3, 4, 5, 6)
) -> dict[int, int]:
    """Number of self-return walks of length k per k.

    For each k, ``(A^k)_{ii} > 0`` iff there is a walk of length k from i to itself.
    The diagonal count of those strict-positive entries is the count of nodes that have
    such a walk; the length distribution of any single cycle is the count of i for
    which the shortest self-return walk has length k, which we APPROXIMATE with the
    diagonal-of-A^k entries -- a node with a cycle of length k participates in any
    ``A^k`` walk too, so the diagonal counts are over-approximations when cycles share
    nodes.
    """
    out: dict[int, int] = {}
    current = sp.identity(A.shape[0], format="csr")
    for k in ks:
        current = current @ A
        # cap the NNZ to keep memory bounded
        if current.nnz > 2_000_000:
            # prune to largest entries per row
            dense = current.toarray()
            keep = np.zeros_like(dense)
            r = np.arange(dense.shape[0])
            top_k = 16
            idx = np.argpartition(-np.abs(dense), top_k, axis=1)[:, :top_k]
            keep[r[:, None], idx] = dense[r[:, None], idx]
            current = sp.csr_matrix(keep)
        diag = current.diagonal()
        out[int(k)] = int(np.count_nonzero(np.asarray(diag) > 0))
    return out


def orn_reachable_recurrent_core(
    A: sp.spmatrix, orn_rows: np.ndarray, scc_labels: np.ndarray
) -> tuple[float, float]:
    """``(fraction_orn_reachable, fraction_of_those_in_non_trivial_SCC)``.

    ORN-reachable = nodes that have at least one walk from an ORN.
    recurrent core = ORN-reachable AND in an SCC of size >= ``MIN_SCC_SIZE_FOR_RECURRENT``.
    """
    import scipy.sparse as _sp

    orn_rows = np.asarray(orn_rows, dtype=np.int64).ravel()
    n = A.shape[0]
    # reachability: do a BFS on the directed graph from all ORN rows together
    n_comp, labels = _sp.csgraph.connected_components(A, directed=True, return_labels=True)
    orn_components = set(np.unique(labels[orn_rows]).tolist())
    reachable = np.zeros(n, dtype=bool)
    for comp in orn_components:
        reachable |= (labels == comp)
    orn_reachable_frac = float(reachable.sum()) / n if n else float("nan")
    nontrivial_scc = np.bincount(scc_labels) >= MIN_SCC_SIZE_FOR_RECURRENT
    recurrent_core_mask = reachable & nontrivial_scc[scc_labels]
    if reachable.sum() == 0:
        return orn_reachable_frac, 0.0
    return orn_reachable_frac, float(recurrent_core_mask.sum()) / float(reachable.sum())


def top_eigenvalues(A: sp.spmatrix, *, k: int = 8) -> tuple[tuple[float, ...], float]:
    """``(|lambda_1|, |lambda_2|, ...)`` and ``|lambda_1|/|lambda_2|`` (``inf`` if tied).

    Uses sparse eigsh on the symmetrised operator; for a non-symmetric A we form
    ``A + A.T`` for the spectrum shape (the dominant non-symmetric mode is typically
    captured by the symmetrised operator too; this is an approximation labelled as such).
    """
    import scipy.sparse.linalg as _spla

    sym = (A + A.T).tocsc() * 0.5
    if sym.nnz == 0:
        return (0.0,), float("inf")
    n = sym.shape[0]
    k = min(k, n - 2) if n > 2 else 1
    try:
        vals = _spla.eigsh(sym, k=k, which="LM", return_eigenvectors=False)
        vals = np.sort(np.abs(vals))[::-1]
    except Exception:  # pragma: no cover - eigsh can fail on tiny matrices
        return (0.0,), float("inf")
    vals = tuple(float(v) for v in vals)
    gap = vals[0] / vals[1] if len(vals) >= 2 and vals[1] > 0 else float("inf")
    return vals, gap


@dataclass(frozen=True)
class ControllabilityKrylov:
    """The Krylov expansion ``[B, A·B, A²·B, ..., A^{K-1}·B]`` and its effective rank.

    ``effective_rank`` is the participation ratio ``(sum sigma)^2 / sum sigma^2`` over
    the full SVD of the (N x K·Din) concatenated matrix. This is what the C3
    dynamics measure with D_eff: the effective rank of the STATE matrix is bounded by
    the effective rank of the KRYLOV expansion's coefficient matrix when the system is
    linear; for a nonlinear (tanh) reservoir it is a necessary-condition lower bound.
    """
    K: int
    Din: int
    rank: int
    effective_rank: float
    singular_values_top: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "K": self.K,
            "Din": self.Din,
            "rank": self.rank,
            "effective_rank": self.effective_rank,
            "singular_values_top": list(self.singular_values_top),
        }


def controllability_krylov(
    A: sp.spmatrix, B: np.ndarray, *, K: int = 16
) -> ControllabilityKrylov:
    """Build ``[B, A·B, A²·B, ..., A^{K-1}·B]`` (shape ``(N, K*Din)``) and report rank and
    effective rank.

    ``B`` is shape ``(N, Din)``. The result is the linear-algebra fact about the
    Krylov expansion that bounds the linear-system reachable subspace; for the tanh
    reservoir it is a NECESSARY lower bound on the effective rank of the state matrix
    over ``K`` steps.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(f"B must be shape ({A.shape[0]}, Din); got {B.shape}")
    N, Din = B.shape
    cols = [B]
    cur = B
    for _ in range(K - 1):
        cur = A @ cur
        cols.append(cur)
    Ck = np.concatenate(cols, axis=1)
    # SVD
    s = np.linalg.svd(Ck, compute_uv=False)
    pos = s[s > 1e-10]
    eff_rank = float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0
    rank = int(np.sum(s > 1e-10))
    return ControllabilityKrylov(
        K=K, Din=Din, rank=rank, effective_rank=eff_rank,
        singular_values_top=tuple(float(v) for v in s[:8]),
    )


def measure_substrate(
    A: sp.spmatrix,
    *,
    orn_rows: np.ndarray,
    B: np.ndarray | None = None,
    K_list: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """The structural-dynamics audit for one substrate.

    Returns a dict with the SCC/WCC/cycle spectrum, the ORN-reachable recurrent
    core, the controllability effective rank at each ``K`` in ``K_list``, and the top
    spectrum. ``B`` defaults to ``identity[: orn_rows, :]`` -- one-hot ORN inputs --
    so the audit can run without the v2 mapping; pass the v2 W_in to use the actual
    input layout.
    """
    n_nodes = int(A.shape[0])
    n_edges = int(A.nnz)
    nw, wcc_labels, scc_labels, in_deg, out_deg = wcc_scc_decomposition(A)
    sizes = np.bincount(scc_labels)
    nontrivial = sizes >= MIN_SCC_SIZE_FOR_RECURRENT

    # WCC / SCC aggregates
    wcc_largest_fraction = float(sizes_wcc := np.bincount(wcc_labels).max()) / n_nodes if n_nodes else 0
    n_wcc = int(np.unique(wcc_labels).size)
    n_scc = int(np.unique(scc_labels).size)
    scc_largest_fraction = float(sizes.max()) / n_nodes if n_nodes else 0
    nodes_in_nontrivial_scc = float(sizes[nontrivial].sum()) / n_nodes if n_nodes else 0

    # cycle-edge fraction: edges INSIDE an SCC of size >= 2 (non-trivial SCCs are
    # exactly the sets where every pair of nodes is on a directed cycle, so every
    # edge between two nodes of the same non-trivial SCC is on some cycle).
    coo = A.tocoo()
    src, dst = coo.row.astype(np.int64), coo.col.astype(np.int64)
    edge_in_nontrivial_scc = sizes[scc_labels[src]] >= MIN_SCC_SIZE_FOR_RECURRENT
    cycle_edge_fraction = float(edge_in_nontrivial_scc.sum()) / n_edges if n_edges else 0.0

    # cycle length distribution
    cyc_len = cycle_length_distribution(A)

    # ORN-reachable core
    orn_reachable_frac, recurrent_core_frac = orn_reachable_recurrent_core(A, orn_rows, scc_labels)

    # spectrum
    spec, gap = top_eigenvalues(A, k=min(8, max(1, n_nodes - 2)))

    # controllability at each K
    if B is None:
        # default: one-hot ORN inputs
        rng = np.random.default_rng(0)
        B_dense = np.zeros((n_nodes, len(orn_rows)), dtype=np.float64)
        B_dense[orn_rows, np.arange(len(orn_rows))] = 1.0
    else:
        B_dense = np.asarray(B, dtype=np.float64)
    controllability = {}
    for K in K_list:
        ck = controllability_krylov(A, B_dense, K=K)
        controllability[f"K={K}"] = ck.as_dict()

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "wcc_largest_fraction": wcc_largest_fraction,
        "wcc_count": n_wcc,
        "scc_count": n_scc,
        "scc_largest_fraction": scc_largest_fraction,
        "nodes_in_nontrivial_scc_fraction": nodes_in_nontrivial_scc,
        "cycle_edge_fraction": cycle_edge_fraction,
        "cycle_length_counts": {int(k): int(v) for k, v in cyc_len.items()},
        "orn_reachable_nodes": orn_reachable_frac,
        "orn_reachable_in_nontrivial_scc_fraction": recurrent_core_frac,
        "spectrum_top": list(spec),
        "spectrum_gap": float(gap) if gap != float("inf") else None,
        "controllability_K": controllability,
        "orn_rows_size": int(len(orn_rows)),
        "B_shape": list(B_dense.shape),
    }


__all__ = [
    "MIN_SCC_SIZE_FOR_RECURRENT",
    "StructuralMetrics",
    "ControllabilityKrylov",
    "wcc_scc_decomposition",
    "cycle_edge_fraction",
    "cycle_length_distribution",
    "orn_reachable_recurrent_core",
    "top_eigenvalues",
    "controllability_krylov",
    "measure_substrate",
]