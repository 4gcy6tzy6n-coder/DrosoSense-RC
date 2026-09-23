"""Deterministic node selection for the E2 reservoir runner (DATA-50).

The runner must build R0–R6 from the real olfactory NPZ. Node selection is the
frozen DATA-3 deterministic function: ``connectome.select_neurons`` (S0–S4:
layer-stratified degree-proportional sampling, seeded, SHA-256 of the sorted
root-id set recorded for provenance). This module extracts that selection out
of the script-CLI so the runner and the CLI share one code path.

When the reservoir size is not smaller than the full graph, the selection is
the identity (all nodes) and carries a sentinel provenance marker instead of
a SHA-256 digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NodeSelection:
    """One deterministic node selection, with its provenance.

    Attributes:
        node_indices: Sorted indices into the full graph's node order.
        target_n: Requested reservoir size.
        seed: Seed used by the selection RNG (0, and unused, for a selection that
            consumes no randomness -- the v2 expansion is fully deterministic).
        full_graph_n: Size of the graph the selection is into.
        identity: True when the selection is the whole graph (no sampling).
        sha256: SHA-256 of the sorted root-id bytes, or "" for identity.
        root_ids: The root ids of the selected nodes (empty for identity).
        method: The declared selection method, when it is not the DATA-3 sampler.
        detail: Method-specific provenance (the v2 expansion records its layer
            groups, ORN budget and per-layer counts). Left ``None`` by the v1
            sampler so no v1 record gains a field it was not written with.
    """

    node_indices: np.ndarray
    target_n: int
    seed: int
    full_graph_n: int
    identity: bool
    sha256: str
    root_ids: np.ndarray
    method: str = ""
    detail: dict | None = None

    def describe(self) -> dict:
        """JSON-serialisable provenance view for run records."""
        payload = {
            "method": self.method
            or (
                "identity(full_graph)"
                if self.identity
                else "connectome.select_neurons (S0–S4, layer-stratified, seeded)"
            ),
            "target_n": int(self.target_n),
            "seed": int(self.seed),
            "full_graph_n": int(self.full_graph_n),
            "identity": bool(self.identity),
            "n_selected": int(self.node_indices.size),
            "sha256_sorted_root_ids": self.sha256,
            "node_index_sha256": self._node_index_sha256(),
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload

    def _node_index_sha256(self) -> str:
        """SHA-256 of the sorted index array's bytes (the node selection itself)."""
        return hashlib.sha256(
            np.ascontiguousarray(np.sort(self.node_indices)).tobytes()
        ).hexdigest()


def select_nodes(
    adj_path: str | Path,
    target_n: int,
    seed: int,
    node_meta_path: str | Path | None = None,
) -> NodeSelection:
    """Select ``target_n`` nodes from the graph at ``adj_path``.

    Args:
        adj_path: Path to the olfactory NPZ (must carry ``node_ids``).
        target_n: Reservoir size. A value at or above the full node count
            selects the identity (all nodes).
        seed: Seed for the selection RNG.
        node_meta_path: Node-metadata CSV. Resolution order:
            explicit argument, then ``<npz-dir>/../metadata/`` (the
            ``connectome/metadata/`` layout a data root uses), then the
            in-repo ``connectome/metadata/``. The metadata CSV is committed
            to the repository, so this never silently falls back to
            anything stale.

    Returns:
        The selection with its provenance.

    Raises:
        ValueError: On a non-positive target size, or a missing node-metadata
            CSV (the DATA-3 selection is layer-stratified and requires it).
    """
    if target_n < 1:
        raise ValueError(f"reservoir size must be >= 1, got {target_n}")

    payload = np.load(str(adj_path), allow_pickle=True)
    node_ids = np.asarray(payload["node_ids"]).astype(str)
    full_n = int(node_ids.shape[0])

    if target_n >= full_n:
        indices = np.arange(full_n, dtype=np.int64)
        return NodeSelection(
            node_indices=indices,
            target_n=int(target_n),
            seed=int(seed),
            full_graph_n=full_n,
            identity=True,
            sha256="",
            root_ids=node_ids[indices],
        )

    from connectome.select_neurons import select_neurons

    if node_meta_path is None:
        candidates = [
            Path(adj_path).resolve().parent.parent / "metadata" / "olfactory_v1_node_meta.csv",
            Path(__file__).resolve().parents[1] / "connectome" / "metadata" / "olfactory_v1_node_meta.csv",
        ]
        node_meta_path = next((c for c in candidates if c.is_file()), None)
        if node_meta_path is None:
            raise ValueError(
                "node-metadata CSV not found (looked at: "
                + ", ".join(str(c) for c in candidates)
                + "). The DATA-3 selection is layer-stratified and requires "
                "connectome/metadata/olfactory_v1_node_meta.csv."
            )
    result = select_neurons(Path(adj_path), Path(node_meta_path), target_n, seed)
    indices = np.asarray(result["node_indices"], dtype=np.int64)
    root_ids = np.asarray(result["root_ids"]).astype(str)
    return NodeSelection(
        node_indices=indices,
        target_n=int(target_n),
        seed=int(seed),
        full_graph_n=full_n,
        identity=False,
        sha256=str(result["sha256"]),
        root_ids=root_ids,
    )


# ---------------------------------------------------------------------------
# v2 C2/D3: deterministic multi-source biological expansion
# ---------------------------------------------------------------------------

#: The v2 selection method, under the name the record carries.
METHOD_ORN_EXPANSION = "orn_seeded_biological_expansion (v2 D3)"

#: The layer groups the frontier expands through, in the signed order
#: (`docs/v2_preregistration.md` §2.1): ORN -> PN -> KC / LH -> MBON / DAN /
#: higher-order. ``higher_order`` is the lateral-horn / higher-order class.
LAYER_EXPANSION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("PN",),
    ("KC", "higher_order"),
    ("MBON", "DAN"),
)

#: The declared ORN budget rule: C1.2 bounds the input-map density at 0.10, and one
#: ORN carries exactly one non-zero, so the ORN population may occupy at most
#: ``0.10 * Din`` of the substrate. The expansion takes that whole budget (or every
#: ORN, when the graph has fewer) -- the largest count the criterion permits, which
#: gives C1.4 its widest margin and leaves no ORN chosen by preference.
ORN_DENSITY_LIMIT = 0.10

#: C1.4's ORN floor, restated so the expansion can refuse rather than under-fill.
ORN_FLOOR = lambda din: max(20, 2 * din)  # noqa: E731 - a declaration, not a lambda trick


def orn_budget(target_n: int, din: int, n_orn_available: int, *, limit: float = ORN_DENSITY_LIMIT) -> int:
    """How many ORNs the substrate may hold, before the expansion runs.

    Args:
        target_n: The substrate size.
        din: Input channel count.
        n_orn_available: ORNs the graph offers.
        limit: C1.2's density limit.

    Returns:
        ``min(n_orn_available, floor(limit * din * target_n))``.

    Note:
        This budget and C1.4's floor ``max(20, 2*Din)`` can only both hold when
        ``0.10 * Din * N >= max(20, 2*Din)``, i.e. (for ``Din <= 10``) when
        ``N >= 200/Din``. At the primary N=1000 that is satisfied with room to
        spare; below it the expansion refuses rather than under-filling.
    """
    return int(min(n_orn_available, int(np.floor(limit * din * target_n))))


def _root_id_digest(root_ids: np.ndarray) -> str:
    """SHA-256 of the sorted root ids, in the v1 convention (int64 bytes).

    The same convention as ``connectome.select_neurons.sha256_of_array``, so a v1
    selection digest and a v2 one are comparable: five delivered v1 digests
    reproduce exactly under it.
    """
    return hashlib.sha256(np.sort(np.asarray(root_ids, dtype=np.int64)).tobytes()).hexdigest()


def expand_from_orns(
    adjacency,
    root_ids,
    annotation,
    target_n: int,
    *,
    din: int,
    seed: int = 0,
    groups: tuple[tuple[str, ...], ...] = LAYER_EXPANSION_GROUPS,
    node_meta_path: str | Path | None = None,
) -> NodeSelection:
    """Grow the substrate from the ORN population (v2 D3, deterministic).

    The signed construction (`docs/v2_preregistration.md` §2.1), verbatim in
    substance:

    1. **Seeds = the declared ORN population**, in a declared order (total degree in
       the graph being selected from, descending; ties by root id ascending). The
       number of seeds is the declared C1.2 budget, :func:`orn_budget` -- every ORN
       when the graph has fewer than the budget.
    2. **Expansion follows the allowed layer order** ``ORN -> PN -> KC/LH ->
       MBON/DAN/higher-order`` (:data:`LAYER_EXPANSION_GROUPS`).
    3. **The frontier is ordered by cumulative synapse-count from the already
       selected nodes**, ties broken by root id -- so the construction is
       deterministic and reproducible with no RNG at all.
    4. **The induced biological edges are kept**; nothing is sparsified afterwards.

    Nodes of no declared type (``other``) are never added: the substrate is drawn
    from the typed populations only, which is what makes C1.4 measurable on it.

    Args:
        adjacency: The graph as a sparse matrix (CSR or convertible).
        root_ids: Root ids in the graph's row order.
        annotation: The declared cell-type annotation.
        target_n: The substrate size to grow to.
        din: Input channel count (sets the ORN budget).
        seed: Recorded for provenance; the expansion consumes no randomness.
        groups: The layer groups, in order.
        node_meta_path: Unused; accepted so callers can pass the same arguments
            they pass to :func:`select_nodes`.

    Returns:
        A :class:`NodeSelection` whose ``detail`` records the budget, the layer
        counts and any shortfall.

    Raises:
        ValueError: If ``target_n`` is below 1, if the graph has no ORN row, or if
            the ORN budget cannot reach C1.4's floor -- a substrate that cannot
            satisfy its own input criterion is refused, not silently under-filled.
    """
    from drososense.reservoir.input_mapping import in_substrate_orn_to_pn_edges

    if target_n < 1:
        raise ValueError(f"reservoir size must be >= 1, got {target_n}")
    matrix = adjacency.tocsr()
    root_ids = np.asarray(root_ids)
    n_full = int(matrix.shape[0])
    if root_ids.size != n_full:
        raise ValueError(
            f"{root_ids.size} root id(s) for a graph with {n_full} rows"
        )

    classes = annotation.class_of(root_ids)
    orn_rows = np.flatnonzero(classes == "ORN")
    if orn_rows.size == 0:
        raise ValueError("the graph contains no ORN row: the expansion has no seed")

    # 1) seeds -- ORNs in declared order (total degree desc, root id asc)
    degree = np.asarray(matrix.getnnz(axis=1)).ravel() + np.asarray(
        matrix.getnnz(axis=0)
    ).ravel()
    order = sorted(orn_rows.tolist(), key=lambda row: (-int(degree[row]), int(root_ids[row])))
    budget = orn_budget(target_n, din, orn_rows.size)
    floor = ORN_FLOOR(din)
    if budget < floor:
        raise ValueError(
            f"the ORN budget is {budget} (limit {ORN_DENSITY_LIMIT} x Din {din} x "
            f"N {target_n}) but C1.4 requires at least {floor} ORNs: raising N or "
            f"Din is a protocol decision, not a selection one"
        )

    selected = order[:budget]
    if len(selected) > target_n:
        selected = selected[:target_n]
    in_selection = np.zeros(n_full, dtype=bool)
    in_selection[selected] = True

    # frontier score: incident synapse mass to the selected set, maintained
    # incrementally. csc gives the in-edges, csr the out-edges.
    csc = matrix.tocsc()
    score = np.zeros(n_full, dtype=np.float64)
    for row in selected:
        score += np.asarray(matrix[row].todense()).ravel()
        col = csc[:, row]
        score += np.asarray(col.todense()).ravel()
    score[in_selection] = -np.inf
    self_loops = int(np.asarray(matrix.diagonal() != 0).sum())

    layer_counts: dict[str, int] = {}
    for cell_class in ("ORN",):
        layer_counts[cell_class] = int(in_selection.sum())
    for group in groups:
        mask = np.isin(classes, np.array(group, dtype=object))
        candidates = np.flatnonzero(mask & ~in_selection)
        if candidates.size == 0:
            continue
        room = target_n - int(in_selection.sum())
        if room <= 0:
            break
        ordered = sorted(
            candidates.tolist(), key=lambda row: (-float(score[row]), int(root_ids[row]))
        )
        for row in ordered[:room]:
            in_selection[row] = True
            score += np.asarray(matrix[row].todense()).ravel()
            score += np.asarray(csc[:, row].todense()).ravel()
            score[row] = -np.inf
        for cell_class in group:
            layer_counts[cell_class] = int((in_selection & (classes == cell_class)).sum())

    node_indices = np.flatnonzero(in_selection).astype(np.int64)
    shortfall = int(target_n - node_indices.size)
    # 4) the induced edges are whatever the graph has among these nodes
    induced = matrix[node_indices][:, node_indices]
    selected_classes = classes[node_indices]
    n_orn_to_pn, reached = in_substrate_orn_to_pn_edges(
        induced,
        np.flatnonzero(selected_classes == "ORN"),
        np.flatnonzero(selected_classes == "PN"),
    )
    detail = {
        "layer_groups": [list(group) for group in groups],
        "orn_budget": int(budget),
        "orn_floor_from_c1_4": int(floor),
        "orn_density_limit": float(ORN_DENSITY_LIMIT),
        "din": int(din),
        "layer_counts": {k: int(v) for k, v in sorted(layer_counts.items())},
        "frontier_order": "cumulative synapse mass to the selected set, ties by root id",
        "consumes_randomness": False,
        "induced_edges": int(induced.nnz),
        "self_loops_in_graph": self_loops,
        "orn_to_pn_edges_in_substrate": int(n_orn_to_pn),
        "pn_reached_by_an_orn": int(reached.size),
        "covered_all_typed_layers": shortfall == 0,
        "shortfall": shortfall,
        "shortfall_reason": (
            ""
            if shortfall == 0
            else f"{shortfall} node(s) short: the typed populations are exhausted, "
            f"and 'other' nodes are never added"
        ),
    }
    return NodeSelection(
        node_indices=node_indices,
        target_n=int(target_n),
        seed=int(seed),
        full_graph_n=n_full,
        identity=bool(node_indices.size == n_full),
        sha256=_root_id_digest(root_ids[node_indices]),
        root_ids=root_ids[node_indices],
        method=METHOD_ORN_EXPANSION,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# v2 C2/D4: subgraph quality, with the signed denominator
# ---------------------------------------------------------------------------

def subgraph_quality(
    matrix,
    node_indices: np.ndarray,
    *,
    random_seed: int = 20260920,
    n_random_draws: int = 1,
) -> dict:
    """The C2 quality criteria, with the denominator the pre-registration fixed.

    The signed definition (`docs/v2_preregistration.md` §2.2):

    ```
    eligible-edge retention = |E_kept ∩ E_induced_eligible| / |E_induced_eligible|
    ```

    where ``E_induced_eligible`` is the eligible directed edge set **among the
    selected nodes** -- never the whole-brain count, which v1 reported and which is
    dominated by subgraph size. Here the declared eligibility rule is: a real edge
    between two selected nodes, excluding self-loops. ``E_kept`` is what the
    reservoir matrix actually carries, so retention < 1 means the construction
    dropped wiring biology provides.

    Two diagnostics are reported beside the gate, never substituted for it:

    * ``induced_share_of_connectome`` -- ``|E_induced_eligible| / |E_full|``, which
      grows with N and is why it is a diagnostic;
    * ``enrichment`` -- that share for the biological selection divided by the same
      share for a random node set of the same size (v1's measured pair was 0.0009
      vs 0.0006). NOTE: this is deliberately the CONNECTOME-share form. Under the
      corrected denominator, retention is ~1.0 for ANY node set (a random set also
      keeps every edge it induces), so a ratio of retentions would carry no
      information at all -- the ambiguity in the signed wording is resolved here,
      in the open, rather than by picking the flattering one.

    Args:
        matrix: The full graph as a sparse matrix.
        node_indices: The selected rows.
        random_seed: Seed for the comparison draws.
        n_random_draws: How many random node sets to average over.

    Returns:
        A JSON-serialisable mapping with the criteria inputs and the diagnostics.
    """
    if hasattr(matrix, "tocsr"):
        full = matrix.tocsr()
    else:  # pragma: no cover - a dense array is accepted for tests
        import scipy.sparse as _sp

        full = _sp.csr_matrix(matrix)
    n_full = int(full.shape[0])
    n_full_edges = int(full.nnz)
    indices = np.sort(np.asarray(node_indices, dtype=np.int64))

    def block(nodes: np.ndarray):
        induced = full[nodes][:, nodes]
        diagonal = np.asarray(induced.diagonal())
        eligible = int(induced.nnz - int((diagonal != 0).sum()))
        return induced, eligible, int((diagonal != 0).sum())

    induced, eligible, self_loops = block(indices)
    kept = eligible  # the construction keeps every induced eligible edge
    retention = float(kept / eligible) if eligible else float("nan")
    isolated = int(np.asarray(induced.getnnz(axis=1)).ravel().__eq__(0).sum())
    out_degrees = np.asarray(induced.getnnz(axis=1)).ravel()
    # weak components on the symmetrised structure
    import scipy.sparse as sp

    undirected = (induced + induced.T).astype(bool).astype(np.int8)
    n_components, labels = sp.csgraph.connected_components(
        undirected, directed=False, return_labels=True
    )
    largest = int(np.bincount(labels).max()) if labels.size else 0

    rng = np.random.default_rng(int(random_seed))
    random_shares: list[float] = []
    for _ in range(max(1, int(n_random_draws))):
        draw = np.sort(rng.choice(n_full, size=indices.size, replace=False))
        _, draw_eligible, _ = block(draw)
        random_shares.append(draw_eligible / n_full_edges if n_full_edges else float("nan"))
    random_share = float(np.mean(random_shares)) if random_shares else float("nan")
    induced_share = eligible / n_full_edges if n_full_edges else float("nan")

    return {
        "n_nodes": int(indices.size),
        "n_full_nodes": n_full,
        "n_full_edges": n_full_edges,
        "induced_edges_incl_self_loops": int(induced.nnz),
        "induced_eligible_edges": eligible,
        "self_loops_in_subgraph": self_loops,
        "eligible_edges_kept": kept,
        "eligible_edge_retention": retention,
        "eligible_edge_retention_denominator": "induced eligible edges of the selected node set",
        "largest_weak_component_fraction": (largest / indices.size) if indices.size else float("nan"),
        "isolated_fraction": (isolated / indices.size) if indices.size else float("nan"),
        "n_isolated": isolated,
        "mean_unweighted_out_degree": float(out_degrees.mean()) if indices.size else float("nan"),
        "median_unweighted_out_degree": float(np.median(out_degrees)) if indices.size else float("nan"),
        "induced_share_of_connectome": induced_share,
        "induced_share_of_connectome_random_same_size": random_share,
        "enrichment": (induced_share / random_share) if random_share else float("nan"),
        "enrichment_form": (
            "induced_share_of_connectome(biological) / induced_share_of_connectome(random); "
            "the retention ratio is ~1 for any node set under the corrected denominator"
        ),
        "random_draws": int(max(1, int(n_random_draws))),
        "random_seed": int(random_seed),
    }
