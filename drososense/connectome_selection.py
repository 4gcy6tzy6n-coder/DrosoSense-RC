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
        seed: Seed used by the selection RNG.
        full_graph_n: Size of the graph the selection is into.
        identity: True when the selection is the whole graph (no sampling).
        sha256: SHA-256 of the sorted root-id bytes, or "" for identity.
        root_ids: The root ids of the selected nodes (empty for identity).
    """

    node_indices: np.ndarray
    target_n: int
    seed: int
    full_graph_n: int
    identity: bool
    sha256: str
    root_ids: np.ndarray

    def describe(self) -> dict:
        """JSON-serialisable provenance view for run records."""
        return {
            "method": "identity(full_graph)" if self.identity
            else "connectome.select_neurons (S0–S4, layer-stratified, seeded)",
            "target_n": int(self.target_n),
            "seed": int(self.seed),
            "full_graph_n": int(self.full_graph_n),
            "identity": bool(self.identity),
            "n_selected": int(self.node_indices.size),
            "sha256_sorted_root_ids": self.sha256,
            "node_index_sha256": self._node_index_sha256(),
        }

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
