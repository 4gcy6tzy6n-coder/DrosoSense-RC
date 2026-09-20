#!/usr/bin/env python3
"""
Compute and append extended topology stats to the olfactory connectome metadata.
Memory-light: expensive metrics computed in sequence, path length via BFS sampling.

Also computes the size-structure curve:
  min_synapses ∈ {1, 2, 3, 5} × cell_type_filter ∈ {all, core_only}
  → (N, M, density, spectral_radius)

Run AFTER build_olfactory_connectome.py:
  python connectome/compute_topology_stats.py [--size-curve]
"""

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from itertools import product

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

# Data-root resolution — large raw inputs live outside the repo (connectome/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import raw_path  # noqa: E402

ROOT = Path(__file__).parent.parent
ADJ_PATH = ROOT / "connectome/adjacency/olfactory_v1.npz"
META_JSON = ROOT / "connectome/metadata/olfactory_v1_meta.json"
OLF_TABLE = raw_path("neuron_class_ranking_df_783-olfactory-10000.feather")

# BFS sampling params — aligned with build script RNG_SEED
BFS_NODES = 200
MAX_PATHS = 2_000_000  # global cap on sampled paths
RNG_SEED = 20260920

# Size-structure curve params
MIN_SYN_VALUES = [1, 2, 3, 5]
CORE_ONLY_VALUES = [False, True]

parser = argparse.ArgumentParser()
parser.add_argument("--size-curve", action="store_true",
                    help="Compute size-structure curve (N, M, density, rho) × (min_syn, core_only)")
parser.add_argument("--spectral-only", action="store_true",
                    help="Only compute spectral radius")
args = parser.parse_args()


def _load_sparse(npz_obj, prefix: str) -> sp.spmatrix:
    """Reconstruct a sparse matrix from dict-of-arrays npz entry."""
    data_key = f"{prefix}_data"
    if data_key not in npz_obj.files:
        return None
    fmt = str(npz_obj[f"{prefix}_format"].item())
    data = npz_obj[data_key]
    indices = npz_obj[f"{prefix}_indices"]
    indptr = npz_obj[f"{prefix}_indptr"]
    shape = tuple(npz_obj[f"{prefix}_shape"])
    if fmt == "csr":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    elif fmt == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    return sp.coo_matrix((data, indices), shape=shape)


def load_adj_and_meta():
    """Load adjacency and node metadata."""
    npz_data = np.load(str(ADJ_PATH), allow_pickle=True)
    adj = _load_sparse(npz_data, "adj")
    nm = None
    nm_path = ROOT / "connectome/metadata/olfactory_v1_node_meta.csv"
    if nm_path.exists():
        import pandas as pd
        nm = pd.read_csv(nm_path)
    return adj, nm


def spectral_radius(adj_ud):
    """Compute undirected spectral radius."""
    try:
        ev = eigsh(adj_ud.astype(float), k=1, which='LM', return_eigenvectors=False)
        return float(np.abs(ev[0]))
    except Exception:
        return float("nan")


def bfs_path_stats(adj_ud, N, rng_seed, bfs_nodes=200, max_paths=2_000_000):
    """BFS-based path length statistics (undirected, unweighted)."""
    rng = random.Random(rng_seed)
    bfs_roots = rng.sample(list(range(N)), k=min(bfs_nodes, N))

    # Build adjacency list (undirected)
    adj_coo = adj_ud.tocoo()
    adj_list = [[] for _ in range(N)]
    for i, j in zip(adj_coo.row, adj_coo.col):
        if i != j:
            adj_list[i].append(j)
    del adj_coo

    path_lengths = []
    paths_sampled = 0
    for bi, root in enumerate(bfs_roots):
        if bi % 20 == 0:
            print(f"    BFS {bi + 1}/{len(bfs_roots)} ...", file=sys.stderr)
        seen = {root}
        queue = deque([(root, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in adj_list[node]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, dist + 1))
                    path_lengths.append(dist + 1)
                    paths_sampled += 1
        if paths_sampled >= max_paths:
            print(f"    cap reached after {paths_sampled:,} paths", file=sys.stderr)
            break

    del adj_list
    if not path_lengths:
        return None
    return {
        "avg_path_length": float(np.mean(path_lengths)),
        "avg_path_length_std": float(np.std(path_lengths)),
        "avg_path_length_max": int(np.max(path_lengths)),
        "avg_path_length_N": len(path_lengths),
    }


def compute_curve_point(olf_root_ids, proof_df, min_syn, core_only):
    """
    Compute (N, M, density, rho) for a given min_syn and core_only setting.
    Uses the same aggregation approach as build_olfactory_connectome.py.
    """
    if core_only:
        import pandas as pd
        olf = pd.read_feather(OLF_TABLE)
        core_ids = set(olf[olf["layer_mean"] <= 4]["root_id"].values)
        olf_root_ids_filtered = olf_root_ids & core_ids
    else:
        olf_root_ids_filtered = olf_root_ids

    mask = (
        proof_df["pre_pt_root_id"].isin(olf_root_ids_filtered) &
        proof_df["post_pt_root_id"].isin(olf_root_ids_filtered) &
        (proof_df["syn_count"] >= min_syn)
    )
    edges = proof_df[mask]
    N = len(olf_root_ids_filtered)
    M = len(edges)
    density = float(M / (N * (N - 1))) if N > 1 else 0.0

    # Build adjacency for spectral radius
    all_nodes = sorted(olf_root_ids_filtered)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    rows = edges["pre_pt_root_id"].map(node_to_idx).values
    cols = edges["post_pt_root_id"].map(node_to_idx).values
    data = np.ones(len(rows), dtype=float)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    adj_ud = (adj + adj.T).tocsr()
    adj_ud.data = np.ones(len(adj_ud.data), dtype=float)
    rho = spectral_radius(adj_ud)

    return {
        "min_synapses": min_syn,
        "core_only": core_only,
        "N": N,
        "M": M,
        "density": density,
        "spectral_radius": rho,
    }


def main():
    print("Loading adjacency ...", file=sys.stderr)
    npz_data = np.load(str(ADJ_PATH), allow_pickle=True)
    adj = _load_sparse(npz_data, "adj")
    N = adj.shape[0]
    print(f"  N={N}, edges={adj.nnz:,}", file=sys.stderr)

    # Undirected adjacency for connectivity metrics
    adj_ud = (adj + adj.T).tocsr()
    adj_ud.data = np.ones(len(adj_ud.data), dtype=bool)

    stats = {}

    # Spectral radius of base adjacency
    print("Computing spectral radius ...", file=sys.stderr)
    adj_ud_f = adj_ud.astype(float)
    rho = spectral_radius(adj_ud_f)
    stats["olfactory_spectral_radius"] = rho
    print(f"  spectral_radius = {rho:.4f}", file=sys.stderr)
    del adj_ud_f

    # Path length via BFS sampling
    print(f"Computing path length via BFS ({BFS_NODES} roots, cap {MAX_PATHS:,} paths) ...",
          file=sys.stderr)
    rng = random.Random(RNG_SEED)
    bfs_result = bfs_path_stats(adj_ud, N, RNG_SEED, BFS_NODES, MAX_PATHS)
    if bfs_result:
        stats["olfactory_avg_path_length"] = bfs_result["avg_path_length"]
        stats["olfactory_avg_path_length_std"] = bfs_result["avg_path_length_std"]
        stats["olfactory_avg_path_length_max"] = bfs_result["avg_path_length_max"]
        stats["olfactory_avg_path_length_N"] = bfs_result["avg_path_length_N"]
        print(f"  avg_path_length = {bfs_result['avg_path_length']:.4f} "
              f"(std={bfs_result['avg_path_length_std']:.4f}, "
              f"max={bfs_result['avg_path_length_max']}, "
              f"N_paths={bfs_result['avg_path_length_N']:,})", file=sys.stderr)
    else:
        stats["olfactory_avg_path_length"] = None

    # Clustering coefficient (undirected, weighted) — use sampling for large graphs
    print("Computing clustering coefficient (sampling 2000 nodes) ...", file=sys.stderr)
    try:
        # Sample 2000 nodes for clustering (full graph O(N) per node is too expensive)
        rng2 = random.Random(RNG_SEED)
        sample_nodes = rng2.sample(list(range(N)), k=min(2000, N))
        G_ud = nx.Graph()
        adj_coo = adj_ud.tocoo()
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i != j:
                G_ud.add_edge(i, j)
        # Average clustering on sampled subgraph
        clust_values = []
        for node in sample_nodes:
            if node in G_ud:
                nb = list(G_ud.neighbors(node))
                if len(nb) < 2:
                    continue
                # Local clustering
                neighbors_set = set(nb)
                possible = len(nb) * (len(nb) - 1)
                if possible == 0:
                    continue
                actual = sum(1 for u in nb for v in nb if u < v and v in neighbors_set and G_ud.has_edge(u, v))
                clust_values.append(actual / possible)
        stats["olfactory_clustering"] = float(np.mean(clust_values)) if clust_values else 0.0
        print(f"  clustering = {stats['olfactory_clustering']:.6f} (sampled {len(clust_values)} nodes)", file=sys.stderr)
    except Exception as e:
        stats["olfactory_clustering"] = None
        print(f"  clustering FAILED: {e}", file=sys.stderr)

    # Assortativity (undirected, degree Pearson)
    print("Computing assortativity ...", file=sys.stderr)
    try:
        G_ud2 = nx.Graph()
        adj_coo2 = adj_ud.tocoo()
        for i, j in zip(adj_coo2.row, adj_coo2.col):
            if i != j:
                G_ud2.add_edge(i, j)
        stats["olfactory_assortativity"] = float(nx.degree_assortativity_coefficient(G_ud2))
        print(f"  assortativity = {stats['olfactory_assortativity']:.6f}", file=sys.stderr)
    except Exception as e:
        stats["olfactory_assortativity"] = None
        print(f"  assortativity FAILED: {e}", file=sys.stderr)

    # Modularity via Louvain (undirected, using networkx 3.6.1)
    print("Computing modularity ...", file=sys.stderr)
    try:
        G_ud3 = nx.Graph()
        adj_coo3 = adj_ud.tocoo()
        for i, j in zip(adj_coo3.row, adj_coo3.col):
            if i != j:
                G_ud3.add_edge(i, j)
        communities = nx.algorithms.community.louvain_communities(G_ud3, seed=RNG_SEED)
        partition = {node: i for i, comm in enumerate(communities) for node in comm}
        stats["olfactory_modularity"] = float(nx.algorithms.community.modularity(G_ud3, communities))
        print(f"  modularity = {stats['olfactory_modularity']:.6f} ({len(communities)} communities)", file=sys.stderr)
    except Exception as e:
        stats["olfactory_modularity"] = None
        print(f"  modularity FAILED: {e}", file=sys.stderr)

    # Update meta JSON
    if META_JSON.exists():
        with open(META_JSON) as f:
            meta = json.load(f)
    else:
        meta = {}

    if "topology_stats" not in meta:
        meta["topology_stats"] = {}
    meta["topology_stats"].update(stats)

    # Size-structure curve
    if args.size_curve:
        print("\nComputing size-structure curve ...", file=sys.stderr)
        import pandas as pd
        proof_path = raw_path("proofread_connections_783.feather")
        print(f"  loading {proof_path} ...", file=sys.stderr)
        proof_df = pd.read_feather(proof_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])

        olf = pd.read_feather(OLF_TABLE)
        olf_root_ids = set(olf["root_id"].values)
        print(f"  olfactory neurons: {len(olf_root_ids):,}", file=sys.stderr)

        curve = []
        for min_syn, core_only in product(MIN_SYN_VALUES, CORE_ONLY_VALUES):
            label = f"min_syn={min_syn}, core_only={core_only}"
            print(f"  computing {label} ...", file=sys.stderr)
            try:
                pt = compute_curve_point(olf_root_ids, proof_df, min_syn, core_only)
                curve.append(pt)
                print(f"    N={pt['N']:,}, M={pt['M']:,}, density={pt['density']:.6f}, "
                      f"ρ={pt['spectral_radius']:.2f}", file=sys.stderr)
            except Exception as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                curve.append({
                    "min_synapses": min_syn,
                    "core_only": core_only,
                    "N": None, "M": None,
                    "density": None, "spectral_radius": None,
                    "error": str(e),
                })

        meta["size_structure_curve"] = curve
        print(f"  curve saved: {len(curve)} points", file=sys.stderr)

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nStats appended to {META_JSON}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
