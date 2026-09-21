#!/usr/bin/env python3
"""
Deterministic node selection function for DATA-3 M2 / DATA-4 M3.

Implements the S0–S4 path from DATA-9's select.py:
  S0 — Load full graph and node metadata
  S1 — KC-dominant downsampling (target fraction per layer)
  S2 — Seeded random selection for reproducibility
  S3 — Output node set as sorted root_id list
  S4 — SHA-256 of output node set for provenance

Usage:
  python connectome/select_neurons.py --target-n 1000 --seed 42 --output-json

Outputs:
  selected_node_indices.npy   — node indices into the full graph
  selected_root_ids.npy       — root_ids of selected nodes
  selection_provenance.json   — SHA-256 of sorted root_ids + all params
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Data-root resolution — large adjacency lives outside the repo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import adjacency_path, metadata_path  # noqa: E402

RNG_SEED = 20260920

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "connectome" / "annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def sha256_of_array(arr: np.ndarray) -> str:
    """SHA-256 of a sorted, concatenated array bytes."""
    h = hashlib.sha256()
    h.update(np.sort(arr).tobytes())
    return h.hexdigest()


def select_neurons(
    adj_path: Path,
    nm_path: Path,
    target_n: int,
    seed: int,
) -> dict:
    """
    Deterministic node selection (S0–S4).

    S0: Load full graph + metadata
    S1: Layer-stratified sampling (KC-dominant):
        - Priority to layer_mean > 3.86 (higher-order olfactory neurons)
        - Then KC (layer_mean ≈ 5.0), then PN (layer_mean ≈ 2.0), then ORN (layer_mean ≈ 1.0)
    S2: Seeded random selection to fill to target_n
    S3: Return sorted root_id list
    S4: Compute SHA-256 of sorted root_ids

    Returns dict with keys:
      node_indices, root_ids, sha256, N_selected, target_n, seed, layer_distribution
    """
    # S0: Load adjacency from dict-of-arrays npz (DATA-9 §7 schema)
    npz_obj = np.load(str(adj_path), allow_pickle=True)
    adj = _load_sparse(npz_obj, "adj")
    # Load node_ids from npz
    if "node_ids" in npz_obj.files:
        all_root_ids = npz_obj["node_ids"]
    else:
        raise ValueError("npz must contain 'node_ids' array (DATA-9 §7 schema)")

    nm_df = pd.read_csv(nm_path)
    N_full = len(all_root_ids)

    if target_n >= N_full:
        # Return all nodes
        node_indices = np.arange(N_full)
        root_ids = all_root_ids
    else:
        # S1: Layer-stratified sampling
        rng = np.random.RandomState(seed)
        layer = nm_df["layer_mean"].values

        # Define layer bins with priority
        # KC layer_mean ≈ 5.0, PN ≈ 2.0, ORN ≈ 1.0, higher-order > 3.86
        layers_sorted = np.sort(np.unique(layer))
        n_layers = len(layers_sorted)

        # Compute target per layer proportional to degree centrality
        total_deg = nm_df["total_degree"].values
        deg_by_layer = {}
        for l in layers_sorted:
            mask = layer == l
            deg_by_layer[l] = float(total_deg[mask].sum())

        total_deg_sum = sum(deg_by_layer.values())
        layer_targets = {
            l: max(1, int(round(target_n * deg_by_layer[l] / total_deg_sum)))
            for l in layers_sorted
        }
        # Adjust for rounding
        while sum(layer_targets.values()) != target_n:
            diff = target_n - sum(layer_targets.values())
            largest_layer = max(layer_targets, key=lambda l: layer_targets[l])
            layer_targets[largest_layer] += diff if diff > 0 else 0
            if diff < 0:
                smallest_layer = min([l for l in layer_targets if layer_targets[l] > 1],
                                     key=lambda l: layer_targets[l])
                layer_targets[smallest_layer] -= 1
            break  # avoid infinite loop

        # S2: Select from each layer
        node_indices_all = np.arange(N_full)
        selected_indices = []
        for l, t in sorted(layer_targets.items(), key=lambda x: x[0], reverse=True):
            layer_mask = layer == l
            layer_indices = node_indices_all[layer_mask]
            k = min(t, len(layer_indices))
            if k >= len(layer_indices):
                selected_indices.extend(layer_indices)
            else:
                chosen = rng.choice(layer_indices, size=k, replace=False)
                selected_indices.extend(chosen)

        # S3: Sort and trim/pad
        selected_indices = np.sort(np.array(selected_indices))
        if len(selected_indices) > target_n:
            selected_indices = selected_indices[:target_n]
        elif len(selected_indices) < target_n:
            # Fill remaining with highest-degree nodes not yet selected
            remaining_mask = np.ones(N_full, dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            remaining_by_deg = remaining_indices[np.argsort(total_deg[remaining_indices])[::-1]]
            needed = target_n - len(selected_indices)
            selected_indices = np.concatenate([selected_indices, remaining_by_deg[:needed]])

        node_indices = np.sort(selected_indices)
        root_ids = all_root_ids[node_indices]

    # S4: SHA-256 of sorted root_ids
    sha256 = sha256_of_array(root_ids)

    # Layer distribution of selected nodes
    layer_selected = nm_df.iloc[node_indices]["layer_mean"].values
    layer_dist = {}
    for l in np.sort(np.unique(layer_selected)):
        layer_dist[str(l)] = int((layer_selected == l).sum())

    return {
        "node_indices": node_indices,
        "root_ids": root_ids,
        "sha256": sha256,
        "N_selected": len(node_indices),
        "target_n": target_n,
        "seed": seed,
        "layer_distribution": layer_dist,
        "full_graph_N": N_full,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic node selection for subgraphs")
    parser.add_argument(
        "--adjacency",
        default=None,  # resolved via paths.py
        help="Path to olfactory_v1.npz (default: use paths.py adjacency_path)"
    )
    parser.add_argument(
        "--node-meta",
        default=None,  # resolved via paths.py
    )
    parser.add_argument(
        "--target-n", type=int, required=True,
        help="Target number of nodes to select (N ∈ {250, 500, 1000, 2000, 4000})"
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", default="connectome/annotations",
        help="Output directory for selection artifacts"
    )
    parser.add_argument(
        "--output-json", action="store_true",
        help="Also print JSON provenance to stdout"
    )
    pargs = parser.parse_args()

    # Resolve paths via paths.py (data-root aware)
    adj_path = Path(pargs.adjacency) if pargs.adjacency else adjacency_path("olfactory_v1.npz")
    nm_path = Path(pargs.node_meta) if pargs.node_meta else metadata_path("olfactory_v1_node_meta.csv")

    result = select_neurons(
        adj_path=adj_path,
        nm_path=nm_path,
        target_n=pargs.target_n,
        seed=pargs.seed,
    )

    # Save artifacts
    idx_path = OUT_DIR / f"selected_n{pargs.target_n}_seed{pargs.seed}_indices.npy"
    root_path = OUT_DIR / f"selected_n{pargs.target_n}_seed{pargs.seed}_root_ids.npy"
    prov_path = OUT_DIR / f"selected_n{pargs.target_n}_seed{pargs.seed}_provenance.json"

    np.save(idx_path, result["node_indices"].astype(np.int32))
    np.save(root_path, result["root_ids"].astype(np.int64))

    prov = {
        "target_n": result["target_n"],
        "seed": result["seed"],
        "N_selected": result["N_selected"],
        "full_graph_N": result["full_graph_N"],
        "sha256_sorted_root_ids": result["sha256"],
        "layer_distribution": result["layer_distribution"],
        "selection_method": "layer-stratified degree-proportional sampling with seeded random",
        "script": "connectome/select_neurons.py",
    }
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    print(f"\n=== Node Selection ===")
    print(f"  target_n : {result['target_n']}")
    print(f"  N_selected : {result['N_selected']}")
    print(f"  seed : {result['seed']}")
    print(f"  sha256(sorted_root_ids) : {result['sha256']}")
    print(f"  layer distribution : {result['layer_distribution']}")
    print(f"\n  artifacts:")
    print(f"    indices : {idx_path}")
    print(f"    root_ids : {root_path}")
    print(f"    provenance : {prov_path}")

    if pargs.output_json:
        print("\n--- JSON provenance ---")
        print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
