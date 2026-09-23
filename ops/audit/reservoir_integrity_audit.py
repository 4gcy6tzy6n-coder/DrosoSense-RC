#!/usr/bin/env python3
"""M4-Audit — Reservoir Integrity Audit (graph-level half: A1, A7, A8).

PURPOSE. The delivered M4/E1 evidence is baseline-only (R0-R6 never ran), and the
open question the project must answer before any "R0 ~= R2" reading is admissible
is: *did R0 and R2 ever produce different, non-degenerate dynamics at all?*

This script answers the graph-level half of that question. It is READ-ONLY with
respect to the repository: it never writes a run record, never touches a test
split, never edits a frozen protocol file, and never starts a training run. Its
only outputs are JSON/CSV evidence under ``results/audit/m4_audit/``.

Audits implemented here
-----------------------
A1  R0/R2 adjacency identity: are they the same object? same bytes? how many
    edges differ? what is the edge-overlap |E_R0 & E_R2| / |E_R0|?
A7  R2 rewiring quality: does the Maslov-Sneppen swap preserve the *unweighted*
    degree sequence and edge count? self-loops? multi-edges? how many swaps
    actually succeeded, how far is the result from the fully-mixed
    configuration-model expectation, do 10 seeds give 10 distinct graphs, and is
    the routine even feasible at the full graph size the runner defaults to?
A8  Subgraph degeneracy: for each candidate N, how many nodes/edges does the
    induced subgraph the reservoir actually sees have, how many nodes are
    isolated, how many connected components, and (the load-bearing number) how
    much of each selected node's out-degree survives the sub-selection?

Usage
-----
    python ops/audit/reservoir_integrity_audit.py --only A1 A7 A8
    python ops/audit/reservoir_integrity_audit.py --sizes 250 500 1000 2000 4000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from connectome.paths import adjacency_path  # noqa: E402
from drososense.connectome_selection import select_nodes  # noqa: E402
from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    DOUBLE_EDGE_SWAP_ATTEMPTS,
    load_reservoir_topology_from_npz,
    make_degree_rewired,
)

OUT_DIR = REPO_ROOT / "results" / "audit" / "m4_audit"
SELECTION_SEED = 20260920  # RNG_SEED of the frozen DATA-3 selection


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sha16(obj) -> str:
    """SHA-256 (first 16 hex) of an array's bytes or a string."""
    if isinstance(obj, str):
        return hashlib.sha256(obj.encode()).hexdigest()[:16]
    arr = np.ascontiguousarray(obj)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _csc_edges(matrix: sp.spmatrix) -> set[tuple[int, int]]:
    """The directed edge set as Python tuples (small N only)."""
    coo = matrix.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def _binarize(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Structure only, weights dropped (this is what R2 actually builds)."""
    coo = matrix.tocoo()
    return sp.csr_matrix(
        (np.ones(coo.nnz, dtype=np.float64), (coo.row, coo.col)), shape=matrix.shape
    )


def _structure_sha(matrix: sp.spmatrix) -> str:
    """A digest that actually identifies the WIRING, not just the degree sequence.

    Degree-only digests (``indptr``, row sums) are useless here precisely
    because a degree-preserving rewiring deliberately holds them fixed: two
    different R2 draws have identical degrees by construction. The column
    indices are what change, so they must be in the digest.
    """
    binary = _binarize(matrix)
    return (
        _sha16(np.asarray(binary.indices))
        + "|"
        + _sha16(np.asarray(binary.indptr))
        + "|"
        + _sha16(np.asarray(binary.shape))
    )


def _weight_uniform(matrix: sp.spmatrix) -> bool | None:
    """True when every stored weight is the same value (R2's actual invariant)."""
    if matrix.nnz == 0:
        return None
    data = np.asarray(matrix.data, dtype=np.float64)
    return bool(data.max() - data.min() <= 1e-15)


def _degrees(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(matrix.sum(axis=1)).ravel(),
        np.asarray(matrix.sum(axis=0)).ravel(),
    )


def _unweighted_degrees(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    binary = _binarize(matrix)
    return (
        np.asarray(binary.sum(axis=1)).ravel(),
        np.asarray(binary.sum(axis=0)).ravel(),
    )


def _components(matrix: sp.spmatrix) -> dict:
    """Weak/strong component structure of the directed graph."""
    from scipy.sparse.csgraph import connected_components

    n = matrix.shape[0]
    n_wcc, lab_wcc = connected_components(matrix, directed=True, connection="weak")
    n_scc, lab_scc = connected_components(matrix, directed=True, connection="strong")
    sizes_wcc = np.bincount(lab_wcc, minlength=n_wcc) if n_wcc else np.array([])
    sizes_scc = np.bincount(lab_scc, minlength=n_scc) if n_scc else np.array([])
    return {
        "n_weak_components": int(n_wcc),
        "largest_weak_fraction": float(sizes_wcc.max() / n) if n and n_wcc else 0.0,
        "n_strong_components": int(n_scc),
        "largest_strong_fraction": float(sizes_scc.max() / n) if n and n_scc else 0.0,
    }


def _self_loops(matrix: sp.spmatrix) -> int:
    coo = matrix.tocoo()
    return int(np.sum(coo.row == coo.col))


def _multiplicity(matrix: sp.spmatrix) -> dict:
    """CSR sums duplicate coordinates, so compare raw COO nnz with CSR nnz."""
    coo = matrix.tocoo()
    deduped = sp.csr_matrix(
        (coo.data, (coo.row, coo.col)), shape=matrix.shape
    )
    return {
        "coo_nnz": int(coo.nnz),
        "csr_nnz": int(deduped.nnz),
        "n_duplicate_coordinates": int(coo.nnz - deduped.nnz),
    }


def _expected_shared_edges_under_mixing(
    out_deg: np.ndarray, in_deg: np.ndarray, n_edges: int
) -> float:
    """Configuration-model expectation of |E & E'| when both are fully mixed.

    For a directed configuration model with the same degree sequences, the
    probability of an edge i->j is ``d_out[i] * d_in[j] / M``, so the expected
    number of *shared* edges between two independent draws is

        sum_ij ( d_out[i] d_in[j] / M )^2
          = (sum_i d_out[i]^2)(sum_j d_in[j]^2) / M^2

    This is the principled "the rewiring went all the way" baseline. An observed
    overlap far above it means the swap chain did NOT mix.
    """
    if n_edges <= 0:
        return 0.0
    return float(
        (np.sum(out_deg.astype(np.float64) ** 2) * np.sum(in_deg.astype(np.float64) ** 2))
        / (n_edges**2)
    )


# ---------------------------------------------------------------------------
# A1 — R0 / R2 adjacency identity
# ---------------------------------------------------------------------------

def audit_A1(npz: Path, sizes: list[int], seeds: list[int]) -> dict:
    print("\n" + "=" * 78)
    print("A1  R0 / R2 ADJACENCY IDENTITY")
    print("=" * 78)
    out: dict = {"audit": "A1", "npz_path": str(npz), "per_size": {}}

    for n in sizes:
        sel = select_nodes(npz, n, SELECTION_SEED)
        R0 = load_reservoir_topology_from_npz(
            str(npz), normalization="n1_pre_l1", node_indices=sel.node_indices, seed=0
        )
        R2 = make_degree_rewired(R0, seed=seeds[0])

        A0, A2 = R0.matrix, R2.matrix
        b0, b2 = _binarize(A0), _binarize(A2)
        e0, e2 = _csc_edges(b0), _csc_edges(b2)
        inter = e0 & e2
        only0 = e0 - e2
        only2 = e2 - e0

        out_deg_0, in_deg_0 = _unweighted_degrees(A0)
        exp_shared = _expected_shared_edges_under_mixing(out_deg_0, in_deg_0, b0.nnz)
        overlap = len(inter) / len(e0) if e0 else 0.0

        # value-level difference (weights included)
        b0f, b2f = b0.astype(np.float64), b2.astype(np.float64)
        diff_nnz = int((b0f - b2f).nnz) if b0.nnz and b2.nnz else int(max(b0.nnz, b2.nnz))

        row = {
            "target_n": n,
            "N": int(A0.shape[0]),
            "M_R0": int(b0.nnz),
            "M_R2": int(b2.nnz),
            "same_object": A0 is A2,
            "sha16_R0_data": _sha16(A0.data),
            "sha16_R2_data": _sha16(A2.data),
            "sha16_R0_indptr": _sha16(A0.indptr),
            "sha16_R2_indptr": _sha16(A2.indptr),
            "identical_bytes": (
                A0.shape == A2.shape
                and _sha16(A0.data) == _sha16(A2.data)
                and _sha16(A0.indptr) == _sha16(A2.indptr)
            ),
            "nnz_binary_difference": diff_nnz,
            "n_shared_edges": len(inter),
            "n_edges_only_in_R0": len(only0),
            "n_edges_only_in_R2": len(only2),
            "edge_overlap_fraction": overlap,
            "expected_shared_edges_if_fully_mixed": exp_shared,
            "expected_overlap_if_fully_mixed": (
                exp_shared / len(e0) if e0 else 0.0
            ),
            "implied_successful_swaps": len(only0) / 2.0,
            "R0_weight_range": [float(A0.data.min()), float(A0.data.max())] if A0.nnz else None,
            "R2_weight_range": [float(A2.data.min()), float(A2.data.max())] if A2.nnz else None,
            "R2_weight_uniform": _weight_uniform(A2),
            "R0_weight_uniform": _weight_uniform(A0),
            "R2_destroys_weight_structure": bool(
                _weight_uniform(A2) and not _weight_uniform(A0)
            ),
            "weighted_degree_preserved": bool(
                np.allclose(
                    np.asarray(A0.sum(axis=1)).ravel(),
                    np.asarray(A2.sum(axis=1)).ravel(),
                )
            ),
            "sha16_R0_structure": _structure_sha(A0),
            "sha16_R2_structure": _structure_sha(A2),
        }
        out["per_size"][str(n)] = row

        print(f"\n  N={n}  N_row={row['N']}  M_R0={row['M_R0']}  M_R2={row['M_R2']}")
        print(f"    same python object ............ {row['same_object']}")
        print(f"    identical bytes ............... {row['identical_bytes']}")
        print(f"    |E_R0 ^ E_R2| ................. {diff_nnz}")
        print(f"    shared edges .................. {len(inter)}  "
              f"(fully-mixed expectation {exp_shared:.2f})")
        print(f"    EDGE OVERLAP |E0&E2|/|E0| ..... {overlap:.4f}  "
              f"(fully-mixed {row['expected_overlap_if_fully_mixed']:.4f})")
        print(f"    edges only in R0 / R2 ......... {len(only0)} / {len(only2)}")
        print(f"    implied successful swaps ...... {len(only0)/2.0:.0f}")
        print(f"    R0 uniform weights? {row['R0_weight_uniform']}   "
              f"R2 uniform weights? {row['R2_weight_uniform']}")

    (OUT_DIR / "A1_r0_r2_adjacency_identity.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A7 — R2 rewiring quality
# ---------------------------------------------------------------------------

def audit_A7(npz: Path, sizes: list[int], seeds: list[int]) -> dict:
    print("\n" + "=" * 78)
    print("A7  R2 REWIRING QUALITY (degree preservation, mixing, seeds, cost)")
    print("=" * 78)
    out: dict = {"audit": "A7", "npz_path": str(npz), "per_size": {}}

    for n in sizes:
        sel = select_nodes(npz, n, SELECTION_SEED)
        R0 = load_reservoir_topology_from_npz(
            str(npz), normalization="n1_pre_l1", node_indices=sel.node_indices, seed=0
        )
        A0 = R0.matrix
        b0 = _binarize(A0)
        out0_u, in0_u = _unweighted_degrees(A0)
        M = int(b0.nnz)

        t0 = time.time()
        R2 = make_degree_rewired(R0, seed=seeds[0])
        elapsed = time.time() - t0
        A2 = R2.matrix
        b2 = _binarize(A2)
        out2_u, in2_u = _unweighted_degrees(A2)

        e0, e2 = _csc_edges(b0), _csc_edges(b2)
        inter = len(e0 & e2)
        exp_shared = _expected_shared_edges_under_mixing(out0_u, in0_u, M)

        attempts = M * DOUBLE_EDGE_SWAP_ATTEMPTS
        row = {
            "target_n": n,
            "N": int(A0.shape[0]),
            "M": M,
            "n_swap_attempts_per_edge": DOUBLE_EDGE_SWAP_ATTEMPTS,
            "swap_attempts_total": int(attempts),
            "elapsed_seconds": elapsed,
            "seconds_per_attempt": elapsed / attempts if attempts else None,
            "n_edges_only_in_R0": len(e0 - e2),
            "implied_successful_swaps": len(e0 - e2) / 2.0,
            "implied_success_rate": (len(e0 - e2) / 2.0) / attempts if attempts else None,
            "edge_overlap_fraction": inter / M if M else 0.0,
            "expected_overlap_if_fully_mixed": exp_shared / M if M else 0.0,
            "unweighted_out_degree_preserved": bool(np.array_equal(out0_u, out2_u)),
            "unweighted_in_degree_preserved": bool(np.array_equal(in0_u, in2_u)),
            "unweighted_total_degree_preserved": bool(
                np.array_equal(out0_u + in0_u, out2_u + in2_u)
            ),
            "edge_count_preserved": int(b0.nnz) == int(b2.nnz),
            "weighted_degree_preserved": bool(
                np.allclose(
                    np.asarray(A0.sum(axis=1)).ravel(),
                    np.asarray(A2.sum(axis=1)).ravel(),
                )
            ),
            "self_loops_R0": _self_loops(A0),
            "self_loops_R2": _self_loops(A2),
            "multiplicity_R0": _multiplicity(A0),
            "multiplicity_R2": _multiplicity(A2),
            "R2_weight_uniform": _weight_uniform(A2),
            "R0_weight_uniform": _weight_uniform(A0),
            "R2_destroys_weight_structure": bool(
                _weight_uniform(A2) and not _weight_uniform(A0)
            ),
        }

        # 10 seeds -> are they 10 distinct graphs? The digest must be WIRING
        # based: degrees are preserved by construction, so any degree-only
        # digest would report "1 distinct graph" for 10 genuinely different ones.
        per_seed = {}
        hashes = []
        for s in seeds:
            Rs = make_degree_rewired(R0, seed=s)
            bs = _binarize(Rs.matrix)
            h = _structure_sha(bs)
            hashes.append(h)
            per_seed[str(s)] = {
                "structure_sha16": h,
                "overlap_with_R0": len(e0 & _csc_edges(bs)) / M if M else 0.0,
                "M": int(bs.nnz),
                "unweighted_out_degree_preserved": bool(
                    np.array_equal(out0_u, _unweighted_degrees(Rs.matrix)[0])
                ),
            }
        # pairwise wiring differences between seeds
        seed_edges = {}
        for s in seeds:
            seed_edges[s] = _csc_edges(_binarize(make_degree_rewired(R0, seed=s).matrix))
        pairwise = {}
        for i, si in enumerate(seeds):
            for sj in seeds[i + 1:]:
                inter_ss = len(seed_edges[si] & seed_edges[sj])
                pairwise[f"{si}_vs_{sj}"] = inter_ss / M if M else 0.0
        row["per_seed"] = per_seed
        row["n_distinct_graphs_over_seeds"] = len(set(hashes))
        row["n_seeds"] = len(seeds)
        row["seed_pairwise_edge_overlap_mean"] = (
            float(np.mean(list(pairwise.values()))) if pairwise else None
        )
        row["seed_pairwise_edge_overlap_min"] = (
            float(np.min(list(pairwise.values()))) if pairwise else None
        )
        row["seed_pairwise_edge_overlap_max"] = (
            float(np.max(list(pairwise.values()))) if pairwise else None
        )
        out["per_size"][str(n)] = row

        print(f"\n  N={n}  M={M}  attempts={attempts}  elapsed={elapsed:.2f}s "
              f"({row['seconds_per_attempt']:.3e} s/attempt)")
        print(f"    unweighted out-degree preserved ... {row['unweighted_out_degree_preserved']}")
        print(f"    unweighted in-degree preserved .... {row['unweighted_in_degree_preserved']}")
        print(f"    edge count preserved .............. {row['edge_count_preserved']}")
        print(f"    WEIGHTED degree preserved ......... {row['weighted_degree_preserved']}   "
              f"(R2 uniformises weights: {row['R2_destroys_weight_structure']})")
        print(f"    self-loops R0/R2 .................. {row['self_loops_R0']}/{row['self_loops_R2']}")
        print(f"    duplicate coordinates R0/R2 ....... "
              f"{row['multiplicity_R0']['n_duplicate_coordinates']}/"
              f"{row['multiplicity_R2']['n_duplicate_coordinates']}")
        print(f"    implied successful swaps .......... {row['implied_successful_swaps']:.0f} "
              f"of {attempts} attempts ({100*row['implied_success_rate']:.2f}%)")
        print(f"    edge overlap with R0 .............. {row['edge_overlap_fraction']:.4f}  "
              f"(fully-mixed {row['expected_overlap_if_fully_mixed']:.4f})")
        print(f"    distinct graphs over {len(seeds)} seeds ....... "
              f"{row['n_distinct_graphs_over_seeds']}")
        print(f"    seed-vs-seed edge overlap ......... mean "
              f"{row['seed_pairwise_edge_overlap_mean']:.4f} "
              f"[{row['seed_pairwise_edge_overlap_min']:.4f}, "
              f"{row['seed_pairwise_edge_overlap_max']:.4f}]")

    # feasibility extrapolation to the runner's default size (the full graph).
    # The per-attempt cost is dominated by the duplicate scan, which is O(M):
    # ``rest_rows = rows[unchanged_mask]`` copies ~M int64 per attempt. So the
    # honest cost model is seconds_per_attempt = c * M, with c fitted from the
    # small-N measurements, NOT the raw small-N per-attempt time.
    full = select_nodes(npz, 10**9, SELECTION_SEED)
    full_n = int(full.node_indices.size)
    payload = np.load(str(npz), allow_pickle=True)
    full_M = int(sp.csr_matrix(
        (
            payload["norm_n1_pre_l1_data"],
            payload["norm_n1_pre_l1_indices"],
            payload["norm_n1_pre_l1_indptr"],
        ),
        shape=tuple(int(x) for x in payload["norm_n1_pre_l1_shape"]),
    ).nnz)
    fits = [
        (v["M"], v["seconds_per_attempt"])
        for v in out["per_size"].values()
        if v["seconds_per_attempt"] and v["M"]
    ]
    c_per_edge = None
    if fits:
        # least-squares through the origin: cost per attempt ~ c * M
        num = sum(m * t for m, t in fits)
        den = sum(m * m for m, _ in fits)
        c_per_edge = num / den if den else None
    projected_s = None
    if c_per_edge:
        projected_s = full_M * DOUBLE_EDGE_SWAP_ATTEMPTS * (c_per_edge * full_M)
    out["full_graph_feasibility"] = {
        "full_graph_N": full_n,
        "full_graph_M": full_M,
        "runner_default_target_n_is_full_graph": True,
        "swap_attempts_total": int(full_M * DOUBLE_EDGE_SWAP_ATTEMPTS),
        "cost_model": "seconds_per_attempt = c * M (duplicate scan is O(M))",
        "c_seconds_per_edge_from_fit": c_per_edge,
        "measured_points": [{"M": m, "seconds_per_attempt": t} for m, t in fits],
        "projected_seconds_one_r2_build": projected_s,
        "projected_days_one_r2_build": projected_s / 86400 if projected_s else None,
        "projected_days_for_10_seeds": projected_s * 10 / 86400 if projected_s else None,
        "note": (
            "The runner builds one family per (window_length, seed), so a "
            "10-seed sweep needs 10 R2 builds. The runner's default "
            "(reservoir_size=None) is the full graph."
        ),
        "full_graph_memory": {
            "what": (
                "_ReservoirBase._drive materialises "
                "input_projection = x @ w_in.T of shape "
                "(n_windows, length, N) in float64"
            ),
            "per_window_length_16_bytes": int(16 * full_n * 8),
            "projected_gib_for_D2_train_windows_6435": float(
                6435 * 16 * full_n * 8 / 1024**3
            ),
            "projected_gib_for_D3_train_windows_20000": float(
                20000 * 16 * full_n * 8 / 1024**3
            ),
        },
    }
    print("\n  --- feasibility at the runner's DEFAULT size (full graph) ---")
    print(f"    full graph N={full_n}  M={full_M}")
    print(f"    swap attempts required ......... "
          f"{out['full_graph_feasibility']['swap_attempts_total']:,}")
    print(f"    fitted cost per attempt ........ "
          f"{c_per_edge * full_M:.4f} s  (c={c_per_edge:.3e} s/edge)")
    print(f"    projected ONE R2 build ......... "
          f"{out['full_graph_feasibility']['projected_days_one_r2_build']:.2f} days")
    print(f"    projected 10 seeds ............. "
          f"{out['full_graph_feasibility']['projected_days_for_10_seeds']:.2f} days")

    (OUT_DIR / "A7_r2_rewiring_quality.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A8 — subgraph degeneracy
# ---------------------------------------------------------------------------

def audit_A8(npz: Path, sizes: list[int], include_full: bool = True) -> dict:
    print("\n" + "=" * 78)
    print("A8  SUBGRAPH DEGENERACY (what the reservoir actually sees)")
    print("=" * 78)
    out: dict = {"audit": "A8", "npz_path": str(npz), "per_size": {}}

    payload = np.load(str(npz), allow_pickle=True)
    for key in ("adj_data", "adj_indices", "adj_indptr", "adj_shape"):
        if key not in payload.files:
            raise ValueError(f"NPZ lacks {key}")
    full = sp.csr_matrix(
        (
            payload["adj_data"],
            payload["adj_indices"],
            payload["adj_indptr"],
        ),
        shape=tuple(int(x) for x in payload["adj_shape"]),
    )
    full_out = np.asarray(full.sum(axis=1)).ravel()
    full_in = np.asarray(full.sum(axis=0)).ravel()
    full_n = int(full.shape[0])
    full_M = int(full.nnz)
    mean_out_unweighted = full_M / full_n

    out["full_graph"] = {
        "N": full_n,
        "M": full_M,
        "mean_unweighted_out_degree": mean_out_unweighted,
        "mean_weighted_out_degree": float(full_out.mean()),
        "note": (
            "meta.json reports in/out degree means of ~430; that is the WEIGHTED "
            "degree (sum of synapse counts). The unweighted mean out-degree is "
            "M/N and governs whether an induced subgraph stays connected."
        ),
    }
    print(f"\n  FULL GRAPH: N={full_n}  M={full_M}")
    print(f"    mean UNWEIGHTED out-degree = {mean_out_unweighted:.2f}")
    print(f"    mean WEIGHTED   out-degree = {full_out.mean():.2f}  "
          f"(<- the ~430 figure in meta.json)")

    sizes_to_run = list(sizes) + ([full_n] if include_full else [])
    for n in sizes_to_run:
        if n > full_n:
            continue
        if n == full_n:
            sub = full
            sel_root_ids = np.asarray(payload["node_ids"]).astype(str)
            identity = True
        else:
            sel = select_nodes(npz, n, SELECTION_SEED)
            idx = sel.node_indices
            sub = full[idx, :][:, idx].tocsr()
            sel_root_ids = sel.root_ids
            identity = sel.identity

        sub_bin = _binarize(sub)
        out_deg_u, in_deg_u = _unweighted_degrees(sub)
        out_w = np.asarray(sub.sum(axis=1)).ravel()
        n_nodes = int(sub.shape[0])
        isolated = int(np.sum((out_deg_u + in_deg_u) == 0))

        # in-block vs out-of-block fan-out for the SELECTED nodes
        if n < full_n:
            idx = select_nodes(npz, n, SELECTION_SEED).node_indices
            outdeg_full_selected = full_out[idx]
            outdeg_inblock = out_deg_u
            rng = np.random.default_rng(0)
            rand_idx = rng.choice(full_n, size=n, replace=False)
            rand_sub = full[rand_idx, :][:, rand_idx].tocsr()
            rand_deg_u, _ = _unweighted_degrees(rand_sub)
            rand_outdeg_full = full_out[rand_idx]
        else:
            outdeg_full_selected = full_out
            outdeg_inblock = out_deg_u
            rand_deg_u = out_deg_u
            rand_outdeg_full = full_out

        kept_fraction = (
            float(outdeg_inblock.sum() / outdeg_full_selected.sum())
            if outdeg_full_selected.sum() > 0 else None
        )
        row = {
            "target_n": n,
            "identity_selection": bool(identity),
            "N": n_nodes,
            "M_weighted_nnz": int(sub.nnz),
            "M_unweighted_edges": int(sub_bin.nnz),
            "density": sub_bin.nnz / (n_nodes * n_nodes) if n_nodes else 0.0,
            "n_isolated_nodes": isolated,
            "isolated_fraction": isolated / n_nodes if n_nodes else 0.0,
            "mean_unweighted_out_degree": float(out_deg_u.mean()),
            "median_unweighted_out_degree": float(np.median(out_deg_u)),
            "max_unweighted_out_degree": float(out_deg_u.max()),
            "mean_weighted_out_degree": float(out_w.mean()),
            "out_degree_kept_fraction_vs_full_graph": kept_fraction,
            "out_degree_kept_fraction_random_baseline": (
                float(rand_deg_u.sum() / rand_outdeg_full.sum())
                if rand_outdeg_full.sum() > 0 else None
            ),
            "structural_zero_rows": int(np.sum(out_deg_u == 0)),
            "structural_zero_cols": int(np.sum(in_deg_u == 0)),
            **_components(sub_bin),
        }
        out["per_size"][str(n)] = row

        label = "FULL" if identity else f"N={n}"
        print(f"\n  {label}: N={n_nodes}  M_edges={row['M_unweighted_edges']}  "
              f"density={row['density']:.6f}")
        print(f"    ISOLATED NODES ................. {isolated} / {n_nodes} "
              f"({100*row['isolated_fraction']:.1f}%)")
        print(f"    structural zero rows / cols .... {row['structural_zero_rows']} / "
              f"{row['structural_zero_cols']}")
        print(f"    mean out-degree (unweighted) ... {row['mean_unweighted_out_degree']:.3f}  "
              f"median={row['median_unweighted_out_degree']:.0f}  "
              f"max={row['max_unweighted_out_degree']:.0f}")
        if kept_fraction is not None:
            print(f"    out-degree KEPT by sub-selection  {kept_fraction:.4f}  "
                  f"(random-node baseline {row['out_degree_kept_fraction_random_baseline']:.4f})")
        print(f"    weak components ................ {row['n_weak_components']}  "
              f"largest={100*row['largest_weak_fraction']:.1f}% of nodes")
        print(f"    strong components .............. {row['n_strong_components']}  "
              f"largest={100*row['largest_strong_fraction']:.1f}%")

    (OUT_DIR / "A8_subgraph_degeneracy.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=["A1", "A7", "A8"],
                        choices=["A1", "A7", "A8"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--no-full-graph", action="store_true")
    args = parser.parse_args()

    npz = Path(args.npz_path) if args.npz_path else adjacency_path()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"M4-Audit reservoir integrity audit (graph-level)\n  NPZ: {npz}")

    if "A1" in args.only:
        audit_A1(npz, args.sizes, args.seeds)
    if "A7" in args.only:
        audit_A7(npz, args.sizes, args.seeds)
    if "A8" in args.only:
        audit_A8(npz, args.sizes, include_full=not args.no_full_graph)

    print(f"\nEvidence written under {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
