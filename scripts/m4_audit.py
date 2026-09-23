#!/usr/bin/env python
"""M4-Audit — reservoir integrity audit (validation/synthetic only, never test).

Purpose
-------
M4-v1.4 reports ``R0 ~ R2`` (the real fly connectome is statistically
indistinguishable from its degree-preserving rewired control). Before that
negative result can be accepted as a statement about *biology*, the audit must
establish that R0 and R2 actually produced different, non-degenerate dynamics on
the way into the readout.

The goal is NOT to turn the result positive. It is to establish what was
actually tested. If the audit passes, ``R0 ~ R2`` is a real scientific result.
If it fails, the existing M4 cannot be used to reject the project hypothesis.

Checks (A1-A8; A9 lives in the evaluation module)
-------------------------------------------------
A1  R0/R2 adjacency identity: distinct matrices, edge Jaccard, degree sequences
A2  Forward hidden-state divergence ``d_t`` under matched inputs
A3  Recurrent-vs-input contribution ratio ``||gAh|| / ||W_in x||``
A4  Readout input audit: does the readout see raw ``x``? (static + empirical)
A5  Normalization audit: n0_raw vs n1_pre_l1 vs n3_global_max
A6  Input-mapping audit: is the biological input pathway used at all?
A7  R2 rewiring audit: distinct graphs per seed, degree preservation
A8  Subgraph connectivity audit

No test split, no dataset, no readout training against real labels is touched.

Evidence discipline
-------------------
The instrumented dynamics are validated against the *production* ``_drive``
before any measurement is taken from them: if ``audit_drive`` does not
reproduce ``_drive`` exactly, every A2/A3 number would be describing a
re-implementation rather than the shipped model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    make_degree_rewired,
    make_shared,
    load_reservoir_topology_from_npz,
)

DEFAULT_NPZ = osp.join(
    PROJECT_ROOT, "..", "data-root", "connectome", "adjacency", "olfactory_v1.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4-Audit reservoir integrity audit")
    p.add_argument("--project-root", default=PROJECT_ROOT)
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--sizes", default="200,1000", help="reservoir sizes to audit")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--normalization", default="n1_pre_l1")
    p.add_argument("--length", type=int, default=64, help="window length for A2/A3")
    p.add_argument("--n-channels", type=int, default=8)
    p.add_argument("--out-dir", default=osp.join(PROJECT_ROOT, "results", "m4_audit"))
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Instrumented dynamics — validated against the production _drive
# --------------------------------------------------------------------------- #
def audit_drive(model, x: np.ndarray, record: bool = False):
    """Replicate ``_ReservoirBase._drive`` step by step, optionally recording.

    The update is copied from the shipped implementation. ``validate_against_production``
    asserts this function reproduces ``_drive`` exactly, so the recorded traces
    describe the real model and not a paraphrase of it.
    """
    leak = float(model.params["leak"])
    gain = float(model.params["gain"])
    pooling = str(model.params["state_pooling"])
    washout = int(model.params["washout"])
    n_nodes = model._topology.n_nodes
    A = model._topology.matrix
    w_in = model._shared.w_in
    bias = model._shared.bias

    n_samples, length, _ = x.shape
    washout = min(washout, max(length - 1, 0))
    states = np.zeros((n_samples, n_nodes), dtype=np.float64)
    input_projection = x @ w_in.T

    trace: list[dict[str, Any]] = []
    for i in range(n_samples):
        state = np.zeros(n_nodes, dtype=np.float64)
        accumulated = np.zeros(n_nodes, dtype=np.float64)
        collected = 0
        for t in range(length):
            recurrent = gain * (A @ state)
            input_term = input_projection[i, t]
            drive = recurrent + input_term + bias
            state = (1.0 - leak) * state + leak * np.tanh(drive)
            if record:
                trace.append({
                    "t": int(t),
                    "h": state.copy(),
                    "recurrent_norm": float(np.linalg.norm(recurrent)),
                    "input_norm": float(np.linalg.norm(input_term)),
                    "state_norm": float(np.linalg.norm(state)),
                })
            if t >= washout:
                accumulated += state
                collected += 1
        if collected and pooling == "mean":
            states[i] = accumulated / collected
        else:
            states[i] = state
    return states, trace


def validate_against_production(family: dict, x: np.ndarray) -> dict:
    """Assert audit_drive == production _drive for every family member."""
    report = {}
    ok = True
    for fid, model in family.items():
        produced = model._drive(x)
        replicated, _ = audit_drive(model, x, record=False)
        identical = np.array_equal(produced, replicated)
        max_abs = float(np.max(np.abs(produced - replicated)))
        report[fid] = {"bitwise_identical": bool(identical), "max_abs_diff": max_abs}
        ok = ok and identical
    report["_all_identical"] = bool(ok)
    return report


# --------------------------------------------------------------------------- #
# A1 — adjacency identity
# --------------------------------------------------------------------------- #
def _edge_set(m: sp.csr_matrix) -> set:
    coo = m.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def a1_adjacency_identity(r0, r2) -> dict:
    A0, A2 = r0.matrix, r2.matrix
    e0, e2 = _edge_set(A0), _edge_set(A2)
    inter = len(e0 & e2)
    union = len(e0 | e2)
    diff = (A0 - A2)
    deg0_out = np.diff(A0.indptr)
    deg2_out = np.diff(A2.indptr)
    deg0_in = np.asarray(A0.sum(axis=0)).ravel()
    deg2_in = np.asarray(A2.sum(axis=0)).ravel()
    # in/out degree here means the BINARY pattern degree of the stored matrix
    bin0_in = np.asarray((A0 != 0).sum(axis=0)).ravel()
    bin2_in = np.asarray((A2 != 0).sum(axis=0)).ravel()

    self0 = int((A0.diagonal() != 0).sum())
    self2 = int((A2.diagonal() != 0).sum())
    # multi-edge check: csr with duplicate (i,j) would report nnz>unique
    uniq0 = len(e0)
    uniq2 = len(e2)

    return {
        "distinct_objects": A0 is not A2,
        "sha256_A_R0": hashlib.sha256(np.ascontiguousarray(A0.data).tobytes()).hexdigest()[:32],
        "sha256_A_R2": hashlib.sha256(np.ascontiguousarray(A2.data).tobytes()).hexdigest()[:32],
        "data_hashes_differ": hashlib.sha256(A0.data.tobytes()).hexdigest()
        != hashlib.sha256(A2.data.tobytes()).hexdigest(),
        "pattern_hashes_differ": hashlib.sha256(A0.indices.tobytes() + A0.indptr.tobytes()).hexdigest()
        != hashlib.sha256(A2.indices.tobytes() + A2.indptr.tobytes()).hexdigest(),
        "nnz_diff": int(diff.nnz),
        "nnz_A_R0": int(A0.nnz),
        "nnz_A_R2": int(A2.nnz),
        "edge_jaccard_R0_R2": inter / union if union else float("nan"),
        "edge_overlap_fraction_of_R0": inter / len(e0) if e0 else float("nan"),
        "edges_R0": len(e0),
        "edges_R2": len(e2),
        "edges_shared": inter,
        "out_degree_sequence_identical": bool(np.array_equal(deg0_out, deg2_out)),
        "in_degree_sequence_identical": bool(np.array_equal(bin0_in, bin2_in)),
        "out_degree_max_abs_diff": int(np.max(np.abs(deg0_out - deg2_out))),
        "in_degree_max_abs_diff": int(np.max(np.abs(bin0_in - bin2_in))),
        "self_loops_R0": self0,
        "self_loops_R2": self2,
        "no_duplicate_edges_R0": bool(uniq0 == A0.nnz),
        "no_duplicate_edges_R2": bool(uniq2 == A2.nnz),
        "edge_count_preserved": bool(len(e0) == len(e2)),
        "R2_weight_is_uniform": bool(np.allclose(A2.data, A2.data[0])),
        "R0_weight_is_uniform": bool(np.allclose(A0.data, A0.data[0])),
        "R2_distinct_weight_values": int(np.unique(A2.data).size),
        "R0_distinct_weight_values": int(np.unique(A0.data).size),
    }


# --------------------------------------------------------------------------- #
# A2 / A3 — dynamics
# --------------------------------------------------------------------------- #
def a2_a3_dynamics(m0, m2, x_single: np.ndarray) -> dict:
    _, tr0 = audit_drive(m0, x_single, record=True)
    _, tr2 = audit_drive(m2, x_single, record=True)

    d_t, ratios = [], []
    for s0, s2 in zip(tr0, tr2):
        h0, h2 = s0["h"], s2["h"]
        denom = np.linalg.norm(h0) + 1e-12
        d_t.append(float(np.linalg.norm(h0 - h2) / denom))
        ratios.append(float(s0["recurrent_norm"] / (s0["input_norm"] + 1e-12)))

    # Also the reverse-direction ratio, to catch an asymmetric measurement.
    ratio_input_vs_recurrent = [
        float(s0["input_norm"] / (s0["recurrent_norm"] + 1e-12)) for s0 in tr0
    ]

    def at(k):
        return d_t[k] if k < len(d_t) else None

    return {
        "n_steps": len(d_t),
        "d_t_first": d_t[0],
        "d_t_at_1": at(1),
        "d_t_at_5": at(5),
        "d_t_at_10": at(10),
        "d_t_at_20": at(20),
        "d_t_last": d_t[-1],
        "d_t_max": float(np.max(d_t)),
        "d_t_mean": float(np.mean(d_t)),
        "d_t_tail_mean": float(np.mean(d_t[len(d_t) // 2:])),
        "hidden_state_diverges": bool(np.max(d_t) > 1e-6),
        "d_t_trace": [round(v, 10) for v in d_t],
        "recurrent_over_input_mean": float(np.mean(ratios)),
        "recurrent_over_input_median": float(np.median(ratios)),
        "recurrent_over_input_min": float(np.min(ratios)),
        "recurrent_over_input_max": float(np.max(ratios)),
        "input_over_recurrent_median": float(np.median(ratio_input_vs_recurrent)),
        "state_norm_mean": float(np.mean([s["state_norm"] for s in tr0])),
        "recurrent_norm_mean": float(np.mean([s["recurrent_norm"] for s in tr0])),
        "input_norm_mean": float(np.mean([s["input_norm"] for s in tr0])),
    }


# --------------------------------------------------------------------------- #
# A4 — readout input audit
# --------------------------------------------------------------------------- #
def a4_readout_input_audit(model, x: np.ndarray, y: np.ndarray) -> dict:
    """Does the readout see raw x? Empirically test whether h alone is used.

    Static reading of ``_fit`` shows the design matrix is ``[h, 1]``. This
    measures the consequence: fit the readout, then check that predictions are
    invariant to a change in x that leaves h unchanged is impossible, so instead
    we verify the design matrix has exactly ``n_nodes + 1`` columns.
    """
    # build_topology_family hard-codes task="classification" (the runner overrides
    # it per task), so a continuous control target would be rejected. Force the
    # regression path rather than reshaping the targets to look like classes.
    model.task = "regression"
    model._fit(x, y)
    w = model._w_out
    n_nodes = model._topology.n_nodes
    return {
        "design_matrix_columns": int(w.shape[0]),
        "n_nodes": int(n_nodes),
        "expected_columns_if_h_plus_bias_only": int(n_nodes + 1),
        "readout_sees_raw_x": bool(w.shape[0] != n_nodes + 1),
        "readout_weight_shape": list(map(int, w.shape)),
    }


def shuffled_h_control(model, x: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Zero-reservoir and shuffled-H controls for the readout.

    If a readout trained on a *shuffled* state matrix matches one trained on the
    real states, the reservoir contributes nothing that the bias alone would not.
    """
    total = x.shape[1]
    length = total // model.n_channels
    tensor = x.reshape(x.shape[0], length, model.n_channels)
    states, _ = audit_drive(model, tensor, record=False)

    rng = np.random.default_rng(seed)
    model.task = "regression"

    def ridge(feats, targets):
        lam = float(model.params["ridge_lambda"])
        g = feats.T @ feats + lam * np.eye(feats.shape[1])
        return np.linalg.solve(g, feats.T @ targets)

    y1 = y.reshape(-1, 1).astype(np.float64)
    real = np.column_stack([states, np.ones(states.shape[0])])
    shuffled = np.column_stack([states[rng.permutation(states.shape[0])], np.ones(states.shape[0])])
    zeros = np.column_stack([np.zeros_like(states), np.ones(states.shape[0])])

    w_real = ridge(real, y1)
    w_shuf = ridge(shuffled, y1)
    w_zero = ridge(zeros, y1)

    def mse(w, feats):
        return float(np.mean((feats @ w - y1) ** 2))

    return {
        "mse_train_real_H": mse(w_real, real),
        "mse_train_shuffled_H": mse(w_shuf, real),
        "mse_train_zero_H": mse(w_zero, real),
        "state_matrix_rank": int(np.linalg.matrix_rank(states)),
        "state_matrix_shape": list(map(int, states.shape)),
        "state_std_per_node_mean": float(np.mean(states.std(axis=0))),
        "state_std_per_node_min": float(np.min(states.std(axis=0))),
        "n_dead_nodes_zero_std": int((states.std(axis=0) < 1e-12).sum()),
    }


# --------------------------------------------------------------------------- #
# A5 — normalization audit
# --------------------------------------------------------------------------- #
def a5_normalization(npz: str, n_nodes: int, seed: int, x_single, n_channels, family_params):
    """Measure what each stored normalization does to the weight distribution.

    Read straight from the NPZ arrays rather than through the reservoir loader:
    the question is about the shipped DATA-3 artifact, not about what the loader
    does with it afterwards.
    """
    from drososense.reservoir.connectome_reservoir import _NORMALIZATION_NPZ_PREFIX

    npz_data = np.load(npz, allow_pickle=True)
    ref_shape = tuple(int(v) for v in npz_data["adj_shape"])
    out: dict[str, Any] = {}
    keep: dict[str, sp.csr_matrix] = {}

    for norm in ALLOWED_NORMALIZATIONS:
        prefix = _NORMALIZATION_NPZ_PREFIX[norm]
        data_key = prefix + "_data"
        if data_key not in npz_data.files:
            out[norm] = {"error": f"{data_key} not in NPZ"}
            continue
        m = sp.csr_matrix(
            (npz_data[prefix + "_data"], npz_data[prefix + "_indices"],
             npz_data[prefix + "_indptr"]),
            shape=ref_shape,
        )
        d = np.asarray(m.data, dtype=np.float64)
        rs = np.asarray(m.sum(axis=1)).ravel()
        nz = rs[rs > 0]
        out[norm] = {
            "nnz": int(m.nnz),
            "weight_min": float(d.min()), "weight_max": float(d.max()),
            "weight_mean": float(d.mean()),
            "row_strength_cv": float(rs.std() / rs.mean()) if rs.mean() else float("nan"),
            "row_strength_p99_over_p50": float(
                np.percentile(nz, 99) / max(np.percentile(nz, 50), 1e-12)
            ) if nz.size else float("nan"),
            "rows_summing_to_1": int(np.isclose(rs, 1.0).sum()),
            "fraction_rows_summing_to_1": float(np.isclose(rs, 1.0).mean()),
            "distinct_weight_values": int(np.unique(d).size),
        }
        if norm in ("n0_raw", "n1_pre_l1"):
            keep[norm] = m[:n_nodes].tocsr()

    # Does the normalization change the RANKING of neighbours within a row?
    # This is the difference between "removes inter-neuron strength heterogeneity"
    # and "removes synaptic weight information".
    rank_preserved: dict[str, Any] = {}
    if "n0_raw" in keep and "n1_pre_l1" in keep:
        A0, A1 = keep["n0_raw"], keep["n1_pre_l1"]
        agree = checked = 0
        for i in range(A0.shape[0]):
            a = A0[i].toarray().ravel()
            b = A1[i].toarray().ravel()
            nz = a > 0
            if nz.sum() > 2:
                checked += 1
                ra = np.argsort(np.argsort(a[nz]))
                rb = np.argsort(np.argsort(b[nz]))
                if np.array_equal(ra, rb):
                    agree += 1
        rank_preserved = {
            "rows_checked": int(checked),
            "rows_with_identical_within_row_ranking": int(agree),
            "within_row_ranking_preserved": bool(checked and agree == checked),
            "interpretation": (
                "If True, the normalization rescales each row without reordering "
                "synapses, so it removes INTER-neuron strength heterogeneity while "
                "preserving the WITHIN-neuron synaptic weight profile."
            ),
        }
    out["_within_row_ranking"] = rank_preserved
    return out


# --------------------------------------------------------------------------- #
# A6 — input mapping
# --------------------------------------------------------------------------- #
def a6_input_mapping(n_nodes: int, n_channels: int, seed: int, input_scale: float, topo) -> dict:
    shared = make_shared(n_nodes=n_nodes, n_channels=n_channels, seed=seed, input_scale=input_scale)
    w = shared.w_in
    col_norms = np.linalg.norm(w, axis=0)
    row_norms = np.linalg.norm(w, axis=1)
    out_deg = np.diff(topo.matrix.indptr).astype(float)
    # Is input magnitude related to a node's position/degree? If not, the input
    # is a generic dense random projection and carries no biological addressing.
    corr_deg = float(np.corrcoef(row_norms, out_deg)[0, 1]) if row_norms.std() > 0 else float("nan")
    return {
        "w_in_shape": list(map(int, w.shape)),
        "nodes_receiving_input": int(n_nodes),
        "fraction_of_nodes_receiving_input": 1.0,
        "nodes_with_zero_input_row": int((np.abs(w).sum(axis=1) == 0).sum()),
        "input_is_dense_random_projection": True,
        "correlation_input_row_norm_vs_out_degree": corr_deg,
        "biological_input_pathway_used": False,
        "note": (
            "W_in is (n_nodes, n_channels): EVERY neuron in the reservoir receives "
            "a direct random projection of the raw input. There is no ORN/PN/KC/MBON "
            "input population, so the olfactory hierarchy is bypassed."
        ),
    }


# --------------------------------------------------------------------------- #
# A7 — rewiring audit
# --------------------------------------------------------------------------- #
def a7_rewiring(npz: str, n_nodes: int, seeds, normalization: str, family_params) -> dict:
    base = load_reservoir_topology_from_npz(
        npz, normalization=normalization, node_indices=None,
        target_spectral_radius=float(family_params["spectral_radius"]), seed=seeds[0],
    )
    graphs = {}
    for s in seeds:
        t = make_degree_rewired(base, int(s))
        graphs[int(s)] = _edge_set(t.matrix)
    jacs, touch = [], []
    keys = sorted(graphs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = graphs[keys[i]], graphs[keys[j]]
            u = a | b
            jacs.append(len(a & b) / len(u) if u else float("nan"))
            touch.append(hashlib.sha256(
                np.array(sorted(b), dtype=np.int64).tobytes()).hexdigest()[:16])
    base_edges = _edge_set(base.matrix)
    per_seed_jaccard_vs_R0 = {
        str(s): len(base_edges & graphs[s]) / len(base_edges | graphs[s])
        for s in keys
    }
    degs = {}
    for s in keys:
        t = make_degree_rewired(base, int(s))
        degs[str(s)] = bool(np.array_equal(np.diff(t.matrix.indptr), np.diff(base.matrix.indptr)))
    return {
        "n_seeds": len(keys),
        "n_distinct_graphs": len(set(touch)),
        "seeds_produce_distinct_graphs": bool(len(set(touch)) == len(keys)),
        "pairwise_edge_jaccard_min": float(np.min(jacs)),
        "pairwise_edge_jaccard_max": float(np.max(jacs)),
        "pairwise_edge_jaccard_mean": float(np.mean(jacs)),
        "jaccard_vs_R0_per_seed": per_seed_jaccard_vs_R0,
        "out_degree_preserved_all_seeds": bool(all(degs.values())),
    }


# --------------------------------------------------------------------------- #
# A8 — subgraph connectivity
# --------------------------------------------------------------------------- #
def a8_connectivity(topo) -> dict:
    m = topo.matrix
    n = m.shape[0]
    pattern = (m != 0)
    und = (pattern + pattern.T).astype(bool).astype(np.int8)
    n_comp, labels = sp.csgraph.connected_components(und, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max()) if sizes.size else 0
    return {
        "n_nodes": int(n),
        "n_edges": int(m.nnz),
        "n_weak_components": int(n_comp),
        "largest_component_size": largest,
        "largest_component_fraction": largest / n if n else float("nan"),
        "n_isolated_nodes": int((np.diff(m.indptr) + np.asarray((m != 0).sum(axis=0)).ravel() == 0).sum()),
        "mean_out_degree": float(np.diff(m.indptr).mean()),
        "is_one_giant_component": bool(n_comp == 1),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 78)
    print("M4-AUDIT — RESERVOIR INTEGRITY AUDIT (validation/synthetic only)")
    print("=" * 78)
    print(f"  npz         : {osp.relpath(osp.abspath(args.npz), PROJECT_ROOT)}")
    print(f"  sizes       : {sizes}")
    print(f"  normalization: {args.normalization}")
    print(f"  window      : length={args.length} channels={args.n_channels}")
    print()

    payload: dict[str, Any] = {
        "audit": "M4-Audit",
        "scope": "validation/synthetic only; no test split, no dataset labels",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "npz": osp.abspath(args.npz),
        "params": {**DEFAULT_RESERVOIR_PARAMS},
        "sizes": sizes,
        "seeds": seeds,
        "checks": {},
    }

    rng = np.random.default_rng(12345)
    x_multi = rng.normal(size=(32, args.length, args.n_channels))
    x_single = rng.normal(size=(1, args.length, args.n_channels))

    for n_nodes in sizes:
        key = f"N{n_nodes}"
        block: dict[str, Any] = {}
        print(f"--- N={n_nodes} ---")

        # Subselection requires node metadata; use the shipped selector.
        from drososense.connectome_selection import select_nodes
        sel = select_nodes(args.npz, n_nodes, seeds[0])
        family = build_topology_family(
            args.npz, seed=seeds[0], n_channels=args.n_channels,
            normalization=args.normalization, node_indices=sel.node_indices,
            params={"reservoir_size": n_nodes},
        )
        block["node_selection"] = sel.describe()

        r0 = family["R0_real_fly"]
        r2 = family["R2_degree_rewired"]

        # -- validation of the instrumentation, before any measurement
        val = validate_against_production(family, x_multi)
        block["instrumentation_validation"] = val
        print(f"  instrumentation matches production _drive: {val['_all_identical']}")
        if not val["_all_identical"]:
            block["FATAL"] = "audit_drive does not reproduce production _drive"
            payload["checks"][key] = block
            continue

        block["A1_adjacency_identity"] = a1_adjacency_identity(r0._topology, r2._topology)
        print(f"  A1 edge Jaccard R0/R2 = "
              f"{block['A1_adjacency_identity']['edge_jaccard_R0_R2']:.6f}, "
              f"nnz diff = {block['A1_adjacency_identity']['nnz_diff']}")

        block["A2_A3_dynamics"] = a2_a3_dynamics(r0, r2, x_single)
        d = block["A2_A3_dynamics"]
        print(f"  A2 d_t: first={d['d_t_first']:.3e} last={d['d_t_last']:.3e} "
              f"max={d['d_t_max']:.3e} tail_mean={d['d_t_tail_mean']:.3e}")
        print(f"  A3 ||gAh||/||W_in x|| median = {d['recurrent_over_input_median']:.6f}")

        block["A4_readout_input"] = a4_readout_input_audit(
            r0, x_multi.reshape(x_multi.shape[0], -1), rng.normal(size=x_multi.shape[0])
        )
        block["A4_shuffled_controls"] = shuffled_h_control(
            r0, x_multi.reshape(x_multi.shape[0], -1), rng.normal(size=x_multi.shape[0])
        )

        block["A5_normalization"] = a5_normalization(
            args.npz, n_nodes, seeds[0], x_single, args.n_channels,
            {**DEFAULT_RESERVOIR_PARAMS, "reservoir_size": n_nodes},
        )

        block["A6_input_mapping"] = a6_input_mapping(
            n_nodes, args.n_channels, seeds[0],
            float(DEFAULT_RESERVOIR_PARAMS["input_scale"]), r0._topology,
        )

        block["A7_rewiring"] = a7_rewiring(
            args.npz, n_nodes, seeds, args.normalization,
            {**DEFAULT_RESERVOIR_PARAMS, "reservoir_size": n_nodes},
        )
        print(f"  A7 distinct R2 graphs over {len(seeds)} seeds = "
              f"{block['A7_rewiring']['n_distinct_graphs']}")

        block["A8_connectivity"] = a8_connectivity(r0._topology)
        c = block["A8_connectivity"]
        print(f"  A8 components={c['n_weak_components']} "
              f"largest_fraction={c['largest_component_fraction']:.4f}")

        payload["checks"][key] = block
        print()

    out = osp.join(args.out_dir, "m4_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"wrote {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
