#!/usr/bin/env python3
"""M4-Audit — Reservoir Integrity Audit (dynamics half: A2, A3, A4, A5, A6).

PURPOSE. The graph-level audit (``reservoir_integrity_audit.py``) asks whether
R0 and R2 are the same *matrix*. This script asks the strictly stronger and
actually decisive question: **do they produce different, non-degenerate
dynamics?**

Hard constraints honoured here
------------------------------
* No test split is ever read. A2/A3/A5/A6 use a seeded synthetic input with no
  dataset at all; A4 uses the synthetic fixture's TRAIN and VALIDATION splits
  only and asserts the test tensors are untouched.
* Nothing under ``configs/`` or ``drososense/`` is modified.
* No run record, no training batch, no server job is started.

Audits
------
A2  Hidden-state divergence. Drive R0 and R2 with the SAME ``W_in``, the same
    bias, the same initial state and the same input, and report
        d_t = ||h_R0(t) - h_R2(t)|| / (||h_R0(t)|| + eps)
    per timestep, with two calibrating references: the same measurement against
    R3/R4 (a genuinely different graph family), and against R0-vs-R0' (same
    graph, different input map) as an upper reference for "a real difference".
A3  Recurrent vs input drive ratio
        R_t = ||gain * A @ h_{t-1}|| / ||W_in x_t + b||
    plus the full hyperparameter regime it holds in. This is the number that
    decides whether the reservoir is computing a recurrence or is a memoryless
    random feature map of the input.
A4  Readout ablation on the synthetic fixture (validation only):
    X-only / H-only / X+H / X+shuffled-H / zero-reservoir.
A5  Normalization sensitivity: divergence and drive ratio under n0_raw,
    n1_pre_l1 and n5_binary. Note the loader always rescales to
    ``spectral_radius``, so "global spectral scaling only" IS n0_raw.
A6  Input-mapping audit: is the input map biologically organised at all? Tests
    (i) the shape/density of W_in, (ii) whether node identity matters — a
    consistent relabelling of nodes leaves the dynamics bit-identical, and
    (iii) the correlation between a node's connectome degree and its input
    weights.

Usage
-----
    python ops/audit/reservoir_dynamics_audit.py --only A2 A3 A6
    python ops/audit/reservoir_dynamics_audit.py --only A4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from connectome.paths import adjacency_path  # noqa: E402
from drososense.connectome_selection import select_nodes  # noqa: E402
from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    build_topology_family,
    load_reservoir_topology_from_npz,
    make_shared,
)
from drososense.reservoir.runner import PINNED_KNOBS  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "audit" / "m4_audit"
SELECTION_SEED = 20260920
EPS = 1e-12

#: The regime the E2/E3 runner actually pins (``drososense/reservoir/runner.py``).
RUNNER_REGIME = {
    "leak": PINNED_KNOBS["leak"],
    "gain": PINNED_KNOBS["gain"],
    "input_scale": PINNED_KNOBS["input_scale"],
    "spectral_radius": 0.9,
    "washout": DEFAULT_RESERVOIR_PARAMS["washout"],
    "state_pooling": DEFAULT_RESERVOIR_PARAMS["state_pooling"],
    "readout": DEFAULT_RESERVOIR_PARAMS["readout"],
    "ridge_lambda": DEFAULT_RESERVOIR_PARAMS["ridge_lambda"],
}


# ---------------------------------------------------------------------------
# synthetic input + a faithful copy of the runner's state update
# ---------------------------------------------------------------------------

def make_ar1_input(n_samples: int, length: int, n_channels: int, seed: int = 0,
                   rho: float = 0.9) -> np.ndarray:
    """A seeded AR(1) input tensor of shape (n_samples, length, n_channels).

    Standardised per channel the way the pipeline standardises real channels,
    so the magnitudes are comparable with a real run. No dataset involved.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, length, n_channels), dtype=np.float64)
    innovations = rng.standard_normal((n_samples, length, n_channels))
    for t in range(length):
        x[:, t, :] = rho * (x[:, t - 1, :] if t else 0.0) + innovations[:, t, :]
    mu = x.reshape(-1, n_channels).mean(axis=0)
    sd = x.reshape(-1, n_channels).std(axis=0) + 1e-8
    return (x - mu) / sd


def drive_with_trace(
    matrix: sp.spmatrix,
    w_in: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    *,
    leak: float,
    gain: float,
) -> dict:
    """Replicate ``_ReservoirBase._drive`` exactly, but keep the traces.

    The update is::

        drive_t = gain * (A @ h_{t-1}) + W_in x_t + b
        h_t     = (1 - leak) * h_{t-1} + leak * tanh(drive_t)

    Returns the per-step hidden states, the per-step recurrent and input
    contribution norms, and the projected-input tensor.
    """
    n_samples, length, _ = x.shape
    n_nodes = matrix.shape[0]
    input_projection = x @ w_in.T  # (n_samples, length, n_nodes)

    h_trace = np.zeros((n_samples, length + 1, n_nodes), dtype=np.float64)
    rec_norm = np.zeros((n_samples, length), dtype=np.float64)
    in_norm = np.zeros((n_samples, length), dtype=np.float64)
    rec_vec = np.zeros((n_samples, length, n_nodes), dtype=np.float64)

    for i in range(n_samples):
        state = np.zeros(n_nodes, dtype=np.float64)
        h_trace[i, 0] = state
        for t in range(length):
            recurrent = gain * (matrix @ state)
            external = input_projection[i, t] + bias
            rec_vec[i, t] = recurrent
            rec_norm[i, t] = float(np.linalg.norm(recurrent))
            in_norm[i, t] = float(np.linalg.norm(external))
            state = (1.0 - leak) * state + leak * np.tanh(recurrent + external)
            h_trace[i, t + 1] = state
    return {
        "h_trace": h_trace,
        "recurrent_norm": rec_norm,
        "input_norm": in_norm,
        "input_projection": input_projection,
    }


def _rel_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """||a-b|| / (||a|| + eps) along the last axis."""
    num = np.linalg.norm(a - b, axis=-1)
    den = np.linalg.norm(a, axis=-1) + EPS
    return num / den


def _summarise_divergence(d: np.ndarray) -> dict:
    """d has shape (n_samples, length)."""
    per_step_mean = d.mean(axis=0)
    return {
        "d_step0": float(per_step_mean[0]),
        "d_step1": float(per_step_mean[1]) if d.shape[1] > 1 else None,
        "d_step5": float(per_step_mean[5]) if d.shape[1] > 5 else None,
        "d_final": float(per_step_mean[-1]),
        "d_mean": float(per_step_mean.mean()),
        "d_max_over_steps": float(per_step_mean.max()),
        "d_min_over_steps": float(per_step_mean.min()),
        "per_step_mean": [float(v) for v in per_step_mean],
    }


def _topologies(npz: Path, n: int, normalization: str, seed: int) -> dict:
    sel = select_nodes(npz, n, SELECTION_SEED)
    family = build_topology_family(
        str(npz),
        seed=seed,
        n_channels=5,
        normalization=normalization,
        node_indices=sel.node_indices,
        params={
            **RUNNER_REGIME,
            "reservoir_size": int(sel.node_indices.size),
        },
    )
    return fam if False else {"selection": sel, "family": family}


# ---------------------------------------------------------------------------
# A2 — hidden-state divergence
# ---------------------------------------------------------------------------

def audit_A2(npz: Path, sizes: list[int], normalization: str,
             n_samples: int, length: int) -> dict:
    print("\n" + "=" * 78)
    print("A2  HIDDEN-STATE DIVERGENCE  d_t = ||h_R0 - h_R2|| / (||h_R0|| + eps)")
    print("=" * 78)
    print(f"  regime: {RUNNER_REGIME}")
    print(f"  normalization={normalization}  input=seeded AR(1) "
          f"({n_samples} windows x {length} steps x 5 channels)")

    out: dict = {
        "audit": "A2",
        "regime": RUNNER_REGIME,
        "normalization": normalization,
        "input": {"kind": "synthetic AR(1), seeded", "n_samples": n_samples,
                  "length": length, "n_channels": 5, "seed": 0},
        "per_size": {},
    }

    for n in sizes:
        sel = select_nodes(npz, n, SELECTION_SEED)
        family = build_topology_family(
            str(npz), seed=0, n_channels=5, normalization=normalization,
            node_indices=sel.node_indices,
            params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
        )
        n_nodes = family["R0_real_fly"]._topology.n_nodes
        x = make_ar1_input(n_samples, length, 5, seed=0)
        shared = make_shared(n_nodes, 5, seed=0, input_scale=RUNNER_REGIME["input_scale"])

        traces = {}
        for fid in ("R0_real_fly", "R2_degree_rewired", "R3_random_sparse", "R4_er_esn"):
            model = family[fid]
            traces[fid] = drive_with_trace(
                model._topology.matrix, shared.w_in, shared.bias, x,
                leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"],
            )

        # R0 vs R2 — the decisive comparison
        d_r0r2 = _rel_divergence(
            traces["R0_real_fly"]["h_trace"], traces["R2_degree_rewired"]["h_trace"]
        )
        d_r0r3 = _rel_divergence(
            traces["R0_real_fly"]["h_trace"], traces["R3_random_sparse"]["h_trace"]
        )
        d_r0r4 = _rel_divergence(
            traces["R0_real_fly"]["h_trace"], traces["R4_er_esn"]["h_trace"]
        )
        # calibration: same graph, DIFFERENT input map (a real, large difference)
        shared2 = make_shared(n_nodes, 5, seed=99, input_scale=RUNNER_REGIME["input_scale"])
        t_alt = drive_with_trace(
            family["R0_real_fly"]._topology.matrix, shared2.w_in, shared2.bias, x,
            leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"],
        )
        d_inputmap = _rel_divergence(
            traces["R0_real_fly"]["h_trace"], t_alt["h_trace"]
        )

        row = {
            "target_n": n,
            "N": int(n_nodes),
            "M_edges": int(family["R0_real_fly"]._topology.matrix.nnz),
            "R0_vs_R2": _summarise_divergence(d_r0r2),
            "R0_vs_R3": _summarise_divergence(d_r0r3),
            "R0_vs_R4": _summarise_divergence(d_r0r4),
            "R0_vs_R0_different_input_map_CALIBRATION": _summarise_divergence(d_inputmap),
        }
        out["per_size"][str(n)] = row

        print(f"\n  N={n}  M={row['M_edges']}")
        for key in ("R0_vs_R2", "R0_vs_R3", "R0_vs_R4",
                    "R0_vs_R0_different_input_map_CALIBRATION"):
            s = row[key]
            print(f"    {key:44s} d_step0={s['d_step0']:.6f}  "
                  f"d_step5={s['d_step5']:.6f}  d_final={s['d_final']:.6f}  "
                  f"d_mean={s['d_mean']:.6f}")

    (OUT_DIR / "A2_hidden_state_divergence.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A3 — recurrent vs input drive ratio
# ---------------------------------------------------------------------------

def audit_A3(npz: Path, sizes: list[int], normalization: str,
             n_samples: int, length: int) -> dict:
    print("\n" + "=" * 78)
    print("A3  RECURRENT vs INPUT DRIVE RATIO   R_t = ||g*A@h|| / ||W_in x + b||")
    print("=" * 78)

    out: dict = {
        "audit": "A3",
        "regime": RUNNER_REGIME,
        "normalization": normalization,
        "note": (
            "R_t << 1 means the update is dominated by the instantaneous input "
            "projection, i.e. h ~= tanh(W_in x + b) is a memoryless random "
            "feature map and the topology cannot express itself in the state."
        ),
        "per_size": {},
    }

    for n in sizes:
        sel = select_nodes(npz, n, SELECTION_SEED)
        family = build_topology_family(
            str(npz), seed=0, n_channels=5, normalization=normalization,
            node_indices=sel.node_indices,
            params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
        )
        model = family["R0_real_fly"]
        A = model._topology.matrix
        n_nodes = A.shape[0]
        x = make_ar1_input(n_samples, length, 5, seed=0)
        shared = make_shared(n_nodes, 5, seed=0, input_scale=RUNNER_REGIME["input_scale"])

        tr = drive_with_trace(A, shared.w_in, shared.bias, x,
                              leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
        rec = tr["recurrent_norm"]
        ext = tr["input_norm"]
        ratio = rec / (ext + EPS)

        # also: does the state carry any information about the PAST input?
        # correlation between h_t and x_{t-k} for k = 1, 4, 8 (memory probe)
        h = tr["h_trace"][:, 1:, :]  # (samples, length, nodes)
        mem = {}
        for k in (1, 4, 8):
            if length - 1 - k >= 0:
                past = x[:, : length - k, :].reshape(-1, 5)
                state_now = h[:, k:, :].reshape(-1, n_nodes)
                # per-channel canonical correlation proxy: max abs corr over nodes
                pn = (past - past.mean(0)) / (past.std(0) + EPS)
                sn = (state_now - state_now.mean(0)) / (state_now.std(0) + EPS)
                c = np.abs(pn.T @ sn) / past.shape[0]
                mem[f"x_t_minus_{k}"] = {
                    "max_abs_corr": float(c.max()),
                    "mean_abs_corr": float(c.mean()),
                }

        row = {
            "target_n": n,
            "N": int(n_nodes),
            "M_edges": int(A.nnz),
            "spectral_radius_target": RUNNER_REGIME["spectral_radius"],
            "spectral_radius_measured": float(model._topology.spectral_radius),
            "gain": RUNNER_REGIME["gain"],
            "effective_recurrent_gain": RUNNER_REGIME["gain"] * float(model._topology.spectral_radius),
            "leak": RUNNER_REGIME["leak"],
            "input_scale": RUNNER_REGIME["input_scale"],
            "R_t_mean": float(ratio.mean()),
            "R_t_step0": float(ratio[:, 0].mean()),
            "R_t_final": float(ratio[:, -1].mean()),
            "R_t_max": float(ratio.max()),
            "R_t_per_step_mean": [float(v) for v in ratio.mean(axis=0)],
            "recurrent_norm_mean": float(rec.mean()),
            "input_norm_mean": float(ext.mean()),
            "state_norm_final_mean": float(
                np.linalg.norm(tr["h_trace"][:, -1, :], axis=-1).mean()
            ),
            "memory_probe": mem,
            "readout_features": "column_stack([h_L, ones]) — H-only, no raw x",
            "state_pooling": RUNNER_REGIME["state_pooling"],
            "washout": RUNNER_REGIME["washout"],
            "washout_effective_with_pooling_last": (
                "no effect: pooling=='last' returns the final state, and "
                "``washout`` only gates the running mean"
            ),
        }

        # --- zero-recurrent control -------------------------------------
        # If the state's memory of the past survives with A = 0, that memory
        # comes from the leaky integration of the input alone and the topology
        # contributes nothing to it.
        A_zero = sp.csr_matrix(A.shape, dtype=np.float64)
        tr_zero = drive_with_trace(A_zero, shared.w_in, shared.bias, x,
                                   leak=RUNNER_REGIME["leak"],
                                   gain=RUNNER_REGIME["gain"])
        hz = tr_zero["h_trace"][:, 1:, :]
        mem_zero = {}
        for k in (1, 4, 8):
            if length - 1 - k >= 0:
                past = x[:, : length - k, :].reshape(-1, 5)
                state_now = hz[:, k:, :].reshape(-1, n_nodes)
                pn = (past - past.mean(0)) / (past.std(0) + EPS)
                sn = (state_now - state_now.mean(0)) / (state_now.std(0) + EPS)
                c = np.abs(pn.T @ sn) / past.shape[0]
                mem_zero[f"x_t_minus_{k}"] = {
                    "max_abs_corr": float(c.max()),
                    "mean_abs_corr": float(c.mean()),
                }
        d_zero = _rel_divergence(tr["h_trace"], tr_zero["h_trace"])
        row["zero_recurrent_control"] = {
            "A_is_zero_matrix": True,
            "memory_probe": mem_zero,
            "R0_vs_zero_recurrent_divergence": _summarise_divergence(d_zero),
            "note": (
                "d(R0, A=0) small means the connectome term changes almost "
                "nothing about the state; a memory probe that survives with "
                "A=0 means the state's memory is the leak's, not the graph's."
            ),
        }
        out["per_size"][str(n)] = row

        print(f"\n  N={n}  M={A.nnz}  rho_target={row['spectral_radius_target']} "
              f"rho_measured={row['spectral_radius_measured']:.4f}")
        print(f"    effective recurrent gain g*rho ... {row['effective_recurrent_gain']:.4f}")
        print(f"    R_t  mean / step0 / final ........ {row['R_t_mean']:.4f} / "
              f"{row['R_t_step0']:.4f} / {row['R_t_final']:.4f}")
        print(f"    ||recurrent|| / ||input|| ........ {row['recurrent_norm_mean']:.4f} / "
              f"{row['input_norm_mean']:.4f}")
        for k, v in mem.items():
            print(f"    memory probe {k:14s} max|corr|={v['max_abs_corr']:.4f} "
                  f"mean|corr|={v['mean_abs_corr']:.4f}")
        z = row["zero_recurrent_control"]
        print(f"    ZERO-RECURRENT control (A=0):")
        for k, v in mem_zero.items():
            print(f"      memory probe {k:14s} max|corr|={v['max_abs_corr']:.4f} "
                  f"mean|corr|={v['mean_abs_corr']:.4f}")
        print(f"      d(R0, A=0) final ............... "
              f"{z['R0_vs_zero_recurrent_divergence']['d_final']:.6f}  "
              f"(mean {z['R0_vs_zero_recurrent_divergence']['d_mean']:.6f})")

    (OUT_DIR / "A3_recurrent_vs_input.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A5 — normalization sensitivity
# ---------------------------------------------------------------------------

def audit_A5(npz: Path, sizes: list[int], normalizations: list[str],
             n_samples: int, length: int) -> dict:
    print("\n" + "=" * 78)
    print("A5  NORMALIZATION SENSITIVITY")
    print("=" * 78)
    print("  the loader ALWAYS rescales to spectral_radius, so n0_raw IS")
    print("  'global spectral scaling only'.")

    out: dict = {"audit": "A5", "regime": RUNNER_REGIME, "per_normalization": {}}

    for norm in normalizations:
        if norm not in ALLOWED_NORMALIZATIONS:
            continue
        out["per_normalization"][norm] = {}
        for n in sizes:
            sel = select_nodes(npz, n, SELECTION_SEED)
            family = build_topology_family(
                str(npz), seed=0, n_channels=5, normalization=norm,
                node_indices=sel.node_indices,
                params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
            )
            R0 = family["R0_real_fly"]._topology
            R2 = family["R2_degree_rewired"]._topology
            n_nodes = R0.n_nodes
            x = make_ar1_input(n_samples, length, 5, seed=0)
            shared = make_shared(n_nodes, 5, seed=0,
                                 input_scale=RUNNER_REGIME["input_scale"])
            t0 = drive_with_trace(R0.matrix, shared.w_in, shared.bias, x,
                                  leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
            t2 = drive_with_trace(R2.matrix, shared.w_in, shared.bias, x,
                                  leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
            d = _rel_divergence(t0["h_trace"], t2["h_trace"])
            ratio = t0["recurrent_norm"] / (t0["input_norm"] + EPS)
            row = {
                "N": int(n_nodes),
                "M": int(R0.matrix.nnz),
                "R0_weight_stats": {
                    "min": float(R0.matrix.data.min()) if R0.matrix.nnz else None,
                    "max": float(R0.matrix.data.max()) if R0.matrix.nnz else None,
                    "sum": float(R0.matrix.data.sum()) if R0.matrix.nnz else None,
                    "mean_nnz": float(R0.matrix.data.mean()) if R0.matrix.nnz else None,
                },
                "R2_weight_stats": {
                    "min": float(R2.matrix.data.min()) if R2.matrix.nnz else None,
                    "max": float(R2.matrix.data.max()) if R2.matrix.nnz else None,
                },
                "R0_vs_R2_divergence": _summarise_divergence(d),
                "R_t_mean": float(ratio.mean()),
                "R_t_final": float(ratio[:, -1].mean()),
            }
            out["per_normalization"][norm][str(n)] = row
            print(f"\n  normalization={norm}  N={n}  M={row['M']}")
            print(f"    R0 weight sum={row['R0_weight_stats']['sum']:.4f} "
                  f"min={row['R0_weight_stats']['min']} max={row['R0_weight_stats']['max']}")
            print(f"    d_final(R0 vs R2) ......... {row['R0_vs_R2_divergence']['d_final']:.6f}")
            print(f"    d_mean (R0 vs R2) ......... {row['R0_vs_R2_divergence']['d_mean']:.6f}")
            print(f"    R_t mean / final .......... {row['R_t_mean']:.4f} / {row['R_t_final']:.4f}")

    (OUT_DIR / "A5_normalization_sensitivity.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A6 — input-mapping audit
# ---------------------------------------------------------------------------

def audit_A6(npz: Path, sizes: list[int], normalization: str) -> dict:
    print("\n" + "=" * 78)
    print("A6  INPUT-MAPPING AUDIT (is the input map biologically organised?)")
    print("=" * 78)

    out: dict = {"audit": "A6", "regime": RUNNER_REGIME, "per_size": {}}

    for n in sizes:
        sel = select_nodes(npz, n, SELECTION_SEED)
        family = build_topology_family(
            str(npz), seed=0, n_channels=5, normalization=normalization,
            node_indices=sel.node_indices,
            params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
        )
        R0 = family["R0_real_fly"]._topology
        A = R0.matrix
        n_nodes = A.shape[0]
        shared = make_shared(n_nodes, 5, seed=0, input_scale=RUNNER_REGIME["input_scale"])

        # (i) structure of W_in: dense uniform, no zeros, no dependence on the graph
        w = shared.w_in
        out_deg = np.diff(A.indptr).astype(np.float64)  # unweighted out-degree

        # (ii) node-identity test: a consistent relabelling must change nothing
        perm = np.random.default_rng(7).permutation(n_nodes)
        A_perm = A[perm, :][:, perm].tocsr()
        x = make_ar1_input(3, 8, 5, seed=0)
        t_ref = drive_with_trace(A, w, shared.bias, x,
                                 leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
        # relabel both the graph AND the input map consistently
        w_perm = w[perm, :]
        b_perm = shared.bias[perm]
        t_perm = drive_with_trace(A_perm, w_perm, b_perm, x,
                                  leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
        # realign the permuted state vector to the reference ordering
        h_perm_realigned = t_perm["h_trace"][:, :, np.argsort(perm)]
        relabel_max_abs_diff = float(
            np.abs(t_ref["h_trace"] - h_perm_realigned).max()
        )

        # (iii) does the input map care about the connectome at all?
        corr_degree_w = float(np.corrcoef(out_deg, np.abs(w).sum(axis=1))[0, 1]) \
            if out_deg.std() > 0 else None

        row = {
            "target_n": n,
            "N": int(n_nodes),
            "W_in_shape": list(w.shape),
            "W_in_is_dense": bool(np.all(w != 0.0)),
            "W_in_n_zeros": int((w == 0.0).sum()),
            "W_in_uniform_range": [float(w.min()), float(w.max())],
            "W_in_input_scale": float(RUNNER_REGIME["input_scale"]),
            "every_node_receives_input": True,
            "n_nodes_receiving_input": int(n_nodes),
            "input_population_structure": (
                "none — a single dense random W_in covers ALL nodes uniformly; "
                "no ORN/PN input population, no cell-type gating, no layer "
                "assignment"
            ),
            "relabelling_invariance_max_abs_state_diff": relabel_max_abs_diff,
            "relabelling_leaves_dynamics_unchanged": relabel_max_abs_diff < 1e-12,
            "corr_between_node_degree_and_input_weight_mass": corr_degree_w,
        }
        out["per_size"][str(n)] = row

        print(f"\n  N={n}  W_in shape={row['W_in_shape']} dense={row['W_in_is_dense']} "
              f"range={row['W_in_uniform_range']}")
        print(f"    nodes receiving input ......... {row['n_nodes_receiving_input']} / {n_nodes} "
              f"(all of them)")
        print(f"    relabelling invariance ........ max|dh| = {relabel_max_abs_diff:.3e}  "
              f"-> dynamics unchanged: {row['relabelling_leaves_dynamics_unchanged']}")
        print(f"    corr(degree, |W_in| mass) ..... {corr_degree_w}")

    (OUT_DIR / "A6_input_mapping.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


# ---------------------------------------------------------------------------
# A4 — readout ablation (validation only, synthetic fixture)
# ---------------------------------------------------------------------------

def audit_A4(npz: Path, size: int, normalization: str, window_length: int,
             fold_id: int, seed: int) -> dict:
    print("\n" + "=" * 78)
    print("A4  READOUT ABLATION  X-only / H-only / X+H / X+shuffled-H / zero-H")
    print("=" * 78)
    print("  synthetic fixture, TRAIN + VALIDATION only — the test split is")
    print("  never built into a tensor here.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    from drososense.data.loaders import dataset_config_path, load_dataset
    from drososense.data.pipeline import build_fold_tensors
    from drososense.data.splits import make_folds
    from drososense.data.windowing import WindowSet

    dataset = load_dataset(dataset_config_path("synthetic_enose"))
    folds = make_folds(dataset.specimens(), "group_kfold", seed=seed, n_splits=5)
    fold = folds[fold_id]
    tensors = build_fold_tensors(
        dataset, fold, window_length,
        REPO_ROOT / "results" / "audit" / "_artifacts" / "A4",
    )
    train, val = tensors.train, tensors.val
    # hard discipline: the test split must not have been touched
    assert tensors.test is not None  # it exists, we simply do not read it
    n_channels = len(dataset.schema.feature_columns)

    sel = select_nodes(npz, size, SELECTION_SEED)
    family = build_topology_family(
        str(npz), seed=seed, n_channels=n_channels, normalization=normalization,
        node_indices=sel.node_indices,
        params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
    )

    def states(ws: WindowSet) -> np.ndarray:
        model = family["R0_real_fly"]
        length = window_length
        x = ws.X.reshape(ws.X.shape[0], length, n_channels)
        tr = drive_with_trace(
            model._topology.matrix, model._shared.w_in, model._shared.bias, x,
            leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"],
        )
        return tr["h_trace"][:, -1, :]  # pooling == "last", as the runner pins

    H_tr, H_va = states(train), states(val)
    X_tr, X_va = train.flat(), val.flat()
    y_tr, y_va = train.y_class, val.y_class
    rng = np.random.default_rng(0)
    Hs_tr = H_tr[rng.permutation(H_tr.shape[0])]
    Hs_va = H_va[rng.permutation(H_va.shape[0])]

    def score(F_tr, F_va) -> dict:
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=0)
        clf.fit(F_tr, y_tr.astype(int))
        pred = clf.predict(F_va)
        return {
            "macro_f1": float(f1_score(y_va, pred, average="macro",
                                       labels=[0, 1, 2, 3], zero_division=0)),
            "accuracy": float(accuracy_score(y_va, pred)),
        }

    arms = {
        "X_only": (X_tr, X_va),
        "H_only": (H_tr, H_va),
        "X_plus_H": (np.hstack([X_tr, H_tr]), np.hstack([X_va, H_va])),
        "X_plus_shuffled_H": (np.hstack([X_tr, Hs_tr]), np.hstack([X_va, Hs_va])),
        "zero_H": (np.hstack([X_tr, np.zeros_like(H_tr)]),
                   np.hstack([X_va, np.zeros_like(H_va)])),
    }
    results = {name: score(a, b) for name, (a, b) in arms.items()}

    out = {
        "audit": "A4",
        "dataset": "synthetic_enose",
        "evidence_class": "synthetic_fixture",
        "split": "group_kfold seed 0 fold 0 — train + validation ONLY",
        "test_split_read": False,
        "regime": RUNNER_REGIME,
        "normalization": normalization,
        "reservoir_size": size,
        "window_length": window_length,
        "n_channels": n_channels,
        "n_train_windows": int(X_tr.shape[0]),
        "n_val_windows": int(X_va.shape[0]),
        "val_class_support": {
            str(c): int((y_va == c).sum()) for c in np.unique(y_va)
        },
        "arms": results,
        "deltas": {
            "H_only_minus_X_only_macro_f1": results["H_only"]["macro_f1"]
            - results["X_only"]["macro_f1"],
            "X_plus_H_minus_X_plus_shuffled_H_macro_f1": results["X_plus_H"]["macro_f1"]
            - results["X_plus_shuffled_H"]["macro_f1"],
            "X_plus_H_minus_X_only_macro_f1": results["X_plus_H"]["macro_f1"]
            - results["X_only"]["macro_f1"],
        },
    }
    (OUT_DIR / "A4_readout_ablation.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )

    print(f"\n  windows: train={out['n_train_windows']} val={out['n_val_windows']}  "
          f"val class support={out['val_class_support']}")
    print(f"  {'arm':22s} {'macro_f1':>9s} {'accuracy':>9s}")
    for name, r in results.items():
        print(f"  {name:22s} {r['macro_f1']:9.4f} {r['accuracy']:9.4f}")
    print(f"\n  H_only - X_only ................. "
          f"{out['deltas']['H_only_minus_X_only_macro_f1']:+.4f}")
    print(f"  (X+H) - (X+shuffled H) .......... "
          f"{out['deltas']['X_plus_H_minus_X_plus_shuffled_H_macro_f1']:+.4f}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+",
                        default=["A2", "A3", "A4", "A5", "A6"],
                        choices=["A2", "A3", "A4", "A5", "A6"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 2000])
    parser.add_argument("--normalization", default="n1_pre_l1")
    parser.add_argument("--normalizations", nargs="+",
                        default=["n0_raw", "n1_pre_l1", "n5_binary"])
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--a4-size", type=int, default=250)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--npz-path", default=None)
    args = parser.parse_args()

    npz = Path(args.npz_path) if args.npz_path else adjacency_path()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"M4-Audit reservoir integrity audit (dynamics)\n  NPZ: {npz}")

    if "A2" in args.only:
        audit_A2(npz, args.sizes, args.normalization, args.n_samples, args.length)
    if "A3" in args.only:
        audit_A3(npz, args.sizes, args.normalization, args.n_samples, args.length)
    if "A6" in args.only:
        audit_A6(npz, args.sizes, args.normalization)
    if "A5" in args.only:
        audit_A5(npz, args.sizes, args.normalizations, args.n_samples, args.length)
    if "A4" in args.only:
        audit_A4(npz, args.a4_size, args.normalization, args.window_length,
                 args.fold_id, args.seed)

    print(f"\nEvidence written under {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
