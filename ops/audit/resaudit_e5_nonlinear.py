#!/usr/bin/env python
"""E5 — nonlinear construct validation of frozen A3.

QUESTION: does frozen A3 have empirical construct validity with respect to nonlinear
reservoir state richness and memory?

Uses the FROZEN dynamics and the FROZEN task-blind probe protocol. Changes nothing:
A3_FACTOR stays 2.0, the rho convention stays 0.95 for the primary measurement, and no
Stage-1 outcome is touched. No new gate is defined and no replacement metric is selected.

Frozen ingredients (asserted, not re-chosen):
    dynamics   x_{t+1} = (1-leak) x_t + leak * tanh(gain * A x_t + W_in u_t + bias)
               with leak = 1.0, gain = 1.0
    probe      48 windows x 256 steps, seed 20260923, washout 16  (battery.probe_input)
    A3         (sum sigma)^2 / sum sigma^2 over sigma > tau0, blocks [B,...,A^16 B]

Nonlinear observables, all computed on the SAME trajectories:
    state_PReff     participation ratio of the state-covariance eigenvalues
    state_entropy   exp(entropy) of the normalised state-covariance eigenvalues
    kernel_erank    participation ratio of the window-Gram eigenvalue spectrum
    state_separation mean pairwise distance between window states / mean state norm
    MC              linear memory capacity, sum_tau R_tau^2 over tau = 1..TAU_MAX
                    (ridge/OLS readout from the top-r state components)

Correlations are Spearman with a substrate-level bootstrap 95% CI, because a Pearson
coefficient on a 5-point family distribution would be meaningless.

Run:
    PYTHONPATH=. python ops/audit/resaudit_e5_nonlinear.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from resaudit._kernels import krylov_score, reservoir_drive  # noqa: E402
from resaudit.battery import (  # noqa: E402
    FROZEN_RHO_TARGET,
    PROBE_GAIN,
    PROBE_LEAK,
    PROBE_WASHOUT,
    probe_input,
    scale_to_spectral_radius,
)

DEFAULT_OUT = REPO / "results/audit/resaudit_a3_nonlinear"
F1_A = REPO / "results/audit/resaudit_stage1/f1/F1_A.npz"

A3_FACTOR = 2.0            # frozen; asserted only
RHO_LADDER = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.05)
N_BOOT = 2000
TAU_MAX = 20               # memory lags 1..20
PCA_RANK = 128             # components used for the memory readout
RIDGE_ALPHA = 1e-6


# ---------------------------------------------------------------------------
# substrates (same frozen semantics as E1/E3)
# ---------------------------------------------------------------------------


def load_f1() -> tuple[sp.csr_matrix, np.ndarray]:
    npz = np.load(F1_A)
    shape = tuple(int(v) for v in npz["adj_shape"])
    A = sp.csr_matrix((np.asarray(npz["adj_data"], float), np.asarray(npz["adj_indices"]),
                       np.asarray(npz["adj_indptr"])), shape=shape)
    B = np.asarray(np.load(REPO / "results/audit/resaudit_stage1/f1/F1_B_Din5_historical.npz")["w_in"],
                   dtype=np.float64)
    return A, B


def uniform_input(n: int, din: int, *, seed: int, density: float = 0.10) -> np.ndarray:
    from resaudit.contracts import apportion_din
    alloc = apportion_din(din)
    rng = np.random.default_rng(int(seed) ^ 0xA1B2)
    B = np.zeros((n, din)); col = 0
    for _l, count in alloc.items():
        cuts = np.linspace(0, n, count + 1, dtype=int)
        for k in range(count):
            blk = np.arange(cuts[k], cuts[k + 1])
            if blk.size == 0:
                col += 1; continue
            nr = max(1, int(round(density * blk.size)))
            B[rng.choice(blk, size=min(nr, blk.size), replace=False), col] = 1.0
            col += 1
    return B


def gen_pool(kind: str, A_F1: sp.csr_matrix, *, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(int(seed)); n, m = A_F1.shape[0], A_F1.nnz
    if kind == "F2":
        from resaudit.families import f2_swaps_at_budget
        G, _a, _d = f2_swaps_at_budget(A_F1, seed=seed, accepted_target=10 * m)
        return G
    if kind == "F3":
        chosen: set[tuple[int, int]] = set(); g = 0
        while len(chosen) < m and g < 200 * m + 1000:
            g += 1
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b:
                chosen.add((a, b))
        r = np.fromiter((e[0] for e in chosen), np.int64, len(chosen))
        c = np.fromiter((e[1] for e in chosen), np.int64, len(chosen))
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    if kind == "CM":
        out_deg = np.diff(A_F1.indptr).astype(np.int64)
        in_deg = np.diff(A_F1.tocsc().indptr).astype(np.int64)
        ins = np.repeat(np.arange(n), in_deg); outs = np.repeat(np.arange(n), out_deg)
        rng.shuffle(ins)
        return sp.csr_matrix((np.ones(ins.size), (outs, ins)), shape=(n, n))
    if kind == "F5":
        lengths = (101, 103, 107, 109, 113, 127, 131, 209)
        srcs, tgts, offs, off = [], [], [], 0
        for L in lengths:
            offs.append(off)
            local = np.repeat(np.arange(L), L - 2)
            delta = np.concatenate([np.arange(2, L) for _ in range(L)])
            srcs.append(off + local); tgts.append(off + (local + delta) % L); off += L
        cs, ct = np.concatenate(srcs), np.concatenate(tgts)
        cy = np.concatenate([offs[b] + np.arange(L) for b, L in enumerate(lengths)])
        cyt = np.concatenate([offs[b] + (np.arange(L) + 1) % L for b, L in enumerate(lengths)])
        pick = rng.permutation(cs.size)[: m - cy.size]
        r = np.concatenate([cy, cs[pick]]); c = np.concatenate([cyt, ct[pick]])
        return sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# nonlinear observables
# ---------------------------------------------------------------------------


def _pr(vals: np.ndarray) -> float:
    pos = vals[vals > 0]
    if pos.size == 0:
        return 0.0
    return float(pos.sum() ** 2 / (pos ** 2).sum())


def _entropy_rank(vals: np.ndarray) -> float:
    pos = vals[vals > 0]
    if pos.size == 0:
        return 0.0
    p = pos / pos.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def nonlinear_metrics(states: np.ndarray, X: np.ndarray, *, washout: int = PROBE_WASHOUT) -> dict:
    """All nonlinear observables from one trajectory batch.

    ``states`` (n_win, length, N); ``X`` (n_win, length, Din).
    """
    s = states[:, washout:, :]                      # (W, T, N)
    w, t, n = s.shape
    flat = s.reshape(w * t, n)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # state covariance spectrum via SVD of the centered data
    sv = np.linalg.svd(flat, compute_uv=False)
    lam = sv ** 2
    state_preff = _pr(lam)
    state_entropy = _entropy_rank(lam)

    # window-level states for kernel quality / separation
    centers = s.mean(axis=1)                        # (W, N)
    gram = centers @ centers.T                      # (W, W)
    eig = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    kernel_erank = _pr(eig)
    norms = np.linalg.norm(centers, axis=1)
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    iu = np.triu_indices(w, k=1)
    sep = float(dmat[iu].mean() / max(norms.mean(), 1e-12))

    # linear memory capacity: OLS readout from top-r state components
    r = min(PCA_RANK, flat.shape[0] - 1, n)
    u_, sv_, vt_ = np.linalg.svd(flat, full_matrices=False)
    comp = u_[:, :r] * sv_[:r]                      # (W*T, r) PCA scores
    comp = comp - comp.mean(axis=0, keepdims=True)
    y_all = X[:, washout:, :].reshape(w * t, X.shape[2])
    mc = 0.0
    mc_by_tau = {}
    for tau in range(1, TAU_MAX + 1):
        if t - tau <= 0:
            break
        y = np.roll(y_all, tau, axis=0)[:, 0]
        # avoid wrap contamination at window boundaries
        A_ = comp[: (t - tau) * w]
        yv = y[: (t - tau) * w]
        if A_.shape[0] < r + 2:
            break
        G = A_.T @ A_ + RIDGE_ALPHA * np.eye(r)
        wgt = np.linalg.solve(G, A_.T @ yv)
        pred = A_ @ wgt
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mc += r2
        mc_by_tau[tau] = r2
    return {
        "state_PReff": state_preff,
        "state_entropy_rank": state_entropy,
        "kernel_erank": kernel_erank,
        "state_separation": sep,
        "MC": float(mc),
        "MC_by_tau": mc_by_tau,
        "state_sigma_max": float(sv[0]) if sv.size else 0.0,
    }


def measure(A: sp.spmatrix, B: np.ndarray, X: np.ndarray) -> dict:
    n = A.shape[0]
    d = reservoir_drive(A, sp.csr_matrix(B), np.zeros(n), X, gain=PROBE_GAIN, leak=PROBE_LEAK)
    out = nonlinear_metrics(d.states, X)
    s = krylov_score(A, B, K=16)
    out["frozen_A3"] = float(s.effective_rank)
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, float); r[order] = np.arange(v.size, dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_boot(x: np.ndarray, y: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = np.array([spearman(x[i], y[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"spearman": rho, "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}
    return {"spearman": rho, "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)), "n": int(x.size),
            "n_boot_used": int(vals.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-f2", type=int, default=50)
    ap.add_argument("--n-f3", type=int, default=50)
    ap.add_argument("--n-cm", type=int, default=50)
    ap.add_argument("--n-f5", type=int, default=30)
    ap.add_argument("--ladder-n", type=int, default=25,
                    help="substrates per family used for the rho ladder (cost control)")
    ap.add_argument("--skip-ladder", action="store_true")
    args = ap.parse_args(argv)

    assert A3_FACTOR == 2.0, "A3_FACTOR frozen at 2.0"
    A_F1, B_F1 = load_f1()
    n, m = A_F1.shape[0], A_F1.nnz
    din = B_F1.shape[1]
    print(f"real F1 N={n} M={m} Din={din}; frozen probe {probe_input(din).shape}")

    # ---------------- pool ----------------
    t0 = time.monotonic()
    pool: list[dict] = []
    A_s = scale_to_spectral_radius(A_F1, FROZEN_RHO_TARGET)
    pool.append({"family": "F1", "seed": 0, "A": A_s, "B": B_F1})
    for kind, count in (("F2", args.n_f2), ("F3", args.n_f3), ("CM", args.n_cm), ("F5", args.n_f5)):
        for k in range(count):
            seed = 7000 + k
            G = gen_pool(kind, A_F1, seed=seed)
            pool.append({"family": kind, "seed": seed,
                         "A": scale_to_spectral_radius(G, FROZEN_RHO_TARGET),
                         "B": uniform_input(n, din, seed=seed)})
    print(f"pool: {len(pool)} substrates in {time.monotonic()-t0:.1f}s")

    # ---------------- primary measurement at frozen rho ----------------
    X = probe_input(din)
    t0 = time.monotonic()
    rows = []
    for i, item in enumerate(pool):
        r = measure(item["A"], item["B"], X)
        r.update({"family": item["family"], "seed": item["seed"]})
        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  measured {i+1}/{len(pool)}  ({time.monotonic()-t0:.0f}s)")
    print(f"primary measurements done in {time.monotonic()-t0:.1f}s")

    A3 = np.array([r["frozen_A3"] for r in rows])
    metrics = ("state_PReff", "state_entropy_rank", "kernel_erank", "state_separation", "MC")

    correlations = {mm: spearman_boot(A3, np.array([r[mm] for r in rows])) for mm in metrics}

    fams = ("F1", "F2", "F3", "CM", "F5")
    family_dist = {}
    for mm in ("frozen_A3",) + metrics:
        family_dist[mm] = {}
        for f in fams:
            v = np.array([r[mm] for r in rows if r["family"] == f])
            if v.size:
                family_dist[mm][f] = {
                    "n": int(v.size), "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                    "min": float(v.min()), "max": float(v.max()),
                }

    # ---------------- rho ladder (operating-point stability) ----------------
    ladder = None
    if not args.skip_ladder:
        sub = []
        for f in fams:
            items = [it for it in pool if it["family"] == f]
            sub.extend(items[: min(args.ladder_n, len(items))])
        print(f"ladder: {len(sub)} substrates x {len(RHO_LADDER)} rho values")
        ladder = {}
        for rho in RHO_LADDER:
            t1 = time.monotonic()
            a3v, mv = [], {mm: [] for mm in metrics}
            for it in sub:
                Ar = scale_to_spectral_radius(it["A"], float(rho))
                r = measure(Ar, it["B"], X)
                a3v.append(r["frozen_A3"])
                for mm in metrics:
                    mv[mm].append(r[mm])
            a3v = np.array(a3v)
            ladder[f"{rho}"] = {
                "rho": float(rho),
                "correlations": {mm: spearman_boot(a3v, np.array(mv[mm])) for mm in metrics},
                "A3_mean": float(a3v.mean()),
            }
            print(f"  rho={rho}: " + "  ".join(
                f"{mm}={ladder[f'{rho}']['correlations'][mm]['spearman']:+.2f}" for mm in metrics)
                + f"   ({time.monotonic()-t1:.0f}s)")

    report = {
        "report": "E5 nonlinear construct validation of frozen A3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": {
            "A3_FACTOR": A3_FACTOR, "rho_primary": FROZEN_RHO_TARGET,
            "leak": PROBE_LEAK, "gain": PROBE_GAIN, "washout": PROBE_WASHOUT,
            "probe_shape": list(probe_input(din).shape),
            "tau_max": TAU_MAX, "pca_rank": PCA_RANK, "n_boot": N_BOOT,
            "real_F1_N": n, "real_F1_M": m, "Din": din,
        },
        "pool": {f: sum(1 for r in rows if r["family"] == f) for f in fams},
        "primary_rho": float(FROZEN_RHO_TARGET),
        "correlations_spearman_bootstrap": correlations,
        "family_distributions": family_dist,
        "rho_ladder": ladder,
        "rows": [{k: v for k, v in r.items() if k != "MC_by_tau"} for r in rows],
        "no_new_gate": True,
        "note": ("Reported as construct-validity evidence. No gate is defined on any nonlinear "
                 "metric, no replacement metric is selected, and A3_FACTOR is unchanged."),
    }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "E5_nonlinear.json").write_text(json.dumps(report, indent=1, default=str),
                                              encoding="utf-8")
    print(f"\nwritten: {outdir/'E5_nonlinear.json'}")
    print(f"\n=== PRIMARY (rho=0.95): Spearman(A3, metric) with 95pct bootstrap CI, n={len(rows)} ===")
    for mm, c in correlations.items():
        print(f"  {mm:22} rho={c['spearman']:+.3f}  CI=[{c['ci_low']:+.3f},{c['ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
