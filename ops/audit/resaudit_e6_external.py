#!/usr/bin/env python
"""E6 — post-hoc external validity: do synthetic task performances track frozen A3?

READ-ONLY on food data (none is touched). Changes nothing: A3_FACTOR stays 2.0, the Stage-1
result is untouched, the food stage stays NOT RELEASED, and no replacement gate or winning
metric is selected.

Reuses EXACTLY the 181-substrate pool of E5 (same deterministic seeds), at the frozen primary
operating point rho = 0.95, with a limited robustness check at rho in {0.50, 0.95, 1.05}.

One drive per (substrate, rho) with a single task-blind stream u ~ U(0,1), T = TASK_STEPS,
serves all three tasks:

  1. delayed recall      target u_{t-tau}, tau = 1..50
                         reports MC(total) = sum_{tau=1..50} R^2_tau
                                 MC(long)  = sum_{tau=26..50} R^2_tau
  2. NARMA-10            standard recursion driven by 0.5*u (the benchmark's U(0,0.5) input)
                         reports NRMSE on the held-out test segment
  3. delayed XOR/parity  y_t = bit(u_{t-t1}) XOR bit(u_{t-t2}) at (3,7) and (5,15)
                         reports accuracy and balanced accuracy

Ridge readout protocol, IDENTICAL for every substrate: train-only standardisation, a fixed
alpha grid selected on the validation segment, scored once on the test segment. No
per-substrate hyperparameter tuning and no per-task tuning beyond the shared grid.

Run:
    PYTHONPATH=. python ops/audit/resaudit_e6_external.py [--out DIR]
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
    scale_to_spectral_radius,
)

DEFAULT_OUT = REPO / "results/audit/resaudit_a3_external"
F1_A = REPO / "results/audit/resaudit_stage1/f1/F1_A.npz"
F1_B = REPO / "results/audit/resaudit_stage1/f1/F1_B_Din5_historical.npz"

A3_FACTOR = 2.0          # frozen; asserted only
TASK_STEPS = 4500
WASHOUT = 200
TRAIN_N, VAL_N = 2400, 800
N_BOOT = 2000
TAU_TOTAL = 50
TAU_LONG_START = 26
XOR_PAIRS = ((3, 7), (5, 15))
ALPHA_GRID = (1e-6, 1e-4, 1e-2, 1.0)
RHO_ROBUST = (0.50, 0.95, 1.05)
ROBUST_N = 20            # per family for the robustness subset


# ---------------------------------------------------------------------------
# identical pool to E5
# ---------------------------------------------------------------------------


def load_f1() -> tuple[sp.csr_matrix, np.ndarray]:
    npz = np.load(F1_A)
    shape = tuple(int(v) for v in npz["adj_shape"])
    A = sp.csr_matrix((np.asarray(npz["adj_data"], float), np.asarray(npz["adj_indices"]),
                       np.asarray(npz["adj_indptr"])), shape=shape)
    return A, np.asarray(np.load(F1_B)["w_in"], dtype=np.float64)


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


def build_pool(n_f2=50, n_f3=50, n_cm=50, n_f5=30):
    """The E5 pool, reproduced from the same seeds."""
    A_F1, B_F1 = load_f1()
    n = A_F1.shape[0]; din = B_F1.shape[1]
    pool = [{"family": "F1", "seed": 0,
             "A": scale_to_spectral_radius(A_F1, FROZEN_RHO_TARGET), "B": B_F1}]
    for kind, count in (("F2", n_f2), ("F3", n_f3), ("CM", n_cm), ("F5", n_f5)):
        for k in range(count):
            seed = 7000 + k
            G = gen_pool(kind, A_F1, seed=seed)
            pool.append({"family": kind, "seed": seed,
                         "A": scale_to_spectral_radius(G, FROZEN_RHO_TARGET),
                         "B": uniform_input(n, din, seed=seed)})
    return pool, A_F1, din


# ---------------------------------------------------------------------------
# task definitions
# ---------------------------------------------------------------------------


def task_stream(*, seed: int, steps: int, din: int) -> tuple[np.ndarray, np.ndarray]:
    """Task-blind stream with ``din`` channels, matching B.

    Channel 0 carries the task signal u ~ U(0,1); channels 1..din-1 carry independent
    uniform noise. All tasks are defined on channel 0, so the extra channels are
    task-blind input dimension, not task information.
    """
    rng = np.random.default_rng(20260924 + int(seed))
    U = rng.uniform(0.0, 1.0, size=(steps, max(1, int(din))))
    return U, (U[:, 0] > 0.5).astype(np.float64)


def narma10_target(u_half: np.ndarray) -> np.ndarray:
    """Standard NARMA-10 driven by u ~ U(0, 0.5).

        y_t = 0.3 y_{t-1} + 0.05 y_{t-1} sum_{i=1..10} y_{t-i}
              + 1.5 u_{t-10} u_t + 0.1
    """
    t = len(u_half)
    y = np.zeros(t)
    for k in range(10, t):
        y[k] = (0.3 * y[k - 1]
                + 0.05 * y[k - 1] * float(y[k - 10:k].sum())
                + 1.5 * u_half[k - 10] * u_half[k]
                + 0.1)
    return y


def ridge_fit_predict(Ztr, ytr, Zva, yva, Zte, alphas=ALPHA_GRID):
    """Train-only standardisation, alpha chosen on validation, scored once on test.

    IDENTICAL for every substrate and every task in this experiment.
    """
    mu = Ztr.mean(axis=0, keepdims=True)
    sd = Ztr.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Ztr = (Ztr - mu) / sd; Zva = (Zva - mu) / sd; Zte = (Zte - mu) / sd
    best = None
    ym = ytr.mean()
    for a in alphas:
        G = Ztr.T @ Ztr + a * np.eye(Ztr.shape[1])
        w = np.linalg.solve(G, Ztr.T @ (ytr - ym))
        pred_va = Zva @ w + ym
        err = float(np.mean((pred_va - yva) ** 2))
        if best is None or err < best[0]:
            best = (err, w, a, ym)
    _, w, a, ym = best
    return Zte @ w + ym, a


def run_tasks(Zf: np.ndarray, u: np.ndarray, ub: np.ndarray) -> dict:
    """All three tasks from one reduced feature matrix. ``Zf`` shape (T, r)."""
    s = Zf[WASHOUT:]
    n = s.shape[0]
    i_tr = slice(0, TRAIN_N)
    i_va = slice(TRAIN_N, TRAIN_N + VAL_N)
    i_te = slice(TRAIN_N + VAL_N, n)
    out: dict = {"n_steps_used": int(n)}

    # --- 1. delayed recall / memory curve ---
    mc = 0.0; mc_long = 0.0; curve = {}
    # ALIGNMENT: s[k] is the state at ABSOLUTE time WASHOUT + k, so the target for s[k]
    # is the input at absolute time WASHOUT + k - tau. Indexing y from absolute 0 instead
    # would misalign by WASHOUT samples; on an i.i.d. stream that alone drives R^2 to ~0.
    u_abs = u[:, 0]
    for tau in range(1, TAU_TOTAL + 1):
        start = WASHOUT - tau
        if start < 0:
            break
        y = u_abs[start : start + n]
        Z = s
        if y.size != Z.shape[0]:
            y = y[: Z.shape[0]]
        if Z.shape[0] < TRAIN_N + VAL_N + 100:
            break
        tr = slice(0, TRAIN_N); va = slice(TRAIN_N, TRAIN_N + VAL_N)
        te = slice(TRAIN_N + VAL_N, Z.shape[0])
        pred, _a = ridge_fit_predict(Z[tr], y[tr], Z[va], y[va], Z[te])
        yt = y[te]
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        r2 = max(0.0, 1.0 - float(((yt - pred) ** 2).sum()) / ss_tot) if ss_tot > 0 else 0.0
        curve[tau] = r2
        mc += r2
        if tau >= TAU_LONG_START:
            mc_long += r2
    out["MC_total"] = float(mc)
    out["MC_long"] = float(mc_long)
    out["MC_curve"] = curve

    # --- 2. NARMA-10 ---
    u_half = (u[:, 0] * 0.5)
    y = narma10_target(u_half)
    yv = y[WASHOUT:]
    if yv.size > TRAIN_N + VAL_N + 100:
        tr = slice(0, TRAIN_N); va = slice(TRAIN_N, TRAIN_N + VAL_N)
        te = slice(TRAIN_N + VAL_N, yv.size)
        pred, _a = ridge_fit_predict(s[tr], yv[tr], s[va], yv[va], s[te])
        yt = yv[te]
        out["NARMA_NRMSE"] = float(
            np.sqrt(np.mean((yt - pred) ** 2)) / max(float(yt.std()), 1e-12))
    else:
        out["NARMA_NRMSE"] = float("nan")

    # --- 3. delayed XOR / parity ---
    for (t1, t2) in XOR_PAIRS:
        lag = max(t1, t2)
        if lag >= n:
            out[f"XOR_{t1}_{t2}_acc"] = float("nan")
            out[f"XOR_{t1}_{t2}_bal"] = float("nan")
            continue
        # ALIGNMENT: same convention -- target for s[k] uses absolute time WASHOUT + k
        idx = np.arange(n)
        a1 = WASHOUT + idx - t1
        a2 = WASHOUT + idx - t2
        ok = (a1 >= 0) & (a2 >= 0)
        idx = idx[ok]
        y = (ub[WASHOUT + idx - t1] != ub[WASHOUT + idx - t2]).astype(np.float64)
        Z = s[idx]
        if Z.shape[0] < TRAIN_N + VAL_N + 100:
            out[f"XOR_{t1}_{t2}_acc"] = float("nan")
            out[f"XOR_{t1}_{t2}_bal"] = float("nan")
            continue
        tr = slice(0, TRAIN_N); va = slice(TRAIN_N, TRAIN_N + VAL_N)
        te = slice(TRAIN_N + VAL_N, Z.shape[0])
        pred, _a = ridge_fit_predict(Z[tr], y[tr], Z[va], y[va], Z[te])
        yt = y[te]
        lab = (pred > 0.5).astype(np.float64)
        acc = float((lab == yt).mean())
        bal = 0.0; ncls = 0
        for c in (0.0, 1.0):
            m = yt == c
            if m.sum() > 0:
                bal += float((lab[m] == c).mean()); ncls += 1
        out[f"XOR_{t1}_{t2}_acc"] = acc
        out[f"XOR_{t1}_{t2}_bal"] = bal / ncls if ncls else float("nan")
    return out


# ---------------------------------------------------------------------------
# nonlinear state descriptors (recomputed on the same trajectories, same formulas as E5)
# ---------------------------------------------------------------------------


def state_descriptors(states: np.ndarray) -> dict:
    s = states[WASHOUT:]
    flat = s - s.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(flat, compute_uv=False)
    lam = sv ** 2
    pos = lam[lam > 0]
    preff = float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0
    p = pos / pos.sum() if pos.size else np.array([])
    ent = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    # window-level descriptors from the task trajectory, using TRAIN_N-sized blocks
    nb = max(2, flat.shape[0] // 512)
    chunks = np.array_split(flat, nb)
    centers = np.stack([c.mean(axis=0) for c in chunks])
    gram = centers @ centers.T
    eig = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    epos = eig[eig > 0]
    k_er = float(epos.sum() ** 2 / (epos ** 2).sum()) if epos.size else 0.0
    norms = np.linalg.norm(centers, axis=1)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    iu = np.triu_indices(centers.shape[0], k=1)
    sep = float(d[iu].mean() / max(norms.mean(), 1e-12))
    return {"state_PReff": preff, "state_entropy_rank": ent,
            "kernel_erank": k_er, "state_separation": sep}


def reduce_features(S: np.ndarray, r: int = 128) -> np.ndarray:
    """Top-r PCA scores of the (post-washout) state trajectory.

    Identical reduction for every substrate and every task, so the readout protocol is
    unchanged; it only bounds the ridge problem size. The nonlinear STATE DESCRIPTORS are
    still computed on the full state matrix, not on this reduction.
    """
    # basis from the post-washout block, but scores returned for the FULL length so that
    # run_tasks applies the washout exactly once
    X = S[WASHOUT:]
    mu = X.mean(axis=0, keepdims=True)
    _U, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(r, sv.size)
    V = Vt[:k].T
    return (S - mu) @ V


def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rk(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(v.size, float)
        r[o] = np.arange(v.size, dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rk(x), rk(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_boot(x, y, *, n_boot=N_BOOT, seed=0) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = np.array([spearman(x[i], y[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"spearman": rho, "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}
    return {"spearman": rho, "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)), "n": int(x.size)}


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--robust-n", type=int, default=ROBUST_N)
    ap.add_argument("--skip-robust", action="store_true")
    args = ap.parse_args(argv)

    assert A3_FACTOR == 2.0, "A3_FACTOR frozen at 2.0"
    t0 = time.monotonic()
    pool, A_F1, din = build_pool()
    print(f"pool reproduced: {len(pool)} substrates ({time.monotonic()-t0:.0f}s)")

    u, ub = task_stream(seed=0, steps=TASK_STEPS, din=din)
    print(f"task stream: u~U(0,1), T={TASK_STEPS}, washout={WASHOUT}, "
          f"train/val/test={TRAIN_N}/{VAL_N}/{TASK_STEPS-WASHOUT-TRAIN_N-VAL_N}")

    descriptors = ("state_PReff", "state_entropy_rank", "kernel_erank", "state_separation")
    targets = ("MC_total", "MC_long", "NARMA_NRMSE",
               "XOR_3_7_acc", "XOR_3_7_bal", "XOR_5_15_acc", "XOR_5_15_bal")

    def drive_and_score(A, B):
        # ONE long window: X must be (n_samples, length, Din) -> (1, TASK_STEPS, din)
        X = u[None, :, :]
        d = reservoir_drive(A, sp.csr_matrix(B), np.zeros(A.shape[0]), X,
                            gain=PROBE_GAIN, leak=PROBE_LEAK)
        S = d.states[0]                       # (TASK_STEPS, N)
        Zf = reduce_features(S, 128)          # (TASK_STEPS, 128) for the readouts
        r = run_tasks(Zf, u, ub)
        r.update(state_descriptors(S))
        r["frozen_A3"] = float(krylov_score(A, B, K=16).effective_rank)
        return r

    # ---------------- primary: rho = 0.95 ----------------
    rows = []
    t0 = time.monotonic()
    for i, it in enumerate(pool):
        r = drive_and_score(it["A"], it["B"])
        r["family"] = it["family"]; r["seed"] = it["seed"]
        rows.append(r)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(pool)} ({time.monotonic()-t0:.0f}s)")
    print(f"primary task measurements done in {time.monotonic()-t0:.0f}s")

    predictors = ("frozen_A3",) + descriptors
    corr = {t: {p: spearman_boot([r[p] for r in rows], [r[t] for r in rows]) for p in predictors}
            for t in targets}

    # ---------------- robustness at rho in {0.50, 0.95, 1.05} ----------------
    robust = None
    if not args.skip_robust:
        sub = []
        for f in ("F1", "F2", "F3", "CM", "F5"):
            items = [it for it in pool if it["family"] == f]
            sub.extend(items[: max(1, min(args.robust_n, len(items)))])
        print(f"robustness: {len(sub)} substrates x {len(RHO_ROBUST)} rho values")
        robust = {}
        for rho in RHO_ROBUST:
            t1 = time.monotonic()
            a3v, tv, dv = [], {t: [] for t in targets}, {p: [] for p in descriptors}
            for it in sub:
                Ar = scale_to_spectral_radius(it["A"], float(rho))
                r = drive_and_score(Ar, it["B"])
                a3v.append(r["frozen_A3"])
                for t in targets:
                    tv[t].append(r[t])
                for p in descriptors:
                    dv[p].append(r[p])
            robust[f"{rho}"] = {
                "rho": float(rho),
                "A3_vs_task": {t: spearman_boot(a3v, tv[t]) for t in targets},
                "descriptor_vs_task": {t: {p: spearman_boot(dv[p], tv[t]) for p in descriptors}
                                       for t in targets},
            }
            print(f"  rho={rho}: A3~MC_total="
                  f"{robust[f'{rho}']['A3_vs_task']['MC_total']['spearman']:+.2f} "
                  f"A3~NARMA={robust[f'{rho}']['A3_vs_task']['NARMA_NRMSE']['spearman']:+.2f} "
                  f"({time.monotonic()-t1:.0f}s)")

    # family distributions of the task metrics
    fam_dist = {}
    for t in targets:
        fam_dist[t] = {}
        for f in ("F1", "F2", "F3", "CM", "F5"):
            v = np.array([r[t] for r in rows if r["family"] == f], dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                fam_dist[t][f] = {"n": int(v.size), "mean": float(v.mean()),
                                  "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}

    report = {
        "report": "E6 post-hoc external validity on synthetic tasks",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": {"A3_FACTOR": A3_FACTOR, "rho_primary": FROZEN_RHO_TARGET,
                   "leak": PROBE_LEAK, "gain": PROBE_GAIN,
                   "task_steps": TASK_STEPS, "washout": WASHOUT,
                   "train": TRAIN_N, "val": VAL_N, "alpha_grid": list(ALPHA_GRID),
                   "n_boot": N_BOOT, "tau_total": TAU_TOTAL,
                   "tau_long_start": TAU_LONG_START, "xor_pairs": [list(p) for p in XOR_PAIRS]},
        "pool": {f: sum(1 for r in rows if r["family"] == f) for f in ("F1", "F2", "F3", "CM", "F5")},
        "primary_correlations": corr,
        "robustness_rho": robust,
        "family_distributions": fam_dist,
        "rows": [{k: v for k, v in r.items() if k != "MC_curve"} for r in rows],
        "no_replacement_gate": True,
        "food_data_accessed": False,
        "note": ("Post-hoc exploratory external-validity study. No gate is defined on any task "
                 "metric, no winning metric is selected, and A3_FACTOR / Stage-1 are untouched."),
    }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "E6_external.json").write_text(json.dumps(report, indent=1, default=str),
                                             encoding="utf-8")
    print(f"\nwritten: {outdir/'E6_external.json'}")

    print(f"\n=== PRIMARY (rho=0.95, n={len(rows)}): Spearman(task, predictor) ===")
    hdr = f"{'task':16}" + "".join(f"{p[:14]:>16}" for p in predictors)
    print(hdr)
    for t in targets:
        line = f"{t:16}"
        for p in predictors:
            c = corr[t][p]
            line += f"{c['spearman']:>+8.3f}[{c['ci_low']:+.2f},{c['ci_high']:+.2f}]"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
