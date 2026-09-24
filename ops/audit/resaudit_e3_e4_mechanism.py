#!/usr/bin/env python
"""E3 + E4 — mechanism audit of frozen A3, and scale-invariant measurement comparison.

READ-ONLY on data. Writes one JSON (+ spectra). Does NOT change A3_FACTOR (=2.0), does NOT
redefine the frozen Stage-1 result, and does NOT use any sweep to select a replacement
threshold. The frozen gate is reported in every sweep purely as a DIAGNOSTIC.

The frozen estimator, recovered exactly (E3 item 1)
---------------------------------------------------
    D_eff(blocks) = (sum s)^2 / sum s^2      over singular values s > tau0
    blocks        = [B, A*B, ..., A^K*B]     i.e. K+1 = 17 blocks at the frozen K = 16
    tau0          = 1e-10                    ABSOLUTE, from NUMERICAL_TOLERANCE
    gate          = 2 * Din                  unchanged, diagnostic only here

Efficiency: the Krylov block is built once per (substrate, rho) up to K_max = 64. The K sweep
then re-uses column prefixes, and the tau and beta sweeps re-use the SAME singular values
(tau by re-thresholding, beta by rescaling s -> beta*s, which is exact because B -> beta*B
scales every block by beta). Only the rho sweep rebuilds the block.

Run:
    PYTHONPATH=. python ops/audit/resaudit_e3_e4_mechanism.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from resaudit._kernels import KRYLOV_K, NUMERICAL_TOLERANCE  # noqa: E402
from resaudit.battery import FROZEN_RHO_TARGET, scale_to_spectral_radius  # noqa: E402

DEFAULT_OUT = REPO / "results/audit/resaudit_a3_mechanism"
F1_A_PATH = REPO / "results/audit/resaudit_stage1/f1/F1_A.npz"
F1_B_PATH = REPO / "results/audit/resaudit_stage1/f1/F1_B_Din5_historical.npz"

A3_FACTOR = 2.0          # frozen; asserted, never searched
TAU0 = float(NUMERICAL_TOLERANCE)
K_MAX = 64

TAU_RATIOS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
BETAS = (1e-2, 1e-1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 1e2)
RHOS = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.05)
KS = (4, 8, 16, 32, 64)
POOL_PER_FAMILY = 6


# ---------------------------------------------------------------------------
# F1 loader + generators (same frozen semantics as E1)
# ---------------------------------------------------------------------------


def load_f1() -> tuple[sp.csr_matrix, np.ndarray]:
    npz = np.load(F1_A_PATH)
    shape = tuple(int(v) for v in npz["adj_shape"])
    A = sp.csr_matrix((np.asarray(npz["adj_data"], float), np.asarray(npz["adj_indices"]),
                       np.asarray(npz["adj_indptr"])), shape=shape)
    B5 = np.asarray(np.load(F1_B_PATH)["w_in"], dtype=np.float64)
    return A, B5


def uniform_input(n: int, din: int, *, seed: int, density: float = 0.10) -> np.ndarray:
    from resaudit.contracts import apportion_din
    alloc = apportion_din(din)
    rng = np.random.default_rng(int(seed) ^ 0xA1B2)
    B = np.zeros((n, din))
    col = 0
    for _layer, count in alloc.items():
        cuts = np.linspace(0, n, count + 1, dtype=int)
        for k in range(count):
            block = np.arange(cuts[k], cuts[k + 1])
            if block.size == 0:
                col += 1
                continue
            n_recv = max(1, int(round(density * block.size)))
            B[rng.choice(block, size=min(n_recv, block.size), replace=False), col] = 1.0
            col += 1
    return B


def gen_degree_preserving(A: sp.spmatrix, *, seed: int, accepted_target: int) -> sp.csr_matrix:
    from resaudit.families import f2_swaps_at_budget
    rw, _acc, _d = f2_swaps_at_budget(A, seed=seed, accepted_target=accepted_target)
    return rw


def gen_directed_G(n: int, m: int, *, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(int(seed))
    chosen: set[tuple[int, int]] = set()
    guard = 0
    while len(chosen) < m and guard < 200 * m + 1000:
        guard += 1
        a = int(rng.integers(0, n)); b = int(rng.integers(0, n))
        if a != b:
            chosen.add((a, b))
    rows = np.fromiter((e[0] for e in chosen), dtype=np.int64, count=len(chosen))
    cols = np.fromiter((e[1] for e in chosen), dtype=np.int64, count=len(chosen))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def gen_cm(A: sp.spmatrix, *, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(int(seed))
    n = A.shape[0]
    out_deg = np.diff(A.indptr).astype(np.int64)
    in_deg = np.diff(A.tocsc().indptr).astype(np.int64)
    in_stubs = np.repeat(np.arange(n), in_deg)
    out_stubs = np.repeat(np.arange(n), out_deg)
    rng.shuffle(in_stubs)
    return sp.csr_matrix((np.ones(in_stubs.size), (out_stubs, in_stubs)), shape=(n, n))


def gen_f5_attempt(A: sp.spmatrix, *, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(int(seed))
    n, m = A.shape[0], A.nnz
    lengths = (101, 103, 107, 109, 113, 127, 131, 209)
    srcs, tgts, offsets, off = [], [], [], 0
    for L in lengths:
        offsets.append(off)
        local = np.repeat(np.arange(L), L - 2)
        delta = np.concatenate([np.arange(2, L) for _ in range(L)])
        srcs.append(off + local); tgts.append(off + (local + delta) % L)
        off += L
    cs, ct = np.concatenate(srcs), np.concatenate(tgts)
    cy = np.concatenate([offsets[b] + np.arange(L) for b, L in enumerate(lengths)])
    cyt = np.concatenate([offsets[b] + (np.arange(L) + 1) % L for b, L in enumerate(lengths)])
    need = m - cy.size
    pick = rng.permutation(cs.size)[:need]
    rows = np.concatenate([cy, cs[pick]]); cols = np.concatenate([cyt, ct[pick]])
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# The Krylov spectrum and the four estimators
# ---------------------------------------------------------------------------


def krylov_spectrum(A: sp.spmatrix, B: np.ndarray, K: int) -> np.ndarray:
    """Singular values of [B, A*B, ..., A^K*B] (K+1 blocks)."""
    cur = np.asarray(B, dtype=np.float64)
    blocks = [cur]
    for _ in range(int(K)):
        cur = A @ cur
        blocks.append(cur)
    return np.linalg.svd(np.concatenate(blocks, axis=1), compute_uv=False)


def frozen_a3(s: np.ndarray, tau: float = TAU0) -> tuple[float, int]:
    """THE frozen estimator: PR over singular values strictly above an ABSOLUTE tau."""
    pos = s[s > tau]
    if pos.size == 0:
        return 0.0, 0
    return float(pos.sum() ** 2 / (pos ** 2).sum()), int(pos.size)


def stable_rank(s: np.ndarray) -> float:
    """||X||_F^2 / ||X||_2^2 = sum s^2 / s_max^2. Scale-invariant."""
    if s.size == 0 or s[0] <= 0:
        return float("nan")
    return float((s ** 2).sum() / (s[0] ** 2))


def pr_all(s: np.ndarray) -> float:
    """Participation ratio over ALL singular values, i.e. frozen A3 with tau -> 0."""
    pos = s[s > 0]
    if pos.size == 0:
        return 0.0
    return float(pos.sum() ** 2 / (pos ** 2).sum())


def entropy_effective_rank(s: np.ndarray) -> float:
    """exp(H(p)) with p_i = s_i / sum s. Scale-invariant."""
    pos = s[s > 0]
    if pos.size == 0:
        return 0.0
    p = pos / pos.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def all_metrics(s: np.ndarray, tau: float = TAU0) -> dict:
    pr, rank = frozen_a3(s, tau)
    return {
        "frozen_A3": pr,
        "frozen_numerical_rank": rank,
        "stable_rank": stable_rank(s),
        "pr_all": pr_all(s),
        "entropy_effective_rank": entropy_effective_rank(s),
        "sigma_max": float(s[0]) if s.size else 0.0,
        "n_singular_values": int(s.size),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def dist(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size), "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Between-family effect size, pooled SD. Reported as a magnitude, not a threshold."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp2 = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    return float((a.mean() - b.mean()) / math.sqrt(sp2)) if sp2 > 0 else float("nan")


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram-intersection overlap in [0,1]; 1.0 = identical distributions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, 21)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    ha = ha / max(ha.sum(), 1); hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


def pct_of(score: float, pool: np.ndarray) -> float:
    pool = np.asarray(pool, float)
    return float((pool <= score).sum() / pool.size) if pool.size else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pool-per-family", type=int, default=POOL_PER_FAMILY)
    args = ap.parse_args(argv)

    assert A3_FACTOR == 2.0, "A3_FACTOR must stay frozen at 2.0"
    A_F1, B5 = load_f1()
    n, m = A_F1.shape[0], A_F1.nnz
    print(f"real F1: N={n} M={m}  tau0={TAU0:g}  K frozen={KRYLOV_K} (-> {KRYLOV_K+1} blocks)")

    # ---------------- substrate pool (built once, reused by every sweep) ----------------
    t0 = time.monotonic()
    pool: dict[str, list[tuple[sp.csr_matrix, np.ndarray]]] = {}
    din = 6
    # F1 itself
    pool["F1"] = [(scale_to_spectral_radius(A_F1, FROZEN_RHO_TARGET), uniform_input(n, din, seed=0))]
    labels = ("F2_degree_preserving", "F3_directed_G", "CM_degree_matched", "F5_attempt_coprime")
    for fam in labels:
        pool[fam] = []
    for k in range(args.pool_per_family):
        seed = 5000 + k
        for fam in labels:
            if fam == "F2_degree_preserving":
                G = gen_degree_preserving(A_F1, seed=seed, accepted_target=10 * m)
            elif fam == "F3_directed_G":
                G = gen_directed_G(n, m, seed=seed)
            elif fam == "CM_degree_matched":
                G = gen_cm(A_F1, seed=seed)
            else:
                G = gen_f5_attempt(A_F1, seed=seed)
            G = scale_to_spectral_radius(G, FROZEN_RHO_TARGET)
            pool[fam].append((G, uniform_input(n, din, seed=seed)))
    print(f"pool built: {sum(len(v) for v in pool.values())} substrates in {time.monotonic()-t0:.1f}s")

    gate = A3_FACTOR * din

    # ---------------- base spectra at the frozen setting ----------------
    base: dict[str, list[np.ndarray]] = {}
    for fam, items in pool.items():
        base[fam] = [krylov_spectrum(G, B, K_MAX) for G, B in items]

    # ---------------- E3.2 tolerance sweep (re-threshold the SAME spectra) ------------
    tau_sweep = []
    for ratio in TAU_RATIOS:
        tau = TAU0 * ratio
        per = {}
        for fam, specs in base.items():
            vals = np.array([frozen_a3(s, tau)[0] / din for s in specs])
            per[fam] = {"ratio_to_Din": dist(vals),
                        "pass_rate": float((vals >= A3_FACTOR).mean())}
        ranking = [f for f, _ in sorted(
            ((f, per[f]["ratio_to_Din"]["mean"]) for f in per), key=lambda x: -x[1])]
        allvals = np.concatenate([
            np.array([frozen_a3(s, tau)[0] / din for s in base[f]]) for f in base])
        tau_sweep.append({
            "tau_ratio": float(ratio), "tau": float(tau),
            "overall_ratio_to_Din": dist(allvals),
            "overall_pass_rate": float((allvals >= A3_FACTOR).mean()),
            "family_ranking": ranking, "per_family": per,
        })
    print("E3.2 tau sweep done")

    # ---------------- E3.3 input scaling (s -> beta*s, exact) ----------------
    beta_sweep = []
    for beta in BETAS:
        per = {}
        for fam, specs in base.items():
            vals = np.array([frozen_a3(s * beta)[0] / din for s in specs])
            per[fam] = {"ratio_to_Din": dist(vals),
                        "pass_rate": float((vals >= A3_FACTOR).mean())}
        ranking = [f for f, _ in sorted(
            ((f, per[f]["ratio_to_Din"]["mean"]) for f in per), key=lambda x: -x[1])]
        beta_sweep.append({"beta": float(beta), "family_ranking": ranking, "per_family": per})
    print("E3.3 beta sweep done")

    # ---------------- E3.4 rho sweep (rebuild blocks) ----------------
    rho_sweep = []
    for rho in RHOS:
        per = {}
        for fam, items in pool.items():
            vals = []
            for G, B in items:
                Gr = scale_to_spectral_radius(G, float(rho))
                s = krylov_spectrum(Gr, B, KRYLOV_K)
                vals.append(frozen_a3(s)[0] / din)
            vals = np.array(vals)
            per[fam] = {"ratio_to_Din": dist(vals),
                        "pass_rate": float((vals >= A3_FACTOR).mean())}
        ranking = [f for f, _ in sorted(
            ((f, per[f]["ratio_to_Din"]["mean"]) for f in per), key=lambda x: -x[1])]
        rho_sweep.append({"rho": float(rho), "family_ranking": ranking, "per_family": per})
    print("E3.4 rho sweep done")

    # ---------------- E3.5 horizon sweep (column prefixes of the K_MAX block) --------
    K_sweep = []
    for K in KS:
        # recompute at exactly K (prefix reuse changes the SVD, so compute exactly)
        per = {}
        for fam, items in pool.items():
            vals = []
            for G, B in items:
                s = krylov_spectrum(G, B, int(K))
                vals.append(frozen_a3(s)[0] / din)
            vals = np.array(vals)
            per[fam] = {"ratio_to_Din": dist(vals),
                        "pass_rate": float((vals >= A3_FACTOR).mean()),
                        "theoretical_columns": int((K + 1) * din)}
        ranking = [f for f, _ in sorted(
            ((f, per[f]["ratio_to_Din"]["mean"]) for f in per), key=lambda x: -x[1])]
        K_sweep.append({"K": int(K), "family_ranking": ranking, "per_family": per})
    print("E3.5 K sweep done")

    # ---------------- E3.6 spectra, with tau0 marked ----------------
    spectra = {}
    for fam, specs in base.items():
        spectra[fam] = [{
            "sigma": [float(v) for v in s[:64]],
            "n_above_tau0": int((s > TAU0).sum()),
            "n_total": int(s.size),
            "sigma_max": float(s[0]),
            "smallest_above_tau0": float(s[s > TAU0].min()) if (s > TAU0).any() else None,
            "largest_below_tau0": float(s[s <= TAU0].max()) if (s <= TAU0).any() else None,
        } for s in specs]
    # the decisive diagnostic: how many singular values sit near tau0
    counts = []
    for fam, specs in base.items():
        for s in specs:
            counts.append({"family": fam, "n_total": int(s.size),
                           "n_above_tau0": int((s > TAU0).sum()),
                           "frac_pruned": float(1.0 - (s > TAU0).sum() / s.size)})

    # ---------------- E4 scale-invariant comparison ----------------
    e4_families = {}
    for fam, specs in base.items():
        e4_families[fam] = {mname: dist(np.array([all_metrics(s)[mname] for s in specs]))
                            for mname in ("frozen_A3", "stable_rank", "pr_all",
                                          "entropy_effective_rank")}
    e4_families_ratio = {}
    for fam, specs in base.items():
        e4_families_ratio[fam] = {
            mname: dist(np.array([all_metrics(s)[mname] / din for s in specs]))
            for mname in ("frozen_A3", "stable_rank", "pr_all", "entropy_effective_rank")}

    # invariance to beta: the metric that should NOT move is the scale-invariant one
    beta_invariance = {}
    for mname in ("frozen_A3", "stable_rank", "pr_all", "entropy_effective_rank"):
        rows = []
        for beta in BETAS:
            vals = []
            for fam, specs in base.items():
                for s in specs:
                    vals.append(all_metrics(s * beta)[mname])
            rows.append({"beta": float(beta), "mean": float(np.mean(vals)),
                         "sd": float(np.std(vals, ddof=1))})
        means = np.array([r["mean"] for r in rows])
        beta_invariance[mname] = {
            "per_beta": rows,
            "relative_spread": float((means.max() - means.min()) / means.mean())
            if means.mean() > 0 else float("nan"),
        }

    # F1 percentile within each control ensemble, per metric
    f1_pct = {}
    for mname in ("frozen_A3", "stable_rank", "pr_all", "entropy_effective_rank"):
        f1_val = np.mean([all_metrics(s)[mname] for s in base["F1"]])
        f1_pct[mname] = {
            fam: pct_of(f1_val, np.array([all_metrics(s)[mname] for s in specs]))
            for fam, specs in base.items() if fam != "F1"
        }

    # between-family effect sizes and overlap (frozen A3 vs each alternative)
    effect_sizes = {}
    for mname in ("frozen_A3", "stable_rank", "pr_all", "entropy_effective_rank"):
        rows = []
        fams = [f for f in base if f != "F1"]
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a = np.array([all_metrics(s)[mname] for s in base[fams[i]]])
                b = np.array([all_metrics(s)[mname] for s in base[fams[j]]])
                rows.append({"pair": f"{fams[i]} vs {fams[j]}",
                             "cohens_d": cohens_d(a, b), "overlap": overlap(a, b)})
        effect_sizes[mname] = rows

    # within-family variance (relative)
    within = {}
    for mname in ("frozen_A3", "stable_rank", "pr_all", "entropy_effective_rank"):
        within[mname] = {fam: {"mean": d["mean"], "sd": d["sd"],
                               "cv": float(d["sd"] / d["mean"]) if d["mean"] else float("nan")}
                         for fam, d in ((f, dist(np.array([all_metrics(s)[mname] for s in specs])))
                                        for f, specs in base.items())}

    report = {
        "report": "E3 mechanism audit + E4 scale-invariant comparison",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_estimator": {
            "D_eff": "(sum sigma)^2 / sum sigma^2 over sigma > tau0",
            "blocks": "[B, A*B, ..., A^K*B]  -> K+1 blocks",
            "K_frozen": int(KRYLOV_K), "n_blocks": int(KRYLOV_K + 1),
            "tau0": TAU0, "tau0_kind": "ABSOLUTE (NUMERICAL_TOLERANCE)",
            "A3_gate_factor": A3_FACTOR, "gate_status": "frozen, diagnostic only in this run",
            "Din": din, "gate_value": gate,
        },
        "conventions": {"rho_frozen": FROZEN_RHO_TARGET, "pool_per_family": args.pool_per_family,
                        "real_F1_N": n, "real_F1_M": m},
        "pool": {fam: len(v) for fam, v in pool.items()},
        "E3_tau_sweep": tau_sweep,
        "E3_beta_sweep": beta_sweep,
        "E3_rho_sweep": rho_sweep,
        "E3_K_sweep": K_sweep,
        "E3_spectra": {"per_family": spectra, "pruning_counts": counts,
                       "tau0": TAU0, "note": "sigma listed to 64 entries; tau0 drawn explicitly"},
        "E4_per_family": e4_families,
        "E4_per_family_ratio_to_Din": e4_families_ratio,
        "E4_beta_invariance": beta_invariance,
        "E4_F1_percentile": f1_pct,
        "E4_effect_sizes": effect_sizes,
        "E4_within_family_variance": within,
        "no_threshold_selected": True,
        "note": ("The frozen 2*Din gate is reported in every sweep as a DIAGNOSTIC only. No "
                 "sweep selects a replacement threshold, and no alternative metric is compared "
                 "against 2*Din."),
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "E3_E4_mechanism.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten: {outdir / 'E3_E4_mechanism.json'}")

    # ---------------- console summary ----------------
    print("\n=== E3.2 tolerance sweep: overall pass rate and D_eff/Din ===")
    for row in tau_sweep:
        print(f"  tau={row['tau']:.3g} (x{row['tau_ratio']:g}): mean ratio={row['overall_ratio_to_Din']['mean']:.3f} "
              f"pass={row['overall_pass_rate']:.3f} rank1={row['family_ranking'][0]}")
    print("\n=== E3.3 beta sweep: overall D_eff/Din (FROZEN estimator) ===")
    for row in beta_sweep:
        vals = np.concatenate([np.array([all_metrics(s * row["beta"])["frozen_A3"] for s in base[f]]) for f in base])
        print(f"  beta={row['beta']:g}: mean={vals.mean():.3f} pass={(vals/ (din)).__ge__(A3_FACTOR).mean():.3f}")
    print("\n=== E3.4 rho sweep ===")
    for row in rho_sweep:
        means = [row["per_family"][f]["ratio_to_Din"]["mean"] for f in row["family_ranking"]]
        print(f"  rho={row['rho']:.2f}: rank={row['family_ranking']} means={[round(x,3) for x in means]}")
    print("\n=== E3.5 K sweep ===")
    for row in K_sweep:
        means = [row["per_family"][f]["ratio_to_Din"]["mean"] for f in row["family_ranking"]]
        print(f"  K={row['K']:2d}: rank={row['family_ranking']} means={[round(x,3) for x in means]}")
    print("\n=== E4 beta invariance (relative spread of the mean across beta) ===")
    for mname, d in beta_invariance.items():
        print(f"  {mname:26} relative spread = {d['relative_spread']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
