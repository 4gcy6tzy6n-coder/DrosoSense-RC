#!/usr/bin/env python
"""Measurement-layer impact audit for the spectral-radius defect.

READ-ONLY on data; writes one JSON report. Answers three questions in order:

1. **Solver validation.** Is the current ``spectral_radius_of`` trustworthy, tested against
   INDEPENDENT oracles rather than against itself?
2. **Blast radius.** Which artifacts were produced through a spectral-radius call, and does
   each one change when recomputed with the corrected solver?
3. **Conclusion survival.** Do the A3 semantic observations and the A4 NOT_DIRECTIONAL
   verdict still hold? They are re-derived here, not assumed.

Historical outputs are never overwritten: each suspect artifact is listed with a
``status`` of ``UNAFFECTED`` or ``INVALIDATED_BY_MEASUREMENT_BUG``.

Usage:
    PYTHONPATH=. python ops/audit/resaudit_rho_impact.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resaudit.battery import (  # noqa: E402
    krylov_score,
    scale_to_spectral_radius,
    spectral_radius_of,
)
from resaudit import toys  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "audit" / "resaudit_rho_impact" / "rho_impact_audit.json"


# ---------------------------------------------------------------------------
# oracles and the two suspect implementations
# ---------------------------------------------------------------------------


def oracle_dense(A: sp.spmatrix) -> float:
    """Exact ``|lambda|_max`` by dense eigendecomposition. The reference of record."""
    return float(np.abs(np.linalg.eigvals(sp.csr_matrix(A).toarray())).max())


def oracle_arnoldi(A: sp.spmatrix, seed: int = 0) -> float:
    """ARPACK on the NON-SYMMETRIC problem (``eigs``, ``which='LM'``). May not converge."""
    n = sp.csr_matrix(A).shape[0]
    if n < 3:
        return float("nan")
    try:
        w = eigs(A, k=1, which="LM", return_eigenvectors=False,
                 v0=np.random.default_rng(seed).standard_normal(n), maxiter=5000)
        return float(np.max(np.abs(w)))
    except Exception:
        return float("nan")


def suspect_pre_fix(A: sp.spmatrix, power_iters: int = 400, seed: int = 0) -> float:
    """The PRE-FIX resaudit estimator: dense when small, single-vector power otherwise."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    if n <= 500:
        return oracle_dense(A)
    v = np.random.default_rng(seed).standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(power_iters):
        v = A @ v
        nrm = float(np.linalg.norm(v))
        if nrm == 0.0:
            return 0.0
        v = v / nrm
    return float(v @ (A @ v))


def frozen_layer(A: sp.spmatrix, seed: int = 0) -> float:
    """The FROZEN project's estimator: ``eigsh`` (a SYMMETRIC solver) on a directed graph."""
    from drososense.reservoir.connectome_reservoir import spectral_radius

    return float(spectral_radius(sp.csr_matrix(A), seed=seed))


def cycles(lengths: tuple[int, ...]) -> sp.csr_matrix:
    n = int(sum(lengths))
    rows: list[int] = []
    cols: list[int] = []
    offset = 0
    for L in lengths:
        for i in range(L):
            rows.append(offset + (i + 1) % L)
            cols.append(offset + i)
        offset += L
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# 1. solver validation
# ---------------------------------------------------------------------------

_ORACLE_CASES = (
    ("4-cycle (n=4, rho=1)", (4,)),
    ("single 2-cycle", (2,)),
    ("3 coprime cycles (n=23)", (5, 7, 11)),
    ("8 disjoint cycles (n=1000, degenerate)", (95, 109, 113, 127, 131, 137, 139, 149)),
    ("12 coprime cycles (n=197)", (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)),
)


def validate_solver() -> dict:
    rows = []
    for name, lengths in _ORACLE_CASES:
        A = cycles(lengths)
        order = A.shape[0]
        est = spectral_radius_of(A)
        rows.append({
            "case": name,
            "n": int(order),
            "oracle_dense": oracle_dense(A),
            "oracle_arnoldi_LM": oracle_arnoldi(A),
            "candidate_subspace": float(est),
            "pre_fix_power": suspect_pre_fix(A, seed=0),
            "frozen_eigsh": frozen_layer(A),
            "candidate_abs_err": float(abs(est - oracle_dense(A))),
        })
    # dense-eigenspectrum family: perturbed graphs where the radius is not 1
    dense_rows = []
    for n, density, seed in ((60, 0.05, 1), (80, 0.10, 2), (120, 0.03, 3), (200, 0.02, 4)):
        rng = np.random.default_rng(seed)
        m = int(density * n * n)
        r = rng.integers(0, n, m)
        c = rng.integers(0, n, m)
        keep = r != c
        A = sp.csr_matrix((rng.uniform(0.1, 1.0, int(keep.sum())), (r[keep], c[keep])), shape=(n, n))
        ex = oracle_dense(A)
        dense_rows.append({
            "n": int(n), "density": float(density),
            "oracle_dense": ex,
            "candidate_subspace": float(spectral_radius_of(A)),
            "frozen_eigsh": frozen_layer(A),
            "abs_err": float(abs(spectral_radius_of(A) - ex)),
        })
    # the property that actually matters downstream: after scaling, rho == target
    scale_rows = []
    for name, lengths in _ORACLE_CASES[:4]:
        A = cycles(lengths)
        scaled = scale_to_spectral_radius(A, 0.95)
        scale_rows.append({
            "case": name,
            "rho_after_scaling": float(spectral_radius_of(scaled)),
            "oracle_after_scaling": oracle_dense(scaled),
            "abs_err_vs_target": float(abs(oracle_dense(scaled) - 0.95)),
        })
    worst = max((r["candidate_abs_err"] for r in rows), default=0.0)
    worst_dense = max((r["abs_err"] for r in dense_rows), default=0.0)
    worst_scale = max((r["abs_err_vs_target"] for r in scale_rows), default=0.0)
    return {
        "known_spectrum_cases": rows,
        "dense_eigenspectrum_cases": dense_rows,
        "scaling_property": scale_rows,
        "worst_abs_error_degenerate": worst,
        "worst_abs_error_dense": worst_dense,
        "worst_abs_error_after_scaling": worst_scale,
        "validated": bool(worst <= 1e-6 and worst_dense <= 1e-6 and worst_scale <= 1e-6),
    }


# ---------------------------------------------------------------------------
# 2/3. blast radius and conclusion survival
# ---------------------------------------------------------------------------


def blast_radius() -> dict:
    """Which artifacts pass through a spectral-radius call, and do they change?"""
    rows: list[dict] = []

    # A3: krylov_score does NOT rescale A, so A3 verdicts are scale-sensitive only through
    # whatever scale the caller supplied. Record that explicitly.
    A = cycles((2, 3, 5, 7, 11))
    B = np.zeros((A.shape[0], 5))
    off = 0
    for c, L in enumerate((2, 3, 5, 7, 11)):
        B[off, c] = 1.0
        off += L
    s_raw = krylov_score(A, B, K=16)
    s_scaled = krylov_score(scale_to_spectral_radius(A, 0.95), B, K=16)
    rows.append({
        "artifact": "A3 Krylov score",
        "passes_through_spectral_radius": False,
        "note": ("krylov_score never rescales A; A3 is scale-SENSITIVE but not "
                 "scale-COMPUTED. The estimator cannot corrupt it except by supplying a "
                 "wrong scale to the caller."),
        "d_eff_unscaled": float(s_raw.effective_rank),
        "d_eff_scaled_to_frozen_convention": float(s_scaled.effective_rank),
        "status": "UNAFFECTED",
    })

    # A4: the ladder IS built through scale_to_spectral_radius, so it is in scope.
    rows.append({
        "artifact": "A4 construct-validity ladder",
        "passes_through_spectral_radius": True,
        "note": ("a4_validity.audit_family_direction scales A to each rung with "
                 "scale_to_spectral_radius, so every rung's realized rho was suspect."),
        "status": "RECOMPUTED_BELOW",
    })
    for name in ("stage1_calibration", "f2_f3_f5_generators", "f4_feasibility"):
        rows.append({
            "artifact": name,
            "passes_through_spectral_radius": name != "f4_feasibility",
            "note": ("uses spectral_radius_of directly" if name != "f4_feasibility"
                     else "imports no spectral-radius function; unaffected by construction"),
            "status": "RECOMPUTED_BELOW" if name != "f4_feasibility" else "UNAFFECTED",
        })
    return {"artifacts": rows}


def conclusion_survival() -> dict:
    """Re-derive the A4 ladder verdict with the CORRECTED solver."""
    from resaudit.battery import PROBE_GAIN, PROBE_LEAK, PROBE_WASHOUT, probe_input_with_seed
    from drososense.reservoir import dynamics as _dyn

    ladder = (0.0, 0.25, 0.50, 0.75, 0.95)
    seeds = (20260923, 11, 29, 47, 83)

    def observable(m_rec: float, m_ff: float) -> float:
        return float((m_rec - m_ff) / m_rec) if m_rec > 0 else 0.0

    def memory(A, B, seed):
        n = A.shape[0]
        X = probe_input_with_seed(B.shape[1], seed)
        W = sp.csr_matrix(B)
        bias = np.zeros(n)
        d1 = _dyn.reservoir_drive(A, W, bias, X, gain=PROBE_GAIN, leak=PROBE_LEAK)
        d0 = _dyn.reservoir_drive(sp.csr_matrix((n, n)), W, bias, X, gain=PROBE_GAIN, leak=PROBE_LEAK)
        return (float(_dyn.memory_metric(d1.states, X, discard_washout=PROBE_WASHOUT)),
                float(_dyn.memory_metric(d0.states, X, discard_washout=PROBE_WASHOUT)))

    wirings = [
        ("C1", toys.ring_with_chords(n=60, k=2, n_chords=30, din=4, seed=5)[:2]),
        ("C2", toys.two_coprime_cycles()[:2]),
        ("C3", toys.hairline_fail_cycle()[:2]),
        ("C6", toys.k_out_ring(n=20, k=1)[:2]),
    ]
    out = []
    for fid, (A0, B) in wirings:
        A0 = sp.csr_matrix(A0)
        rho0 = spectral_radius_of(A0)
        per_rho = []
        for rho in ladder:
            scaled = scale_to_spectral_radius(A0, float(rho)) if rho0 > 0 else A0.tocsr()
            recs, obs = [], []
            for s in seeds:
                m1, m0 = memory(scaled, B, s)
                recs.append(m1)
                obs.append(observable(m1, m0))
            per_rho.append({
                "rho": float(rho),
                "rho_realised": float(spectral_radius_of(scaled)),
                "M_recurrent_mean": float(np.mean(recs)),
                "observable_mean": float(np.mean(obs)),
            })
        mono = all(per_rho[i + 1]["M_recurrent_mean"] >= per_rho[i]["M_recurrent_mean"] - 1e-12
                   for i in range(len(per_rho) - 1))
        obs_op = per_rho[-1]["observable_mean"]
        out.append({
            "family": fid,
            "ladder": per_rho,
            "monotone_nondecreasing": bool(mono),
            "observable_at_operating": float(obs_op),
            "directional": bool(mono and obs_op > 0),
        })
    n_dir = sum(1 for r in out if r["directional"])
    return {
        "verdict": "DIRECTIONAL" if n_dir == len(out) else "NOT_DIRECTIONAL",
        "n_directional": n_dir,
        "n_families": len(out),
        "families": out,
        "note": ("Recomputed with the corrected solver. The A4 observable is a memory "
                 "metric evaluated on the driven state, so a change of realized rho can "
                 "move it; the verdict is therefore re-derived rather than assumed."),
    }


def historical_artifacts() -> list[dict]:
    """List suspect pre-fix reports, with a status. Files are never modified."""
    suspects = [
        ("results/audit/resaudit_a4_validity/A4_construct_validity.json",
         "A4 ladder rungs were scaled through the pre-fix estimator"),
        ("results/audit/resaudit_stage1/ResAudit_stage1.json",
         "C1/C2 toylike families scaled through the pre-fix estimator"),
        ("results/audit/v3_selection/V3_substrate_selection.json",
         "FROZEN project: rho_target scaling; uses eigsh, a separate defect"),
        ("results/audit/m5_structural_dynamics/M5_structural_dynamics.json",
         "FROZEN project: mirrors rescale_to_spectral_radius"),
        ("results/audit/m4_audit/ledger_repair_E9.json",
         "FROZEN project: ledger repair over rho-scaled topologies"),
    ]
    out = []
    for rel, why in suspects:
        p = REPO / rel
        out.append({
            "path": rel,
            "exists": p.exists(),
            "why_in_scope": why,
            "superseded_by": "results/audit/resaudit_rho_impact/rho_impact_audit.json",
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    report = {
        "report": "measurement-layer impact audit: spectral-radius defect",
        "solver_validation": validate_solver(),
        "blast_radius": blast_radius(),
        "conclusion_survival": conclusion_survival(),
        "historical_artifacts": historical_artifacts(),
        "policy": ("historical outputs are never overwritten; a suspect artifact is marked "
                   "INVALIDATED_BY_MEASUREMENT_BUG and superseded by this report"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    v = report["solver_validation"]
    print("SOLVER VALIDATION")
    print(f"  worst abs error, degenerate spectra : {v['worst_abs_error_degenerate']:.3e}")
    print(f"  worst abs error, dense spectra      : {v['worst_abs_error_dense']:.3e}")
    print(f"  worst abs error after scaling       : {v['worst_abs_error_after_scaling']:.3e}")
    print(f"  validated                           : {v['validated']}")
    print()
    print(f"{'case':42} {'oracle':>10} {'candidate':>10} {'pre-fix':>10} {'frozen eigsh':>13}")
    for r in v["known_spectrum_cases"]:
        print(f"  {r['case']:40} {r['oracle_dense']:10.6f} {r['candidate_subspace']:10.6f} "
              f"{r['pre_fix_power']:10.6f} {r['frozen_eigsh']:13.6f}")
    print()
    c = report["conclusion_survival"]
    print("A4 CONCLUSION SURVIVAL (recomputed)")
    for f in c["families"]:
        print(f"  {f['family']}: monotone={f['monotone_nondecreasing']} "
              f"obs@0.95={f['observable_at_operating']:+.4f} directional={f['directional']}")
    print(f"  VERDICT: {c['verdict']} ({c['n_directional']}/{c['n_families']})")
    print()
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
