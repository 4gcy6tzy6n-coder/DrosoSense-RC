#!/usr/bin/env python
"""E7 — A4 retirement replication: does the preregistered A4 observable show reproducible
directional validity under repeated calibration?

READ-ONLY. Changes nothing: the frozen A4 definition, the A4 threshold, the ladder, the
Stage-1 logic and every prior outcome are untouched. No A4b is defined, no threshold is
modified, and no replacement observable is searched for.

The frozen A4 (recovered verbatim, not re-derived)
--------------------------------------------------
    M_rec    = memory_metric(states | driven by A)        max_k |corr(h_t, u_{t-k})|,
                                                          k in {1,4,8,16}, washout 16
    M_A:=0   = memory_metric(states | driven by A := 0)   the SAME probe, zero recurrence
    A4       = (M_rec - M_A:=0) / M_rec                   0.0 when M_rec <= 0
    threshold A4_MIN = 0.20 (unchanged, reported not applied)

Calibration families: the ORIGINAL C1, C2, C3, C4, C6, C7. **C5 is excluded deliberately**
(its base spectral radius is 0, so the recurrence-strength knob does not exist for it); no
family was added to chase a prettier result.

Recurrence ladder (frozen): rho in {0.00, 0.25, 0.50, 0.75, 0.95}. The knob is the spectral
radius of the SAME wiring with the SAME input mapping, and rho = 0 reproduces the A := 0
control, so the ladder contains its own control.

30 independent probe seeds per family x ladder level (the frozen protocol had 5).

Run:
    PYTHONPATH=. python ops/audit/resaudit_e7_a4_replication.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from resaudit import toys  # noqa: E402
from resaudit._kernels import MEMORY_LAGS, memory_metric, reservoir_drive  # noqa: E402
from resaudit.a4_validity import (  # noqa: E402
    RECURRENCE_LADDER,
    _RHO_FLOOR,
    _observable,
)
from resaudit.battery import (  # noqa: E402
    PROBE_GAIN,
    PROBE_LEAK,
    PROBE_WASHOUT,
    probe_input_with_seed,
    scale_to_spectral_radius,
    spectral_radius_of,
)
from resaudit.criteria import A4_MIN  # noqa: E402

DEFAULT_OUT = REPO / "results/audit/resaudit_a4_replication"
N_SEEDS = 30
N_BOOT = 2000

#: The original calibration families, C5 EXCLUDED by design (documented in the report).
FAMILY_SPECS = (
    ("C1", "ring+chords (k=2, 30 chords)", lambda: toys.ring_with_chords(n=60, k=2, n_chords=30, din=4, seed=5)),
    ("C2", "coprime cycles (2,3,5,7,11)", lambda: toys.two_coprime_cycles()),
    ("C3", "single 2-cycle", lambda: toys.hairline_fail_cycle()),
    ("C4", "recurrent graph, zero input", lambda: toys.zero_input_graph()),
    ("C6", "20-cycle (k_out_ring k=1)", lambda: toys.k_out_ring(n=20, k=1)),
    ("C7", "20-cycle, reversal permutation", None),   # built from C6's wiring below
)


def build_families() -> list[dict]:
    out = []
    for fid, label, fn in FAMILY_SPECS:
        if fid == "C7":
            A0, B0, _ = toys.k_out_ring(n=20, k=1)
            perm = (-np.arange(A0.shape[0])) % A0.shape[0]
            A = sp.csr_matrix(A0[perm][:, perm])
            B = np.asarray(B0, dtype=np.float64)
        else:
            A, B, _ = fn()
        A = sp.csr_matrix(A)
        B = np.asarray(B, dtype=np.float64)
        out.append({
            "family": fid, "label": label, "A": A, "B": B,
            "N": int(A.shape[0]), "E": int(A.nnz), "Din": int(B.shape[1]),
            "rho0_unscaled": float(spectral_radius_of(A)),
            "self_loops": int(np.count_nonzero(A.diagonal() != 0)),
            "ladder_applicable": bool(spectral_radius_of(A) > _RHO_FLOOR),
        })
    return out


def a4_cell(A: sp.spmatrix, B: np.ndarray, rho: float, seed: int) -> dict:
    """One (family, rho, seed) cell: the exact frozen A4 with its intermediates."""
    n = A.shape[0]
    scaled = scale_to_spectral_radius(A, float(rho)) if float(rho) > 0 else sp.csr_matrix((n, n))
    X = probe_input_with_seed(B.shape[1], seed)
    W_in = sp.csr_matrix(B)
    bias = np.zeros(n, dtype=np.float64)
    driven = reservoir_drive(scaled, W_in, bias, X, gain=PROBE_GAIN, leak=PROBE_LEAK)
    ff = reservoir_drive(sp.csr_matrix((n, n), dtype=np.float64), W_in, bias, X,
                         gain=PROBE_GAIN, leak=PROBE_LEAK)
    m_rec = float(memory_metric(driven.states, X, discard_washout=PROBE_WASHOUT))
    m_ff = float(memory_metric(ff.states, X, discard_washout=PROBE_WASHOUT))
    a4 = float(_observable(m_rec, m_ff))
    return {"rho": float(rho), "seed": int(seed), "M_recurrent": m_rec,
            "M_A0": m_ff, "A4": a4, "A4_difference": m_rec - m_ff,
            "passes_A4_MIN": bool(a4 >= A4_MIN)}


def rank_corr(xs, ys) -> float:
    def rk(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort"); r = np.empty(v.size, float)
        r[o] = np.arange(v.size, dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rk(xs), rk(ys)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci(xs, ys, *, n_boot=N_BOOT, seed=0) -> dict:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    rho = rank_corr(xs, ys)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xs.size, size=(n_boot, xs.size))
    vals = np.array([rank_corr(xs[i], ys[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"spearman": rho, "ci_low": float("nan"), "ci_high": float("nan")}
    return {"spearman": rho, "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args(argv)

    fams = build_families()
    n_seeds = int(args.seeds)
    # frozen family seeds plus n_seeds-5 extra, all deterministic
    seed_list = [20260923, 11, 29, 47, 83] + [101, 103, 107, 109, 113, 127, 131, 137, 139,
                                             149, 151, 157, 163, 167, 173, 179, 181, 191,
                                             193, 197, 199, 211, 223, 227, 229]
    seed_list = seed_list[:n_seeds]
    assert len(seed_list) == n_seeds, f"need {n_seeds} seeds"

    print(f"E7 A4 replication: {len(fams)} families x {len(RECURRENCE_LADDER)} rho levels x "
          f"{n_seeds} seeds = {len(fams)*len(RECURRENCE_LADDER)*n_seeds} cells")
    print(f"frozen A4 = (M_rec - M_A:=0)/M_rec ; A4_MIN = {A4_MIN} (reported, not applied)")

    t0 = time.monotonic()
    cells: list[dict] = []
    for f in fams:
        if not f["ladder_applicable"]:
            print(f"  {f['family']}: ladder NOT APPLICABLE (rho0={f['rho0_unscaled']:.3g})")
            continue
        for rho in RECURRENCE_LADDER:
            for s in seed_list:
                c = a4_cell(f["A"], f["B"], rho, s)
                c["family"] = f["family"]
                cells.append(c)
        print(f"  {f['family']}: {len(RECURRENCE_LADDER)*n_seeds} cells done "
              f"({time.monotonic()-t0:.1f}s)")
    print(f"all cells in {time.monotonic()-t0:.1f}s")

    ladder = list(RECURRENCE_LADDER)
    per_family: dict[str, dict] = {}
    for f in fams:
        fid = f["family"]
        if not f["ladder_applicable"]:
            per_family[fid] = {"ladder_applicable": False,
                               "reason": f"base spectral radius {f['rho0_unscaled']:.3g}; "
                                         "no recurrence-strength knob exists"}
            continue
        levels = {}
        for rho in ladder:
            v = np.array([c["A4"] for c in cells if c["family"] == fid and c["rho"] == rho])
            mrec = np.array([c["M_recurrent"] for c in cells if c["family"] == fid and c["rho"] == rho])
            ma0 = np.array([c["M_A0"] for c in cells if c["family"] == fid and c["rho"] == rho])
            levels[f"{rho}"] = {
                "rho": float(rho), "n": int(v.size),
                "A4_mean": float(v.mean()), "A4_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "A4_median": float(np.median(v)),
                "A4_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "frac_A4_gt_0": float((v > 0).mean()),
                "frac_A4_lt_0": float((v < 0).mean()),
                "frac_A4_eq_0": float((v == 0).mean()),
                "M_recurrent_mean": float(mrec.mean()), "M_A0_mean": float(ma0.mean()),
                "passes_A4_MIN_frac": float((v >= A4_MIN).mean()),
            }
        observed = np.array([levels[f"{r}"]["rho"] for r in ladder])
        means = np.array([levels[f"{r}"]["A4_mean"] for r in ladder])
        # monotonicity violations against the DECLARED expectation (recurrence stronger -> A4 higher)
        viol = int(sum(1 for i in range(len(means) - 1) if means[i + 1] < means[i]))
        # use per-seed values, not just the level means, for the correlation
        rho_per_seed = np.array([c["rho"] for c in cells if c["family"] == fid])
        a4_per_seed = np.array([c["A4"] for c in cells if c["family"] == fid])
        ci = boot_ci(rho_per_seed, a4_per_seed, seed=hash(fid) % 10_000)
        allpos = bool(np.all(means > 0))
        directional = bool(ci["spearman"] > 0 and ci["ci_low"] > 0 and allpos)
        per_family[fid] = {
            "ladder_applicable": True,
            "levels": levels,
            "spearman_rho_vs_A4": ci["spearman"],
            "spearman_ci95": [ci["ci_low"], ci["ci_high"]],
            "expected_direction": "+",
            "monotonicity_violations": viol,
            "n_monotonicity_checks": len(means) - 1,
            "all_level_means_positive": allpos,
            "A4_positive_fraction_overall": float((a4_per_seed > 0).mean()),
            "A4_negative_fraction_overall": float((a4_per_seed < 0).mean()),
            "directional_validity": "PASS" if directional else "FAIL",
        }
        print(f"  {fid}: spearman={ci['spearman']:+.3f} "
              f"CI=[{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}] "
              f"viol={viol}/{len(means)-1} fracA4>0={per_family[fid]['A4_positive_fraction_overall']:.2f} "
              f"-> {per_family[fid]['directional_validity']}")

    applicable = [fid for fid, d in per_family.items() if d.get("ladder_applicable")]
    n_pass = sum(1 for fid in applicable if per_family[fid]["directional_validity"] == "PASS")

    spec = [{
        "family": f["family"], "construction_rule": f["label"],
        "recurrence_control_parameter": "spectral radius of the same wiring+input (rho)",
        "N": f["N"], "E": f["E"], "Din": f["Din"],
        "weight_rule": "unit weights as constructed by the toy builder",
        "self_loops": f["self_loops"],
        "rho_convention": "scaled to each ladder level; rho=0 reproduces A:=0",
        "input_mapping": "the toy family's own B, never rescaled across rho",
        "rho0_unscaled": f["rho0_unscaled"],
        "ladder_applicable": f["ladder_applicable"],
        "n_seeds": n_seeds,
    } for f in fams]

    report = {
        "report": "E7 A4 retirement replication",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_definition": {
            "M_recurrent": "memory_metric(states driven by A): max_k |corr(h_t, u_{t-k})|, "
                           f"k in {list(MEMORY_LAGS)}, washout {PROBE_WASHOUT}",
            "M_A0": "the SAME probe driven by A := 0 (zero recurrence)",
            "A4": "(M_recurrent - M_A:=0) / M_recurrent; 0.0 when M_recurrent <= 0",
            "A4_MIN": A4_MIN, "A4_MIN_note": "unchanged; reported, never applied as a filter",
            "A4_status": "descriptive only (retired as a gate by the construct-validity audit)",
            "dynamics": {"leak": PROBE_LEAK, "gain": PROBE_GAIN},
        },
        "family_specification": spec,
        "C5_excluded": ("C5 (nilpotent chain) is excluded because its base spectral radius is 0, "
                        "so no recurrence-strength knob exists for it. No family was added."),
        "recurrence_ladder": list(RECURRENCE_LADDER),
        "n_seeds": n_seeds, "seeds": seed_list, "n_boot": N_BOOT,
        "per_family": per_family,
        "cells": cells,
        "summary": {
            "n_ladder_applicable": len(applicable),
            "n_directional_pass": n_pass,
            "directional_validity": f"{n_pass}/{len(applicable)}",
        },
        "no_replacement_observable": True,
        "note": ("Replication of the frozen A4 under 30 seeds per cell. No A4b is defined, the "
                 "threshold is untouched, and no alternative observable is searched for."),
    }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "E7_a4_replication.json").write_text(json.dumps(report, indent=1, default=str),
                                                   encoding="utf-8")

    # compact paper-ready CSV: one row per family x ladder level
    with (outdir / "E7_a4_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "rho", "n", "A4_mean", "A4_sd", "A4_median",
                    "A4_ci_lo", "A4_ci_hi", "frac_A4_gt_0", "frac_A4_lt_0",
                    "M_recurrent_mean", "M_A0_mean", "passes_A4_MIN_frac",
                    "spearman_rho_vs_A4", "spearman_ci_lo", "spearman_ci_hi",
                    "monotonicity_violations", "expected_direction", "directional_validity"])
        for f in fams:
            fid = f["family"]; d = per_family[fid]
            if not d.get("ladder_applicable"):
                w.writerow([fid, "n/a", 0] + [""] * 16)
                continue
            for rho in ladder:
                L = d["levels"][f"{rho}"]
                w.writerow([fid, L["rho"], L["n"], f"{L['A4_mean']:.6f}", f"{L['A4_sd']:.6f}",
                            f"{L['A4_median']:.6f}", f"{L['A4_ci95'][0]:.6f}", f"{L['A4_ci95'][1]:.6f}",
                            f"{L['frac_A4_gt_0']:.4f}", f"{L['frac_A4_lt_0']:.4f}",
                            f"{L['M_recurrent_mean']:.6f}", f"{L['M_A0_mean']:.6f}",
                            f"{L['passes_A4_MIN_frac']:.4f}",
                            f"{d['spearman_rho_vs_A4']:.6f}", f"{d['spearman_ci95'][0]:.6f}",
                            f"{d['spearman_ci95'][1]:.6f}", d["monotonicity_violations"],
                            d["expected_direction"], d["directional_validity"]])
    print(f"\nwritten: {outdir/'E7_a4_replication.json'}")
    print(f"written: {outdir/'E7_a4_summary.csv'}")
    print(f"\nDIRECTIONAL VALIDITY: {n_pass}/{len(applicable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
