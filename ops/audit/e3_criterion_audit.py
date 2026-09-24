#!/usr/bin/env python
"""E3 -- A3 construct-validity audit: calibration controls, Din sweep, K sweep.

Authority: the operator's spec for E3 (2026-09-23). NO change to A3 threshold, family construction,
or input geometry. The three components are fixed before measurement:

  (i)  calibration controls: at least one known-low and one known-high effective-dimension system.
       Verify A3 discriminates in the right direction on the calibration families (which the engine
       was designed against) -- this is the construct-validity check before extending to the real
       F1/F2/F3/F5.
  (ii) Din sweep: F1/F2/F3/F5 at Din in {4, 6, 8, 10, 12}, K=16. Per family, regress D_eff/Din on Din
       and report slope with 95% CI. If slope is close to zero, the width-invariance claim is
       quantified.
  (iii) K sweep: K in {8, 16, 32}, Din=6, F1/F2/F3/F5. Report D_eff stability across K -- this is a
       sensitivity check, NOT a model selection. The threshold (2*Din) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
SYS_PATH = (REPO, str(REPO / "ops" / "audit"))

DINS = (4, 6, 8, 10, 12)
KS = (8, 16, 32)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _scale_to_spectral_radius(A, rho_target=0.95):
    from resaudit.battery import scale_to_spectral_radius
    return scale_to_spectral_radius(A, target=rho_target)


def _load_F1_raw():
    import scipy.sparse as sp
    raw_path = REPO / "results/audit/resaudit_stage1/f1/F1_A_raw.npz"
    z = np.load(raw_path, allow_pickle=True)
    return sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])
    )


def _F1_B_at(din):
    """Reuse the already-materialized F1 wiring (frozen S0) and build only the typed-aligned B at
    the requested Din via the amendment-6 allocation rule.

    For Din in {6, 8} the saved F1_B_Din{din}.npz is used verbatim. For Din in {4, 10, 12} (new
    widths the materializer did not pre-build) we construct B from F1_A_raw.npz's node_ids and
    the node_meta annotation, using build_typed_aligned_mapping with the frozen
    receiving_fraction and density_limit.
    """
    from collections import Counter
    from connectome.cell_types import load_node_classes
    from drososense.connectome_candidates import generate_candidate
    from drososense.reservoir.input_mapping import (CellTypeAnnotation, build_typed_aligned_mapping)
    from resaudit.din_allocation import apportion_din

    p = REPO / "results/audit/resaudit_stage1/f1" / ("F1_B_Din" + str(din) + ".npz")
    if p.exists():
        return np.asarray(np.load(p, allow_pickle=True)["w_in"], dtype=np.float64)

    # Reconstruct the frozen S0 selection so we have the F1 root_ids (the 1000 selected nodes)
    # and the typed annotation. The full raw adjacency (124k x 124k) is needed to run the
    # selection at the historical Din=5 setting; this matches the materializer byte-for-byte.
    meta_path = REPO / "connectome/metadata/olfactory_v1_node_meta.csv"
    raw_meta = np.genfromtxt(str(meta_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    node_idx = np.asarray(raw_meta["node_idx"], dtype=np.int64)
    root_id = np.asarray(raw_meta["root_id"], dtype=np.int64)
    assert node_idx.min() == 0 and node_idx.max() == 124184
    full_root_ids = np.empty(124185, dtype=np.int64); full_root_ids[node_idx] = root_id
    classes = load_node_classes(str(meta_path))
    counts = dict(Counter(classes.values()))
    annotation = CellTypeAnnotation(classes=classes, source="node_meta", per_class_counts=counts)

    # Load the full raw adjacency (124k x 124k) so generate_candidate can run the selection.
    full_raw = REPO.parent / "connectome/adjacency/olfactory_v1.npz"
    raw_z = np.load(full_raw, allow_pickle=True)
    if {"adj_data", "adj_indices", "adj_indptr", "adj_shape"} <= set(raw_z.files):
        full_graph = sp.csr_matrix(
            (raw_z["adj_data"], raw_z["adj_indices"], raw_z["adj_indptr"]),
            shape=tuple(raw_z["adj_shape"]),
        )
    else:
        full_graph = sp.csr_matrix(
            (raw_z["data"], raw_z["indices"], raw_z["indptr"]),
            shape=tuple(raw_z["shape"]),
        )
    cand = generate_candidate("S0", raw_graph=full_graph, root_ids=full_root_ids,
                               annotation=annotation, din=5, target_n=1000, seed=0)

    alloc = apportion_din(din)
    m = build_typed_aligned_mapping(
        cand.root_ids, annotation, din, seed=0,
        din_per_layer=alloc, receiving_fraction=0.80, density_limit=0.16,
    )
    return np.asarray(m.w_in.toarray(), dtype=np.float64)


def _audit_one(spec, k_override=None):
    """Return (criterion_result, k_used, singular_values_top).

    audit_a3 returns (CriterionResult, krylov_diagnostics); we discard the diagnostics. The
    K override uses krylov_score directly because audit_a3 reads A3_KRYLOV_K as a module
    constant and cannot be parameterised without modifying the engine.
    """
    from resaudit.battery import audit_a3
    from resaudit.criteria import CriterionState
    A, B = spec.A, spec.B
    if k_override is not None:
        from collections import namedtuple
        from drososense.substrate_scores import krylov_score
        s = krylov_score(A, B, K=int(k_override))
        eff = s.effective_rank
        threshold = float(2.0 * int(spec.din))
        passed = bool(eff >= threshold)
        _R = namedtuple("_R", ("value", "threshold", "passed", "state"))
        return _R(value=float(eff), threshold=threshold, passed=passed,
                   state=CriterionState.PASS if passed else CriterionState.FAIL),                s.K, list(s.singular_values)
    cri, _diag = audit_a3(spec)
    from drososense.substrate_scores import krylov_score
    s = krylov_score(A, B)
    return cri, s.K, list(s.singular_values)


def _make_families(A, B):
    """Build F1, F2, F3, F5 specs at the given Din."""
    from resaudit.family import FamilySpec
    from resaudit.families import generate_f2, generate_f3, generate_f5

    def f1():
        return FamilySpec(
            family_id="F1", label="Drosophila olfactory connectome (frozen S0)",
            role="negative_biological_control", A=A, B=B, n_nodes=A.shape[0], n_edges=int(A.nnz),
            din=B.shape[1], counterfactual_parent=None, seed=0,
            construction={"kind": "frozen_substrate"},
        )
    f1s = f1()
    f2s, _ = generate_f2(A, B)
    f3s, _ = generate_f3(A, B)
    f5s, _ = generate_f5(A, B)
    return f1s, f2s, f3s, f5s


def component_1_calibration_controls():
    """A3 on resaudit toys: known-low (zero_input_graph) and known-high (two_coprime_cycles).
    Each toy is run at its own declared Din (the toy's B determines the width); K=16. The question
    is whether A3 discriminates in the right direction on the calibration systems, not whether it
    is satisfied at any particular Din."""
    from resaudit.toys import two_coprime_cycles, zero_input_graph
    from resaudit.family import FamilySpec

    out = []
    for label, maker in (("known_high_two_coprime_cycles", two_coprime_cycles),
                           ("known_low_zero_input_graph", zero_input_graph)):
        A, B, meta = maker()
        din = int(B.shape[1])
        spec = FamilySpec(family_id=label, label=label, role="calibration",
                          A=A, B=B, n_nodes=A.shape[0], n_edges=int(A.nnz), din=din,
                          counterfactual_parent=None, seed=0,
                          construction={"kind": "calibration_toy", **meta})
        r, _K, _sv = _audit_one(spec)
        out.append({"family": label, "Din": int(din), "K": _K,
                     "D_eff": float(r.value), "gate": float(r.threshold),
                     "pass": bool(r.passed), "state": r.state.value})
    return out


def component_2_din_sweep():
    """F1/F2/F3/F5 at Din in {4,6,8,10,12}, K=16. Regression slope of D_eff/Din on Din per family."""
    from scipy import stats as sstats
    A = _scale_to_spectral_radius(_load_F1_raw(), 0.95)
    results = {}
    for fam_id, builder in (("F1", lambda B: None), ("F2", None), ("F3", None), ("F5", None)):
        del builder  # unused; we build specs via _make_families below
    per_family = {f: [] for f in ("F1", "F2", "F3", "F5")}
    for din in DINS:
        B = _F1_B_at(din)
        f1, f2, f3, f5 = _make_families(A, B)
        for spec, fam in ((f1, "F1"), (f2, "F2"), (f3, "F3"), (f5, "F5")):
            r, _K, _sv = _audit_one(spec)
            per_family[fam].append({"Din": din, "D_eff": float(r.value),
                                     "gate": float(r.threshold), "pass": bool(r.passed),
                                     "D_eff_over_Din": float(r.value) / din,
                                     "K_used": int(_K)})
    # Regression per family.
    summary = {}
    for fam, pts in per_family.items():
        x = np.array([p["Din"] for p in pts], dtype=float)
        y = np.array([p["D_eff_over_Din"] for p in pts], dtype=float)
        slope, intercept, r, p, se = sstats.linregress(x, y)
        # 95% CI on the slope: slope +/- t_{0.025, n-2} * stderr, n=5 -> df=3, t~3.182
        from scipy.stats import t as tdist
        tcrit = float(tdist.ppf(0.975, len(x) - 2))
        summary[fam] = {
            "Din_values": x.tolist(),
            "D_eff": [p["D_eff"] for p in pts],
            "D_eff_over_Din": y.tolist(),
            "pass_at_each_Din": [p["pass"] for p in pts],
            "regression_slope_per_Din": float(slope),
            "regression_intercept": float(intercept),
            "regression_slope_se": float(se),
            "regression_slope_ci95": [float(slope - tcrit * se), float(slope + tcrit * se)],
            "regression_r": float(r),
            "regression_pvalue": float(p),
            "width_invariant_claim_holds": bool(abs(slope) < 0.05),  # ad-hoc threshold; reported with
                                                                # justification in the findings doc.
        }
    return summary


def component_3_k_sweep():
    """F1/F2/F3/F5 at Din=6, K in {8,16,32}."""
    A = _scale_to_spectral_radius(_load_F1_raw(), 0.95)
    B = _F1_B_at(6)
    f1, f2, f3, f5 = _make_families(A, B)
    families = (("F1", f1), ("F2", f2), ("F3", f3), ("F5", f5))
    summary = {}
    for fam, spec in families:
        summary[fam] = {}
        for K in KS:
            r, used_K, svs = _audit_one(spec, k_override=K)
            summary[fam][f"K={K}"] = {
                "K_used": int(used_K), "Din": 6,
                "D_eff": float(r.value), "gate": float(r.threshold), "pass": bool(r.passed),
                "singular_values_top5": list(svs[:5]),
            }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/audit/resaudit_stage1/e3_criterion_audit.json"))
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== E3 / component 1: calibration controls (Din=6, K=16) ===")
    c1 = component_1_calibration_controls()
    for r in c1:
        print(f"  {r['family']}: D_eff={r['D_eff']:.4f} gate={r['gate']} {r['state']}")

    print("\n=== E3 / component 2: Din sweep (Din in {4,6,8,10,12}, K=16) ===")
    c2 = component_2_din_sweep()
    for fam, summ in c2.items():
        pts = [{"Din": d, "D_eff": e, "pass": p}
               for d, e, p in zip(summ["Din_values"], summ["D_eff"], summ["pass_at_each_Din"])]
        vals = " ".join(f"Din={p['Din']}:{p['D_eff']:.2f}/{'P' if p['pass'] else 'F'}" for p in pts)
        print(f"  {fam}: {vals}")
        print(f"    slope={summ['regression_slope_per_Din']:+.4f}/Din "
              f"CI95=[{summ['regression_slope_ci95'][0]:+.4f},{summ['regression_slope_ci95'][1]:+.4f}]"
              f" r={summ['regression_r']:.3f}")

    print("\n=== E3 / component 3: K sweep (K in {8,16,32}, Din=6) ===")
    c3 = component_3_k_sweep()
    for fam, ks in c3.items():
        for k, r in ks.items():
            print(f"  {fam} {k}: D_eff={r['D_eff']:.4f} gate={r['gate']} {r['pass']}")

    report = {
        "experiment": "E3 A3 construct-validity audit",
        "authority": "operator spec for E3 (2026-09-23); no threshold or construction changes",
        "design": {
            "Din_sweep": list(DINS),
            "K_sweep": list(KS),
            "threshold": "2*Din (unchanged)",
            "K_default": 16,
            "Din_default": 6,
            "rho_target": 0.95,
            "seed_for_F1_selection": 0,
        },
        "calibration_controls": c1,
        "Din_sweep_regression": c2,
        "K_sensitivity": c3,
        "git_head": git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
