#!/usr/bin/env python3
"""M4-Audit — Reservoir Integrity Audit (real-data half: A2r, A4r, A5r).

The synthetic probe (`reservoir_dynamics_audit.py`) answers A2/A3/A5/A6 with a
seeded AR(1) input, which is the right thing for an integrity question but leaves
two items open that the audit brief explicitly required on **real** data:

  A2r  the hidden-state divergence d_t = ||h_R0 - h_R2|| / (||h_R0|| + eps)
       driven by real e-nose windows rather than a synthetic sequence;
  A4r  the readout ablation X-only / H-only / X+H / X+shuffled-H / zero-H on a
       real specimen-level **validation** split;
  A5r  the effective rank of the real hidden state, which is the mechanism A2/A3
       claim (rank ~= n_channels, independent of N).

Discipline
----------
* Only ``fold.train`` and ``fold.val`` are ever built into features. The script
  reads ``fold.test`` exactly once, to assert it is a distinct object that it does
  NOT use.
* No training run, no protocol file touched, no test-split scoring.
* Everything here is a *measurement*, not a deliverable experiment result: the
  ablation uses one seed and is labelled as such.

Usage
-----
    python ops/audit/reservoir_realdata_audit.py --dataset d2_beef_uncontrolled
    python ops/audit/reservoir_realdata_audit.py --dataset d3_rainbow_trout \
        --split-strategy group_kfold --folds 0 1 2
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
if str(REPO_ROOT / "ops" / "audit") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ops" / "audit"))

from connectome.paths import adjacency_path  # noqa: E402
from drososense.connectome_selection import select_nodes  # noqa: E402
from drososense.data.loaders import dataset_config_path, load_dataset  # noqa: E402
from drososense.data.pipeline import build_fold_tensors  # noqa: E402
from drososense.data.splits import make_folds  # noqa: E402
from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    build_topology_family,
    make_shared,
)
from drososense.reservoir.runner import PINNED_KNOBS  # noqa: E402
from reservoir_dynamics_audit import (  # noqa: E402
    RUNNER_REGIME,
    SELECTION_SEED,
    _rel_divergence,
    drive_with_trace,
)

OUT_DIR = REPO_ROOT / "results" / "audit" / "m4_audit"
EPS = 1e-12


def effective_rank(H: np.ndarray) -> dict:
    """Participation ratio of the singular values of the centred state matrix."""
    Hc = H - H.mean(axis=0)
    s = np.linalg.svd(Hc, compute_uv=False)
    return {
        "effective_rank": float((s.sum() ** 2) / (s ** 2).sum()),
        "top1_variance_fraction": float(s[0] ** 2 / (s ** 2).sum()),
        "dims": int(min(H.shape)),
    }


def states_for(model, ws, window_length: int, n_channels: int) -> np.ndarray:
    """Reservoir state (pooling == 'last') for every window of a WindowSet."""
    x = ws.X.reshape(ws.X.shape[0], window_length, n_channels)
    tr = drive_with_trace(
        model._topology.matrix, model._shared.w_in, model._shared.bias, x,
        leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"],
    )
    return tr["h_trace"][:, -1, :]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="d2_beef_uncontrolled")
    ap.add_argument("--split-strategy", default="loso")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="fold ids to aggregate; default = all")
    ap.add_argument("--reservoir-size", type=int, default=250)
    ap.add_argument("--window-length", type=int, default=16)
    ap.add_argument("--normalization", default="n1_pre_l1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz-path", default=None)
    args = ap.parse_args()

    npz = Path(args.npz_path) if args.npz_path else adjacency_path()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_config_path(args.dataset))
    n_channels = len(dataset.schema.feature_columns)
    folds = make_folds(dataset.specimens(), args.split_strategy,
                       seed=args.seed, n_splits=args.n_splits)
    if args.folds:
        folds = [f for f in folds if f.fold_id in set(args.folds)]

    sel = select_nodes(npz, args.reservoir_size, SELECTION_SEED)
    family = build_topology_family(
        str(npz), seed=args.seed, n_channels=n_channels,
        normalization=args.normalization, node_indices=sel.node_indices,
        params={**RUNNER_REGIME, "reservoir_size": int(sel.node_indices.size)},
    )
    R0 = family["R0_real_fly"]._topology

    print("=" * 78)
    print("M4-Audit real-data half (A2r / A4r / A5r)")
    print("=" * 78)
    print(f"  dataset={args.dataset}  channels={n_channels}  "
          f"split={args.split_strategy}  folds={[f.fold_id for f in folds]}")
    print(f"  N={R0.n_nodes}  M={R0.matrix.nnz}  normalization={args.normalization}")
    print(f"  regime: {RUNNER_REGIME}")

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    per_fold = []
    arms_all: dict[str, list[dict]] = {}
    div_all: list[float] = []
    rank_all: list[float] = []

    for fold in folds:
        artifact_dir = (REPO_ROOT / "results" / "audit" / "_artifacts"
                        / "realdata" / args.dataset / f"w{args.window_length}_seed{args.seed:02d}_fold{fold.fold_id:02d}")
        t = build_fold_tensors(dataset, fold, args.window_length, artifact_dir)
        train, val = t.train, t.val
        # discipline: test exists and is a distinct object; we never build features from it
        assert t.test is not None and t.test.X is not t.train.X

        # ---- A2r: state divergence on REAL input, R0 vs R2 and R0 vs R3 ----
        tr_t = train.X.reshape(train.X.shape[0], args.window_length, n_channels)
        v_t = val.X.reshape(val.X.shape[0], args.window_length, n_channels)
        traces = {}
        for fid in ("R0_real_fly", "R2_degree_rewired", "R3_random_sparse"):
            m = family[fid]
            traces[fid] = drive_with_trace(
                m._topology.matrix, m._shared.w_in, m._shared.bias, v_t,
                leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"],
            )
        d_r0r2 = _rel_divergence(traces["R0_real_fly"]["h_trace"],
                                 traces["R2_degree_rewired"]["h_trace"])
        d_r0r3 = _rel_divergence(traces["R0_real_fly"]["h_trace"],
                                 traces["R3_random_sparse"]["h_trace"])
        zero = drive_with_trace(sp.csr_matrix(R0.matrix.shape), family["R0_real_fly"]._shared.w_in,
                                family["R0_real_fly"]._shared.bias, v_t,
                                leak=RUNNER_REGIME["leak"], gain=RUNNER_REGIME["gain"])
        d_zero = _rel_divergence(traces["R0_real_fly"]["h_trace"], zero["h_trace"])
        div_all.append({
            "fold_id": fold.fold_id,
            "d_R0_R2_final": float(d_r0r2[:, -1].mean()),
            "d_R0_R2_mean": float(d_r0r2.mean()),
            "d_R0_R3_final": float(d_r0r3[:, -1].mean()),
            "d_R0_zero_final": float(d_zero[:, -1].mean()),
        })

        # ---- A5r: effective rank of the real hidden states ----
        H_tr = states_for(family["R0_real_fly"], train, args.window_length, n_channels)
        H_va = states_for(family["R0_real_fly"], val, args.window_length, n_channels)
        rk = effective_rank(np.vstack([H_tr, H_va]))
        rk["fold_id"] = fold.fold_id
        rank_all.append(rk)

        # ---- A4r: readout ablation, train+val only ----
        X_tr, X_va = train.flat(), val.flat()
        y_tr, y_va = train.y_class, val.y_class
        rng = np.random.default_rng(0)
        Hs_tr, Hs_va = H_tr[rng.permutation(H_tr.shape[0])], H_va[rng.permutation(H_va.shape[0])]

        def score(F_tr, F_va, task: str) -> dict:
            out = {}
            if task == "classification":
                sc = StandardScaler().fit(F_tr)
                clf = LogisticRegression(C=1.0, max_iter=5000, random_state=0)
                clf.fit(sc.transform(F_tr), y_tr.astype(int))
                pred = clf.predict(sc.transform(F_va))
                out["macro_f1"] = float(f1_score(y_va, pred, average="macro",
                                                 labels=[0, 1, 2, 3], zero_division=0))
                out["accuracy"] = float(accuracy_score(y_va, pred))
                counts = np.bincount(y_va.astype(int), minlength=4)
                out["majority_accuracy"] = float(counts.max() / counts.sum())
            else:
                sc = StandardScaler().fit(F_tr)
                rg = Ridge(alpha=1.0).fit(sc.transform(F_tr), train.y_reg)
                p = rg.predict(sc.transform(F_va))
                ss_res = float(((val.y_reg - p) ** 2).sum())
                ss_tot = float(((val.y_reg - val.y_reg.mean()) ** 2).sum())
                out["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
                out["mae"] = float(np.abs(val.y_reg - p).mean())
            return out

        arms = {"X_only": (X_tr, X_va), "H_only": (H_tr, H_va)}
        arms["X_plus_H"] = (np.hstack([X_tr, H_tr]), np.hstack([X_va, H_va]))
        arms["X_plus_shuffled_H"] = (np.hstack([X_tr, Hs_tr]), np.hstack([X_va, Hs_va]))
        arms["zero_H"] = (np.hstack([X_tr, np.zeros_like(H_tr)]),
                          np.hstack([X_va, np.zeros_like(H_va)]))
        row = {"fold_id": fold.fold_id, "n_train": int(X_tr.shape[0]),
               "n_val": int(X_va.shape[0]),
               "val_class_support": {str(c): int((y_va == c).sum()) for c in np.unique(y_va)}}
        for name, (a, b) in arms.items():
            row[name] = score(a, b, "classification")
            arms_all.setdefault(name, []).append(row[name])
        per_fold.append(row)

        print(f"\n  fold {fold.fold_id}: train={row['n_train']} val={row['n_val']} "
              f"val_support={row['val_class_support']}")
        print(f"    A2r  d(R0,R2) final={div_all[-1]['d_R0_R2_final']:.6f}  "
              f"d(R0,R3)={div_all[-1]['d_R0_R3_final']:.6f}  "
              f"d(R0,A:=0)={div_all[-1]['d_R0_zero_final']:.6f}")
        print(f"    A5r  effective rank of H = {rk['effective_rank']:.3f} "
              f"(channels={n_channels}), top1 var={rk['top1_variance_fraction']:.4f}")
        print(f"    A4r  " + "  ".join(
            f"{n}:F1={row[n]['macro_f1']:.4f}/acc={row[n]['accuracy']:.4f}"
            for n in ("X_only", "H_only", "X_plus_H", "X_plus_shuffled_H")))
        print(f"         majority-class accuracy = {row['X_only']['majority_accuracy']:.4f}")

    def agg(name: str, key: str) -> float:
        vals = [a[key] for a in arms_all[name] if a.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "audit": "A2r/A4r/A5r",
        "dataset": args.dataset,
        "split": f"{args.split_strategy} seed={args.seed} folds={[f.fold_id for f in folds]}",
        "channels": n_channels,
        "reservoir_N": int(R0.n_nodes),
        "reservoir_M": int(R0.matrix.nnz),
        "normalization": args.normalization,
        "regime": RUNNER_REGIME,
        "test_split_read": False,
        "per_fold": per_fold,
        "A2r_divergence": div_all,
        "A5r_effective_rank": rank_all,
        "A4r_arms_mean": {
            name: {"macro_f1": agg(name, "macro_f1"), "accuracy": agg(name, "accuracy")}
            for name in arms_all
        },
        "A4r_majority_accuracy_mean": float(np.mean(
            [a["majority_accuracy"] for a in arms_all["X_only"]])),
        "A4r_deltas": {
            "H_only_minus_X_only": agg("H_only", "macro_f1") - agg("X_only", "macro_f1"),
            "X_plus_H_minus_X_plus_shuffled_H": (
                agg("X_plus_H", "macro_f1") - agg("X_plus_shuffled_H", "macro_f1")),
            "X_plus_H_minus_zero_H": agg("X_plus_H", "macro_f1") - agg("zero_H", "macro_f1"),
        },
        "A2r_means": {
            "d_R0_R2_final": float(np.mean([d["d_R0_R2_final"] for d in div_all])),
            "d_R0_R3_final": float(np.mean([d["d_R0_R3_final"] for d in div_all])),
            "d_R0_zero_final": float(np.mean([d["d_R0_zero_final"] for d in div_all])),
        },
        "A5r_effective_rank_mean": float(np.mean([r["effective_rank"] for r in rank_all])),
    }

    out_path = OUT_DIR / f"A2r_A4r_A5r_{args.dataset}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(per_fold)} folds — {args.dataset}")
    print("=" * 78)
    print(f"  A2r  d(R0,R2)={summary['A2r_means']['d_R0_R2_final']:.6f}   "
          f"d(R0,R3)={summary['A2r_means']['d_R0_R3_final']:.6f}   "
          f"d(R0,A:=0)={summary['A2r_means']['d_R0_zero_final']:.6f}")
    print(f"  A5r  effective rank of H = {summary['A5r_effective_rank_mean']:.3f} "
          f"vs channels = {n_channels}")
    print(f"  A4r  majority-class baseline accuracy = "
          f"{summary['A4r_majority_accuracy_mean']:.4f}")
    print(f"  {'arm':22s} {'macro_f1':>9s} {'accuracy':>9s}")
    for name, v in summary["A4r_arms_mean"].items():
        print(f"  {name:22s} {v['macro_f1']:9.4f} {v['accuracy']:9.4f}")
    print(f"\n  H_only - X_only ................. "
          f"{summary['A4r_deltas']['H_only_minus_X_only']:+.4f}")
    print(f"  (X+H) - (X+shuffled H) .......... "
          f"{summary['A4r_deltas']['X_plus_H_minus_X_plus_shuffled_H']:+.4f}")
    print(f"  (X+H) - zero_H .................. "
          f"{summary['A4r_deltas']['X_plus_H_minus_zero_H']:+.4f}")
    print(f"\nEvidence written to {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
