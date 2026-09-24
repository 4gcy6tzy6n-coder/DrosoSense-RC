#!/usr/bin/env python
"""Recompute the frozen v3 substrate SELECTION with the corrected spectral-radius estimator.

Answers the one question that governs whether the Stage-1 negative can be frozen:

    Did the defective estimator change WHICH substrate the frozen selection rule chooses?

The distinction this script exists to make:

    current F1 measurement  -- valid, already verified by content hashes
    current F1 selection provenance -- possibly invalid, if a wrong rho reordered candidates

Why the selection can depend on rho at all: `score_matrix` normalises each candidate to a
common `rho_target` using a single global scalar, and `krylov_score` prunes singular values
with an ABSOLUTE tolerance (1e-10). So the scalar changes how many Krylov directions survive
pruning, which changes S1. S1 is therefore NOT scale-invariant -- measured, `D_eff` moved from
1.02 to 13.44 under a 0.5x rescale of the same matrix.

IMPORTANT: the frozen `score_matrix` symmetrises before `eigsh`, i.e. it estimates the radius
of `(A + A^T)/2`, which is a legitimate symmetric problem. It is therefore NOT affected by the
directed-`eigsh` defect found in `connectome_reservoir.spectral_radius`. This script compares
the frozen SYMMETRISED radius against the corrected directed radius to quantify the
convention difference rather than assume either.

READ-ONLY with respect to data. Writes one JSON. Never overwrites the frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as _spla

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from connectome.cell_types import load_node_classes  # noqa: E402
from drososense.connectome_candidates import (  # noqa: E402
    CANDIDATE_IDS,
    build_candidate_input,
    generate_candidate,
)
from drososense.substrate_scores import eigenmode_score, krylov_score  # noqa: E402
from resaudit.battery import spectral_radius_of  # noqa: E402

DEFAULT_DATA_ROOT = REPO.parent
DEFAULT_NODE_META = REPO / "connectome/metadata/olfactory_v1_node_meta.csv"
DEFAULT_OUT = REPO / "results/audit/resaudit_v3_reselection/V3_reselection.json"

SELECTION_GATE_FACTOR = 2.0


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def frozen_symmetrised_rho(matrix: sp.spmatrix) -> float:
    """The frozen `score_matrix` convention: radius of the SYMMETRISED matrix."""
    sym = ((matrix + matrix.T) * 0.5).tocsc()
    if sym.shape[0] < 3:
        return float(np.abs(np.linalg.eigvals(matrix.toarray())).max())
    k = min(6, sym.shape[0] - 2)
    try:
        vals = _spla.eigsh(sym, k=k, which="LM", return_eigenvectors=False)
        return float(np.abs(vals).max())
    except Exception:
        return float(np.abs(np.linalg.eigvals(matrix.toarray())).max())


def score_matrix(matrix: sp.csr_matrix, *, rho_target: float, estimator) -> sp.csr_matrix:
    rho = estimator(matrix)
    if rho <= 1e-12:
        return matrix.copy()
    return (matrix * (rho_target / rho)).tocsr()


def score_candidate(adjacency, annotation, din, rho_target, seed, estimator):
    A = score_matrix(adjacency, rho_target=rho_target, estimator=estimator)
    B = build_candidate_input.__wrapped__ if False else None
    return A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--node-meta", default=str(DEFAULT_NODE_META))
    ap.add_argument("--din", type=int, default=5)
    ap.add_argument("--target-n", type=int, default=1000)
    ap.add_argument("--rho-target", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    node_meta = Path(args.node_meta)
    npz = data_root / "connectome/adjacency/olfactory_v1.npz"
    if not npz.exists():
        raise SystemExit(f"raw connectome not found at {npz}")

    # reuse the SAME loader the successful F1 materialization used, so the two agree
    sys.path.insert(0, str(REPO / "ops/audit"))
    from resaudit_materialize_f1 import load_raw  # type: ignore

    raw, node_ids, annotation, layout, npz_path = load_raw(data_root, node_meta)
    print(f"loader layout: {layout}")
    print(f"raw adjacency: shape={raw.shape} nnz={raw.nnz}")

    rows = []
    for cid in CANDIDATE_IDS:
        cand = generate_candidate(cid, raw_graph=raw, root_ids=node_ids, annotation=annotation,
                                  din=args.din, target_n=args.target_n, seed=args.seed)
        adj = sp.csr_matrix(cand.adjacency)
        mapping = build_candidate_input(cand, annotation, args.din, seed=args.seed)
        B = mapping.w_in.toarray() if sp.issparse(mapping.w_in) else np.asarray(mapping.w_in)

        rho_sym = frozen_symmetrised_rho(adj)
        rho_dir = spectral_radius_of(adj)
        A_frozen = score_matrix(adj, rho_target=args.rho_target, estimator=frozen_symmetrised_rho)
        A_fixed_old_conv = score_matrix(adj, rho_target=args.rho_target, estimator=spectral_radius_of)
        # the convention the frozen artifact actually used
        A_sym = A_frozen
        # corrected DIRECTED radius, same target
        A_dir = A_fixed_old_conv

        s1_sym = krylov_score(A_sym, B, K=16)
        s1_dir = krylov_score(A_dir, B, K=16)
        s2 = eigenmode_score(A_sym, B)
        rows.append({
            "candidate_id": cid,
            "n_nodes": int(adj.shape[0]),
            "n_edges": int(adj.nnz),
            "rho_symmetrised": rho_sym,
            "rho_directed_corrected": rho_dir,
            "rho_ratio_directed_over_sym": (rho_dir / rho_sym) if rho_sym else float("nan"),
            "S1_normalised_with_sym_rho": float(s1_sym.effective_rank),
            "S1_normalised_with_directed_rho": float(s1_dir.effective_rank),
            "S2_participation_ratio": float(s2.participation_ratio),
            "gate": SELECTION_GATE_FACTOR * args.din,
        })
        print(f"  {cid}: N={adj.shape[0]} M={adj.nnz} rho_sym={rho_sym:.4f} "
              f"rho_dir={rho_dir:.4f} S1_sym={s1_sym.effective_rank:.4f} "
              f"S1_dir={s1_dir.effective_rank:.4f}")

    def select(rows, key):
        eligible = [r for r in rows if r[key] >= r["gate"]]
        if not eligible:
            return {"eligible": [], "selected": None, "stop_loss": True,
                    "reason": "no candidate reached S1 >= 2*Din; stop-loss"}
        ordered = sorted(eligible, key=lambda r: (-r[key], -r["S2_participation_ratio"], r["n_nodes"]))
        return {"eligible": [r["candidate_id"] for r in ordered],
                "selected": ordered[0]["candidate_id"], "stop_loss": False,
                "reason": "max S1, ties by S2 participation ratio then smaller N"}

    sel_sym = select(rows, "S1_normalised_with_sym_rho")
    sel_dir = select(rows, "S1_normalised_with_directed_rho")
    report = {
        "report": "v3 substrate selection recomputed under two rho conventions",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(REPO),
        "settings": {"din": args.din, "target_n": args.target_n,
                     "rho_target": args.rho_target, "seed": args.seed,
                     "gate_factor": SELECTION_GATE_FACTOR},
        "candidates": rows,
        "selection_frozen_convention_symmetrised_rho": sel_sym,
        "selection_corrected_directed_rho": sel_dir,
        "selection_changed": sel_sym["selected"] != sel_dir["selected"],
        "stop_loss_either_way": sel_sym["stop_loss"] or sel_dir["stop_loss"],
        "note": ("The frozen score_matrix symmetrises before eigsh, so its rho is a "
                 "legitimate SYMMETRIC estimate and is NOT the directed-eigsh defect. "
                 "This run quantifies the convention difference; it does not assume it."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print()
    print(f"frozen convention (symmetrised rho): selected={sel_sym['selected']} "
          f"stop_loss={sel_sym['stop_loss']}")
    print(f"corrected convention (directed rho): selected={sel_dir['selected']} "
          f"stop_loss={sel_dir['stop_loss']}")
    print(f"SELECTION CHANGED: {report['selection_changed']}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
