#!/usr/bin/env python
"""v3 substrate selection — generate S0–S4, score them, apply the pre-registered rule.

READ-ONLY with respect to data: no food label, no dataset metric, no test split. The
selection rule, the gate and the stop-loss are fixed in `docs/v3_preregistration.md` §3-§6
and are applied here **mechanically**:

    eligible = [c for c in candidates if S1(c) >= 2 * Din]
    if not eligible:                     -> STOP-LOSS
    else: max S1, ties by higher S2 participation ratio, then by smaller N
    freeze the selected candidate

Exact float ordering is used; **no tolerance-based "approximately tied" judgement is
introduced** (per the owner's constraint 6). The selection engine reads ONLY ``S1``,
``S2_participation_ratio`` and ``n_nodes`` -- never any S3 field, and a unit test asserts
that changing S3 cannot change the selected candidate.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectome.cell_types import load_node_classes
from drososense.connectome_candidates import (
    CANDIDATE_IDS,
    build_candidate_input,
    generate_candidate,
)
from drososense.reservoir.input_mapping import CellTypeAnnotation
from drososense.substrate_scores import eigenmode_score, krylov_score, spectral_score

DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/drososense/data-root")
DEFAULT_NODE_META = Path(
    "/root/autodl-tmp/drososense/repo/connectome/metadata/olfactory_v1_node_meta.csv"
)
DEFAULT_OUT = Path(
    "/root/autodl-tmp/drososense/repo/results/audit/v3_selection/V3_substrate_selection.json"
)

#: The pre-registered selection gate: S1 >= SELECTION_GATE_FACTOR * Din.
SELECTION_GATE_FACTOR = 2.0


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def score_matrix(matrix: sp.csr_matrix, *, rho_target: float) -> sp.csr_matrix:
    """The declared A normalization used for scoring: ONE global factor to rho_target.

    Identical for every candidate (owner's constraint 1): the raw induced synapse-count
    matrix scaled by a single global gamma. This is the scoring normalization only; the
    selected candidate is later taken through the full C1-C4 pipeline to fix its final
    preprocessing.
    """
    import scipy.sparse.linalg as _spla

    sym = ((matrix + matrix.T) * 0.5).tocsc()
    if sym.shape[0] < 3:
        return matrix.copy()
    k = min(6, sym.shape[0] - 2)
    try:
        vals = _spla.eigsh(sym, k=k, which="LM", return_eigenvectors=False)
        rho = float(np.abs(vals).max())
    except Exception:  # pragma: no cover
        rho = float(np.abs(np.linalg.eigvals(matrix.toarray())).max())
    if rho <= 1e-12:
        return matrix.copy()
    return (matrix * (rho_target / rho)).tocsr()


def select_candidate(rows: list[dict]) -> dict:
    """The pre-registered selection rule, applied mechanically.

    Args:
        rows: One dict per candidate with at least ``candidate_id``, ``S1``,
            ``S2_participation_ratio`` and ``n_nodes``.

    Returns:
        ``{"eligible": [...], "selected": id|None, "stop_loss": bool, "reason": str}``.
    """
    eligible = [r for r in rows if r["S1"] >= r["gate"]]
    if not eligible:
        return {
            "eligible": [],
            "selected": None,
            "stop_loss": True,
            "reason": (
                "no candidate reached S1 >= 2*Din; per docs/v3_preregistration.md section 6 "
                "the connectome-reservoir main line STOPS and no v4 is opened"
            ),
        }
    # exact float ordering; ties (bit-identical S1) broken by higher S2 participation
    # ratio, then by smaller N. No tolerance is introduced.
    eligible_sorted = sorted(
        eligible,
        key=lambda r: (-r["S1"], -r["S2_participation_ratio"], r["n_nodes"]),
    )
    winner = eligible_sorted[0]
    return {
        "eligible": [r["candidate_id"] for r in eligible_sorted],
        "selected": winner["candidate_id"],
        "stop_loss": False,
        "reason": (
            "max S1 among the eligible candidates, ties by higher S2 participation ratio, "
            "then by smaller N"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--node-meta", default=str(DEFAULT_NODE_META))
    parser.add_argument("--din", type=int, default=5)
    parser.add_argument("--target-n", type=int, default=1000)
    parser.add_argument("--rho-target", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root)
    npz = data_root / "connectome/adjacency/olfactory_v1.npz"
    node_meta = Path(args.node_meta)

    with np.load(npz, allow_pickle=True) as payload:
        shape = tuple(int(v) for v in payload["adj_shape"])
        node_ids = np.asarray(payload["node_ids"])
        raw_graph = sp.csr_matrix(
            (np.asarray(payload["adj_data"], dtype=float),
             np.asarray(payload["adj_indices"]),
             np.asarray(payload["adj_indptr"])),
            shape=shape,
        )
    classes = load_node_classes(node_meta)
    counts: dict[str, int] = {}
    for v in classes.values():
        counts[v] = counts.get(v, 0) + 1
    annotation = CellTypeAnnotation(classes=classes, source=str(node_meta), per_class_counts=counts)

    gate = SELECTION_GATE_FACTOR * float(args.din)
    report = {
        "report_schema": "v3_substrate_selection/1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head": git_head(repo),
            "validation_only": True,
            "touches_test_split": False,
            "fits_no_model": True,
            "reads_food_labels": False,
            "preregistration": "docs/v3_preregistration.md",
            "note": (
                "selection uses only structure and linear dynamics; the selection engine "
                "reads S1, S2_participation_ratio and n_nodes and never any S3 field"
            ),
        },
        "settings": {
            "din": args.din,
            "target_n": args.target_n,
            "rho_target": args.rho_target,
            "krylov_K": 16,
            "selection_gate": gate,
            "selection_gate_factor": SELECTION_GATE_FACTOR,
            "tie_break": "higher S2 participation ratio, then smaller N; exact float ordering",
            "input_rule": (
                "amendment 2/3 typed-aligned mapping, receiving_fraction 0.80, "
                "density_limit 0.16, identical for every candidate"
            ),
            "A_normalization_for_scoring": (
                "raw induced synapse-count matrix scaled by ONE global factor to rho_target, "
                "identical for every candidate"
            ),
        },
        "candidates": [],
    }

    rows_for_selection: list[dict] = []
    for cid in CANDIDATE_IDS:
        try:
            cand = generate_candidate(
                cid, raw_graph=raw_graph, root_ids=node_ids, annotation=annotation,
                din=args.din, target_n=args.target_n, seed=args.seed,
            )
        except Exception as exc:
            report["candidates"].append(
                {"candidate_id": cid, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        mapping = build_candidate_input(cand, annotation, args.din, seed=args.seed)
        B = mapping.w_in.toarray()
        A = score_matrix(cand.adjacency, rho_target=args.rho_target)
        s1 = krylov_score(A, B)
        s2 = eigenmode_score(A, B)
        s3 = spectral_score(A)
        entry = {
            "candidate_id": cid,
            "n_nodes": cand.n_nodes,
            "n_edges": cand.n_edges,
            "input_nodes": int(mapping.support_rows.size),
            "input_density": float(mapping.density()),
            "Din": args.din,
            "normalization": f"global gamma to rho={args.rho_target}",
            "B_hash": mapping.describe()["w_in_sha256"],
            "A_hash": _matrix_sha256(A),
            "K": s1.K,
            "singular_values": list(s1.singular_values),
            "numerical_rank": s1.numerical_rank,
            "effective_rank": s1.effective_rank,
            "S1": s1.S1,
            "gate": gate,
            "clears_gate": bool(s1.S1 >= gate),
            "krylov_diagnostics": {
                "theoretical_columns": s1.theoretical_columns,
                "krylov_capacity_used": s1.as_dict()["krylov_capacity_used"],
                "S1_per_node": s1.as_dict()["S1_per_node"],
                "S1_over_min_capacity": s1.as_dict()["S1_over_min_capacity"],
            },
            "S2": s2.as_dict(),
            "S3": s3,
            "provenance": cand.provenance,
        }
        report["candidates"].append(entry)
        rows_for_selection.append(
            {
                "candidate_id": cid,
                "S1": float(s1.S1),
                "gate": gate,
                "S2_participation_ratio": float(s2.participation_ratio),
                "n_nodes": cand.n_nodes,
            }
        )
        print(
            f"  {cid}: N={cand.n_nodes:4d} M={cand.n_edges:7d} in={mapping.support_rows.size:4d} "
            f"| S1={s1.S1:6.3f} (gate {gate:.1f}, {'PASS' if s1.S1 >= gate else 'fail'}) "
            f"| S2_PR={s2.participation_ratio:5.2f} ent={s2.mode_energy_entropy:5.2f} "
            f"basis={s2.basis} cond={s2.eigenvector_condition_number:.1e} "
            f"| S3 rho={s3['S3_spectral_radius']:.3f} nonnorm={s3['S3_henrici_non_normality']:.4f}"
        )

    selection = select_candidate(rows_for_selection)
    report["selection"] = selection
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    if selection["stop_loss"]:
        print(f"STOP-LOSS: {selection['reason']}")
    else:
        print(f"SELECTED: {selection['selected']} (eligible: {selection['eligible']})")
    return 0


def _matrix_sha256(matrix: sp.spmatrix) -> str:
    import hashlib

    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())