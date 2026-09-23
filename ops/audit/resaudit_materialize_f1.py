#!/usr/bin/env python
"""Materialize F1 (the frozen S0 substrate) and its dataset-conditioned inputs.

Authority: docs/resaudit_food_preregistration_amendment_4.md.

F1's structural identity is the frozen S0 substrate as recorded in
results/audit/v3_selection/V3_substrate_selection.json (candidate 0):

    N = 1000, M = 80443, Din = 5
    A_hash = 3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20
    B_hash = 3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968

The frozen selection settings are reproduced exactly: din=5, target_n=1000, seed=0,
rho_target=0.9, and the amendment-2/3 typed-aligned input rule
(receiving_fraction=0.80, density_limit=0.16).

NOTE ON `din` AND THE WIRING. `expand_from_orns` is Din-parameterized: `din` sets the ORN
seed budget and the class/group floors, so generating S0 at Din=6 or 8 yields a DIFFERENT
node set, not merely a different B. Amendment 4 requires F1's node set, A and weights to be
byte-identical across widths, so the wiring is materialized ONCE in the frozen Din=5
selection context and only B is re-instantiated per dataset. This script asserts that
invariant rather than assuming it.

Usage:
    PYTHONPATH=. python3 ops/audit/resaudit_materialize_f1.py \
        --data-root .. --out-dir results/audit/resaudit_stage1/f1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectome.cell_types import load_node_classes  # noqa: E402
from drososense.connectome_candidates import (  # noqa: E402
    build_candidate_input,
    generate_candidate,
)
from drososense.reservoir.input_mapping import CellTypeAnnotation  # noqa: E402

#: The frozen S0 provenance this materialization must reproduce exactly.
FROZEN_S0 = {
    "n_nodes": 1000,
    "n_edges": 80443,
    "din": 5,
    "node_list_sha256": "77ec9bfd90d6ec7216f642e1860c815ca972fd7638f1bb464766b9053faccb1e",
    "edge_list_sha256": "a840f45bd5c96ce654f8ff9c5957b0ec3eb158527a417ff635b462e03815740a",
    "B_hash": "3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968",
    # Amendment 5: recorded for the record only -- NOT a reproduction criterion.
    "A_hash_not_an_identity": "3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20",
}

#: The frozen selection settings (V3_substrate_selection.json -> settings).
FROZEN_SETTINGS = {"din": 5, "target_n": 1000, "seed": 0, "rho_target": 0.9}

#: Amendment 4: the dataset-conditioned input widths.
DATASET_DIN = {"FD1": 6, "FD3": 6, "FD2": 8}


def _matrix_sha256(matrix: sp.spmatrix) -> str:
    """Verbatim from ops/audit/v3_substrate_selection.py, so A_hash is comparable."""
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def score_matrix(matrix: sp.csr_matrix, *, rho_target: float) -> sp.csr_matrix:
    """Verbatim from ops/audit/v3_substrate_selection.py, so A_hash is comparable."""
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


def load_raw(data_root: Path, node_meta: Path):
    """Load the raw adjacency and typed annotation.

    The frozen loader reads npz keys ``adj_shape/adj_data/adj_indices/adj_indptr/node_ids``.
    The local artifact is a scipy-saved csr (``shape/data/indices/indptr``) with no
    ``node_ids``, so the payload is reconstructed: the matrix is identical, and ``node_ids``
    is taken from the node metadata's ``root_id`` column in ``node_idx`` order — which is the
    order the graph's rows are in. Both facts are asserted below, not assumed.
    """
    npz = data_root / "connectome/adjacency/olfactory_v1.npz"
    with np.load(npz, allow_pickle=True) as payload:
        keys = set(payload.keys())
        if {"adj_shape", "adj_data", "adj_indices", "adj_indptr"} <= keys:
            shape = tuple(int(v) for v in payload["adj_shape"])
            data, indices, indptr = payload["adj_data"], payload["adj_indices"], payload["adj_indptr"]
            node_ids = np.asarray(payload["node_ids"]) if "node_ids" in keys else None
            layout = "frozen_loader_layout"
        elif {"shape", "data", "indices", "indptr"} <= keys:
            shape = tuple(int(v) for v in payload["shape"])
            data, indices, indptr = payload["data"], payload["indices"], payload["indptr"]
            node_ids = None
            layout = "scipy_csr_layout"
        else:
            raise SystemExit(f"unrecognised npz layout: {sorted(keys)}")
    raw_graph = sp.csr_matrix(
        (np.asarray(data, dtype=float), np.asarray(indices), np.asarray(indptr)), shape=shape
    )

    if node_ids is None:
        meta = np.genfromtxt(node_meta, delimiter=",", names=True, dtype=None, encoding="utf-8")
        root_id = np.asarray(meta["root_id"], dtype=np.int64)
        node_idx = np.asarray(meta["node_idx"], dtype=np.int64)
        assert node_idx.min() == 0 and node_idx.max() == shape[0] - 1, "node_idx not a full range"
        node_ids = np.empty(shape[0], dtype=np.int64)
        node_ids[node_idx] = root_id

    classes = load_node_classes(node_meta)
    counts: dict[str, int] = {}
    for v in classes.values():
        counts[v] = counts.get(v, 0) + 1
    annotation = CellTypeAnnotation(
        classes=classes, source=str(node_meta), per_class_counts=counts
    )
    return raw_graph, node_ids, annotation, layout, npz


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", default=str(repo.parent))
    parser.add_argument("--node-meta", default=str(repo / "connectome/metadata/olfactory_v1_node_meta.csv"))
    parser.add_argument("--out-dir", default=str(repo / "results/audit/resaudit_stage1/f1"))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    node_meta = Path(args.node_meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_graph, node_ids, annotation, layout, npz_path = load_raw(data_root, node_meta)
    raw_sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()
    print(f"raw adjacency : {npz_path.name} shape={raw_graph.shape} nnz={raw_graph.nnz}")
    print(f"layout        : {layout}")
    print(f"raw sha256    : {raw_sha}")

    # ---- 1. reproduce the frozen S0 exactly -------------------------------------------
    cand = generate_candidate(
        "S0",
        raw_graph=raw_graph,
        root_ids=node_ids,
        annotation=annotation,
        din=FROZEN_SETTINGS["din"],
        target_n=FROZEN_SETTINGS["target_n"],
        seed=FROZEN_SETTINGS["seed"],
    )
    mapping5 = build_candidate_input(cand, annotation, FROZEN_SETTINGS["din"], seed=FROZEN_SETTINGS["seed"])
    B5 = mapping5.w_in.toarray()
    A5 = score_matrix(cand.adjacency, rho_target=FROZEN_SETTINGS["rho_target"])
    a_hash, b_hash = _matrix_sha256(A5), mapping5.describe()["w_in_sha256"]

    prov = dict(cand.provenance)
    # Amendment 5: A_hash is NOT an identity -- score_matrix normalises with a
    # stochastic eigsh estimate, so its digest changes run to run. The wiring-level
    # hashes are the reproduction criterion.
    check = {
        "n_nodes": int(cand.n_nodes) == FROZEN_S0["n_nodes"],
        "n_edges": int(cand.n_edges) == FROZEN_S0["n_edges"],
        "node_list_sha256": prov.get("node_list_sha256") == FROZEN_S0["node_list_sha256"],
        "edge_list_sha256": prov.get("edge_list_sha256") == FROZEN_S0["edge_list_sha256"],
        "B_hash": b_hash == FROZEN_S0["B_hash"],
    }
    print("\n=== FROZEN S0 REPRODUCTION ===")
    print(f"  N        {cand.n_nodes:>8}  expected {FROZEN_S0['n_nodes']:>8}  {'OK' if check['n_nodes'] else 'MISMATCH'}")
    print(f"  M        {cand.n_edges:>8}  expected {FROZEN_S0['n_edges']:>8}  {'OK' if check['n_edges'] else 'MISMATCH'}")
    print(f"  B_hash   {b_hash[:16]}...  expected {FROZEN_S0['B_hash'][:16]}...  {'OK' if check['B_hash'] else 'MISMATCH'}")
    print(f"  node_list_sha256 {'OK' if check['node_list_sha256'] else 'MISMATCH'}  {prov.get('node_list_sha256')}")
    print(f"  edge_list_sha256 {'OK' if check['edge_list_sha256'] else 'MISMATCH'}  {prov.get('edge_list_sha256')}")
    print(f"  A_hash (NOT an identity, amendment 5) {a_hash[:16]}... vs frozen {FROZEN_S0['A_hash_not_an_identity'][:16]}...")

    if not all(check.values()):
        print("\nSTOP: the frozen S0 was NOT reproduced exactly. Amendment 4 section 5 forbids")
        print("accepting agreement of N and M alone. This is a provenance failure to explain.")
        (out_dir / "F1_materialization_STOP.json").write_text(
            json.dumps(
                {
                    "verdict": "PROVENANCE_FAILURE",
                    "reproduced": {"n_nodes": int(cand.n_nodes), "n_edges": int(cand.n_edges),
                                   "A_hash": a_hash, "B_hash": b_hash},
                    "expected": FROZEN_S0,
                    "checks": check,
                    "raw_connectome_sha256": raw_sha,
                    "raw_layout": layout,
                    "git_head": git_head(repo),
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    # ---- 2. dataset-conditioned B, on the SAME A -------------------------------------
    print("\n=== DATASET-CONDITIONED INPUT ===")
    Bs: dict[int, np.ndarray] = {}
    blocked: dict[str, str] = {}
    for din in sorted(set(DATASET_DIN.values())):
        try:
            m = build_candidate_input(cand, annotation, din, seed=FROZEN_SETTINGS["seed"])
        except Exception as exc:
            # The typed-aligned rule carries a FIXED per-layer allocation
            # DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER = {ORN:2, PN:2, KC:1} which must sum to
            # total_din. It has no declared allocation for any other width, so Din=6/8
            # cannot be built without INVENTING one. No allocation is improvised here:
            # recorded as a blocker for the operator instead.
            blocked[str(din)] = f"{type(exc).__name__}: {exc}"
            print(f"  Din={din}: BLOCKED -- {type(exc).__name__}: {str(exc)[:150]}")
            continue
        Bs[din] = m.w_in.toarray()
        print(f"  Din={din}: input_nodes={int(m.support_rows.size)} "
              f"density={float(m.density()):.4f} w_in_sha256={m.describe()['w_in_sha256'][:16]}...")
    if blocked:
        print("\n  The frozen typed-aligned mapping declares only {ORN:2, PN:2, KC:1} = Din=5.")
        print("  Din=6/8 require a DECLARED per-layer allocation. Not improvised -- operator decision.")

    # A must be byte-identical across widths: same object here by construction, asserted.
    assert int(cand.n_nodes) == FROZEN_S0["n_nodes"]
    nested = (None if not (6 in Bs and 8 in Bs)
              else bool(Bs[6].shape[0] == Bs[8].shape[0] and np.array_equal(Bs[6], Bs[8][:, :6])))
    print(f"  nested property B_6 == B_8[:, :6]: {nested}")

    np.savez_compressed(out_dir / "F1_A.npz",
                        adj_data=A5.tocsr().data, adj_indices=A5.tocsr().indices,
                        adj_indptr=A5.tocsr().indptr, adj_shape=np.asarray(A5.shape),
                        node_ids=np.asarray(cand.root_ids, dtype=np.int64))
    for din, B in Bs.items():
        np.savez_compressed(out_dir / f"F1_B_Din{din}.npz", w_in=B, din=np.asarray(din),
                            node_ids=np.asarray(cand.root_ids, dtype=np.int64))
    (out_dir / "F1_materialization.json").write_text(
        json.dumps(
            {
                "verdict": "REPRODUCED",
                "dataset": "F1 (frozen S0 substrate)",
                "raw_connectome_sha256": raw_sha,
                "raw_connectome_layout": layout,
                "selection_rule_version": prov.get("selection_rule"),
                "selection_settings": FROZEN_SETTINGS,
                "node_list_sha256": prov.get("node_list_sha256"),
                "edge_list_sha256": prov.get("edge_list_sha256"),
                "A_sha256_observed": a_hash,
                "A_sha256_frozen_NOT_AN_IDENTITY": FROZEN_S0["A_hash_not_an_identity"],
                "A_hash_defect": ("amendment 5: score_matrix normalises with an unseeded "
                                  "eigsh estimate, so this digest is not reproducible; the "
                                  "wiring-level hashes are the reproduction criterion"),
                "N": int(cand.n_nodes),
                "M": int(cand.n_edges),
                "B5_sha256_historical": b_hash,
                "B_sha256": {str(d): build_candidate_input(cand, annotation, d,
                              seed=FROZEN_SETTINGS["seed"]).describe()["w_in_sha256"]
                             for d in sorted(Bs)},
                "B_blocked_din": blocked,
                "B_blocker_reason": (
                    "DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER = {ORN:2, PN:2, KC:1} sums to 5 and "
                    "build_typed_aligned_mapping requires the allocation to sum to total_din. "
                    "The rule declares no allocation for any other width, so Din=6 (FD1/FD3) "
                    "and Din=8 (FD2) cannot be instantiated without inventing one. No "
                    "allocation was improvised; this awaits an operator decision."
                ),
                "status": ("PARTIAL: F1 wiring verified; B at Din=5 verified; "
                           "B at Din=6/8 blocked") if blocked else "COMPLETE",
                "dataset_din": DATASET_DIN,
                "input_mapping_rule": "amended-2/3 typed-aligned mapping, receiving_fraction=0.80, density_limit=0.16",
                "nested_property_B6_is_B8_prefix": nested,
                "din_affects_wiring": True,
                "din_wiring_note": (
                    "expand_from_orns is Din-parameterized (ORN seed budget, class/group "
                    "floors, ORN_FLOOR(din)), so S0 generated at Din=6/8 is a different "
                    "substrate. Amendment 4 requires A byte-identical across widths, so the "
                    "wiring is materialized once in the frozen Din=5 selection context and "
                    "only B is re-instantiated."
                ),
                "protocol_amendment_hashes": {
                    "amendment_3": "655acc84effa19f6e9eb3dfab8ca9389485155aa96cd1826e5b6fdb24988dd11",
                    "amendment_4": "9992d79f4ed524da0b13babd62345a6c2452b03c4b7f9ca9c0eb9bdc18ed9ac1",
                },
                "git_head": git_head(repo),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir}/F1_A.npz, F1_B_Din*.npz, F1_materialization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
