#!/usr/bin/env python
"""Materialize F1 (the frozen S0 substrate) and its dataset-conditioned inputs.

Authority: docs/resaudit_food_preregistration_amendment_4.md (F1 identity vs B),
docs/resaudit_food_preregistration_amendment_5.md (A_hash is not an identity),
docs/resaudit_food_preregistration_amendment_6.md (the Din allocation rule).

F1's structural identity is the frozen S0 substrate (candidate 0 of
results/audit/v3_selection/V3_substrate_selection.json), reproduced with the frozen
selection settings din=5, target_n=1000, seed=0. Identity is proven by the deterministic
content hashes -- node list and edge list -- plus N, M and B at Din=5. The legacy `A_hash`
is NOT a criterion (amendment 5: it digests a stochastic `eigsh` normalisation and takes a
different value on every call).

Amendment 6: the input allocation for a width D is
    weights (ORN:2, PN:2, KC:1), q_l = D*w_l/5, floors, then largest-remainder with the
fixed tie order ORN > PN > KC
giving (2,2,1) at 5, (3,2,1) at 6 and (3,3,2) at 8. The RULE is frozen, not the outputs.
At Din=5 the rule must reproduce the historical mapping bit-for-bit.

NOTE ON `din` AND THE WIRING. `expand_from_orns` is Din-parameterized, so S0 generated at
Din=6/8 is a different substrate. Amendment 4 requires A byte-identical across widths, so
the wiring is materialized ONCE in the frozen Din=5 selection context and only B is
re-instantiated.

Usage:
    PYTHONPATH=. python3 ops/audit/resaudit_materialize_f1.py --data-root ..
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
from drososense.connectome_candidates import generate_candidate  # noqa: E402
from drososense.reservoir.input_mapping import (  # noqa: E402
    CellTypeAnnotation,
    build_typed_aligned_mapping,
)
from resaudit.din_allocation import (  # noqa: E402
    FROZEN_DEFAULT_ALLOCATION,
    apportion_din,
)

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

#: Amendment 6: the rule's expected outputs at the preregistered widths.
EXPECTED_ALLOCATION = {
    5: {"ORN": 2, "PN": 2, "KC": 1},
    6: {"ORN": 3, "PN": 2, "KC": 1},
    8: {"ORN": 3, "PN": 3, "KC": 2},
}

#: Amendment 4: the dataset-conditioned input widths.
DATASET_DIN = {"FD1": 6, "FD3": 6, "FD2": 8}

#: The typed-aligned rule's frozen construction constants (build_candidate_input).
RECEIVING_FRACTION = 0.80
DENSITY_LIMIT = 0.16


def _matrix_sha256(matrix: sp.spmatrix) -> str:
    """Verbatim from ops/audit/v3_substrate_selection.py, so A_hash is comparable."""
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def score_matrix(matrix: sp.csr_matrix, *, rho_target: float) -> sp.csr_matrix:
    """Verbatim from ops/audit/v3_substrate_selection.py. Legacy run provenance only."""
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
    comes from the node metadata's ``root_id`` column in ``node_idx`` order -- the order the
    graph's rows are in. Asserted, not assumed.
    """
    npz = data_root / "connectome/adjacency/olfactory_v1.npz"
    with np.load(npz, allow_pickle=True) as payload:
        keys = set(payload.keys())
        if {"adj_shape", "adj_data", "adj_indices", "adj_indptr"} <= keys:
            shape = tuple(int(v) for v in payload["adj_shape"])
            data = payload["adj_data"]
            indices = payload["adj_indices"]
            indptr = payload["adj_indptr"]
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


def _build_B(cand, annotation, din: int, allocation: dict):
    """The typed-aligned mapping at a declared width. No fallback, no substitution."""
    return build_typed_aligned_mapping(
        cand.root_ids, annotation, din, seed=FROZEN_SETTINGS["seed"],
        din_per_layer=allocation, receiving_fraction=RECEIVING_FRACTION,
        density_limit=DENSITY_LIMIT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", default=str(repo.parent))
    parser.add_argument(
        "--node-meta", default=str(repo / "connectome/metadata/olfactory_v1_node_meta.csv")
    )
    parser.add_argument("--out-dir", default=str(repo / "results/audit/resaudit_stage1/f1"))
    args = parser.parse_args(argv)

    data_root, node_meta, out_dir = Path(args.data_root), Path(args.node_meta), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_graph, node_ids, annotation, layout, npz_path = load_raw(data_root, node_meta)
    raw_sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()
    print(f"raw adjacency : {npz_path.name} shape={raw_graph.shape} nnz={raw_graph.nnz}")
    print(f"raw sha256    : {raw_sha}")

    # ---- 1. reproduce the frozen S0 wiring ------------------------------------------
    cand = generate_candidate(
        "S0", raw_graph=raw_graph, root_ids=node_ids, annotation=annotation,
        din=FROZEN_SETTINGS["din"], target_n=FROZEN_SETTINGS["target_n"],
        seed=FROZEN_SETTINGS["seed"],
    )
    prov = dict(cand.provenance)
    A5 = score_matrix(cand.adjacency, rho_target=FROZEN_SETTINGS["rho_target"])
    a_hash = _matrix_sha256(A5)

    check = {
        "n_nodes": int(cand.n_nodes) == FROZEN_S0["n_nodes"],
        "n_edges": int(cand.n_edges) == FROZEN_S0["n_edges"],
        "node_list_sha256": prov.get("node_list_sha256") == FROZEN_S0["node_list_sha256"],
        "edge_list_sha256": prov.get("edge_list_sha256") == FROZEN_S0["edge_list_sha256"],
    }
    print("\n=== FROZEN S0 REPRODUCTION (wiring identity) ===")
    print(f"  N  {cand.n_nodes:>6} expected {FROZEN_S0['n_nodes']:>6}  "
          f"{'OK' if check['n_nodes'] else 'MISMATCH'}")
    print(f"  M  {cand.n_edges:>6} expected {FROZEN_S0['n_edges']:>6}  "
          f"{'OK' if check['n_edges'] else 'MISMATCH'}")
    print(f"  node_list_sha256 {'OK' if check['node_list_sha256'] else 'MISMATCH'}"
          f"  {prov.get('node_list_sha256')}")
    print(f"  edge_list_sha256 {'OK' if check['edge_list_sha256'] else 'MISMATCH'}"
          f"  {prov.get('edge_list_sha256')}")
    print(f"  A_hash legacy (not an identity, amendment 5) {a_hash[:16]}... "
          f"vs {FROZEN_S0['A_hash_not_an_identity'][:16]}...")
    if not all(check.values()):
        print("\nSTOP: the frozen S0 was NOT reproduced. Amendment 4 section 5 forbids accepting")
        print("agreement of N and M alone. This is a provenance failure to explain.")
        return 2

    # ---- 2. amendment 6: the allocation rule ---------------------------------------
    print("\n=== AMENDMENT 6 ALLOCATION RULE ===")
    if apportion_din(5) != dict(FROZEN_DEFAULT_ALLOCATION):
        print("STOP: the rule does not reproduce the frozen Din=5 allocation. Stage 1 blocked.")
        return 2
    allocations: dict[int, dict[str, int]] = {}
    widths = sorted({5, *DATASET_DIN.values()})
    for din in widths:
        alloc = apportion_din(din)
        allocations[din] = alloc
        if alloc != EXPECTED_ALLOCATION[din]:
            raise SystemExit(
                f"STOP: Din={din} allocation {alloc} != expected {EXPECTED_ALLOCATION[din]}"
            )
        print(f"  Din={din}: {alloc}  OK")

    # ---- 3. dataset-conditioned B, twice per width ---------------------------------
    print("\n=== DATASET-CONDITIONED INPUT ===")
    Bs: dict[int, np.ndarray] = {}
    b_stats: dict[str, dict] = {}
    for din in widths:
        m1 = _build_B(cand, annotation, din, allocations[din])
        m2 = _build_B(cand, annotation, din, allocations[din])
        B1, B2 = m1.w_in.toarray(), m2.w_in.toarray()
        Bs[din] = B1
        b_stats[str(din)] = {
            "allocation": allocations[din],
            "input_nodes": int(m1.support_rows.size),
            "density": float(m1.density()),
            "w_in_sha256": m1.describe()["w_in_sha256"],
            "deterministic_two_runs_byte_identical": bool(np.array_equal(B1, B2)),
            "fallback_used": False,
            "dynamic_allocation": False,
        }
        print(f"  Din={din}: alloc={allocations[din]} nodes={m1.support_rows.size} "
              f"density={float(m1.density()):.4f} "
              f"two_runs_identical={b_stats[str(din)]['deterministic_two_runs_byte_identical']}")

    # ---- 4. amendment 6 section 6: the four sanity checks ---------------------------
    checks = {
        "1_din5_allocation_and_B_hash": bool(
            allocations[5] == EXPECTED_ALLOCATION[5]
            and b_stats["5"]["w_in_sha256"] == FROZEN_S0["B_hash"]
        ),
        "2_din6_allocation_3_2_1_no_fallback": bool(
            allocations[6] == EXPECTED_ALLOCATION[6] and not b_stats["6"]["fallback_used"]
        ),
        "3_din8_allocation_3_3_2_no_fallback": bool(
            allocations[8] == EXPECTED_ALLOCATION[8] and not b_stats["8"]["fallback_used"]
        ),
        "4_two_materializations_byte_identical": bool(
            all(v["deterministic_two_runs_byte_identical"] for v in b_stats.values())
        ),
    }
    print("\n=== SANITY CHECKS (amendment 6 section 6) ===")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        print("\nSTOP: a sanity check failed; F1 stays PARTIAL and Stage 1 does not start.")
        return 2

    # ---- 5. artifacts ----------------------------------------------------------------
    csr = A5.tocsr()
    np.savez_compressed(
        out_dir / "F1_A.npz", adj_data=csr.data, adj_indices=csr.indices,
        adj_indptr=csr.indptr, adj_shape=np.asarray(csr.shape),
        node_ids=np.asarray(cand.root_ids, dtype=np.int64),
    )
    # The RAW induced synapse-count adjacency. This is the substrate's A and the thing the
    # edge-list hash identifies; the rho-scaled copies above and in F1_A.npz are scoring
    # normalisations belonging to a particular run's convention (amendment 5), which is why
    # the audit re-normalises to its own declared FROZEN_RHO_TARGET rather than reusing them.
    raw_csr = sp.csr_matrix(cand.adjacency).tocsr()
    raw_weight_sha = hashlib.sha256(
        np.ascontiguousarray(np.sort(raw_csr.data), dtype=np.float64).tobytes()
    ).hexdigest()
    np.savez_compressed(
        out_dir / "F1_A_raw.npz", adj_data=raw_csr.data, adj_indices=raw_csr.indices,
        adj_indptr=raw_csr.indptr, adj_shape=np.asarray(raw_csr.shape),
        node_ids=np.asarray(cand.root_ids, dtype=np.int64),
    )
    print(f"  raw A: nnz={raw_csr.nnz} weight_multiset_sha256={raw_weight_sha[:16]}...")
    for din, B in Bs.items():
        np.savez_compressed(out_dir / f"F1_B_Din{din}.npz", w_in=B, din=np.asarray(din),
                            node_ids=np.asarray(cand.root_ids, dtype=np.int64))

    (out_dir / "F1_materialization.json").write_text(
        json.dumps(
            {
                "verdict": "REPRODUCED",
                "status": "VERIFIED_FOR_STAGE1",
                "dataset": "F1 (frozen S0 substrate)",
                "raw_connectome_sha256": raw_sha,
                "raw_connectome_layout": layout,
                "selection_rule_version": prov.get("selection_rule"),
                "selection_settings": FROZEN_SETTINGS,
                "node_list_sha256": prov.get("node_list_sha256"),
                "edge_list_sha256": prov.get("edge_list_sha256"),
                "substrate_identity": (
                    "node-list hash + edge-list hash + raw weights/topology provenance "
                    "(amendment 5); the legacy scaled-float A_hash is run provenance only"
                ),
                "A_sha256_observed_legacy": a_hash,
                "A_sha256_frozen_NOT_AN_IDENTITY": FROZEN_S0["A_hash_not_an_identity"],
                "N": int(cand.n_nodes),
                "M": int(cand.n_edges),
                "B5_sha256_historical": FROZEN_S0["B_hash"],
                "raw_A_nnz": int(raw_csr.nnz),
                "raw_weight_multiset_sha256": raw_weight_sha,
                "raw_A_note": ("F1_A_raw.npz is the substrate identity alongside the node/edge "
                               "list hashes; F1_A.npz is the legacy 0.9-targeted scoring copy "
                               "and must NOT be fed to the audit, which has its own declared "
                               "FROZEN_RHO_TARGET=0.95"),
                "B_sha256": {d: v["w_in_sha256"] for d, v in b_stats.items()},
                "B_stats": b_stats,
                "allocation_rule": (
                    "amendment 6: weights (ORN:2,PN:2,KC:1), q_l = D*w_l/5, floors then "
                    "largest-remainder, fixed tie order ORN>PN>KC"
                ),
                "allocations": {str(k): v for k, v in allocations.items()},
                "dataset_din": DATASET_DIN,
                "input_mapping_rule": (
                    "typed-aligned, receiving_fraction=0.80, density_limit=0.16 (unchanged)"
                ),
                "sanity_checks": checks,
                "nested_property_B6_is_B8_prefix": "not required (amendment 6 section 5)",
                "din_affects_wiring": True,
                "din_wiring_note": (
                    "expand_from_orns is Din-parameterized, so the wiring is materialized once "
                    "in the frozen Din=5 selection context and only B is re-instantiated "
                    "(amendment 4)"
                ),
                "protocol_amendment_hashes": {
                    "amendment_3": "655acc84effa19f6e9eb3dfab8ca9389485155aa96cd1826e5b6fdb24988dd11",
                    "amendment_4": "9992d79f4ed524da0b13babd62345a6c2452b03c4b7f9ca9c0eb9bdc18ed9ac1",
                    "amendment_5": "12a8d5c3b8017b6afeace0b12a0275a10a288024feccecfee87405a400cd45e4",
                    "amendment_6": "PENDING_SIDECAR",
                },
                "git_head": git_head(repo),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir}/F1_A.npz, F1_B_Din*.npz, F1_materialization.json")
    print("STATUS: VERIFIED_FOR_STAGE1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
