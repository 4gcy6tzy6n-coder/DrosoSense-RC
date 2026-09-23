#!/usr/bin/env python
"""M5 - Structural Dynamics Audit.

READ-ONLY. Measures on the C2 expansion's induced block (the same substrate the C3
gate measured):

  1. SCC / cycle statistics (largest SCC, count, non-trivial SCC fraction,
     cycle-edge fraction);
  2. ORN-reachable recurrent core;
  3. controllability effective rank (A^k B Krylov) at K = 1, 2, 4, 8, 16;
  4. spectrum of A (symmetrised, top |lambda|);
  5. raw vs row-L1 comparison: the same metrics on the raw substrate, on the raw
     substrate scaled to rho_target (no row normalization), and on the row-L1
     normalized substrate.

The point is to separate two structural hypotheses that C3's failure leaves open:

  - the graph is near-feedforward even at the WCC level (huge SCC, but few directed
    cycles); or
  - the graph has cycles but the row-L1 normalization suppresses their effect on
    the dynamics.

The decision the audit feeds:

  graph lacks recurrence               ->  change substrate
  graph has recurrence but suppressed   ->  change dynamics (e.g. v3-A)

Nothing is written anywhere this script commits to; the report is written to
results/audit/m5_structural_dynamics/M5_structural_dynamics.json by the caller.
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
from drososense.connectome_selection import expand_from_orns
from drososense.connectome_cycles import measure_substrate


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/drososense/data-root")
DEFAULT_NODE_META_REL = Path("connectome/metadata/olfactory_v1_node_meta.csv")
DEFAULT_OUT = Path(
    "/root/autodl-tmp/drososense/repo/results/audit/m5_structural_dynamics/M5_structural_dynamics.json"
)


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def rescale_to_spectral_radius(matrix: sp.csr_matrix, target: float) -> sp.csr_matrix:
    """Mirror of drososense.reservoir.connectome_reservoir.rescale_to_spectral_radius."""
    import scipy.sparse.linalg as _spla

    sym = ((matrix + matrix.T) * 0.5).tocsc()
    if sym.shape[0] < 2:
        return matrix.copy()
    k = min(6, sym.shape[0] - 2)
    vals = _spla.eigsh(sym, k=k, which="LM", return_eigenvectors=False)
    rho = float(np.abs(vals).max())
    if rho <= 1e-12:
        return matrix.copy()
    return (matrix * (target / rho)).tocsr()


def row_l1_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Presynaptic L1 normalization (rows of A sum to 1)."""
    csr = matrix.tocsr()
    row = np.asarray(csr.sum(axis=1)).ravel()
    row[row == 0] = 1.0
    return sp.diags(1.0 / row) @ csr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--node-meta", default=str(DEFAULT_NODE_META_REL))
    parser.add_argument("--din", type=int, default=5)
    parser.add_argument("--target-n", type=int, default=1000)
    parser.add_argument("--rho-target", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    node_meta = Path(args.node_meta)
    if not node_meta.is_file():
        node_meta = repo / DEFAULT_NODE_META_REL
    data_root = Path(args.data_root)
    npz = data_root / "connectome/adjacency/olfactory_v1.npz"

    with np.load(npz, allow_pickle=True) as payload:
        shape = tuple(int(v) for v in payload["adj_shape"])
        node_ids = np.asarray(payload["node_ids"])
        raw = sp.csr_matrix(
            (np.asarray(payload["adj_data"], dtype=float),
             np.asarray(payload["adj_indices"]),
             np.asarray(payload["adj_indptr"])),
            shape=shape,
        )

    classes = load_node_classes(node_meta)
    counts: dict[str, int] = {}
    for v in classes.values():
        counts[v] = counts.get(v, 0) + 1
    from drososense.reservoir.input_mapping import CellTypeAnnotation
    annotation = CellTypeAnnotation(classes=classes, source=str(node_meta), per_class_counts=counts)

    expansion = expand_from_orns(raw, node_ids, annotation, args.target_n, din=args.din, seed=args.seed)
    sel = expansion.node_indices
    selected_classes = annotation.class_of(node_ids[sel])
    orn_rows = np.asarray(expansion.detail["orn_budget"], dtype=np.int64)
    # orn_budget is the SIZE; the actual rows are the top-orn_budget ORN rows in declared
    # order. Get them from the substrate via the annotation.
    orn_rows = np.flatnonzero(selected_classes == "ORN")
    substrate_raw = raw[sel][:, sel].tocsr()
    n_nodes = substrate_raw.shape[0]

    # Three variants
    rho_target = float(args.rho_target)
    substrate_row = row_l1_normalize(substrate_raw)
    substrate_row = rescale_to_spectral_radius(substrate_row, rho_target)
    substrate_gamma = rescale_to_spectral_radius(substrate_raw, rho_target)
    # Substrate A itself has its own (default) spectral radius; record it for context
    from drososense.reservoir.connectome_reservoir import spectral_radius
    rho_raw = float(spectral_radius(substrate_raw, seed=0))

    # ORN-aligned input on the substrate (so controllability is measured with the
    # actual W_in the dynamics would see, not a generic one-hot)
    from drososense.reservoir.input_mapping import build_orn_aligned_mapping
    mapping = build_orn_aligned_mapping(
        substrate_raw, node_ids[sel], annotation, args.din, seed=args.seed
    )
    W_in = mapping.w_in.toarray()  # (N, Din)

    report = {
        "report_schema": "m5_structural_dynamics/1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head": git_head(repo),
            "validation_only": True,
            "touches_test_split": False,
            "fits_no_model": True,
            "substrate": "the C2 expansion's induced block on the olfactory v1 NPZ",
            "W_in": "v2 ORN-aligned input mapping (per docs/v2_construct_c1_input_mapping.md)",
            "input_layer_note": (
                "controllability is measured with this W_in so the rank figures "
                "are the same objects the dynamics would see, not a generic "
                "one-hot stimulus"
            ),
        },
        "substrate": {
            "n_nodes": int(n_nodes),
            "rho_raw": rho_raw,
            "rho_row_l1_rescaled": float(spectral_radius(substrate_row, seed=0)),
            "rho_gamma_only_rescaled": float(spectral_radius(substrate_gamma, seed=0)),
        },
        "orn_substrate": {
            "n_orn_rows": int(orn_rows.size),
        },
    }

    for name, A in (("raw", substrate_raw), ("row_l1_rescaled", substrate_row),
                     ("gamma_only_rescaled", substrate_gamma)):
        report[name] = measure_substrate(A, orn_rows=orn_rows, B=W_in)
        report[name]["rho_measured"] = float(spectral_radius(A, seed=0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    for name in ("raw", "row_l1_rescaled", "gamma_only_rescaled"):
        d = report[name]
        print(
            f"  {name:18s} SCCs={d['scc_count']:4d} largest={d['scc_largest_fraction']:.3f} "
            f"nontrivial_nodes={d['nodes_in_nontrivial_scc_fraction']:.3f} "
            f"cycle_edges={d['cycle_edge_fraction']:.3f} "
            f"orn_recurrent_core={d['orn_reachable_in_nontrivial_scc_fraction']:.3f} "
            f"K=1..16 D_eff={[report[name]['controllability_K'][f'K={k}']['effective_rank'] for k in (1,2,4,8,16)]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())