#!/usr/bin/env python
"""M5b — the input-geometry curve (v2 amendment 3).

READ-ONLY. Measures the C3 gates as a function of the input support size, on the same
substrate, the same (gain x leak) grid and the same dynamics as amendment 2. The question
amendment 3 was signed to answer is bounded and declared in advance:

    if some support size clears both gates  ->  the C3 failure was input-geometry-driven
    if the curve plateaus below both gates  ->  input geometry is excluded as sufficient

Support sizes (declared): 243 (ORN only), ~350, ~473 (amendment 2), ~600 (full typed
union), ~800 (best effort). Each is realised by `build_typed_aligned_mapping` at the
fraction that produces it; density is allowed up to amendment 3's 0.16 ceiling.

Nothing is written except the JSON report.
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
from drososense.reservoir.connectome_reservoir import (
    load_reservoir_topology_from_npz,
    spectral_radius,
)
from drososense.reservoir.dynamics import evaluate_c3
from drososense.reservoir.input_mapping import (
    CellTypeAnnotation,
    build_typed_aligned_mapping,
)
from drososense.reservoir.r2_counterfactual import build_wiring_counterfactual

DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/drososense/data-root")
DEFAULT_NODE_META = Path(
    "/root/autodl-tmp/drososense/repo/connectome/metadata/olfactory_v1_node_meta.csv"
)
DEFAULT_OUT = Path(
    "/root/autodl-tmp/drososense/repo/results/audit/m5_structural_dynamics/"
    "M5b_input_geometry_curve.json"
)

#: Amendment 3's relaxed C1.2 ceiling.
AMENDMENT_3_DENSITY_LIMIT = 0.16

#: The declared support sizes and the layer fractions that realise them.
SUPPORT_TARGETS: tuple[float, ...] = (243, 350, 473, 600, 800)

#: The (gain, leak) grid, identical to amendment 2's C3 run.
GAIN_VALUES = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
LEAK_VALUES = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--node-meta", default=str(DEFAULT_NODE_META))
    parser.add_argument("--din", type=int, default=5)
    parser.add_argument("--target-n", type=int, default=1000)
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

    expansion = expand_from_orns(
        raw_graph, node_ids, annotation, args.target_n, din=args.din, seed=args.seed
    )
    sel = expansion.node_indices
    selected_roots = node_ids[sel]
    selected_classes = annotation.class_of(selected_roots)
    raw_block = raw_graph[sel][:, sel].tocsr()

    # the two reservoirs (amendment 1's counterfactual)
    reference = load_reservoir_topology_from_npz(
        npz, normalization="n1_pre_l1", node_indices=sel
    )
    cf = build_wiring_counterfactual(
        raw_block, normalization="n1_pre_l1",
        target_spectral_radius=float(reference.spectral_radius),
        seed=20260920, time_budget_s=120.0,
    )
    reservoirs = {"R0_shared": cf.r0, "R2_shared": cf.r2}

    rng = np.random.default_rng(args.seed + 1)
    X_val = rng.standard_normal((10, 16, args.din))
    noise = rng.standard_normal(X_val.shape)
    for t in range(1, 16):
        X_val[:, t, :] = 0.6 * X_val[:, t - 1, :] + 0.4 * noise[:, t, :]

    # the typed populations available, so the fraction that realises each target support
    typed_total = int(np.isin(selected_classes, ["ORN", "PN", "KC"]).sum())
    orn_total = int((selected_classes == "ORN").sum())

    report = {
        "report_schema": "m5b_input_geometry_curve/1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head": git_head(repo),
            "validation_only": True,
            "touches_test_split": False,
            "fits_no_model": True,
            "amendment": "v2 amendment 3 (C1.2 density ceiling 0.10 -> 0.16)",
            "note": (
                "measures the input-geometry curve the amendment was signed to answer; "
                "the decision rule was declared before the data"
            ),
        },
        "settings": {
            "din": args.din,
            "target_n": args.target_n,
            "density_limit": AMENDMENT_3_DENSITY_LIMIT,
            "support_targets": list(SUPPORT_TARGETS),
            "grid_gain": list(GAIN_VALUES),
            "grid_leak": list(LEAK_VALUES),
            "typed_population_total": typed_total,
            "orn_population_total": orn_total,
        },
        "curve": [],
    }

    for target in SUPPORT_TARGETS:
        if target <= orn_total:
            fraction = 1.0  # ORN-only baseline is handled separately below
        else:
            fraction = min(1.0, float(target) / float(typed_total))
        try:
            mapping = build_typed_aligned_mapping(
                selected_roots, annotation, args.din, seed=args.seed,
                receiving_fraction=fraction, density_limit=AMENDMENT_3_DENSITY_LIMIT,
            )
        except Exception as exc:  # reported, never swallowed
            report["curve"].append({"support_target": target, "fraction": fraction,
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue
        W_in = mapping.w_in.tocsr()
        bias = np.zeros(raw_block.shape[0])
        entry = {
            "support_target": int(target),
            "fraction": float(fraction),
            "support_actual": int(mapping.support_rows.size),
            "nnz": int(W_in.nnz),
            "density": float(mapping.density()),
            "per_reservoir": {},
        }
        for name, A in reservoirs.items():
            best_mem = max_mem_args = None
            best_deff = None
            in_band = 0
            for gain in GAIN_VALUES:
                for leak in LEAK_VALUES:
                    r = evaluate_c3(A, W_in, bias, X_val, gain=gain, leak=leak, din=args.din)
                    mem = r["memory"]["drop_fraction"]
                    deff = r["effective_rank"]["D_eff_factor"]
                    if in_band is not None and r["criterion_lines"]["C3.1_R_t_in_band"]:
                        in_band += 1
                    if best_mem is None or mem > best_mem:
                        best_mem = mem
                        max_mem_args = (gain, leak)
                    if best_deff is None or deff > best_deff:
                        best_deff = deff
            entry["per_reservoir"][name] = {
                "best_memory_drop": float(best_mem),
                "best_memory_drop_at": list(max_mem_args),
                "best_D_eff_factor": float(best_deff),
                "grid_points_with_R_t_in_band": int(in_band),
                "clears_memory_gate": bool(best_mem >= 0.20),
                "clears_deff_gate": bool(best_deff >= 1.5),
            }
        entry["clears_both_gates"] = all(
            v["clears_memory_gate"] and v["clears_deff_gate"]
            for v in entry["per_reservoir"].values()
        )
        report["curve"].append(entry)
        print(
            f"  support={entry['support_actual']:4d} (frac {fraction:.3f}, density "
            f"{entry['density']:.4f}) | "
            + " | ".join(
                f"{n}: mem {v['best_memory_drop']:+.4f} D_eff {v['best_D_eff_factor']:.3f}"
                f" {'PASS' if (v['clears_memory_gate'] and v['clears_deff_gate']) else 'fail'}"
                for n, v in entry["per_reservoir"].items()
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    any_clear = any(e.get("clears_both_gates") for e in report["curve"])
    print(f"verdict: {'input geometry CAN clear both gates' if any_clear else 'NO support size clears both gates -> input geometry excluded as sufficient'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())