#!/usr/bin/env python
"""v2 construct checks — C1 (ORN-aligned sparse input mapping), READ-ONLY.

The signed pre-registration (`docs/v2_preregistration.md` §1) makes every C1
criterion a property of the **construction**, measured validation-only, before any
formal experiment. This script measures them and writes one JSON report per
criterion with the observed value, the threshold, the verdict and the source of
every number. It fits no model, reads no test split, and changes no record.

WHAT IT MEASURES, AND ON WHAT SUBSTRATE

1. **The full delivered olfactory graph** as the *construct* substrate: it is the
   only place where the real ORN->PN edges and the real populations are all
   present, so it is where C1.1/C1.2/C1.4/C1.5 can be measured at all. It is NOT
   the formal substrate (124,185 nodes is not a trainable reservoir); the formal
   substrate is C2's deliverable and must itself satisfy C1.4.
2. **The delivered v1 selections** (N = 250 … 4000). Measured here for the same
   reason the audit measured them: they are what the project actually ran. The
   mapping is expected to REFUSE them, and the refusal is the finding — a
   substrate with 0-19 ORNs cannot carry an ORN-aligned input pathway, and the
   refusal is reported with the counts rather than patched by densifying.

A failed criterion is a **design limitation**: it is reported, never relaxed.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from connectome.cell_types import DECLARED_CLASSES, load_node_classes  # noqa: E402
from drososense.connectome_selection import (  # noqa: E402
    expand_from_orns,
    select_nodes,
    subgraph_quality,
)
from drososense.reservoir.input_mapping import (  # noqa: E402
    INPUT_MAPPING_ORN_ALIGNED,
    ORN_DENSITY_LIMIT,
    CellTypeAnnotation,
    InputMappingError,
    build_orn_aligned_mapping,
    compare_annotations,
    load_classified_edge_annotation,
)

OUT_DIR = REPO_ROOT / "results" / "audit" / "v2_construct"
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/drososense/data-root")
DEFAULT_SIZES = (250, 500, 1000, 2000, 4000)
SELECTION_SEED = 20260920  # the runner's frozen selection seed (runner.py)
REPORT_SCHEMA = "c1_input_mapping/1"
REPORT_SCHEMA_C2 = "c2_subgraph/1"

#: C1.4's thresholds, as signed. `Din` is the input channel count.
C1_4_THRESHOLDS = {
    "ORN": lambda din: max(20, 2 * din),
    "PN": lambda din: 20,
    "KC": lambda din: max(50, 4 * din),
    "MBON+DAN+higher_order": lambda din: 20,
}


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return ""


def population_counts(annotation: CellTypeAnnotation, root_ids) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in annotation.class_of(root_ids).tolist():
        counts[str(value)] = counts.get(str(value), 0) + 1
    counts["MBON+DAN+higher_order"] = sum(
        counts.get(name, 0) for name in ("MBON", "DAN", "higher_order")
    )
    return counts


def criterion(
    cid: str,
    statement: str,
    observed,
    threshold,
    passed: bool | None,
    source: str,
    note: str = "",
) -> dict:
    """One criterion row: what was measured, against what, and where it came from.

    ``passed=None`` means NOT MEASURED, which is a different statement from a pass
    and from a fail -- a criterion that could not be measured is never reported as
    satisfied.
    """
    return {
        "criterion": cid,
        "statement": statement,
        "observed": observed,
        "threshold": threshold,
        "pass": passed,
        "source": source,
        "note": note,
    }


def measure_full_graph(
    adjacency_path: Path, node_meta: Path, din: int, seed: int, input_scale: float
) -> tuple[dict, np.ndarray, CellTypeAnnotation]:
    """C1.1/C1.2/C1.4/C1.5/C1.6 on the full delivered olfactory graph."""
    with np.load(adjacency_path, allow_pickle=True) as npz:
        shape = tuple(int(v) for v in npz["adj_shape"])
        node_ids = np.asarray(npz["node_ids"])
        matrix = sp.csr_matrix(
            (np.asarray(npz["adj_data"], dtype=float),
             np.asarray(npz["adj_indices"]),
             np.asarray(npz["adj_indptr"])),
            shape=shape,
        )
    classes = load_node_classes(node_meta)
    annotation = CellTypeAnnotation(
        classes=classes,
        source=str(node_meta),
        per_class_counts={},
    )
    annotation = CellTypeAnnotation(
        classes=classes,
        source=str(node_meta),
        per_class_counts={
            name: sum(1 for v in classes.values() if v == name)
            for name in set(classes.values())
        },
    )
    counts = population_counts(annotation, node_ids)

    rows: list[dict] = []
    substrate = {
        "substrate": "full delivered olfactory graph",
        "n_nodes": shape[0],
        "n_edges": int(matrix.nnz),
        "populations": counts,
        "din": din,
        "node_ids_source": str(adjacency_path),
        "class_source": str(node_meta),
        "class_source_sha256": annotation.digest(),
        "is_formal_substrate": False,
        "why_not_formal": (
            "the full graph is not a trainable reservoir; it is the construct "
            "substrate where the real populations and ORN->PN edges are all present"
        ),
    }

    for name, threshold_fn in C1_4_THRESHOLDS.items():
        threshold = threshold_fn(din)
        observed = int(counts.get(name, 0))
        rows.append(
            criterion(
                f"C1.4[{name}]",
                f"population {name} present in the substrate",
                observed,
                threshold,
                observed >= threshold,
                f"{adjacency_path} + {node_meta}",
            )
        )

    try:
        mapping = build_orn_aligned_mapping(
            matrix, node_ids, annotation, din, seed=seed, input_scale=input_scale
        )
    except InputMappingError as exc:
        substrate["mapping_refused"] = str(exc)
        rows.append(
            criterion(
                "C1.1",
                "the input population is the declared ORN population",
                None,
                "exact equality",
                None,
                str(adjacency_path),
                note=str(exc),
            )
        )
        return {"substrate": substrate, "criteria": rows}, node_ids, annotation

    described = mapping.describe()
    substrate["mapping"] = described
    rows += [
        criterion(
            "C1.1",
            "every node receiving external input is an ORN, and the receiving set "
            "equals the declared ORN population",
            {
                "n_support_rows": described["n_support_rows"],
                "n_orn": described["n_orn"],
                "support_is_orn_only": described["support_is_orn_only"],
            },
            "exact equality",
            bool(described["support_is_orn_only"]),
            str(node_meta),
        ),
        criterion(
            "C1.2",
            "input-map density nnz(W_in)/(N*Din)",
            described["density"],
            f"<= {ORN_DENSITY_LIMIT}",
            bool(described["density"] <= ORN_DENSITY_LIMIT + 1e-12),
            "computed from the built mapping",
            note=(
                f"one non-zero per ORN, so this is the ORN fraction "
                f"{described['orn_fraction']:.6f} over Din={din} "
                f"(bound {described['max_orn_fraction_allowed']:.2f})"
            ),
        ),
        criterion(
            "C1.5a",
            "no W_in support on any PN row",
            described["n_support_rows_on_pn"],
            0,
            described["n_support_rows_on_pn"] == 0,
            "computed from the built mapping",
        ),
        criterion(
            "C1.5b",
            "PNs receive signal only through REAL ORN->PN edges of the substrate",
            {
                "n_orn_to_pn_edges": described["n_orn_to_pn_edges"],
                "n_pn": described["n_pn"],
                "n_pn_receiving_a_real_orn_edge": described[
                    "n_pn_receiving_a_real_orn_edge"
                ],
            },
            "every receiving PN has >= 1 real edge",
            (
                described["n_pn_receiving_a_real_orn_edge"] > 0
                if described["n_pn"] > 0
                else None
            ),
            str(adjacency_path),
            note=(
                f"{described['n_pn'] - described['n_pn_receiving_a_real_orn_edge']} of "
                f"{described['n_pn']} PNs receive NO real ORN edge, so the admissible "
                f"receiving-PN pool is the "
                f"{described['n_pn_receiving_a_real_orn_edge']}-node ORN-recipient subset"
            ),
        ),
        criterion(
            "C1.6",
            "the input mapping is a named, declared object with a reproducible digest",
            {
                "name": mapping.name,
                "declared_input_population": described["declared_input_population"],
                "w_in_sha256": described["w_in_sha256"][:16],
                "annotation_sha256": described["annotation"]["mapping_sha256"][:16],
            },
            f"name == {INPUT_MAPPING_ORN_ALIGNED!r}",
            mapping.name == INPUT_MAPPING_ORN_ALIGNED,
            "computed from the built mapping",
        ),
    ]
    assert mapping.name == INPUT_MAPPING_ORN_ALIGNED or not rows
    return {"substrate": substrate, "criteria": rows}, node_ids, annotation


def measure_v1_selections(
    node_meta: Path, sizes: tuple[int, ...], din: int, seed: int
) -> dict:
    """Whether the delivered v1 substrate can carry the pathway, per size.

    The selection function needs the graph's ``node_ids``. When the adjacency is
    absent the root ids are read from the committed node metadata, whose
    ``node_idx`` column IS the graph's row order (verified: the digests this
    reproduces match the delivered records exactly); the reconstruction is
    labelled as such and the EDGE-dependent criteria are left NOT MEASURED.
    """
    import pandas as pd

    from connectome.select_neurons import select_neurons

    meta = pd.read_csv(node_meta)
    root_ids = np.array([str(int(v)) for v in meta["root_id"]])
    classes = load_node_classes(node_meta)
    annotation = CellTypeAnnotation(
        classes=classes,
        source=str(node_meta),
        per_class_counts={
            name: sum(1 for v in classes.values() if v == name)
            for name in set(classes.values())
        },
    )
    delivered = delivered_selection_digests()
    rows: list[dict] = []
    for size in sizes:
        try:
            selection = select_neurons(_stub_npz(root_ids), node_meta, int(size), SELECTION_SEED)
            selected = np.asarray(selection["root_ids"]).astype(str)
        except Exception as exc:  # pragma: no cover - reported, never swallowed
            rows.append(
                {
                    "n": int(size),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        counts = population_counts(annotation, selected)
        orn_threshold = C1_4_THRESHOLDS["ORN"](din)
        kc_threshold = C1_4_THRESHOLDS["KC"](din)
        rows.append(
            {
                "n": int(size),
                "populations": counts,
                "selection_sha256": str(selection["sha256"]),
                "selection_digest_matches_delivered": (
                    None
                    if int(size) not in delivered
                    else str(selection["sha256"]) == delivered[int(size)]
                ),
                "delivered_selection_sha256": delivered.get(int(size)),
                "mapping_buildable": counts.get("ORN", 0) >= din and counts.get("PN", 0) > 0,
                "refusal": (
                    f"{counts.get('ORN', 0)} ORN row(s) for {din} channel(s): one "
                    f"channel per ORN needs at least one ORN per channel"
                    if counts.get("ORN", 0) < din
                    else ""
                ),
                "c1_4_orn": {
                    "observed": counts.get("ORN", 0),
                    "threshold": orn_threshold,
                    "pass": counts.get("ORN", 0) >= orn_threshold,
                },
                "c1_4_kc": {
                    "observed": counts.get("KC", 0),
                    "threshold": kc_threshold,
                    "pass": counts.get("KC", 0) >= kc_threshold,
                },
                "c1_4_pn": {
                    "observed": counts.get("PN", 0),
                    "threshold": C1_4_THRESHOLDS["PN"](din),
                    "pass": counts.get("PN", 0) >= C1_4_THRESHOLDS["PN"](din),
                },
                "c1_5b": "NOT MEASURED: needs the adjacency (edge structure)",
            }
        )
    return {
        "substrate": "delivered v1 selections (connectome.select_neurons, "
        f"seed={SELECTION_SEED})",
        "din": din,
        "node_ids_source": (
            "reconstructed from the committed node metadata (node_idx is the graph "
            "row order); the digests below are cross-checked against the delivered "
            "records by tests/test_input_mapping_c1.py and the C1 report"
        ),
        "sizes": rows,
    }


def _stub_npz(root_ids: np.ndarray) -> Path:
    """A minimal ``node_ids``-carrying NPZ for the selection function.

    ``connectome.select_neurons`` reads only ``node_ids`` from the graph file (the
    adjacency itself is not needed to CHOOSE nodes), so the committed metadata is
    enough to reproduce the delivered selections when the 775 MB adjacency is not
    on this machine. The ids are written as **int64**, which is how the delivered
    NPZ stores them: the selection digest is a SHA-256 of the id BYTES, so an
    ``S32`` stub produces a different (and meaningless) digest for the same
    selection. With int64 all five delivered digests reproduce exactly.
    """
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    np.savez(handle.name, node_ids=np.asarray(root_ids, dtype=np.int64))
    return Path(handle.name)


def delivered_selection_digests() -> dict[int, str]:
    """``target_n -> sha256`` from the DELIVERED v1 records, when they are present.

    Used to verify that this report reproduces the substrates the project actually
    ran, rather than a similar-looking selection of its own.
    """
    pattern = "results/audit/m4_audit/server_evidence/ev/e9_size_d2*/d2_beef_uncontrolled/R0/*.json"
    out: dict[int, str] = {}
    for path in sorted(REPO_ROOT.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - a corrupt copy is not a digest
            continue
        selection = (payload.get("model_description") or {}).get("node_selection") or {}
        digest = selection.get("sha256_sorted_root_ids")
        target = selection.get("target_n")
        if digest and target is not None:
            out[int(target)] = str(digest)
    return out


def _load_graph(adjacency_path: Path):
    """The delivered adjacency as a CSR matrix plus its node ids."""
    with np.load(adjacency_path, allow_pickle=True) as npz:
        shape = tuple(int(v) for v in npz["adj_shape"])
        node_ids = np.asarray(npz["node_ids"])
        matrix = sp.csr_matrix(
            (np.asarray(npz["adj_data"], dtype=float),
             np.asarray(npz["adj_indices"]),
             np.asarray(npz["adj_indptr"])),
            shape=shape,
        )
    return matrix, node_ids


def _declared_annotation(node_meta: Path) -> CellTypeAnnotation:
    classes = load_node_classes(node_meta)
    counts: dict[str, int] = {}
    for value in classes.values():
        counts[value] = counts.get(value, 0) + 1
    return CellTypeAnnotation(classes=classes, source=str(node_meta), per_class_counts=counts)


def measure_c2(
    adjacency_path: Path,
    node_meta: Path,
    *,
    din: int,
    target_n: int,
    normalization: str,
    seed: int,
) -> dict:
    """C2.1-C2.6 on the delivered graph: expand from the ORNs, then measure.

    The v1 selection is measured on the same graph and at the same N, so the two
    selections are compared like for like -- that comparison IS the finding.
    """
    matrix, node_ids = _load_graph(adjacency_path)
    annotation = _declared_annotation(node_meta)

    expansion = expand_from_orns(matrix, node_ids, annotation, target_n, din=din, seed=seed)
    quality = subgraph_quality(matrix, expansion.node_indices)

    # E_kept: what the reservoir matrix actually carries, i.e. the same induced
    # block after the declared DATA-3 normalization. If the normalization dropped
    # wiring, retention < 1 and that is the point of the check.
    from drososense.reservoir.connectome_reservoir import load_reservoir_topology_from_npz

    topology = load_reservoir_topology_from_npz(
        adjacency_path, normalization=normalization, node_indices=expansion.node_indices
    )
    built = topology.matrix.tocsr()
    built_self_loops = int(np.asarray(built.diagonal() != 0).sum())
    built_eligible = int(built.nnz - built_self_loops)
    retention_after_build = (
        built_eligible / quality["induced_eligible_edges"]
        if quality["induced_eligible_edges"]
        else float("nan")
    )

    v1 = select_nodes(adjacency_path, target_n=target_n, seed=SELECTION_SEED)
    v1_quality = subgraph_quality(matrix, v1.node_indices)
    v1_mapping_refusal = ""
    try:
        build_orn_aligned_mapping(
            matrix[node_indices_or_all(matrix, v1.node_indices)],
            node_ids[v1.node_indices],
            annotation,
            din,
            seed=seed,
        )
    except InputMappingError as exc:
        v1_mapping_refusal = str(exc)

    # C2.1 is C1.4 applied to THIS substrate: measured from the classes of the nodes
    # actually selected, not from the allocation the expansion intended.
    selected_classes = annotation.class_of(node_ids[expansion.node_indices])
    composition: dict[str, int] = {}
    for value in selected_classes.tolist():
        composition[str(value)] = composition.get(str(value), 0) + 1
    composition["MBON+DAN+higher_order"] = sum(
        composition.get(name, 0) for name in ("MBON", "DAN", "higher_order")
    )
    c2_1_checks = {
        name: {
            "observed": int(composition.get(name, 0)),
            "threshold": int(threshold_fn(din)),
            "pass": bool(composition.get(name, 0) >= threshold_fn(din)),
        }
        for name, threshold_fn in C1_4_THRESHOLDS.items()
    }
    c2_1_pass = all(check["pass"] for check in c2_1_checks.values())

    rows = [
        criterion(
            "C2.1",
            "input population present and typed in the selected subgraph (C1.4)",
            {"composition": composition, "checks": c2_1_checks},
            {name: threshold_fn(din) for name, threshold_fn in C1_4_THRESHOLDS.items()},
            c2_1_pass,
            f"{adjacency_path} + {node_meta}",
            note=(
                "measured from the classes of the nodes actually selected; the "
                "expansion never adds an untyped node, and it assigns every declared "
                "layer group at least its C1.4 floor before sharing the remainder"
            ),
        ),
        criterion(
            "C2.2",
            "eligible-edge retention (denominator = induced eligible edges)",
            {
                "before_normalization": quality["eligible_edge_retention"],
                "after_the_declared_normalization": retention_after_build,
                "induced_eligible_edges": quality["induced_eligible_edges"],
                "kept_eligible_edges": built_eligible,
                "normalization": normalization,
            },
            ">= 0.90",
            bool(retention_after_build >= 0.90),
            f"{adjacency_path} + the built R0 matrix",
            note=quality["eligible_edge_retention_denominator"],
        ),
        criterion(
            "C2.3",
            "largest weak component / N",
            quality["largest_weak_component_fraction"],
            ">= 0.90",
            bool(quality["largest_weak_component_fraction"] >= 0.90),
            "computed on the induced subgraph",
        ),
        criterion(
            "C2.4",
            "isolated fraction",
            quality["isolated_fraction"],
            "<= 0.02",
            bool(quality["isolated_fraction"] <= 0.02),
            "computed on the induced subgraph",
            note=f"{quality['n_isolated']} isolated node(s)",
        ),
        criterion(
            "C2.5",
            "mean in-subgraph unweighted out-degree",
            quality["mean_unweighted_out_degree"],
            ">= 2.0",
            bool(quality["mean_unweighted_out_degree"] >= 2.0),
            "computed on the induced subgraph",
            note=f"median {quality['median_unweighted_out_degree']}",
        ),
        criterion(
            "C2.6",
            "deterministic, declared selection with target_n, seed and sha256 on the record",
            {
                "method": expansion.method,
                "target_n": int(expansion.target_n),
                "sha256_sorted_root_ids": expansion.sha256[:16],
                "consumes_randomness": expansion.detail["consumes_randomness"],
            },
            "declared + reproducible",
            bool(expansion.detail["consumes_randomness"] is False and expansion.sha256),
            "the NodeSelection's own describe()",
        ),
    ]

    # determinism, measured rather than asserted
    repeat = expand_from_orns(matrix, node_ids, annotation, target_n, din=din, seed=seed)
    deterministic = bool(
        np.array_equal(repeat.node_indices, expansion.node_indices)
        and repeat.sha256 == expansion.sha256
    )

    return {
        "report_schema": REPORT_SCHEMA_C2,
        "settings": {
            "target_n": int(target_n),
            "din": int(din),
            "normalization": normalization,
            "selection_seed": SELECTION_SEED,
            "mapping_seed": int(seed),
        },
        "expansion": {
            "composition": composition,
            "selection": expansion.describe(),
            "quality": quality,
            "retention_after_the_declared_normalization": retention_after_build,
            "deterministic_on_repeat": deterministic,
        },
        "v1_selection_at_the_same_n": {
            "selection": v1.describe(),
            "quality": v1_quality,
            "orn_aligned_mapping_refusal": v1_mapping_refusal,
        },
        "criteria": rows,
    }


def node_indices_or_all(matrix, indices):
    """The rows of ``indices`` (identity when the selection is the whole graph)."""
    return np.asarray(indices, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="C1")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--adjacency", default=None)
    parser.add_argument(
        "--node-meta",
        default=str(REPO_ROOT / "connectome/metadata/olfactory_v1_node_meta.csv"),
    )
    parser.add_argument("--classes-csv", default=None, help="classified edge metadata (cross-check)")
    parser.add_argument("--din", type=int, default=5, help="input channel count")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--target-n", type=int, default=1000, help="C2 substrate size")
    parser.add_argument("--normalization", default="n1_pre_l1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    wanted = {part.strip().upper() for part in str(args.only).split(",") if part.strip()}
    unknown = wanted - {"C1", "C2"}
    if unknown:
        raise SystemExit(f"--only {args.only!r}: unknown section(s) {sorted(unknown)}")

    data_root = Path(args.data_root)
    adjacency = (
        Path(args.adjacency)
        if args.adjacency
        else data_root / "connectome/adjacency/olfactory_v1.npz"
    )
    node_meta = Path(args.node_meta)
    sizes = tuple(int(v) for v in str(args.sizes).split(",") if v.strip())

    if wanted == {"C2"}:
        report = {
            "report_schema": REPORT_SCHEMA_C2,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provenance": {
                "git_head": git_head(),
                "validation_only": True,
                "touches_test_split": False,
                "fits_no_model": True,
            },
        }
        if not adjacency.is_file():
            raise SystemExit(f"--only C2 needs the adjacency; not found at {adjacency}")
        report["C2"] = measure_c2(
            adjacency,
            node_meta,
            din=args.din,
            target_n=args.target_n,
            normalization=args.normalization,
            seed=args.seed,
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "C2_subgraph.json"
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(target)
        for row in report["C2"]["criteria"]:
            flag = {True: "PASS", False: "FAIL", None: "NOT MEASURED"}[row["pass"]]
            print(f"  {row['criterion']:6s} {flag}  observed={str(row['observed'])[:100]}")
        for key in ("expansion", "v1_selection_at_the_same_n"):
            q = report["C2"][key]["quality"]
            print(
                f"  {key:28s} N={q['n_nodes']} edges={q['induced_eligible_edges']} "
                f"retention={q['eligible_edge_retention']:.4f} "
                f"wcc={q['largest_weak_component_fraction']:.4f} "
                f"isolated={q['isolated_fraction']:.4f} "
                f"mean_out_deg={q['mean_unweighted_out_degree']:.3f} "
                f"enrichment={q['enrichment']:.2f}"
            )
        return 0

    report: dict = {
        "report_schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head": git_head(),
            "validation_only": True,
            "touches_test_split": False,
            "fits_no_model": True,
            "note": (
                "every number here is a property of the construction, measured on "
                "delivered data; no model is fitted, no test split is read, and no "
                "record is written"
            ),
        },
        "settings": {
            "din": args.din,
            "sizes": list(sizes),
            "selection_seed": SELECTION_SEED,
            "mapping_seed": args.seed,
            "input_scale": args.input_scale,
            "orn_density_limit": ORN_DENSITY_LIMIT,
        },
    }

    if adjacency.is_file():
        full, node_ids, annotation = measure_full_graph(
            adjacency, node_meta, args.din, args.seed, args.input_scale
        )
        report["construct_substrate"] = full
        report["edge_criteria_measured"] = True
    else:
        report["construct_substrate"] = {
            "substrate": "full delivered olfactory graph",
            "criteria": [],
            "why_absent": f"adjacency not present at {adjacency}",
            "note": (
                "the edge-dependent criteria (C1.5b) are NOT MEASURED without the "
                "adjacency; they are not reported as satisfied"
            ),
        }
        report["edge_criteria_measured"] = False
        from drososense.reservoir.input_mapping import load_declared_annotation

        annotation = load_declared_annotation(node_meta)

    report["v1_selections"] = measure_v1_selections(node_meta, sizes, args.din, args.seed)

    if args.classes_csv:
        path = Path(args.classes_csv)
        if path.is_file():
            classified = load_classified_edge_annotation(path)
            report["vocabulary_cross_check"] = compare_annotations(annotation, classified)
        else:
            report["vocabulary_cross_check"] = {
                "why_absent": f"classified edge metadata not present at {path}"
            }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "C1_input_mapping.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(target)

    rows = report["construct_substrate"]["criteria"]
    for row in rows:
        flag = {True: "PASS", False: "FAIL", None: "NOT MEASURED"}[row["pass"]]
        print(f"  {row['criterion']:18s} {flag}")
    if report["v1_selections"]["sizes"]:
        print("  delivered v1 selections:")
        for row in report["v1_selections"]["sizes"]:
            if "error" in row:
                print(f"    N={row['n']:<6} ERROR {row['error']}")
                continue
            print(
                f"    N={row['n']:<6} ORN={row['populations'].get('ORN', 0):<4} "
                f"PN={row['populations'].get('PN', 0):<4} "
                f"KC={row['populations'].get('KC', 0):<4} "
                f"buildable={row['mapping_buildable']}"
            )
    try:
        shown = out_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:  # an out-dir outside the repository (a smoke run)
        shown = out_dir
    print(f"Evidence written under {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
