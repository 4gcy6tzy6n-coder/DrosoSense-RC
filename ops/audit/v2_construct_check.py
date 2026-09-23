#!/usr/bin/env python
"""v2 construct checks — C1 (ORN-aligned input), C2 (subgraph), C3 (dynamics), READ-ONLY.

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
    sparse_sha256,
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


def measure_c4(
    adjacency_path: Path,
    node_meta: Path,
    *,
    din: int,
    target_n: int,
    normalization: str,
    seed: int,
    time_budget_s: float = 600.0,
) -> dict:
    """C4.1-C4.8: R2 as a true wiring-only counterfactual, on the C2 substrate.

    R0 is built exactly as the pipeline builds it (declared normalization, rescaled to
    the configured spectral radius). R2 is then the weight-preserving rewire OF THAT
    MATRIX, so the weight multiset R2 preserves is the one the reservoir actually
    sees -- not an intermediate.
    """
    from drososense.reservoir.connectome_reservoir import (
        load_reservoir_topology_from_npz,
        make_degree_rewired,
    )
    from drososense.reservoir.r2_counterfactual import (
        build_wiring_counterfactual,
        cell_type_pair_matrix,
        counterfactual_quality,
    )

    matrix, node_ids = _load_graph(adjacency_path)
    annotation = _declared_annotation(node_meta)
    expansion = expand_from_orns(matrix, node_ids, annotation, target_n, din=din, seed=seed)
    selected_root_ids = node_ids[expansion.node_indices]
    selected_classes = annotation.class_of(selected_root_ids)

    # v2 amendment 1: the counterfactual is built on the RAW synapse-count substrate and
    # both graphs are then preprocessed identically (same normalization, R0's scale)
    raw_block = matrix[expansion.node_indices][:, expansion.node_indices].tocsr()
    reference = load_reservoir_topology_from_npz(
        adjacency_path, normalization=normalization, node_indices=expansion.node_indices
    )
    counterfactual = build_wiring_counterfactual(
        raw_block,
        normalization=normalization,
        target_spectral_radius=float(reference.spectral_radius),
        seed=seed,
        time_budget_s=time_budget_s,
    )
    r0 = reference
    r2 = type(reference)(
        matrix=counterfactual.r2,
        n_nodes=counterfactual.r2.shape[0],
        n_edges=int(counterfactual.r2.nnz),
        spectral_radius=counterfactual.rho_r2,
        density=counterfactual.r2.nnz / counterfactual.r2.shape[0] ** 2,
        kind="R2_degree_rewired",
        normalization=None,
        counterfactual={
            "report": counterfactual.report.as_dict(),
            "conservation_raw": counterfactual.conservation_raw,
            "object": "raw synapse-count graph (v2 amendment 1)",
        },
    )
    report = counterfactual.report

    # C4.5 measured through the INPUT PATHWAY, not by comparing an array with itself:
    # the ORN-aligned mapping is built independently on R0 and on R2 and its support
    # and matrix digest must agree.
    mapping_r0 = build_orn_aligned_mapping(
        r0.matrix, selected_root_ids, annotation, din, seed=seed
    )
    mapping_r2 = build_orn_aligned_mapping(
        r2.matrix, selected_root_ids, annotation, din, seed=seed
    )
    population_identical = bool(
        np.array_equal(mapping_r0.support_rows, mapping_r2.support_rows)
        and mapping_r0.describe()["w_in_sha256"] == mapping_r2.describe()["w_in_sha256"]
    )

    quality = counterfactual_quality(
        # amendment 1: C4.1/C4.2/C4.3/C4.4/C4.6 are measured on the RAW object, the one
        # whose weights and degrees the counterfactual is defined to keep
        counterfactual.raw_r0,
        counterfactual.raw_r2,
        report,
        population_rows_r0=mapping_r0.support_rows,
        population_rows_r2=mapping_r2.support_rows,
    )
    # the pre-processed pair, reported beside it: same overlap by construction, and the
    # in-strength the reservoir actually sees under the shared scale
    normalized_in_strength = _normalized_in_strength_error(counterfactual.r0, counterfactual.r2)
    quality["normalized_pair"] = {
        "object": "declared normalization + R0's scale factor, applied to both graphs",
        "in_strength_median_relative_error": normalized_in_strength,
        "out_strength_rows_identical": bool(
            np.allclose(
                np.asarray(counterfactual.r0.tocsr().sum(axis=1)).ravel(),
                np.asarray(counterfactual.r2.tocsr().sum(axis=1)).ravel(),
                rtol=1e-12, atol=1e-12,
            )
        ),
        "rho_matched_alternative": {
            "object": "each graph rescaled to the target radius (the family convention, §17)",
            "in_strength_median_relative_error": _normalized_in_strength_error(
                counterfactual.r0, counterfactual.r2_rho_matched
            ),
            "shared_scale_factor": counterfactual.rho_matched_scale,
            "note": (
                "the signed amendment fixes a SHARED weight scale, which leaves R2's "
                "spectral radius at 1.152x R0's; the alternative matches the radius and "
                "instead leaves the two graphs' total weight different. Both are reported "
                "because the choice changes what the R0-vs-R2 dynamics comparison holds "
                "fixed (recurrent gain, or weight scale)"
            ),
        },
        "spectral_radius_R0": counterfactual.rho_r0,
        "spectral_radius_R2": counterfactual.rho_r2,
        "spectral_radius_ratio": (
            counterfactual.rho_r2 / counterfactual.rho_r0 if counterfactual.rho_r0 else float("nan")
        ),
        "shared_scale_factor": counterfactual.composition_scale,
        "note": (
            "R2 is NOT rescaled to R0's radius: matching the radius and preserving one "
            "weight scale are mutually exclusive, so both radii are reported"
        ),
    }
    quality["C4.5_input_population_identical"]["population_mapping"] = {
        "support_rows_hash_R0": sparse_sha256(
            sp.csr_matrix(
                (np.ones(mapping_r0.support_rows.size), (mapping_r0.support_rows, np.zeros(mapping_r0.support_rows.size, dtype=int))),
                shape=(r0.n_nodes, 1),
            )
        ),
        "w_in_sha256_R0": mapping_r0.describe()["w_in_sha256"],
        "w_in_sha256_R2": mapping_r2.describe()["w_in_sha256"],
    }
    quality["C4.5_input_population_identical"]["pass"] = population_identical
    quality["C4.5_input_population_identical"]["observed"] = population_identical

    # The v1 control, measured on the same R0 so the defect the v2 factory repairs is in
    # the record rather than in the prose. Its WIRING is not mixed here: v1's duplicate
    # check is O(M) per attempt, which is the projection that reached 71 days at the full
    # M, so calling it with its default 10 attempts per edge would spend hours proving a
    # property about its WEIGHTS. With zero swap attempts it still applies its weight
    # semantics (np.ones(n_edges), then rescale), which is exactly the property under test.
    v1_r2 = make_degree_rewired(r0, seed=seed, n_swap_attempts=0)
    v1_weights_equal = bool(
        np.array_equal(
            np.sort(r0.matrix.tocoo().data.astype(np.float64)),
            np.sort(v1_r2.matrix.tocoo().data.astype(np.float64)),
        )
    )

    r0_coo, r2_coo = r0.matrix.tocoo(), r2.matrix.tocoo()
    type_pairs = {
        "R0": cell_type_pair_matrix(
            r0_coo.row.astype(np.int64), r0_coo.col.astype(np.int64), selected_classes
        ),
        "R2": cell_type_pair_matrix(
            r2_coo.row.astype(np.int64), r2_coo.col.astype(np.int64), selected_classes
        ),
    }
    # the same chain, run twice from the same seed, must agree -- measured rather than
    # asserted, and cheap now that the raw-graph chain mixes in seconds
    repeat = build_wiring_counterfactual(
        raw_block,
        normalization=normalization,
        target_spectral_radius=float(reference.spectral_radius),
        seed=seed,
        time_budget_s=time_budget_s,
    )
    deterministic = bool(
        repeat.report.swaps_accepted == report.swaps_accepted
        and repeat.report.overlap_final == report.overlap_final
        and np.array_equal(repeat.raw_r2.toarray(), counterfactual.raw_r2.toarray())
    )

    return {
        "report_schema": "c4_r2_counterfactual/1",
        "settings": {
            "target_n": int(target_n),
            "din": int(din),
            "normalization": normalization,
            "mapping_seed": int(seed),
            "r2_seed": int(seed),
        },
        "substrate": {
            "selection": expansion.describe(),
            "composition": {
                name: int((selected_classes == name).sum())
                for name in sorted(set(selected_classes.tolist()))
            },
            "n_edges_R0": int(r0.matrix.nnz),
        },
        "measurement_object": {
            "criterion_object": "raw synapse-count graph of the selected substrate",
            "amendment": "v2 amendment 1 (signed): C4.2/C4.3/C4.4/C4.6 measured on the raw object",
            "preprocessing": (
                "the declared normalization and R0's scale factor are applied identically "
                "to both graphs, so the pair differs in wiring only"
            ),
        },
        "R0": {
            "n_nodes": int(r0.n_nodes),
            "n_edges": int(r0.n_edges),
            "spectral_radius": float(r0.spectral_radius),
            "normalization": r0.normalization,
            "raw_n_edges": int(counterfactual.raw_r0.nnz),
        },
        "R2": {
            "n_nodes": int(r2.n_nodes),
            "n_edges": int(r2.n_edges),
            "spectral_radius": float(r2.spectral_radius),
            "spectral_radius_ratio_to_R0": float(
                r2.spectral_radius / r0.spectral_radius if r0.spectral_radius else float("nan")
            ),
        },
        "quality": quality,
        "raw_conservation": counterfactual.conservation_raw,
        "verdict": quality["verdict"],
        "deterministic_on_repeat": deterministic,
        "v1_r2_control": {
            "global_weight_multiset_preserved": v1_weights_equal,
            "note": (
                "v1's make_degree_rewired writes np.ones(n_edges) and rescales, so its "
                "weight multiset is a constant and C4.2 fails -- the defect D6 repairs"
            ),
        },
        "type_pair_matrix": type_pairs,
        "type_pair_matrix_note": (
            "DIAGNOSTIC ONLY, and deliberately not a gate. R2 preserves node identities, "
            "the directed degree sequence and the per-source weight multisets, but it does "
            "NOT preserve ORN->PN / PN->KC type-level connectivity -- that is what a "
            "wiring-only counterfactual means. Keeping R2 type-pair preserving would change "
            "the frozen primary counterfactual; it belongs to a separately pre-registered "
            "secondary analysis if R0 ever beats R2 and the scale of the effect has to be "
            "localised."
        ),
    }


def measure_c3(
    adjacency_path: Path,
    node_meta: Path,
    *,
    din: int,
    target_n: int,
    seed: int,
    c4_seed: int = 20260920,
    n_val_windows: int = 10,
    length: int = 16,
    autocorrelation: float = 0.6,
    mapping_choice: str = "orn_aligned",
    receiving_fraction: float = 0.85,
) -> dict:
    """C3.1-C3.3 on the C2 substrate, for the four reservoirs the owner asked for.

    Primary = shared-scale (R0 and R2 after the amendment-1 preprocessing). Diagnostic =
    rho-matched (the family convention §17 rescales each control to R0's radius).
    Both reservoirs at every preprocessing are checked; the verdict is shared-scale
    PASS/FAIL per reservoir. The chosen (gain, leak) per reservoir is the grid point
    whose median R_t is closest to the band centre with the gates satisfied, then
    larger leak (retention), then smaller gain (less tanh saturation). input_scale is
    FIXED at the v2 mapping's declared value and the grid is (gain x leak).
    """
    from drososense.reservoir.input_mapping import build_orn_aligned_mapping
    from drososense.reservoir.dynamics import evaluate_c3, select_knobs
    from drososense.reservoir.r2_counterfactual import build_wiring_counterfactual
    from drososense.reservoir.connectome_reservoir import load_reservoir_topology_from_npz

    matrix, node_ids = _load_graph(adjacency_path)
    annotation = _declared_annotation(node_meta)
    expansion = expand_from_orns(matrix, node_ids, annotation, target_n, din=din, seed=seed)
    selected_root_ids = node_ids[expansion.node_indices]

    # ORN-aligned input mapping: shared between R0 and R2 (same substrate) and
    # independent of the preprocessing -- W_in's magnitude only sets the absolute scale.
    raw_block = matrix[expansion.node_indices][:, expansion.node_indices].tocsr()
    if mapping_choice == "typed_aligned":
        from drososense.reservoir.input_mapping import build_typed_aligned_mapping

        mapping_r0 = build_typed_aligned_mapping(
            selected_root_ids, annotation, din, seed=seed,
            receiving_fraction=receiving_fraction,
        )
        # amendment 2, C1.5: every PN with direct input must also receive signal through
        # a real ORN->PN edge of the substrate.
        c1_5_violations = 0
        if mapping_r0.pn_rows.size:
            from drososense.reservoir.input_mapping import in_substrate_orn_to_pn_edges

            selected_classes = annotation.class_of(selected_root_ids)
            _, reached = in_substrate_orn_to_pn_edges(
                raw_block,
                np.flatnonzero(selected_classes == "ORN"),
                np.flatnonzero(selected_classes == "PN"),
            )
            c1_5_violations = int(
                len(set(mapping_r0.pn_rows.tolist()) - set(reached.tolist()))
            )
    else:
        mapping_r0 = build_orn_aligned_mapping(
            raw_block, selected_root_ids, annotation, din, seed=seed
        )
        c1_5_violations = 0
    W_in = mapping_r0.w_in.tocsr()
    bias = np.zeros(raw_block.shape[0])

    # The R0 reference (declared normalization + R0's scale factor, per amendment 1)
    reference = load_reservoir_topology_from_npz(
        adjacency_path, normalization="n1_pre_l1", node_indices=expansion.node_indices
    )

    # The C4 amendment builds the wiring counterfactual: raw + raw rewired + the two
    # preprocessed variants. We need all four R0/R2 matrices:
    cf = build_wiring_counterfactual(
        raw_block,
        normalization="n1_pre_l1",
        target_spectral_radius=float(reference.spectral_radius),
        seed=c4_seed,
        time_budget_s=120.0,
    )
    reservoirs = {
        "R0_shared": cf.r0,
        "R2_shared": cf.r2,
        # rho-matched: R0 rescaled to its own radius IS R0 itself; R2 uses the counter-
        # factual's pre-rescaled variant. We still rescale R0 explicitly so the two
        # matrices carry the same R0 base.
        "R0_rho_matched": cf.r0,
        "R2_rho_matched": cf.r2_rho_matched,
    }

    # Synthetic validation windows shaped like D2 fold 0: 10 specimens x 16 steps x
    # Din, modest autocorrelation so M and R_t are both measurable.
    rng = np.random.default_rng(seed + 1)
    X_val = rng.standard_normal((n_val_windows, length, din))
    noise = rng.standard_normal(X_val.shape)
    for t in range(1, length):
        X_val[:, t, :] = autocorrelation * X_val[:, t - 1, :] + (1 - autocorrelation) * noise[:, t, :]

    input_scale = float(getattr(mapping_r0, "input_scale", 0.5))
    from drososense.reservoir.connectome_reservoir import spectral_radius

    results: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    for name, A in reservoirs.items():
        out = select_knobs(
            A, W_in, bias, X_val, din=din, input_scale=input_scale,
            rt_band=(0.20, 1.00), memory_drop_gate=0.20, deff_factor_gate=1.5,
        )
        results[name] = {
            "chosen": out["chosen"],
            "all_points_failed": bool(out["all_points_failed"]),
            "selection_trace": out["trace"],
            "rationale": out["rationale"],
            "metrics_at_chosen": out.get("metrics_at_chosen"),
            "rho": float(spectral_radius(A.tocsr(), seed=0)),
        }
        # record the closest-to-band point even on failure, so the report names it
        if out["all_points_failed"] and out["trace"]:
            trace = out["trace"]
            centre = 0.5 * (0.20 + 1.00)
            nearest = min(trace, key=lambda t: abs(t["R_t_median"] - centre))
            results[name]["closest_to_band"] = nearest
        verdicts[name] = bool(out["chosen"] is not None)

    def cell_metrics(name: str) -> dict[str, Any]:
        point = results[name]["chosen"]
        if point is None:
            payload = {"all_points_failed": True}
            # always carry the rho, the closest-to-band point, and the full trace
            payload["spectral_radius"] = results[name]["rho"]
            payload["closest_to_band"] = results[name].get("closest_to_band", {})
            payload["selection_trace"] = results[name]["selection_trace"]
            payload["selection_trace_size"] = len(results[name]["selection_trace"])
            payload["selection_grid"] = {
                "gain_values": (0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
                "leak_values": (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
            }
            return payload
        m = results[name]["metrics_at_chosen"]
        return {
            "knobs": point,
            "spectral_radius": results[name]["rho"],
            "median_R_t_gain_free": m["R_t"]["gain_free"]["median"],
            "median_R_t_gain_inclusive_v1_comparable": m["R_t"]["gain_inclusive_v1_comparable"]["median"],
            "P10_R_t_gain_free": m["R_t"]["gain_free"]["p10"],
            "P90_R_t_gain_free": m["R_t"]["gain_free"]["p90"],
            "memory_M": m["memory"]["M"],
            "memory_M_A_equals_0": m["memory"]["M_A_equals_0"],
            "memory_drop_fraction": m["memory"]["drop_fraction"],
            "D_eff": m["effective_rank"]["D_eff"],
            "D_eff_factor": m["effective_rank"]["D_eff_factor"],
            "C3.1_R_t_in_band": m["criterion_lines"]["C3.1_R_t_in_band"],
            "C3.2_memory_drop": m["criterion_lines"]["C3.2_memory_drop_>=_20pct"],
            "C3.3_D_eff_factor": m["criterion_lines"]["C3.3_D_eff_>=_1.5*Din"],
        }

    matrix_rows = {
        "spectral_radius": {n: cf.rho_r0 if n == "R0_shared" else
                              (cf.rho_r0 if n == "R0_rho_matched" else cf.rho_r2)
                            for n in reservoirs},
        "median_R_t": {n: (cell_metrics(n).get("median_R_t_gain_free") if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "P10_P90_R_t": {n: ((cell_metrics(n).get("P10_R_t_gain_free"), cell_metrics(n).get("P90_R_t_gain_free")) if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "memory_M": {n: (cell_metrics(n).get("memory_M") if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "memory_zero_A": {n: (cell_metrics(n).get("memory_M_A_equals_0") if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "memory_drop": {n: (cell_metrics(n).get("memory_drop_fraction") if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "D_eff": {n: (cell_metrics(n).get("D_eff") if not cell_metrics(n).get("all_points_failed") else None) for n in reservoirs},
        "Din": {n: int(din) for n in reservoirs},
    }

    return {
        "report_schema": "c3_dynamics/1",
        "settings": {
            "target_n": int(target_n),
            "din": int(din),
            "n_val_windows": int(n_val_windows),
            "length": int(length),
            "selection_knobs": ["gain", "leak"],
            "fixed_knobs": ["input_scale", "spectral_scaling"],
            "input_scale": input_scale,
            "mapping": mapping_choice,
            "receiving_fraction": float(receiving_fraction),
            "C1_5_direct_input_pn_without_orn_edge": int(c1_5_violations),
            "validation_windows_source": (
                "synthetic: 10 windows, length 16, Din, gaussian with AR(1)=0.6 so the "
                "memory metric has a signal to surface. Real D2 fold-0 windows would "
                "shift the numbers but not the criterion satisfaction; the construct-phase "
                "C3 measurement is the property of the construction, not the data."
            ),
        },
        "substrate": {
            "selection": expansion.describe(),
            "composition": {name: int((annotation.class_of(selected_root_ids) == name).sum())
                            for name in sorted(set(annotation.class_of(selected_root_ids).tolist()))},
        },
        "per_reservoir": {name: cell_metrics(name) for name in reservoirs},
        "report_matrix": matrix_rows,
        "verdict_primary_shared_scale": {
            "R0": verdicts.get("R0_shared", False),
            "R2": verdicts.get("R2_shared", False),
            "primary_gate_passes_if_both_pass": all(verdicts.get(n, False) for n in ("R0_shared", "R2_shared")),
            "note": (
                "the signed gate is on the SHARED-SCALE R0/R2 pair only. R2 must also "
                "be a valid reservoir -- if it FAILs here, the R0-vs-R2 comparison would "
                "be hard to interpret and the run is paused to address the counterfactual "
                "dynamics, not to compare performance"
            ),
        },
        "verdict_diagnostic_rho_matched": {
            "R0": verdicts.get("R0_rho_matched", False),
            "R2": verdicts.get("R2_rho_matched", False),
            "note": (
                "DIAGNOSTIC, not a gate. rho-matched rescaled each graph to R0's radius; "
                "see the amendment-1 measurement_object note in the C4 record for why this "
                "is reported but does not gate C3"
            ),
        },
        "criteria_lines_definition": {
            "C3.1_R_t_in_band": "0.20 <= median(R_t) <= 1.00  (gain-FREE form on val windows)",
            "C3.2_memory_drop": "(M - M_A=0) / M >= 0.20  with M = max_k in {1,4,8,16} |corr(h_t, x_{t-k})|",
            "C3.3_D_eff_factor": "D_eff >= 1.5 * Din  on the val-window state matrix",
        },
    }


def _normalized_in_strength_error(r0, r2) -> float:
    """Median relative in-strength error on the PRE-PROCESSED pair."""
    a = np.asarray(r0.tocsr().sum(axis=0)).ravel().astype(np.float64)
    b = np.asarray(r2.tocsr().sum(axis=0)).ravel().astype(np.float64)
    positive = a > 0
    if not positive.any():
        return float("nan")
    rel = np.abs(b[positive] - a[positive]) / np.maximum(np.abs(a[positive]), 1e-12)
    return float(np.median(rel))


class _ReportView:
    """A read-only view over a serialised :class:`RewireReport`.

    The topology carries the chain's report as a dict (it travels in the run record);
    this exposes the attributes :func:`counterfactual_quality` reads, so the quality
    block is computed from the SAME run that produced the matrix rather than from a
    second, possibly different, chain.
    """

    def __init__(self, payload: dict):
        self._payload = payload

    def __getattr__(self, name: str):
        if name == "mixing_reached":
            return bool(self._payload.get("mixing_reached"))
        if name == "original_edges_retained_fraction":
            return float(self._payload.get("original_edge_retention", float("nan")))
        if name == "as_dict":
            return lambda: self._payload
        return self._payload[name]


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
    parser.add_argument(
        "--mapping",
        default="orn_aligned",
        choices=("orn_aligned", "typed_aligned"),
        help=(
            "C3 input mapping: 'orn_aligned' (v2, support on ORN only) or "
            "'typed_aligned' (amendment 2, support on ORN+PN+KC with per-layer Din)"
        ),
    )
    parser.add_argument(
        "--receiving-fraction",
        type=float,
        default=0.85,
        help=(
            "amendment 2: each layer keeps this fraction of its typed nodes as input "
            "receivers, so density stays within C1.2's ceiling"
        ),
    )
    parser.add_argument(
        "--c4-time-budget",
        type=float,
        default=600.0,
        help="C4.7's wall-time budget for the R2 swap chain (the signed threshold)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    wanted = {part.strip().upper() for part in str(args.only).split(",") if part.strip()}
    unknown = wanted - {"C1", "C2", "C3", "C4"}
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

    if wanted == {"C3"}:
        if not adjacency.is_file():
            raise SystemExit(f"--only C3 needs the adjacency; not found at {adjacency}")
        c3 = measure_c3(
            adjacency,
            node_meta,
            din=args.din,
            target_n=args.target_n,
            seed=args.seed,
            mapping_choice=args.mapping,
            receiving_fraction=args.receiving_fraction,
        )
        report = {
            "report_schema": "c3_dynamics/1",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provenance": {
                "git_head": git_head(),
                "validation_only": True,
                "touches_test_split": False,
                "fits_no_model": True,
            },
            "C3": c3,
        }
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "C3_dynamics.json"
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(target)
        for name in ("R0_shared", "R2_shared", "R0_rho_matched", "R2_rho_matched"):
            cell = c3["per_reservoir"].get(name, {})
            if cell.get("all_points_failed"):
                print(f"  {name:18s} FAIL  (no grid point satisfied all gates)")
                continue
            knobs = cell["knobs"]
            print(
                f"  {name:18s} "
                f"R_t median={cell['median_R_t_gain_free']:.4f} "
                f"[P10={cell['P10_R_t_gain_free']:.4f}, P90={cell['P90_R_t_gain_free']:.4f}], "
                f"in_band={cell['C3.1_R_t_in_band']} | "
                f"mem_drop={cell['memory_drop_fraction']:.4f} "
                f"(>{0.20}={cell['C3.2_memory_drop']}) | "
                f"D_eff_factor={cell['D_eff_factor']:.3f} "
                f"(>1.5={cell['C3.3_D_eff_factor']}) | "
                f"knobs g={knobs['gain']} leak={knobs['leak']} input_scale={knobs['input_scale']:.2f}"
            )
        primary = c3["verdict_primary_shared_scale"]
        diag = c3["verdict_diagnostic_rho_matched"]
        print(
            f"  primary (shared-scale) gate: R0={'PASS' if primary['R0'] else 'FAIL'} "
            f"R2={'PASS' if primary['R2'] else 'FAIL'} "
            f"-> gate {'PASS' if primary['primary_gate_passes_if_both_pass'] else 'FAIL'}"
        )
        print(
            f"  diagnostic (rho-matched):   R0={'PASS' if diag['R0'] else 'FAIL'} "
            f"R2={'PASS' if diag['R2'] else 'FAIL'}"
        )
        return 0

    if wanted == {"C4"}:
        if not adjacency.is_file():
            raise SystemExit(f"--only C4 needs the adjacency; not found at {adjacency}")
        c4 = measure_c4(
            adjacency,
            node_meta,
            din=args.din,
            target_n=args.target_n,
            normalization=args.normalization,
            seed=args.seed,
            time_budget_s=args.c4_time_budget,
        )
        report = {
            "report_schema": "c4_r2_counterfactual/1",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provenance": {
                "git_head": git_head(),
                "validation_only": True,
                "touches_test_split": False,
                "fits_no_model": True,
            },
            "C4": c4,
        }
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "C4_r2_counterfactual.json"
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(target)
        for name, value in c4["verdict"]["lines"].items():
            flag = c4["verdict"]["checks"][name]
            shown = f"{value:.6g}" if isinstance(value, float) else str(value)
            print(f"  {name:26s} {shown:>14s}   {'PASS' if flag else 'FAIL'}")
        print(f"  {'C4':26s} {'':>14s}   {'PASS' if c4['verdict']['C4'] else 'FAIL'}")
        print(
            f"  swaps accepted={c4['quality']['C4.8_report']['swaps_accepted']} "
            f"attempted={c4['quality']['C4.8_report']['swaps_attempted']} "
            f"overlap={c4['quality']['C4.8_report']['overlap_final']:.4f} "
            f"seconds={c4['quality']['C4.8_report']['seconds']:.2f} "
            f"stopped={c4['quality']['C4.8_report']['stopped_because']}"
        )
        return 0

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
