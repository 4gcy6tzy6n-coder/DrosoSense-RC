#!/usr/bin/env python
"""C1 — the cell-type annotation the v2 input pathway stands on (READ-ONLY).

v2 constraint 1 (`docs/v2_preregistration.md` §1) requires

    X (e-nose channels) -> ORN -> PN -> downstream

with only ORNs receiving external input and PNs receiving it only through REAL
ORN->PN connectome edges. Every acceptance criterion C1.x is therefore a statement
about the delivered annotation, and this script measures that annotation instead of
assuming it:

* which file carries the cell-type vocabulary, and which does not;
* the size of each typed population in the delivered adjacency;
* the REAL ORN->PN edge structure, including how many PNs receive an ORN edge at
  all -- the population from which a legitimate PN set can be drawn is bounded by
  that number, not by the PN class size;
* the numbers C1.4's thresholds are compared against.

It is validation-only: no model is fitted, no test split is read, and nothing is
written except the JSON report it is asked for.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/drososense/data-root")

#: The classes C1.4 names, in the vocabulary the delivered metadata uses.
DECLARED_CLASSES = ("ORN", "PN", "KC", "MBON", "DAN", "higher_order")

REPORT_SCHEMA = "c1_annotation_probe/1"


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return ""


def stream_classes(metadata: Path) -> tuple[dict[int, str], collections.Counter]:
    """The cell type of every node the classified edge metadata names.

    `pre_class` and `post_class` are checked for agreement per node rather than
    assumed: a node whose class depends on which end of an edge it sits is not a
    cell type, and the report has to say so if that ever happens.
    """
    classes: dict[int, str] = {}
    conflicts = collections.Counter()
    for chunk in pd.read_csv(
        metadata,
        usecols=["pre_root_id", "post_root_id", "pathway_class", "pre_class", "post_class"],
        dtype={"pathway_class": "string", "pre_class": "string", "post_class": "string"},
        chunksize=2_000_000,
    ):
        for column, id_column in (("pre_class", "pre_root_id"), ("post_class", "post_root_id")):
            for node, cls in zip(
                chunk[id_column].to_numpy(), chunk[column].fillna("<NA>").to_numpy()
            ):
                key = int(node)
                value = str(cls)
                known = classes.get(key)
                if known is None:
                    classes[key] = value
                elif known != value:
                    conflicts[f"{known}->{value}"] += 1
    return classes, conflicts


def adjacency_classes(adjacency: Path, classes: dict[int, str]):
    """Per-node classes aligned to the adjacency's own index space."""
    with np.load(adjacency, allow_pickle=True) as npz:
        shape = tuple(int(v) for v in npz["adj_shape"])
        node_ids = np.asarray(npz["node_ids"])
        indptr = np.asarray(npz["adj_indptr"])
        indices = np.asarray(npz["adj_indices"])
        nnz = int(np.asarray(npz["adj_data"]).size)
    aligned = np.array([classes.get(int(n), "<absent>") for n in node_ids], dtype=object)
    return shape, node_ids, indptr, indices, nnz, aligned


def measure(data_root: Path) -> dict:
    metadata = data_root / "connectome/metadata/olfactory_v1_edge_meta_classified.csv"
    adjacency = data_root / "connectome/adjacency/olfactory_v1.npz"
    ranking = data_root / "connectome/raw/neuron_class_ranking_df_783-olfactory-10000.feather"
    for required in (metadata, adjacency):
        if not required.is_file():
            raise SystemExit(f"required input absent: {required}")

    classes, conflicts = stream_classes(metadata)
    shape, node_ids, indptr, indices, nnz, aligned = adjacency_classes(adjacency, classes)

    population = collections.Counter(aligned.tolist())
    edge_pairs: collections.Counter = collections.Counter()
    out_edges = collections.Counter()
    orn_edges_per_pn: collections.Counter = collections.Counter()
    for row in range(shape[0]):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if start == stop:
            continue
        source = aligned[row]
        out_edges[source] += stop - start
        targets = aligned[indices[start:stop]]
        for target in targets:
            edge_pairs[(str(source), str(target))] += 1
        if source == "ORN":
            for index in np.flatnonzero(targets == "PN"):
                orn_edges_per_pn[int(node_ids[indices[start + index]])] += 1

    per_pn = np.array(sorted(orn_edges_per_pn.values()), dtype=int)
    ranking_columns: list[str] = []
    if ranking.is_file():
        ranking_columns = list(pd.read_feather(ranking).columns)

    return {
        "report_schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head": git_head(Path(__file__).resolve().parents[2]),
            "validation_only": True,
            "touches_test_split": False,
            "fits_no_model": True,
            "note": (
                "measures the DELIVERED annotation; it changes no data, no split, no "
                "endpoint and no record"
            ),
        },
        "sources": {
            "cell_type_vocabulary": str(metadata),
            "adjacency": str(adjacency),
            "modality_ranking": str(ranking) if ranking.is_file() else None,
            "modality_ranking_columns": ranking_columns,
            "modality_ranking_carries_class_labels": any(
                column in {"class", "cell_type", "pre_class", "post_class", "super_class"}
                for column in ranking_columns
            ),
        },
        "vocabulary": {
            "n_nodes_with_a_class": len(classes),
            "class_conflicts_pre_vs_post": dict(conflicts),
            "classes": sorted({str(v) for v in classes.values()}),
        },
        "adjacency": {
            "shape": list(shape),
            "nnz": nnz,
            "n_node_ids": int(node_ids.size),
            "mean_out_degree": nnz / shape[0] if shape[0] else None,
        },
        "population_in_adjacency": {
            name: int(population.get(name, 0))
            for name in (*DECLARED_CLASSES, "other", "<absent>")
        },
        "out_edges_by_source_class": {
            str(name): int(count) for name, count in out_edges.most_common()
        },
        "edge_pairs": [
            {"source_class": source, "target_class": target, "edges": int(count)}
            for (source, target), count in edge_pairs.most_common()
        ],
        "orn_to_pn": {
            "edges": int(edge_pairs.get(("ORN", "PN"), 0)),
            "pn_nodes_receiving_at_least_one_orn_edge": int(per_pn.size),
            "pn_population": int(population.get("PN", 0)),
            "pn_without_any_orn_edge": int(population.get("PN", 0)) - int(per_pn.size),
            "orn_edges_per_receiving_pn": {
                "min": int(per_pn.min()) if per_pn.size else 0,
                "median": float(np.median(per_pn)) if per_pn.size else 0.0,
                "max": int(per_pn.max()) if per_pn.size else 0,
            },
        },
        "criterion_inputs": {
            "note": (
                "the populations C1.4 can draw from, before any subgraph selection: "
                "the PN set admissible for C1.5 is the ORN-RECIPIENT subset, not the "
                "whole PN class"
            ),
            "orn_available": int(population.get("ORN", 0)),
            "pn_available": int(population.get("PN", 0)),
            "pn_available_with_real_orn_edge": int(per_pn.size),
            "kc_available": int(population.get("KC", 0)),
            "mbon_dan_higher_order_available": int(
                sum(population.get(name, 0) for name in ("MBON", "DAN", "higher_order"))
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=None, help="JSON report path (default: stdout summary only)")
    args = parser.parse_args(argv)

    report = measure(Path(args.data_root))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(target)

    print(f"adjacency  {report['adjacency']['shape']} nnz={report['adjacency']['nnz']:,}")
    print("populations (delivered adjacency):")
    for name, count in report["population_in_adjacency"].items():
        print(f"  {name:14s} {count:,}")
    orn = report["orn_to_pn"]
    print(
        f"ORN->PN real edges: {orn['edges']:,} reaching "
        f"{orn['pn_nodes_receiving_at_least_one_orn_edge']:,} of {orn['pn_population']:,} PNs "
        f"({orn['pn_without_any_orn_edge']:,} PNs receive none)"
    )
    print("class vocabulary source:", report["sources"]["cell_type_vocabulary"])
    print(
        "modality ranking carries class labels:",
        report["sources"]["modality_ranking_carries_class_labels"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
