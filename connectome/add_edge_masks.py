#!/usr/bin/env python3
"""
Add edge pathway classification to olfactory_v1_edge_meta.csv.

Adds three columns per R1.2 §2 (DATA-9 §3.2):
  pathway_class ∈ {forward, modulatory, lateral, other}
  pre_class      — cell type of pre-synaptic neuron
  post_class     — cell type of post-synaptic neuron

Classification is based on layer_mean from neuron_class_ranking_df_783-olfactory-10000.feather.
The layer_mean is a continuous ranking score; we define tier boundaries to map to cell types.

Tier boundaries (approximate, derived from DATA-9 biological scope):
  layer_mean ≈ 1.0  → ORN   (olfactory receptor neuron)
  layer_mean ≈ 2.0  → PN    (projection neuron / ALPN)
  layer_mean ∈ (3, 4] → higher-order olfactory (uPN, mPN, LHN, etc.)
  layer_mean ≈ 4.5  → DAN   (dopaminergic neuron)
  layer_mean ≈ 5.0  → KC    (Kenyon cell)
  layer_mean ≈ 6.0  → MBON  (MB output neuron)
  layer_mean > 6.0  → other/higher-order

Pathway rules:
  forward    : pre_class in {ORN, PN, KC, higher-order} AND
               post_class in {PN, KC, MBON, higher-order} AND
               follows sensory pathway direction (pre layer <= post layer or cross-tier forward)
  modulatory : pre_class in {DAN, APL} OR post_class in {DAN, APL}
  lateral    : pre_class == post_class in {higher-order, other} at similar layers
  other      : anything not matching above

Usage:
  python connectome/add_edge_masks.py [--write]

Run --write to actually write the classified CSV (large file; default is dry-run).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_types import DECLARED_LAYER_TIERS, layer_to_class  # noqa: E402
from paths import metadata_path, raw_path  # noqa: E402

EDGE_CSV = metadata_path("olfactory_v1_edge_meta.csv")
OLF_TABLE = raw_path("neuron_class_ranking_df_783-olfactory-10000.feather")

# ── Tier definitions ────────────────────────────────────────────────────────
# The vocabulary is DECLARED ONCE, in ``connectome/cell_types.py``, so this
# classifier and the v2 construct phase cannot drift apart: they used to hold two
# copies of the same rule and disagreed on 32 of 124,185 nodes. ``layer_mean`` is a
# per-neuron olfactory-modality SCORE from the ranking table and the boundaries are
# convention, not measurement -- a receptor-level claim would need data this bundle
# does not carry.
_TIER_BOUNDARIES = DECLARED_LAYER_TIERS


# ── Pathway classification ──────────────────────────────────────────────────────
def classify_pathway(pre_class: str, post_class: str) -> str:
    """Classify edge pathway direction.

    forward    : main olfactory circuit direction
                  ORN→PN, ORN→higher-order, PN→KC, PN→MBON, KC→MBON,
                  higher-order→PN, higher-order→KC, higher-order→MBON
    modulatory : involves DAN or APL
    lateral    : same class at similar tier (excluding self-loops and unknown)
    other      : anything not matching above, including unknown/unclassified
    """
    # Unknown class → cannot classify
    if pre_class == "unknown" or post_class == "unknown":
        return "other"

    # Modulatory: any DAN or APL involvement
    if pre_class in ("DAN", "APL") or post_class in ("DAN", "APL"):
        return "modulatory"

    # Forward: follows sensory pathway hierarchy
    forward_pairs = {
        ("ORN", "PN"), ("ORN", "higher_order"),
        ("PN", "KC"), ("PN", "MBON"), ("PN", "higher_order"),
        ("KC", "MBON"), ("KC", "higher_order"),
        ("higher_order", "PN"), ("higher_order", "KC"),
        ("higher_order", "MBON"),
    }
    if (pre_class, post_class) in forward_pairs:
        return "forward"

    # Lateral: same class at same/similar tier (excluding self-loops)
    if pre_class == post_class:
        return "lateral"

    return "other"


def main():
    parser = argparse.ArgumentParser(description="Add pathway classification to edge_meta.csv")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write classified CSV to data-root (default is dry-run with count summary)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int, default=1_000_000,
        help="Chunk size for reading edge CSV (default 1M rows)"
    )
    pargs = parser.parse_args()

    print(f"Loading neuron class table: {OLF_TABLE}", file=sys.stderr)
    olf = pd.read_feather(OLF_TABLE, columns=["root_id", "layer_mean"])
    root_to_class = {row["root_id"]: layer_to_class(row["layer_mean"]) for _, row in olf.iterrows()}
    class_counts = {}
    for c in root_to_class.values():
        class_counts[c] = class_counts.get(c, 0) + 1
    print(f"  Neuron class distribution: {class_counts}", file=sys.stderr)
    print(f"  Total neurons: {len(root_to_class):,}", file=sys.stderr)

    print(f"\nReading edge CSV: {EDGE_CSV}", file=sys.stderr)
    print(f"  (dry-run mode — use --write to save)", file=sys.stderr)

    pathway_counts = {"forward": 0, "modulatory": 0, "lateral": 0, "other": 0}
    chunk_iter = pd.read_csv(EDGE_CSV, chunksize=pargs.chunk_size)

    total_rows = 0
    for chunk_idx, chunk in enumerate(chunk_iter):
        pre_classes = chunk["pre_root_id"].map(root_to_class).fillna("unknown")
        post_classes = chunk["post_root_id"].map(root_to_class).fillna("unknown")
        pathways = [
            classify_pathway(p, q)
            for p, q in zip(pre_classes.values, post_classes.values)
        ]
        for pw in pathways:
            pathway_counts[pw] += 1
        total_rows += len(chunk)
        print(f"  processed {total_rows:,} / ~14,828,657 edges ...", file=sys.stderr)

    print(f"\n=== Edge Pathway Classification Summary ===", file=sys.stderr)
    print(f"Total edges classified: {sum(pathway_counts.values()):,}", file=sys.stderr)
    for pw, cnt in sorted(pathway_counts.items(), key=lambda x: -x[1]):
        pct = cnt / sum(pathway_counts.values()) * 100
        print(f"  {pw:12s}: {cnt:>12,}  ({pct:5.2f}%)", file=sys.stderr)

    # Write if --write
    if pargs.write:
        print(f"\nWriting classified edge CSV ...", file=sys.stderr)
        out_path = metadata_path("olfactory_v1_edge_meta_classified.csv")
        first_chunk = True
        chunk_iter2 = pd.read_csv(EDGE_CSV, chunksize=pargs.chunk_size)
        for chunk_idx, chunk in enumerate(chunk_iter2):
            pre_classes = chunk["pre_root_id"].map(root_to_class).fillna("unknown")
            post_classes = chunk["post_root_id"].map(root_to_class).fillna("unknown")
            pathways = [
                classify_pathway(p, q)
                for p, q in zip(pre_classes.values, post_classes.values)
            ]
            chunk = chunk.copy()
            chunk["pathway_class"] = pathways
            chunk["pre_class"] = pre_classes.values
            chunk["post_class"] = post_classes.values
            chunk.to_csv(
                out_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            print(f"  wrote chunk {chunk_idx + 1} ...", file=sys.stderr)

        print(f"\nClassified edge CSV written to: {out_path}", file=sys.stderr)
        print(f"Columns: pre_root_id, post_root_id, syn_count, pathway_class, pre_class, post_class",
              file=sys.stderr)
    else:
        print(f"\n(Dry run — no file written. Use --write to save.)", file=sys.stderr)

    return pathway_counts


if __name__ == "__main__":
    main()
