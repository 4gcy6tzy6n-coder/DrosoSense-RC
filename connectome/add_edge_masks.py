#!/usr/bin/env python3
"""
add_edge_masks.py — Assign pathway_class / pre_class / post_class to olfactory edges.

DATA-3 post-R1 fix: align pathway_class with DATA-9 §3.2 semantics.

Rule order (DATA-9 §3.2):
  1. modulatory  — pre/post in {MBDAN, APL, DAN}  → DAN (=MBDAN); APL not in cell-type table
  2. lateral    — pre/post == ALLN                → ALLN not in cell-type table; cannot identify
  3. forward    — exactly ORN→PN, PN→KC, or KC→MBON  (DATA-9 §3.2 three-pairwise chain)
  4. other      — everything else

Biological scope note:
  ALLN cannot be reliably distinguished from ALPN using layer_mean alone.
  APL cannot be reliably distinguished from other cell types using layer_mean alone.
  The FlyWire cell-type classifier used for this connectome (v783 olfactory rank table)
  does not expose an explicit ALLN or APL label.  Consequently the lateral category is
  empty and APL involvement is unreported; these limitations are recorded in
  meta.json.edge_mask_counts.note.

Inputs (resolved via connectome/paths.py data-root):
  - neuron_class_ranking_df_783-olfactory-10000.feather  : root_id + layer_mean
  - olfactory_v1_edge_meta.csv                          : pre_root_id, post_root_id, syn_count
  - olfactory_v1_edge_meta_classified.csv (existing)    : pre_class, post_class from old script

Output:
  - data-root/connectome/metadata/olfactory_v1_edge_meta_classified.csv
    (overwrites the incorrectly-classified previous version)

Provenance recorded in meta.json.edge_mask_counts.

Usage:
  python connectome/add_edge_masks.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "connectome"))
from paths import metadata_path  # noqa: E402


# ── Pathway classification constants (DATA-9 §3.2) ────────────────────────────
MODULATORY_CLASSES = frozenset({"MBDAN", "APL", "DAN"})  # DAN = MBDAN in the classifier

# The classifier uses PN for ALPN (antennal-lobe projection neurons).
# ALLN (AL local neurons) and APL are NOT present in the cell-type table — cannot identify.
# DATA-9 §3.2 forward chain is exactly three pairwise types:
#   ORN→PN  (ORN→ALPN in DATA-9 naming)
#   PN→KC   (ALPN→KC in DATA-9 naming)
#   KC→MBON
# Cross-level ORN→KC and intra-AL PN→PN are excluded from forward per §3.2.


def classify_edge(pre_class: str, post_class: str) -> str:
    """Apply DATA-9 §3.2 pathway rules in priority order."""
    # 1. Modulatory: MBDAN/APL involvement
    if pre_class in MODULATORY_CLASSES or post_class in MODULATORY_CLASSES:
        return "modulatory"
    # 2. Lateral: ALLN involvement — ALLN not in cell-type table; rule unreachable
    #    (annotator: if ALLN annotation is added to the cell-type table, activate this)
    # if pre_class == "ALLN" or post_class == "ALLN":
    #     return "lateral"
    # 3. Forward: exactly the three pairwise types in DATA-9 §3.2
    if (pre_class == "ORN" and post_class == "PN") or \
       (pre_class == "PN" and post_class == "KC") or \
       (pre_class == "KC" and post_class == "MBON"):
        return "forward"
    # 4. Everything else
    return "other"


def main():
    # Load existing classified CSV which has pre_class/post_class already assigned
    # by the FlyWire cell-type classifier (old script). We only re-assign pathway_class.
    classified_csv = metadata_path("olfactory_v1_edge_meta_classified.csv")
    out_csv = metadata_path("olfactory_v1_edge_meta_classified.csv")

    print(f"Loading existing classified edges: {classified_csv}", file=sys.stderr)
    em = pd.read_csv(classified_csv)
    print(f"  edges: {len(em):,}", file=sys.stderr)
    print(f"  columns: {list(em.columns)}", file=sys.stderr)

    # Verify required columns are present
    required = {"pre_root_id", "post_root_id", "syn_count", "pre_class", "post_class", "pathway_class"}
    missing = required - set(em.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        print("Run build_olfactory_connectome.py first to produce edge_meta.csv,", file=sys.stderr)
        print("then apply the cell-type classifier to produce pre_class/post_class.", file=sys.stderr)
        sys.exit(1)

    # Apply classification
    print("Re-classifying pathway_class with DATA-9 §3.2 rules ...", file=sys.stderr)
    em["pathway_class"] = em.apply(
        lambda r: classify_edge(r["pre_class"], r["post_class"]),
        axis=1
    )

    # Write output CSV
    print(f"Writing: {out_csv}", file=sys.stderr)
    em.to_csv(out_csv, index=False)

    # Report counts
    counts = em["pathway_class"].value_counts().to_dict()
    total = len(em)
    print("\npathway_class counts:", file=sys.stderr)
    for cls in ["forward", "modulatory", "lateral", "other"]:
        n = counts.get(cls, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cls:12s}: {n:>12,}  ({pct:.1f}%)", file=sys.stderr)

    print(f"\nTotal: {total:,}", file=sys.stderr)

    # Update meta.json
    meta_path = ROOT / "connectome" / "metadata" / "olfactory_v1_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Build new edge_mask_counts from the freshly-classified CSV
    new_counts = {k: int(v) for k, v in counts.items()}
    new_counts["total"] = int(total)

    meta["edge_mask_counts"] = {
        "forward": new_counts.get("forward", 0),
        "modulatory": new_counts.get("modulatory", 0),
        "lateral": new_counts.get("lateral", 0),
        "other": new_counts.get("other", 0),
        "total": total,
        "note": (
            "pathway_class assigned by add_edge_masks.py (DATA-9 §3.2 rules). "
            "ALLN cannot be identified from layer_mean alone — lateral category is empty. "
            "APL also cannot be identified from layer_mean alone — modulatory DAN-only "
            "edges are classified as modulatory; APL involvement may be present but "
            "unreported. "
            "pre_class/post_class are the FlyWire cell-type classifier labels "
            "(PN ≈ ALPN; ALLN and APL not exposed in v783 olfactory rank table). "
            "higher_order neurons are classified as 'other' (not forward) per DATA-9 §3.2 rule."
        ),
    }

    # Update commit_sha to reflect this fix
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True
        ).strip()
        meta["commit_sha"] = sha
    except Exception:
        pass  # keep existing value if git fails

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nUpdated meta.json: {meta_path}", file=sys.stderr)
    print("  edge_mask_counts:", meta["edge_mask_counts"], file=sys.stderr)

    # Also update edge_mask_mapping_rules in meta.json
    meta["edge_mask_mapping_rules"] = {
        "modulatory": "pre_class or post_class in {MBDAN, APL, DAN} (MBDAN=DAN in classifier; APL NOT in cell-type table)",
        "lateral": (
            "pre_class==ALLN or post_class==ALLN — ALLN NOT in cell-type table; "
            "category empty; requires explicit ALLN annotation to activate"
        ),
        "forward": "exactly (ORN→PN) OR (PN→KC) OR (KC→MBON) per DATA-9 §3.2",
        "other": "all remaining edges",
        "class_names": {
            "ORN": "olfactory receptor neuron (afferent sensory)",
            "PN":  "projection neuron (≈ ALPN in DATA-9; antennal-lobe output)",
            "KC":  "Kenyon cell (mushroom-body intrinsic)",
            "MBON": "MB output neuron",
            "DAN": "dopaminergic neuron (≈ MBDAN in DATA-9)",
            "higher_order": "higher-order olfactory neuron (outside the primary forward chain)",
            "other": "unclassified or boundary-audit class",
            "ALLN": "AL local neuron — NOT in cell-type table (annotate to activate lateral rule)",
            "APL": "APL neuron — NOT in cell-type table (annotate to activate modulatory rule)",
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
