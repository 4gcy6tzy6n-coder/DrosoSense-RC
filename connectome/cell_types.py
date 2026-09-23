"""The declared cell-type vocabulary of the olfactory connectome (single source).

WHY THIS FILE EXISTS. The ORN/PN/KC/MBON/DAN/higher_order labels used in this
project are **derived**, never shipped: no committed file carries a `cell_type` or
`neuron_class` column. Two code paths had grown their own copy of the rule --

* ``connectome/add_edge_masks.py`` (``_TIER_BOUNDARIES`` / ``layer_to_class``),
  which produced ``olfactory_v1_edge_meta_classified.csv``;
* the v2 construct phase, which needs the same vocabulary to build an
  ORN-aligned input mapping and to check criteria C1.1/C1.4;

-- and they disagree on 32 of 124,185 nodes (measured: ORN 2,267 tiered from the
committed ``neuron_class_ranking_df_783-olfactory-10000`` scores vs 2,299 in the
delivered classified edge metadata; see
``results/audit/m4_audit/C1_annotation_probe.json``). Two vocabularies is one too
many: a criterion that says "every node receiving input is an ORN" has to name
which table it is measured against.

This module declares the rule once, so ``add_edge_masks.py`` and the v2 construct
phase cannot drift. The tiers are a **proxy**: `layer_mean` is a per-neuron
olfactory-modality score from the ranking table, and the boundaries below are
convention, not measurement. The documentation says so at every use site -- a
claim about a specific receptor would need receptor-level data this bundle does
not carry.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

#: The declared tier boundaries, as ONE table. Lower bound exclusive, upper
#: inclusive, matching the delivered classified edge metadata. A node whose score
#: falls in no tier is ``other`` -- and no node of the delivered graph sits exactly
#: on a boundary (measured), so the exclusivity cannot silently move a population.
DECLARED_LAYER_TIERS: tuple[tuple[float, float, str], ...] = (
    (0.5, 1.8, "ORN"),
    (1.8, 3.5, "PN"),
    (3.5, 4.2, "higher_order"),
    (4.2, 4.8, "DAN"),
    (4.8, 5.5, "KC"),
    (5.5, 6.5, "MBON"),
    (6.5, float("inf"), "other"),
)

#: Every class name the vocabulary can produce, in descending ORN-proximity order.
DECLARED_CLASSES: tuple[str, ...] = (
    "ORN",
    "PN",
    "KC",
    "MBON",
    "DAN",
    "higher_order",
    "other",
)

#: The class an unknown or unranked node gets. Never ORN: an untyped node must not
#: become part of the olfactory input population by omission.
FALLBACK_CLASS = "other"

ORN = "ORN"
PN = "PN"


def layer_to_class(layer_mean: float) -> str:
    """The declared class of one olfactory-modality score.

    Args:
        layer_mean: The per-neuron score from the FlyWire olfactory ranking table.

    Returns:
        One of :data:`DECLARED_CLASSES`.
    """
    for low, high, cell_class in DECLARED_LAYER_TIERS:
        if low < layer_mean <= high:
            return cell_class
    return FALLBACK_CLASS


def _as_root_id(value: str) -> str:
    """A FlyWire root id as a canonical string.

    Root ids are 60-bit integers (~7.2e17), which a float64 CANNOT represent
    exactly: parsing through ``float`` silently collapses distinct neurons onto one
    key (measured: it turned 124,185 nodes into 100,984). They are integer strings,
    so they are parsed as integers.
    """
    text = value.strip()
    if "." in text:  # a table written through a float column
        text = text.split(".", 1)[0]
    return str(int(text))


def load_node_classes(
    node_meta_path: str | Path,
    *,
    key: str = "root_id",
    score_column: str = "layer_mean",
) -> dict[str, str]:
    """``root_id -> declared class`` from the committed node metadata.

    The metadata table is committed to the repository
    (``connectome/metadata/olfactory_v1_node_meta.csv``), so the vocabulary is
    readable without the 775 MB adjacency and without the data root.

    Args:
        node_meta_path: The node-metadata CSV.
        key: Column holding the node identity (``root_id``).
        score_column: Column holding the modality score (``layer_mean``).

    Returns:
        ``str(root_id) -> class``.

    Raises:
        FileNotFoundError: If the table is absent.
        ValueError: If the required columns are missing.
    """
    path = Path(node_meta_path)
    if not path.is_file():
        raise FileNotFoundError(f"node metadata not found at {path}")
    classes: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {key, score_column} - fields
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        for row in reader:
            node = row[key]
            if node is None or node == "":
                continue
            score = row[score_column]
            classes[_as_root_id(node)] = (
                FALLBACK_CLASS if score in (None, "") else layer_to_class(float(score))
            )
    if not classes:
        raise ValueError(f"{path} yielded no node classes")
    return classes


def class_counts(classes: Mapping[str, str]) -> dict[str, int]:
    """How many nodes carry each declared class."""
    counts: dict[str, int] = {}
    for cell_class in classes.values():
        counts[cell_class] = counts.get(cell_class, 0) + 1
    return counts


__all__ = [
    "DECLARED_CLASSES",
    "DECLARED_LAYER_TIERS",
    "FALLBACK_CLASS",
    "ORN",
    "PN",
    "class_counts",
    "layer_to_class",
    "load_node_classes",
]
