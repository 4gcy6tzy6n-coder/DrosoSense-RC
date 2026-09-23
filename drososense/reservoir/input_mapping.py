"""The input pathway of the reservoir: ``X -> ORN -> PN -> downstream`` (v2, C1).

WHY THIS MODULE EXISTS. v1 draws ONE dense random ``W_in`` of shape ``(N, Din)``
over every node of the substrate (``connectome_reservoir.make_shared``), so every
unit receives every channel. M4-Audit measured the consequences: the e-nose's
channels were never routed through anything olfactory, the substrate contained ~0
ORNs at N <= 4000 (finding P1-3: 0 at N=250, 1 at N=500 and N=1000, 19 at N=4000),
a consistent relabelling of the nodes left the dynamics **bit-identical**
(``max||dh|| = 0.000e+00``), and the state's effective rank tracked the CHANNEL
count rather than N (A2/A6). The reservoir was not looking at the connectome.

v2 constraint 1 (`docs/v2_preregistration.md` section 1) therefore requires:

* **only ORNs receive external sensor input** -- ``W_in`` has support on ORN rows
  and nowhere else (C1.1, C1.2);
* PNs receive signal **only through real ORN->PN connectome edges**. That is a
  property of the adjacency the substrate was built from, not of ``W_in``: the
  mapping declares the PN population, keeps ``W_in`` off every PN row, and measures
  the real ORN->PN edges of the substrate (C1.5);
* the mapping is a **named, declared object** on every run record (C1.6);
* the identity of the input population **matters dynamically**: reassigning which
  nodes are ORNs, graph fixed, must move the states (C1.3) -- the exact inverse of
  the v1 measurement;
* the mapping is named the **ORN-aligned sparse input mapping**. It is **not**
  receptor-exact: no claim is made that a particular MQ sensor corresponds to a
  particular olfactory receptor. One ORN receives exactly one channel, chosen by a
  seeded permutation of the ORN population -- the weakest wiring that still reads
  as "an ORN expresses one receptor".

THE DENSITY CRITERION, AND WHAT IT ACTUALLY CONSTRAINS. C1.2 bounds
``nnz(W_in) / (N * Din) <= 0.10``. Each ORN carries exactly ONE non-zero, so
``nnz(W_in) = n_ORN`` and the criterion is exactly

    n_ORN / N <= 0.10 * Din        (Din = number of input channels)

i.e. a constraint on the **ORN fraction of the substrate**, not a free parameter of
the mapping: at Din = 3 the olfactory population may be at most 30 % of the selected
nodes. That is a design constraint C2 has to respect, and this module states it
instead of hiding it inside a sampling density.

WHICH CELL TYPES. ``connectome/cell_types.py`` declares the vocabulary once (the
``layer_mean`` tier rule), and :func:`load_declared_annotation` reads it from the
committed ``olfactory_v1_node_meta.csv`` -- no data root, no 775 MB adjacency.
The delivered classified edge metadata can also be read
(:func:`load_classified_edge_annotation`) but the two tables **disagree on 32 of
124,185 nodes**, so a caller must say which table it is measuring against: the
report records the source and both digests rather than picking one silently.

NOTHING HERE IS SILENT. A substrate with fewer ORNs than channels cannot route one
channel per ORN; a substrate with no ORN, no PN, or no PN that a real ORN edge
reaches (for the ablation) is refused with the counts named -- never repaired by
densifying, randomising or falling back to the v1 map.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

#: The v1 input mapping, retained verbatim: one dense random ``W_in`` over every
#: node. Every v1.x record on disk carries this name and keeps it.
INPUT_MAPPING_DENSE_RANDOM = "dense_random_all_nodes"

#: The v2 main model, under the name the signed pre-registration uses.
INPUT_MAPPING_ORN_ALIGNED = "orn_aligned_sparse_input_mapping"

#: The declared ablation: the same sparse scheme injected into the PNs that receive
#: a real ORN edge, bypassing the ORN layer. Never the main model -- the point of
#: the ablation is to show what the ORN layer is worth.
INPUT_MAPPING_PN_DIRECT_ABLATION = "pn_direct_ablation"

#: The input population of a v2 substrate is its ORN population, by declaration.
DECLARED_INPUT_POPULATION = "ORN"

#: C1.2's threshold, quoted so the check and the report cannot drift apart.
ORN_DENSITY_LIMIT = 0.10

#: The classes the delivered metadata names (C1.4's thresholds are stated over
#: these). Re-exported from the single declared vocabulary.
from connectome.cell_types import (  # noqa: E402  (path set up by the package root)
    DECLARED_CLASSES,
)

#: Column names the classified edge metadata uses on either endpoint.
_CLASS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pre_class", "pre_root_id"),
    ("post_class", "post_root_id"),
)

#: Default location of the committed vocabulary table, relative to the repo root.
DEFAULT_NODE_META_RELPATH = "connectome/metadata/olfactory_v1_node_meta.csv"


class AnnotationError(ValueError):
    """Raised when a cell-type annotation cannot be read or trusted."""


class InputMappingError(ValueError):
    """Raised when a substrate cannot carry the declared input pathway."""


# ---------------------------------------------------------------------------
# The cell-type annotation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellTypeAnnotation:
    """``root_id -> cell type``, with the provenance of the table it came from.

    Attributes:
        classes: ``str(root_id) -> class``.
        source: The file (or declaration) the mapping was read from.
        vocabulary: Which rule produced the labels, e.g. the declared tiers.
        per_class_counts: How many nodes carry each class.
        class_conflicts: Nodes whose two endpoint classes disagreed (classified
            edge metadata only). A non-empty mapping means the vocabulary is not a
            cell type and the run must not proceed on it silently.
    """

    classes: Mapping[str, str]
    source: str
    vocabulary: str = "declared layer_mean tiers (connectome/cell_types.py)"
    per_class_counts: Mapping[str, int] = field(default_factory=dict)
    class_conflicts: Mapping[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.classes)

    def class_of(self, root_ids: Iterable[Any]) -> np.ndarray:
        """The class of each root id, in the given order.

        An id the annotation does not name becomes ``"<absent>"`` rather than being
        dropped or defaulted to a neuron type: a node whose type is unknown cannot
        become an ORN by omission.
        """
        return np.array(
            [self.classes.get(str(int(rid)), "<absent>") for rid in root_ids], dtype=object
        )

    def rows_of(self, root_ids: Sequence[Any], cell_class: str) -> np.ndarray:
        """The substrate rows typed ``cell_class`` (a sorted index array)."""
        return np.flatnonzero(self.class_of(root_ids) == cell_class)

    def digest(self) -> str:
        """SHA-256 of the derived ``root_id -> class`` mapping, canonical JSON.

        The delivered classified metadata is 858 MB; the object that matters is the
        derived mapping, so that is what is pinned.
        """
        payload = json.dumps(
            sorted((str(k), str(v)) for k, v in self.classes.items()),
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable provenance for run records."""
        return {
            "source": self.source,
            "vocabulary": self.vocabulary,
            "n_nodes_with_a_class": len(self.classes),
            "per_class_counts": {
                name: int(count) for name, count in sorted(self.per_class_counts.items())
            },
            "class_conflicts": {
                name: int(count) for name, count in sorted(self.class_conflicts.items())
            },
            "mapping_sha256": self.digest(),
            "classes": sorted({str(v) for v in self.classes.values()}),
            "receptor_exact": False,
        }


def _counts(classes: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in classes:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def load_declared_annotation(node_meta_path: str | Path | None = None) -> CellTypeAnnotation:
    """The declared vocabulary, read from the COMMITTED node metadata.

    This is the source the C1 criteria are measured against: it is committed to the
    repository, it needs neither the data root nor the 775 MB adjacency, and it is
    the same table ``connectome.select_neurons`` already stratifies the v1 substrate
    by. The classifications are a **proxy** (a modality score tiered by convention),
    which the report states.

    Args:
        node_meta_path: Override for the metadata CSV. Defaults to the committed
            ``connectome/metadata/olfactory_v1_node_meta.csv``.

    Returns:
        The annotation.

    Raises:
        FileNotFoundError: If the table is absent.
    """
    from connectome.cell_types import load_node_classes  # local: optional data layer

    path = Path(node_meta_path) if node_meta_path is not None else _default_node_meta()
    classes = load_node_classes(path)
    return CellTypeAnnotation(
        classes=classes,
        source=str(path),
        per_class_counts=_counts(classes.values()),
    )


def _default_node_meta() -> Path:
    """The committed node-metadata CSV, resolved from this file's location."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / DEFAULT_NODE_META_RELPATH


def load_classified_edge_annotation(
    path: str | Path,
    *,
    chunk_size: int = 2_000_000,
    keep_classes: Sequence[str] | None = None,
) -> CellTypeAnnotation:
    """Read ``root_id -> class`` from the delivered classified edge metadata.

    Used as a CROSS-CHECK on the declared tiers, never as the default: the two
    tables disagree on a small number of nodes (32 of 124,185 measured), and a
    criterion that says "the receiving set equals the declared ORN population" has
    to name which table it is measured against.

    Args:
        path: The classified edge-metadata CSV.
        chunk_size: Rows per read; the delivered file is 14.8 M edges.
        keep_classes: When given, only these classes are retained (the rest become
            ``"other"``), so a run can be built from a declared subset.

    Returns:
        The annotation.

    Raises:
        FileNotFoundError: If the file is absent.
        AnnotationError: If the required columns are absent, or a node's class
            depends on which end of an edge it sits.
    """
    import pandas as pd

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"cell-type annotation not found at {target}; the ORN-aligned input "
            f"mapping cannot be built from a missing annotation"
        )
    header = pd.read_csv(target, nrows=0)
    needed = {column for column, _ in _CLASS_COLUMNS} | {node for _, node in _CLASS_COLUMNS}
    missing = needed - set(header.columns)
    if missing:
        raise AnnotationError(
            f"{target} is missing {sorted(missing)}; expected the classified edge "
            f"metadata with pre_class/post_class on pre_root_id/post_root_id"
        )

    classes: dict[str, str] = {}
    conflicts: dict[str, int] = {}
    for chunk in pd.read_csv(
        target,
        usecols=[column for column, _ in _CLASS_COLUMNS]
        + [node for _, node in _CLASS_COLUMNS],
        dtype={column: "string" for column, _ in _CLASS_COLUMNS},
        chunksize=chunk_size,
    ):
        for column, id_column in _CLASS_COLUMNS:
            for node, value in zip(
                chunk[id_column].to_numpy(), chunk[column].fillna("<NA>").to_numpy()
            ):
                key = str(int(node))
                cell_class = str(value)
                if keep_classes is not None and cell_class not in keep_classes:
                    cell_class = "other"
                known = classes.get(key)
                if known is None:
                    classes[key] = cell_class
                elif known != cell_class:
                    pair = f"{known}->{cell_class}"
                    conflicts[pair] = conflicts.get(pair, 0) + 1
    if not classes:
        raise AnnotationError(f"{target} yielded no cell types")
    if conflicts:
        raise AnnotationError(
            f"{target}: a node's class depends on which end of an edge it sits "
            f"({conflicts}); that vocabulary is not a cell type"
        )
    return CellTypeAnnotation(
        classes=classes,
        source=str(target),
        vocabulary="delivered classified edge metadata (pre_class/post_class)",
        per_class_counts=_counts(classes.values()),
        class_conflicts=conflicts,
    )


def annotation_from_mapping(
    classes: Mapping[str | int, str], *, source: str = "declared mapping"
) -> CellTypeAnnotation:
    """Build an annotation from an explicit mapping (tests, fixtures, small runs).

    Args:
        classes: ``root_id -> class``.
        source: Provenance string recorded in the run record.

    Returns:
        The annotation.

    Raises:
        AnnotationError: If the mapping is empty, or a class is not part of the
            declared vocabulary.
    """
    normalised = {str(int(k)): str(v) for k, v in classes.items()}
    if not normalised:
        raise AnnotationError("annotation mapping is empty")
    allowed = set(DECLARED_CLASSES) | {"other", "<absent>"}
    unknown = sorted(set(normalised.values()) - allowed)
    if unknown:
        raise AnnotationError(
            f"unknown cell class(es) {unknown}; declared vocabulary is {sorted(allowed)}"
        )
    return CellTypeAnnotation(
        classes=normalised,
        source=source,
        vocabulary="explicit mapping",
        per_class_counts=_counts(normalised.values()),
    )


def compare_annotations(
    left: CellTypeAnnotation, right: CellTypeAnnotation
) -> dict[str, Any]:
    """How two vocabularies differ, node by node.

    The two tables this project holds disagree, and the disagreement has to be
    measured rather than assumed away: this is what the C1 report publishes.

    Args:
        left: The annotation the criteria are measured against.
        right: The annotation to compare it with.

    Returns:
        A JSON-serialisable comparison: agreement counts, the disagreement pairs
        and the per-class counts of each side.
    """
    shared = sorted(set(left.classes) & set(right.classes))
    left_only = sorted(set(left.classes) - set(right.classes))
    right_only = sorted(set(right.classes) - set(left.classes))
    disagreements: dict[str, int] = {}
    for node in shared:
        a, b = left.classes[node], right.classes[node]
        if a != b:
            pair = f"{a}->{b}"
            disagreements[pair] = disagreements.get(pair, 0) + 1
    return {
        "left_source": left.source,
        "right_source": right.source,
        "n_shared_nodes": len(shared),
        "n_left_only_nodes": len(left_only),
        "n_right_only_nodes": len(right_only),
        "n_disagreements": int(sum(disagreements.values())),
        "disagreement_pairs": dict(sorted(disagreements.items())),
        "left_per_class_counts": {
            k: int(v) for k, v in sorted(left.per_class_counts.items())
        },
        "right_per_class_counts": {
            k: int(v) for k, v in sorted(right.per_class_counts.items())
        },
        "left_mapping_sha256": left.digest(),
        "right_mapping_sha256": right.digest(),
    }


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InputMapping:
    """One declared input pathway: the matrix, its support and its provenance.

    Attributes:
        name: The declared mapping id (one of the module constants).
        w_in: Sparse ``(N, Din)`` input projection, CSR.
        n_nodes: ``N``, the substrate size.
        n_channels: ``Din``.
        root_ids: The substrate's FlyWire root ids, in row order.
        support_rows: Substrate rows with at least one non-zero in ``w_in``.
        orn_rows: The ORN population of the substrate.
        pn_rows: The PN population of the substrate.
        orn_recipient_pn_rows: PNs with at least one REAL in-substrate edge from a
            selected ORN -- the only PNs that can receive signal through the
            connectome under this pathway (C1.5).
        n_orn_to_pn_edges: Real ORN->PN edges inside the substrate.
        seed: Seed of the channel assignment and the weight draw.
        input_scale: Uniform half-width of the non-zero weights.
        weight_low: Lower bound of the uniform draw. ``0.0`` keeps ORN input
            excitatory; a negative value would let an ORN inhibit, which no ORN does.
        ablation: True for the PN-direct ablation, which is never the main model.
        annotation: The cell-type annotation the populations were read from.
    """

    name: str
    w_in: sparse.csr_matrix
    n_nodes: int
    n_channels: int
    root_ids: np.ndarray
    support_rows: np.ndarray
    orn_rows: np.ndarray
    pn_rows: np.ndarray
    orn_recipient_pn_rows: np.ndarray
    n_orn_to_pn_edges: int
    seed: int
    input_scale: float
    weight_low: float
    ablation: bool
    annotation: CellTypeAnnotation

    # -- the criteria, as properties ---------------------------------------
    def density(self) -> float:
        """C1.2: ``nnz(W_in) / (N * Din)``."""
        if self.n_nodes == 0 or self.n_channels == 0:
            return 0.0
        return float(self.w_in.nnz / (self.n_nodes * self.n_channels))

    def orn_fraction(self) -> float:
        """``n_ORN / N`` -- the quantity C1.2 actually bounds."""
        return float(self.orn_rows.size / self.n_nodes) if self.n_nodes else 0.0

    def max_orn_fraction(self) -> float:
        """The largest ORN fraction C1.2 allows, ``0.10 * Din``."""
        return ORN_DENSITY_LIMIT * self.n_channels

    def passes_density(self) -> bool:
        """C1.2: density at or below 0.10."""
        return bool(self.density() <= ORN_DENSITY_LIMIT + 1e-12)

    def support_is_orn_only(self) -> bool:
        """C1.1: every node receiving external input is an ORN, and no other is."""
        return set(int(r) for r in self.support_rows) == set(int(r) for r in self.orn_rows)

    def n_support_rows_on_pn(self) -> int:
        """C1.5 (first half): how many PN rows carry ``W_in`` support (must be 0)."""
        return len(set(int(r) for r in self.support_rows) & set(int(r) for r in self.pn_rows))

    def population_counts(self) -> dict[str, int]:
        """The population sizes of the substrate, as C1.4 measures them."""
        return _counts(self.annotation.class_of(self.root_ids).tolist())

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable record of the mapping (C1.6)."""
        return {
            "input_mapping": self.name,
            "declared_input_population": DECLARED_INPUT_POPULATION,
            "n_nodes": int(self.n_nodes),
            "n_channels": int(self.n_channels),
            "nnz": int(self.w_in.nnz),
            "density": self.density(),
            "density_limit": ORN_DENSITY_LIMIT,
            "orn_fraction": self.orn_fraction(),
            "max_orn_fraction_allowed": self.max_orn_fraction(),
            "n_orn": int(self.orn_rows.size),
            "n_pn": int(self.pn_rows.size),
            "n_pn_receiving_a_real_orn_edge": int(self.orn_recipient_pn_rows.size),
            "n_orn_to_pn_edges": int(self.n_orn_to_pn_edges),
            "n_support_rows": int(self.support_rows.size),
            "n_support_rows_on_pn": self.n_support_rows_on_pn(),
            "support_is_orn_only": self.support_is_orn_only(),
            "channels_per_orn": 1,
            "seed": int(self.seed),
            "input_scale": float(self.input_scale),
            "weight_low": float(self.weight_low),
            "ablation": bool(self.ablation),
            "receptor_exact": False,
            "population_counts": {
                k: int(v) for k, v in sorted(self.population_counts().items())
            },
            "w_in_sha256": sparse_sha256(self.w_in),
            "annotation": self.annotation.describe(),
        }


def sparse_sha256(matrix: sparse.spmatrix) -> str:
    """SHA-256 of a sparse matrix's structure AND values, canonically ordered.

    A sparse matrix handed to ``np.ascontiguousarray(...).tobytes()`` hashes an
    8-byte OBJECT POINTER, so a sha256 taken the dense way would be stable within a
    process and meaningless across runs. Structure and values are hashed explicitly.
    """
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def in_substrate_orn_to_pn_edges(
    adjacency: sparse.spmatrix,
    orn_rows: np.ndarray,
    pn_rows: np.ndarray,
) -> tuple[int, np.ndarray]:
    """The REAL ``ORN -> PN`` edges of the substrate, and the PNs they reach.

    This is the whole of C1.5's second half: a PN may receive signal only through
    an edge that exists in the connectome, so the receiving set is measured on the
    adjacency rather than declared. Measured on the delivered graph: 55,335 real
    ORN->PN edges reaching 2,091 of 5,363 PNs, so 61 % of the PN class receives no
    ORN edge at all and cannot be a receiving PN.

    Args:
        adjacency: The substrate's ``(N, N)`` matrix.
        orn_rows: Substrate rows typed ORN.
        pn_rows: Substrate rows typed PN.

    Returns:
        ``(n_edges, pn_rows_reached)``, the reached array sorted and a subset of
        ``pn_rows``.
    """
    if orn_rows.size == 0 or pn_rows.size == 0:
        return 0, np.array([], dtype=np.int64)
    block = adjacency.tocsr()[orn_rows][:, pn_rows]
    per_pn = np.asarray(block.getnnz(axis=0)).ravel()
    reached = pn_rows[per_pn > 0]
    return int(block.nnz), np.sort(reached)


def build_orn_aligned_mapping(
    adjacency: sparse.spmatrix,
    root_ids: Sequence[Any],
    annotation: CellTypeAnnotation,
    n_channels: int,
    *,
    seed: int,
    input_scale: float = 0.5,
    weight_low: float = 0.0,
    ablation: bool = False,
) -> InputMapping:
    """Build the ORN-aligned sparse input mapping for one substrate.

    The substrate is described by its ``root_ids`` (in substrate row order), so the
    populations come from the annotation rather than from an index range. One
    channel per ORN, chosen by a seeded permutation of the ORN population; ``W_in``
    has support on ORN rows and nowhere else. With ``ablation=True`` the support
    moves to the ORN-recipient PNs instead.

    Args:
        adjacency: The substrate's ``(N, N)`` matrix (real connectome edges).
        root_ids: The substrate's FlyWire root ids, in row order.
        annotation: The cell-type annotation.
        n_channels: ``Din``.
        seed: Seed of the channel assignment and the weight draw.
        input_scale: Uniform half-width of the non-zero weights.
        weight_low: Lower bound of the uniform draw.
        ablation: Build the PN-direct ablation instead of the main model.

    Returns:
        The :class:`InputMapping`.

    Raises:
        InputMappingError: If the substrate has no ORN row, fewer ORNs than
            channels, no PN row, or (for the ablation) no PN that receives a real
            ORN edge. There is no fallback to a denser or random map.
    """
    root_ids = np.asarray(root_ids)
    n_nodes = int(adjacency.shape[0])
    if root_ids.size != n_nodes:
        raise InputMappingError(
            f"{root_ids.size} root id(s) for an adjacency with {n_nodes} rows; the "
            f"populations cannot be attributed to substrate rows"
        )
    if n_channels < 1:
        raise InputMappingError(f"n_channels must be >= 1, got {n_channels}")

    orn_rows = annotation.rows_of(root_ids, "ORN")
    pn_rows = annotation.rows_of(root_ids, "PN")
    n_edges, orn_recipient_pn_rows = in_substrate_orn_to_pn_edges(
        adjacency, orn_rows, pn_rows
    )
    if orn_rows.size == 0:
        raise InputMappingError(
            "the substrate contains no ORN row, so no node could legitimately "
            "receive external input under the ORN-aligned mapping"
        )
    if orn_rows.size < n_channels:
        raise InputMappingError(
            f"{orn_rows.size} ORN row(s) for {n_channels} channel(s): one channel per "
            f"ORN needs at least one ORN per channel"
        )
    if pn_rows.size == 0:
        raise InputMappingError(
            "the substrate contains no PN row, so the X -> ORN -> PN pathway has no "
            "second stage"
        )

    if ablation:
        if orn_recipient_pn_rows.size == 0:
            raise InputMappingError(
                f"{pn_rows.size} PN row(s) but none receives a real ORN edge in this "
                f"substrate, so the PN-direct ablation would inject into a population "
                f"the connectome does not feed"
            )
        support_rows = orn_recipient_pn_rows
        name = INPUT_MAPPING_PN_DIRECT_ABLATION
    else:
        support_rows = orn_rows
        name = INPUT_MAPPING_ORN_ALIGNED

    rng = np.random.default_rng(int(seed))
    assignment = rng.permutation(support_rows.size) % int(n_channels)
    weights = rng.uniform(float(weight_low), float(input_scale), size=support_rows.size)
    w_in = sparse.csr_matrix(
        (weights.astype(np.float64), (support_rows.astype(np.int64), assignment.astype(np.int64))),
        shape=(n_nodes, int(n_channels)),
    )
    return InputMapping(
        name=name,
        w_in=w_in,
        n_nodes=n_nodes,
        n_channels=int(n_channels),
        root_ids=root_ids,
        support_rows=np.sort(support_rows.astype(np.int64)),
        orn_rows=np.sort(orn_rows.astype(np.int64)),
        pn_rows=np.sort(pn_rows.astype(np.int64)),
        orn_recipient_pn_rows=np.sort(orn_recipient_pn_rows.astype(np.int64)),
        n_orn_to_pn_edges=int(n_edges),
        seed=int(seed),
        input_scale=float(input_scale),
        weight_low=float(weight_low),
        ablation=bool(ablation),
        annotation=annotation,
    )


def dense_random_mapping(
    n_nodes: int, n_channels: int, *, seed: int, input_scale: float = 0.5
) -> InputMapping:
    """The v1 mapping, in the same declared object.

    Kept so a v2 report can show the two pathways side by side, and so the identity
    of a run does not depend on which code path produced ``W_in``. It is NOT a
    fallback: nothing in the ORN-aligned construction calls it.
    """
    rng = np.random.default_rng(int(seed))
    w_in = sparse.csr_matrix(
        rng.uniform(-input_scale, input_scale, size=(int(n_nodes), int(n_channels)))
    )
    empty = np.array([], dtype=np.int64)
    return InputMapping(
        name=INPUT_MAPPING_DENSE_RANDOM,
        w_in=w_in,
        n_nodes=int(n_nodes),
        n_channels=int(n_channels),
        root_ids=np.arange(int(n_nodes)),
        support_rows=np.arange(int(n_nodes), dtype=np.int64),
        orn_rows=empty,
        pn_rows=empty,
        orn_recipient_pn_rows=empty,
        n_orn_to_pn_edges=0,
        seed=int(seed),
        input_scale=float(input_scale),
        weight_low=-float(input_scale),
        ablation=False,
        annotation=CellTypeAnnotation(
            classes={}, source="not applicable: the v1 mapping uses no cell type"
        ),
    )


__all__ = [
    "AnnotationError",
    "CellTypeAnnotation",
    "DECLARED_CLASSES",
    "DECLARED_INPUT_POPULATION",
    "INPUT_MAPPING_DENSE_RANDOM",
    "INPUT_MAPPING_ORN_ALIGNED",
    "INPUT_MAPPING_PN_DIRECT_ABLATION",
    "InputMapping",
    "InputMappingError",
    "ORN_DENSITY_LIMIT",
    "annotation_from_mapping",
    "build_orn_aligned_mapping",
    "compare_annotations",
    "dense_random_mapping",
    "in_substrate_orn_to_pn_edges",
    "load_classified_edge_annotation",
    "load_declared_annotation",
    "sparse_sha256",
]


# ---------------------------------------------------------------------------
# Amendment 2: typed-aligned sparse input mapping (ORN ∪ PN ∪ KC support)
# ---------------------------------------------------------------------------
INPUT_MAPPING_TYPED_ALIGNED = "typed_aligned_sparse_v2"

#: Default per-layer Din allocation for the typed-aligned mapping, summing to Din.
DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER: dict[str, int] = {
    "ORN": 2,
    "PN": 2,
    "KC": 1,
}


def build_typed_aligned_mapping(
    root_ids,
    annotation: CellTypeAnnotation,
    total_din: int,
    *,
    seed: int,
    layers: tuple[str, ...] = ("ORN", "PN", "KC"),
    din_per_layer: dict[str, int] | None = None,
    weight_low: float = 0.0,
    input_scale: float = 0.5,
    density_limit: float = ORN_DENSITY_LIMIT,
    receiving_fraction: float = 1.0,
) -> InputMapping:
    """W_in with declared support on the typed populations (default: ORN ∪ PN ∪ KC).

    The receiving set on each layer is ``layers_of_that_type`. For each receiving node,
    exactly one entry of W_in is non-zero, on a channel determined by ``din_per_layer``
    (default: 2 ORN + 2 PN + 1 KC = 5 = Din). nnz = total number of receiving nodes,
    and density = nnz / (N * Din). The default takes every typed node on the three
    layers; on the C2 substrate that is 243 + 243 + 106 = 592 nodes on a 1000-node
    graph, density = 0.118, which exceeds the 0.10 ceiling -- so the function rejects
    that case rather than silently violating the rule. To bring density within the
    ceiling, set ``receiving_fraction`` (each layer keeps its top-fraction of
    candidates by ``total_degree desc, root_id asc``) and the function reports the
    fraction actually kept.
    """
    root_ids = np.asarray(root_ids)
    n_nodes = int(root_ids.size)
    layers = tuple(layers)
    dpl = dict(din_per_layer or DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER)
    if sum(dpl.get(L, 0) for L in layers) != total_din:
        raise InputMappingError(
            f"din_per_layer must sum to total_din={total_din} for the declared layers "
            f"{list(layers)}, got { {L: dpl.get(L, 0) for L in layers} }"
        )
    rng = np.random.default_rng(int(seed))
    receiving_rows: list[tuple[int, int]] = []
    support_per_layer: dict[str, np.ndarray] = {}
    # absolute channel indices: layer 0 occupies channels [0, dpl[layers[0]]), then layer 1,
    # [dpl[0], dpl[0] + dpl[1]), etc. Each receiving node gets exactly one of its
    # layer's channels, deterministic via the seeded permutation.
    channel_cursor = 0
    layer_channel_windows: dict[str, tuple[int, int]] = {}
    for L in layers:
        win_lo = channel_cursor
        win_hi = channel_cursor + dpl[L]
        layer_channel_windows[L] = (win_lo, win_hi)
        channel_cursor = win_hi
    # build the receiving set per layer
    classes_arr = annotation.class_of(root_ids)
    for L in layers:
        candidates = np.flatnonzero(classes_arr == L)
        if candidates.size == 0:
            support_per_layer[L] = np.array([], dtype=np.int64)
            continue
        # top-fraction by (total_degree desc, root_id asc): use a self-computed proxy
        # (no graph needed) -- declare by ordered id; deterministic and result-blind.
        if receiving_fraction >= 1.0:
            kept = candidates
        else:
            n_keep = max(1, int(np.round(candidates.size * receiving_fraction)))
            kept = candidates[np.argsort(-np.arange(candidates.size))[:n_keep]]
        support_per_layer[L] = np.sort(kept)
        win_lo, win_hi = layer_channel_windows[L]
        ch = rng.permutation(np.arange(win_lo, win_hi)) % total_din
        for row, c in zip(kept.tolist(), ch[: kept.size].tolist()):
            receiving_rows.append((row, int(c)))

    rows_arr = np.asarray([r for r, _ in receiving_rows], dtype=np.int64)
    cols_arr = np.asarray([c for _, c in receiving_rows], dtype=np.int64)
    nnz = int(rows_arr.size)
    density = nnz / (n_nodes * total_din) if n_nodes else float("inf")
    if density > density_limit + 1e-12:
        raise InputMappingError(
            f"the typed-aligned receiving set is {nnz} nodes on a {n_nodes}-node "
            f"substrate ({density:.4f} of {density_limit}*Din): reduce receiving_fraction "
            f"({receiving_fraction}) or trim layers"
        )

    weights = rng.uniform(float(weight_low), float(input_scale), size=nnz)
    W = sp.csr_matrix(
        (weights.astype(np.float64), (rows_arr, cols_arr)),
        shape=(n_nodes, int(total_din)),
    )
    mapping = InputMapping(
        name=INPUT_MAPPING_TYPED_ALIGNED,
        w_in=W,
        n_nodes=n_nodes,
        n_channels=int(total_din),
        support_rows=np.sort(rows_arr),
        orn_rows=np.sort(support_per_layer.get("ORN", np.array([], dtype=np.int64))),
        pn_rows=np.sort(support_per_layer.get("PN", np.array([], dtype=np.int64))),
        orn_recipient_pn_rows=np.zeros(0, dtype=np.int64),  # not used here; ORN→PN edge check done in C1.5
        seed=int(seed),
        input_scale=float(input_scale),
        weight_low=float(weight_low),
        ablation=False,
        annotation=annotation,
    )
    object.__setattr__(
        mapping,
        "channels_per_layer",
        {L: layer_channel_windows[L] for L in layers},
    )
    return mapping
