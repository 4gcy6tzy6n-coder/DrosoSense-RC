"""Frozen connectome reservoir (R0) and matched topology controls.

This module delivers the M3 deliverable for DATA-4:

* a frozen, sparse :class:`ConnectomeReservoir` (R0) that drives the standard
  leaky dynamics ``h_t = (1 - alpha) h_{t-1} + alpha * tanh(g * A_hat h_{t-1}
  + W_in x_t + b)``,
* seven topology families (R0–R6) that share the readout, the input mapping,
  the bias, the split policy and the seed used to draw ``W_in`` / ``b`` — so
  every difference in performance is attributable to the reservoir matrix
  itself, and the **degree-rewired (R2)** family is the primary scientific
  control,
* sparse propagation via ``scipy.sparse`` ``matvec`` so the system stays
  tractable for ``N in {250, 500, 1000, 2000, 4000}``,
* configuration-driven ``alpha / g / input_scale / spectral_radius`` so the
  same hyperparameters cross over every topology.

The biological weights are :data:`synapse-count-informed structural weights
<PROVENANCE>`, uncalibrated. They are not conductance, not efficacy, and not
connection probability, and nothing in this module supports such a claim.

Design choices that show up in the code, stated up front:

* **One readout, many reservoirs.** The readout code lives in
  :class:`_ReservoirBase` and is shared verbatim across R0–R6. Adding a new
  topology is a one-matrix change.
* **Matching constraints are explicit, not implicit.** When R0 is built, it
  freezes ``(N, M, rho)``. Every control factory takes that reference and
  emits a matrix with the same ``(N, M, rho)`` up to a configurable tolerance.
  Tests pin those numbers.
* **Readout stays a black box.** The leaky reservoir feeds it window-level
  feature vectors; the readout does not see the reservoir matrix and so
  cannot leak the test set's structure into the comparison.
* **Sparse propagation, no surprise.** The reservoir matvec is the standard
  CSR ``@ state``; the linear readout is a single ``(H^T H + lambda I) W =
  H^T Y`` solve. Nothing in between requires a dense conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from drososense.reservoir.input_mapping import (  # noqa: E402
    INPUT_MAPPING_DENSE_RANDOM,
    InputMapping,
    sparse_sha256,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from drososense.baselines.base import BaseModel, TaskType
from drososense.utils.seeding import make_rng

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

#: Default hyperparameters for every reservoir in this module. The same dict
#: also drives the matched controls (R1–R6) so the comparison cannot drift
#: because R0 and R2 used a different ``alpha``.
DEFAULT_RESERVOIR_PARAMS: dict[str, Any] = {
    "reservoir_size": 200,
    "leak": 0.3,
    "gain": 1.0,
    "input_scale": 0.5,
    "ridge_lambda": 1.0e-3,
    "washout": 10,
    "state_pooling": "last",  # last | mean
    "readout": "ridge",  # ridge | logistic
    "spectral_radius": 0.9,
}

#: Allowed normalizations for the biological reservoir. These are the six
#: stored in ``olfactory_v1.npz`` (see ``meta.json::normalization``).
ALLOWED_NORMALIZATIONS: tuple[str, ...] = (
    "n0_raw",
    "n1_pre_l1",
    "n2_post_l1",
    "n3_global_max",
    "n4_log_pre_l1",
    "n5_binary",
)

#: NPZ key prefix used by each normalization when reading CSR components.
#: ``n0_raw`` is the raw adjacency — the build script (DATA-3) stores it
#: under the ``adj_*`` keys rather than emitting a redundant ``norm_n0_raw_*``
#: copy, per meta.json's "n0_raw ... same as adjacency_data" semantic. The
#: other five normalizations are stored under their ``norm_<name>_*`` prefix.
#: This mapping keeps :func:`load_reservoir_topology_from_npz` symmetric with
#: the build script and avoids inventing a third NPZ layout for the raw case.
_NORMALIZATION_NPZ_PREFIX: dict[str, str] = {
    "n0_raw": "adj",
    "n1_pre_l1": "norm_n1_pre_l1",
    "n2_post_l1": "norm_n2_post_l1",
    "n3_global_max": "norm_n3_global_max",
    "n4_log_pre_l1": "norm_n4_log_pre_l1",
    "n5_binary": "norm_n5_binary",
}

#: Topology family identifiers. Used in run records, the registry and the
#: constraint-checker. The order is the comparison order in the protocol.
TOPOLOGY_FAMILY_IDS: tuple[str, ...] = (
    "R0_real_fly",
    "R1_weight_shuffled",
    "R2_degree_rewired",
    "R3_random_sparse",
    "R4_er_esn",
    "R5_small_world",
    "R6_dense_random",
)

#: Controls that match ``(N, M)`` exactly with R0; R3, R5, R6 do not match
#: the degree sequence and so are "structure-broken" controls. R2 keeps the
#: degree sequence — the **primary** scientific control.
PRIMARY_CONTROL = "R2_degree_rewired"

#: Maximum absolute eigenvalue to treat as "effectively zero" for the
#: spectral-radius rescale. Avoiding division by zero on an empty matrix.
SPECTRAL_RADIUS_EPS: float = 1.0e-12

#: The input mapping of the v1 implementation: ONE dense random ``W_in`` over every
#: node of the substrate, with no input population, no cell-type gating and no layer
#: assignment. Re-exported from
#: :mod:`drososense.reservoir.input_mapping`, which is the single declaration of the
#: pathway names (v2 adds ``orn_aligned_sparse_input_mapping`` and the PN-direct
#: ablation beside it) — one name per pathway, so a record cannot describe a
#: ``W_in`` that the reservoir did not use.
INPUT_MAPPING_DENSE_RANDOM: str = INPUT_MAPPING_DENSE_RANDOM

#: Maximum double-edge swap attempts per edge in the degree-rewired control.
#: 10 is enough for the configuration-model to converge on sparse graphs.
DOUBLE_EDGE_SWAP_ATTEMPTS: int = 10


# ---------------------------------------------------------------------------
# Frozen provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservoirTopology:
    """The frozen reservoir matrix and the metrics used to match it.

    Attributes:
        matrix: Sparse (CSR) ``(N, N)`` adjacency.
        n_nodes: ``N`` — node count, matched across the family.
        n_edges: ``M`` — edge count, matched across the family.
        spectral_radius: Largest ``|lambda|`` after rescale, matched across
            the family.
        density: ``M / N^2``.
        kind: Family id (``R0_real_fly``, ``R2_degree_rewired``, ...).
        normalization: Which DATA-3 normalization produced this matrix; only
            meaningful for R0.
        weight_semantics: Always the structural-weight sentence. Cached so a
            run record cannot accidentally lose it.
        counterfactual: For a v2 R2 built as a weight-preserving counterfactual, the
            swap chain's own report (C4.8: swap counts, original-edge retention, the
            mixing curve, the conservation hashes). ``None`` for every v1 control, so
            no v1 record gains a field it was not written with.
    """

    matrix: csr_matrix
    n_nodes: int
    n_edges: int
    spectral_radius: float
    density: float
    kind: str
    normalization: str | None
    counterfactual: dict[str, Any] | None = None
    weight_semantics: str = (
        "synapse-count-informed structural weight "
        "(uncalibrated — not conductance, efficacy, or connection probability)"
    )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for run records.

        Returns:
            Mapping of structural metrics and provenance.
        """
        payload = {
            "kind": self.kind,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "spectral_radius": self.spectral_radius,
            "density": self.density,
            "normalization": self.normalization,
            "weight_semantics": self.weight_semantics,
            "nnz": int(self.matrix.nnz),
        }
        if self.counterfactual is not None:
            payload["counterfactual"] = self.counterfactual
        return payload


@dataclass(frozen=True)
class ReservoirShared:
    """The components that every reservoir in a family must share.

    A ``ConnectomeReservoir`` and its degree-rewired control share
    ``W_in``, ``b`` and the seed that drew them, so any performance gap comes
    from the reservoir matrix rather than from the input mapping or the bias.

    Attributes:
        w_in: ``(N, n_channels)`` input projection, drawn once. Dense for the v1
            mapping, sparse (CSR) for the v2 ORN-aligned mapping -- the state
            update is a matvec either way.
        bias: ``(N,)`` per-node bias, drawn once.
        seed: Seed used to draw both. Reproducibility is a property of this
            seed alone.
        input_scale: Uniform-``[-s, s]`` half-width used for ``W_in`` and ``b``.
        input_mapping: The DECLARED input pathway (protocol v1.5 identity
            component). ``dense_random_all_nodes`` is v1; v2 declares
            ``orn_aligned_sparse_input_mapping``, whose support is the ORN
            population of the substrate and nowhere else (criterion C1.1).
        input_detail: ``InputMapping.describe()`` when the mapping was built as a
            declared object -- nnz, density, the populations and the digests. The
            v1 path leaves it None, so every v1.x record keeps the exact shape it
            was written with.
    """

    w_in: np.ndarray | sp.csr_matrix
    bias: np.ndarray
    seed: int
    input_scale: float
    input_mapping: str = INPUT_MAPPING_DENSE_RANDOM
    input_detail: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for run records.

        Returns:
            Mapping of shared-component fingerprints.
        """
        payload: dict[str, Any] = {
            "w_in_sha256": _sha256(self.w_in),
            "bias_sha256": _sha256(self.bias),
            "seed": int(self.seed),
            "input_scale": float(self.input_scale),
            "input_mapping": self.input_mapping,
        }
        if self.input_detail is not None:
            payload["input_population"] = self.input_detail
        return payload


def _sha256(arr: np.ndarray | sp.spmatrix) -> str:
    """Stable hash of a numeric array's bytes (no payload slicing).

    A sparse matrix is hashed through :func:`input_mapping.sparse_sha256`:
    ``np.ascontiguousarray(csr).tobytes()`` would hash an 8-byte object POINTER,
    which is stable within one process and meaningless across runs.
    """
    import hashlib

    if sp.issparse(arr):
        return sparse_sha256(arr)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def spectral_radius(matrix: csr_matrix, seed: int) -> float:
    """Largest ``|lambda|`` of a sparse matrix, pinned to a reproducible start.

    A free start vector would let two runs of the same seed disagree by
    ARPACK convergence noise. The caller passes the seed; we derive a fixed
    start vector from it.

    Args:
        matrix: Sparse matrix to estimate.
        seed: Seed for the start vector.

    Returns:
        Largest absolute eigenvalue, ``0.0`` if the matrix is empty.
    """
    if matrix.nnz == 0:
        return 0.0
    n = matrix.shape[0]
    if n <= 3:
        return float(np.max(np.abs(np.linalg.eigvals(matrix.toarray()))))
    start = make_rng(seed).standard_normal(n)
    largest = eigsh(matrix, k=1, return_eigenvectors=False, v0=start)
    return float(np.max(np.abs(largest)))


def rescale_to_spectral_radius(matrix: csr_matrix, target: float) -> csr_matrix:
    """Scale ``matrix`` so its largest ``|lambda|`` equals ``target``.

    Args:
        matrix: Input matrix.
        target: Target spectral radius.

    Returns:
        ``matrix * (target / rho)``; the original matrix is left untouched
        (CSR copies on slice).
    """
    rho = spectral_radius(matrix, seed=0)
    if rho <= SPECTRAL_RADIUS_EPS:
        return matrix.copy()
    return (matrix * (target / rho)).tocsr()


def constraint_match(
    reference: ReservoirTopology,
    candidate: csr_matrix,
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, float]:
    """Compare ``candidate`` to a reference topology on (N, M, rho).

    The comparison is reported, not enforced, so the test suite can both
    ``assert`` on it and store the gap.

    Args:
        reference: The frozen R0 topology.
        candidate: A candidate matrix from R1–R6.
        tolerance: Absolute tolerance on ``spectral_radius`` match; edge and
            node counts match exactly (modulo degree-rewired which may differ
            in self-loop policy).

    Returns:
        Mapping with ``n_nodes``, ``n_edges``, ``density``, ``spectral_radius``
        and their deltas versus the reference.
    """
    n = int(candidate.shape[0])
    m = int(candidate.nnz)
    rho = spectral_radius(candidate, seed=0)
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": m / (n * n) if n else 0.0,
        "spectral_radius": rho,
        "delta_n_nodes": n - reference.n_nodes,
        "delta_n_edges": m - reference.n_edges,
        "delta_spectral_radius": rho - reference.spectral_radius,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# R0 — the real fly connectome
# ---------------------------------------------------------------------------

def load_reservoir_topology_from_npz(
    npz_path: str,
    *,
    normalization: str = "n1_pre_l1",
    node_indices: np.ndarray | None = None,
    target_spectral_radius: float = 0.9,
    seed: int = 0,
) -> ReservoirTopology:
    """Build the R0 (real fly) reservoir topology.

    The six DATA-3 normalizations share the same loader shape — read the CSR
    triples at the normalization's NPZ key prefix, build a :class:`csr_matrix`,
    optionally subselect nodes, then rescale to ``target_spectral_radius``.
    The ``n0_raw`` case is the only special one: the build script (DATA-3)
    does not emit a redundant ``norm_n0_raw_*`` copy, because per meta.json
    "n0_raw ... same as adjacency_data". :data:`_NORMALIZATION_NPZ_PREFIX`
    routes that case to the ``adj_*`` keys, keeping the loader symmetric with
    the build.

    Args:
        npz_path: Path to ``olfactory_v1.npz`` (or any compatible artifact).
        normalization: One of :data:`ALLOWED_NORMALIZATIONS`. The biological
            normalization is taken from the stored array, never recomputed.
        node_indices: Subgraph indices into the full graph. ``None`` means
            "the full graph". E9 picks N ∈ {250, 500, 1000, 2000, 4000} by
            selecting nodes through :mod:`connectome.select_neurons`.
        target_spectral_radius: The ``rho`` every control will be rescaled
            to.
        seed: Seed for the spectral-radius estimate's start vector.

    Returns:
        A :class:`ReservoirTopology` describing R0.

    Raises:
        FileNotFoundError: If ``npz_path`` is missing.
        ValueError: On an unknown normalization or a non-CSR-shaped NPZ.
    """
    if normalization not in ALLOWED_NORMALIZATIONS:
        raise ValueError(
            f"normalization {normalization!r} is not one of "
            f"{list(ALLOWED_NORMALIZATIONS)}"
        )
    data = np.load(npz_path, allow_pickle=True)
    prefix = _NORMALIZATION_NPZ_PREFIX[normalization]
    for suffix in ("_data", "_indices", "_indptr", "_shape"):
        key = prefix + suffix
        if key not in data.files:
            raise ValueError(
                f"normalization {normalization!r} is not stored in the NPZ "
                f"(missing {key!r}); use one of {list(ALLOWED_NORMALIZATIONS)}"
            )
    shape = tuple(int(x) for x in data[prefix + "_shape"])
    matrix = csr_matrix(
        (data[prefix + "_data"], data[prefix + "_indices"], data[prefix + "_indptr"]),
        shape=shape,
    )

    if node_indices is not None:
        idx = np.asarray(node_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("node_indices must be a non-empty 1-D array")
        if idx.min() < 0 or idx.max() >= matrix.shape[0]:
            raise ValueError(
                f"node_indices out of range: min={idx.min()} max={idx.max()} "
                f"but matrix has {matrix.shape[0]} rows"
            )
        # Per-node subselection preserves the bidirectionality of the inner
        # square block only if all out-neighbours of selected nodes are also
        # selected. We accept leakage outside the block because the reservoir
        # sees only the ``(N, N)`` square; out-of-block contributions are
        # dropped. The DATA-3 select_neurons function picks nodes whose
        # out-of-block fan-out is small relative to M.
        matrix = matrix[idx, :][:, idx].tocsr()

    matrix = rescale_to_spectral_radius(matrix, target_spectral_radius)
    n = int(matrix.shape[0])
    m = int(matrix.nnz)
    rho = spectral_radius(matrix, seed=seed)
    density = m / (n * n) if n else 0.0
    return ReservoirTopology(
        matrix=matrix,
        n_nodes=n,
        n_edges=m,
        spectral_radius=rho,
        density=density,
        kind="R0_real_fly",
        normalization=normalization,
    )


# ---------------------------------------------------------------------------
# Topology factories — R1..R6
# ---------------------------------------------------------------------------

def _rng_shared(seed: int) -> np.random.Generator:
    """Independent stream for the topology draw (W_in / b use their own)."""
    return make_rng(int(seed) + 7919)


def make_weight_shuffled(
    reference: ReservoirTopology,
    seed: int,
) -> ReservoirTopology:
    """R1 — keep ``(rows, cols)`` of R0, shuffle the values.

    This nulls out *weight structure* but keeps the *graph*. R1 differs from
    R0 only by destroying the synapse-count ordering along each edge.

    Args:
        reference: The R0 topology.
        seed: Seed for the shuffle.

    Returns:
        The R1 topology, with the same shape, ``(N, M, rho)``.
    """
    matrix = reference.matrix.copy()
    rng = _rng_shared(seed)
    if matrix.nnz > 0:
        coo = matrix.tocoo()
        shuffled = rng.permutation(coo.data)
        matrix = csr_matrix(
            (shuffled, (coo.row, coo.col)),
            shape=reference.matrix.shape,
        )
    matrix = rescale_to_spectral_radius(matrix, reference.spectral_radius)
    return _wrap_topology(matrix, reference, kind="R1_weight_shuffled", normalization=None)


def make_degree_rewired(
    reference: ReservoirTopology,
    seed: int,
    *,
    n_swap_attempts: int = DOUBLE_EDGE_SWAP_ATTEMPTS,
    preserve_self_loops: bool = True,
) -> ReservoirTopology:
    """R2 — Maslov–Sneppen double-edge swap, preserving the degree sequence.

    R2 is the **primary** scientific control: it keeps the same nodes, the
    same degree distribution and the same edge count as R0, but breaks the
    specific wiring. If R0 does not beat R2, no specific-wiring claim is
    available.

    Args:
        reference: The R0 topology.
        seed: Seed for the swap sequence.
        n_swap_attempts: Attempts per edge; ``10`` is sufficient on sparse
            graphs of the densities encountered here.
        preserve_self_loops: If ``True``, swaps that would introduce or
            remove a self-loop are rejected.

    Returns:
        The R2 topology with the same ``(N, M, rho)`` and the same degree
        sequences as R0.
    """
    coo = reference.matrix.tocoo()
    rows = coo.row.astype(np.int64).copy()
    cols = coo.col.astype(np.int64).copy()
    n_edges = rows.size
    n_nodes = int(reference.matrix.shape[0])

    rng = _rng_shared(seed)
    if n_edges < 2:
        return _wrap_topology(reference.matrix.copy(), reference,
                              kind="R2_degree_rewired", normalization=None)

    attempts = n_edges * int(n_swap_attempts)
    for _ in range(attempts):
        # Pick two distinct edges (i, j) and (k, l).
        a_idx, b_idx = rng.integers(0, n_edges, size=2)
        if a_idx == b_idx:
            continue
        i, j = rows[a_idx], cols[a_idx]
        k, l = rows[b_idx], cols[b_idx]
        if i == k or j == l:
            continue
        # The double-edge swap replaces (i, j) and (k, l) with (i, l) and
        # (k, j). We must avoid creating duplicates; with the original edges
        # removed we check only the new ones against the rest of the list.
        if preserve_self_loops and (i == l or k == j):
            continue
        new_a_row, new_a_col = i, l
        new_b_row, new_b_col = k, j
        # Deduplicate against the unchanged edges.
        unchanged_mask = np.ones(n_edges, dtype=bool)
        unchanged_mask[[a_idx, b_idx]] = False
        rest_rows = rows[unchanged_mask]
        rest_cols = cols[unchanged_mask]
        # Check neither new edge duplicates a rest edge.
        dup_a = np.any((rest_rows == new_a_row) & (rest_cols == new_a_col))
        dup_b = np.any((rest_rows == new_b_row) & (rest_cols == new_b_col))
        if dup_a or dup_b:
            continue
        # Apply.
        rows[a_idx], cols[a_idx] = new_a_row, new_a_col
        rows[b_idx], cols[b_idx] = new_b_row, new_b_col

    rewired = csr_matrix(
        (np.ones(n_edges, dtype=np.float64), (rows, cols)),
        shape=(n_nodes, n_nodes),
    )
    rewired = rescale_to_spectral_radius(rewired, reference.spectral_radius)
    return _wrap_topology(rewired, reference, kind="R2_degree_rewired", normalization=None)


def make_degree_rewired_weight_preserving(
    reference: ReservoirTopology,
    seed: int,
    *,
    target_overlap: float = 0.20,
    time_budget_s: float = 600.0,
    max_attempts: int | None = None,
) -> ReservoirTopology:
    """R2 as a TRUE wiring-only counterfactual (v2 D6, criteria C4.1-C4.8).

    The v1 :func:`make_degree_rewired` rewrites the matrix with ``np.ones(n_edges)``
    and rescales it, so R2 was a **uniform-weight graph**: R0-vs-R2 was a joint
    contrast of wiring AND weight structure, and "the wiring does not matter" is not
    what that comparison measured.

    This factory keeps the nodes, the directed degree sequence, the GLOBAL weight
    multiset and every per-source outgoing weight multiset, and changes only the
    wiring -- a directed double-edge swap that leaves each weight with its own source.
    It does NOT rescale: preserving the weight multiset and matching a spectral radius
    are mutually exclusive, and both radii are reported so the difference is visible.

    Args:
        reference: The R0 topology.
        seed: Seed of the swap sequence.
        target_overlap: C4.6's ceiling on the R0/R2 edge overlap.
        time_budget_s: C4.7's wall-time budget (the signed threshold is 600 s; the
            caller may pass the 1 h hard cap, but a run over 600 s is a FAIL).
        max_attempts: Cap on swap proposals.

    Returns:
        The R2 topology, with ``counterfactual`` carrying the chain's report.
    """
    from drososense.reservoir.r2_counterfactual import (
        counterfactual_quality,
        weight_preserving_degree_rewire,
    )

    rewired, report = weight_preserving_degree_rewire(
        reference.matrix,
        seed=seed,
        target_overlap=target_overlap,
        time_budget_s=time_budget_s,
        max_attempts=max_attempts,
    )
    topology = _wrap_topology(
        rewired, reference, kind="R2_degree_rewired", normalization=None
    )
    quality = counterfactual_quality(reference.matrix, rewired, report)
    return ReservoirTopology(
        matrix=topology.matrix,
        n_nodes=topology.n_nodes,
        n_edges=topology.n_edges,
        spectral_radius=topology.spectral_radius,
        density=topology.density,
        kind=topology.kind,
        normalization=topology.normalization,
        counterfactual={
            "report": report.as_dict(),
            "quality": quality,
            "reference_spectral_radius": float(reference.spectral_radius),
        },
    )


def make_random_sparse(
    reference: ReservoirTopology,
    seed: int,
) -> ReservoirTopology:
    """R3 — Erdős–Rényi-style draw with the same ``(N, M)`` as R0.

    Args:
        reference: The R0 topology.
        seed: Seed for the draw.

    Returns:
        The R3 topology; same ``N, M`` and rescaled ``rho``, no degree or
        weight structure preserved.
    """
    n = int(reference.matrix.shape[0])
    m = int(reference.n_edges)
    if m <= 0 or n <= 0:
        return _wrap_topology(csr_matrix((n, n)), reference,
                              kind="R3_random_sparse", normalization=None)
    rng = _rng_shared(seed)
    # Sample without replacement from the off-diagonal cells. Using
    # ``rng.choice`` over ``n * (n - 1)`` then mapping ``(i, j) = (flat //
    # (n - 1), flat % (n - 1))`` with a conditional bump for the diagonal
    # skip avoids the post-draw dedup that would silently drop edges from
    # CSR's ``(data, (row, col))`` constructor.
    cells = n * n - n
    m = min(m, cells)
    flat = rng.choice(cells, size=m, replace=False)
    rows = flat // (n - 1)
    offsets = flat % (n - 1)
    cols = offsets + (offsets >= rows).astype(np.int64)
    values = rng.uniform(-1.0, 1.0, size=m)
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    matrix = rescale_to_spectral_radius(matrix, reference.spectral_radius)
    return _wrap_topology(matrix, reference, kind="R3_random_sparse", normalization=None)


def make_er_esn(
    reference: ReservoirTopology,
    seed: int,
    *,
    target_density: float = 0.05,
) -> ReservoirTopology:
    """R4 — standard echo-state-network random sparse draw.

    This is the existing :class:`drososense.reservoir.esn.EchoStateNetwork`
    topology, surfaced as a control with the same ``(N, rho)`` as R0.

    Args:
        reference: The R0 topology.
        seed: Seed for the draw.
        target_density: Density of the random draw. ``0.05`` matches the
            ``EchoStateNetwork`` default.

    Returns:
        The R4 topology. The edge count is ``target_density * N^2``, not
            matched to R0; that is the cost of "different topology". The
            spectral radius is matched.
    """
    n = int(reference.matrix.shape[0])
    if n <= 0:
        return _wrap_topology(csr_matrix((n, n)), reference,
                              kind="R4_er_esn", normalization=None)
    rng = _rng_shared(seed)
    n_edges = max(1, int(round(target_density * n * n)))
    cells = n * n - n
    n_edges = min(n_edges, cells)
    flat = rng.choice(cells, size=n_edges, replace=False)
    rows = flat // (n - 1)
    offsets = flat % (n - 1)
    cols = offsets + (offsets >= rows).astype(np.int64)
    values = rng.uniform(-1.0, 1.0, size=n_edges)
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    matrix = rescale_to_spectral_radius(matrix, reference.spectral_radius)
    return _wrap_topology(matrix, reference, kind="R4_er_esn", normalization=None)


def make_small_world(
    reference: ReservoirTopology,
    seed: int,
    *,
    rewire_probability: float = 0.1,
) -> ReservoirTopology:
    """R5 — Watts–Strogatz ring lattice, rewired with probability ``p``.

    The ring-lattice base matches the structural motif: locally connected,
    small characteristic path length. ``p=0.1`` is the canonical
    Watts–Strogatz parameter.

    Args:
        reference: The R0 topology.
        seed: Seed for the rewire draw.
        rewire_probability: Per-edge rewiring probability.

    Returns:
        The R5 topology. Edge count matches R0 only when ``k * N / 2 ==
            M``; otherwise the run record reports the gap explicitly.
    """
    n = int(reference.matrix.shape[0])
    m = int(reference.n_edges)
    if n < 3 or m <= 0:
        return _wrap_topology(csr_matrix((n, n)), reference,
                              kind="R5_small_world", normalization=None)
    rng = _rng_shared(seed)
    # Each node has k half-edges to its right neighbours; total edges = k * N / 2.
    k = max(2, int(round(2 * m / n)))
    if k % 2 == 1:
        k += 1  # Watts–Strogatz requires k even.
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for offset in range(1, k // 2 + 1):
            j = (i + offset) % n
            edges.add((i, j))
    # Cap at M to match R0 edge count exactly.
    if len(edges) > m:
        edges = set(list(edges)[:m])
    elif len(edges) < m:
        # Add random extras (still undirected; not self-loops).
        extras_needed = m - len(edges)
        attempts = 0
        while extras_needed > 0 and attempts < 10 * extras_needed:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                attempts += 1
                continue
            key = (min(i, j), max(i, j))
            if key in edges:
                attempts += 1
                continue
            edges.add(key)
            extras_needed -= 1
            attempts += 1

    # Rewire with probability p.
    rewired: set[tuple[int, int]] = set()
    for (i, j) in edges:
        if rng.random() < rewire_probability:
            attempts = 0
            while attempts < 10:
                new_j = int(rng.integers(0, n))
                if new_j != i and (min(i, new_j), max(i, new_j)) not in rewired:
                    rewired.add((min(i, new_j), max(i, new_j)))
                    break
                attempts += 1
            else:
                rewired.add((i, j))
        else:
            rewired.add((i, j))

    rows = np.array([u for (u, v) in rewired], dtype=np.int64)
    cols = np.array([v for (u, v) in rewired], dtype=np.int64)
    values = np.ones(rows.size, dtype=np.float64)
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    matrix = rescale_to_spectral_radius(matrix, reference.spectral_radius)
    return _wrap_topology(matrix, reference, kind="R5_small_world", normalization=None)


def make_dense_random(
    reference: ReservoirTopology,
    seed: int,
    *,
    target_density: float = 0.5,
) -> ReservoirTopology:
    """R6 — random dense Erdős–Rényi.

    Supplied as a "high-density random" sanity check: if a 500-cell reservoir
    with 50% density matches R0, then "more edges" is the explanation and
    the specific wiring does not matter.

    Args:
        reference: The R0 topology.
        seed: Seed for the draw.
        target_density: Density of the random draw.

    Returns:
        The R6 topology. Edge count is ``target_density * N^2``; the
            spectral radius is rescaled to R0's.
    """
    n = int(reference.matrix.shape[0])
    if n <= 0:
        return _wrap_topology(csr_matrix((n, n)), reference,
                              kind="R6_dense_random", normalization=None)
    rng = _rng_shared(seed)
    n_edges = max(1, int(round(target_density * n * n)))
    cells = n * n - n
    n_edges = min(n_edges, cells)
    flat = rng.choice(cells, size=n_edges, replace=False)
    rows = flat // (n - 1)
    offsets = flat % (n - 1)
    cols = offsets + (offsets >= rows).astype(np.int64)
    values = rng.uniform(-1.0, 1.0, size=n_edges)
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    matrix = rescale_to_spectral_radius(matrix, reference.spectral_radius)
    return _wrap_topology(matrix, reference, kind="R6_dense_random", normalization=None)


def _wrap_topology(
    matrix: csr_matrix,
    reference: ReservoirTopology,
    *,
    kind: str,
    normalization: str | None,
) -> ReservoirTopology:
    """Build a :class:`ReservoirTopology` for a control matrix."""
    n = int(matrix.shape[0])
    m = int(matrix.nnz)
    rho = spectral_radius(matrix, seed=0)
    density = m / (n * n) if n else 0.0
    return ReservoirTopology(
        matrix=matrix,
        n_nodes=n,
        n_edges=m,
        spectral_radius=rho,
        density=density,
        kind=kind,
        normalization=normalization,
    )


# ---------------------------------------------------------------------------
# Shared input / bias
# ---------------------------------------------------------------------------

def make_shared(
    n_nodes: int,
    n_channels: int,
    seed: int,
    *,
    input_scale: float = 0.5,
    mapping: InputMapping | None = None,
) -> ReservoirShared:
    """Draw ``W_in`` and ``b`` once for the family.

    With ``mapping`` given, ``W_in`` IS that mapping's matrix -- the declared
    pathway object carries its own support, density and provenance, and this
    function does not second-guess it. The bias is still drawn here: the bias is
    per-node and is not part of the input pathway.

    Args:
        n_nodes: Reservoir size (``N``).
        n_channels: Number of input channels (``C``).
        seed: Seed for the draws.
        input_scale: Uniform-``[-s, s]`` half-width.
        mapping: A declared :class:`~drososense.reservoir.input_mapping.InputMapping`.
            Its shape must match ``(n_nodes, n_channels)``, or the W_in a record
            describes would not be the W_in the reservoir used.

    Returns:
        A :class:`ReservoirShared`.

    Raises:
        ValueError: On non-positive node or channel counts, or a mapping whose
            shape contradicts them.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    if n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {n_channels}")
    rng = make_rng(int(seed))
    if mapping is not None:
        if mapping.w_in.shape != (int(n_nodes), int(n_channels)):
            raise ValueError(
                f"input mapping {mapping.name!r} has shape {mapping.w_in.shape}, "
                f"expected {(int(n_nodes), int(n_channels))}"
            )
        return ReservoirShared(
            w_in=mapping.w_in.tocsr(),
            bias=rng.uniform(-input_scale, input_scale, size=n_nodes).astype(np.float64),
            seed=int(seed),
            input_scale=float(input_scale),
            input_mapping=mapping.name,
            input_detail=mapping.describe(),
        )
    w_in = rng.uniform(-input_scale, input_scale, size=(n_nodes, n_channels))
    bias = rng.uniform(-input_scale, input_scale, size=n_nodes)
    return ReservoirShared(
        w_in=w_in.astype(np.float64),
        bias=bias.astype(np.float64),
        seed=int(seed),
        input_scale=float(input_scale),
    )


# ---------------------------------------------------------------------------
# Reservoir model
# ---------------------------------------------------------------------------

class _ReservoirBase(BaseModel):
    """Common implementation of the leaky dynamics and the linear readout.

    Every R0–R6 model delegates here so the comparison is like-for-like. The
    only thing the subclasses override is ``_build_topology``.

    The state-update is the canonical leaky ESN update::

        h_t = (1 - alpha) h_{t-1} + alpha * tanh(g * A_hat h_{t-1} + W_in x_t + b)

    followed by a per-window state reduction (``last`` or ``mean``) and a
    ridge or logistic readout. The reservoir matrix, ``W_in`` and ``b`` are
    drawn once, frozen for the rest of the run, and never updated by the
    readout training.

    Note:
        ``A`` here is :math:`\\hat A`, the *rescaled* biological or control
        matrix — its largest absolute eigenvalue equals the configured
        ``spectral_radius``. The rescale is one-shot, at build time.
    """

    model_id = "reservoir_base"

    def __init__(
        self,
        task: TaskType,
        seed: int,
        params: dict[str, Any] | None,
        topology: ReservoirTopology,
        shared: ReservoirShared,
        *,
        n_channels: int,
    ) -> None:
        merged = {**DEFAULT_RESERVOIR_PARAMS, **(params or {})}
        super().__init__(task, seed, merged)
        self._topology = topology
        self._shared = shared
        self.n_channels = n_channels
        self._w_out: np.ndarray | None = None
        self._n_outputs: int | None = None
        self._readout_kind = str(merged["readout"])

    # Subclasses implement this. Returns a :class:`ReservoirTopology`.
    def _build_topology(self) -> ReservoirTopology:
        raise NotImplementedError

    # -- reservoir dynamics -------------------------------------------------
    def _drive(self, x: np.ndarray) -> np.ndarray:
        """Run the reservoir on a window tensor and return per-window states.

        Args:
            x: Tensor of shape ``(n_samples, length, n_channels)``.

        Returns:
            Per-window states of shape ``(n_samples, n_nodes)``.

        Raises:
            RuntimeError: If the reservoir is uninitialised (call ``fit``
                first).
        """
        leak = float(self.params["leak"])
        gain = float(self.params["gain"])
        pooling = str(self.params["state_pooling"])
        washout = int(self.params["washout"])
        n_nodes = self._topology.n_nodes
        A = self._topology.matrix
        w_in = self._shared.w_in
        bias = self._shared.bias

        n_samples, length, _ = x.shape
        washout = min(washout, max(length - 1, 0))
        states = np.zeros((n_samples, n_nodes), dtype=np.float64)
        # Pre-transposed input — keeps the inner loop to one sparse matvec.
        if sp.issparse(w_in):
            # A sparse W_in (the v2 ORN-aligned map) is applied as one sparse
            # matmul: x @ w_in.T would densify it back to (n, L, N).
            flat = np.asarray((w_in @ x.reshape(-1, x.shape[-1]).T).T, dtype=np.float64)
            input_projection = flat.reshape(n_samples, length, n_nodes)
        else:
            input_projection = x @ w_in.T  # (n_samples, length, n_nodes)

        for i in range(n_samples):
            state = np.zeros(n_nodes, dtype=np.float64)
            accumulated = np.zeros(n_nodes, dtype=np.float64)
            collected = 0
            for t in range(length):
                # Sparse matvec: scales linearly with nnz, not N^2.
                drive = gain * (A @ state) + input_projection[i, t] + bias
                state = (1.0 - leak) * state + leak * np.tanh(drive)
                if t >= washout:
                    accumulated += state
                    collected += 1
            if collected and pooling == "mean":
                states[i] = accumulated / collected
            else:
                states[i] = state
        return states

    # -- readout ------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Drive the reservoir, train the readout.

        Args:
            x: Flattened window matrix, reshaped back to ``(n, L, C)``.
            y: Targets.

        Raises:
            ValueError: On a non-multiple flattened width.
        """
        total = x.shape[1]
        if total % self.n_channels != 0:
            raise ValueError(
                f"flattened width {total} is not a multiple of "
                f"n_channels {self.n_channels}"
            )
        length = total // self.n_channels
        tensor = x.reshape(x.shape[0], length, self.n_channels)

        states = self._drive(tensor)
        augmented = np.column_stack([states, np.ones(states.shape[0])])

        if self.task == "classification":
            classes = np.unique(y)
            if not np.array_equal(classes, np.arange(len(classes))):
                raise ValueError(
                    f"{self.model_id}: classification labels must be contiguous "
                    f"from 0, got {classes.tolist()}"
                )
            self._n_outputs = int(len(classes))
            targets = np.zeros((y.shape[0], self._n_outputs), dtype=np.float64)
            targets[np.arange(y.shape[0]), y.astype(int)] = 1.0
        else:
            self._n_outputs = 1
            targets = y.reshape(-1, 1).astype(np.float64)

        if self._readout_kind == "logistic":
            self._w_out = self._fit_logistic_readout(augmented, y)
        else:
            self._w_out = self._ridge_solve(augmented, targets)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self._w_out is None:
            raise RuntimeError("readout not fitted")
        length = x.shape[1] // self.n_channels
        states = self._drive(x.reshape(x.shape[0], length, self.n_channels))
        augmented = np.column_stack([states, np.ones(states.shape[0])])
        scores = augmented @ self._w_out
        if self.task == "classification":
            return scores.argmax(axis=1)
        return scores[:, 0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        flat = self.flatten(x)
        length = flat.shape[1] // self.n_channels
        states = self._drive(
            flat.reshape(flat.shape[0], length, self.n_channels)
        )
        augmented = np.column_stack([states, np.ones(states.shape[0])])
        scores = augmented @ self._w_out
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    # -- readout solvers ---------------------------------------------------
    def _ridge_solve(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Solve ``(H^T H + lambda I) W = H^T Y`` in float64.

        Args:
            features: Design matrix ``(n_samples, n_features)``.
            targets: Target matrix ``(n_samples, n_outputs)``.

        Returns:
            Readout weights of shape ``(n_features, n_outputs)``.
        """
        ridge_lambda = float(self.params["ridge_lambda"])
        gram = features.T @ features
        regulariser = ridge_lambda * np.eye(gram.shape[0])
        return np.linalg.solve(gram + regulariser, features.T @ targets)

    def _fit_logistic_readout(
        self, features: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Fit a multinomial logistic regression on the reservoir states.

        Args:
            features: Design matrix.
            y: Integer class labels.

        Returns:
            Weight matrix of shape ``(n_features + 1, n_classes)``.

        Raises:
            ValueError: If used for regression.
        """
        if self.task != "classification":
            raise ValueError(
                "the logistic readout is only available for classification"
            )
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=1.0 / max(float(self.params["ridge_lambda"]), 1e-12),
            max_iter=2000,
            random_state=self.seed,
        )
        model.fit(features, y.astype(int))
        coefficients = model.coef_
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        return np.vstack([coefficients.T, model.intercept_.reshape(1, -1)])

    # -- reporting ---------------------------------------------------------
    def n_trainable_parameters(self) -> int | None:
        if self._w_out is None or self._n_outputs is None:
            return None
        return int(self._w_out.size)

    def n_frozen_parameters(self) -> int:
        if self._topology is None or self._shared is None:
            return 0
        # nnz for the adjacency AND for a sparse W_in: this counts non-zero
        # connections, and for a dense matrix nnz == size.
        w_in_terms = (
            int(self._shared.w_in.nnz)
            if sp.issparse(self._shared.w_in)
            else int(self._shared.w_in.size)
        )
        return int(self._topology.matrix.nnz + w_in_terms + self._shared.bias.size)

    def reservoir_sparsity(self) -> float:
        if self._topology is None:
            return 0.0
        n = self._topology.n_nodes
        if n == 0:
            return 0.0
        return float(1.0 - self._topology.matrix.nnz / (n * n))

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["frozen_parameters"] = self.n_frozen_parameters()
        info["reservoir_sparsity"] = self.reservoir_sparsity()
        info["readout"] = self._readout_kind
        info["topology"] = self._topology.describe()
        info["shared"] = self._shared.describe()
        info["match_to_R0"] = constraint_match(
            _reference_or_self(self._topology), self._topology.matrix
        )
        return info


def _reference_or_self(topology: ReservoirTopology) -> ReservoirTopology:
    """Pass-through helper; tests inject an explicit reference instead."""
    return topology


# ---------------------------------------------------------------------------
# Concrete R0–R6 model classes
# ---------------------------------------------------------------------------

class R0RealFlyReservoir(_ReservoirBase):
    """R0 — the real olfactory connectome, frozen.

    The biological matrix is loaded from a DATA-3 NPZ artifact and rescaled
    to the configured ``spectral_radius``. No synapse-level claim is made:
    the entries are :data:`synapse-count-informed structural weights
    <ReservoirTopology.weight_semantics>`.
    """

    model_id = "R0_real_fly"


class R1WeightShuffledReservoir(_ReservoirBase):
    """R1 — same graph as R0, shuffled values.
    """

    model_id = "R1_weight_shuffled"


class R2DegreeRewiredReservoir(_ReservoirBase):
    """R2 — primary scientific control.

    Same nodes, same edge count, same degree sequence as R0; only the
    specific wiring is randomised through Maslov–Sneppen double-edge swaps.
    """

    model_id = "R2_degree_rewired"


class R3RandomSparseReservoir(_ReservoirBase):
    """R3 — random sparse control with R0's (N, M).
    """

    model_id = "R3_random_sparse"


class R4ErEsnReservoir(_ReservoirBase):
    """R4 — random Erdős–Rényi-style ESN matrix.
    """

    model_id = "R4_er_esn"


class R5SmallWorldReservoir(_ReservoirBase):
    """R5 — Watts–Strogatz small-world control.
    """

    model_id = "R5_small_world"


class R6DenseRandomReservoir(_ReservoirBase):
    """R6 — random dense control.
    """

    model_id = "R6_dense_random"


# ---------------------------------------------------------------------------
# Family construction
# ---------------------------------------------------------------------------

def build_topology_family(
    npz_path: str,
    *,
    seed: int,
    n_channels: int,
    normalization: str = "n1_pre_l1",
    node_indices: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, _ReservoirBase]:
    """Build the R0–R6 family on a shared ``W_in`` and ``b``.

    The family is reproducible from ``(npz_path, n_channels, seed,
    normalization, node_indices, params)`` alone.

    Args:
        npz_path: Path to ``olfactory_v1.npz`` (DATA-3 artifact).
        seed: Run seed for ``W_in`` / ``b`` / topology draws.
        n_channels: Number of input channels.
        normalization: R0 normalization (DATA-3 scheme).
        node_indices: Subgraph selection.
        params: Hyperparameter overrides.

    Returns:
        Mapping from family id to model instance.

    Raises:
        FileNotFoundError: If ``npz_path`` is missing.
        ValueError: On a bad normalization or empty subgraph.
    """
    merged = {**DEFAULT_RESERVOIR_PARAMS, **(params or {})}
    r0 = load_reservoir_topology_from_npz(
        npz_path,
        normalization=normalization,
        node_indices=node_indices,
        target_spectral_radius=float(merged["spectral_radius"]),
        seed=seed,
    )
    shared = make_shared(
        n_nodes=r0.n_nodes,
        n_channels=n_channels,
        seed=seed,
        input_scale=float(merged["input_scale"]),
    )
    controls: dict[str, ReservoirTopology] = {
        "R0_real_fly": r0,
        "R1_weight_shuffled": make_weight_shuffled(r0, seed),
        "R2_degree_rewired": make_degree_rewired(r0, seed),
        "R3_random_sparse": make_random_sparse(r0, seed),
        "R4_er_esn": make_er_esn(r0, seed),
        "R5_small_world": make_small_world(r0, seed),
        "R6_dense_random": make_dense_random(r0, seed),
    }

    family: dict[str, _ReservoirBase] = {}
    for family_id, topo in controls.items():
        task: TaskType = "classification"  # The runner overrides per task.
        cls = {
            "R0_real_fly": R0RealFlyReservoir,
            "R1_weight_shuffled": R1WeightShuffledReservoir,
            "R2_degree_rewired": R2DegreeRewiredReservoir,
            "R3_random_sparse": R3RandomSparseReservoir,
            "R4_er_esn": R4ErEsnReservoir,
            "R5_small_world": R5SmallWorldReservoir,
            "R6_dense_random": R6DenseRandomReservoir,
        }[family_id]
        family[family_id] = cls(
            task=task,
            seed=seed,
            params=merged,
            topology=topo,
            shared=shared,
            n_channels=n_channels,
        )
    return family


__all__ = [
    "ALLOWED_NORMALIZATIONS",
    "DEFAULT_RESERVOIR_PARAMS",
    "PRIMARY_CONTROL",
    "ReservoirShared",
    "ReservoirTopology",
    "TOPOLOGY_FAMILY_IDS",
    "build_topology_family",
    "constraint_match",
    "load_reservoir_topology_from_npz",
    "make_dense_random",
    "make_degree_rewired",
    "make_degree_rewired_weight_preserving",
    "make_er_esn",
    "make_random_sparse",
    "make_shared",
    "make_small_world",
    "make_weight_shuffled",
    "rescale_to_spectral_radius",
    "spectral_radius",
    "R0RealFlyReservoir",
    "R1WeightShuffledReservoir",
    "R2DegreeRewiredReservoir",
    "R3RandomSparseReservoir",
    "R4ErEsnReservoir",
    "R5SmallWorldReservoir",
    "R6DenseRandomReservoir",
]