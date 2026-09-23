"""Frozen input apportionment, canonical adjacency identity, and the amended scale rule.

Three contracts that had to be fixed before any real Stage-1 run, all deterministic and all
independent of any A3 verdict, any family score and any food measurement.

1. :func:`apportion_din` -- extends the frozen ``ORN:PN:KC = 2:2:1`` typed-alignment ratio to
   ANY ``Din`` by largest-remainder apportionment with a permanently fixed ``ORN -> PN -> KC``
   tie-break. The point is not that the resulting numbers are optimal; it is that they are
   **not chosen after seeing a result**.

2. :func:`canonical_adjacency_hash` -- F1's identity from its CONTENT, not from a digest
   produced through a spectral-radius estimator. The historical ``A_hash`` came from an
   unseeded ``eigsh`` path and is therefore non-identifying (amendment 5); this replaces it
   with a hash over a canonicalised CSR serialisation.

3. :func:`global_normalisation_scalar` and :func:`weights_up_to_global_scalar` -- the amended
   weight rule. ``rho = 0.95`` holds EXACTLY for every dynamical family, and the weight
   structure is matched *up to one global scalar* rather than by absolute equality. The
   earlier "absolute weight multiset wins, rho may drift 0.64 %" rule is superseded, because
   A3 is known to be scale-sensitive.
"""

from __future__ import annotations

import hashlib
import struct
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "TYPED_ALIGNMENT_ORDER",
    "TYPED_ALIGNMENT_WEIGHTS",
    "FROZEN_RHO_EXACT",
    "apportion_din",
    "apportionment_table",
    "canonical_adjacency_bytes",
    "canonical_adjacency_hash",
    "global_normalisation_scalar",
    "weights_up_to_global_scalar",
]

#: The layer order, and the tie-break order. FIXED: declared once, never reselected.
TYPED_ALIGNMENT_ORDER: tuple[str, ...] = ("ORN", "PN", "KC")

#: The frozen typed-alignment ratio, from the v2 amendment 2/3 rule (`din=5` -> 2:2:1).
TYPED_ALIGNMENT_WEIGHTS: tuple[int, ...] = (2, 2, 1)

#: The amended exact scale. Every dynamical family reaches this radius EXACTLY.
FROZEN_RHO_EXACT = 0.95


def apportion_din(din: int) -> dict[str, int]:
    """Split ``din`` input channels across the typed layers by largest remainder.

    Deterministic everywhere, including ties: seats are awarded by descending fractional
    remainder, and ties break by the permanently fixed ``ORN -> PN -> KC`` order.

    Verified against the stated expectations: ``din=5 -> {2,2,1}`` (reproducing the frozen
    ``din=5`` allocation exactly, which is the consistency requirement), ``din=6 ->
    {3,2,1}``, ``din=8 -> {3,3,2}``.

    Args:
        din: Total input channels. Must be >= 1.

    Returns:
        ``{layer: channels}`` summing to ``din``.

    Raises:
        ValueError: if ``din < 1``.
    """
    d = int(din)
    if d < 1:
        raise ValueError(f"din must be >= 1, got {din}")
    total = sum(TYPED_ALIGNMENT_WEIGHTS)
    quotas = [Fraction(d * w, total) for w in TYPED_ALIGNMENT_WEIGHTS]
    floors = [int(q) for q in quotas]
    remaining = d - sum(floors)
    # descending fractional remainder; ties by the frozen layer order
    ranked = sorted(
        range(len(TYPED_ALIGNMENT_WEIGHTS)),
        key=lambda i: (-(quotas[i] - floors[i]), TYPED_ALIGNMENT_ORDER[i]),
    )
    out = floors[:]
    for i in ranked[:remaining]:
        out[i] += 1
    return dict(zip(TYPED_ALIGNMENT_ORDER, (int(v) for v in out)))


def apportionment_table(dins: Sequence[int]) -> dict[int, dict[str, int]]:
    """The allocation for several widths, for the amendment's record."""
    return {int(d): apportion_din(int(d)) for d in dins}


# ---------------------------------------------------------------------------
# canonical identity
# ---------------------------------------------------------------------------


def canonical_adjacency_bytes(
    A: sp.spmatrix, *, dtype: str = "<f8"
) -> bytes:
    """A canonical byte serialisation of a sparse adjacency, with no estimator involved.

    Canonicalisation is the whole point: the same graph must hash identically regardless of
    construction order, matrix format, stored zeros, or duplicated entries. So the matrix is

    * converted to CSR,
    * reduced to a canonical form (sorted indices, duplicates summed, explicit zeros dropped),
    * re-indexed so only the nonzero pattern and its values remain,

    and the serialisation is a fixed little-endian layout of
    ``(n_rows, n_cols, nnz, dtype)`` followed by ``indptr``, ``indices`` and ``data``. NaN
    and infinity are refused rather than hashed, because a non-finite weight would make two
    runs disagree without either being wrong.

    No spectral-radius estimator, no normalisation and no floating-point accumulation order
    enters this function.
    """
    M = sp.csr_matrix(A).copy()
    M.sum_duplicates()
    M.eliminate_zeros()
    M.sort_indices()
    M = M.astype(np.dtype(dtype).name)
    if M.nnz and not np.all(np.isfinite(M.data)):
        raise ValueError("adjacency contains non-finite weights; refusing to hash it")
    header = b"".join(
        struct.pack("<q", int(v)) for v in (M.shape[0], M.shape[1], int(M.nnz))
    ) + np.dtype(dtype).str.encode("ascii")
    return b"".join(
        [
            header,
            np.ascontiguousarray(M.indptr, dtype=np.int64).tobytes(),
            np.ascontiguousarray(M.indices, dtype=np.int64).tobytes(),
            np.ascontiguousarray(M.data, dtype=np.dtype(dtype)).tobytes(),
        ]
    )


def canonical_adjacency_hash(A: sp.spmatrix, *, dtype: str = "<f8") -> str:
    """SHA-256 of :func:`canonical_adjacency_bytes`. Independent of any rho estimator."""
    return hashlib.sha256(canonical_adjacency_bytes(A, dtype=dtype)).hexdigest()


# ---------------------------------------------------------------------------
# the amended scale rule
# ---------------------------------------------------------------------------


def global_normalisation_scalar(rho_raw: float, target: float = FROZEN_RHO_EXACT) -> float:
    """The single scalar that takes a family to ``target``: ``alpha = target / rho_raw``."""
    r = float(rho_raw)
    if r <= 0.0:
        raise ValueError(f"rho_raw must be > 0 to normalise, got {rho_raw}")
    return float(target) / r


def weights_up_to_global_scalar(
    reference_weights: np.ndarray | Sequence[float],
    rho_raw: float,
    *,
    target: float = FROZEN_RHO_EXACT,
) -> tuple[np.ndarray, float]:
    """The amended weight rule: the SAME weights, times one global scalar.

    Returns:
        ``(alpha * W, alpha)`` where ``alpha = target / rho_raw``.

    What this preserves is what actually matters for a topology contrast: the weight
    *ranking*, the pairwise weight *ratios*, and the *shape* of the distribution. What it
    deliberately does NOT claim is absolute equality of the released weights, because that
    claim is incompatible with reaching ``rho = target`` exactly -- and A3 is
    scale-sensitive, so exact ``rho`` wins.
    """
    w = np.asarray(reference_weights, dtype=np.float64)
    alpha = global_normalisation_scalar(rho_raw, target)
    return (w * alpha), alpha
