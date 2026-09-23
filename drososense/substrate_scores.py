"""v3 substrate scores: S1 (Krylov controllability), S2 (eigenmode participation), S3 (spectral).

Pre-registered in `docs/v3_preregistration.md` §3. S1 is the PRIMARY selection score; S2 and
S3 are DIAGNOSTICS and the selection engine must never read S3.

**Identical measurement discipline for S1 across candidates** (declared, per the owner's
implementation constraints): same ``Din``, same typed-aligned input rule, same ``B``
determinism, same ``A`` normalization, same Krylov depth ``K = 16``, same numerical
tolerance, same effective-rank definition (participation ratio over singular values). ``B``
is NEVER rescaled per candidate -- rescaling would make S1 a comparison of substrate AND
input scale at once.

**S2 uses LEFT eigenvectors.** ``A`` is directed and generally non-normal, so for
``p_i = ||v_i^T B||`` the right eigenvectors are the wrong accessibility object: the direct
input coupling to mode ``i`` is ``l_i^T B`` with ``l_i`` the LEFT eigenvector
(``l_i^T A = lambda_i l_i^T``). When the eigenbasis is ill-conditioned we fall back to the
**left Schur vectors**, which are orthonormal and numerically stable, and we always report
the eigenvector condition number and the non-normality so a wildly non-normal candidate
cannot look good by accident.

**S3 is report-only**: spectral radius, eigenvalue-modulus entropy, singular-value spread,
Henrici departure from normality, and a transient-amplification proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

#: The Krylov depth, fixed by pre-registration.
KRYLOV_K = 16

#: Numerical tolerance for rank / effective-rank decisions, fixed across candidates.
NUMERICAL_TOLERANCE = 1e-10

#: Dense eigendecomposition is used when a candidate is at most this large (candidates are
#: <= 1000 nodes by construction, so this is always true in practice).
DENSE_LIMIT = 2000


@dataclass(frozen=True)
class KrylovScore:
    """S1 -- the primary selection score, plus the scale-robust diagnostics.

    Attributes:
        effective_rank: the pre-registered S1 -- participation ratio
            ``(sum sigma)^2 / sum sigma^2`` over the singular values of the
            concatenated Krylov block, with an ABSOLUTE tolerance.
        numerical_rank: singular values above the absolute tolerance.
        numerical_rank_relative: singular values above
            ``tol_rel * sigma_max``. This is the scale-robust reachable dimension, and it
            is reported BECAUSE the participation ratio is not: the singular values of
            ``[B, A·B, …, A^K·B]`` span a huge dynamic range (the dominant power's norm is
            ``~rho^K`` times the others), so the participation ratio is dominated by
            whichever power has the largest norm. Both are published so the selection's
            sensitivity to the definition is visible rather than hidden.
        power_norms: ``||A^k B||_F`` for k = 0..K -- the reason the block's singular values
            are so unevenly scaled.
    """

    K: int
    din: int
    effective_rank: float
    numerical_rank: int
    numerical_rank_relative: int
    power_norms: tuple[float, ...]
    singular_values: tuple[float, ...]
    n_nodes: int
    theoretical_columns: int

    @property
    def S1(self) -> float:
        return self.effective_rank

    def as_dict(self) -> dict[str, Any]:
        return {
            "S1": float(self.effective_rank),
            "K": int(self.K),
            "Din": int(self.din),
            "effective_rank": float(self.effective_rank),
            "numerical_rank": int(self.numerical_rank),
            "numerical_rank_relative": int(self.numerical_rank_relative),
            "power_norms": [float(v) for v in self.power_norms],
            "singular_values": [float(v) for v in self.singular_values],
            "n_nodes": int(self.n_nodes),
            "theoretical_columns": int(self.theoretical_columns),
            "krylov_capacity_used": (
                float(self.effective_rank / self.theoretical_columns)
                if self.theoretical_columns
                else float("nan")
            ),
            "S1_per_node": float(self.effective_rank / self.n_nodes) if self.n_nodes else float("nan"),
            "S1_over_min_capacity": float(
                self.effective_rank / min(self.n_nodes, self.theoretical_columns)
            )
            if self.n_nodes and self.theoretical_columns
            else float("nan"),
        }


def krylov_score(
    A: sp.spmatrix,
    B: np.ndarray,
    *,
    K: int = KRYLOV_K,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> KrylovScore:
    """S1 = effective rank of ``[B, A·B, A²·B, …, A^K·B]`` (Krylov depth ``K``).

    ``B`` must already be the candidate's input mapping built by the same declared rule for
    every candidate; this function never rescales it.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(f"B must be ({A.shape[0]}, Din); got {B.shape}")
    N, din = B.shape
    blocks = [B]
    power_norms = [float(np.linalg.norm(B, "fro"))]
    cur = B
    for _ in range(K):
        cur = A @ cur
        blocks.append(cur)
        power_norms.append(float(np.linalg.norm(cur, "fro")))
    Ck = np.concatenate(blocks, axis=1)
    s = np.linalg.svd(Ck, compute_uv=False)
    pos = s[s > tolerance]
    eff = float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0
    relative = s > (tolerance * float(s[0])) if s.size and s[0] > 0 else np.zeros_like(s, dtype=bool)
    return KrylovScore(
        K=int(K),
        din=int(din),
        effective_rank=eff,
        numerical_rank=int(pos.size),
        numerical_rank_relative=int(relative.sum()),
        power_norms=tuple(power_norms),
        singular_values=tuple(float(v) for v in s[:16]),
        n_nodes=int(N),
        theoretical_columns=int((K + 1) * din),
    )


@dataclass(frozen=True)
class EigenmodeScore:
    """S2 -- eigenmode participation of ``B``, via LEFT eigenvectors (or left Schur vectors)."""

    mode_count: int
    participation_ratio: float
    mode_energy_entropy: float
    coupling: tuple[float, ...]
    basis: str
    eigenvector_condition_number: float
    non_normality: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "S2_mode_count": int(self.mode_count),
            "S2_participation_ratio": float(self.participation_ratio),
            "S2_mode_energy_entropy": float(self.mode_energy_entropy),
            "S2_coupling_top": [float(v) for v in self.coupling[:16]],
            "S2_basis": self.basis,
            "eigenvector_condition_number": float(self.eigenvector_condition_number),
            "non_normality": float(self.non_normality),
        }


def _participation_and_entropy(p: np.ndarray) -> tuple[float, float]:
    total = float(p.sum())
    if total <= 0:
        return 0.0, 0.0
    ratio = float(total ** 2 / (p ** 2).sum()) if (p ** 2).sum() > 0 else 0.0
    q = p / total
    nz = q[q > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
    return ratio, entropy


def eigenmode_score(A: sp.spmatrix, B: np.ndarray, *, max_modes: int = 64) -> EigenmodeScore:
    """S2 -- how many of ``A``'s modes does ``B`` actually reach?

    Uses LEFT eigenvectors ``l_i`` (``l_i^T A = lambda_i l_i^T``): the direct input coupling
    to mode ``i`` is ``l_i^T B``. Falls back to **left Schur vectors** when the eigenbasis is
    ill-conditioned (condition number above ``1e8``), which is the numerically stable object
    for the same question.

    Args:
        A: The candidate matrix (sparse or dense).
        B: The candidate's input mapping, shape ``(N, Din)``.
        max_modes: How many modes to retain for the coupling vector (largest |lambda|).

    Returns:
        The :class:`EigenmodeScore`.
    """
    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    n = A_dense.shape[0]
    if n > DENSE_LIMIT:  # pragma: no cover - candidates are <= 1000 by construction
        raise ValueError(f"candidate has {n} nodes; dense eigen-decomposition is capped at {DENSE_LIMIT}")
    B = np.asarray(B, dtype=np.float64)

    non_normality = henrici_departure(A_dense)
    vals, left = sla.eig(A_dense, left=True, right=False)
    # scipy returns left eigenvectors as columns of `left` satisfying A^H v = conj(lambda) v;
    # the mode coupling is |v^H B| which is the same accessibility question for our purposes.
    cond = float(np.linalg.cond(left)) if left.size else float("inf")
    basis = "left_eigenvectors"
    vectors = left
    if not np.isfinite(cond) or cond > 1e8:
        # numerically stable fallback: left Schur vectors (orthonormal)
        T, Z = sla.schur(A_dense, output="real")
        vectors = Z  # Schur vectors are orthonormal; Z^T A Z = T
        basis = "left_schur_vectors"
        cond = float(np.linalg.cond(vectors))

    order = np.argsort(-np.abs(vals))
    keep = order[: min(max_modes, n)]
    coupling = np.abs(vectors[:, keep].conj().T @ B).sum(axis=1)
    ratio, entropy = _participation_and_entropy(coupling)
    return EigenmodeScore(
        mode_count=int(coupling.size),
        participation_ratio=ratio,
        mode_energy_entropy=entropy,
        coupling=tuple(float(v) for v in coupling),
        basis=basis,
        eigenvector_condition_number=cond,
        non_normality=non_normality,
    )


def henrici_departure(A: np.ndarray) -> float:
    """``||A Aᵀ − Aᵀ A||_F / ||A||_F²`` -- 0 for normal matrices, larger for non-normal."""
    fro2 = float((A ** 2).sum())
    if fro2 <= 0:
        return 0.0
    comm = A @ A.T - A.T @ A
    return float(np.sqrt((comm ** 2).sum()) / fro2)


def spectral_score(A: sp.spmatrix) -> dict[str, Any]:
    """S3 -- report-only spectral diagnostics. **Never read by the selection engine.**"""
    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    vals = np.linalg.eigvals(A_dense)
    moduli = np.abs(vals)
    order = np.argsort(-moduli)
    rho = float(moduli[order[0]]) if moduli.size else 0.0
    # eigenvalue-modulus entropy over the normalised modulus spectrum
    total = float(moduli.sum())
    entropy = 0.0
    if total > 0:
        q = moduli / total
        nz = q[q > 0]
        entropy = float(-(nz * np.log(nz)).sum())
    sv = np.linalg.svd(A_dense, compute_uv=False)
    spread = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
    # transient amplification proxy: the largest ||A^k||_2 over k = 1..8 relative to rho^k
    amplification = 0.0
    cur = np.eye(A_dense.shape[0])
    for k in range(1, 9):
        cur = cur @ A_dense
        norm = float(np.linalg.norm(cur, 2))
        amplification = max(amplification, norm / (rho ** k) if rho > 0 else norm)
    return {
        "S3_spectral_radius": rho,
        "S3_eigenvalue_modulus_entropy": entropy,
        "S3_singular_value_spread": spread,
        "S3_henrici_non_normality": henrici_departure(A_dense),
        "S3_transient_amplification_proxy": amplification,
        "S3_eigenvalues_top8_modulus": [float(moduli[i]) for i in order[:8]],
        "S3_note": "REPORT-ONLY: the selection engine must not read any S3 field",
    }


__all__ = [
    "DENSE_LIMIT",
    "KRYLOV_K",
    "NUMERICAL_TOLERANCE",
    "EigenmodeScore",
    "KrylovScore",
    "eigenmode_score",
    "henrici_departure",
    "krylov_score",
    "spectral_score",
]