"""The sealed family object: what the battery is allowed to see.

Task-blindness is enforced here STRUCTURALLY, not by convention. A
:class:`FamilySpec` cannot carry a food label, a macro-F1, an MAE, a test split, or any
task metric, because it has no field for one and sets ``__slots__`` so none can be
attached later. The battery's entry point takes ``FamilySpec`` values and nothing else,
so there is no channel through which a task result could reach a qualification verdict.

The same object serves both roles the pre-registration needs:

* it declares what a family IS (``family_id``, ``role``, construction, seed), and
* it carries the only two objects the battery measures (the adjacency ``A`` and the
  input mapping ``B``).

``counterfactual_parent`` is the field that makes A2 answerable without collapsing
``not_applicable`` into ``FAIL``: a family that names no parent has no wiring
counterfactual, so A2's fairness conditions are not defined for it and A2 returns
``NOT_APPLICABLE`` rather than a failure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp


class TaskBlindnessError(TypeError):
    """Raised when a value carries or names a task/food metric."""

    #: Substrings that may not appear in a construction parameter KEY. Checked
    #: case-insensitively. This is a guard against a construction description becoming a
    #: smuggling channel for a task result, not a filter on scientific prose.
    FORBIDDEN_KEY_PARTS: tuple[str, ...] = (
        "macro_f1",
        "macrof1",
        "f1",
        "mae",
        "mean_absolute_error",
        "rmse",
        "mse",
        "accuracy",
        "label",
        "labels",
        "test_split",
        "test_data",
        "food",
        "beef",
        "trout",
        "tvc",
        "target",
        "y_true",
        "y_pred",
    )

    @classmethod
    def check_construction(cls, construction: Mapping[str, Any]) -> None:
        """Reject a construction record that names a task or food quantity."""
        for key in construction:
            low = str(key).lower()
            for part in cls.FORBIDDEN_KEY_PARTS:
                if part in low:
                    raise TaskBlindnessError(
                        f"construction key {key!r} names a task/food quantity "
                        f"(matched {part!r}); the construction record must be task-blind"
                    )


@dataclass(frozen=True)
class FamilySpec:
    """A reservoir family, sealed against task information.

    Attributes:
        family_id: Stable identifier, e.g. ``"F1"``. Purely a label; no verdict may
            depend on it (see ``tests/test_resaudit_invariants.py``).
        label: Human-readable name, for reports only.
        role: The pre-registration's declared role, e.g. ``"negative_biological_control"``.
        A: Adjacency, shape ``(N, N)``, directed, sparse or dense.
        B: Input mapping, shape ``(N, Din)``. Built by the single declared rule.
        n_nodes: ``N``.
        n_edges: Edge count (nonzero entries of ``A``).
        din: Input dimension ``Din``.
        counterfactual_parent: ``family_id`` of the wiring parent whose degree sequence,
            weight multiset and per-source weight multisets this family must preserve,
            or ``None`` when the family has no wiring counterfactual. ``None`` makes A2
            ``NOT_APPLICABLE``.
        seed: Construction seed, recorded for reproducibility.
        construction: Task-blind description of how the family was built. Keys are
            screened by :class:`TaskBlindnessError`.
    """

    __slots__ = (
        "family_id",
        "label",
        "role",
        "A",
        "B",
        "n_nodes",
        "n_edges",
        "din",
        "counterfactual_parent",
        "seed",
        "construction",
    )

    family_id: str
    label: str
    role: str
    A: sp.spmatrix
    B: np.ndarray
    n_nodes: int
    n_edges: int
    din: int
    counterfactual_parent: str | None
    seed: int
    construction: Mapping[str, Any]

    def __post_init__(self) -> None:
        A = sp.csr_matrix(self.A) if not sp.issparse(self.A) else self.A.tocsr()
        B = np.asarray(self.B, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError(f"{self.family_id}: B must be 2-D (N, Din); got shape {B.shape}")
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"{self.family_id}: A must be square; got {A.shape}")
        if B.shape[0] != A.shape[0]:
            raise ValueError(
                f"{self.family_id}: B rows ({B.shape[0]}) must equal A nodes ({A.shape[0]})"
            )
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "construction", dict(self.construction))
        TaskBlindnessError.check_construction(self.construction)
        if int(self.n_nodes) != A.shape[0]:
            raise ValueError(f"{self.family_id}: n_nodes={self.n_nodes} != A.shape[0]={A.shape[0]}")
        if int(self.din) != B.shape[1]:
            raise ValueError(f"{self.family_id}: din={self.din} != B.shape[1]={B.shape[1]}")
        if int(self.n_edges) != int(A.nnz):
            raise ValueError(
                f"{self.family_id}: n_edges={self.n_edges} != A.nnz={int(A.nnz)}"
            )
        if self.counterfactual_parent == self.family_id:
            raise ValueError(f"{self.family_id}: a family cannot be its own counterfactual parent")

    # -- identity -----------------------------------------------------------

    def wiring_hash(self) -> str:
        """Content hash of ``(A, B)`` -- what the battery actually measures."""
        h = hashlib.sha256()
        a = self.A.tocoo()
        h.update(np.asarray([self.n_nodes, self.n_edges, self.din], dtype=np.int64).tobytes())
        h.update(a.row.astype(np.int64).tobytes())
        h.update(a.col.astype(np.int64).tobytes())
        h.update(np.ascontiguousarray(a.data, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.B, dtype=np.float64).tobytes())
        return h.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Report header. Deliberately excludes the matrices."""
        return {
            "family_id": self.family_id,
            "label": self.label,
            "role": self.role,
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "din": int(self.din),
            "counterfactual_parent": self.counterfactual_parent,
            "a2_applicable": self.counterfactual_parent is not None,
            "seed": int(self.seed),
            "wiring_hash": self.wiring_hash(),
            "construction": _jsonable(self.construction),
        }


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-ification that never silently drops information."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def wiring_hash_of(A: sp.spmatrix, B: np.ndarray) -> str:
    """Standalone content hash, for callers that hold matrices but no spec."""
    A = A.tocsr()
    B = np.asarray(B, dtype=np.float64)
    h = hashlib.sha256()
    a = A.tocoo()
    h.update(np.asarray([A.shape[0], int(A.nnz), B.shape[1]], dtype=np.int64).tobytes())
    h.update(a.row.astype(np.int64).tobytes())
    h.update(a.col.astype(np.int64).tobytes())
    h.update(np.ascontiguousarray(a.data, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(B, dtype=np.float64).tobytes())
    return h.hexdigest()


def dump_specs(specs: list[FamilySpec], path: str) -> None:
    """Write the report headers of several families as one JSON array."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([s.as_dict() for s in specs], fh, indent=2, sort_keys=True)


__all__ = [
    "TaskBlindnessError",
    "FamilySpec",
    "wiring_hash_of",
    "dump_specs",
]
