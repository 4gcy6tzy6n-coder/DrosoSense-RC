"""The independent unit of the cluster-level statistics: the SPECIMEN (protocol v1.5.3).

WHY THIS MODULE EXISTS. The frozen `pairing.resample_unit_detail` justifies the
fold as the cluster on a stated premise: *"For LOSO a fold is a single specimen, so
the fold bootstrap IS a specimen bootstrap"*, which requires that the seeds share
ONE specimen partition — *"ten seeds on one specimen partition"*. The
implementation does not satisfy that premise: under a seeded LOSO permutation each
seed gets a different partition, so a given `fold_id` holds a different specimen
under each seed (measured on the delivered D3 evidence: all 62 of 62 `fold_id`s
change specimen across the 10 seeds).

So a fold-*index* cluster mean averages ten DIFFERENT specimens, and the clusters
are re-partitions of the same specimens rather than independent specimens — the
sign test then assumes an independence the design does not provide. v1.5.3
declares the cluster unit explicitly as the specimen, which is the faithful reading
of the frozen text under either split design.

Everything here is a *derivation and a check*, never a silent repair: a fold that
cannot be attributed to one specimen makes the contrast unevaluable, and a missing
per-specimen result is never filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: The frame column carrying the specimen a row was evaluated on. The committed
#: per-run tables write it; a frame without it cannot be clustered under v1.5.3.
SPECIMEN_COLUMN = "test_specimens_joined"

#: Fallback column names, in priority order.
SPECIMEN_COLUMN_CANDIDATES: tuple[str, ...] = (
    "test_specimens_joined",
    "specimen_id",
    "test_specimen",
)

#: Field separators a multi-specimen cell may use.
_SPLIT_TOKENS = (",", ";", "|")


class ClusterUnitError(ValueError):
    """Raised when a contrast cannot be clustered on the specimen."""


@dataclass(frozen=True)
class SpecimenCluster:
    """One independent unit, with the provenance behind it.

    Attributes:
        specimen_id: The cluster's identity — the specimen tested.
        source_folds: The fold indices whose observations belong to it.
        source_seeds: The seeds whose observations belong to it.
        n_rows: How many paired observations it aggregates.
        models_present: Models that produced a result on it.
        tasks_present: Tasks that produced a result on it.
    """

    specimen_id: str
    source_folds: tuple[int, ...] = ()
    source_seeds: tuple[int, ...] = ()
    n_rows: int = 0
    models_present: tuple[str, ...] = ()
    tasks_present: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "specimen_id": self.specimen_id,
            "source_folds": list(self.source_folds),
            "source_seeds": list(self.source_seeds),
            "n_rows": int(self.n_rows),
            "models_present": list(self.models_present),
            "tasks_present": list(self.tasks_present),
        }

    def __len__(self) -> int:
        return self.n_rows


def _specimen_column(frame: pd.DataFrame) -> str:
    for candidate in SPECIMEN_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ClusterUnitError(
        f"no specimen column in the frame (looked for {list(SPECIMEN_COLUMN_CANDIDATES)}); "
        f"protocol v1.5.3 clusters on the specimen and will not fall back to fold_id"
    )


def specimen_ids(frame: pd.DataFrame) -> pd.Series:
    """The specimen each row was evaluated on.

    Args:
        frame: A per-run frame carrying a test-specimen column.

    Returns:
        A string Series of specimen ids, aligned to ``frame``.

    Raises:
        ClusterUnitError: If the column is absent, empty, or a cell names more than
            one specimen — the last case is a fold that cannot be attributed to one
            specimen, and v1.5.3 refuses it rather than clustering on the fold.
    """
    column = _specimen_column(frame)
    raw = frame[column].astype(str).str.strip()
    multi = raw[raw.apply(lambda value: any(t in value for t in _SPLIT_TOKENS))]
    if not multi.empty:
        offending = sorted(set(multi.tolist()))[:3]
        raise ClusterUnitError(
            f"{len(multi)} row(s) carry more than one test specimen (e.g. {offending}); a fold "
            f"whose test set holds several specimens cannot be attributed to one specimen, so "
            f"the contrast is UNEVALUABLE under protocol v1.5.3 rather than clustered on the "
            f"fold index"
        )
    return raw


def n_clusters(frame: pd.DataFrame) -> int:
    """The number of independent units: distinct specimens."""
    return int(specimen_ids(frame).nunique())


def cluster_provenance(
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...] | None = None,
    tasks: tuple[str, ...] | None = None,
) -> list[SpecimenCluster]:
    """One :class:`SpecimenCluster` per specimen, with the provenance behind it.

    This is the machine-checkable answer to "who is the independent unit?" that
    every CI, bootstrap and sign test can be traced back to.

    Args:
        frame: A per-run frame carrying a test-specimen column.
        models: The models that produced results for every cluster. Supplied when
            the frame itself no longer carries a ``model`` column — a paired
            contrast is merged on the pair key, so both models are present by
            construction, and an empty ``models_present`` would be a field that
            says nothing.
        tasks: The task(s) the clusters were evaluated on, same convention.

    Returns:
        The clusters, sorted by specimen id.
    """
    ids = specimen_ids(frame)
    work = frame.assign(__specimen__=ids.to_numpy())

    def column(name: str, dtype: Any) -> pd.Series:
        """A column if present, else an empty Series of the right dtype."""
        if name in work.columns:
            return work[name]
        return pd.Series(dtype=dtype)

    def values(name: str, dtype: Any, cast) -> dict[str, tuple]:
        column_data = column(name, dtype)
        return {
            str(specimen): tuple(sorted({cast(v) for v in group}))
            for specimen, group in column_data.groupby(work["__specimen__"])
        }

    folds = values("fold_id", int, int)
    seeds = values("seed", int, int)
    sizes = work.groupby("__specimen__").size().to_dict()
    declared_models = (
        tuple(sorted(set(models))) if models is not None else None
    )
    declared_tasks = tuple(sorted(set(tasks))) if tasks is not None else None
    observed_models = values("model", str, str) if declared_models is None else {}
    observed_tasks = values("task", str, str) if declared_tasks is None else {}

    def per_specimen(
        declared: tuple[str, ...] | None, observed: dict[str, tuple], specimen: Any
    ) -> tuple[str, ...]:
        if declared is not None:
            return declared
        return observed.get(str(specimen), ())

    return [
        SpecimenCluster(
            specimen_id=str(specimen),
            source_folds=folds.get(str(specimen), ()),
            source_seeds=seeds.get(str(specimen), ()),
            n_rows=int(sizes[specimen]),
            models_present=per_specimen(declared_models, observed_models, specimen),
            tasks_present=per_specimen(declared_tasks, observed_tasks, specimen),
        )
        for specimen in sorted(sizes)
    ]


def expected_cluster_counts(declaration: Mapping[str, Any] | None = None) -> dict[str, int]:
    """The counts v1.5.3 states in advance, as a check on the implementation."""
    if declaration is None:
        declaration = load_cluster_unit_declaration()
    block = declaration.get("cluster_unit") or {}
    return {str(k): int(v) for k, v in (block.get("expected_cluster_counts") or {}).items()}


def check_expected_cluster_counts(
    counts: Mapping[str, int], declaration: Mapping[str, Any] | None = None
) -> list[str]:
    """Compare observed cluster counts against the declared expectations.

    A mismatch is a defect in the implementation, not a result: the counts are a
    property of the datasets.

    Returns:
        A list of human-readable problems; empty when every declared dataset agrees.
    """
    expected = expected_cluster_counts(declaration)
    problems: list[str] = []
    for dataset, want in expected.items():
        got = counts.get(dataset)
        if got is None:
            continue
        if int(got) != want:
            problems.append(
                f"{dataset}: {got} clusters against the {want} protocol v1.5.3 declares"
            )
    return problems


def load_cluster_unit_declaration(path: str | Path | None = None) -> dict[str, Any]:
    """Read the v1.5.3 ``cluster_unit`` block.

    Raises:
        FileNotFoundError: If the amendment is absent — a cluster unit that cannot
            be read is not one that can be defaulted to the fold index.
    """
    import yaml

    from drososense.utils.paths import PROTOCOL_V1_5_3_PATH

    target = Path(path) if path is not None else PROTOCOL_V1_5_3_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"cluster-unit declaration not found at {target}; protocol v1.5.3 must be "
            f"present — the cluster unit is never defaulted"
        )
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if "cluster_unit" not in payload:
        raise ValueError(f"{target} carries no cluster_unit block")
    return payload

