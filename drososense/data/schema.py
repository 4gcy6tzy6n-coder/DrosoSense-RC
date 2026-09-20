"""Canonical dataset schema.

Raw CSVs differ wildly in column naming and in whether a specimen identifier
exists at all. Loaders normalise every dataset into this schema before any
split, scaler or window is constructed, so downstream code never branches on
dataset identity.

The specimen identifier is the load-bearing field: ``SpecimenSource`` records
where it came from and whether that satisfies the frozen protocol. A dataset
whose specimen identifier is *assumed* rather than *published* is marked
non-compliant and may not be used for a protocol result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from drososense.utils.config import load_yaml

# Canonical column names every loaded dataset is normalised to.
SPECIMEN_COLUMN = "specimen_id"
TIME_COLUMN = "time_index"
CLASS_COLUMN = "freshness_class"
REGRESSION_COLUMN = "tvc"


class SpecimenSource(str, Enum):
    """Where the specimen identifier came from.

    Attributes:
        PUBLISHED: The dataset ships a specimen/sample identifier column.
        ASSUMED_TIME_BLOCK: No identifier exists; contiguous time blocks are
            used as a stand-in grouping. This does NOT satisfy the frozen
            protocol and any result built on it is non-compliant.
        UNAVAILABLE: No defensible grouping exists; the dataset cannot be used
            with specimen-level splitting at all.
    """

    PUBLISHED = "published"
    ASSUMED_TIME_BLOCK = "assumed_time_block"
    UNAVAILABLE = "unavailable"

    @property
    def protocol_compliant(self) -> bool:
        """Whether this source satisfies ``split_unit: specimen``."""
        return self is SpecimenSource.PUBLISHED


# Config files use short, readable spellings; the enum values are the canonical
# ones. Both are accepted so that a config reads naturally without every author
# having to know the enum.
_SPECIMEN_SOURCE_ALIASES: dict[str, SpecimenSource] = {
    "column": SpecimenSource.PUBLISHED,
    "published": SpecimenSource.PUBLISHED,
    "time_block": SpecimenSource.ASSUMED_TIME_BLOCK,
    "assumed_time_block": SpecimenSource.ASSUMED_TIME_BLOCK,
    "none": SpecimenSource.UNAVAILABLE,
    "unavailable": SpecimenSource.UNAVAILABLE,
}


def resolve_specimen_source(value: object) -> SpecimenSource:
    """Map a config spelling onto a :class:`SpecimenSource`.

    Args:
        value: Raw config value, e.g. ``"column"``.

    Returns:
        The corresponding enum member.

    Raises:
        ValueError: If the spelling is not recognised.
    """
    key = str(value).strip().lower()
    if key not in _SPECIMEN_SOURCE_ALIASES:
        raise ValueError(
            f"specimen.source must be one of {sorted(_SPECIMEN_SOURCE_ALIASES)}, got {value!r}"
        )
    return _SPECIMEN_SOURCE_ALIASES[key]


@dataclass(frozen=True)
class DatasetSchema:
    """Declarative description of one normalised dataset.

    Attributes:
        dataset_id: Stable identifier, e.g. ``d1_beef_controlled``.
        display_name: Human-readable name for tables and figures.
        feature_columns: Canonical sensor/climate channels, in fixed order.
        class_names: Ordered freshness class labels, index == class id.
        specimen_source: Provenance of the specimen identifier.
        specimen_column_raw: Raw column used as specimen id, if any.
        specimen_note: Free-text justification, surfaced in reports.
        class_derivation: How the class label was produced (e.g. TVC thresholds).
        license: License string as published by the data provider.
        source_url: Canonical landing page.
        citation: Required citation text.
    """

    dataset_id: str
    display_name: str
    feature_columns: tuple[str, ...]
    class_names: tuple[str, ...]
    specimen_source: SpecimenSource = SpecimenSource.PUBLISHED
    specimen_column_raw: str | None = None
    specimen_note: str = ""
    class_derivation: str = ""
    license: str = ""
    source_url: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError(f"{self.dataset_id}: feature_columns must not be empty")
        if not self.class_names:
            raise ValueError(f"{self.dataset_id}: class_names must not be empty")
        duplicates = {c for c in self.feature_columns if self.feature_columns.count(c) > 1}
        if duplicates:
            raise ValueError(f"{self.dataset_id}: duplicate feature columns {sorted(duplicates)}")

    @property
    def n_features(self) -> int:
        """Number of input channels."""
        return len(self.feature_columns)

    @property
    def n_classes(self) -> int:
        """Number of freshness classes."""
        return len(self.class_names)

    @property
    def protocol_compliant(self) -> bool:
        """Whether this dataset may be used for a protocol-compliant result."""
        return self.specimen_source.protocol_compliant


@dataclass(frozen=True)
class Dataset:
    """A normalised dataset: a tidy frame plus its schema.

    Attributes:
        schema: The :class:`DatasetSchema` describing the columns.
        frame: Tidy frame with canonical columns only.
        provenance: Loading diagnostics — how the specimen column was derived
            and how many rows were dropped for non-finite channels.
    """

    schema: DatasetSchema
    frame: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {SPECIMEN_COLUMN, TIME_COLUMN, CLASS_COLUMN, REGRESSION_COLUMN}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.schema.dataset_id}: frame missing {sorted(missing)}")
        absent = [c for c in self.schema.feature_columns if c not in self.frame.columns]
        if absent:
            raise ValueError(f"{self.schema.dataset_id}: frame missing features {absent}")

    @property
    def dataset_id(self) -> str:
        """Stable dataset identifier."""
        return self.schema.dataset_id

    def specimens(self) -> tuple[str, ...]:
        """Return every specimen identifier, sorted for determinism.

        Returns:
            Tuple of unique specimen ids as strings.
        """
        return tuple(sorted(self.frame[SPECIMEN_COLUMN].astype(str).unique()))

    def features(self) -> pd.DataFrame:
        """Return only the feature channels, in schema order.

        Returns:
            A frame with ``schema.feature_columns`` in order.
        """
        return self.frame.loc[:, list(self.schema.feature_columns)]

    def label_summary(self) -> dict[str, Any]:
        """Summarise label distribution and per-specimen counts.

        Returns:
            Mapping with ``n_rows``, ``n_specimens``, ``class_counts`` and
            ``rows_per_specimen`` statistics.
        """
        per_specimen = self.frame.groupby(SPECIMEN_COLUMN, observed=True).size()
        return {
            "n_rows": int(len(self.frame)),
            "n_specimens": int(per_specimen.size),
            "class_counts": {
                str(k): int(v)
                for k, v in self.frame[CLASS_COLUMN].value_counts().sort_index().items()
            },
            "rows_per_specimen": {
                "min": int(per_specimen.min()),
                "max": int(per_specimen.max()),
                "median": float(per_specimen.median()),
            },
        }


def schema_from_config(config: dict[str, Any]) -> DatasetSchema:
    """Build a :class:`DatasetSchema` from a ``configs/datasets/*.yaml`` mapping.

    Args:
        config: Parsed dataset config.

    Returns:
        The corresponding schema.

    Raises:
        ValueError: If a required field is absent or malformed.
    """
    try:
        dataset_id = config["dataset_id"]
        display_name = config["display_name"]
        features = tuple(config["features"]["columns"])
        class_names = tuple(config["labels"]["class_names"])
    except KeyError as exc:
        raise ValueError(f"dataset config missing required field: {exc}") from exc

    specimen = config.get("specimen", {})
    try:
        specimen_source = resolve_specimen_source(specimen.get("source", "column"))
    except ValueError as exc:
        raise ValueError(f"{dataset_id}: {exc}") from exc

    return DatasetSchema(
        dataset_id=dataset_id,
        display_name=display_name,
        feature_columns=features,
        class_names=class_names,
        specimen_source=specimen_source,
        specimen_column_raw=specimen.get("column"),
        specimen_note=specimen.get("note", ""),
        class_derivation=config.get("labels", {}).get("derivation", ""),
        license=config.get("license", ""),
        source_url=config.get("source", {}).get("url", ""),
        citation=config.get("source", {}).get("citation", ""),
    )


def load_dataset_config(path: str | Path) -> tuple[DatasetSchema, dict[str, Any]]:
    """Load a dataset config, returning both the schema and the raw mapping.

    Args:
        path: Path to a ``configs/datasets/*.yaml`` file.

    Returns:
        ``(schema, raw_config)``.
    """
    raw = load_yaml(path)
    return schema_from_config(raw), raw
