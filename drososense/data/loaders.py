"""Config-driven normalisation of raw dataset files into the canonical schema.

Each ``configs/datasets/*.yaml`` declares how a raw file maps onto the
canonical columns. Nothing about a particular dataset is hardcoded here, so
adding D4 means adding a config and a manifest — not editing this module.

The specimen identifier is the delicate part. When the published data contains
no specimen column, the config may declare ``specimen.source: time_block``,
which cuts the time-ordered series into contiguous blocks and treats each block
as a stand-in grouping. That is explicitly NOT protocol-compliant, and
:class:`~drososense.data.schema.SpecimenSource` records the fact so the runner
can label any resulting numbers as non-compliant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drososense.data.schema import (
    CLASS_COLUMN,
    REGRESSION_COLUMN,
    SPECIMEN_COLUMN,
    TIME_COLUMN,
    Dataset,
    DatasetSchema,
    SpecimenSource,
    load_dataset_config,
    resolve_specimen_source,
)
from drososense.utils.paths import dataset_raw_dir


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset cannot be loaded at all, with the reason attached."""


def _read_raw(config: dict[str, Any], dataset_id: str) -> pd.DataFrame:
    """Read the dataset's raw file into a frame.

    Args:
        config: Parsed dataset config.
        dataset_id: Dataset identifier, used to resolve the raw directory.

    Returns:
        The raw frame with column names stripped.

    Raises:
        DatasetUnavailableError: If the file is absent.
        ValueError: If the declared raw file is missing from the config.
    """
    raw = config.get("raw", {})
    filename = raw.get("file")
    if not filename:
        raise ValueError(f"{dataset_id}: raw.file is not declared in the config")

    base = dataset_raw_dir(dataset_id)
    direct = base / filename
    candidates = [direct, *sorted(base.rglob(filename))] if base.is_dir() else [direct]
    for candidate in candidates:
        if candidate.is_file():
            frame = pd.read_csv(candidate)
            if raw.get("strip_whitespace", True):
                frame.columns = [str(c).strip() for c in frame.columns]
            return frame

    raise DatasetUnavailableError(
        f"{dataset_id}: raw file {filename!r} not found under {base}. "
        f"Run `python scripts/download_data.py --dataset {dataset_id}` first, or follow the "
        f"manual steps in data/manifests/{dataset_id}.yaml."
    )


def _derive_specimen(
    raw_frame: pd.DataFrame,
    frame: pd.DataFrame,
    config: dict[str, Any],
    dataset_id: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Produce the canonical specimen column and a provenance note.

    Args:
        raw_frame: Frame as read from disk, for looking up a specimen column
            that need not appear in ``raw.column_map`` or ``raw.features``.
        frame: Frame already carrying canonical ``time_index``.
        config: Parsed dataset config.
        dataset_id: Dataset identifier for error messages.

    Returns:
        ``(specimen_series, provenance)``.

    Raises:
        DatasetUnavailableError: If no specimen grouping is available.
        ValueError: If the specimen config is malformed.
    """
    specimen_cfg = config.get("specimen", {})
    source = resolve_specimen_source(specimen_cfg.get("source", "column"))

    if source is SpecimenSource.PUBLISHED:
        column = specimen_cfg.get("column")
        if not column:
            raise ValueError(f"{dataset_id}: specimen.source is 'column' but no column is named")
        # The specimen column may be declared in column_map, or may simply exist
        # in the raw file without being a feature.
        if column in frame.columns:
            series = frame[column]
        elif column in raw_frame.columns:
            series = raw_frame[column]
        else:
            raise DatasetUnavailableError(
                f"{dataset_id}: declared specimen column {column!r} is not present in the raw "
                f"file (available: {sorted(raw_frame.columns)})"
            )
        return series.astype(str), {"source": source.value, "column": column}

    if source is SpecimenSource.ASSUMED_TIME_BLOCK:
        size = int(specimen_cfg.get("block_size", 0))
        if size < 1:
            raise ValueError(f"{dataset_id}: time_block specimen requires block_size >= 1")
        block_by = str(specimen_cfg.get("block_by", "rows"))
        ordered = frame.sort_values(TIME_COLUMN, kind="stable").index
        if block_by == "time":
            unique_times = frame.loc[ordered, TIME_COLUMN].drop_duplicates().to_numpy()
            ranks = pd.Series(np.arange(len(unique_times)), index=unique_times)
            rank = frame[TIME_COLUMN].map(ranks).to_numpy()
        elif block_by == "rows":
            position = pd.Series(np.arange(len(frame)), index=ordered)
            rank = frame.index.map(position).to_numpy()
        else:
            raise ValueError(f"{dataset_id}: block_by must be 'rows' or 'time', got {block_by!r}")
        block_id = rank // size
        return (
            pd.Series([f"block_{int(b):04d}" for b in block_id], index=frame.index),
            {
                "source": source.value,
                "block_by": block_by,
                "block_size": size,
                "n_blocks": int(len(np.unique(block_id))),
            },
        )

    raise DatasetUnavailableError(
        f"{dataset_id}: specimen.source is {source.value!r} — the dataset publishes no specimen "
        f"identifier and none was assumed, so specimen-level splitting cannot be applied. "
        f"See data/manifests/{dataset_id}.yaml for the acquisition steps that would unblock it."
    )


def normalise_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: DatasetSchema,
) -> Dataset:
    """Map a raw frame onto the canonical schema.

    Args:
        frame: Raw frame as read from disk.
        config: Parsed dataset config.
        schema: Schema derived from the same config.

    Returns:
        The normalised :class:`Dataset`.

    Raises:
        ValueError: If a declared column is missing or labels are unmappable.
    """
    dataset_id = schema.dataset_id
    raw = config.get("raw", {})
    column_map: dict[str, str] = dict(raw.get("column_map", {}))
    features: list[str] = list(raw.get("features", []))

    missing_sources = [
        src for src in (*column_map.values(), *features) if src not in frame.columns
    ]
    if missing_sources:
        raise ValueError(
            f"{dataset_id}: raw file is missing declared columns {sorted(set(missing_sources))}; "
            f"available: {sorted(frame.columns)}"
        )

    out = pd.DataFrame(index=frame.index)
    for canonical, source in column_map.items():
        out[canonical] = frame[source]
    for feature in features:
        out[feature] = frame[feature]

    for column in (TIME_COLUMN, CLASS_COLUMN, REGRESSION_COLUMN):
        if column not in out.columns:
            raise ValueError(f"{dataset_id}: column_map does not define {column!r}")

    label_map = raw.get("class_label_map")
    if label_map:
        normalised_labels = {str(k).strip().lower(): int(v) for k, v in label_map.items()}
        raw_labels = out[CLASS_COLUMN].astype(str).str.strip().str.lower()
        unmapped = sorted(set(raw_labels.unique()) - set(normalised_labels))
        if unmapped:
            raise ValueError(
                f"{dataset_id}: class labels {unmapped} have no entry in raw.class_label_map"
            )
        out[CLASS_COLUMN] = raw_labels.map(normalised_labels).astype(int)
    else:
        out[CLASS_COLUMN] = out[CLASS_COLUMN].astype(int)

    out[TIME_COLUMN] = pd.to_numeric(out[TIME_COLUMN], errors="raise")
    out[REGRESSION_COLUMN] = pd.to_numeric(out[REGRESSION_COLUMN], errors="raise")
    for feature in features:
        # Cast to float even when the source column is integral. The frozen
        # scaler produces float values, and an int64 column would raise on
        # assignment — as it did for D2, whose raw MQ channels are integers.
        out[feature] = pd.to_numeric(out[feature], errors="raise").astype(np.float64)

    specimens, provenance = _derive_specimen(frame, out, config, dataset_id)
    out[SPECIMEN_COLUMN] = specimens

    # Rows with non-finite channels are dropped here, once, and counted. The
    # frozen protocol rejects rather than imputes.
    feature_frame = out.loc[:, features]
    finite = np.isfinite(feature_frame.to_numpy(dtype=np.float64)).all(axis=1)
    dropped = int((~finite).sum())
    out = out.loc[finite].copy()

    label_range = set(out[CLASS_COLUMN].unique())
    if not label_range <= set(range(schema.n_classes)):
        raise ValueError(
            f"{dataset_id}: labels {sorted(label_range)} fall outside "
            f"0..{schema.n_classes - 1} declared by labels.class_names"
        )

    out = out.sort_values([SPECIMEN_COLUMN, TIME_COLUMN], kind="stable").reset_index(drop=True)
    return Dataset(
        schema=schema,
        frame=out,
        provenance={
            "dropped_non_finite_rows": dropped,
            "specimen": provenance,
            "protocol_compliant_specimen": schema.protocol_compliant,
        },
    )


def load_dataset(config_path: str | Path) -> Dataset:
    """Load and normalise a dataset described by a config file.

    Args:
        config_path: Path to ``configs/datasets/*.yaml``.

    Returns:
        The normalised :class:`Dataset`.
    """
    schema, config = load_dataset_config(config_path)
    frame = _read_raw(config, schema.dataset_id)
    return normalise_frame(frame, config, schema)


def dataset_config_path(dataset_id: str) -> Path:
    """Resolve a dataset id to its config file.

    Args:
        dataset_id: Dataset identifier.

    Returns:
        Path to ``configs/datasets/<dataset_id>.yaml``.
    """
    from drososense.utils.paths import CONFIGS_DIR

    return CONFIGS_DIR / "datasets" / f"{dataset_id}.yaml"
