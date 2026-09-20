"""Config-driven normalisation of raw dataset files into the canonical schema.

Each ``configs/datasets/*.yaml`` declares how raw files map onto the canonical
columns. Nothing about a particular dataset is hardcoded here, so adding D4
means adding a config and a manifest — not editing this module.

Two groupings are derived here, and they are not the same thing:

* the **specimen** — the independent physical sample. This is the unit the
  frozen protocol splits on, and the red line the whole benchmark rests on.
* the **session** — one uninterrupted acquisition block. A window may never
  span two sessions, because a session boundary inside a specimen is a different
  measurement occasion (D3 stores the same fillet on several days, with a
  different TVC each day); a window straddling the boundary would have channels
  from one occasion and a label from another.

Where a dataset publishes no specimen column, the config may declare
``specimen.source: time_block``, which cuts the series into contiguous blocks
and treats each block as a stand-in grouping. That is explicitly NOT
protocol-compliant, and :class:`~drososense.data.schema.SpecimenSource` records
the fact so the runner can label any resulting numbers as non-compliant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drososense.data.schema import (
    CLASS_COLUMN,
    REGRESSION_COLUMN,
    SESSION_COLUMN,
    SERIES_SPECIMEN_COLUMN,
    SPECIMEN_COLUMN,
    TIME_COLUMN,
    Dataset,
    DatasetSchema,
    SpecimenSource,
    load_dataset_config,
    resolve_specimen_source,
)
from drososense.utils.paths import dataset_raw_dir

# Session policies understood by ``sessions.source``.
SESSION_PER_SPECIMEN = "specimen"
SESSION_PER_FILE = "per_file"
_SESSION_SOURCES = (SESSION_PER_SPECIMEN, SESSION_PER_FILE)

# Sentinel a dataset config may use as the ``time_index`` source when the
# provider ships no time column at all (D3's files are 120 bare sensor rows).
# The row's position inside its own source file becomes the time index.
ROW_INDEX_SENTINEL = "@row_index"


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset cannot be loaded at all, with the reason attached."""


def _read_csv(path: Path, strip_whitespace: bool) -> pd.DataFrame:
    """Read one CSV, stripping the ragged header whitespace providers ship.

    Args:
        path: CSV to read.
        strip_whitespace: Whether to strip whitespace from column names.

    Returns:
        The frame as read.
    """
    frame = pd.read_csv(path)
    if strip_whitespace:
        frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _series_entries(
    raw: dict[str, Any], dataset_id: str, base: Path
) -> list[tuple[Path, str, str, dict[str, Any]]]:
    """Resolve the dataset's raw files into per-file provenance records.

    Args:
        raw: The config's ``raw`` mapping.
        dataset_id: Dataset identifier, for error messages.
        base: The dataset's raw directory.

    Returns:
        One ``(path, specimen, session, provenance)`` tuple per file, in a
        deterministic order. ``provenance`` records any filename alias that was
        applied to reach the specimen id.

    Raises:
        DatasetUnavailableError: If no declared file exists on disk.
        ValueError: If the raw declaration is malformed or an alias is stale.
    """
    explicit = raw.get("series_files")
    if explicit:
        entries: list[tuple[Path, str, str, dict[str, Any]]] = []
        for entry in explicit:
            filename = entry.get("file")
            specimen = entry.get("specimen")
            if not filename or specimen is None:
                raise ValueError(
                    f"{dataset_id}: every raw.series_files entry needs 'file' and 'specimen'"
                )
            path = _locate(base, str(filename))
            if path is None:
                raise DatasetUnavailableError(_missing_message(dataset_id, base, str(filename)))
            entries.append(
                (path, str(specimen), str(entry.get("session", specimen)), {"alias": None})
            )
        return entries

    pattern = raw.get("glob")
    if not pattern:
        return []
    if not base.is_dir():
        raise DatasetUnavailableError(_missing_message(dataset_id, base, str(pattern)))

    specimen_regex = re.compile(str(raw.get("glob_specimen_pattern", r"^(?P<specimen>.+?)\.csv$")))
    aliases: dict[str, dict[str, Any]] = {
        str(k): dict(v) for k, v in (raw.get("specimen_aliases") or {}).items()
    }
    applied_aliases: list[str] = []
    entries = []
    seen_names: set[str] = set()
    for path in sorted(base.rglob(str(pattern))):
        seen_names.add(path.name)
        match = specimen_regex.match(path.name)
        if not match:
            raise ValueError(
                f"{dataset_id}: filename {path.name!r} does not match "
                f"raw.glob_specimen_pattern {specimen_regex.pattern!r}; refusing to guess the "
                f"specimen it belongs to"
            )
        groups = match.groupdict()
        specimen = groups.get("specimen")
        # `sessions.source: per_file` means "one session per source file", so the
        # session defaults to the file stem rather than to the specimen: a
        # specimen measured on seven days has seven sessions, and the windowing
        # boundary has to fall between them.
        session = groups.get("session") or path.stem
        if specimen is None:
            raise ValueError(
                f"{dataset_id}: raw.glob_specimen_pattern must define a 'specimen' group"
            )
        alias = aliases.get(path.name)
        applied: dict[str, Any] | None = None
        if alias is not None:
            if "specimen" not in alias or not alias.get("reason"):
                raise ValueError(
                    f"{dataset_id}: raw.specimen_aliases[{path.name!r}] needs both 'specimen' "
                    f"and a 'reason' explaining the identity claim"
                )
            applied_aliases.append(path.name)
            specimen = alias["specimen"]
            applied = {
                "file": path.name,
                "specimen": str(alias["specimen"]),
                "original_token": str(groups.get("specimen")),
                "reason": str(alias["reason"]),
            }
        entries.append((path, str(specimen), str(session), {"alias": applied}))
    if not entries:
        raise DatasetUnavailableError(_missing_message(dataset_id, base, str(pattern)))

    # An alias for a file that is not there is a stale claim about the archive;
    # failing loudly keeps the alias list honest as the dataset is re-fetched.
    unused = sorted(set(aliases) - set(applied_aliases))
    if unused:
        raise ValueError(
            f"{dataset_id}: raw.specimen_aliases names files that were not found: {unused}"
        )
    return entries


def _missing_message(dataset_id: str, base: Path, selector: str) -> str:
    """Build the standard "not on disk" message.

    Args:
        dataset_id: Dataset identifier.
        base: The dataset's raw directory.
        selector: Filename or glob that matched nothing.

    Returns:
        The message.
    """
    return (
        f"{dataset_id}: no raw file matching {selector!r} under {base}. "
        f"Run `python scripts/download_data.py --dataset {dataset_id}` first, or follow the "
        f"manual steps in data/manifests/{dataset_id}.yaml."
    )


def _locate(base: Path, filename: str) -> Path | None:
    """Find a raw file directly under ``base`` or anywhere beneath it.

    Args:
        base: The dataset's raw directory.
        filename: Basename to look for.

    Returns:
        The path found, or ``None``.
    """
    direct = base / filename
    if direct.is_file():
        return direct
    if base.is_dir():
        matches = sorted(base.rglob(filename))
        if matches:
            return matches[0]
    return None


@dataclass(frozen=True)
class RawRead:
    """The result of reading a dataset's raw files.

    Attributes:
        frame: The concatenated raw frame.
        aliases: Filename-identity claims that were applied, each with its
            reason, so the record shows that a specimen id was corrected rather
            than read straight off the archive.
        source_files: Names of the files that were read.
    """

    frame: pd.DataFrame
    aliases: list[dict[str, Any]]
    source_files: list[str]


def _read_raw(config: dict[str, Any], dataset_id: str) -> RawRead:
    """Read the dataset's raw file(s) into one frame.

    Three shapes are supported:

    * ``raw.file`` — a single CSV.
    * ``raw.series_files`` — an explicit list of CSVs, each naming the series it
      represents.
    * ``raw.glob`` — every CSV matching a pattern, with the specimen and session
      recovered from the filename by ``raw.glob_specimen_pattern``.

    Args:
        config: Parsed dataset config.
        dataset_id: Dataset identifier, used to resolve the raw directory.

    Returns:
        A :class:`RawRead` holding the frame (with column names stripped and, for
        series datasets, an extra :data:`SERIES_SPECIMEN_COLUMN` column naming
        the series each row came from) plus the per-file provenance.

    Raises:
        DatasetUnavailableError: If a declared file is absent.
        ValueError: If no raw shape is declared.
    """
    raw = config.get("raw", {})
    strip = bool(raw.get("strip_whitespace", True))
    base = dataset_raw_dir(dataset_id)
    synthesise_time = str(dict(raw.get("column_map", {})).get(TIME_COLUMN, "")) == (
        ROW_INDEX_SENTINEL
    )

    entries = _series_entries(raw, dataset_id, base)
    if entries:
        frames = []
        for path, specimen, session, provenance in entries:
            frame = _read_csv(path, strip)
            if synthesise_time:
                frame[TIME_COLUMN] = np.arange(len(frame), dtype=np.int64)
            frame[SERIES_SPECIMEN_COLUMN] = specimen
            frame[f"{SERIES_SPECIMEN_COLUMN}session"] = session
            frame[f"{SERIES_SPECIMEN_COLUMN}file"] = path.name
            frames.append(frame)
        return RawRead(
            frame=pd.concat(frames, ignore_index=True),
            aliases=[p["alias"] for _, _, _, p in entries if p.get("alias")],
            source_files=[path.name for path, _, _, _ in entries],
        )

    filename = raw.get("file")
    if not filename:
        raise ValueError(
            f"{dataset_id}: none of raw.file, raw.series_files or raw.glob is declared"
        )
    path = _locate(base, str(filename))
    if path is None:
        raise DatasetUnavailableError(_missing_message(dataset_id, base, str(filename)))
    frame = _read_csv(path, strip)
    if synthesise_time:
        frame[TIME_COLUMN] = np.arange(len(frame), dtype=np.int64)
    return RawRead(frame=frame, aliases=[], source_files=[path.name])


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

    if source is SpecimenSource.SERIES_FILE:
        if SERIES_SPECIMEN_COLUMN not in raw_frame.columns:
            raise DatasetUnavailableError(
                f"{dataset_id}: specimen.source is 'series_file' but the raw declaration did not "
                f"produce per-file series labels; declare raw.file, raw.series_files or raw.glob"
            )
        series = raw_frame[SERIES_SPECIMEN_COLUMN]
        counts = series.value_counts()
        return series.astype(str), {
            "source": source.value,
            "derivation": "one independent series per source file, named by the provider",
            "n_series": int(counts.size),
            "files_per_series": int(round(float(counts.mean()))) if counts.size else 0,
        }

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


def _derive_session(
    raw_frame: pd.DataFrame,
    specimens: pd.Series,
    config: dict[str, Any],
    dataset_id: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Produce the canonical session column and a provenance note.

    Args:
        raw_frame: Frame as read from disk, carrying the per-file series labels.
        specimens: The already-derived specimen column.
        config: Parsed dataset config.
        dataset_id: Dataset identifier for error messages.

    Returns:
        ``(session_series, provenance)``.

    Raises:
        ValueError: If the declared session source is unknown.
    """
    policy = str(config.get("sessions", {}).get("source", SESSION_PER_SPECIMEN)).strip().lower()
    if policy not in _SESSION_SOURCES:
        raise ValueError(
            f"{dataset_id}: sessions.source must be one of {list(_SESSION_SOURCES)}, got {policy!r}"
        )

    if policy == SESSION_PER_FILE:
        column = f"{SERIES_SPECIMEN_COLUMN}session"
        if column not in raw_frame.columns:
            raise ValueError(
                f"{dataset_id}: sessions.source is 'per_file' but the raw declaration produced no "
                f"file-level labels"
            )
        session = raw_frame[column].astype(str)
        note: dict[str, Any] = {
            "source": policy,
            "derivation": "one acquisition session per source file",
            "n_sessions": int(session.nunique()),
        }
    else:
        session = specimens.astype(str)
        note = {
            "source": policy,
            "derivation": "one acquisition session per specimen",
            "n_sessions": int(session.nunique()),
        }

    straddling = (
        pd.DataFrame({"specimen": specimens.astype(str), "session": session.astype(str)})
        .drop_duplicates()
        .groupby("session", observed=True)["specimen"]
        .nunique()
    )
    if (straddling > 1).any():
        offenders = sorted(straddling[straddling > 1].index)[:5]
        raise ValueError(
            f"{dataset_id}: sessions {offenders} contain rows from more than one specimen; a "
            f"session must sit inside exactly one specimen"
        )
    return session, note


def normalise_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    schema: DatasetSchema,
    raw_read: RawRead | None = None,
) -> Dataset:
    """Map a raw frame onto the canonical schema.

    Args:
        frame: Raw frame as read from disk.
        config: Parsed dataset config.
        schema: Schema derived from the same config.
        raw_read: Provenance of the read that produced ``frame``, when it came
            through :func:`_read_raw`.

    Returns:
        The normalised :class:`Dataset`.

    Raises:
        ValueError: If a declared column is missing or labels are unmappable.
    """
    dataset_id = schema.dataset_id
    raw = config.get("raw", {})
    column_map: dict[str, str] = dict(raw.get("column_map", {}))
    features: list[str] = list(raw.get("features", []))
    # Canonical feature name -> raw column name. Providers spell the same channel
    # differently ('Mq-2', 'MQ2', 'MQ2 '), and the canonical spelling is what the
    # cross-dataset channel comparison relies on, so the mapping is explicit
    # rather than normalised by guesswork. Entries absent from the map default to
    # the identity, which keeps single-spelling datasets from needing one.
    feature_map: dict[str, str] = dict(raw.get("feature_map", {}))
    feature_sources = {name: str(feature_map.get(name, name)) for name in features}

    missing_sources = [
        src
        for src in (*column_map.values(), *feature_sources.values())
        if src != ROW_INDEX_SENTINEL and src not in frame.columns
    ]
    if missing_sources:
        raise ValueError(
            f"{dataset_id}: raw file is missing declared columns {sorted(set(missing_sources))}; "
            f"available: {sorted(frame.columns)}"
        )

    out = pd.DataFrame(index=frame.index)
    for canonical, source in column_map.items():
        if source == ROW_INDEX_SENTINEL:
            if TIME_COLUMN not in frame.columns:
                raise ValueError(
                    f"{dataset_id}: column_map.time_index is {ROW_INDEX_SENTINEL!r} but the read "
                    f"did not synthesise a row index; only raw.glob/series reads can"
                )
            out[canonical] = frame[TIME_COLUMN]
        else:
            out[canonical] = frame[source]
    for feature, source in feature_sources.items():
        out[feature] = frame[source]

    for column in (TIME_COLUMN, CLASS_COLUMN, REGRESSION_COLUMN):
        if column not in out.columns:
            raise ValueError(f"{dataset_id}: column_map does not define {column!r}")

    out[TIME_COLUMN] = pd.to_numeric(out[TIME_COLUMN], errors="raise")
    out[REGRESSION_COLUMN] = pd.to_numeric(out[REGRESSION_COLUMN], errors="raise")
    for feature in features:
        # Cast to float even when the source column is integral. The frozen
        # scaler produces float values, and an int64 column would raise on
        # assignment — as it did for D2, whose raw MQ channels are integers.
        out[feature] = pd.to_numeric(out[feature], errors="raise").astype(np.float64)

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

    # An optional TVC-threshold derivation. D3 publishes only a binary
    # Fresh/Spoiled Label, so the four-level task it must share with the beef
    # datasets is derived from TVC at thresholds that protocol v1.1 froze before
    # any D3 model was run. The derivation is recorded, not implied: a derived
    # class is never presented as the provider's own label.
    class_derivation: dict[str, Any] | None = None
    derivation = raw.get("class_thresholds")
    if derivation:
        thresholds = [float(t) for t in derivation["thresholds"]]
        class_names = list(derivation.get("names", schema.class_names))
        if len(class_names) != len(thresholds) + 1:
            raise ValueError(
                f"{dataset_id}: raw.class_thresholds needs len(names) == len(thresholds) + 1"
            )
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError(f"{dataset_id}: raw.class_thresholds must be strictly increasing")
        out[CLASS_COLUMN] = np.digitize(
            out[REGRESSION_COLUMN].to_numpy(dtype=np.float64), thresholds
        ).astype(int)
        schema = DatasetSchema(
            **{
                **{f: getattr(schema, f) for f in schema.__dataclass_fields__},
                "class_names": tuple(class_names),
            }
        )
        class_derivation = {
            "source": "derived_from_tvc_thresholds",
            "thresholds": thresholds,
            "class_names": class_names,
            "provider_native_labels_used": bool(derivation.get("use_provider_labels", False)),
            "rationale": derivation.get("rationale", ""),
        }

    specimens, specimen_provenance = _derive_specimen(frame, out, config, dataset_id)
    out[SPECIMEN_COLUMN] = specimens
    sessions, session_provenance = _derive_session(frame, specimens, config, dataset_id)
    out[SESSION_COLUMN] = sessions

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

    out = out.sort_values([SPECIMEN_COLUMN, SESSION_COLUMN, TIME_COLUMN], kind="stable")
    out = out.reset_index(drop=True)
    return Dataset(
        schema=schema,
        frame=out,
        provenance={
            "dropped_non_finite_rows": dropped,
            "specimen": specimen_provenance,
            "session": session_provenance,
            "class_derivation": class_derivation,
            "filename_aliases": list(raw_read.aliases) if raw_read is not None else [],
            "source_files": list(raw_read.source_files) if raw_read is not None else [],
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
    raw_read = _read_raw(config, schema.dataset_id)
    return normalise_frame(raw_read.frame, config, schema, raw_read)


def dataset_config_path(dataset_id: str) -> Path:
    """Resolve a dataset id to its config file.

    Args:
        dataset_id: Dataset identifier.

    Returns:
        Path to ``configs/datasets/<dataset_id>.yaml``.
    """
    from drososense.utils.paths import CONFIGS_DIR

    return CONFIGS_DIR / "datasets" / f"{dataset_id}.yaml"
