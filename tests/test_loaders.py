"""Loader behaviour: normalisation, specimen derivation, and failure modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drososense.data.loaders import DatasetUnavailableError, normalise_frame
from drososense.data.schema import (
    CLASS_COLUMN,
    SPECIMEN_COLUMN,
    TIME_COLUMN,
    Dataset,
    schema_from_config,
)

BASE_CONFIG: dict = {
    "dataset_id": "toy",
    "display_name": "Toy",
    "raw": {
        "file": "toy.csv",
        "strip_whitespace": True,
        "column_map": {"time_index": "t", "freshness_class": "cls", "tvc": "tvc"},
        "features": ["a", "b"],
        "class_label_map": {"fresh": 0, "off": 1},
    },
    "labels": {"class_names": ["Fresh", "Off"]},
    "specimen": {"source": "column", "column": "sample"},
    "features": {"columns": ["a", "b"]},
    "source": {"url": "generated", "citation": "n/a"},
    "license": "n/a",
}


def _raw_frame() -> pd.DataFrame:
    """A raw frame with INTEGRAL sensor channels, as the real D2 CSV has."""
    return pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5, 6],
            "sample": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "cls": ["fresh", "fresh", "off", "fresh", "off", "off"],
            "tvc": [2.0, 2.5, 3.5, 2.1, 3.9, 4.2],
            "a": [100, 120, 300, 110, 290, 310],
            "b": [7, 8, 9, 7, 9, 10],
        }
    )


@pytest.mark.unit
def test_integral_channels_are_cast_to_float(tmp_path):
    """Integer sensor columns become float.

    The frozen scaler emits floats; leaving a column as int64 makes the
    standardisation assignment raise. This regressed once on D2, whose raw MQ
    channels are all integers, while D1's are floats and hid the bug.
    """
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    for feature in schema.feature_columns:
        assert dataset.frame[feature].dtype == np.float64, feature


@pytest.mark.unit
def test_standardising_an_integral_dataset_does_not_raise(tmp_path):
    """End-to-end: an integer-typed dataset can be scaled without error."""
    from drososense.data.scaling import fit_standardizer

    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    features = list(schema.feature_columns)
    scaler = fit_standardizer(dataset.frame[features].to_numpy())
    scaled = dataset.frame.copy()
    scaled.loc[:, features] = scaler.transform(dataset.frame[features].to_numpy())
    assert np.isfinite(scaled[features].to_numpy()).all()


@pytest.mark.unit
def test_class_labels_are_mapped_to_declared_order():
    """Text labels map through the config's class_label_map."""
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    assert set(dataset.frame[CLASS_COLUMN]) == {0, 1}
    by_time = dataset.frame.set_index(TIME_COLUMN)[CLASS_COLUMN]
    assert by_time.loc[1] == 0  # "fresh"
    assert by_time.loc[3] == 1  # "off"


@pytest.mark.unit
def test_canonical_columns_replace_the_raw_names():
    """Downstream code only ever sees canonical column names."""
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    for column in (TIME_COLUMN, SPECIMEN_COLUMN, CLASS_COLUMN, "tvc", "a", "b"):
        assert column in dataset.frame.columns
    for raw_name in ("t", "cls", "sample"):
        assert raw_name not in dataset.frame.columns


@pytest.mark.unit
def test_unknown_class_label_is_rejected():
    """A label with no mapping fails instead of becoming NaN or an index."""
    config = {**BASE_CONFIG}
    frame = _raw_frame()
    frame.loc[0, "cls"] = "mystery"
    with pytest.raises(ValueError, match="no entry in raw.class_label_map"):
        normalise_frame(frame, config, schema_from_config(config))


@pytest.mark.unit
def test_missing_declared_column_is_rejected():
    """A config promising a column the file lacks fails with both lists."""
    config = {**BASE_CONFIG}
    frame = _raw_frame().drop(columns=["b"])
    with pytest.raises(ValueError, match="missing declared columns"):
        normalise_frame(frame, config, schema_from_config(config))


@pytest.mark.unit
def test_non_finite_rows_are_dropped_and_counted():
    """Rows with a non-finite channel are dropped, not imputed."""
    config = {**BASE_CONFIG}
    frame = _raw_frame()
    frame.loc[2, "a"] = np.nan
    dataset = normalise_frame(frame, config, schema_from_config(config))
    assert len(dataset.frame) == 5
    assert dataset.provenance["dropped_non_finite_rows"] == 1


# ---------------------------------------------------------------------------
# Specimen derivation
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_published_specimen_column_passes_through():
    """A dataset that ships a specimen id keeps it verbatim."""
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    assert dataset.specimens() == ("s1", "s2")
    assert schema.protocol_compliant is True
    assert dataset.provenance["specimen"]["source"] == "published"


@pytest.mark.unit
def test_time_block_specimens_are_assumed_and_marked_non_compliant():
    """Without a published id, blocks are a labelled assumption."""
    config = {
        **BASE_CONFIG,
        "specimen": {"source": "time_block", "block_by": "rows", "block_size": 3},
    }
    dataset = normalise_frame(_raw_frame(), config, schema_from_config(config))
    assert dataset.specimens() == ("block_0000", "block_0001")
    assert dataset.schema.protocol_compliant is False
    assert dataset.provenance["protocol_compliant_specimen"] is False


@pytest.mark.unit
def test_time_block_by_time_uses_distinct_time_values():
    """Blocking by time counts distinct timestamps, not rows."""
    frame = _raw_frame()
    # Three rows share each timestamp, so row-based and time-based blocking differ.
    frame["t"] = [1, 1, 2, 2, 3, 3]
    config = {
        **BASE_CONFIG,
        "specimen": {"source": "time_block", "block_by": "time", "block_size": 2},
    }
    dataset = normalise_frame(frame, config, schema_from_config(config))
    assert dataset.specimens() == ("block_0000", "block_0001")


@pytest.mark.unit
def test_unavailable_specimen_source_blocks_loading():
    """A dataset with no defensible grouping refuses to load."""
    config = {**BASE_CONFIG, "specimen": {"source": "none"}}
    with pytest.raises(DatasetUnavailableError, match="cannot be applied"):
        normalise_frame(_raw_frame(), config, schema_from_config(config))


@pytest.mark.unit
def test_declared_but_absent_specimen_column_is_reported():
    """A config naming a specimen column the file lacks fails clearly."""
    config = {**BASE_CONFIG, "specimen": {"source": "column", "column": "beef_id"}}
    with pytest.raises(DatasetUnavailableError, match="beef_id"):
        normalise_frame(_raw_frame(), config, schema_from_config(config))


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dataset_rejects_a_frame_missing_canonical_columns():
    """The container validates its own invariant."""
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    with pytest.raises(ValueError, match="frame missing"):
        Dataset(schema=schema, frame=dataset.frame.drop(columns=[CLASS_COLUMN]))


@pytest.mark.unit
def test_label_summary_describes_the_data():
    """The summary reports counts a reader can check against the manifest."""
    schema = schema_from_config(BASE_CONFIG)
    dataset = normalise_frame(_raw_frame(), BASE_CONFIG, schema)
    summary = dataset.label_summary()
    assert summary["n_rows"] == 6
    assert summary["n_specimens"] == 2
    assert summary["class_counts"] == {"0": 3, "1": 3}
    assert summary["rows_per_specimen"]["min"] == 3
