"""The dataset layer, tested against the real files when they are present.

These are the tests that would have caught the R0 audit's X1 and X2 before the
data ever reached a model: that the two beef datasets are bound to the right
DOIs, that D2's five cuts group into five specimens, that D3's fillet token is a
repeated-measures identifier rather than a per-file label, and that nothing reads
a column by position — which is what makes D2's reversed last two columns
harmless instead of silently fatal.

They skip when the raw data is absent, because the repository does not commit it;
``scripts/download_data.py`` fetches it and the checksums in the manifests pin it.
"""

from __future__ import annotations

import pytest

from drososense.data.loaders import dataset_config_path, load_dataset, normalise_frame
from drososense.data.schema import CLASS_COLUMN, SESSION_COLUMN, SPECIMEN_COLUMN, load_dataset_config
from drososense.utils.paths import DATA_RAW_DIR

REAL_DATASETS = ("d1_beef_controlled", "d2_beef_uncontrolled", "d3_rainbow_trout")


def _require(dataset_id: str):
    """Load a real dataset, skipping the test when its raw files are absent.

    Args:
        dataset_id: Dataset identifier.

    Returns:
        The loaded :class:`~drososense.data.schema.Dataset`.
    """
    base = DATA_RAW_DIR / dataset_id
    if not base.is_dir() or not any(base.glob("*")):
        pytest.skip(f"{dataset_id}: raw data not present; run scripts/download_data.py")
    return load_dataset(dataset_config_path(dataset_id))


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_d1_is_the_controlled_classification_dataset():
    """D1 is n8mc3nspfn: 20,815 rows, provider integer classes, no specimen id."""
    dataset = _require("d1_beef_controlled")
    assert dataset.frame.shape[0] == 20815
    assert dataset.schema.n_features == 10
    assert not dataset.schema.protocol_compliant
    assert dataset.provenance["specimen"]["source"] == "assumed_time_block"
    assert dataset.provenance["specimen"]["n_blocks"] == 12


@pytest.mark.integration
def test_d2_is_the_uncontrolled_five_cut_dataset():
    """D2 is mwmhh766fc v3: five cuts, five specimens, LOSO(5) is exhaustive."""
    dataset = _require("d2_beef_uncontrolled")
    assert dataset.frame.shape[0] == 10800
    assert dataset.specimens() == ("TS1", "TS2", "TS3", "TS4", "TS5")
    assert dataset.sessions() == dataset.specimens()
    assert dataset.schema.protocol_compliant
    assert dataset.provenance["specimen"]["source"] == "series_file"
    counts = dataset.frame.groupby(SPECIMEN_COLUMN, observed=True).size().unique()
    assert list(counts) == [2160]


@pytest.mark.integration
def test_d2_column_order_difference_does_not_affect_the_loaded_values():
    """TS1 swaps its last two columns; reading by name makes that invisible.

    The check is direct: the loaded temperature and humidity series must be
    plausible for their names on every cut. Under a positional read, TS1's
    temperature would be TS2's humidity — roughly 60 rather than roughly 30.
    """
    dataset = _require("d2_beef_uncontrolled")
    for specimen in dataset.specimens():
        rows = dataset.frame[dataset.frame[SPECIMEN_COLUMN] == specimen]
        assert 5.0 < rows["temperature"].mean() < 45.0, f"{specimen}: temperature looks wrong"
        assert 5.0 < rows["humidity"].mean() < 100.0, f"{specimen}: humidity looks wrong"


@pytest.mark.integration
def test_d3_has_62_specimens_and_210_sessions():
    """X2/X4: the fillet token is a repeated-measures id, not a per-file label."""
    dataset = _require("d3_rainbow_trout")
    assert dataset.frame.shape[0] == 25080
    assert len(dataset.specimens()) == 62
    assert len(dataset.sessions()) == 210
    assert dataset.schema.protocol_compliant
    # A fillet measured on several days is one specimen with several sessions.
    per_specimen = dataset.frame.groupby(SPECIMEN_COLUMN, observed=True)[SESSION_COLUMN].nunique()
    assert per_specimen.max() > 1
    assert per_specimen.sum() == 210


@pytest.mark.integration
def test_d3_four_level_label_is_derived_and_declared():
    """The provider ships Fresh/Spoiled; the four levels come from frozen thresholds."""
    dataset = _require("d3_rainbow_trout")
    derivation = dataset.provenance["class_derivation"]
    assert derivation["source"] == "derived_from_tvc_thresholds"
    assert derivation["thresholds"] == [4.3, 5.0, 6.5]
    assert derivation["provider_native_labels_used"] is False
    counts = dataset.frame[CLASS_COLUMN].value_counts().sort_index().tolist()
    assert counts == [3479, 3601, 10800, 7200]
    assert all(count > 0 for count in counts), "all four classes must be non-empty"


@pytest.mark.integration
def test_d3_filename_aliases_are_recorded_in_provenance():
    """A corrected specimen id must be visible in the record, not silent."""
    dataset = _require("d3_rainbow_trout")
    aliases = dataset.provenance["filename_aliases"]
    assert {a["file"] for a in aliases} == {"202600515F1F1.csv", "20260513F4F1M.csv"}
    assert {a["specimen"] for a in aliases} == {"F1F1", "F4F1"}
    assert all(a["reason"] for a in aliases)


@pytest.mark.integration
@pytest.mark.parametrize("dataset_id", REAL_DATASETS)
def test_no_dataset_has_missing_values_after_loading(dataset_id):
    """The protocol rejects rather than imputes, and the count is recorded."""
    dataset = _require(dataset_id)
    numeric = dataset.frame.select_dtypes("number")
    assert not numeric.isna().any().any()
    assert "dropped_non_finite_rows" in dataset.provenance


@pytest.mark.integration
@pytest.mark.parametrize("dataset_id", REAL_DATASETS)
def test_sessions_never_straddle_specimens(dataset_id):
    """A session belongs to exactly one specimen — the windowing boundary."""
    dataset = _require(dataset_id)
    per_session = dataset.frame.groupby(SESSION_COLUMN, observed=True)[SPECIMEN_COLUMN].nunique()
    assert (per_session == 1).all()


# ---------------------------------------------------------------------------
# Config-level rules that hold without any data
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_series_file_dataset_without_a_glob_is_rejected():
    """series_file needs a raw shape that produces per-file series labels."""
    import pandas as pd

    from drososense.data.loaders import DatasetUnavailableError
    from drososense.data.schema import schema_from_config

    config = {
        "dataset_id": "broken",
        "display_name": "Broken",
        "raw": {
            "file": "x.csv",
            "column_map": {"time_index": "t", "freshness_class": "c", "tvc": "v"},
            "features": ["s1"],
            "class_label_map": {"0": 0},
        },
        "labels": {"class_names": ["A", "B"]},
        "specimen": {"source": "series_file"},
        "features": {"columns": ["s1"]},
    }
    schema = schema_from_config(config)
    frame = pd.DataFrame({"t": [1, 2], "c": ["0", "0"], "v": [1.0, 2.0], "s1": [0.1, 0.2]})
    with pytest.raises(DatasetUnavailableError, match="did not produce per-file series labels"):
        normalise_frame(frame, config, schema)


@pytest.mark.unit
def test_class_thresholds_override_a_partial_label_map():
    """D3 maps two provider labels but derives four classes from TVC."""
    import pandas as pd

    from drososense.data.schema import schema_from_config

    config = {
        "dataset_id": "derived",
        "display_name": "Derived",
        "raw": {
            "file": "x.csv",
            "column_map": {"time_index": "t", "freshness_class": "label", "tvc": "tvc"},
            "features": ["s1"],
            "class_label_map": {"fresh": 0, "spoiled": 3},
            "class_thresholds": {
                "thresholds": [1.0, 2.0, 3.0],
                "names": ["A", "B", "C", "D"],
                "use_provider_labels": False,
                "rationale": "test",
            },
        },
        "labels": {"class_names": ["A", "B", "C", "D"]},
        "specimen": {"source": "time_block", "block_by": "rows", "block_size": 1},
        "features": {"columns": ["s1"]},
    }
    schema = schema_from_config(config)
    frame = pd.DataFrame(
        {
            "t": [1, 2, 3, 4],
            "label": ["fresh", "fresh", "spoiled", "spoiled"],
            "tvc": [0.5, 1.5, 2.5, 3.5],
            "s1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    dataset = normalise_frame(frame, config, schema)
    assert dataset.frame[CLASS_COLUMN].tolist() == [0, 1, 2, 3]
    assert dataset.provenance["class_derivation"]["provider_native_labels_used"] is False
    assert dataset.schema.class_names == ("A", "B", "C", "D")


@pytest.mark.unit
def test_a_filename_alias_for_an_absent_file_is_rejected(tmp_path):
    """An alias naming a file that is not in the archive is a rotten claim.

    The check is the point: without it, an alias could survive a re-download and
    keep correcting an id that no longer exists, or stop applying silently.
    """
    from drososense.data.loaders import _series_entries

    (tmp_path / "20260509F1F1.csv").write_text("a\n1\n", encoding="utf-8")
    raw = {
        "glob": "*.csv",
        "glob_specimen_pattern": r"^(?P<date>\d+)(?P<specimen>F\d+F\d+M?)\.csv$",
        "specimen_aliases": {
            "202600515F1F1.csv": {"specimen": "F1F1", "reason": "not present on disk"}
        },
    }
    with pytest.raises(ValueError, match="names files that were not found"):
        _series_entries(raw, "stale_alias", tmp_path)


@pytest.mark.unit
def test_a_filename_alias_must_carry_a_reason(tmp_path):
    """A specimen id may be corrected, but only with the evidence attached."""
    from drososense.data.loaders import _series_entries

    (tmp_path / "202600515F1F1.csv").write_text("a\n1\n", encoding="utf-8")
    raw = {
        "glob": "*.csv",
        "glob_specimen_pattern": r"^(?P<date>\d+)(?P<specimen>F\d+F\d+M?)\.csv$",
        "specimen_aliases": {"202600515F1F1.csv": {"specimen": "F1F1"}},
    }
    with pytest.raises(ValueError, match="needs both 'specimen' and a 'reason'"):
        _series_entries(raw, "unjustified", tmp_path)


@pytest.mark.unit
def test_a_filename_alias_is_applied_and_reported(tmp_path):
    """The applied alias travels back with the entry, so provenance can record it."""
    from drososense.data.loaders import _series_entries

    (tmp_path / "202600515F1F1.csv").write_text("a\n1\n", encoding="utf-8")
    raw = {
        "glob": "*.csv",
        "glob_specimen_pattern": r"^(?P<date>\d+)(?P<specimen>F\d+F\d+M?)\.csv$",
        "specimen_aliases": {
            "202600515F1F1.csv": {"specimen": "F1F1", "reason": "date typo; TVC confirms day 7"}
        },
    }
    entries = _series_entries(raw, "aliased", tmp_path)
    assert len(entries) == 1
    _, specimen, _, provenance = entries[0]
    assert specimen == "F1F1"
    assert provenance["alias"]["original_token"] == "F1F1"
    assert "day 7" in provenance["alias"]["reason"]
