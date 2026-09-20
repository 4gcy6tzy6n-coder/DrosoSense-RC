"""Shared test fixtures.

Tests never touch ``data/raw`` or the real configuration: they build a small
synthetic fixture and a temporary dataset config in ``tmp_path``, so the suite
runs offline and leaves no artifacts behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.data.loaders import normalise_frame  # noqa: E402
from drososense.data.schema import Dataset, schema_from_config  # noqa: E402
from drososense.data.synthetic import FixtureSpec, generate_fixture  # noqa: E402

FIXTURE_SPEC = FixtureSpec(
    n_specimens=6,
    n_timesteps=60,
    n_channels=3,
    seed=7,
    noise_sd=0.3,
    specimen_offset_sd=6.0,
)

FIXTURE_CONFIG: dict = {
    "dataset_id": "unit_fixture",
    "display_name": "Unit fixture",
    "raw": {
        "file": "unit_fixture.csv",
        "strip_whitespace": True,
        "column_map": {
            "time_index": "time_index",
            "freshness_class": "freshness_class",
            "tvc": "tvc",
        },
        "features": ["s1", "s2", "s3", "temperature", "humidity"],
        "class_label_map": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "labels": {"class_names": ["Excellent", "Good", "Acceptable", "Spoiled"]},
    "specimen": {"source": "column", "column": "specimen_id"},
    "features": {"columns": ["s1", "s2", "s3", "temperature", "humidity"]},
    "source": {"url": "generated", "citation": "n/a"},
    "license": "n/a",
}


@pytest.fixture(scope="session")
def fixture_frame() -> pd.DataFrame:
    """A small synthetic frame with real specimen structure.

    Returns:
        The generated fixture frame.
    """
    return generate_fixture(FIXTURE_SPEC)


@pytest.fixture()
def fixture_dataset(fixture_frame: pd.DataFrame) -> Dataset:
    """The fixture normalised into the canonical schema.

    Args:
        fixture_frame: Session-scoped generated frame.

    Returns:
        The normalised :class:`Dataset`.
    """
    schema = schema_from_config(FIXTURE_CONFIG)
    return normalise_frame(fixture_frame.copy(), FIXTURE_CONFIG, schema)


@pytest.fixture()
def fixture_csv(tmp_path: Path, fixture_frame: pd.DataFrame) -> Path:
    """Write the fixture to a CSV inside ``tmp_path``.

    Args:
        tmp_path: pytest temporary directory.
        fixture_frame: Session-scoped generated frame.

    Returns:
        Path of the written CSV.
    """
    path = tmp_path / "unit_fixture.csv"
    fixture_frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def protocol() -> dict:
    """Load the frozen protocol.

    Returns:
        The parsed ``configs/protocol_v1.yaml``.
    """
    from drososense.utils.config import load_protocol

    return load_protocol()


@pytest.fixture()
def write_config(tmp_path: Path):
    """Return a helper that writes a dataset config YAML to disk.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Callable ``(config: dict, name: str) -> Path``.
    """

    def _write(config: dict, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    return _write
