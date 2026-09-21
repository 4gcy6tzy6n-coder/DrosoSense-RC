"""Synthetic e-nose fixture with genuine specimen structure.

The fixture exists for ONE purpose: to exercise the full pipeline — split,
scaler, window, model, metric, result record — end to end, deterministically
and without network access. It is a smoke-test fixture.

It is registered as a dataset with ``synthetic: true`` in its manifest, and
every result produced from it is written under ``experiment: smoke`` with
``evidence_class: synthetic_fixture``. It must never appear in a results table
alongside real datasets, and no conclusion may rest on it.

The generator deliberately gives each specimen its own sensor offset and its
own spoilage trajectory. That is what makes it useful as a leakage canary: a
row-level split lets a model memorise per-specimen offsets and scores far above
what a specimen-level split permits. ``tests/test_leakage.py`` uses the fixture
to demonstrate exactly that gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drososense.data.schema import CLASS_COLUMN, REGRESSION_COLUMN, SPECIMEN_COLUMN, TIME_COLUMN

# Microbiological spoilage thresholds on log10(TVC), matching the thresholds
# the published beef datasets use (3.0 / 4.0 / 5.0 log10 CFU/g).
TVC_CLASS_THRESHOLDS: tuple[float, ...] = (3.0, 4.0, 5.0)
FIXTURE_DATASET_ID = "synthetic_enose"
FIXTURE_FILENAME = "synthetic_enose.csv"


@dataclass(frozen=True)
class FixtureSpec:
    """Parameters of the synthetic fixture.

    Attributes:
        n_specimens: Number of independent specimens.
        n_timesteps: Timesteps recorded per specimen.
        n_channels: Number of gas-sensor channels (temperature and humidity are
            added on top of these).
        seed: Master seed; the generator is fully deterministic given this.
        noise_sd: Standard deviation of per-channel measurement noise.
        specimen_offset_sd: Standard deviation of per-specimen sensor offsets.
            Larger values make a row-level split leak more badly.
    """

    n_specimens: int = 12
    n_timesteps: int = 120
    n_channels: int = 6
    seed: int = 20260920
    noise_sd: float = 0.35
    specimen_offset_sd: float = 6.0

    def __post_init__(self) -> None:
        if self.n_specimens < 2:
            raise ValueError("n_specimens must be >= 2")
        if self.n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1")
        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")


def generate_fixture(spec: FixtureSpec | None = None) -> pd.DataFrame:
    """Generate the synthetic fixture as a canonical tidy frame.

    Args:
        spec: Fixture parameters; defaults to :class:`FixtureSpec`.

    Returns:
        Frame with canonical columns: ``specimen_id``, ``time_index``, sensor
        channels ``s1..sN``, ``temperature``, ``humidity``, ``freshness_class``
        and ``tvc``.
    """
    spec = spec or FixtureSpec()
    rng = np.random.default_rng(spec.seed)
    channel_names = [f"s{i + 1}" for i in range(spec.n_channels)]

    records: list[dict[str, float | str | int]] = []
    for specimen_index in range(spec.n_specimens):
        specimen_id = f"sp{specimen_index:03d}"

        # Per-specimen spoilage trajectory: a logistic curve in log10(TVC).
        tvc_start = rng.uniform(2.2, 2.9)
        tvc_end = rng.uniform(6.0, 8.5)
        midpoint = rng.uniform(0.35, 0.75) * spec.n_timesteps
        steepness = rng.uniform(0.06, 0.16)

        # Per-specimen sensor baseline offsets — the leakage trap.
        offsets = rng.normal(0.0, spec.specimen_offset_sd, size=spec.n_channels)
        gains = rng.uniform(3.0, 9.0, size=spec.n_channels)
        base_levels = rng.uniform(40.0, 220.0, size=spec.n_channels)

        ambient_temp = rng.uniform(3.0, 5.0)
        ambient_humidity = rng.uniform(70.0, 90.0)

        for t in range(spec.n_timesteps):
            progress = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
            tvc = tvc_start + (tvc_end - tvc_start) * progress
            # Sensor response saturates with spoilage and is specimen-specific.
            response = np.log1p(np.maximum(tvc - 2.0, 0.0))
            sensors = (
                base_levels
                + offsets
                + gains * response
                + rng.normal(0.0, spec.noise_sd, size=spec.n_channels)
            )
            record: dict[str, float | str | int] = {
                SPECIMEN_COLUMN: specimen_id,
                TIME_COLUMN: t,
                REGRESSION_COLUMN: float(tvc),
                "temperature": float(ambient_temp + rng.normal(0.0, 0.15)),
                "humidity": float(ambient_humidity + rng.normal(0.0, 0.6)),
            }
            for name, value in zip(channel_names, sensors):
                record[name] = float(value)
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    labels = np.digitize(frame[REGRESSION_COLUMN].to_numpy(), TVC_CLASS_THRESHOLDS)
    frame[CLASS_COLUMN] = labels.astype(int)
    frame = frame.sort_values([SPECIMEN_COLUMN, TIME_COLUMN], kind="stable").reset_index(drop=True)

    ordered = [SPECIMEN_COLUMN, TIME_COLUMN, *channel_names, "temperature", "humidity",
               CLASS_COLUMN, REGRESSION_COLUMN]
    return frame.loc[:, ordered]


def feature_columns(spec: FixtureSpec | None = None) -> list[str]:
    """Return the fixture's sensor channel names in schema order.

    Args:
        spec: Fixture parameters; must match the one used to generate.

    Returns:
        Channel names followed by ``temperature`` and ``humidity``.
    """
    spec = spec or FixtureSpec()
    return [f"s{i + 1}" for i in range(spec.n_channels)] + ["temperature", "humidity"]


def write_fixture(destination: str | Path, spec: FixtureSpec | None = None) -> Path:
    """Generate the fixture and write it to ``destination``.

    Args:
        destination: Path of the CSV to write; parents are created.
        spec: Fixture parameters.

    Returns:
        The path written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_fixture(spec).to_csv(destination, index=False)
    return destination
