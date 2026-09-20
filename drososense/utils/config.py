"""Configuration loading and fingerprinting.

Every run record stores the hash of the config it was produced from, so a
result can be traced back to the exact protocol and hyperparameters that
generated it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from drososense.utils.paths import PROTOCOL_PATH

REQUIRED_PROTOCOL_KEYS: tuple[str, ...] = (
    "protocol_version",
    "frozen",
    "split_unit",
    "seeds",
    "tasks",
    "preprocessing",
    "primary_hypothesis",
    "secondary_hypotheses",
    "statistical_tests",
    "datasets",
    "gates",
    "narrative_adjustment_rules",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed mapping. An empty file yields ``{}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping at the top level of {path}")
    return data


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate ``configs/protocol_v1.yaml``.

    Args:
        path: Override for the protocol location (used by tests).

    Returns:
        The protocol mapping.

    Raises:
        ValueError: If a required top-level key is missing.
    """
    protocol = load_yaml(path or PROTOCOL_PATH)
    missing = [key for key in REQUIRED_PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"protocol is missing required keys: {sorted(missing)}")
    return protocol


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable short fingerprint of a config mapping.

    Key order does not affect the digest, so re-serialising a config does not
    invalidate previously recorded runs.

    Args:
        config: Any JSON-serialisable mapping.

    Returns:
        First 12 hex characters of the SHA-256 of the canonical JSON form.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
