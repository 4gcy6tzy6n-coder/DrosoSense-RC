"""Canonical project paths.

Every module resolves paths through this file so that the project can be moved
or mounted read-only without editing code. No module hardcodes a path.
"""

from __future__ import annotations

from pathlib import Path

# drososense/utils/paths.py -> drososense/utils -> drososense -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
DATA_MANIFESTS_DIR: Path = PROJECT_ROOT / "data" / "manifests"
RESULTS_RAW_DIR: Path = PROJECT_ROOT / "results" / "raw"
RESULTS_TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"

# v1 is kept unchanged on disk: the freeze rule is that an amendment adds a new
# version file rather than editing the old one, so that results produced under an
# earlier protocol stay attributable to the text that produced them.
PROTOCOL_V1_PATH: Path = CONFIGS_DIR / "protocol_v1.yaml"
PROTOCOL_V1_1_PATH: Path = CONFIGS_DIR / "protocol_v1.1.yaml"

# The active protocol. Everything that reads "the protocol" reads this.
PROTOCOL_PATH: Path = PROTOCOL_V1_1_PATH

# A protocol cannot contain its own hash. The digest of the frozen YAML lives in
# a sidecar file next to it, which is what makes a post-freeze edit detectable:
# `python -m drososense.utils.protocol --check` recomputes it.
PROTOCOL_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.1.sha256"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it.

    Args:
        path: Directory to create.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_raw_dir(dataset_id: str) -> Path:
    """Return the raw-data directory for one dataset, creating it if needed.

    Args:
        dataset_id: Dataset identifier, e.g. ``d1_beef_controlled``.

    Returns:
        ``data/raw/<dataset_id>``.
    """
    return ensure_dir(DATA_RAW_DIR / dataset_id)
