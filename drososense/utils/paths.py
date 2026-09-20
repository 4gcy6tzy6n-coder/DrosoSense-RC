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

PROTOCOL_PATH: Path = CONFIGS_DIR / "protocol_v1.yaml"


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
