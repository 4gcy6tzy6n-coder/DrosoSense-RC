#!/usr/bin/env python3
"""
Connectome data-root resolution (DATA-3 M2).

All large raw inputs (≈10 GB FlyWire v783 feather/npy) live OUTSIDE the Git
repository so that the repo working tree contains only code and small derived
artifacts. That keeps the tree replayable into isolated task worktrees, which
cannot carry oversized or symlinked payloads.

Resolution order (first existing candidate wins):

  1. $DROSOSENSE_DATA                    — explicit override (absolute path)
  2. <repo-parent>/data-root             — shared local data root (default layout)
  3. <repo>/connectome                   — self-contained clone fallback

Set the override when the data lives elsewhere:

  export DROSOSENSE_DATA=/Volumes/big/flywire

Layout expected under the data root:

  $DROSOSENSE_DATA/connectome/raw/<feather,npy inputs>
  $DROSOSENSE_DATA/connectome/metadata/olfactory_v1_edge_meta.csv   (large output)
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "DATA_SOURCE",
    "RAW_DIR",
    "adjacency_path",
    "require_data_file",
    "metadata_path",
    "raw_path",
]


def adjacency_path(name: str = "olfactory_v1.npz") -> Path:
    """Return the path to a connectome adjacency NPZ.

    Resolution order (first existing candidate wins):

      1. ``$DROSOSENSE_DATA/connectome/adjacency/<name>`` — the layout the
         dispatch documented (DATA-3 / DATA-4).
      2. ``$DROSOSENSE_DATA/connectome/<name>`` — the flat layout some
         server-side syncs have produced.
      3. ``<repo>/connectome/adjacency/<name>`` — in-repo fallback.
      4. ``<repo>/connectome/<name>`` — final in-repo fallback.

    Args:
        name: Filename of the adjacency artifact; defaults to the olfactory
            v1 NPZ. The function never raises if the file is absent — that
            is the caller's responsibility, and a missing file is the
            actionable signal to provision the data root.

    Returns:
        The first candidate that resolves; never raises.
    """
    candidates: list[Path] = []
    if DATA_ROOT is not None:
        candidates.append(DATA_ROOT / "connectome" / "adjacency" / name)
        candidates.append(DATA_ROOT / "connectome" / name)
    candidates.append(REPO_ROOT / "connectome" / "adjacency" / name)
    candidates.append(REPO_ROOT / "connectome" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return the most-likely path so the caller can present a useful error.
    return candidates[0]

# <repo>/connectome/paths.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_VAR = "DROSOSENSE_DATA"

_CANDIDATES = (
    (f"${_ENV_VAR}", lambda: os.environ.get(_ENV_VAR)),
    ("<repo-parent>/data-root", lambda: str(REPO_ROOT.parent / "data-root")),
    ("<repo>/connectome", lambda: str(REPO_ROOT / "connectome")),
)

DATA_ROOT: Path | None = None
DATA_SOURCE: str = ""
for _label, _resolve in _CANDIDATES:
    _raw = _resolve()
    if not _raw:
        continue
    _candidate = Path(_raw).expanduser()
    if (_candidate / "connectome" / "raw").is_dir() or (_candidate / "raw").is_dir():
        DATA_ROOT = _candidate
        DATA_SOURCE = _label
        break

# <dir holding raw/*>
RAW_DIR: Path | None = None
if DATA_ROOT is not None:
    RAW_DIR = (DATA_ROOT / "connectome" / "raw"
               if (DATA_ROOT / "connectome" / "raw").is_dir()
               else DATA_ROOT / "raw")


def raw_path(name: str) -> Path:
    """Absolute path to a raw input file; falls back to the in-repo path.

    The fallback keeps the script runnable (with a clear downstream
    FileNotFoundError) on a clone where the data root has not been provisioned.
    """
    if RAW_DIR is not None:
        return RAW_DIR / name
    return REPO_ROOT / "connectome" / "raw" / name


def metadata_path(name: str) -> Path:
    """Path to a metadata artifact.

    Small artifacts are committed and live in the repo. The oversized edge CSV
    lives in the data root when one is provisioned.
    """
    if DATA_ROOT is not None:
        candidate = DATA_ROOT / "connectome" / "metadata" / name
        if candidate.exists() or name == "olfactory_v1_edge_meta.csv":
            return candidate
    return REPO_ROOT / "connectome" / "metadata" / name


def require_data_file(path: Path) -> Path:
    """Fail loudly and actionably when a raw input is not provisioned."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw connectome input not found: {path}\n"
            f"Provision the data root (currently resolved from: "
            f"{DATA_SOURCE or 'none — no data root found'}) and re-run. "
            f"Override with: export {_ENV_VAR}=/path/to/data-root"
        )
    return path
