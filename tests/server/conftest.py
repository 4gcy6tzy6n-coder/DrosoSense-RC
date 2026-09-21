"""Shared pytest fixtures for the DATA-28 server-harness tests.

The fixtures exist for one reason: the harness scripts bootstrap the project
root onto ``sys.path`` themselves, so they can be run as ``python
scripts/server/...`` from anywhere. The tests, however, run under pytest, which
imports modules by package path. The fixtures here make that work and also
provide a temp dir for CSV writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _ensure_project_root_on_path() -> None:
    """Make sure the project root is importable as ``scripts`` and ``drososense``."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the resolved project root."""
    return PROJECT_ROOT


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """A scratch directory for CSVs written by the harness."""
    return tmp_path / "results"