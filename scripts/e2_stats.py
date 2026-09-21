#!/usr/bin/env python
"""Pre-registered E2 statistics for the R0–R6 reservoir family (DATA-5, M4).

Reads the per-run CSVs written by ``scripts/run_reservoir.py`` and computes,
for every contrast the frozen protocol declares, the pre-registered paired
test (cluster sign test, decisive; paired Wilcoxon, descriptive) plus the
bootstrap CI, effect size and minimum achievable p — then applies the
protocol's Holm families. Nothing here evaluates a model: it is pure
post-hoc analysis of already-scored records, so it is safe to run at any
time, against synthetic fixture records as well as approved real ones.

Examples
--------
::

    python scripts/e2_stats.py --records results/raw/m4_reservoir/batch1_per_run.csv
    python scripts/e2_stats.py --records a_per_run.csv b_per_run.csv --dataset d2_beef_uncontrolled --task classification --print
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation import e2_stats  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point; see :func:`drososense.evaluation.e2_stats.main`."""
    return e2_stats.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
