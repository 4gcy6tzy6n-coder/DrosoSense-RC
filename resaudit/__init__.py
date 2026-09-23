"""ResAudit-Food -- a task-blind, construct-validity-driven reservoir audit framework.

The package contains:

* :mod:`resaudit.criteria` -- the frozen A1-A5 thresholds, transcribed from
  ``docs/resaudit_food_preregistration.md`` Section 3, and the three-valued outcome type.
* :mod:`resaudit.family` -- the sealed :class:`~resaudit.family.FamilySpec`, which
  structurally cannot carry task or food information.
* :mod:`resaudit.battery` -- the A1-A5 gate engine and the Stage 1 admissibility table.

Nothing in this package reads a food dataset, a label, a test split or a task metric.
"""

from __future__ import annotations

__all__ = ["criteria", "family", "battery"]
