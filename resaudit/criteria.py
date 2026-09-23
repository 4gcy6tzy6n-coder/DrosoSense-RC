"""ResAudit-Food -- the frozen audit battery, as data rather than as prose.

Every threshold in this module is transcribed from
``docs/resaudit_food_preregistration.md`` Section 3, which was written BEFORE any
reservoir in this project was built or scored. The values are declared here as module
constants so that no criterion can be moved after seeing a task result, and so that
``tests/test_resaudit_criteria.py`` can assert the transcription is exact.

The three-valued outcome is deliberate. ``NOT_APPLICABLE`` is a distinct state from
``FAIL`` and from ``PASS``: a criterion whose precondition is absent has not been
evaluated, and folding it into ``FAIL`` would penalise a family for a property the
battery never asked of it. This matters concretely for A2, whose fairness conditions
(degree sequence, weight multiset and per-source weight multisets preserved exactly)
are only meaningful for a counterfactual that shares a wiring parent with the
candidate. Families without such a parent are ``NOT_APPLICABLE``, not failed.

A ``NOT_APPLICABLE`` criterion never blocks qualification. Only ``FAIL`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Frozen thresholds. Source: docs/resaudit_food_preregistration.md, Section 3.
# ---------------------------------------------------------------------------

#: A1 -- structure present. Source: DrosoSense-RC C2.3-C2.5.
A1_SCC_FRACTION_MIN = 0.90
A1_ISOLATED_FRACTION_MAX = 0.02
A1_MEAN_OUT_DEGREE_MIN = 2.0

#: A2 -- counterfactual fair. Source: DrosoSense-RC C4, amendment 1.
#: The exactness requirements are boolean (degree sequence, global weight multiset,
#: per-source weight multisets: preserved EXACTLY), so only the overlap has a number.
A2_OVERLAP_MAX = 0.20

#: A3 -- Krylov controllability, the PRIMARY qualification gate.
#: ``D_eff([B, A*B, ..., A^16*B]) >= A3_FACTOR * Din``. Source: DrosoSense-RC v3.
A3_FACTOR = 2.0
A3_KRYLOV_K = 16

#: A4 -- recurrent memory. ``(M - M_{A:=0}) / M >= A4_MIN``. Source: DrosoSense-RC C3.2.
A4_MIN = 0.20

#: A5 -- state expansion. ``D_eff(state) >= A5_FACTOR * Din``. Source: DrosoSense-RC C3.3.
A5_FACTOR = 1.5

#: Identity of the primary qualification gate. A family that fails this is not run on
#: food data, which is the framework's entire point and what makes the food stage
#: affordable (pre-registration Sections 3 and 6.1).
PRIMARY_GATE = "A3"

#: A3's NARROWED name. The criterion was introduced as "Krylov controllability" and read
#: as a proxy for a substrate being a useful reservoir. Calibration falsified that
#: reading: a pure feed-forward nilpotent chain PASSES A3, so A3 cannot be evidence of
#: recurrence. What it measures is how many independent state directions the input can
#: reach within a finite horizon of ``K`` steps. The id and threshold are unchanged; only
#: the claimed meaning is narrowed. See
#: ``docs/resaudit_food_preregistration_amendment_1.md`` Section 2.
A3_NAME = "finite-horizon input-reachable state diversity"

#: The caveat every report must carry, so a downstream reader cannot read an A3 PASS as a
#: statement about dynamics.
A3_CAVEAT = (
    "A3 PASS is neither sufficient evidence of recurrence nor of useful memory. "
    "A pure feed-forward chain passes A3, so A3 measures finite-horizon "
    "input-reachable state diversity, not recurrence quality."
)

#: A1 and A2 are admission requirements.
ADMISSION_CRITERIA: tuple[str, ...] = ("A1", "A2")

#: DOWNSTREAM GATES -- criteria that can disqualify a family. A5 only.
#:
#: A4 was one of these and has been REMOVED, by the decision rule the pre-registration
#: wrote for exactly this situation (Section 6.2). ``docs/resaudit_a4_construct_validity.md``
#: measures the A4 observable on a pre-declared recurrence-strength ladder over the
#: calibration families and finds it is NOT directional: the recurrent memory falls as
#: recurrence strengthens, is O(0.01) and does not separate the extreme rungs, and is
#: negative at the operating point on 5 of 6 applicable families. A criterion that does
#: not move in the direction its own definition requires cannot gate anything, so it is
#: reported descriptively. The THRESHOLD is not moved: no value of it would fix a
#: non-directional observable, and moving it is the tuning the pre-registration forbids.
DOWNSTREAM_CRITERIA: tuple[str, ...] = ("A5",)

#: Criteria that can BLOCK qualification. A4 is deliberately absent.
BLOCKING_CRITERIA: tuple[str, ...] = ("A1", "A2", "A3", "A5")

#: Retired as a gate; still measured and reported.
DESCRIPTIVE_ONLY_CRITERIA: tuple[str, ...] = ("A4",)

ALL_CRITERIA: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5")

#: Why A4 is descriptive only, carried into every report that shows an A4 number.
A4_DESCRIPTIVE_CAVEAT = (
    "A4 is DESCRIPTIVE ONLY, not a qualification gate. Its construct-validity audit "
    "(docs/resaudit_a4_construct_validity.md) found it is not directional in recurrence "
    "strength: the recurrent memory decreases as recurrence increases, is O(0.01) and "
    "does not separate rho=0 from rho=0.95, and is negative at the operating point on 5 "
    "of 6 ladder-applicable calibration families. Retired by pre-registration decision "
    "rule 6.2. The threshold was NOT moved."
)


class CriterionState(str, Enum):
    """Three-valued outcome. ``NOT_APPLICABLE`` is not a soft failure."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "not_applicable"

    @property
    def blocks_qualification(self) -> bool:
        """Only an evaluated FAIL blocks. N/A and PASS do not."""
        return self is CriterionState.FAIL


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's outcome for one family, with an auditable trace.

    Attributes:
        criterion: ``"A1"``..``"A5"``.
        state: the three-valued outcome.
        applicable: whether the criterion's precondition held. ``False`` iff
            ``state is NOT_APPLICABLE``.
        value: the measured quantity the threshold is compared against, or ``None``
            when the criterion could not be evaluated.
        threshold: the frozen threshold, echoed so a report is self-contained.
        expression: the comparison in words, e.g. ``"D_eff >= 2.0 * Din"``.
        reason: for ``NOT_APPLICABLE``, WHY the precondition was absent; for other
            states a short note. Never empty for ``NOT_APPLICABLE``.
        detail: supporting numbers (diagnostics), never used for the verdict.
    """

    criterion: str
    state: CriterionState
    applicable: bool
    value: float | None
    threshold: float | None
    expression: str
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.criterion not in ALL_CRITERIA:
            raise ValueError(f"unknown criterion {self.criterion!r}; expected one of {ALL_CRITERIA}")
        if self.applicable != (self.state is not CriterionState.NOT_APPLICABLE):
            raise ValueError(
                f"{self.criterion}: applicable={self.applicable} contradicts "
                f"state={self.state.value}"
            )
        if self.state is CriterionState.NOT_APPLICABLE and not str(self.reason).strip():
            raise ValueError(
                f"{self.criterion}: a NOT_APPLICABLE result must state WHY it is not applicable"
            )

    @property
    def passed(self) -> bool:
        return self.state is CriterionState.PASS

    @property
    def failed(self) -> bool:
        return self.state is CriterionState.FAIL

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "state": self.state.value,
            "applicable": bool(self.applicable),
            "value": None if self.value is None else float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "expression": self.expression,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def not_applicable(
    criterion: str,
    *,
    reason: str,
    expression: str = "",
    detail: Mapping[str, Any] | None = None,
) -> CriterionResult:
    """Build a ``NOT_APPLICABLE`` result. A reason is mandatory."""
    if not str(reason).strip():
        raise ValueError(f"{criterion}: NOT_APPLICABLE requires a reason")
    return CriterionResult(
        criterion=criterion,
        state=CriterionState.NOT_APPLICABLE,
        applicable=False,
        value=None,
        threshold=None,
        expression=expression,
        reason=reason,
        detail=dict(detail or {}),
    )


def evaluate_ge(
    criterion: str,
    value: float,
    threshold: float,
    *,
    expression: str,
    detail: Mapping[str, Any] | None = None,
) -> CriterionResult:
    """A ``value >= threshold`` criterion. "Close" is not a pass."""
    state = CriterionState.PASS if float(value) >= float(threshold) else CriterionState.FAIL
    return CriterionResult(
        criterion=criterion,
        state=state,
        applicable=True,
        value=float(value),
        threshold=float(threshold),
        expression=expression,
        reason="",
        detail=dict(detail or {}),
    )


def evaluate_le(
    criterion: str,
    value: float,
    threshold: float,
    *,
    expression: str,
    detail: Mapping[str, Any] | None = None,
) -> CriterionResult:
    """A ``value <= threshold`` criterion."""
    state = CriterionState.PASS if float(value) <= float(threshold) else CriterionState.FAIL
    return CriterionResult(
        criterion=criterion,
        state=state,
        applicable=True,
        value=float(value),
        threshold=float(threshold),
        expression=expression,
        reason="",
        detail=dict(detail or {}),
    )


def evaluate_bool(
    criterion: str,
    ok: bool,
    *,
    expression: str,
    value: float | None = None,
    threshold: float | None = None,
    detail: Mapping[str, Any] | None = None,
) -> CriterionResult:
    """An exactness criterion with no numeric threshold."""
    return CriterionResult(
        criterion=criterion,
        state=CriterionState.PASS if ok else CriterionState.FAIL,
        applicable=True,
        value=value,
        threshold=threshold,
        expression=expression,
        reason="",
        detail=dict(detail or {}),
    )


__all__ = [
    "A1_SCC_FRACTION_MIN",
    "A1_ISOLATED_FRACTION_MAX",
    "A1_MEAN_OUT_DEGREE_MIN",
    "A2_OVERLAP_MAX",
    "A3_FACTOR",
    "A3_KRYLOV_K",
    "A4_MIN",
    "A5_FACTOR",
    "PRIMARY_GATE",
    "A3_NAME",
    "A3_CAVEAT",
    "ADMISSION_CRITERIA",
    "DOWNSTREAM_CRITERIA",
    "BLOCKING_CRITERIA",
    "DESCRIPTIVE_ONLY_CRITERIA",
    "A4_DESCRIPTIVE_CAVEAT",
    "ALL_CRITERIA",
    "CriterionState",
    "CriterionResult",
    "not_applicable",
    "evaluate_ge",
    "evaluate_le",
    "evaluate_bool",
]
