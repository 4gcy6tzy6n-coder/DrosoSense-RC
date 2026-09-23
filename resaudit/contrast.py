"""The family contrast matrix: a DESIGN GATE, not documentation.

The user's instruction for this stage:

    For each family, record exactly one primary varied structural factor, all held-fixed
    quantities, its contrast parent, the identifiable claim, and forbidden stronger
    claims. ... No family instances or food evaluation until the contrast matrix itself
    passes review.

So this module is executable: a :class:`ContrastDeclaration` cannot be constructed at all
unless it names a single varied factor, at least one held-fixed quantity, an identifiable
claim, and the stronger claims that claim does NOT license. :func:`validate_battery` then
checks the matrix as a whole, which is where the interesting failures live -- a family whose
varied factor duplicates another's, a family with no contrast parent, or a family that
still has an un-named varied factor.

Three rules from the instruction are encoded here rather than left to prose:

1. **One primary varied factor per family.** ``varied_factors`` is a 1-tuple, enforced.
2. **F2 and F3 must be non-redundant controls.** They may not share a varied factor, and
   their held-fixed sets must differ -- F2 retains first-order statistics that F3 discards,
   which is what makes them a gradient rather than two random baselines.
3. **A family whose varied factor is un-named cannot be instantiated.** F4 is in exactly
   that state: its ``varied_factor`` is ``None`` and ``blocked_reason`` says why. The
   validator reports it as blocked rather than letting it silently pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ContrastDeclaration",
    "MatrixValidationError",
    "validate_battery",
    "CONTRAST_MATRIX",
    "blocked_families",
    "ready_families",
    "contrast_matrix_table",
    "F1_REFERENCE_CLAIM_PREFIX",
]

#: Every recognizable identifiability claim about a family-vs-reference contrast must be
#: phrased as an association under matched conditions, never as a cause.
F1_REFERENCE_CLAIM_PREFIX = "Under matched "

#: The STRUCTURAL-STATISTIC vocabulary. "Held fixed" splits into two kinds of quantity, and
#: only these are the ones the F1 -> F2 -> F3 gradient must shrink: measurement conventions
#: (input geometry, spectral radius, leak/gain, probe protocol, readout) are held fixed by
#: every family and shrinking them would be a protocol change, not a contrast.
STRUCTURAL_STATISTICS: frozenset[str] = frozenset({
    "edge_count",
    "density",
    "mean_degree",
    "degree_sequence",
    "weight_multiset",
    "scc_profile",
    "cycle_length_profile",
    "recurrence_statistics",
    "cell_type_assignment",
})


class MatrixValidationError(ValueError):
    """Raised when the contrast matrix does not satisfy the design gate."""


@dataclass(frozen=True)
class ContrastDeclaration:
    """One family's entry in the contrast matrix.

    Attributes:
        family_id: ``"F1"``..``"F5"``.
        role: The role the family plays, e.g. ``"biological_reference"``.
        varied_factor: The SINGLE primary structural factor this family changes relative
            to its contrast parent, or ``None`` when the factor has not been named yet.
            A ``None`` here makes the family un-instantiable and puts it on the blocked
            list -- which is the state F4 is in.
        held_fixed: Quantities held fixed relative to the contrast parent. Must be
            non-empty: a contrast that holds nothing fixed identifies nothing.
        contrast_parent: ``family_id`` this family is compared against, or ``None`` for
            the reference itself.
        identifiable_claim: The strongest statement a comparison against the parent
            licenses. Must begin with :data:`F1_REFERENCE_CLAIM_PREFIX`.
        forbidden_claims: Stronger statements the contrast does NOT license. Must be
            non-empty -- this is the field that stops a result being over-read.
        blocked_reason: Why the family cannot be instantiated yet. Non-empty iff
            ``varied_factor is None``.
        notes: Free-form design notes.
    """

    family_id: str
    role: str
    varied_factor: str | None
    held_fixed: tuple[str, ...]
    contrast_parent: str | None
    identifiable_claim: str
    forbidden_claims: tuple[str, ...]
    blocked_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.varied_factor is not None:
            if not str(self.varied_factor).strip():
                raise MatrixValidationError(
                    f"{self.family_id}: varied_factor is empty; use None to mean 'not yet named'"
                )
        if not self.held_fixed:
            raise MatrixValidationError(
                f"{self.family_id}: held_fixed is empty. A contrast that holds nothing "
                "fixed identifies nothing."
            )
        if not self.forbidden_claims:
            raise MatrixValidationError(
                f"{self.family_id}: forbidden_claims is empty. Every contrast must state "
                "the stronger claim it does NOT license."
            )
        if not self.identifiable_claim.startswith(F1_REFERENCE_CLAIM_PREFIX):
            raise MatrixValidationError(
                f"{self.family_id}: identifiable_claim must be phrased as an association "
                f"under matched conditions, i.e. start with {F1_REFERENCE_CLAIM_PREFIX!r}. "
                f"Got: {self.identifiable_claim!r}"
            )
        if self.contrast_parent == self.family_id:
            raise MatrixValidationError(f"{self.family_id}: cannot be its own contrast parent")
        if self.varied_factor is None and not self.blocked_reason:
            raise MatrixValidationError(
                f"{self.family_id}: an un-named varied_factor requires a blocked_reason "
                "explaining what must be decided before it can be built"
            )
        if self.varied_factor is not None and self.blocked_reason:
            raise MatrixValidationError(
                f"{self.family_id}: blocked_reason is set but varied_factor is named "
                f"({self.varied_factor!r}); a named family is buildable"
            )

    @property
    def blocked(self) -> bool:
        return self.varied_factor is None

    @property
    def held_fixed_set(self) -> frozenset[str]:
        return frozenset(self.held_fixed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "role": self.role,
            "varied_factor": self.varied_factor,
            "held_fixed": list(self.held_fixed),
            "contrast_parent": self.contrast_parent,
            "identifiable_claim": self.identifiable_claim,
            "forbidden_claims": list(self.forbidden_claims),
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
# Held-fixed vocabulary, shared verbatim across families so that "held fixed" means the
# same thing everywhere and the F2/F3 gradient is checkable:
#   n_nodes, edge_count, degree_sequence, density, weight_multiset,
#   input_geometry, spectral_radius, leak_gain, cell_type_assignment, scc_profile,
#   cycle_length_profile, recurrence_statistics, readout, probe_protocol

_CLAIM_F2 = (
    "Under matched N, edge count, degree sequence, weight multiset, input geometry and "
    "spectral radius, any difference from F1 is associated with the specific wiring "
    "arrangement rather than with first-order structure."
)
_CLAIM_F3 = (
    "Under matched N, density, weight multiset, input geometry and spectral radius, any "
    "difference from F1 is associated with higher-order structure, since first-order "
    "degree statistics are deliberately NOT retained."
)
_CLAIM_F5 = (
    "Under matched N, edge budget, input geometry, weight multiset, probe protocol and "
    "spectral radius, F5 is an A3-qualified construction within the same budget, so it "
    "shows what an A3-passing substrate looks like at this scale."
)

_FORBIDDEN_TOPOLOGY_CAUSAL = (
    "biological topology causes better performance",
    "biology is computationally superior",
    "the connectome is the better reservoir",
)
_FORBIDDEN_DENSITY_CAUSAL = (
    "density causes the difference",
    "more edges cause better performance",
)
_FORBIDDEN_A3_CAUSAL = (
    "A3 causes better food performance",
    "A3 PASS predicts food accuracy",
    "guaranteed reachable diversity improves the task",
)
_FORBIDDEN_REFERENCE = (
    "F1 is a baseline to be beaten",
    "F1's value is a performance benchmark",
)


CONTRAST_MATRIX: tuple[ContrastDeclaration, ...] = (
    ContrastDeclaration(
        family_id="F1",
        role="biological_reference",
        # The reference varies nothing; it IS the thing others are contrasted against.
        # Recorded as an explicit sentinel rather than None, because None means "not yet
        # named" and F1's factor is named: it is 'nothing, by construction'.
        varied_factor="none (reference; varies nothing by construction)",
        held_fixed=("n_nodes", "edge_count", "mean_degree", "weight_multiset",
                    "input_geometry", "spectral_radius", "leak_gain", "probe_protocol"),
        contrast_parent=None,
        identifiable_claim=(
            "Under matched measurement conventions, F1 supplies the reference substrate "
            "against which F2-F5 are contrasted."
        ),
        forbidden_claims=_FORBIDDEN_REFERENCE,
        notes=(
            "The frozen project's S0 substrate, reused unchanged. Held-fixed values are "
            "the quantities the OTHER families must match to F1, and are listed here so "
            "there is one canonical list rather than five drifting ones."
        ),
    ),
    ContrastDeclaration(
        family_id="F2",
        role="statistics_matched_random_control",
        varied_factor="wiring arrangement, with first-order statistics retained exactly",
        held_fixed=("n_nodes", "edge_count", "mean_degree", "degree_sequence",
                    "weight_multiset", "input_geometry", "spectral_radius", "leak_gain",
                    "probe_protocol"),
        contrast_parent="F1",
        identifiable_claim=_CLAIM_F2,
        forbidden_claims=_FORBIDDEN_TOPOLOGY_CAUSAL,
        notes=(
            "Constructible with the frozen weight_preserving_degree_rewire, which "
            "preserves the exact degree sequence and per-source weight multisets while "
            "exchanging edges. Note this is the SAME primitive A2 audits, so F2's A2 is "
            "applicable: F2 declares a counterfactual_parent and its replacement must pass "
            "the overlap ceiling. The knob is the number of accepted swaps, which trades "
            "wiring randomization against the A2 overlap ceiling -- that trade must be "
            "declared before generating, not chosen to taste."
        ),
    ),
    ContrastDeclaration(
        family_id="F3",
        role="topology_destroyed_control",
        varied_factor="higher-order structure (degree distribution, clustering, cycle "
                      "profile), deliberately destroyed",
        held_fixed=("n_nodes", "mean_degree", "weight_multiset", "input_geometry",
                    "spectral_radius", "leak_gain", "probe_protocol"),
        contrast_parent="F1",
        identifiable_claim=_CLAIM_F3,
        forbidden_claims=_FORBIDDEN_DENSITY_CAUSAL + _FORBIDDEN_TOPOLOGY_CAUSAL,
        notes=(
            "Constructible as a mean-degree-matched Erdos-Renyi graph plus the frozen "
            "dense random input mapping. The F2/F3 distinction is the gradient: F2 retains "
            "the full degree_sequence and F3 retains only mean_degree, so F3's structural "
            "set is a strict subset of F2's and F3 discards strictly more structure. This "
            "is why F3 declares mean_degree rather than edge_count: given fixed N the two "
            "are the same quantity, and naming it once keeps the hierarchy checkable "
            "instead of letting a vocabulary difference masquerade as a structural one. "
            "F3 must NOT invert the degree distribution deliberately -- its point is that "
            "it does not retain it, not that it opposes it."
        ),
    ),
    ContrastDeclaration(
        family_id="F4",
        role="recurrence_preserving_structural_counterfactual",
        varied_factor=None,   # BLOCKED: the single factor is not yet named.
        held_fixed=("n_nodes", "edge_count", "degree_sequence", "density",
                    "weight_multiset", "input_geometry", "spectral_radius",
                    "scc_profile", "cycle_length_profile", "leak_gain", "probe_protocol"),
        contrast_parent="F1",
        identifiable_claim=(
            "Under matched N, edge count, degree sequence, density, weight multiset, input "
            "geometry, spectral radius and recurrence statistics, any difference from F1 "
            "is associated with the named high-order organization factor."
        ),
        forbidden_claims=_FORBIDDEN_TOPOLOGY_CAUSAL + (
            "the named factor is the biological factor",
            "recurrence statistics explain the difference",
        ),
        blocked_reason=(
            "The single primary varied factor is NOT YET NAMED. A recurrence-preserving "
            "structural counterfactual is not specific enough to build: candidates are "
            "cell-type connectivity organization, motif organization, edge-direction "
            "arrangement, modular organization, and spatial/locality organization. Each "
            "preserves recurrence by a different mechanism and each is verifiable or not "
            "for a different reason, so choosing one changes what the family can be used "
            "for. Recommended first candidate: cell-type connectivity organization, "
            "because the frozen package ships a cell-type annotation and the "
            "(source-type, target-type) edge-count matrix is directly computable, so both "
            "the 'before' and 'after' are measurable without inventing a new statistic."
        ),
        notes=(
            "Two structural facts constrain the choice and are recorded before the "
            "decision, not after: (1) the row sums of the type-pair matrix ARE the "
            "out-degrees by type, so preserving the full degree sequence already pins most "
            "of the type-pair marginals -- the varied factor must therefore be the type "
            "ASSIGNMENT arrangement, not the marginals. (2) If F4 holds the degree "
            "sequence fixed, it is a different degree-preserving rewire of F1 than F2 is; "
            "the two must be distinguished by what they preserve (F4: recurrence "
            "statistics) and by the A2 overlap budget, or they become redundant controls."
        ),
    ),
    ContrastDeclaration(
        family_id="F5",
        role="deliberately_a3_qualified_construction",
        varied_factor="finite-horizon input-reachable state diversity, guaranteed by "
                      "construction",
        held_fixed=("n_nodes", "edge_count", "mean_degree", "weight_multiset",
                    "input_geometry", "spectral_radius", "leak_gain", "probe_protocol",
                    "readout"),
        contrast_parent="F1",
        identifiable_claim=_CLAIM_F5,
        forbidden_claims=_FORBIDDEN_A3_CAUSAL + _FORBIDDEN_TOPOLOGY_CAUSAL,
        notes=(
            "A3 PASS is a CONSTRUCTION PROPERTY, not a result, and F5 cannot be used to "
            "infer that A3 causes food performance: A3-FAIL families are never run on "
            "food data (rule 6.1), so that causal question is unidentifiable by design. "
            "F5's construction may not consult any task metric (pre-registration "
            "Section 4; enforced by FamilySpec's task-blindness guard). The declared "
            "mechanism is composing blocks whose directed cycles have incommensurate "
            "lengths and placing the input across those blocks."
        ),
    ),
)


def blocked_families(matrix: Sequence[ContrastDeclaration] = CONTRAST_MATRIX) -> list[str]:
    """Families that may NOT be instantiated yet."""
    return [d.family_id for d in matrix if d.blocked]


def ready_families(matrix: Sequence[ContrastDeclaration] = CONTRAST_MATRIX) -> list[str]:
    """Families whose varied factor is named, so a generator could be written."""
    return [d.family_id for d in matrix if not d.blocked]


def validate_battery(
    matrix: Sequence[ContrastDeclaration] = CONTRAST_MATRIX,
) -> dict[str, Any]:
    """Check the matrix as a whole. Raises on a design-gate failure.

    The per-family checks were already enforced in ``__post_init__``; what happens here is
    the BATTERY-level check, where the interesting failures are.

    Returns:
        A report with the ready and blocked families and the contrast gradient.

    Raises:
        MatrixValidationError: on any design-gate failure.
    """
    by_id = {d.family_id: d for d in matrix}
    if len(by_id) != len(matrix):
        raise MatrixValidationError("duplicate family_id in the contrast matrix")

    # 1. every non-reference family must name a contrast parent that exists
    for d in matrix:
        if d.contrast_parent is None:
            if d.family_id != "F1":
                raise MatrixValidationError(
                    f"{d.family_id}: only the reference may have no contrast parent"
                )
            continue
        if d.contrast_parent not in by_id:
            raise MatrixValidationError(
                f"{d.family_id}: contrast_parent {d.contrast_parent!r} is not in the matrix"
            )

    # 2. one primary varied factor per family, and it must be unique across the battery
    seen: dict[str, str] = {}
    for d in matrix:
        if d.blocked:
            continue
        key = str(d.varied_factor).strip().lower()
        if key in seen:
            raise MatrixValidationError(
                f"{d.family_id} and {seen[key]} declare the same varied factor "
                f"({d.varied_factor!r}); two families may not vary the same factor unless "
                "their contrast parents differ, which is not expressible here"
            )
        seen[key] = d.family_id

    # 3. F2 and F3 must be NON-REDUNDANT controls
    f2, f3 = by_id.get("F2"), by_id.get("F3")
    if f2 is not None and f3 is not None:
        if f2.varied_factor == f3.varied_factor:
            raise MatrixValidationError(
                "F2 and F3 declare the same varied factor; they must be a gradient, not "
                "two random baselines"
            )
        f2_struct = f2.held_fixed_set & STRUCTURAL_STATISTICS
        f3_struct = f3.held_fixed_set & STRUCTURAL_STATISTICS
        if not (f2_struct - f3_struct):
            raise MatrixValidationError(
                "F2 holds no structural statistic fixed that F3 discards, so F2 and F3 are "
                "redundant controls. F2 must retain first-order statistics "
                "(degree_sequence) that F3 deliberately does not."
            )
        if not (f3_struct < f2_struct):
            raise MatrixValidationError(
                f"F3's structural-statistic set {sorted(f3_struct)} must be a STRICT SUBSET "
                f"of F2's {sorted(f2_struct)}: the F1 -> F2 -> F3 gradient requires F3 to "
                "retain strictly less structure. Otherwise the intended gradient does not "
                "exist and the two are two random baselines."
            )

    # 4. the reference must vary nothing
    f1 = by_id.get("F1")
    if f1 is not None and not str(f1.varied_factor).lower().startswith("none"):
        raise MatrixValidationError(
            "F1 is the reference and must vary nothing; its varied_factor should be the "
            "'none (reference...)' sentinel"
        )

    return {
        "n_families": len(matrix),
        "ready": ready_families(matrix),
        "blocked": blocked_families(matrix),
        "non_redundant_controls_confirmed": bool(f2 is not None and f3 is not None),
        "gradient": _gradient(by_id),
        "instances_generated": 0,
        "food_evaluation": "HOLD",
    }


def _gradient(by_id: Mapping[str, ContrastDeclaration]) -> list[dict[str, Any]]:
    """The F1 -> F2 -> F3 structure-retention gradient, as a checkable ladder."""
    ladder: list[tuple[str, str]] = [("F1", "reference: everything as measured")]
    if "F2" in by_id:
        ladder.append(("F2", "first-order statistics retained (degree_sequence held fixed)"))
    if "F3" in by_id:
        ladder.append(("F3", "only coarse statistics retained (density held fixed)"))
    return [{"family_id": fid, "structure_retained": note} for fid, note in ladder]


def contrast_matrix_table(matrix: Sequence[ContrastDeclaration] = CONTRAST_MATRIX) -> str:
    """Render the matrix as the Markdown table this stage exists to produce."""
    rows = [
        "| Family | Role | Contrast parent | Primary varied factor | Held fixed | "
        "Identifiable claim | Forbidden stronger claims | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in matrix:
        varied = d.varied_factor if d.varied_factor else "**NOT YET NAMED**"
        status = "**BLOCKED**" if d.blocked else "ready"
        rows.append(
            "| {fid} | {role} | {parent} | {varied} | {fixed} | {claim} | {forb} | {status} |".format(
                fid=d.family_id,
                role=d.role,
                parent=d.contrast_parent or "— (reference)",
                varied=varied,
                fixed="; ".join(d.held_fixed),
                claim=d.identifiable_claim,
                forb="; ".join(d.forbidden_claims),
                status=status,
            )
        )
    return "\n".join(rows)
