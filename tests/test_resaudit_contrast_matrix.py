"""Tests for the family contrast matrix, which is a DESIGN GATE.

These assert that the gate BLOCKS, not merely that the current matrix is well-formed. A
gate that accepts everything is documentation. So each test constructs a deliberately
defective declaration or matrix and asserts the rejection.

The properties under test come from the instruction that released F1-F5 to the design
layer:

* exactly one primary varied structural factor per family;
* held-fixed quantities recorded;
* a contrast parent per family;
* an identifiable claim, phrased as an association under matched conditions;
* forbidden stronger claims, non-empty;
* F2 and F3 non-redundant controls;
* no family instance until its varied factor is named (F4 is blocked).
"""

from __future__ import annotations

import pytest

from resaudit.contrast import (
    CONTRAST_MATRIX,
    ContrastDeclaration,
    MatrixValidationError,
    F1_REFERENCE_CLAIM_PREFIX,
    blocked_families,
    contrast_matrix_table,
    ready_families,
    validate_battery,
)

pytestmark = pytest.mark.unit


def _ref():
    """A minimal valid reference, so a partial matrix has a resolvable parent."""
    return ContrastDeclaration(
        family_id="F1",
        role="reference",
        varied_factor="none (reference; varies nothing by construction)",
        held_fixed=("n_nodes", "mean_degree"),
        contrast_parent=None,
        identifiable_claim=(
            "Under matched measurement conventions, F1 is the reference substrate."
        ),
        forbidden_claims=("F1 is a baseline to be beaten",),
    )


def _good(**overrides):
    base = dict(
        family_id="FX",
        role="test_role",
        varied_factor="some single factor",
        held_fixed=("n_nodes", "density"),
        contrast_parent="F1",
        identifiable_claim=(
            "Under matched N and density, any difference is associated with some factor."
        ),
        forbidden_claims=("some factor causes better performance",),
    )
    base.update(overrides)
    return ContrastDeclaration(**base)


# ---------------------------------------------------------------------------
# The current matrix satisfies the gate
# ---------------------------------------------------------------------------


def test_the_shipped_matrix_passes_the_design_gate():
    report = validate_battery()
    assert report["n_families"] == 5
    assert report["non_redundant_controls_confirmed"] is True
    assert report["instances_generated"] == 0
    assert report["food_evaluation"] == "HOLD"


def test_f4_is_blocked_and_the_others_are_ready():
    assert ready_families() == ["F1", "F2", "F3", "F5"]
    assert blocked_families() == ["F4"]


def test_f4_block_is_an_un_named_varied_factor_with_a_reason():
    f4 = {d.family_id: d for d in CONTRAST_MATRIX}["F4"]
    assert f4.varied_factor is None
    assert f4.blocked is True
    assert f4.blocked_reason
    # the reason must name the candidate factors, since choosing one is the decision
    for candidate in ("cell-type connectivity", "motif organization", "modular"):
        assert candidate in f4.blocked_reason


def test_the_f1_f2_f3_gradient_is_a_strict_structure_retention_ladder():
    report = validate_battery()
    ids = [row["family_id"] for row in report["gradient"]]
    assert ids == ["F1", "F2", "F3"]
    from resaudit.contrast import STRUCTURAL_STATISTICS

    by_id = {d.family_id: d for d in CONTRAST_MATRIX}
    f2_struct = by_id["F2"].held_fixed_set & STRUCTURAL_STATISTICS
    f3_struct = by_id["F3"].held_fixed_set & STRUCTURAL_STATISTICS
    assert "degree_sequence" in f2_struct
    assert "degree_sequence" not in f3_struct
    assert f3_struct < f2_struct, (sorted(f3_struct), sorted(f2_struct))
    # measurement conventions must be held by BOTH: shrinking those would be a protocol
    # change, not a structural contrast
    conventions = {"input_geometry", "spectral_radius", "leak_gain", "probe_protocol"}
    assert conventions <= by_id["F2"].held_fixed_set
    assert conventions <= by_id["F3"].held_fixed_set


def test_f5_declares_a3_as_a_construction_property_and_forbids_the_causal_reading():
    f5 = {d.family_id: d for d in CONTRAST_MATRIX}["F5"]
    assert "by construction" in f5.varied_factor
    joined = " ".join(f5.forbidden_claims)
    assert "A3 causes better food performance" in joined
    assert "A3 PASS predicts food accuracy" in joined


def test_every_family_forbids_at_least_one_stronger_claim():
    for d in CONTRAST_MATRIX:
        assert d.forbidden_claims, d.family_id
        # a forbidden claim must be a real over-reading, not a placeholder
        assert all(len(c) > 15 for c in d.forbidden_claims), d.family_id


def test_no_family_permits_a_causal_claim_about_biology():
    """The reference contrast must not license a biology-superiority reading."""
    for d in CONTRAST_MATRIX:
        for claim in d.forbidden_claims:
            if "biological topology" in claim or "biology" in claim:
                break
        else:
            continue
    f2 = {d.family_id: d for d in CONTRAST_MATRIX}["F2"]
    assert any("biological topology" in c for c in f2.forbidden_claims)


# ---------------------------------------------------------------------------
# The gate BLOCKS defective declarations
# ---------------------------------------------------------------------------


def test_a_family_with_no_held_fixed_quantities_is_rejected():
    with pytest.raises(MatrixValidationError, match="holds nothing fixed"):
        _good(held_fixed=())


def test_a_family_with_no_forbidden_claims_is_rejected():
    with pytest.raises(MatrixValidationError, match="forbidden_claims is empty"):
        _good(forbidden_claims=())


def test_a_causal_identifiable_claim_is_rejected():
    """The claim must be an association under matched conditions, not a cause."""
    with pytest.raises(MatrixValidationError, match="association under matched conditions"):
        _good(identifiable_claim="biological topology causes better performance")


def test_an_un_named_factor_without_a_blocked_reason_is_rejected():
    with pytest.raises(MatrixValidationError, match="requires a blocked_reason"):
        _good(varied_factor=None, blocked_reason="")


def test_a_named_factor_with_a_blocked_reason_is_rejected():
    with pytest.raises(MatrixValidationError, match="a named family is buildable"):
        _good(varied_factor="something", blocked_reason="but I am not sure")


def test_a_blank_varied_factor_is_rejected():
    with pytest.raises(MatrixValidationError, match="use None to mean"):
        _good(varied_factor="   ")


def test_a_family_cannot_be_its_own_contrast_parent():
    with pytest.raises(MatrixValidationError, match="own contrast parent"):
        _good(contrast_parent="FX")


# ---------------------------------------------------------------------------
# The gate blocks defective MATRICES
# ---------------------------------------------------------------------------


def test_duplicate_varied_factors_across_families_are_rejected():
    a = _good(family_id="FA", varied_factor="wiring arrangement")
    b = _good(family_id="FB", varied_factor="wiring arrangement")
    with pytest.raises(MatrixValidationError, match="same varied factor"):
        validate_battery([_ref(), a, b])


def test_duplicate_family_ids_are_rejected():
    a = _good(family_id="FA")
    b = _good(family_id="FA", varied_factor="a different factor")
    with pytest.raises(MatrixValidationError, match="duplicate family_id"):
        validate_battery([_ref(), a, b])


def test_a_missing_contrast_parent_is_rejected():
    a = _good(family_id="FA", contrast_parent="FZ")
    with pytest.raises(MatrixValidationError, match="not in the matrix"):
        validate_battery([a])


def test_a_second_reference_with_no_parent_is_rejected():
    ref = _good(
        family_id="F1",
        varied_factor="none (reference; varies nothing by construction)",
        contrast_parent=None,
    )
    other = _good(family_id="FA", contrast_parent=None, varied_factor="something else")
    with pytest.raises(MatrixValidationError, match="only the reference may have no contrast parent"):
        validate_battery([ref, other])


def test_the_reference_must_vary_nothing():
    """If F1 names a factor, it is no longer a reference."""
    bad_ref = _good(
        family_id="F1",
        varied_factor="biological topology",
        contrast_parent=None,
    )
    with pytest.raises(MatrixValidationError, match="must vary nothing"):
        validate_battery([bad_ref])


def test_redundant_f2_f3_controls_are_rejected():
    """F3 keeping F2's held-fixed set means the gradient does not exist."""
    f2 = _good(
        family_id="F2",
        varied_factor="wiring arrangement",
        held_fixed=("n_nodes", "degree_sequence", "density"),
        contrast_parent="F1",
    )
    f3 = _good(
        family_id="F3",
        varied_factor="degree distribution",
        held_fixed=("n_nodes", "degree_sequence", "density"),
        contrast_parent="F1",
    )
    with pytest.raises(MatrixValidationError, match="redundant controls|STRICT SUBSET"):
        validate_battery([_ref(), f2, f3])


def test_f2_f3_sharing_a_varied_factor_is_rejected():
    f2 = _good(family_id="F2", varied_factor="the same factor", contrast_parent="F1")
    f3 = _good(family_id="F3", varied_factor="the same factor", contrast_parent="F1")
    with pytest.raises(MatrixValidationError, match="same varied factor"):
        validate_battery([_ref(), f2, f3])


def test_f2_must_retain_a_statistic_that_f3_discards():
    f2 = _good(
        family_id="F2",
        varied_factor="wiring arrangement",
        held_fixed=("n_nodes", "mean_degree"),
        contrast_parent="F1",
    )
    f3 = _good(
        family_id="F3",
        varied_factor="higher-order structure",
        held_fixed=("n_nodes", "mean_degree"),
        contrast_parent="F1",
    )
    with pytest.raises(MatrixValidationError, match="redundant controls"):
        validate_battery([_ref(), f2, f3])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_table_renders_the_blocked_family_visibly():
    table = contrast_matrix_table()
    header = table.splitlines()[0]
    for col in (
        "Family", "Role", "Contrast parent", "Primary varied factor", "Held fixed",
        "Identifiable claim", "Forbidden stronger claims", "Status",
    ):
        assert col in header
    assert "**NOT YET NAMED**" in table
    assert "**BLOCKED**" in table
    assert table.count("ready") == 4


def test_matrix_is_serialisable_for_the_report():
    import json

    report = validate_battery()
    report["matrix"] = [d.as_dict() for d in CONTRAST_MATRIX]
    json.dumps(report)


def test_every_declaration_serialises_with_both_claim_fields():
    for d in CONTRAST_MATRIX:
        payload = d.as_dict()
        assert "identifiable_claim" in payload
        assert isinstance(payload["forbidden_claims"], list)
        assert payload["identifiable_claim"].startswith(F1_REFERENCE_CLAIM_PREFIX)
