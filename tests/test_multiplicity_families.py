"""Multiplicity families: the protocol declares them, the analysis must read them.

Protocol v1.2 enumerates eight families under ``multiplicity.families`` and states
the rule: "The unit of correction is a ``(contrast, condition)`` pair… Every
declared pair belongs to exactly ONE family. A pair that appears in no family is
reported without correction AND labelled exploratory; a pair appearing in two
families is a protocol error and the analysis stops."

Nothing read them. The correction ran inside a ``(metric, dataset)`` group, which
put the single-test primary comparison, the four topology controls and the GRU
baseline into one family, and would have merged the five robustness families into
a single twenty-odd-test family. That is review item N1: ``gates.sig`` decides on
``p_holm``, so an unregistered correction decides the gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze import apply_family_correction  # noqa: E402
from drososense.evaluation.stats import holm_correction  # noqa: E402
from drososense.utils.config import protocol_family_membership  # noqa: E402


def _row(contrast_id: str, condition: str, p_value: float, dataset: str = "d3_rainbow_trout"):
    return {
        "contrast_id": contrast_id,
        "condition": condition,
        "metric": "macro_f1",
        "dataset": dataset,
        "task": "classification",
        "status": "ok",
        "classification": "declared",
        "p_value": p_value,
        "p_holm": float("nan"),
        "family": "",
        "family_note": "",
    }


def _table(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# The membership comes from the protocol
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_membership_is_read_from_the_protocol(protocol):
    """Each declared pair resolves to the family that enumerates it."""
    membership = protocol_family_membership(protocol)
    assert membership[("R0_vs_R2", "full")] == "F_primary"
    assert membership[("R0_vs_R4", "full")] == "F_secondary_topology"
    assert membership[("R0_vs_GRU", "full")] == "F_secondary_baselines"
    assert membership[("R0_vs_R4", "dropout_p0.3")] == "F_dropout"
    assert membership[("R0_vs_R4", "train10pct")] == "F_lowdata"
    assert membership[("R0_vs_R4", "noise_s0.1")] == "F_noise"
    assert membership[("R0_vs_R4", "drift_gain")] == "F_drift"
    assert membership[("R0_vs_R4", "size1000")] == "F_size"


@pytest.mark.unit
def test_no_pair_is_claimed_by_two_families(protocol):
    """family_rule: every declared pair belongs to exactly one family."""
    families = protocol["multiplicity"]["families"]
    seen: dict[tuple[str, str], str] = {}
    for family in families:
        conditions = family.get("conditions") or ["full"]
        for contrast_id in family["tests"]:
            for condition in conditions:
                pair = (contrast_id, condition)
                assert pair not in seen, f"{pair} is in {seen.get(pair)} and {family['id']}"
                seen[pair] = family["id"]
    # And the same mapping is what the production helper returns.
    assert protocol_family_membership(protocol) == seen


@pytest.mark.unit
def test_the_five_robustness_families_are_disjoint(protocol):
    """They share a contrast (R0_vs_R4) and must NOT share a correction family.

    This is the part that was silently wrong: in a ``(metric, dataset)`` grouping
    every dropout, noise, drift and size test on D3/macro_f1 lands in one family,
    so a claim about p_drop = 0.3 would be corrected against twenty tests from
    three other experiments.
    """
    membership = protocol_family_membership(protocol)
    robustness = {
        family_id
        for (_, condition), family_id in membership.items()
        if condition != "full"
    }
    assert robustness == {"F_lowdata", "F_dropout", "F_noise", "F_drift", "F_size"}

    by_family: dict[str, set[str]] = {}
    for (contrast_id, condition), family_id in membership.items():
        if family_id in robustness:
            by_family.setdefault(family_id, set()).add(condition)
    assert by_family["F_lowdata"] == {"train10pct", "train25pct", "train50pct", "train75pct"}
    assert by_family["F_dropout"] == {
        "dropout_p0.1", "dropout_p0.2", "dropout_p0.3", "dropout_p0.4", "dropout_p0.5",
    }
    assert by_family["F_noise"] == {"noise_s0.01", "noise_s0.05", "noise_s0.1", "noise_s0.2"}
    assert by_family["F_drift"] == {"drift_gain", "drift_offset", "drift_combined"}
    assert by_family["F_size"] == {"size250", "size500", "size1000", "size2000", "size4000"}


@pytest.mark.unit
def test_a_pair_in_two_families_stops_the_analysis(protocol):
    """family_rule: a duplicated pair is a protocol error, not a tie to break."""
    broken = {
        "multiplicity": {
            "families": [
                {"id": "F_a", "tests": ["R0_vs_R2"]},
                {"id": "F_b", "tests": ["R0_vs_R2"]},
            ]
        }
    }
    with pytest.raises(ValueError, match="belongs to exactly one family"):
        protocol_family_membership(broken)


@pytest.mark.unit
def test_every_family_alpha_matches_the_declared_alpha(protocol):
    """A family that quietly declared a different alpha would change its own rule."""
    declared = float(protocol["statistical_tests"]["alpha"])
    for family in protocol["multiplicity"]["families"]:
        assert float(family["alpha"]) == declared, family["id"]


# ---------------------------------------------------------------------------
# The correction is applied inside the family, not inside (metric, dataset)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_primary_comparison_is_corrected_only_against_itself(protocol):
    """F_primary is one test: "no correction applies, and none is claimed".

    Under the old ``(metric, dataset)`` grouping the primary comparison was
    corrected against the five other contrasts on the same dataset and metric —
    a correction the protocol explicitly does not pre-register.
    """
    table = _table(
        _row("R0_vs_R2", "full", 0.001),
        _row("R0_vs_R3", "full", 0.010),
        _row("R0_vs_R4", "full", 0.020),
        _row("R0_vs_R5", "full", 0.030),
        _row("R0_vs_R1", "full", 0.040),
        _row("R0_vs_GRU", "full", 0.050),
    )
    corrected = apply_family_correction(table, protocol)
    by_contrast = corrected.set_index("contrast_id")

    assert by_contrast.at["R0_vs_R2", "family"] == "F_primary"
    assert by_contrast.at["R0_vs_R2", "p_holm"] == pytest.approx(0.001)

    assert by_contrast.at["R0_vs_GRU", "family"] == "F_secondary_baselines"
    assert by_contrast.at["R0_vs_GRU", "p_holm"] == pytest.approx(0.050)

    # The topology family is corrected over its own four members and nothing else.
    topology = corrected[corrected["family"] == "F_secondary_topology"]
    assert len(topology) == 4
    expected = holm_correction([0.010, 0.020, 0.030, 0.040])
    assert dict(zip(topology["contrast_id"], topology["p_holm"])) == pytest.approx(
        dict(zip(["R0_vs_R3", "R0_vs_R4", "R0_vs_R5", "R0_vs_R1"], expected))
    )


@pytest.mark.unit
def test_the_old_grouping_would_have_given_a_different_answer(protocol):
    """The size of the defect, as an assertion rather than as a claim.

    Holm over all six contrasts of one dataset and metric — the shipped rule —
    gives the primary comparison 0.006 instead of 0.001, and that number is what
    ``gates.sig`` reads.
    """
    p_values = [0.001, 0.010, 0.020, 0.030, 0.040, 0.050]
    old = holm_correction(p_values)
    new = apply_family_correction(
        _table(*[_row(c, "full", p) for c, p in zip(
            ["R0_vs_R2", "R0_vs_R3", "R0_vs_R4", "R0_vs_R5", "R0_vs_R1", "R0_vs_GRU"], p_values
        )]),
        protocol,
    ).set_index("contrast_id")["p_holm"]

    assert old[0] == pytest.approx(0.006)
    assert new["R0_vs_R2"] == pytest.approx(0.001)
    assert old[0] != new["R0_vs_R2"]


@pytest.mark.unit
def test_conditions_are_corrected_inside_their_own_family(protocol):
    """The same contrast under two conditions is two tests in two families."""
    table = _table(
        _row("R0_vs_R4", "full", 0.020),
        _row("R0_vs_R1", "full", 0.040),
        _row("R0_vs_R4", "dropout_p0.2", 0.005),
        _row("R0_vs_R4", "dropout_p0.4", 0.030),
        _row("R0_vs_R4", "noise_s0.05", 0.007),
    )
    corrected = apply_family_correction(table, protocol).set_index(["contrast_id", "condition"])

    assert corrected.at[("R0_vs_R4", "full"), "family"] == "F_secondary_topology"
    # Two members present out of four: Holm adjusts by the members PRESENT.
    assert corrected.at[("R0_vs_R4", "full"), "p_holm"] == pytest.approx(0.040)
    assert corrected.at[("R0_vs_R4", "dropout_p0.2"), "family"] == "F_dropout"
    assert corrected.at[("R0_vs_R4", "dropout_p0.2"), "p_holm"] == pytest.approx(0.010)
    assert corrected.at[("R0_vs_R4", "dropout_p0.4"), "p_holm"] == pytest.approx(0.030)
    assert corrected.at[("R0_vs_R4", "noise_s0.05"), "family"] == "F_noise"
    assert corrected.at[("R0_vs_R4", "noise_s0.05"), "p_holm"] == pytest.approx(0.007)


@pytest.mark.unit
def test_datasets_and_metrics_are_never_pooled_into_one_correction(protocol):
    """A test on D2 must not be corrected against the same test on D3."""
    table = _table(
        _row("R0_vs_R3", "full", 0.010, dataset="d2_beef_uncontrolled"),
        _row("R0_vs_R4", "full", 0.020, dataset="d2_beef_uncontrolled"),
        _row("R0_vs_R3", "full", 0.001, dataset="d3_rainbow_trout"),
        _row("R0_vs_R4", "full", 0.002, dataset="d3_rainbow_trout"),
    )
    corrected = apply_family_correction(table, protocol)

    d2 = corrected[corrected["dataset"] == "d2_beef_uncontrolled"].set_index("contrast_id")
    d3 = corrected[corrected["dataset"] == "d3_rainbow_trout"].set_index("contrast_id")
    assert d2.at["R0_vs_R3", "p_holm"] == pytest.approx(0.020)
    assert d3.at["R0_vs_R3", "p_holm"] == pytest.approx(0.002)


@pytest.mark.unit
def test_a_pair_in_no_family_is_uncorrected_and_relabelled_exploratory(protocol):
    """family_rule's third clause, on a condition no family enumerates."""
    table = _table(
        _row("R0_vs_R2", "full", 0.001),
        _row("R0_vs_R2", "drift_combined", 0.004),  # R0_vs_R2 is in F_primary at `full` only
    )
    corrected = apply_family_correction(table, protocol).set_index("condition")

    assert corrected.at["full", "classification"] == "declared"
    assert corrected.at["drift_combined", "classification"] == "exploratory"
    assert corrected.at["drift_combined", "family"] == ""
    assert pd.isna(corrected.at["drift_combined", "p_holm"])
    assert "enumerated by no multiplicity family" in corrected.at["drift_combined", "family_note"]


@pytest.mark.unit
def test_an_exploratory_pair_is_never_corrected(protocol):
    """A pair the caller marked exploratory carries no pre-registration to correct."""
    row = _row("esn_vs_gru", "full", 0.001)
    row["classification"] = "exploratory"
    corrected = apply_family_correction(_table(row), protocol)
    assert corrected.at[0, "family"] == ""
    assert pd.isna(corrected.at[0, "p_holm"])
