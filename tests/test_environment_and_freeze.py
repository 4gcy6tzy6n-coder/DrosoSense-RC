"""Environment claims and the data-contact log.

Two reporting defects the R0 audit caught are turned into tests here:

* C7 — "159 passed" was true in the declaring environment and false on a machine
  without xgboost, and nothing in the repository made the difference visible.
  ``drososense.utils.env_report`` makes it a value, and these tests hold it to
  naming what is missing instead of implying completeness.
* C3 — the freeze rule needs evidence. The protocol cannot hold its own digest,
  and the first test evaluation cannot be recorded by editing a frozen file, so
  both live outside it: a digest sidecar and a contact log the runner writes.
"""

from __future__ import annotations

import json

import pytest

from drososense.evaluation.contact_log import (
    counts_as_contact,
    load_contact_log,
    record_contact,
)
from drososense.utils.env_report import compare_environments, parse_environment_yml
from drososense.utils.paths import PROTOCOL_V1_1_PATH


# ---------------------------------------------------------------------------
# Environment reporting
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_declared_environment_parses_into_packages():
    """Channels and block headers are not packages; the real declarations are."""
    declarations = parse_environment_yml()
    assert "conda-forge" not in declarations, "a channel is not a dependency"
    assert "pip:" not in declarations
    assert "python" in declarations
    assert declarations["numpy"].startswith(">=")
    assert "xgboost" in declarations


@pytest.mark.unit
def test_the_report_separates_declared_from_installed():
    """The gap between the two is exactly what a bare test count hides."""
    report = compare_environments()
    assert report.python_local
    assert report.platform_local
    names = {p.name for p in report.packages}
    assert {"numpy", "pandas", "scikit-learn", "xgboost", "torch"} <= names
    for status in report.packages:
        if status.installed is None:
            assert status.name in report.missing
            assert not status.meets_specifier


@pytest.mark.unit
def test_an_incomplete_environment_qualifies_its_claims():
    """A claim made here must say it is not made in the declared environment."""
    report = compare_environments()
    if report.complete:
        assert "declared environment" in report.qualification()
    else:
        qualification = report.qualification()
        assert "NOT the declared one" in qualification
        for missing in report.missing:
            assert missing in qualification


@pytest.mark.unit
def test_the_report_is_json_serialisable():
    """It is attached to run summaries, so it must survive a round trip."""
    payload = compare_environments().as_dict()
    assert json.loads(json.dumps(payload)) == payload
    for key in ("complete", "missing", "mismatched", "python_local", "packages"):
        assert key in payload


@pytest.mark.unit
def test_a_missing_environment_file_yields_an_empty_declaration(tmp_path):
    """No declaration means nothing to compare against — not a crash."""
    assert parse_environment_yml(tmp_path / "absent.yml") == {}


# ---------------------------------------------------------------------------
# The data-contact log
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_fixture_does_not_count_as_test_contact():
    """The synthetic fixture cannot produce a protocol result, so it cannot
    be used to claim the frozen metrics were already known."""
    assert not counts_as_contact("group_kfold", True, "synthetic_fixture")


@pytest.mark.unit
def test_a_non_compliant_split_does_not_count_as_test_contact():
    """D1's time-block stand-in is not a specimen-level split."""
    assert not counts_as_contact("time_block_holdout", False, "real")


@pytest.mark.unit
def test_a_compliant_real_run_counts():
    """This is the event the freeze rule is about."""
    assert counts_as_contact("loso", True, "real")
    assert counts_as_contact("group_kfold", True, "real")


@pytest.mark.unit
def test_the_contact_log_records_the_first_counting_evaluation(tmp_path):
    """A synthetic run before a real one must not set the timestamp."""
    record_contact(
        experiment="smoke", dataset="synthetic_enose", split_strategy="group_kfold",
        protocol_compliant=True, evidence_class="synthetic_fixture", n_models=8,
        at="2026-09-20T00:00:00Z", base_dir=tmp_path,
    )
    log = load_contact_log(tmp_path)
    assert log.first_test_evaluation_at is None
    assert not log.started

    record_contact(
        experiment="m1", dataset="d3_rainbow_trout", split_strategy="group_kfold",
        protocol_compliant=True, evidence_class="real", n_models=16,
        at="2026-09-20T01:00:00Z", base_dir=tmp_path,
    )
    log = load_contact_log(tmp_path)
    assert log.first_test_evaluation_at == "2026-09-20T01:00:00Z"
    assert log.datasets_touched == ("d3_rainbow_trout",)

    # A later counting run must not move the first-contact timestamp.
    record_contact(
        experiment="m1", dataset="d2_beef_uncontrolled", split_strategy="loso",
        protocol_compliant=True, evidence_class="real", n_models=16,
        at="2026-09-20T02:00:00Z", base_dir=tmp_path,
    )
    log = load_contact_log(tmp_path)
    assert log.first_test_evaluation_at == "2026-09-20T01:00:00Z"
    assert log.datasets_touched == ("d2_beef_uncontrolled", "d3_rainbow_trout")
    assert len(log.entries) == 3


@pytest.mark.unit
def test_an_absent_contact_log_reads_as_not_started(tmp_path):
    """The absence of the file means no test evaluation has happened."""
    log = load_contact_log(tmp_path)
    assert log.first_test_evaluation_at is None
    assert log.entries == ()


@pytest.mark.unit
def test_the_protocol_records_the_contact_rule_not_the_live_value():
    """The frozen file cannot hold the live value without invalidating itself."""
    import yaml

    protocol = yaml.safe_load(PROTOCOL_V1_1_PATH.read_text(encoding="utf-8"))
    entry = protocol["freeze_evidence"]["data_contact_log"]
    assert entry["first_test_evaluation_at"] is None
    assert entry["datasets_touched"] == []
    assert entry["note"].strip()
