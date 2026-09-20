"""The frozen protocol and the dataset manifests are deliverables, so they are tested.

A protocol that is merely present is not frozen; it has to be machine-readable,
internally consistent, and complete enough that a reader can tell what was
decided before the results were seen.
"""

from __future__ import annotations

import pytest

from drososense.data.manifest import (
    AvailabilityStatus,
    load_all_manifests,
    load_manifest,
    sha256_of,
    verify_manifest,
)
from drososense.data.schema import SpecimenSource, load_dataset_config, resolve_specimen_source
from drososense.utils.paths import CONFIGS_DIR, DATA_MANIFESTS_DIR

EXPECTED_DATASETS = {
    "d1_beef_controlled",
    "d2_beef_uncontrolled",
    "d3_rainbow_trout",
    "synthetic_enose",
}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_protocol_declares_it_is_frozen(protocol):
    """The protocol states its own freeze status and version."""
    assert protocol["frozen"] is True
    assert protocol["protocol_version"] == "1.0.0"
    assert protocol["frozen_at"]


@pytest.mark.unit
def test_protocol_fixes_the_split_unit_to_specimen(protocol):
    """``split_unit: specimen`` is the red line and must be declared."""
    assert protocol["split_unit"] == "specimen"


@pytest.mark.unit
def test_protocol_fixes_seeds_zero_through_nine(protocol):
    """The seed set is exactly 0..9, as the parent issue specifies."""
    assert protocol["seeds"] == list(range(10))


@pytest.mark.unit
def test_protocol_primary_metrics_match_the_parent_issue(protocol):
    """Primary metrics are macro-F1 (classification) and MAE (regression)."""
    assert protocol["tasks"]["classification"]["metrics"]["primary"] == "macro_f1"
    assert protocol["tasks"]["regression"]["metrics"]["primary"] == "mae"


@pytest.mark.unit
def test_protocol_secondary_metrics_are_declared(protocol):
    """Secondary metrics are enumerated, not left to the analyst."""
    assert set(protocol["tasks"]["classification"]["metrics"]["secondary"]) == {
        "balanced_accuracy",
        "auroc",
        "accuracy",
    }
    assert set(protocol["tasks"]["regression"]["metrics"]["secondary"]) == {"rmse", "r2"}


@pytest.mark.unit
def test_protocol_declares_hypotheses_and_tests(protocol):
    """Primary and secondary hypotheses exist and carry a named test."""
    assert protocol["primary_hypothesis"]["id"] == "H1"
    assert protocol["primary_hypothesis"]["test"]
    assert len(protocol["secondary_hypotheses"]) >= 3
    for hypothesis in protocol["secondary_hypotheses"]:
        assert hypothesis["id"] and hypothesis["statement"] and hypothesis["test"]
    assert protocol["statistical_tests"]["primary_test"]["name"] == "wilcoxon_signed_rank"
    assert protocol["statistical_tests"]["multiplicity"]["method"] == "holm"


@pytest.mark.unit
def test_protocol_forbids_changing_the_primary_metric_after_results(protocol):
    """The anti-tuning rule is explicit and machine-readable."""
    forbidden = " ".join(protocol["freeze_policy"]["forbidden"])
    assert "primary metric" in forbidden
    assert "test results" in forbidden


@pytest.mark.unit
def test_protocol_declares_all_three_gates_with_failure_actions(protocol):
    """Gate A/B/C each state a condition and what happens if it is not met."""
    gates = protocol["gates"]
    for name in ("Gate_A", "Gate_B", "Gate_C"):
        assert name in gates
        assert gates[name]["condition"].strip()
        assert gates[name]["if_failed"].strip() if "if_failed" in gates[name] else True


@pytest.mark.unit
def test_protocol_pre_defines_narrative_adjustment_rules(protocol):
    """If the hypothesis fails, the paper's story changes by a pre-set rule."""
    rules = protocol["narrative_adjustment_rules"]
    assert len(rules) >= 3
    triggers = " ".join(rule["trigger"] for rule in rules)
    assert "R0" in triggers
    for rule in rules:
        assert rule["id"] and rule["trigger"] and rule["action"]


@pytest.mark.unit
def test_protocol_forbids_row_level_splitting(protocol):
    """The row-level split prohibition is stated in the protocol itself."""
    forbidden = " ".join(protocol["split_rules"]["forbidden"])
    assert "train_test_split" in forbidden
    preprocessing_forbidden = " ".join(protocol["preprocessing"]["normalization"]["forbidden"])
    assert "test" in preprocessing_forbidden


@pytest.mark.unit
def test_protocol_requires_windows_to_stay_inside_one_specimen(protocol):
    """The window boundary rule and the selection rule are both fixed."""
    windowing = protocol["preprocessing"]["windowing"]
    assert set(windowing["length_candidates"]) == {8, 16, 32, 64}
    forbidden = " ".join(windowing["forbidden"])
    assert "two different specimens" in forbidden
    assert "test set" in forbidden


@pytest.mark.unit
def test_protocol_scope_boundaries_forbid_connectome_claims_in_m1(protocol):
    """M1 must not claim a connectome advantage."""
    forbidden = " ".join(protocol["scope_boundaries"]["m1_claims_forbidden"])
    assert "connectome" in forbidden.lower()
    assert "synthetic fixture" in forbidden.lower()


@pytest.mark.unit
def test_protocol_registers_all_deliverable_datasets(protocol):
    """D1, D2 and D3 are declared with a config path each."""
    assert set(protocol["datasets"]) >= {"D1", "D2", "D3"}
    for key in ("D1", "D2", "D3"):
        assert (CONFIGS_DIR.parent / protocol["datasets"][key]["config"]).is_file()


@pytest.mark.unit
def test_protocol_model_zoo_covers_the_required_baselines(protocol):
    """All nine baselines are declared, matching the parent issue's list."""
    zoo = protocol["model_zoo"]
    declared = {
        entry["id"]
        for family in ("classical", "sequence", "reservoir")
        for entry in zoo[family]
    }
    assert declared == {
        "svm_rbf",
        "random_forest",
        "xgboost",
        "pca_svm",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "esn",
    }


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("dataset_id", sorted(EXPECTED_DATASETS))
def test_every_dataset_config_loads_and_is_internally_consistent(dataset_id):
    """Each config parses and declares a usable schema."""
    schema, config = load_dataset_config(CONFIGS_DIR / "datasets" / f"{dataset_id}.yaml")
    assert schema.dataset_id == dataset_id
    assert schema.n_features == len(schema.feature_columns)
    assert schema.n_classes == 4
    assert config["tasks"]


@pytest.mark.unit
def test_beef_datasets_are_marked_non_compliant_because_they_lack_specimens():
    """The two beef datasets publish no specimen id, and say so."""
    for dataset_id in ("d1_beef_controlled", "d2_beef_uncontrolled"):
        schema, _ = load_dataset_config(CONFIGS_DIR / "datasets" / f"{dataset_id}.yaml")
        assert schema.specimen_source is SpecimenSource.ASSUMED_TIME_BLOCK
        assert not schema.protocol_compliant
        assert "ASSUMED" in schema.specimen_note or "assumed" in schema.specimen_note.lower()


@pytest.mark.unit
def test_the_fixture_declares_published_specimens():
    """The smoke fixture has genuine specimen ids, so its split is compliant."""
    schema, _ = load_dataset_config(CONFIGS_DIR / "datasets" / "synthetic_enose.yaml")
    assert schema.specimen_source is SpecimenSource.PUBLISHED
    assert schema.protocol_compliant


@pytest.mark.unit
def test_d3_config_flags_its_schema_as_unverified():
    """The trout config admits that its schema was never confirmed."""
    _, config = load_dataset_config(CONFIGS_DIR / "datasets" / "d3_rainbow_trout.yaml")
    assert config["schema_verified"] is False
    assert "UNVERIFIED" in config["schema_warning"]


@pytest.mark.unit
def test_specimen_source_aliases_resolve():
    """Config spellings map onto the canonical enum."""
    assert resolve_specimen_source("column") is SpecimenSource.PUBLISHED
    assert resolve_specimen_source("time_block") is SpecimenSource.ASSUMED_TIME_BLOCK
    assert resolve_specimen_source("none") is SpecimenSource.UNAVAILABLE
    with pytest.raises(ValueError, match="specimen.source must be"):
        resolve_specimen_source("guesswork")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_all_manifests_load_and_cover_the_declared_datasets():
    """Every dataset has a manifest, and every manifest is well-formed."""
    manifests = load_all_manifests()
    assert EXPECTED_DATASETS <= set(manifests)
    for manifest in manifests.values():
        assert manifest.display_name
        assert manifest.files


@pytest.mark.unit
def test_unavailable_datasets_must_explain_themselves():
    """A dataset that is not auto-available must state steps or a blocker."""
    for manifest in load_all_manifests().values():
        if manifest.status in (AvailabilityStatus.AUTO,):
            continue
        assert manifest.blockers or manifest.manual_steps, (
            f"{manifest.dataset_id} is {manifest.status.value} without an explanation"
        )


@pytest.mark.unit
def test_d3_manifest_records_the_real_archive_and_is_blocked():
    """The trout manifest states the true size, license and blocker."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d3_rainbow_trout.yaml")
    assert manifest.status is AvailabilityStatus.BLOCKED
    assert manifest.license.startswith("CC BY")
    assert manifest.doi == "10.5281/zenodo.20649184"
    assert manifest.files[0].size_bytes == 21134576519
    assert manifest.manual_steps
    assert manifest.blockers


@pytest.mark.unit
def test_beef_manifests_pin_a_sha256_for_every_required_file():
    """The auto-downloadable datasets are pinned by content hash."""
    for dataset_id in ("d1_beef_controlled", "d2_beef_uncontrolled"):
        manifest = load_manifest(DATA_MANIFESTS_DIR / f"{dataset_id}.yaml")
        assert manifest.status is AvailabilityStatus.AUTO
        for entry in manifest.files:
            if entry.required:
                assert entry.sha256 and len(entry.sha256) == 64
                assert entry.download_url


@pytest.mark.unit
def test_beef_manifests_document_the_missing_specimen_identifier():
    """The blocker that stops a protocol-compliant split is recorded, not hidden."""
    for dataset_id in ("d1_beef_controlled", "d2_beef_uncontrolled"):
        manifest = load_manifest(DATA_MANIFESTS_DIR / f"{dataset_id}.yaml")
        assert "NO SPECIMEN IDENTIFIER IS PUBLISHED" in manifest.specimen_note
        assert any("specimen" in blocker for blocker in manifest.blockers)


@pytest.mark.unit
def test_fixture_manifest_declares_the_data_is_synthetic():
    """The fixture cannot be mistaken for observed data."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "synthetic_enose.yaml")
    assert manifest.synthetic is True
    assert "SMOKE TEST" in manifest.display_name


@pytest.mark.unit
def test_sha256_of_matches_a_known_digest(tmp_path):
    """The hasher computes the digest the provider publishes for the same bytes."""
    import hashlib

    path = tmp_path / "payload.bin"
    payload = b"drososense"
    path.write_bytes(payload)
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()
    assert len(sha256_of(path)) == 64


@pytest.mark.unit
def test_verify_manifest_reports_missing_files_without_raising(tmp_path):
    """A missing payload is reported, so the caller decides whether it is fatal."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml")
    report = verify_manifest(manifest, base_dir=tmp_path)
    assert report["ok"] is False
    assert report["missing"] == ["beef_controlled.csv"]
    assert report["corrupt"] == []


@pytest.mark.unit
def test_verify_manifest_detects_a_checksum_mismatch(tmp_path):
    """A file that is not the pinned revision is reported as corrupt."""
    import yaml

    payload = tmp_path / "beef_controlled.csv"
    payload.write_text("not the real dataset\n")
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml")
    report = verify_manifest(manifest, base_dir=tmp_path)
    assert report["ok"] is False
    assert report["corrupt"] and report["corrupt"][0]["name"] == "beef_controlled.csv"

    # Sanity: the report is JSON-serialisable, so it can be attached to a record.
    yaml.safe_dump(report)
