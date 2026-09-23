"""Protocol v1.5 — Evidence Identity Fix: guard tests.

WHAT THIS FILE PINS. Before v1.5, §17's unit of identity was

    sha256(f"{fold_fingerprint}|w{window_length}|{model}|{task}")[:16]

which names the partition, the window length, the model and the task — and
nothing about the *substrate* the model was built from. Two runs differing in
reservoir size, input normalization, input mapping or rewiring seed therefore
shared one identity. The measured consequences are in
``results/audit/m4_audit/server_evidence/``: the E9 size study evaluated the same
splits at five reservoir sizes and registered ONE fingerprint under SIX config
hashes (70 violations by the project's own rule, and
``load_evidence_bundle`` refuses such a bundle), while the E3 low-data sweep was
skipped in full as ``prior_ok_cross_experiment_anchor`` against an E1 unit it does
not duplicate.

The fix has two halves, and a test for each:

  * the identity now carries the substrate (schema 2, ``make_evidence_unit``);
  * the schema-1 fingerprint is retained as a **lookup alias**, so a re-run of an
    already-scored pre-v1.5 unit is still recognised and still refused. Without
    the alias the fix would convert today's false collision into a real
    violation, which is why it is tested explicitly rather than assumed.
"""

from __future__ import annotations

import hashlib

import pytest
import yaml

from drososense.evaluation.results import (
    EVIDENCE_UNIT_COMPONENTS,
    EVIDENCE_UNIT_SCHEMA_VERSION,
    RunRecord,
    legacy_test_fingerprint,
    make_evidence_unit,
    make_evidence_unit_id,
    make_test_fingerprint,
    record_evidence_unit_aliases,
)
from drososense.evaluation.results import (
    test_touched_once_report as touched_once_report,
)
from drososense.utils.config import protocol_sha256, recorded_protocol_sha256
from drososense.utils.paths import (
    CONFIGS_DIR,
    PROTOCOL_V1_5_PATH,
    PROTOCOL_V1_5_SHA256_PATH,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

BASE = {
    "dataset": "d2_beef_uncontrolled",
    "task": "classification",
    "fold_fingerprint": "be5b09b2c961",
    "model": "R0",
    "window_length": 16,
    "reservoir_size": 250,
    "normalization": "n1_pre_l1",
    "input_mapping": "dense_random_all_nodes",
    "topology_variant": "R0_real_fly",
    "rewire_seed": 0,
}


def unit(**overrides):
    """A complete evidence unit with the given components overridden."""
    return make_evidence_unit(**{**BASE, **overrides})


def record(unit_payload=None, *, config_hash="c0ffee", status="ok", run_id="r"):
    """A minimal RunRecord carrying one identity."""
    unit_payload = unit_payload if unit_payload is not None else unit()
    return RunRecord(
        run_id=run_id,
        experiment="e2_topology",
        dataset=BASE["dataset"],
        model=BASE["model"],
        task=BASE["task"],
        seed=0,
        fold_id=0,
        protocol_version="1.5.0",
        window_length=BASE["window_length"],
        metrics={"macro_f1": 0.5} if status == "ok" else {},
        n_train_windows=10,
        n_test_windows=5,
        train_specimens=["TS1"],
        test_specimens=["TS2"],
        duration_s=1.0,
        environment={},
        timestamp_utc="2026-09-23T00:00:00+00:00",
        fold_fingerprint=BASE["fold_fingerprint"],
        config_hash=config_hash,
        test_fingerprint=unit_payload["id"],
        status=status,
        evidence_unit=unit_payload,
    )


# ---------------------------------------------------------------------------
# 1. same config -> same fingerprint (determinism, and order-independence)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_same_components_give_the_same_identity():
    assert unit()["id"] == unit()["id"]
    assert unit()["id"] == make_evidence_unit_id(**BASE)


@pytest.mark.unit
def test_identity_is_independent_of_argument_order_and_of_extra_names():
    """Canonical JSON with sorted keys: the identity is a function of the tuple."""
    reordered = make_evidence_unit(
        task=BASE["task"],
        dataset=BASE["dataset"],
        topology_variant=BASE["topology_variant"],
        model=BASE["model"],
        normalization=BASE["normalization"],
        rewire_seed=BASE["rewire_seed"],
        fold_fingerprint=BASE["fold_fingerprint"],
        input_mapping=BASE["input_mapping"],
        window_length=BASE["window_length"],
        reservoir_size=BASE["reservoir_size"],
    )
    assert reordered["id"] == unit()["id"]
    assert reordered["components"] == unit()["components"]


@pytest.mark.unit
def test_every_declared_component_is_present_in_the_payload():
    components = unit()["components"]
    assert set(components) == set(EVIDENCE_UNIT_COMPONENTS)
    assert unit()["schema"] == EVIDENCE_UNIT_SCHEMA_VERSION == "2"


# ---------------------------------------------------------------------------
# 2. reservoir_size 250 -> 500 must be a DIFFERENT unit (the E9 defect)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("size_a,size_b", [(250, 500), (500, 1000), (1000, 2000), (2000, 4000)])
def test_reservoir_size_change_changes_the_identity(size_a, size_b):
    """The E9 size study's five sizes must be five units, not one collision."""
    assert unit(reservoir_size=size_a)["id"] != unit(reservoir_size=size_b)["id"]


@pytest.mark.unit
def test_the_e9_size_study_no_longer_collides():
    """Regression for the exact measured defect.

    Under schema 1 all five sizes produced one fingerprint; the six E9 config
    hashes then read as 70 §17 violations. Under schema 2 the five sizes are five
    distinct units, so the same six configs are six legitimate units.
    """
    ids = {unit(reservoir_size=n)["id"] for n in (250, 500, 1000, 2000, 4000)}
    assert len(ids) == 5

    legacy = {
        make_test_fingerprint(BASE["fold_fingerprint"], BASE["window_length"],
                              BASE["model"], BASE["task"])
        for _ in (250, 500, 1000, 2000, 4000)
    }
    assert len(legacy) == 1, (
        "schema 1 is expected to collapse all five sizes onto one identity — "
        "that is the defect, recorded here so the fix cannot be quietly undone"
    )


# ---------------------------------------------------------------------------
# 3. normalization n0_raw -> n1_pre_l1 must be a DIFFERENT unit
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "norm_a,norm_b",
    [
        ("n0_raw", "n1_pre_l1"),
        ("n1_pre_l1", "n5_binary"),
        ("n0_raw", "n4_log_pre_l1"),
        (None, "n1_pre_l1"),
    ],
)
def test_normalization_change_changes_the_identity(norm_a, norm_b):
    assert unit(normalization=norm_a)["id"] != unit(normalization=norm_b)["id"]


# ---------------------------------------------------------------------------
# 4. rewire_seed 0 -> 1: different graph, therefore a different unit
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rewire_seed_change_changes_the_identity():
    """The rewiring seed determines the actual control graph.

    It is part of the identity so that a future decoupling of the rewiring seed
    from the run seed is visible rather than silent. The reservoir runner pins
    ``rewire_seed == seed`` today (asserted below), so this is a guard against
    regression, not a description of current behaviour.
    """
    assert unit(rewire_seed=0)["id"] != unit(rewire_seed=1)["id"]


@pytest.mark.unit
def test_reservoir_runner_pins_the_rewiring_seed_to_the_run_seed():
    """`make_degree_rewired` is called with the run seed, so the identity's
    rewire_seed is the seed whose split `fold_fingerprint` already encodes."""
    import inspect

    import drososense.reservoir.runner as rr

    source = inspect.getsource(rr)
    assert "build_topology_family(" in source
    assert "seed=seed," in source, (
        "the family must be built from the run seed; if that changes, "
        "rewire_seed in the identity must change with it"
    )


# ---------------------------------------------------------------------------
# 5. input_mapping / topology_variant are in the identity (the v2 axis)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_input_mapping_is_part_of_the_identity():
    """v2 replaces the dense random map with an ORN/PN-aligned population.

    If those two could share an identity, the v2 redesign would reproduce the
    same defect on its first sweep.
    """
    dense = unit(input_mapping="dense_random_all_nodes")["id"]
    constrained = unit(input_mapping="orn_pn_constrained")["id"]
    assert dense != constrained


@pytest.mark.unit
def test_declared_input_mapping_constant_matches_the_implementation():
    from drososense.reservoir.connectome_reservoir import (
        INPUT_MAPPING_DENSE_RANDOM,
        make_shared,
    )

    assert INPUT_MAPPING_DENSE_RANDOM == "dense_random_all_nodes"
    shared = make_shared(8, 3, seed=0, input_scale=0.1)
    assert shared.describe()["input_mapping"] == INPUT_MAPPING_DENSE_RANDOM


@pytest.mark.unit
def test_topology_variant_is_part_of_the_identity():
    assert unit(topology_variant="R0_real_fly")["id"] != unit(
        topology_variant="R2_degree_rewired"
    )["id"]


@pytest.mark.unit
def test_schema_two_id_can_never_equal_a_schema_one_fingerprint():
    """The schema tag is inside the digest, so the two namespaces are disjoint."""
    v1 = legacy_test_fingerprint(
        BASE["fold_fingerprint"], BASE["window_length"], BASE["model"], BASE["task"]
    )
    assert make_test_fingerprint(
        BASE["fold_fingerprint"], BASE["window_length"], BASE["model"], BASE["task"]
    ) == v1, "make_test_fingerprint must keep its schema-1 behaviour byte-for-byte"
    assert unit()["id"] != v1


# ---------------------------------------------------------------------------
# 6. the legacy alias keeps pre-v1.5 units protected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_legacy_record_is_registered_under_its_schema_one_fingerprint():
    """A pre-v1.5 record has no components, so only schema-1 can be rebuilt."""
    legacy = RunRecord(
        run_id="legacy",
        experiment="e2_topology",
        dataset=BASE["dataset"],
        model=BASE["model"],
        task=BASE["task"],
        seed=0,
        fold_id=0,
        protocol_version="1.4.0",
        window_length=BASE["window_length"],
        metrics={"macro_f1": 0.5},
        n_train_windows=10,
        n_test_windows=5,
        train_specimens=["TS1"],
        test_specimens=["TS2"],
        duration_s=1.0,
        environment={},
        timestamp_utc="2026-09-22T00:00:00+00:00",
        fold_fingerprint=BASE["fold_fingerprint"],
        config_hash="old",
        test_fingerprint=legacy_test_fingerprint(
            BASE["fold_fingerprint"], BASE["window_length"], BASE["model"], BASE["task"]
        ),
    )
    aliases = record_evidence_unit_aliases(legacy)
    assert legacy.test_fingerprint in aliases
    assert unit()["legacy_id"] in aliases, (
        "the new unit's schema-1 alias must match the legacy record's stored "
        "fingerprint, or the fix turns a false collision into a real violation"
    )


@pytest.mark.unit
def test_a_v1_5_record_is_also_registered_under_its_schema_two_id():
    current = record()
    aliases = record_evidence_unit_aliases(current)
    assert current.evidence_unit["id"] in aliases
    assert current.evidence_unit["legacy_id"] in aliases


@pytest.mark.unit
def test_rerunning_a_legacy_unit_under_v1_5_is_still_refused(tmp_path):
    """End-to-end: the ledger must still block a re-touch after the fix.

    This is the test that fails if someone "simplifies" the fix by switching the
    definition without keeping the alias.
    """
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    legacy_fp = legacy_test_fingerprint("foldsplit0001", 16, "svm_rbf", "classification")
    prior = {
        "run_id": "d2_beef_uncontrolled|svm_rbf|classification|seed00|fold00",
        "experiment": "identity_guard",
        "dataset": "synthetic_enose",
        "model": "svm_rbf",
        "task": "classification",
        "seed": 0,
        "fold_id": 0,
        "protocol_version": "1.4.0",
        "window_length": 16,
        "metrics": {"macro_f1": 0.1},
        "n_train_windows": 1,
        "n_test_windows": 1,
        "train_specimens": [],
        "test_specimens": [],
        "duration_s": 0.1,
        "environment": {},
        "timestamp_utc": "2026-09-22T00:00:00+00:00",
        "fold_fingerprint": "foldsplit0001",
        "config_hash": "legacyhash",
        "test_fingerprint": legacy_fp,
        "status": "ok",
    }
    # The record's stored fingerprint is schema 1; a v1.5 run of the same unit
    # computes a schema-2 id. The ledger has to connect the two.
    prior_record = RunRecord.from_dict(prior)
    assert prior_record.evidence_unit == {}, "a v1.4 record carries no components"

    # the v1.5 unit that supersedes exactly that legacy record
    successor = make_evidence_unit(
        dataset="synthetic_enose",
        task="classification",
        fold_fingerprint="foldsplit0001",
        model="svm_rbf",
        window_length=16,
    )
    assert successor["legacy_id"] == legacy_fp, (
        "the successor's schema-1 alias must equal the legacy stored fingerprint"
    )
    assert legacy_fp in record_evidence_unit_aliases(prior_record)
    assert successor["id"] != legacy_fp
    # and the ledger lookup the runners perform finds it under the alias
    ledger = {}
    for identity in record_evidence_unit_aliases(prior_record):
        ledger.setdefault(identity, prior_record.config_hash)
    matched = next(
        (a for a in (successor["id"], successor["legacy_id"]) if a in ledger), None
    )
    assert matched == legacy_fp
    assert ledger[matched] == "legacyhash"


# ---------------------------------------------------------------------------
# 7. failed runs do not occupy the quota; ok runs do (v1.4 semantics, not lost)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_failed_run_does_not_occupy_the_touch_quota_and_ok_run_does():
    """v1.4's clarification must not regress under the identity change."""
    ok = record(config_hash="h1", run_id="ok-run")
    failed = record(config_hash="h2", status="failed", run_id="failed-run")
    failed = RunRecord(**{**failed.to_dict(), "test_fingerprint": failed.evidence_unit["id"]})

    report = touched_once_report([ok, failed])
    assert report["n_with_fingerprint"] == 1, "only the ok record occupies the quota"
    assert report["n_violations"] == 0

    # ... and an ok record under a different config on the SAME identity is one.
    other = record(config_hash="h3", run_id="ok-run-2")
    report = touched_once_report([ok, other])
    assert report["n_violations"] == 1
    assert report["violations"][0]["test_fingerprint"] == ok.evidence_unit["id"]


# ---------------------------------------------------------------------------
# 8. the report stays honest about which identity produced a record
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_report_separates_identity_schemas_without_reinterpreting_legacy_rows():
    """A schema-1 bundle must keep reporting the collisions schema 1 had.

    Re-labelling them would destroy the evidence that the identity bug existed.
    """
    def legacy_record(config_hash, run_id):
        return RunRecord(
            run_id=run_id,
            experiment="e9_size_d2",
            dataset="d2_beef_uncontrolled",
            model="R0",
            task="classification",
            seed=0,
            fold_id=0,
            protocol_version="1.4.0",
            window_length=16,
            metrics={"macro_f1": 0.3},
            n_train_windows=1,
            n_test_windows=1,
            train_specimens=["TS1"],
            test_specimens=["TS2"],
            duration_s=0.1,
            environment={},
            timestamp_utc="2026-09-22T00:00:00+00:00",
            fold_fingerprint="e9fold",
            config_hash=config_hash,
            test_fingerprint="collided0000abcd",
            status="ok",
        )

    report = touched_once_report(
        [legacy_record("c1", "a"), legacy_record("c2", "b"), record()]
    )
    assert report["n_violations"] == 1, "the legacy collision is still reported"
    assert report["identity_schema_counts"] == {"schema_1_legacy": 2, "schema_2": 1}
    assert report["n_records_with_evidence_unit"] == 1
    assert report["n_records_with_colliding_evidence_unit_id"] == 0


# ---------------------------------------------------------------------------
# 9. the amendment file itself: frozen, sidecar-matched, earlier files untouched
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_protocol_v1_5_is_a_frozen_amendment_with_a_matching_sidecar():
    assert PROTOCOL_V1_5_PATH.is_file(), "v1.5 amendment file must be committed"
    assert PROTOCOL_V1_5_SHA256_PATH.is_file(), "v1.5 sidecar must be committed"
    assert protocol_sha256(PROTOCOL_V1_5_PATH) == recorded_protocol_sha256(
        PROTOCOL_V1_5_SHA256_PATH
    )

    parsed = yaml.safe_load(PROTOCOL_V1_5_PATH.read_text(encoding="utf-8"))
    assert parsed["protocol_version"] == "1.5.0"
    assert parsed["frozen"] is True
    assert parsed["supersedes"] == "1.4.0"
    assert "identity only" in parsed["scope_statement"]
    assert parsed["evidence_unit_identity"]["schema_version"] == "2"
    assert parsed["evidence_unit_identity"]["legacy_alias"]["required"] is True
    declared = set(parsed["evidence_unit_identity"]["components"])
    assert declared == set(EVIDENCE_UNIT_COMPONENTS), (
        "the amendment's declared components and the code's must agree"
    )
    assert parsed["test_touched_once"]["enabled"] is True


@pytest.mark.unit
def test_earlier_protocol_files_are_byte_frozen():
    """v1.5 adds a file; it must not edit one. Digests are pinned from the
    committed sidecars, so a post-freeze edit is detectable."""
    for name in ("protocol_v1.yaml", "protocol_v1.1.yaml", "protocol_v1.2.yaml",
                 "protocol_v1.3.yaml", "protocol_v1.4.yaml"):
        path = CONFIGS_DIR / name
        sidecar = CONFIGS_DIR / (name.replace(".yaml", "") + ".sha256")
        if not sidecar.is_file():
            continue  # v1 predates the sidecar convention
        assert path.is_file(), f"{name} must remain on disk"
        assert protocol_sha256(path) == recorded_protocol_sha256(sidecar), (
            f"{name} digest drifted — an amendment must ADD a version file"
        )


@pytest.mark.unit
def test_v1_5_does_not_change_the_loadable_base_protocol():
    """Datasets, splits, metrics, gates and decision rules still read v1.3."""
    from drososense.utils.paths import PROTOCOL_PATH, PROTOCOL_SHA256_PATH

    assert PROTOCOL_PATH.name == "protocol_v1.3.yaml"
    assert PROTOCOL_SHA256_PATH.name == "protocol_v1.3.sha256"

    v1_3 = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    v1_5 = yaml.safe_load(PROTOCOL_V1_5_PATH.read_text(encoding="utf-8"))
    # every field v1.5 lists as unchanged must be declared in non_changes
    for token in ("datasets", "split", "alpha", "AUROC", "Holm", "gate expressions"):
        assert any(token in item for item in v1_5["non_changes"]), token
    assert v1_3["statistical_tests"]["alpha"] == 0.05


# ---------------------------------------------------------------------------
# 10. the E3 half: a declared training fraction must not carry the "full" identity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_e3_fractions_get_distinct_conditions_and_never_the_full_identity():
    """E3_lowdata declares that the test split stays fixed across fractions.

    Without a condition axis the four fractions share the ``full`` identity, and
    that is how E3 produced zero records: 100 units skipped as
    ``prior_ok_cross_experiment_anchor`` against an E1 unit they do not
    duplicate. The label is DERIVED from the fraction so the two cannot drift.
    """
    from drososense.data.splits import condition_from_train_fraction

    assert condition_from_train_fraction(None) == "full"
    assert condition_from_train_fraction(1.0) == "full"
    assert condition_from_train_fraction(0.10) == "train10pct"
    assert condition_from_train_fraction(0.25) == "train25pct"
    assert condition_from_train_fraction(0.50) == "train50pct"
    assert condition_from_train_fraction(0.75) == "train75pct"

    ids = {
        make_evidence_unit(
            dataset=BASE["dataset"],
            task=BASE["task"],
            fold_fingerprint=BASE["fold_fingerprint"],
            model=BASE["model"],
            window_length=BASE["window_length"],
            condition=condition_from_train_fraction(fraction),
            reservoir_size=BASE["reservoir_size"],
            normalization=BASE["normalization"],
        )["id"]
        for fraction in (None, 0.10, 0.25, 0.50, 0.75)
    }
    assert len(ids) == 5, "the full batch and the four fractions are five units"

    with pytest.raises(ValueError):
        condition_from_train_fraction(0.0)
    with pytest.raises(ValueError):
        condition_from_train_fraction(1.5)
    # a fraction that is not a declared whole percent is named exactly rather
    # than rounded onto a declared label the run did not use
    assert condition_from_train_fraction(0.333) == "train0p333"


@pytest.mark.unit
def test_condition_is_part_of_config_hash_so_the_ledger_can_tell_them_apart():
    """A condition change must change the configuration hash too.

    If it did not, the ledger would see "same config" across two conditions and
    treat a genuinely different evaluation as a re-computation.
    """
    from drososense.evaluation.runner import BenchmarkConfig
    from drososense.reservoir.runner import ReservoirConfig
    from drososense.utils.config import config_hash

    base = BenchmarkConfig(dataset_id="d2_beef_uncontrolled", experiment="x")
    other = BenchmarkConfig(
        dataset_id="d2_beef_uncontrolled", experiment="x", condition="train10pct"
    )
    assert config_hash(base.as_dict()) != config_hash(other.as_dict())

    rbase = ReservoirConfig(dataset_id="d2_beef_uncontrolled", experiment="x")
    rother = ReservoirConfig(
        dataset_id="d2_beef_uncontrolled", experiment="x", condition="train10pct"
    )
    assert config_hash(rbase.as_dict()) != config_hash(rother.as_dict())
    assert rbase.condition == "full"
