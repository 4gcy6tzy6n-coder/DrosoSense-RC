"""Protocol v1.4 §17: a crashed run does not register a test_fingerprint.

The pre-v1.4 implementation of ``test_touched_once`` in
``drososense/evaluation/runner.py`` loaded every record with a fingerprint
into the prior-touches map, regardless of status. A crashed run is recorded
with its full config fingerprint and an empty ``metrics`` block, but it
never scored the split. Counting it as a touch made the post-fix re-run of
the four sequence baselines on d3_rainbow_trout impossible (every
``(model, task, seed, fold)`` was refused as ``test_touched_once violated``).

Protocol v1.4 narrows the registration to ``status == "ok"``. DATA-51
adds the skip-existing clarification: an already-``ok`` unit is SKIPPED,
never re-scored, and the skip is disclosed with both config hashes.
These tests pin both directions of the fix:

  1. A failed run does NOT register its fingerprint; a later re-run of the
     same ``(dataset, model, task, seed, fold)`` under a different config is
     allowed.
  2. An ok run DOES register its fingerprint; a later re-run of the same
     ``(dataset, model, task, seed, fold)`` under a DIFFERENT config_hash is
     SKIPPED and disclosed (pre-DATA-51 it was refused as a §17 violation;
     the skip is what lets a re-shard batch complete rather than abort).
  3. An ok run CAN be re-run with the SAME config_hash (re-computation,
     allowed); the re-run now SKIPS the unit and discloses the skip instead
     of overwriting the record in place.

The tests use the throwaway ``unit_fixture`` dataset so the runner never
touches ``data/raw`` and never depends on which datasets are downloaded.
"""

from __future__ import annotations

import pytest

from drososense.evaluation.results import (
    RunRecord,
    load_records,
    make_run_id,
    utc_now_iso,
    write_record,
)


def _crashed_record(**overrides) -> RunRecord:
    """Build a minimal ``status == "failed"`` record carrying a fingerprint.

    Returns:
        A ``RunRecord`` whose ``status`` is ``"failed"`` and which carries a
        populated ``test_fingerprint`` and a populated ``config_hash``. The
        runner can load it without complaining about missing fields, and the
        pre-v1.4 prior-touches map would have registered it as a touch.
    """
    payload = {
        "run_id": make_run_id("unit_fixture", "svm_rbf", "classification", 0, 0),
        "experiment": "fingerprint_probe",
        "dataset": "unit_fixture",
        "model": "svm_rbf",
        "task": "classification",
        "seed": 0,
        "fold_id": 0,
        "protocol_version": "1.1.0",
        "window_length": 8,
        "metrics": {
            "macro_f1": None,
            "balanced_accuracy": None,
            "accuracy": None,
            "auroc": None,
            "auroc_n_classes_scored": 0,
            "auroc_defined": False,
            "mae": None,
            "rmse": None,
            "r2": None,
            "failure_reason": "RuntimeError: device mismatch",
        },
        "n_train_windows": 0,
        "n_test_windows": 0,
        "train_specimens": [],
        "test_specimens": ["sentinel_specimen"],
        "duration_s": 0.0,
        "environment": {"python": "3.x"},
        "timestamp_utc": utc_now_iso(),
        "evidence_class": "real",
        "protocol_compliant": True,
        "model_description": {},
        "fold_fingerprint": "deadbeef",
        "class_coverage": {"train": [], "test": []},
        "config_hash": "ffff" * 4,
        "test_fingerprint": "0123456789abcdef",
        "empty_class_policy": "require_all_classes",
        "n_train_sessions": 0,
        "n_test_sessions": 1,
        "status": "failed",
        "failure_reason": "RuntimeError: device mismatch",
        "notes": "",
        "selection": {},
    }
    payload.update(overrides)
    return RunRecord(**payload)


@pytest.mark.integration
def test_a_crashed_run_does_not_register_test_touched_once(
    temporary_dataset, tmp_path
):
    """A failed record does NOT block a later re-run of the same fingerprint.

    Protocol v1.4 §17: only ``status == "ok"`` records register a fingerprint.
    A failed run is recorded for completeness, but the pre-run guard must
    not refuse a later successful run of the same
    ``(dataset, model, task, seed, fold)`` under a different ``config_hash``.

    Args:
        temporary_dataset: Pytest fixture (loader / runner redirected to
            a throwaway dataset under ``tmp_path``).
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_record(
        _crashed_record(
            dataset=dataset_id,
            model="svm_rbf",
            task="classification",
            seed=0,
            fold_id=0,
            window_length=8,
            test_fingerprint="0123456789abcdef",
            config_hash="ffff" * 4,
            fold_fingerprint="deadbeef",
        ),
        base_dir=raw_dir,
    )

    [failed] = load_records(raw_dir)
    assert failed.status == "failed"
    assert failed.test_fingerprint

    # The benchmark run sees the failed record but does NOT block on its
    # fingerprint; the run finishes and lands its own ok record.
    #
    # Protocol v1.5 changed the on-disk consequence, deliberately. A record's
    # path is now addressable by evidence unit, and this hand-written crashed
    # record carries none (it is a pre-v1.5 shape), so the ok record no longer
    # overwrites it in place: a *failed* record is not evidence, and silently
    # clobbering it with a different unit's record was the old behaviour's side
    # effect, not a requirement. §17's rule is about the quota, not the file, and
    # is asserted below.
    run_benchmark(
        BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="fingerprint_probe",
            models=("svm_rbf",),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            n_splits=3,
            max_folds=1,
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    after = load_records(raw_dir)
    ok_records = [r for r in after if r.status == "ok"]
    failed_records = [r for r in after if r.status == "failed"]
    assert len(ok_records) == 1, (
        "the real run scored the split; it is not blocked by the failed "
        "record that already shared its (dataset, model, task, seed, fold) "
        "fingerprint"
    )
    assert ok_records[0].status == "ok"
    assert ok_records[0].evidence_unit.get("schema") == "2"

    # The crashed record is preserved rather than clobbered, and it still
    # occupies nothing: §17 counts ok records only.
    assert len(failed_records) == 1
    assert failed_records[0].evidence_unit == {}, (
        "the hand-written crashed record is a pre-v1.5 shape and keeps its "
        "unsuffixed legacy path"
    )
    from drososense.evaluation.results import test_touched_once_report

    report = test_touched_once_report(after)
    assert report["n_with_fingerprint"] == 1
    assert report["n_violations"] == 0


@pytest.mark.integration
def test_an_ok_run_is_registered_and_a_different_config_is_skipped_not_refused(
    temporary_dataset, tmp_path
):
    """An ok run counts as a touch; a different-config re-shard SKIPS it.

    This test pins the DATA-51 skip-existing semantics on the protected half
    of the §17 contract. Pre-DATA-51 the guard raised RuntimeError at the
    first already-ok unit and aborted the whole batch; DATA-51 clarifies
    that the unit is instead SKIPPED (nothing is re-fit on the split, so
    §17's intent that a touched test split is not re-fit is preserved) and
    the skip is disclosed with both config hashes so a reader can audit
    which prior run owns the unit and which config the current batch ran
    under. The pre-DATA-51 abort behaviour is recorded in the DATA-51
    issue, not in the test.

    Args:
        temporary_dataset: Pytest fixture.
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config_a = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="fingerprint_probe",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 0}},
    )
    # Step 1: an honest ok run produces a fingerprint and a config_hash.
    run_benchmark(config_a, raw_dir=raw_dir, tables_dir=tables_dir)
    [first] = load_records(raw_dir)
    assert first.status == "ok"
    assert first.test_fingerprint
    assert first.config_hash

    # Step 2: a different config_hash on the same unit must be SKIPPED and
    # disclosed, not raised — the re-shard that pre-DATA-51 would have
    # aborted now completes. The disclosure names the previously-recorded
    # config_hash, which is the same value the pre-DATA-51 RuntimeError
    # would have named, so the audit link is preserved.
    config_b = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="fingerprint_probe",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 7}},
    )
    summary_b = run_benchmark(config_b, raw_dir=raw_dir, tables_dir=tables_dir)

    disclosure = summary_b.attrs["skipped_units"][0]
    assert first.config_hash == disclosure["prior_config_hash"], (
        "the disclosure names the previously-recorded config_hash so a "
        "reader can audit which run it is comparing against"
    )
    assert disclosure["reason"] == "prior_ok_different_config"
    assert "§17" in disclosure["reason"] or disclosure["reason"], (
        "the disclosure reason is the skip category, not a protocol string"
    )
    assert summary_b.attrs["n_skipped_different_config"] == 1, (
        "the different-config prior ok unit is skipped, not aborted"
    )
    [record_on_disk] = load_records(raw_dir)
    assert record_on_disk.config_hash == first.config_hash
    assert record_on_disk.timestamp_utc == first.timestamp_utc


@pytest.mark.integration
def test_an_ok_run_with_the_same_config_is_allowed_as_recomputation(
    temporary_dataset, tmp_path
):
    """An ok run with the same config_hash is a re-computation and is allowed.

    Protocol §17 (and v1.4) permits re-running with the same config: the same
    fingerprint with the same config_hash is reported in the freeze report as
    ``n_repeated_test_fingerprints`` but not as a violation.

    Args:
        temporary_dataset: Pytest fixture.
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="fingerprint_probe",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 0}},
    )

    run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)
    [first] = load_records(raw_dir)
    # Same config, same split — the unit is SKIPPED, not re-scored, and the
    # record on disk is unchanged. Pre-DATA-51 this invocation would have
    # overwritten the record in place; DATA-51's skip semantics mean the
    # record is untouched and the skip is disclosed.
    timestamp_before_rerun = first.timestamp_utc
    second_summary = run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)
    [record_after] = load_records(raw_dir)
    assert record_after.timestamp_utc == timestamp_before_rerun, (
        "the skip must not rewrite the prior record in place — the same ok "
        "record that existed before the second invocation is still the only "
        "record on disk, byte-stable in timestamp"
    )
    assert second_summary.attrs["n_skipped_same_config"] == 1
    disclosure = second_summary.attrs["skipped_units"][0]
    assert disclosure["reason"] == "prior_ok_same_config"
    assert disclosure["prior_config_hash"] == first.config_hash


@pytest.mark.unit
def test_protocol_v1_4_is_a_frozen_sibling_with_a_matching_sidecar():
    """v1.4 is committed as a frozen amendment record, not edited into v1.2.

    The byte-for-byte v1.1 / v1.2 digests are pinned by the SHA-256
    sidecars; v1.4 has its own sidecar with a matching digest.
    """
    import yaml

    from drososense.utils.config import protocol_sha256, recorded_protocol_sha256
    from drososense.utils.paths import CONFIGS_DIR

    for name, recorded in [
        ("protocol_v1.1.yaml", "b87ed6057962b00e41c175acd7c78f314e552c50e2eecf53174e6156d8f7c0b9"),
        ("protocol_v1.2.yaml", "32c57f6efaa8f9f14584d3a153fddc2814c5be8829eb888fc0d6c27d536a0cc1"),
    ]:
        path = CONFIGS_DIR / name
        assert path.is_file(), f"{name} must remain on disk"
        assert protocol_sha256(path) == recorded, f"{name} digest drifted"

    v1_4_path = CONFIGS_DIR / "protocol_v1.4.yaml"
    sidecar_path = CONFIGS_DIR / "protocol_v1.4.sha256"
    assert v1_4_path.is_file(), "v1.4 amendment file must be committed"
    assert sidecar_path.is_file(), "v1.4 sha256 sidecar must be committed"

    actual = protocol_sha256(v1_4_path)
    recorded = recorded_protocol_sha256(sidecar_path)
    assert recorded == actual, (
        f"v1.4 sidecar does not match the file:\n  recorded {recorded}\n  "
        f"actual   {actual}"
    )

    parsed = yaml.safe_load(v1_4_path.read_text(encoding="utf-8"))
    assert parsed["protocol_version"] == "1.4.0"
    assert parsed["frozen"] is True
    assert parsed["supersedes"] == "1.2.0"
    assert "amendment_v1_4" in parsed
    amendment = parsed["amendment_v1_4"]
    assert amendment["trigger"].strip()
    assert "NOT" in amendment["not_driven_by_test_observations"].upper()
    assert amendment["unchanged"]
    assert amendment["runs_affected"]
    assert any(
        "gru" in line or "cnn1d" in line or "tcn" in line or "lstm" in line
        for line in amendment["runs_affected"].splitlines()
    ), "the four sequence baselines must be named in runs_affected"


@pytest.mark.unit
def test_runner_constant_protocol_version_is_v1_5_0():
    """The runner's PROTOCOL_VERSION label moved with the amendment.

    The label tracks the newest frozen amendment, so each generation of rows is
    distinguishable in the contact log and in any aggregation: ``1.1.0``
    (pre-§17-clarification), ``1.4.0`` (crashed runs do not consume the quota),
    and ``1.5.0`` (evidence-unit identity schema 2). v1.5 changes identity only —
    it is not a licence to re-read any earlier row.
    """
    import drososense
    import drososense.evaluation.runner as runner_module
    import drososense.reservoir.runner as reservoir_runner_module

    assert drososense.PROTOCOL_VERSION == "1.5.0"
    assert runner_module.PROTOCOL_VERSION == "1.5.0"
    assert reservoir_runner_module.PROTOCOL_VERSION == "1.5.0"


# ---------------------------------------------------------------------------
# DATA-51 skip-existing: an already-ok unit is skipped, not re-scored, and
# not a batch aborting event. Both directions are pinned:
#   1. same config_hash prior ok record  -> skip, disclosed as a
#      re-computation that is no longer even re-run in place;
#   2. different config_hash prior ok record -> skip + explicit disclosure
#      carrying both config hashes (pre-DATA-51 this raised RuntimeError and
#      aborted the whole batch — the re-shard deadlock that blocked §17).
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_an_ok_run_with_the_same_config_is_skipped_not_recomputed(
    temporary_dataset, tmp_path
):
    """A repeat of a fully-ok unit under the same config is skipped.

    Pre-DATA-51 the second invocation re-ran the unit in place (the same
    record path was overwritten with a second, identical record). DATA-51
    clarifies that the unit is SKIPPED: no re-fit, no re-score, no rewritten
    record, and the skip is disclosed in the summary attrs and on disk.

    Args:
        temporary_dataset: Pytest fixture.
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.results import load_records
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="skip_probe",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 0}},
    )

    first_summary = run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)
    [first] = load_records(raw_dir)
    assert first.status == "ok"
    assert first.timestamp_utc

    # Timestamp captured BEFORE the second run; a re-computation would write
    # a record with a later timestamp at the same run_id path.
    timestamp_before_second_run = first.timestamp_utc

    second_summary = run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)

    [record_after] = load_records(raw_dir)
    assert record_after.status == "ok"
    assert record_after.timestamp_utc == timestamp_before_second_run, (
        "the skip must not rewrite the prior record in place — the same ok "
        "record that existed before the second invocation is still the only "
        "record on disk, byte-for-byte stable in timestamp"
    )

    # The skip is disclosed, not silent: the summary carries the count and the
    # reason, and the on-disk receipt exists next to the summary.
    assert second_summary.attrs["n_skipped_units"] == 1
    assert second_summary.attrs["n_skipped_same_config"] == 1
    assert second_summary.attrs["n_skipped_different_config"] == 0
    disclosure = second_summary.attrs["skipped_units"][0]
    assert disclosure["reason"] == "prior_ok_same_config"
    assert disclosure["prior_config_hash"] == first.config_hash
    assert disclosure["run_config_hash"] == first.config_hash
    assert disclosure["prior_run_id"] == first.run_id
    assert disclosure["prior_status"] == "ok"

    receipt_path = tables_dir / "skip_probe_skip_disclosure.json"
    assert receipt_path.is_file(), (
        "the skip receipt must land on disk so the disclosure is auditable "
        "without a live session"
    )
    import json as _json
    receipt = _json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["n_skipped_units"] == 1
    assert receipt["n_skipped_same_config"] == 1
    assert receipt["disclosures"][0]["reason"] == "prior_ok_same_config"

    # Nothing new was fitted: the second invocation produced zero new records,
    # and its contact-log accounting reflects that — the first invocation's
    # record is the only record the dataset ever produced.
    assert second_summary.empty or second_summary.attrs["n_skipped_units"] == 1


@pytest.mark.integration
def test_an_ok_run_with_a_different_config_is_skipped_with_disclosure_not_aborted(
    temporary_dataset, tmp_path
):
    """A different-config re-shard over an ok unit skips and discloses, not aborts.

    Pre-DATA-51, the second invocation under a different config_hash raised
    ``RuntimeError: test_touched_once violated`` at the first already-ok unit
    and killed the whole batch — the deadlock that prevented re-sharding the
    D3 deep-model sweep into model x seed-half processes. DATA-51's skip
    semantics: the unit is SKIPPED (nothing is re-fit on the split, preserving
    §17's intent that a touched test split is not re-fit) and the skip is
    disclosed with both config hashes. The rest of the batch proceeds.

    Args:
        temporary_dataset: Pytest fixture.
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.results import load_records
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config_a = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="skip_probe_diff",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 0}},
    )
    run_benchmark(config_a, raw_dir=raw_dir, tables_dir=tables_dir)
    [prior_record] = load_records(raw_dir)
    assert prior_record.status == "ok"

    # A DIFFERENT config (different random_state -> different config_hash),
    # aimed at the SAME unit. Pre-DATA-51 this is where the batch died.
    config_b = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="skip_probe_diff",
        models=("svm_rbf",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
        model_params={"svm_rbf": {"random_state": 7}},
    )
    # No raise, no abort: the batch completes.
    summary_b = run_benchmark(config_b, raw_dir=raw_dir, tables_dir=tables_dir)

    [record_on_disk] = load_records(raw_dir)
    # The prior ok record is untouched — it still carries config A's hash and
    # timestamp; the different-config run wrote nothing on top of it.
    assert record_on_disk.status == "ok"
    assert record_on_disk.config_hash == prior_record.config_hash
    assert record_on_disk.timestamp_utc == prior_record.timestamp_utc

    # The skip is disclosed with BOTH config hashes, which is the audit
    # requirement of the issue: which prior ok record owns the unit, and what
    # config this batch that skipped it ran under.
    assert summary_b.attrs["n_skipped_units"] == 1
    assert summary_b.attrs["n_skipped_same_config"] == 0
    assert summary_b.attrs["n_skipped_different_config"] == 1
    disclosure = summary_b.attrs["skipped_units"][0]
    assert disclosure["reason"] == "prior_ok_different_config"
    assert disclosure["prior_config_hash"] == prior_record.config_hash
    assert disclosure["prior_config_hash"] != disclosure["run_config_hash"]
    assert disclosure["prior_run_id"] == prior_record.run_id
    assert disclosure["prior_status"] == "ok"

    # The on-disk receipt carries the same disclosure, so it is auditable
    # without a live session.
    import json as _json
    receipt_path = tables_dir / "skip_probe_diff_skip_disclosure.json"
    assert receipt_path.is_file()
    receipt = _json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["n_skipped_different_config"] == 1
    assert receipt["disclosures"][0]["reason"] == "prior_ok_different_config"
    assert receipt["disclosures"][0]["prior_config_hash"] == prior_record.config_hash


@pytest.mark.integration
def test_skip_does_not_disturb_the_ok_record_set_on_a_mixed_batch(
    temporary_dataset, tmp_path
):
    """A mixed batch: already-ok units are skipped, new units still run.

    The re-shard use case is not an all-skip batch — it is a batch where most
    of the unit space is already covered by a prior sweep and only the new
    (model, task, seed, fold) slice is uncomputed. Those new units must still
    run and land records; the covered units are skipped and disclosed; the
    total ok record count on disk stays exactly the union of the two,
    with no overwrite of any prior record.

    Args:
        temporary_dataset: Pytest fixture.
        tmp_path: Pytest temp directory.
    """
    from drososense.evaluation.results import load_records
    from drososense.evaluation.runner import BenchmarkConfig, run_benchmark

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Batch 1: two models, one task, one seed, one fold -> two ok records.
    config_first = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="skip_mixed",
        models=("svm_rbf", "random_forest"),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
    )
    run_benchmark(config_first, raw_dir=raw_dir, tables_dir=tables_dir)
    first_two = load_records(raw_dir)
    assert len(first_two) == 2
    prior_hashes = {r.run_id: (r.config_hash, r.timestamp_utc) for r in first_two}

    # Batch 2: the RE-SHARD case. Same dataset/window/seed/fold, a DIFFERENT
    # config (a subset model list plus the regression task). Every
    # classification unit batch 1 already scored under a different
    # config_hash is skipped and disclosed; the new regression units for the
    # requested model still run to completion and land fresh records. No
    # prior record is overwritten.
    config_second = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="skip_mixed",
        models=("svm_rbf",),
        tasks=("classification", "regression"),
        seeds=(0,),
        window_lengths=(8,),
        n_splits=3,
        max_folds=1,
    )
    summary_second = run_benchmark(config_second, raw_dir=raw_dir, tables_dir=tables_dir)

    records_now = load_records(raw_dir)
    # The svm_rbf classification unit from batch 1 is skipped, not
    # re-scored, and not overwritten: its record is byte-stable on disk.
    svm_class = next(
        r for r in records_now if r.model == "svm_rbf" and r.task == "classification"
    )
    assert svm_class.status == "ok"
    assert svm_class.config_hash == prior_hashes[svm_class.run_id][0]
    assert svm_class.timestamp_utc == prior_hashes[svm_class.run_id][1]

    # The random_forest records from batch 1 are still on disk, untouched by
    # batch 2 (which did not request that model).
    rf = [r for r in records_now if r.model == "random_forest"]
    assert len(rf) == 1, "batch 1's random_forest record must survive untouched"
    assert rf[0].timestamp_utc == prior_hashes[rf[0].run_id][1]

    # The NEW regression unit ran and produced a fresh record.
    reg = next(
        r for r in records_now if r.model == "svm_rbf" and r.task == "regression"
    )
    assert reg.status == "ok"
    assert reg.run_id not in prior_hashes, "the regression record is new, not a rewrite"

    # Exactly one prior ok unit was skipped: batch 1's svm_rbf classification
    # record (now met under a different config_hash because batch 2's model
    # list and task set differ).
    assert summary_second.attrs["n_skipped_units"] == 1
    assert summary_second.attrs["n_skipped_different_config"] == 1
    disclosure = summary_second.attrs["skipped_units"][0]
    assert disclosure["model"] == "svm_rbf"
    assert disclosure["task"] == "classification"
    assert disclosure["reason"] == "prior_ok_different_config"
    assert disclosure["protocol_version"] == "1.5.0", (
        "the disclosure carries the active protocol version label so it is "
        "auditable without a live session"
    )

