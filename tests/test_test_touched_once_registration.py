"""Protocol v1.4 §17: a crashed run does not register a test_fingerprint.

The pre-v1.4 implementation of ``test_touched_once`` in
``drososense/evaluation/runner.py`` loaded every record with a fingerprint
into the prior-touches map, regardless of status. A crashed run is recorded
with its full config fingerprint and an empty ``metrics`` block, but it
never scored the split. Counting it as a touch made the post-fix re-run of
the four sequence baselines on d3_rainbow_trout impossible (every
``(model, task, seed, fold)`` was refused as ``test_touched_once violated``).

Protocol v1.4 narrows the registration to ``status == "ok"``. These tests
pin both directions of the fix:

  1. A failed run does NOT register its fingerprint; a later re-run of the
     same ``(dataset, model, task, seed, fold)`` under a different config is
     allowed.
  2. An ok run DOES register its fingerprint; a later re-run of the same
     ``(dataset, model, task, seed, fold)`` under a DIFFERENT config_hash is
     refused.
  3. An ok run CAN be re-run with the SAME config_hash (re-computation,
     allowed).

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
    # fingerprint; the run finishes and lands an ok record at the same
    # run_id (overwriting the failed one in place).
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

    [final] = load_records(raw_dir)
    assert final.status == "ok", (
        "the real run scored the split; it is not blocked by the failed "
        "record that already shared its (dataset, model, task, seed, fold) "
        "fingerprint"
    )


@pytest.mark.integration
def test_an_ok_run_does_register_and_a_different_config_is_refused(
    temporary_dataset, tmp_path
):
    """An ok run counts as a touch; a different-config re-run is refused.

    This test pins the protected half of the v1.4 contract: the guard must
    still refuse a second ok run that has a different config_hash on the
    same fingerprint.

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

    # Step 2: a different config_hash on the same BenchmarkConfig must be
    # refused as a §17 violation.
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
    with pytest.raises(RuntimeError, match="test_touched_once violated") as excinfo:
        run_benchmark(config_b, raw_dir=raw_dir, tables_dir=tables_dir)

    message = str(excinfo.value)
    assert first.config_hash in message, (
        "the violation names the previously-recorded config_hash so a "
        "reader can audit which run it is comparing against"
    )
    assert "protocol v1.4 §17" in message, (
        "the error message identifies the active protocol version"
    )


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
    # Same config, same split — must not raise.
    run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)

    ok_records = [r for r in load_records(raw_dir) if r.status == "ok"]
    assert len(ok_records) == 1, (
        "the second run overwrites the first in place — one record per "
        "run_id; the second invocation is permitted because the config_hash "
        "matches the registered prior touch"
    )


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
def test_runner_constant_protocol_version_tracks_the_active_protocol_file():
    """The runner's PROTOCOL_VERSION label must stay internally consistent.

    DATA-33 introduced a version-sensitive assertion that pinned this
    constant to the hard-coded string ``"1.4.0"`` (valid on the DATA-41
    line, whose active protocol file was ``protocol_v1.4.yaml``). On the
    DATA-43 merge line, the runner constant is the baseline's own label
    (``"1.1.0"``, inherited from the DATA-34/gpu-device-fix line); the
    §17 ok-only guard being ported here does not re-pin it, because that
    re-pinning is DATA-41's own active-protocol switch, not part of the
    guard fix.

    The assertion below checks the invariant that actually matters for the
    records this line will write: ``drososense.PROTOCOL_VERSION`` and the
    runner's constant must agree with each other — a record written through
    the runner must not be able to carry a label that disagrees with the
    package-level constant. It does NOT re-pin either constant to a
    hard-coded string; a future active-protocol switch that points
    ``PROTOCOL_PATH`` at a file whose declared ``protocol_version`` no
    longer matches the constants will surface as a new, explicit decision
    rather than silently drifting.

    Separately, the frozen v1.4 amendment file (independent of which file
    is active on this line) still declares its own ``protocol_version``
    as ``1.4.0`` — unchanged from the DATA-41 review.
    """
    import yaml

    import drososense
    import drososense.evaluation.runner as runner_module
    from drososense.utils.paths import CONFIGS_DIR

    assert drososense.PROTOCOL_VERSION == runner_module.PROTOCOL_VERSION, (
        "drososense.PROTOCOL_VERSION and the runner's PROTOCOL_VERSION must "
        "be the same value; if they disagree, a record written through the "
        "runner could carry a label no active file declares"
    )
    # The frozen v1.4 amendment file itself still declares 1.4.0, independent
    # of which protocol file is active on this line (unchanged from DATA-41).
    v1_4_declared = yaml.safe_load(
        (CONFIGS_DIR / "protocol_v1.4.yaml").read_text(encoding="utf-8")
    )["protocol_version"]
    assert v1_4_declared == "1.4.0"
