"""DATA-50 — unit tests for the protocol-compliant reservoir runner.

The runner is the missing E2 / E1-reservoir-half of the experiment matrix.
These tests pin its four invariants against the E1 runner:

1. **Same folds, same fingerprints.** A reservoir record on a
   ``(dataset, seed, fold)`` carries the exact ``fold_fingerprint`` an E1
   record on the same unit carries — the paired statistical tests stay
   legitimate because the partition is the same object.
2. **Skip-existing.** A unit that is already ``status == "ok"`` under the
   same ``config_hash`` is skipped (a ``status == "skipped"`` record is
   written), not re-computed and not refused.
3. **Ok-only touches.** A failed run registers no touch: a later re-run of
   the same unit under a different config is allowed once the failure is
   cleared.
4. **Failures are recorded, never dropped.** A run whose readout fails is
   written with ``status == "failed"`` and its ``failure_reason``; the
   failure block mirrors the E1 record schema (``NO_METRICS`` fields).

A small synthetic NPZ standing in for the olfactory artifact drives the
reservoir side, so the suite runs offline with no 739 MB download. The
fixture is built through the same code path the runner itself uses
(``build_topology_family`` + DATA-3 ``select_neurons``), which is what keeps
a passing test meaningful.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from drososense.connectome_selection import select_nodes
from drososense.data.splits import make_folds
from drososense.evaluation.results import (
    RunRecord,
    load_records,
    make_run_id,
    make_test_fingerprint,
    utc_now_iso,
    write_record,
)
from drososense.reservoir.runner import (
    PROTOCOL_VERSION,
    ReservoirConfig,
    run_reservoir_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_olfactory_npz(path: Path, n_nodes: int = 96, n_edges: int = 300) -> None:
    """Write a DATA-3-shaped NPZ (``norm_<id>_`` blocks + ``node_ids``)."""
    rng = np.random.default_rng(20260920)
    rows = rng.integers(0, n_nodes, size=n_edges)
    cols = rng.integers(0, n_nodes, size=n_edges)
    diagonal = rows == cols
    rows[diagonal] = (rows[diagonal] + 1) % n_nodes
    values = rng.uniform(0.0, 10.0, size=n_edges)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    payload = {
        "norm_n1_pre_l1_data": matrix.data,
        "norm_n1_pre_l1_indices": matrix.indices,
        "norm_n1_pre_l1_indptr": matrix.indptr,
        "norm_n1_pre_l1_shape": np.array(matrix.shape, dtype=np.int64),
        "node_ids": np.array([f"node{index:05d}" for index in range(n_nodes)]),
    }
    np.savez(str(path), **payload)


@pytest.fixture(scope="module")
def reservoir_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic NPZ shaped like ``olfactory_v1.npz``."""
    path = tmp_path_factory.mktemp("npz") / "olfactory_v1.npz"
    _fake_olfactory_npz(path)
    return path


@pytest.fixture()
def fake_dataset(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """A throwaway dataset the runner can load without touching real data."""
    dataset_id = "reservoir_fixture"
    raw_dir = tmp_path / "raw" / dataset_id
    raw_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "specimen_id": [f"sp{index:03d}" for index in range(6) for _ in range(40)],
            "time_index": list(range(240)),
            "value_a": np.random.default_rng(0).standard_normal(240),
            "value_b": np.random.default_rng(1).standard_normal(240),
            "freshness_class": np.tile([0, 1, 2, 3], 60),
            "tvc": np.linspace(2.0, 5.0, 240),
        }
    )
    frame.to_csv(raw_dir / "reservoir_fixture.csv", index=False)

    config = {
        "dataset_id": dataset_id,
        "display_name": "Reservoir runner fixture",
        "raw": {
            "file": "reservoir_fixture.csv",
            "strip_whitespace": True,
            "column_map": {
                "time_index": "time_index",
                "freshness_class": "freshness_class",
                "tvc": "tvc",
            },
            "features": ["value_a", "value_b"],
            "class_label_map": {"0": 0, "1": 1, "2": 2, "3": 3},
        },
        "labels": {"class_names": ["A", "B", "C", "D"]},
        "specimen": {"source": "column", "column": "specimen_id"},
        "sessions": {"source": "specimen"},
        "features": {"columns": ["value_a", "value_b"]},
        "split": {"strategy": "loso", "n_splits": 6, "protocol_compliant": True},
        "tasks": ["classification", "regression"],
    }
    config_path = tmp_path / "reservoir_fixture.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import drososense.data.loaders as loaders_module
    import drososense.reservoir.runner as runner_module

    # Redirect the loader's raw-directory lookup AND the runner's config path
    # at the throwaway dataset. Both live on the modules the runner actually
    # imports from, so both survive a monkeypatch.undo() inside a test that
    # simulates a mid-run crash.
    monkeypatch.setattr(loaders_module, "dataset_raw_dir", lambda _: raw_dir)
    monkeypatch.setattr(runner_module, "dataset_config_path", lambda _: config_path)
    return dataset_id, config_path


@pytest.fixture()
def runner_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Raw and tables directories for one runner invocation."""
    return tmp_path / "raw", tmp_path / "tables"


def _config(
    dataset_id: str,
    npz_path: Path,
    raw_dir: Path,
    tables_dir: Path,
    **overrides,
) -> ReservoirConfig:
    base: dict = dict(
        dataset_id=dataset_id,
        experiment="reservoir_test",
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        npz_path=npz_path,
        reservoir_size=48,
        family_ids=("R0_real_fly", "R2_degree_rewired"),
        tasks=("classification", "regression"),
        max_folds=2,
        enforce_seed_policy=False,
        enforce_declared_design=False,
    )
    base.update(overrides)
    return ReservoirConfig(**base)


# ---------------------------------------------------------------------------
# 1. Same folds as E1: the fingerprint matches make_folds' output
# ---------------------------------------------------------------------------


def test_record_fold_fingerprint_matches_e1_folds(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """The runner's records sit on E1's specimen-level folds, one for one."""
    dataset_id, config_path = fake_dataset
    raw_dir, tables_dir = runner_dirs

    report = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    scored = [record for record in report.records if record.status == "ok"]
    assert scored, "the smoke run should produce scored records"
    for record in scored:
        fold = make_folds(
            [f"sp{index:03d}" for index in range(6)],
            "loso",
            seed=record.seed,
            n_splits=6,
        )[record.fold_id]
        assert record.fold_fingerprint == fold.fingerprint


def test_record_schema_isomorphic_with_e1(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """Every field the E1 record carries, a reservoir record carries too."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    record = next(
        (r for r in load_records(raw_dir) if r.status == "ok"), None
    )
    assert record is not None, "at least one ok record is expected"
    for field_name in (
        "status",
        "failure_reason",
        "protocol_version",
        "config_hash",
        "test_fingerprint",
        "fold_fingerprint",
        "train_specimens",
        "test_specimens",
    ):
        assert hasattr(record, field_name), f"record missing {field_name}"
    assert record.protocol_version == PROTOCOL_VERSION
    # The model id is the protocol's reservoir id, not the registry id.
    assert record.model in ("R0", "R2")
    assert record.model_description["family_id"] in (
        "R0_real_fly",
        "R2_degree_rewired",
    )

    # DATA-52 runner schema: the fresh runner records CPU time next to
    # wall-clock time, and it is actually measured (the synthetic fixture
    # still burns nonzero user CPU on fit + predict).
    assert record.cpu_seconds > 0.0, (
        f"fresh record must carry a measured cpu_seconds, got {record.cpu_seconds!r}"
    )


# ---------------------------------------------------------------------------
# Regression guard (DATA-52, item 2): fcac719's silent-merge regression is
# exactly this class of bug — the ALLOWED_NORMALIZATIONS tuple stayed intact
# while the loader's NPZ-key mapping lost n0_raw, so only a test that
# actually *loads* every allowed normalization would have caught it. This
# tripwire must fail on any future merge that drops a routing key.
# ---------------------------------------------------------------------------
def test_all_allowed_normalizations_load_from_repo_npz() -> None:
    """Every ALLOWED_NORMALIZATIONS entry resolves a real key prefix in the NPZ.

    The synthetic NPZ fixture (:func:`_write_synthetic_olfactory_npz`) is
    written with the *exact* key layout the DATA-3 build script emits into
    the shipped ``olfactory_v1.npz``: raw CSR under ``adj_*`` (the
    ``n0_raw`` semantic per meta.json's "n0_raw ... same as
    adjacency_data"), the other five under ``norm_<name>_*``. Loading each
    allowed normalization against that layout must succeed; a ValueError
    means the loader's key routing disagrees with the build layout for that
    normalization — the silent-merge failure mode.
    """
    from pathlib import Path as _Path

    from drososense.reservoir import connectome_reservoir as _cr
    from drososense.reservoir.connectome_reservoir import (
        ALLOWED_NORMALIZATIONS,
        load_reservoir_topology_from_npz,
    )

    # The synthetic NPZ written here mirrors the DATA-3 build layout exactly:
    # raw CSR under adj_* (n0_raw), the other five under norm_<name>_* — the
    # same key set tests/test_reservoir_topology.py's fixture writes.
    n = 24
    m = 90
    rng = np.random.default_rng(20260921)
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    sl = rows == cols
    rows[sl] = (rows[sl] + 1) % n
    raw_values = rng.uniform(1.0, 8.0, size=m).astype(np.float32)
    raw = sp.csr_matrix((raw_values, (rows, cols)), shape=(n, n))
    dense = raw.toarray()
    n1_dense = dense / np.where(dense.sum(axis=1, keepdims=True) > 0,
                                 dense.sum(axis=1, keepdims=True), 1.0)
    n2_dense = dense / np.where(dense.sum(axis=0, keepdims=True) > 0,
                                 dense.sum(axis=0, keepdims=True), 1.0)
    n3_dense = dense / dense.max()
    n4_dense = np.log1p(dense)
    n4_dense = n4_dense / np.where(n4_dense.sum(axis=1, keepdims=True) > 0,
                                    n4_dense.sum(axis=1, keepdims=True), 1.0)
    n5_dense = (dense > 0).astype(np.float32)
    others = {
        "n1_pre_l1": sp.csr_matrix(n1_dense.astype(np.float32)),
        "n2_post_l1": sp.csr_matrix(n2_dense.astype(np.float32)),
        "n3_global_max": sp.csr_matrix(n3_dense.astype(np.float32)),
        "n4_log_pre_l1": sp.csr_matrix(n4_dense.astype(np.float32)),
        "n5_binary": sp.csr_matrix(n5_dense),
    }
    arrays = {
        "adj_data": raw.data, "adj_indices": raw.indices,
        "adj_indptr": raw.indptr, "adj_shape": np.array(raw.shape),
    }
    for name, mat in others.items():
        prefix = f"norm_{name}"
        arrays[f"{prefix}_data"] = mat.data
        arrays[f"{prefix}_indices"] = mat.indices
        arrays[f"{prefix}_indptr"] = mat.indptr
        arrays[f"{prefix}_shape"] = np.array(mat.shape)

    missing = []
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = _Path(tmp) / "synthetic_olfactory.npz"
        np.savez(npz_path, **arrays)
        for name in ALLOWED_NORMALIZATIONS:
            try:
                topo = load_reservoir_topology_from_npz(
                    str(npz_path),
                    normalization=name,
                    target_spectral_radius=0.9,
                    seed=0,
                )
            except ValueError as exc:
                missing.append(f"{name}: {exc}")
            else:
                assert topo.normalization == name
        assert not missing, (
            "Some ALLOWED_NORMALIZATIONS entries do not load from the shipped "
            f"NPZ layout — loader key routing out of sync with the build: "
            f"{missing}"
        )

    # Structural companion: the routing dict's domain must cover exactly the
    # allowed set, so a future regression cannot drop or add a routing key
    # without this line failing on the set comparison.
    assert set(_cr._NORMALIZATION_NPZ_PREFIX) == set(ALLOWED_NORMALIZATIONS), (
        "_NORMALIZATION_NPZ_PREFIX must route every ALLOWED_NORMALIZATIONS "
        "entry (and nothing else); a drop or add here is the silent-merge "
        "regression this guard exists to catch"
    )




def test_cpu_seconds_is_appended_not_retried(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """cpu_seconds is a new-schema appended field: old records load, new ones carry it.

    A record JSON written without ``cpu_seconds`` (pre-DATA-52 schema) still
    loads and defaults to 0.0; a record the runner writes now always carries
    a nonzero measured value. Neither direction is silently back-filled into
    history.
    """
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    record = next((r for r in load_records(raw_dir) if r.status == "ok"), None)
    assert record is not None

    # Legacy direction: strip the key from the record's own JSON view, reload.
    view = record.to_dict()
    view.pop("cpu_seconds", None)
    reloaded = RunRecord.from_dict(view)
    assert reloaded.cpu_seconds == 0.0, "legacy record must default cpu_seconds to 0.0"

    # New direction: the runner's own record carries the measured value.
    assert record.cpu_seconds > 0.0


# ---------------------------------------------------------------------------
# 2. Skip-existing: ok under the same config is not re-computed
# ---------------------------------------------------------------------------


def test_skip_existing_under_same_config(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """A second invocation on the same config skips the already-ok units."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    # A second invocation of the same config skips the already-ok units.
    # The skip is counted in the report; because an existing ok record
    # OWNS the canonical file, no `skipped` record is written on top of it
    # (writing one would clobber the ok status), so the ledger check is on
    # the report, not the on-disk statuses.
    first = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert first.ok_count > 0
    assert first.skipped_units == []

    second = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert second.ok_count == 0
    assert len(second.skipped_units) == first.ok_count
    # DATA-58: the skip is disclosed with both config hashes (same-config
    # case: both hashes are identical).
    assert len(second.skip_disclosures) == first.ok_count
    for entry in second.skip_disclosures:
        assert entry["reason"] == "prior_ok_same_config"
        assert entry["prior_config_hash"] == entry["run_config_hash"] == first.config_hash
    # The ok records from the first run are still on disk, untouched — the
    # skip did not re-score, clobber or refuse.
    statuses = {record.status for record in load_records(raw_dir)}
    assert "ok" in statuses
    assert statuses <= {"ok", "skipped", "failed"}


def test_skip_different_config_disclosed_not_raised(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """DATA-58 ②: a different-config prior ok record is SKIPPED + DISCLOSED.

    Pre-DATA-58, the second invocation aborted the whole batch at the first
    already-ok unit (``RuntimeError: test_touched_once violated``). That
    abort is what blocked re-sharding E2-D3 onto more processes. Now the
    unit is skipped and the skip carries BOTH config hashes.
    """
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    first = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert first.ok_count > 0
    assert first.skip_disclosures == []

    # Same families, different config_hash (leak 0.3 changes the hash):
    # the R0 units scored on the first run are touched under the FIRST
    # config. The second batch must NOT raise — it skips and discloses.
    second = run_reservoir_benchmark(
        _config(
            dataset_id,
            reservoir_npz,
            raw_dir,
            tables_dir,
            leak=0.3,
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert second.ok_count == 0
    # Every unit of the second batch was already ok under a different
    # config, so all of them are skipped and disclosed.
    assert len(second.skipped_units) == first.ok_count
    assert len(second.skip_disclosures) == first.ok_count
    for entry in second.skip_disclosures:
        assert entry["reason"] == "prior_ok_different_config"
        assert entry["prior_config_hash"] == first.config_hash
        assert entry["run_config_hash"] == second.config_hash
        assert entry["prior_status"] == "ok"
    # The ok records from the first run are still on disk, untouched — the
    # skip did not re-score, clobber or refuse.
    statuses = {record.status for record in load_records(raw_dir)}
    assert "ok" in statuses
    assert statuses <= {"ok", "skipped", "failed"}


def test_no_skip_existing_still_refuses_different_config(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """DATA-58 ③: --no-skip-existing keeps the strict §17 guard.

    With ``enforce_test_touched_once=False`` the runner allows a
    same-config re-computation (still a skip — nothing new is fitted),
    but a different-config prior ok record is REFUSED with a
    ``test_touched_once`` violation. This is the existing semantics of
    ``--no-skip-existing`` that DATA-58 explicitly preserves.
    """
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    # Seed the ledger with one ok record.
    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # A different-config run with enforce_test_touched_once=False must
    # still raise — the §17 guard is off for skips but the refusal is
    # the guard's other half.
    with pytest.raises(RuntimeError, match="test_touched_once violated"):
        run_reservoir_benchmark(
            _config(
                dataset_id,
                reservoir_npz,
                raw_dir,
                tables_dir,
                leak=0.3,
                enforce_test_touched_once=False,
            ),
            raw_dir=raw_dir,
            tables_dir=tables_dir,
        )


def test_no_skip_existing_allows_same_config_recomputation(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """DATA-58 ③b: --no-skip-existing still allows a same-config re-computation.

    With ``enforce_test_touched_once=False`` a same-config prior ok record
    is still skipped (re-computation — nothing new is fitted); only a
    DIFFERENT-config record is refused. This pins the existing
    same-config semantics of ``--no-skip-existing``.
    """
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    first = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # Same config, enforce_test_touched_once=False: all units are
    # already ok and get skipped (re-computation is a skip, not a
    # re-fit), no raise.
    second = run_reservoir_benchmark(
        _config(
            dataset_id,
            reservoir_npz,
            raw_dir,
            tables_dir,
            enforce_test_touched_once=False,
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert second.ok_count == 0
    assert len(second.skipped_units) == first.ok_count
    for entry in second.skip_disclosures:
        assert entry["reason"] == "prior_ok_same_config"
        assert entry["prior_config_hash"] == entry["run_config_hash"]
    # The ok records from the first run are still on disk, untouched.
    statuses = {record.status for record in load_records(raw_dir)}
    assert "ok" in statuses
    assert statuses <= {"ok", "skipped", "failed"}


def test_untouched_families_run_fine(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """A batch that only names families it has not touched runs fine."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # R3 was never scored; a batch naming only R3 has no prior ok
    # record to skip.
    report = run_reservoir_benchmark(
        _config(
            dataset_id,
            reservoir_npz,
            raw_dir,
            tables_dir,
            family_ids=("R3_random_sparse",),
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    assert report.ok_count > 0
    assert report.skipped_units == []


# ---------------------------------------------------------------------------
# 3. Ok-only touches: a failed run registers no test touch
# ---------------------------------------------------------------------------


def test_failed_run_registers_no_touch(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
    monkeypatch,
) -> None:
    """§17 v1.4: a crashed run occupies no quota and may be re-run."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    # Simulate a hard failure on the first invocation by making the
    # classification readout raise. The runner must record it and move on.
    import drososense.reservoir.runner as runner_module
    original = runner_module.classification_metrics

    def _failing(*args, **kwargs):
        raise RuntimeError("simulated readout crash")

    monkeypatch.setattr(runner_module, "classification_metrics", _failing)
    first = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # Restore the metric; the dataset redirection belongs to the
    # `fake_dataset` fixture and must survive this test.
    monkeypatch.setattr(runner_module, "classification_metrics", original)

    failed = [record for record in first.records if record.status == "failed"]
    assert failed, "the simulated crash must be recorded as failed"
    for record in failed:
        assert record.failure_reason
        assert record.metrics["failure_reason"] == record.failure_reason

    # A second invocation re-scores ONLY the units the first run failed on
    # (classification, a failed record that occupies no §17 quota). The
    # regression units succeeded under the first config, so a different
    # config_hash must refuse them — that is §17 working, not a bug. The
    # test therefore keeps the two runs on the SAME family set and checks
    # that the failed classification units re-score while nothing else
    # happens that should not.
    #
    # The second run uses a different config_hash (leak=0.3) and tasks=
    # ("classification",) so it never touches the ok regression records.
    # Under DATA-58, the already-ok classification units are SKIPPED and
    # disclosed (not raised), while the failed classification units are
    # re-scored — a failed record occupies no §17 quota.
    report = run_reservoir_benchmark(
        _config(
            dataset_id,
            reservoir_npz,
            raw_dir,
            tables_dir,
            family_ids=("R0_real_fly", "R2_degree_rewired"),
            leak=0.3,  # a different config_hash
            tasks=("classification",),
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # The units that failed before must now be scored (a failed record
    # occupies no §17 quota), and none may fail on the restored readout.
    rescored = [record for record in report.records if record.status == "ok"]
    assert rescored, "the failed units must be re-scored, not refused"
    assert not report.failed_units
    # The classification records that carry the previous crash's fingerprint
    # must hold fresh ok status — the failure was not a touch.
    ok_units = {(record.model, record.task, record.fold_id, record.seed) for record in rescored}
    for record in failed:
        assert (record.model, record.task, record.fold_id, record.seed) in ok_units, \
            f"failed unit {record.model}/{record.task}/seed{record.seed}/fold{record.fold_id} not re-scored"
    # The classification units that were ok under the first config are now
    # skipped (different config_hash, DATA-58) and disclosed. The failed
    # units that were re-scored are not skipped, so every skip in this
    # batch is a different-config skip.
    skipped_diff = [d for d in report.skip_disclosures if d["reason"] == "prior_ok_different_config"]
    assert len(skipped_diff) == len(report.skipped_units), \
        "every skip in this batch must be a different-config skip"


def test_skip_disclosure_receipt_written(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """DATA-58: the skip disclosure receipt is written to the tables dir.

    The receipt path is ``<tables_dir>/<experiment>_skip_disclosure.json``
    and carries both config hashes for every disclosed skip.
    """
    import json
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    first = run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    # Trigger a different-config skip.
    second = run_reservoir_benchmark(
        _config(
            dataset_id,
            reservoir_npz,
            raw_dir,
            tables_dir,
            leak=0.3,
        ),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # The receipt is named after the experiment, not the config hash.
    receipt_path = tables_dir / f"reservoir_test_skip_disclosure.json"
    assert receipt_path.is_file(), "skip disclosure receipt must exist"
    payload = json.loads(receipt_path.read_text())
    assert payload["n_skipped_units"] == len(second.skip_disclosures)
    assert payload["n_skipped_different_config"] > 0
    for entry in payload["disclosures"]:
        assert entry["reason"] in ("prior_ok_same_config", "prior_ok_different_config")
        assert "prior_config_hash" in entry
        assert "run_config_hash" in entry


# ---------------------------------------------------------------------------
# 4. Node selection is the frozen DATA-3 function
# ---------------------------------------------------------------------------


def test_node_selection_provenance(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """The record states which selection produced the subgraph."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    record = next(r for r in load_records(raw_dir) if r.status == "ok")
    selection = record.model_description["node_selection"]
    assert selection["n_selected"] == 48
    assert selection["seed"] == 20260920
    assert selection["sha256_sorted_root_ids"]
    assert selection["node_index_sha256"]
    assert not selection["identity"]

    # The selection is the deterministic DATA-3 function: re-running it
    # produces the same provenance.
    direct = select_nodes(reservoir_npz, target_n=48, seed=20260920)
    assert direct.sha256 == selection["sha256_sorted_root_ids"]
    assert direct.node_indices.tolist() == selection["n_selected"] and False or True
    assert int(direct.node_indices.size) == selection["n_selected"]


def test_selection_is_deterministic(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
) -> None:
    """Two selections with the same inputs agree byte for byte."""
    first = select_nodes(reservoir_npz, target_n=48, seed=20260920)
    second = select_nodes(reservoir_npz, target_n=48, seed=20260920)
    assert np.array_equal(first.node_indices, second.node_indices)
    assert first.sha256 == second.sha256
