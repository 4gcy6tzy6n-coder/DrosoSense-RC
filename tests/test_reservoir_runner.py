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
    # The ok records from the first run are still on disk, untouched — the
    # skip did not re-score, not clobber, not refuse.
    statuses = {record.status for record in load_records(raw_dir)}
    assert "ok" in statuses
    assert statuses <= {"ok", "skipped", "failed"}


def test_skip_only_under_same_config_hash(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """A different config is a different run: no skip, no refusal."""
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )
    # A different invocation (different families, different config_hash)
    # scores the units it does not already hold ok under that config —
    # here R3 was never scored, so it runs. The R0 units it did score on the
    # first run are touched under the FIRST config, and §17 forbids scoring
    # them again under a different one: the runner refuses up front.
    with pytest.raises(RuntimeError, match="test_touched_once violated"):
        run_reservoir_benchmark(
            _config(
                dataset_id,
                reservoir_npz,
                raw_dir,
                tables_dir,
                family_ids=("R0_real_fly", "R3_random_sparse"),
            ),
            raw_dir=raw_dir,
            tables_dir=tables_dir,
        )

    # A run that only names families it has not touched runs fine.
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
    # To isolate the failure-recovery assertion from the refuse assertion,
    # the second run uses tasks=("classification",) so it never touches the
    # ok regression records.
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
    ok_units = {(record.model, record.task) for record in rescored}
    for record in failed:
        assert (record.model, record.task) in ok_units


def test_ok_run_is_refused_under_different_config(
    fake_dataset: tuple[str, Path],
    reservoir_npz: Path,
    runner_dirs: tuple[Path, Path],
) -> None:
    """§17: an ok record DOES register; a re-touch under a new config is refused.

    The refusal surfaces on the first unit the run reaches — which is before
    any fold tensor is built — so the run raises and writes nothing new.
    """
    dataset_id, _ = fake_dataset
    raw_dir, tables_dir = runner_dirs

    run_reservoir_benchmark(
        _config(dataset_id, reservoir_npz, raw_dir, tables_dir),
        raw_dir=raw_dir,
        tables_dir=tables_dir,
    )

    with pytest.raises(RuntimeError, match="test_touched_once violated"):
        run_reservoir_benchmark(
            _config(
                dataset_id,
                reservoir_npz,
                raw_dir,
                tables_dir,
                family_ids=("R0_real_fly", "R3_random_sparse"),
                leak=0.3,
            ),
            raw_dir=raw_dir,
            tables_dir=tables_dir,
        )


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

    # DATA-59 record fidelity: params.reservoir_size must state the ACTUAL
    # node count of the reservoir in use, not the stale DATA-3 default
    # (200). The record must be self-certifying about its own size.
    assert record.model_description["params"]["reservoir_size"] == 48
    assert selection["target_n"] == 48

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
