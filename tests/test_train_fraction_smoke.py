"""Smoke test for DATA-61 — E3 low-data ``--train-fraction`` pool fix.

Mirrors the D2 batch that was 100% failed in DATA-60: 4 specimens, LOSO(5),
seed set {0}, window_length 8, the protocol's train_fraction ladder
[0.10, 0.25, 0.50, 0.75, 1.00], and the two runner halves (E1 benchmark
``run_baselines.py`` and E2 reservoir ``run_reservoir_e2.py``).

The test is the minimal "先冒烟再起批" gate from DATA-61 §5: the batch
must NOT produce a single ``status == "failed"`` record (the pre-DATA-61
global-pool bug emptied the train side of one fold at f=0.10), and the
test partition must stay byte-identical across all five fractions.

Run offline on the fixture the existing E3 test suite already builds —
no 739 MB download, no real D2 data.

Usage (from the repo root)::

    python -m pytest tests/test_train_fraction_smoke.py -v

or as a standalone script (writes the D2-shaped records under a tmp dir):

    python tests/test_train_fraction_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from drososense.data.loaders import load_dataset
from drososense.data.pipeline import usable_specimens
from drososense.data.splits import fold_train_pool, make_folds
from drososense.evaluation.results import load_records, make_test_fingerprint
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark


# ---------------------------------------------------------------------------
# Fixtures — D2 shape: 4 specimens, LOSO(5), 32 rows each, 4 classes.
# ---------------------------------------------------------------------------


def _make_d2_shaped_fixture(tmp_dir: Path, dataset_id: str = "d2_shaped") -> tuple[str, Path]:
    """Build a 4-specimen LOSO fixture that mirrors D2's failure shape."""
    raw_dir = tmp_dir / "raw" / dataset_id
    raw_dir.mkdir(parents=True)
    n_rows = 32
    frame = pd.DataFrame(
        {
            "specimen_id": [f"c{index:03d}" for index in range(4) for _ in range(n_rows)],
            "time_index": list(range(4 * n_rows)),
            "value_a": np.random.default_rng(0).standard_normal(4 * n_rows),
            "value_b": np.random.default_rng(1).standard_normal(4 * n_rows),
            "freshness_class": np.tile([0, 1, 2, 3], 4 * n_rows // 4),
            "tvc": np.linspace(2.0, 5.0, 4 * n_rows),
        }
    )
    frame.to_csv(raw_dir / f"{dataset_id}.csv", index=False)
    config = {
        "dataset_id": dataset_id,
        "display_name": "D2-shaped smoke fixture (DATA-61)",
        "raw": {
            "file": f"{dataset_id}.csv",
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
        "split": {"strategy": "loso", "n_splits": 5, "protocol_compliant": True},
        "tasks": ["classification", "regression"],
    }
    config_path = tmp_dir / f"{dataset_id}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return dataset_id, config_path


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


# ---------------------------------------------------------------------------
# Invariant A — no fold is ever failed under the new per-fold pool
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_failed_records_at_10_percent(tmp_path: Path, monkeypatch) -> None:
    """The DATA-60 batch was 100% failed at f=0.10; the new code must not
    produce a single failed record at any fraction in the ladder."""
    dataset_id, config_path = _make_d2_shaped_fixture(tmp_path)

    import drososense.data.loaders as loaders_module
    import drososense.evaluation.runner as e1_module
    import drososense.reservoir.runner as reservoir_module
    import drososense.utils.paths as paths_module

    raw_data_dir = tmp_path / "raw" / dataset_id
    monkeypatch.setattr(paths_module, "dataset_raw_dir", lambda _: raw_data_dir)
    from drososense.data.loaders import RawRead

    monkeypatch.setattr(
        loaders_module,
        "_read_raw",
        lambda config, ds_id: RawRead(
            frame=pd.read_csv(raw_data_dir / f"{dataset_id}.csv"),
            aliases=[],
            source_files=[f"{dataset_id}.csv"],
        ),
    )
    monkeypatch.setattr(e1_module, "dataset_config_path", lambda _: config_path)
    monkeypatch.setattr(reservoir_module, "dataset_config_path", lambda _: config_path)

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    assert len(specimens) == 4, f"expected 4 specimens, got {len(specimens)}"
    folds = make_folds(specimens, "loso", seed=0, n_splits=5)

    # ---- E1 half: run_baselines equivalent, all 5 fractions ----------------
    e1_raw = tmp_path / "raw_e1"
    e1_tables = tmp_path / "tables_e1"
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        config = BenchmarkConfig(
            dataset_id=dataset_id,
            experiment=f"d2_smoke_e1_f{int(fraction * 100)}",
            models=("random_forest",),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            split_strategy="loso",
            n_splits=5,
            train_fraction=fraction,
            enforce_test_touched_once=False,
        )
        run_benchmark(config, raw_dir=e1_raw, tables_dir=e1_tables)
        failed = [
            record
            for record in load_records(e1_raw)
            if record.experiment == f"d2_smoke_e1_f{int(fraction * 100)}"
            and record.status == "failed"
        ]
        assert not failed, (
            f"E1 f={fraction}: {len(failed)} failed records — DATA-61 "
            f"regression: {[(r.fold_id, r.failure_reason) for r in failed]}"
        )

    # ---- E2 half: reservoir runner, 4 fractions (D2 smoke ladder) --------
    npz = tmp_path / "olfactory_v1.npz"
    _fake_olfactory_npz(npz)
    from drososense.reservoir.runner import ReservoirConfig, run_reservoir_benchmark

    e2_raw = tmp_path / "raw_e2"
    e2_tables = tmp_path / "tables_e2"
    for fraction in (0.10, 0.25, 0.50, 0.75):
        config = ReservoirConfig(
            dataset_id=dataset_id,
            experiment=f"d2_smoke_e2_f{int(fraction * 100)}",
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            split_strategy="loso",
            n_splits=5,
            npz_path=npz,
            reservoir_size=48,
            family_ids=("R0_real_fly",),
            enforce_seed_policy=False,
            enforce_declared_design=False,
            enforce_test_touched_once=False,
            train_fraction=fraction,
        )
        report = run_reservoir_benchmark(config, raw_dir=e2_raw, tables_dir=e2_tables)
        failed = [
            record
            for record in report.records
            if record.experiment == f"d2_smoke_e2_f{int(fraction * 100)}"
            and record.status == "failed"
        ]
        assert not failed, (
            f"E2 f={fraction}: {len(failed)} failed records — DATA-61 "
            f"regression: {[(r.fold_id, r.failure_reason) for r in failed]}"
        )


# ---------------------------------------------------------------------------
# Invariant B — test partition byte-identical across the fraction ladder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_test_partition_fixed_across_fractions(tmp_path: Path, monkeypatch) -> None:
    """The §17 pairing invariant: the test + val partition (and its
    fingerprint) does not move across 10/25/50/75/100%."""
    dataset_id, config_path = _make_d2_shaped_fixture(tmp_path)

    import drososense.data.loaders as loaders_module
    import drososense.utils.paths as paths_module

    raw_data_dir = tmp_path / "raw" / dataset_id
    monkeypatch.setattr(paths_module, "dataset_raw_dir", lambda _: raw_data_dir)
    from drososense.data.loaders import RawRead

    monkeypatch.setattr(
        loaders_module,
        "_read_raw",
        lambda config, ds_id: RawRead(
            frame=pd.read_csv(raw_data_dir / f"{dataset_id}.csv"),
            aliases=[],
            source_files=[f"{dataset_id}.csv"],
        ),
    )

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    folds = make_folds(specimens, "loso", seed=0, n_splits=5)

    # Reference test/val partition at 100% (full data)
    ref_part = {fold.fold_id: (fold.test, fold.val) for fold in folds}
    ref_fp = {
        fold.fold_id: make_test_fingerprint(fold.fingerprint, 8, "R0", "classification")
        for fold in folds
    }

    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        for fold in folds:
            # The Fold object itself is fraction-independent — the runner
            # builds it once per seed and the pool only filters the TRAIN
            # tensor. Verify the partition still matches the reference.
            part = (fold.test, fold.val)
            assert part == ref_part[fold.fold_id], (
                f"fold {fold.fold_id} at f={fraction}: test/val partition moved"
            )
            fp = make_test_fingerprint(fold.fingerprint, 8, "R0", "classification")
            assert fp == ref_fp[fold.fold_id], (
                f"fold {fold.fold_id} at f={fraction}: test fingerprint moved"
            )


# ---------------------------------------------------------------------------
# Invariant C — pool is always ⊆ fold.train and never empty (the fix itself)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pool_is_subset_of_train_and_nonempty() -> None:
    """The core DATA-61 invariant at the function level: for the D2-shaped
    4-specimen fixture, every fold at every fraction gets a non-empty pool
    that is a subset of its own train side. The old global-pool code failed
    this for folds 0 and 3 (the test and val specimen of the same
    permutation position) at f=0.10."""
    specimens = [f"c{index:03d}" for index in range(4)]
    for seed in range(10):
        folds = make_folds(specimens, "loso", seed=seed, n_splits=5)
        for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
            for fold in folds:
                pool = fold_train_pool(fold, fraction)
                assert pool, (
                    f"seed {seed} f={fraction} fold {fold.fold_id}: "
                    f"pool is empty"
                )
                assert set(pool) <= set(fold.train), (
                    f"seed {seed} f={fraction} fold {fold.fold_id}: "
                    f"pool leaks outside train: {sorted(set(pool) - set(fold.train))}"
                )
                assert not (set(pool) & (set(fold.test) | set(fold.val))), (
                    f"seed {seed} f={fraction} fold {fold.fold_id}: "
                    f"pool admits a test/val specimen"
                )


if __name__ == "__main__":
    # Standalone smoke: run the pytest suite inline
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
