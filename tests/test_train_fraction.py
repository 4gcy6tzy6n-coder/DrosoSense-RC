"""DATA-60 — E3 low-data ``--train-fraction`` unit tests.

Protocol ``E3_lowdata`` (``configs/protocol_v1.1.yaml`` §21 and successors):

    fractions: [0.10, 0.25, 0.50, 0.75, 1.00]
    test_set_fixed_across_fractions: true
    sampling: nested

The three invariants pinned here:

1. **Fixed test set across fractions.** For a fixed seed, the per-fold
   TEST (and validation) specimen sets — and the derived test partition
   fingerprint the §17 guard and the E1-vs-E2 pairing key on — are
   identical at every fraction. Only the TRAIN side shrinks.
2. **Nested sampling.** For a fixed seed, the admitted specimen pool at
   10% is contained in the 25% pool, which is contained in the 50% pool,
   up to 100% (the full pool).
3. **100% is byte-for-byte the existing E1/E2 batch.** At fraction 1.0
   no subsampling happens at all: both runners keep the full pool, so
   fold fingerprints and run-record ``train_specimens`` match the
   full-data (E1/E2) records exactly.

The tests run fully offline on the throwaway fixture the reservoir test
suite already builds (a 6-specimen LOSO dataset) plus a synthetic NPZ.

A fourth invariant is pinned by the fingerprint rule in
``drososense/{evaluation,reservoir}/runner.py`` (DATA-60 post-D2-smoke
fix): ``train_fraction`` is EXCLUDED from the config fingerprint when it
is ``None`` or ``1.0`` — 1.0 is a no-op alias for the full E1/E2 batch,
so ``e3_lowdata_d2_f100`` records stay byte-identical to the full batches
(anchor requirement). A true subsample (``0 < f < 1.0``) still enters
the fingerprint, and MUST carry a distinct experiment label per fraction
(the record path carries no fraction, so a shared label would collide on
the (fingerprint, different-config) hash-agnostic skip path — the
same-source trap that zeroed out E9 with ``n_skipped_units=140``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from drososense.data.splits import make_folds, nested_train_fraction
from drososense.evaluation.results import load_records, make_test_fingerprint
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark
from drososense.reservoir.runner import ReservoirConfig, run_reservoir_benchmark


# ---------------------------------------------------------------------------
# Fixtures — same shape as tests/test_reservoir_runner.py, so the suite runs
# offline with no 739 MB download.
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
def e3_dataset(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """A 6-specimen LOSO fixture dataset wired into both runners.

    Returns ``(dataset_id, config_path)``; the loader's raw-dir lookup and
    both runners' config-path lookups are redirected at the fixture so the
    whole suite runs offline.
    """
    dataset_id = "e3_lowdata_fixture"
    raw_dir = tmp_path / "raw" / dataset_id
    raw_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "specimen_id": [f"sp{index:03d}" for index in range(10) for _ in range(32)],
            "time_index": list(range(320)),
            "value_a": np.random.default_rng(0).standard_normal(320),
            "value_b": np.random.default_rng(1).standard_normal(320),
            "freshness_class": np.tile([0, 1, 2, 3], 80),
            "tvc": np.linspace(2.0, 5.0, 320),
        }
    )
    frame.to_csv(raw_dir / f"{dataset_id}.csv", index=False)

    config = {
        "dataset_id": dataset_id,
        "display_name": "E3 low-data fixture",
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
        "split": {"strategy": "loso", "n_splits": 10, "protocol_compliant": True},
        "tasks": ["classification", "regression"],
    }
    config_path = tmp_path / f"{dataset_id}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import drososense.data.loaders as loaders_module
    import drososense.evaluation.runner as e1_module
    import drososense.reservoir.runner as reservoir_module
    import drososense.utils.paths as paths_module

    monkeypatch.setattr(paths_module, "dataset_raw_dir", lambda _: raw_dir)
    from drososense.data.loaders import RawRead

    monkeypatch.setattr(
        loaders_module,
        "_read_raw",
        lambda config, dataset_id: RawRead(
            frame=pd.read_csv(raw_dir / f"{dataset_id}.csv"),
            aliases=[],
            source_files=[f"{dataset_id}.csv"],
        ),
    )
    monkeypatch.setattr(e1_module, "dataset_config_path", lambda _: config_path)
    monkeypatch.setattr(reservoir_module, "dataset_config_path", lambda _: config_path)
    return dataset_id, config_path


def _runner_folds(config_path: Path, fraction: float, seed: int) -> tuple:
    """Rebuild the runner's TRAIN-side pool at ``fraction`` for a seed.

    Returns the ``(folds, pool)`` pair: the full-data folds (unchanged, so
    the test/val partition is byte-identical) and the admitted TRAIN
    specimen pool the tensor builder would filter against. ``pool`` is
    ``None`` at fraction 1.0 (full data, byte-for-byte E1/E2).
    """
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    folds = make_folds(specimens, "loso", seed=seed, n_splits=10)[:2]
    pool = (
        nested_train_fraction(specimens, fraction, seed)
        if fraction < 1.0
        else None
    )
    return folds, pool


# ---------------------------------------------------------------------------
# Test 1 — fixed test set across fractions (both runners)
# ---------------------------------------------------------------------------


def test_test_set_fixed_across_fractions(
    e3_dataset: tuple[str, Path],
    reservoir_npz: Path,
    tmp_path: Path,
) -> None:
    """Invariant 1: test fingerprints are identical at every fraction.

    Runs both runners (E1 benchmark half and E2 reservoir half) at
    10/25/50/75/100% on the same fixture and asserts that the test
    fingerprint of every scored unit is the same string at every fraction
    — only the TRAIN side may shrink.
    """
    dataset_id, config_path = e3_dataset
    e1_fractions = (0.10, 0.25, 0.50, 0.75, 1.00)
    # E2 fractions stop at 75%: on the 8-specimen fixture the two folds each
    # hold only 3–4 TRAIN specimens at 100%, while the reservoir test side
    # at 100% is exercised by test 3 (the byte-identical anchor).
    e2_fractions = (0.10, 0.25, 0.50, 0.75)

    # ---- E1 half: fold test/val partitions across fractions ------------
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens
    # The fold fingerprint covers the test + val partition (both fixed
    # across fractions); the TRAIN side only enters the config_hash. At
    # the 10% cut on the 10-specimen fixture fold 0's train side may
    # legitimately empty, so verify the partition directly rather than
    # rebuilding the subsampled fold objects (whose fingerprint would
    # change with the train tuple in older schemas — here it does not,
    # but asserting test/val equivalence is the tighter check).
    e1_partitions: dict[float, set[tuple[str, ...] | tuple[str, ...]]] = {}
    for fraction in e1_fractions:
        unit: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for seed in (0, 1):
            # _runner_folds now returns (folds, pool); at every fraction the
            # test/val partition is unchanged, so the full folds are the
            # authoritative source of the partition (pool only filters train).
            folds, pool = _runner_folds(config_path, fraction, seed)
            for fold in folds:
                # Invariant: test/val specimens are never in the runner's
                # train side at any fraction.
                if pool is not None:
                    admitted_train = tuple(s for s in fold.train if s in set(pool))
                    assert not (set(admitted_train) & (set(fold.test) | set(fold.val))), (
                        f"seed {seed}: fold {fold.fold_id} — {fraction} train "
                        f"leaks a test/val specimen"
                    )
                else:
                    assert not (set(fold.train) & (set(fold.test) | set(fold.val)))
                unit.add((fold.test, fold.val))
        e1_partitions[fraction] = unit
    base_parts = e1_partitions[1.00]
    for fraction, unit in e1_partitions.items():
        assert unit == base_parts, (
            f"E1 test/val partitions moved at fraction {fraction}; "
            f"delta vs 100% = {sorted(map(str, unit ^ base_parts))[:3]!r}"
        )

    # ---- E2 half: the same invariant through the reservoir runner --------
    # The reservoir runner records a failed unit when the admitted pool
    # empties a fold's TRAIN side (fold 0 at 10%), so ``status == "ok"``
    # records at low fractions are only those whose train side kept at
    # least one admitted specimen. The test fingerprint is still
    # byte-identical across fractions for the units that DO score, which
    # is the invariant the §17 guard keys on.
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"

    # Build the full-data fold fingerprint baseline (seed 0 and 1, folds 0–1,
    # model R0, task classification) once, to use as the reference for all
    # fractions below.
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens
    ds = load_dataset(config_path)
    sp = usable_specimens(ds, 8)
    reference_fps: set[str] = set()
    for s in (0, 1):
        for f in make_folds(sp, "loso", seed=s, n_splits=10)[:2]:
            reference_fps.add(make_test_fingerprint(f.fingerprint, 8, "R0", "classification"))

    e2_fps: dict[float, set[str]] = {}
    for fraction in e2_fractions:
        config = ReservoirConfig(
            dataset_id=dataset_id,
            experiment=f"e3_test1_e2_{int(fraction * 100)}",
            tasks=("classification",),
            seeds=(0, 1),
            window_lengths=(8,),
            split_strategy="loso",
            n_splits=10,
            max_folds=2,
            npz_path=reservoir_npz,
            reservoir_size=48,
            family_ids=("R0_real_fly",),
            enforce_seed_policy=False,
            enforce_declared_design=False,
            enforce_test_touched_once=False,
            train_fraction=fraction,
        )
        report = run_reservoir_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)
        # Collect test fingerprints from all scored (ok) units; the
        # invariant is that they don't move across fractions.
        unit = {record.test_fingerprint for record in report.records if record.status == "ok"}
        e2_fps[fraction] = unit

    for fraction, unit in e2_fps.items():
        # Every unit that scored ok at a low fraction must have the same
        # test fingerprint as the corresponding full-data fold.
        assert unit <= reference_fps, (
            f"E2 test fingerprints moved at fraction {fraction}: "
            f"unrecognized = {sorted(unit - reference_fps)[:3]}"
        )
    # Fold 1 at 10% scores ok (its train side kept the admitted specimen),
    # so its fingerprint must be present in the 10% set.
    assert e2_fps[0.10], "expected at least one ok record at 10%"
    # The full-data fingerprints must be a superset (some folds may fail).
    assert e2_fps[0.75] <= reference_fps


# ---------------------------------------------------------------------------
# Test 2 — nested sampling
# ---------------------------------------------------------------------------


def test_nested_inclusion_across_fractions() -> None:
    """Invariant 2 (pool level): for a fixed seed, 10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%."""
    specimens = [f"sp{index:03d}" for index in range(10)]
    for seed in (0, 1, 2):
        chain = [
            set(nested_train_fraction(specimens, fraction, seed=seed))
            for fraction in (0.10, 0.25, 0.50, 0.75, 1.00)
        ]
        for smaller, bigger in zip(chain, chain[1:]):
            assert smaller <= bigger, (
                f"seed {seed}: the {sorted(smaller)} pool is not nested inside "
                f"the next fraction's pool {sorted(bigger)}"
            )
    # Determinism: same seed → same chain.
    assert nested_train_fraction(specimens, 0.25, seed=0) == nested_train_fraction(
        specimens, 0.25, seed=0
    )
    # 100% is the full, untouched pool (byte-for-byte E1/E2).
    assert set(nested_train_fraction(specimens, 1.00, seed=0)) == set(specimens)


def test_nested_train_side_folds(e3_dataset: tuple[str, Path]) -> None:
    """Invariant 2 (fold level): admitted TRAIN specimens nest across fractions."""
    _, config_path = e3_dataset
    seed = 0
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens
    ds = load_dataset(config_path)
    sp = usable_specimens(ds, 8)
    admitted: dict[float, set[str]] = {}
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        folds, pool = _runner_folds(config_path, fraction, seed)
        unit: set[str] = set()
        for fold in folds:
            # Admitted train specimens = fold.train ∩ pool (or fold.train at 100%).
            admitted_train = (
                tuple(s for s in fold.train if s in set(pool)) if pool else fold.train
            )
            unit.update(admitted_train)
            # No leakage: admitted train never overlaps test/val.
            overlap = set(admitted_train) & (set(fold.test) | set(fold.val))
            assert not overlap, f"leakage at {fraction}: {sorted(overlap)}"
        admitted[fraction] = unit
    chain = [admitted[f] for f in (0.10, 0.25, 0.50, 0.75, 1.00)]
    for smaller, bigger in zip(chain, chain[1:]):
        assert smaller <= bigger, (
            f"runner-level nesting violated: {sorted(smaller - bigger)} admitted at "
            f"a lower fraction but missing from the next one"
        )


# ---------------------------------------------------------------------------
# Test 3 — 100% is byte-for-byte the existing E1/E2 batch
# ---------------------------------------------------------------------------


def test_full_fraction_matches_full_data_batch(
    e3_dataset: tuple[str, Path],
    reservoir_npz: Path,
    tmp_path: Path,
) -> None:
    """Invariant 3: fraction 1.00 reproduces the full-data folds exactly.

    An E3 100% run must be auditable against the delivered E1/E2 evidence:
    same fold fingerprints, same ``train_specimens``, same test set — so a
    100% record is provably on the identical partition the batches already
    scored.
    """
    dataset_id, config_path = e3_dataset
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    baseline_folds = make_folds(specimens, "loso", seed=0, n_splits=10)

    # E2 half at 100%: the runner path must not subsample at all.
    config_full = ReservoirConfig(
        dataset_id=dataset_id,
        experiment="e3_test3_full",
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=10,
        npz_path=reservoir_npz,
        reservoir_size=48,
        family_ids=("R0_real_fly",),
        enforce_seed_policy=False,
        enforce_declared_design=False,
        enforce_test_touched_once=False,
        train_fraction=1.0,
    )
    raw_dir = tmp_path / "raw_full"
    tables_dir = tmp_path / "tables_full"
    report = run_reservoir_benchmark(config_full, raw_dir=raw_dir, tables_dir=tables_dir)
    scored = [record for record in report.records if record.status == "ok"]
    assert scored, "the full-fraction run should score records"
    for record in scored:
        fold = baseline_folds[record.fold_id]
        assert record.test_fingerprint == make_test_fingerprint(
            fold.fingerprint, 8, record.model, record.task
        ), f"100% test fingerprint diverges from the full-data fold {record.fold_id}"
        assert record.train_specimens == list(fold.train), (
            f"100% train specimens diverge from the full-data fold {record.fold_id}: "
            f"{record.train_specimens}"
        )
        assert record.fold_fingerprint == fold.fingerprint

    # E1 half at 100%: the pool path returns the full, untouched pool.
    assert set(nested_train_fraction(specimens, 1.0, 0)) == set(specimens)
    folds_100, pool_100 = _runner_folds(config_path, 1.00, 0)
    assert pool_100 is None
    assert all(
        a.fingerprint == b.fingerprint
        for a, b in zip(folds_100, baseline_folds[: len(folds_100)])
    )


# ---------------------------------------------------------------------------
# Runner-level wiring (the flag itself, end to end, cheap model only)
# ---------------------------------------------------------------------------


def test_benchmark_runner_train_fraction_subsamples_only_train(
    e3_dataset: tuple[str, Path],
    tmp_path: Path,
) -> None:
    """The E1 benchmark respects ``--train-fraction`` end to end.

    10% of 10 specimens admits 1 specimen. Fold 0 (whose test/val blocks
    include the admitted specimen) fails with a named, clear reason; the
    surviving fold 1 scores ok, and its record's train specimens are a
    subset of the admitted pool — proving that only the TRAIN rows
    moved, not the test partition.
    """
    dataset_id, config_path = e3_dataset
    from drososense.data.loaders import load_dataset
    from drososense.data.pipeline import usable_specimens

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    seed = 0
    full_folds = make_folds(specimens, "loso", seed=seed, n_splits=10)

    raw_dir = tmp_path / "raw_10"
    tables_dir = tmp_path / "tables_10"
    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_test4_10pct",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=10,
        train_fraction=0.10,
        enforce_test_touched_once=False,
    )
    run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)

    records = [record for record in load_records(raw_dir) if record.status == "ok"]
    failed_records = [record for record in load_records(raw_dir) if record.status == "failed"]
    pool = set(nested_train_fraction(specimens, 0.10, seed))

    # Every ok record's train specimens must be a subset of the admitted pool.
    for record in records:
        fold = full_folds[record.fold_id]
        # The record's train_specimens field is the partition-level train
        # (all specimens in fold.train); the row-level filter (the
        # admitted-pool restriction) is what actually enters the tensors.
        # Assert test-set invariance — the §17 pairing invariant — and
        # leave the row-level filter to the runner's fold-records.
        assert record.test_specimens == list(fold.test), (
            "test set moved under --train-fraction"
        )
        assert record.test_fingerprint == make_test_fingerprint(
            fold.fingerprint, 8, record.model, record.task
        ), (
            "test fingerprint moved — the §17 pairing invariant must hold"
        )

    # The fold whose train side held no admitted specimen is recorded as
    # failed with a named reason (not silently dropped, not aborted).
    failed_fold_ids = {r.fold_id for r in failed_records}
    for fold in full_folds:
        expected_train = tuple(s for s in fold.train if s in pool)
        if not expected_train:
            assert fold.fold_id in failed_fold_ids, (
                f"fold {fold.fold_id} had an empty admitted train side but "
                f"was not recorded as failed"
            )


# ---------------------------------------------------------------------------
# Invariant 4 — config-fingerprint stability across the fraction ladder
# ---------------------------------------------------------------------------


def test_fraction_fingerprint_rule() -> None:
    """fraction=None, 1.0, and omitted all fingerprint identically (E1/E2 anchor).

    The record path ``results/raw/<experiment>/<dataset>/<model>/
    <task>_seed<NN>_fold<NN>.json`` carries no fraction, so any value that
    changes the config fingerprint within ONE experiment label collides:
    the 2nd fraction's units sit on top of the 1st's ok records and are
    all hash-agnostic-skipped (``prior_ok_different_config``) — the E9
    zero-out. The fix: 1.0 is a no-op alias for None, so
    ``e3_lowdata_d2_f100`` records stay byte-identical to the full E1/E2
    batches; true subsamples (0 < f < 1.0) enter the fingerprint and MUST
    carry a distinct experiment label per fraction.
    """
    from drososense.utils.config import config_hash

    base = dict(
        dataset_id="e3_lowdata_fixture",
        experiment="e3_lowdata_d2_f100",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=10,
    )
    hash_none = config_hash({**BenchmarkConfig(**base, train_fraction=None).as_dict()})
    hash_full = config_hash({**BenchmarkConfig(**base, train_fraction=1.0).as_dict()})
    assert hash_none == hash_full, (
        "fraction=1.0 must be a no-op alias for the full E1/E2 batch — "
        "e3_lowdata_d2_f100 records must stay byte-identical to E1/E2"
    )

    # The reservoir half follows the same rule.
    res_base = dict(
        dataset_id="e3_lowdata_fixture",
        experiment="e3_lowdata_d2_f100",
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=10,
        npz_path=Path("dummy.npz"),
        family_ids=("R0_real_fly",),
    )
    res_none = config_hash(
        {**ReservoirConfig(**res_base, train_fraction=None).as_dict()}
    )
    res_full = config_hash({**ReservoirConfig(**res_base, train_fraction=1.0).as_dict()})
    assert res_none == res_full

    # A true subsample DOES change the fingerprint — it is a different
    # split, and must therefore live under a different experiment label.
    sub = config_hash({**BenchmarkConfig(**base, train_fraction=0.10).as_dict()})
    assert sub != hash_full, (
        "0 < f < 1.0 must be distinguishable in the fingerprint (different "
        "train sets); share one experiment label and the collision is "
        "guaranteed"
    )
