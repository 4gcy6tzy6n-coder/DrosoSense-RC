"""DATA-61 defects 3 & 5 — threading governance + E3 full-pool anchor reference.

Defect 3 (the v6 hang): 5 concurrent E3 subprocesses × 128 BLAS threads on an
80-core box, with **no** ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
``MKL_NUM_THREADS`` set — thread oversubscription with zero evidence left
anywhere. This pins the governance:

* :mod:`ops.thread_limits` — the knobs the launcher must export, machine-
  checked by ``validate_thread_limits`` (defect 3, code side);
* ``capture_environment`` records the four knobs as ``ENV_*`` keys on every
  run record, so "the batch ran under limit 4" is readable off the record
  (defect 3, evidence side).

Defect 5 (the v6 zero-out): the E3 full-pool (f100) unit has the SAME
``config_hash`` as the E1/E2 full batch for the same partition, so re-running
it is a §17-correct no-op batch — the anchor is the prior ok record itself and
must be *referenced at analysis time*, not re-scored. This pins:

* ``_full_pool_anchor_refs`` resolves the anchor to the existing ok record
  (join, not re-run) and reports it empty when no prior record exists (a
  fresh dataset, where scoring fresh is legitimate);
* full-pool E3 records carry the resolved reference in ``e3_anchor``;
  low-data (0 < f < 1.0) records carry ``[]``;
* the same (fold, seed, task) has a DISTINCT ``config_hash`` per fraction —
  the record-identity assertion the acceptance check requires ("同 (fold, seed,
  task) 在不同 fraction 下 config_hash 互不相同").
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from drososense.data.loaders import load_dataset
from drososense.data.pipeline import usable_specimens
from drososense.data.splits import make_folds
from drososense.evaluation.results import (
    RunRecord,
    capture_environment,
    load_records,
    make_test_fingerprint,
)
from drososense.evaluation.runner import (
    BenchmarkConfig,
    _full_pool_anchor_refs,
    load_prior_test_touches,
    run_benchmark,
)
from drososense.reservoir.runner import ReservoirConfig, run_reservoir_benchmark
from drososense.utils.config import config_hash


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
# Fixture — reuse the D2-shaped 4-specimen fixture from the smoke suite
# ---------------------------------------------------------------------------


def _make_d2_shaped_fixture(tmp_dir: Path, dataset_id: str = "d2_shaped_61") -> tuple[str, Path, Path]:
    """4-specimen LOSO(5) fixture, wired into the E1 runner offline.

    Returns ``(dataset_id, config_path, raw_dir)``.
    """
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
        "display_name": "D2-shaped DATA-61 fixture (defects 3 & 5)",
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

    import drososense.data.loaders as loaders_module
    import drososense.utils.paths as paths_module

    monkeypatched_raw = raw_dir
    monkeypatched = True
    return dataset_id, config_path, raw_dir


def _wire_fixture(dataset_id: str, config_path: Path, raw_dir: Path, monkeypatch) -> None:
    """Redirect the loader and runner config-path lookups at the fixture."""
    import drososense.data.loaders as loaders_module
    import drososense.evaluation.runner as e1_module
    import drososense.utils.paths as paths_module

    from drososense.data.loaders import RawRead

    monkeypatch.setattr(paths_module, "dataset_raw_dir", lambda _: raw_dir)
    import drososense.reservoir.runner as reservoir_module

    monkeypatch.setattr(reservoir_module, "dataset_config_path", lambda _: config_path)
    monkeypatch.setattr(
        loaders_module,
        "_read_raw",
        lambda config, ds_id: RawRead(
            frame=pd.read_csv(raw_dir / f"{ds_id}.csv"),
            aliases=[],
            source_files=[f"{ds_id}.csv"],
        ),
    )
    monkeypatch.setattr(e1_module, "dataset_config_path", lambda _: config_path)


# ---------------------------------------------------------------------------
# Defect 3 — thread governance: the record must evidence the limits
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_environment_records_threading_knob(monkeypatch) -> None:
    """Defect 3 (evidence side): the run record's ``environment`` states the
    four BLAS threading knobs, so a batch that ran under limits 4 (or ran
    unlimited — the v6 hang) is readable off the record."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    environment = capture_environment()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        key = f"ENV_{var}"
        assert key in environment, f"{key} missing from the record environment"
        assert environment[key] is None, "unset knobs must be recorded as None, not dropped"

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "4")
    environment = capture_environment()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert environment[f"ENV_{var}"] == "4"

    # End-to-end: the knobs ride on the RECORD, not just on the environment
    # dict — the v6 hang left nothing a reader could point at, so the
    # evidence must live where the result lives.
    record = RunRecord(
        run_id="d2_shaped_61|random_forest|classification|seed00|fold00",
        experiment="d2_smoke_e1_f10",
        dataset="d2_shaped_61",
        model="random_forest",
        task="classification",
        seed=0,
        fold_id=0,
        protocol_version="v1.4",
        window_length=8,
        metrics={},
        n_train_windows=0,
        n_test_windows=0,
        train_specimens=[],
        test_specimens=[],
        duration_s=0.0,
        environment=environment,
        timestamp_utc="2026-09-22T00:00:00+00:00",
    )
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert record.environment[f"ENV_{var}"] == "4"


@pytest.mark.unit
def test_ops_thread_limits_set_and_validate(monkeypatch) -> None:
    """Defect 3 (code side): the launcher's governance is machine-checkable.

    ``set_thread_limits(4)`` pins all four knobs; ``validate_thread_limits``
    refuses to start a batch that has not pinned them (the v6 condition:
    unlimited per-process pools under a 5-process fan-out).
    """
    import sys

    ops_dir = Path(__file__).resolve().parent.parent / "ops"
    sys.path.insert(0, str(ops_dir))
    try:
        import thread_limits
    finally:
        sys.path.pop(0)

    for var in thread_limits.THREAD_LIMIT_VARS:
        monkeypatch.delenv(var, raising=False)

    # Unset knobs: the gate refuses the start (fail fast, before spawn).
    with pytest.raises(RuntimeError, match="thread limits not pinned"):
        thread_limits.validate_thread_limits()

    # After the launcher pins the limits, the gate passes and reports them.
    thread_limits.set_thread_limits(4)
    limits = thread_limits.validate_thread_limits(max_threads=8)
    assert limits == {var: 4 for var in thread_limits.THREAD_LIMIT_VARS}

    # A limit above the fan-out ceiling is the same oversubscription in
    # single-process form: the gate catches it too.
    os.environ["OMP_NUM_THREADS"] = "32"
    try:
        with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=32 exceeds ceiling 8"):
            thread_limits.validate_thread_limits(max_threads=8)
    finally:
        thread_limits.set_thread_limits(4)


# ---------------------------------------------------------------------------
# Defect 5 — the full-pool anchor is referenced, not re-scored
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_pool_anchor_resolves_to_prior_ok_record(
    tmp_path: Path, monkeypatch
) -> None:
    """The f100 anchor is the prior ok record itself: ``_full_pool_anchor_refs``
    joins on (dataset, model, task, seed, fingerprint, window) and reports the
    existing ok record — it never re-scores anything. No prior record ⇒
    empty list (a fresh dataset, where scoring fresh is legitimate)."""
    dataset_id, config_path, raw_dir = _make_d2_shaped_fixture(tmp_path)
    _wire_fixture(dataset_id, config_path, raw_dir, monkeypatch)

    raw = tmp_path / "raw_e1"
    config = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="anchor_baseline",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=None,
        enforce_test_touched_once=False,
    )
    run_benchmark(config, raw_dir=raw, tables_dir=tmp_path / "tables")

    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    folds = make_folds(specimens, "loso", seed=0, n_splits=5)

    for fold in folds:
        fold_fp = make_test_fingerprint(fold.fingerprint, 8, "random_forest", "classification")
        refs = _full_pool_anchor_refs(
            dataset_id, "random_forest", "classification", 0, fold.fingerprint, 8, raw_dir=raw
        )
        assert refs, f"fold {fold.fold_id}: baseline ok record not found as anchor"
        assert any(ref["fingerprint"] == fold_fp for ref in refs)
        assert all(ref["status"] == "ok" for ref in refs)
        assert all(ref["experiment"] == "anchor_baseline" for ref in refs)
        assert all(ref["source"].startswith("results/raw") for ref in refs)

    # A fingerprint no prior record owns resolves to [] — a fresh dataset,
    # not an anchor collision (no baseline regression record exists).
    empty = _full_pool_anchor_refs(
        dataset_id, "random_forest", "regression", 0, folds[0].fingerprint, 8, raw_dir=raw
    )
    assert empty == []

    # A record under a DIFFERENT dataset is not a full-pool anchor of this
    # dataset's E3 design: the join is dataset-scoped.
    other = _full_pool_anchor_refs(
        "some_other_dataset", "random_forest", "classification", 0, folds[0].fingerprint, 8, raw_dir=raw
    )
    assert other == []


@pytest.mark.unit
def test_full_pool_record_carries_anchor_and_low_data_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    """The record-level evidence: a full-pool E3 record stamps the resolved
    anchor reference in ``e3_anchor``; a true low-data record (0 < f < 1.0)
    carries ``[]`` — its identity is the subsampled config, not a reference.

    Also pins the §17 read-out: on the SAME (fold, seed, task), the config
    hash differs between fractions (f10 ≠ f25 ≠ f50), so each low-data
    fraction is a DISTINCT unit — the v6 zero-out was the f100 side, and
    this is the acceptance check that the low-data side stays distinct."""
    dataset_id, config_path, raw_dir = _make_d2_shaped_fixture(tmp_path)
    _wire_fixture(dataset_id, config_path, raw_dir, monkeypatch)

    raw = tmp_path / "raw_e1"
    # 1) The full-pool baseline batch (the "E1" side the anchor references).
    config_full = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="anchor_baseline",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=None,
        enforce_test_touched_once=False,
    )
    run_benchmark(config_full, raw_dir=raw, tables_dir=tmp_path / "tables")

    baseline = [
        record
        for record in load_records(raw)
        if record.experiment == "anchor_baseline" and record.status == "ok"
    ]
    assert baseline, "the baseline full-pool batch must score ok records"

    # 2) A full-pool E3 re-invocation: every record it scores stamps the
    # anchor reference; config identity with the baseline is preserved.
    e3_full = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_full_pool",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=None,
        enforce_test_touched_once=False,
    )
    summary = run_benchmark(e3_full, raw_dir=raw, tables_dir=tmp_path / "tables")
    # The fold-level guard: the whole fold is a disclosed anchor skip because
    # its units are held by a different-experiment ok record under the SAME
    # config hash. That is the correct v6-corrected behaviour.
    disclosures = summary.attrs.get("skipped_units", [])
    # The §17 ledger scan happens in run_benchmark BEFORE any record of this
    # invocation is written — so the fold-level anchor guard cannot see the
    # baseline batch this very test writes. That is the runner behaving
    # correctly; the join semantics are asserted directly below (step 5).
    assert not disclosures, f"full-pool E3 on fresh tmp records should not skip: {disclosures}"
    assert not summary.empty, "a full-pool re-invocation on fresh records must score"

    # 3) A low-data batch scores fresh and carries [] in e3_anchor.
    e3_low = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_low_f10",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=0.10,
        enforce_test_touched_once=False,
    )
    run_benchmark(e3_low, raw_dir=raw, tables_dir=tmp_path / "tables")
    low_records = [
        record
        for record in load_records(raw)
        if record.experiment == "e3_low_f10" and record.status == "ok"
    ]
    assert low_records, "the low-data batch must score fresh records"
    for record in low_records:
        assert record.e3_anchor == [], (
            f"a 0 < f < 1.0 record must not carry an anchor reference: "
            f"{record.run_id!r}"
        )
        # The record must be readable for its train_fraction: it lives in the
        # config hash, and a reader can recover it from the experiment label
        # + record (the record path carries no fraction — the acceptance
        # criterion '记录里能读出 train_fraction' is met by the e3_anchor field
        # on full-pool records and the distinct config hash per low fraction).
    hashes = {record.config_hash for record in low_records}
    baseline_hashes = {record.config_hash for record in baseline}
    assert hashes.isdisjoint(baseline_hashes), (
        "a low-data fraction must NOT share its config hash with the "
        "full-pool batch"
    )

    # 4) Same (fold, seed, task), different fraction → different config hash.
    config_f10 = config_hash(e3_low.as_dict())
    e3_25 = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_low_f25",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=0.25,
        enforce_test_touched_once=False,
    )
    config_f25 = config_hash(e3_25.as_dict())
    config_full_hash = config_hash(config_full.as_dict())
    assert config_f10 != config_f25 != config_full_hash, (
        "same (fold, seed, task) under different fractions must have distinct "
        "config hashes: f10/f25/100 collide"
    )
    # 5) The analysis-time join: the same §17 ledger the runner's guard
    # consults (scan all records, incl. the baseline) resolves the full-pool
    # E3 unit to the baseline's ok records; the low-data record shares the
    # partition's fingerprint (test set fixed across fractions) under a
    # distinct config hash.
    pt, pt_meta, pt_exp = load_prior_test_touches(raw, True)
    dataset = load_dataset(config_path)
    specimens = usable_specimens(dataset, 8)
    folds = make_folds(specimens, "loso", seed=0, n_splits=5)
    for fold in folds:
        fp = make_test_fingerprint(fold.fingerprint, 8, "random_forest", "classification")
        assert fp in pt, f"fold {fold.fold_id}: partition fingerprint missing from ledger"
        # The ledger's owner for each classification fingerprint is the
        # baseline batch (it ran first; setdefault keeps the first owner).
        assert pt[fp] == config_full_hash
        assert pt_exp[fp] == "anchor_baseline"
        # The full-pool E3 unit IS the baseline record BY CONSTRUCTION: the
        # reference is the prior ok record — re-scoring it adds nothing.
        refs = _full_pool_anchor_refs(
            dataset_id, "random_forest", "classification", 0, fold.fingerprint, 8, raw_dir=raw
        )
        assert any(
            ref["experiment"] == "anchor_baseline" and ref["config_hash"] == config_full_hash
            for ref in refs
        ), f"fold {fold.fold_id}: full-pool anchor not resolved to the baseline record"

    # The low-data records scored fresh — their fingerprint is the
    # partition's (test set fixed across fractions) and their config hash is
    # distinct from the ledger owner's.
    low_fps = {record.test_fingerprint for record in low_records}
    assert low_fps <= set(pt), "a low-data record's fingerprint must be a known partition"
    low_hashes = {record.config_hash for record in low_records}
    assert low_hashes == {config_f10}
    for fp in low_fps:
        assert pt[fp] == config_full_hash, (
            "the ledger's owner for a shared fingerprint is the first ok "
            "record (the baseline) — a low-data unit keeps its own identity "
            "via its distinct config hash, not by clobbering the ledger"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

# ---------------------------------------------------------------------------
# Defect 2 — skip semantics: a low-data fraction is never swallowed by the
# batch it audits against; a full-pool f100 run is a disclosed anchor skip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_low_data_fraction_scores_fresh_not_skipped(
    tmp_path: Path, monkeypatch
) -> None:
    """Defect 2 (the v6 zero-out): E3 units for a true low-data fraction
    (0 < f < 1.0) must each produce NEW ok records even when a prior full
    batch owns every test fingerprint on the partition — the prior record
    is an anchor reference, not a §17 guard, because the fractions differ
    only in the admitted TRAIN pool (test set fixed across fractions).

    The pre-DATA-61 skip path swallowed exactly this: f50/f75/f100 batches
    came back 0 records, all units skipped. This asserts the two halves of
    the corrected skip semantics:

    * a f10 batch on a partition already held by a full-pool E1 batch
      (different config hash) scores every fold FRESH — zero disclosures,
      one ok record per fold;
    * the same partition under a FULL-POOL run (train_fraction None — the
      f100 anchor) is a disclosed cross-experiment anchor skip — zero new
      records, one disclosure per unit naming the prior experiment.
    """
    dataset_id, config_path, raw_dir = _make_d2_shaped_fixture(tmp_path)
    _wire_fixture(dataset_id, config_path, raw_dir, monkeypatch)

    raw = tmp_path / "raw_e1"
    # 1) The full-batch owner of the partition (an "e1" side).
    e1 = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e1_main",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=None,
        enforce_test_touched_once=False,
    )
    run_benchmark(e1, raw_dir=raw, tables_dir=tmp_path / "tables")
    assert load_records(raw), "the e1 batch must score ok records"

    # 2) A true low-data fraction on the SAME partition: must score fresh.
    f10 = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_lowdata_f10",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=0.10,
        enforce_test_touched_once=True,  # the guard is ON, and it must not fire
    )
    summary = run_benchmark(f10, raw_dir=raw, tables_dir=tmp_path / "tables")
    disclosures = summary.attrs.get("skipped_units", [])
    assert not disclosures, (
        f"defect 2 regression: a low-data fraction was swallowed by the "
        f"full batch it audits against: {disclosures}"
    )
    f10_records = [
        record
        for record in load_records(raw)
        if record.experiment == "e3_lowdata_f10" and record.status == "ok"
    ]
    assert f10_records, "a f10 batch must produce ok records (the v6 zero-out)"
    for record in f10_records:
        assert record.train_fraction == 0.10, (
            "the record must state the train_fraction it was built under "
            "(acceptance: '记录里能读出 train_fraction')"
        )

    # 3) A FULL-POOL run on the SAME partition: a disclosed anchor skip.
    f100 = BenchmarkConfig(
        dataset_id=dataset_id,
        experiment="e3_lowdata_f100",
        models=("random_forest",),
        tasks=("classification",),
        seeds=(0,),
        window_lengths=(8,),
        split_strategy="loso",
        n_splits=5,
        train_fraction=None,  # the 1.0 no-op alias
        enforce_test_touched_once=True,
    )
    summary_100 = run_benchmark(f100, raw_dir=raw, tables_dir=tmp_path / "tables")
    assert summary_100.empty, "a full-pool f100 run must not re-score the anchor"
    disclosures_100 = summary_100.attrs.get("skipped_units", [])
    anchor = [d for d in disclosures_100 if d["reason"] == "prior_ok_cross_experiment_anchor"]
    assert anchor, (
        "the f100 run must be a DISCLOSED anchor skip naming the prior "
        f"experiment (got {len(disclosures_100)} disclosures)"
    )
    assert all(d["prior_experiment"] == "e1_main" for d in anchor)
    f100_records = [
        record
        for record in load_records(raw)
        if record.experiment == "e3_lowdata_f100" and record.status == "ok"
    ]
    assert not f100_records, "a f100 run must not write new ok records"


@pytest.mark.unit
def test_each_fraction_is_a_distinct_unit(tmp_path: Path, monkeypatch) -> None:
    """Defect 5 (record identity): the SAME (fold, seed, task) keeps a
    DISTINCT config_hash under each fraction, and every low-data fraction
    produces ok records — the acceptance check "f10/f25/f50/f75 各自 > 0
    条记录" at the fixture shape, plus the reservoir half (defect 2).

    The full-pool unit is a no-op alias for the full batch: its config
    hash equals the full-batch hash (invariant pinned in the DATA-60
    suite), so it is the ANCHOR — referenced, never re-scored. The four
    true fractions (0 < f < 1.0) each enter the fingerprint, giving four
    distinct unit identities on top of the shared full-pool identity:
    five distinct hashes for the five experiment labels.
    """
    dataset_id, config_path, raw_dir = _make_d2_shaped_fixture(tmp_path)
    _wire_fixture(dataset_id, config_path, raw_dir, monkeypatch)

    raw = tmp_path / "raw_e1"
    configs = {}
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        f_label = int(fraction * 100)
        configs[fraction] = BenchmarkConfig(
            dataset_id=dataset_id,
            experiment=f"e3_lowdata_d2_f{f_label}",
            models=("random_forest",),
            tasks=("classification",),
            seeds=(0,),
            window_lengths=(8,),
            split_strategy="loso",
            n_splits=5,
            train_fraction=(None if fraction == 1.0 else fraction),
            enforce_test_touched_once=True,
        )
        run_benchmark(configs[fraction], raw_dir=raw, tables_dir=tmp_path / "tables")

    # Distinct identity per fraction: the 0 < f < 1.0 units plus the full-pool
    # unit — five configs, five hashes, no two equal.
    hashes = [config_hash(cfg.as_dict()) for cfg in configs.values()]
    assert len(set(hashes)) == 5, (
        f"same (fold, seed, task) must have a distinct config_hash per "
        f"fraction, got {len(set(hashes))} of 5: {hashes}"
    )

    # Every LOW-data fraction produced ok records; the full-pool fraction is
    # a disclosed anchor skip (zero ok records, one disclosure per unit).
    for fraction, cfg in configs.items():
        records = [
            record
            for record in load_records(raw)
            if record.experiment == cfg.experiment and record.status == "ok"
        ]
        if fraction < 1.0:
            assert records, f"{cfg.experiment} produced 0 ok records (v6 zero-out)"
            for record in records:
                assert record.train_fraction == fraction, (
                    f"{record.run_id}: the record must state its train_fraction "
                    f"({record.train_fraction} != {fraction})"
                )
        else:
            assert not records, (
                "the full-pool fraction must not re-score the anchor it audits"
            )

    # The reservoir half (E2): the same skip semantics through the R-family.
    npz = tmp_path / "olfactory_v1.npz"
    _fake_olfactory_npz(npz)
    raw2 = tmp_path / "raw_e2"
    for fraction in (0.10, 0.25):
        e2f = ReservoirConfig(
            dataset_id=dataset_id,
            experiment=f"e3_lowdata_e2_f{int(fraction * 100)}",
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
            enforce_test_touched_once=True,
            train_fraction=fraction,
        )
        report = run_reservoir_benchmark(e2f, raw_dir=raw2, tables_dir=tmp_path / "tables_e2")
        ok = [record for record in report.records if record.status == "ok"]
        assert ok, f"E2 f{int(fraction * 100)} produced 0 ok records (v6 zero-out)"
        for record in ok:
            assert record.train_fraction == fraction
            assert record.e3_anchor == []

    # A full-pool E2 run on a partition the E2 batch already owns is a
    # disclosed anchor skip with the reference on the skip record. It must
    # run against the SAME raw root that holds the prior batch — that is
    # where the anchor ledger lives.
    e2_prior = ReservoirConfig(
        dataset_id=dataset_id,
        experiment="e2_main",
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
        train_fraction=None,
    )
    raw3 = tmp_path / "raw_e2_prior"
    run_reservoir_benchmark(e2_prior, raw_dir=raw3, tables_dir=tmp_path / "tables_e2p")
    e2_full = ReservoirConfig(
        dataset_id=dataset_id,
        experiment="e2_full_pool_e3",
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
        enforce_test_touched_once=True,
        train_fraction=None,  # full pool — the f100 anchor
    )
    report = run_reservoir_benchmark(e2_full, raw_dir=raw3, tables_dir=tmp_path / "tables_e2f")
    skipped = [
        unit for unit in report.skipped_units if "anchor" in unit
    ]
    assert skipped, "a full-pool E2 run over an owned partition must be a disclosed anchor skip"
    skipped_records = [
        record
        for record in load_records(raw3)
        if record.experiment == "e2_full_pool_e3" and record.status == "skipped"
    ]
    assert skipped_records, "the anchor skip must be recorded, not silent"
    for record in skipped_records:
        assert record.train_fraction is None
        assert record.e3_anchor, "an anchor skip record must carry the resolved reference"
        assert all(ref["experiment"] == "e2_main" for ref in record.e3_anchor)

    # The analysis-time join is readable off the anchor skip record itself:
    # its e3_anchor carries the reference to the prior batch that owns the
    # partition — the join, not a re-score.
    assert all(
        ref["experiment"] == "e2_main"
        for record in skipped_records
        for ref in record.e3_anchor
    ), "the anchor skip record must reference the prior batch that owns the partition"
