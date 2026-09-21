"""The declared seed set is enforced at the runner entry and at aggregation.

``seeds.root_seeds``, ``primary_seed_count`` and ``extension_to`` were declared
and read by nothing: the runner took seeds from its arguments, so a run outside
0..9 was accepted as readily as one inside it. No delivered artifact used a seed
outside the declared set — this is preventive wiring, not a repair — and the
tests say so by checking the delivered seed sets as well as the rule.

The extension rule has two halves and they fail differently:

* ``seeds 10..19 may be added only as a whole block`` — a partial extension is
  refused at the runner entry, where the run's seed set is known.
* ``results computed over 10 and over 20 seeds are reported separately and never
  pooled`` — pooling happens at aggregation, so the check lives there too.
"""

from __future__ import annotations

import pandas as pd
import pytest

from drososense.evaluation.results import (
    RunRecord,
    aggregate_records,
    assert_seed_blocks_are_not_pooled,
)
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark
from drososense.utils.config import load_protocol
from drososense.utils.paths import RESULTS_TABLES_DIR
from drososense.utils.seeding import EXTENDED_BLOCK, PRIMARY_BLOCK, SeedPolicy


@pytest.fixture(scope="module")
def policy() -> SeedPolicy:
    return SeedPolicy.from_protocol()


# ---------------------------------------------------------------------------
# The declared set
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_policy_reads_the_protocol(policy):
    declared = load_protocol()["seeds"]
    assert list(policy.root_seeds) == declared["root_seeds"]
    assert policy.primary_seed_count == declared["primary_seed_count"]
    assert policy.extension_to == declared["extension_to"]
    assert policy.extension_block == tuple(range(10, 20))
    assert len(policy.extended_seeds) == 20


@pytest.mark.unit
def test_a_run_inside_the_declared_set_is_accepted(policy):
    """A subset of the root seeds is the ordinary case, not a violation."""
    policy.validate([0])
    policy.validate([0, 1])
    policy.validate(list(range(10)))
    policy.validate(list(range(20)))
    assert policy.classify([0]) == PRIMARY_BLOCK
    assert policy.classify(list(range(10))) == PRIMARY_BLOCK
    assert policy.classify(list(range(20))) == EXTENDED_BLOCK


@pytest.mark.unit
def test_a_seed_outside_the_declared_set_is_refused(policy):
    with pytest.raises(ValueError, match="outside the declared seed set"):
        policy.validate([0, 1, 10_000])
    with pytest.raises(ValueError, match="outside the declared seed set"):
        policy.validate([-1])
    with pytest.raises(ValueError, match="outside the declared seed set"):
        policy.validate([20])


@pytest.mark.unit
def test_the_extension_block_is_all_or_nothing(policy):
    """10..19 join whole, and only on top of the full primary set."""
    with pytest.raises(ValueError, match="all-or-nothing"):
        policy.validate([10])
    with pytest.raises(ValueError, match="all-or-nothing"):
        policy.validate(list(range(10)) + [15])
    with pytest.raises(ValueError, match="all-or-nothing"):
        policy.validate([0, 1] + list(range(10, 20)))
    with pytest.raises(ValueError, match="all-or-nothing"):
        policy.validate(list(range(18)))


@pytest.mark.unit
def test_an_empty_or_repeated_seed_set_is_refused(policy):
    with pytest.raises(ValueError, match="no seeds requested"):
        policy.validate([])
    with pytest.raises(ValueError, match="repeats a seed"):
        policy.validate([0, 0, 1])


@pytest.mark.unit
def test_a_self_contradicting_seed_block_is_refused():
    """The two statements of the same thing have to agree, or neither is trusted."""
    protocol = load_protocol()
    contradictory = {
        **protocol,
        "seeds": {**protocol["seeds"], "primary_seed_count": 8},
    }
    with pytest.raises(ValueError, match="primary_seed_count is 8 but .* holds 10"):
        SeedPolicy.from_protocol(contradictory)

    vacuous = {**protocol, "seeds": {**protocol["seeds"], "extension_to": 10}}
    with pytest.raises(ValueError, match="does not extend"):
        SeedPolicy.from_protocol(vacuous)


# ---------------------------------------------------------------------------
# The delivered artifacts
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["m1_benchmark_per_run.csv", "m1_real_validation_per_run.csv", "smoke_per_run.csv"],
)
def test_no_delivered_artifact_used_an_undeclared_seed(name, policy):
    """The claim that this is preventive, checked rather than asserted.

    These are the committed per-run tables. Every seed in them has to be one the
    protocol declares — otherwise the wiring is a repair, and the review note
    saying no delivered result deviated would be wrong.
    """
    path = RESULTS_TABLES_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} is not committed")
    seeds = sorted({int(value) for value in pd.read_csv(path)["seed"]})
    assert seeds, f"{name} records no seed at all"
    policy.validate(seeds)
    assert not set(seeds) & set(policy.extension_block), (
        f"{name} used the extension block; the delivered results are a 10-seed design"
    )


# ---------------------------------------------------------------------------
# The two enforcement points
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_runner_refuses_a_seed_outside_the_declared_set(temporary_dataset, tmp_path):
    """The failure the hole named: an argument outside 0..9 used to be accepted.

    Seed 11 belongs to the extension block and is refused for being taken partly;
    this is the seed that is outside the declared set altogether.
    """
    dataset_id, _ = temporary_dataset
    with pytest.raises(ValueError, match="outside the declared seed set"):
        run_benchmark(
            BenchmarkConfig(
                dataset_id=dataset_id, models=("svm_rbf",), tasks=("classification",),
                seeds=(11_000,), window_lengths=(8,), n_splits=3, max_folds=1,
            ),
            raw_dir=tmp_path / "raw", tables_dir=tmp_path / "tables",
        )


@pytest.mark.unit
def test_the_runner_refuses_a_partial_extension(temporary_dataset, tmp_path):
    dataset_id, _ = temporary_dataset
    with pytest.raises(ValueError, match="all-or-nothing"):
        run_benchmark(
            BenchmarkConfig(
                dataset_id=dataset_id, models=("svm_rbf",), tasks=("classification",),
                seeds=tuple(range(10)) + (15,), window_lengths=(8,), n_splits=3, max_folds=1,
            ),
            raw_dir=tmp_path / "raw", tables_dir=tmp_path / "tables",
        )


def _record(seed: int, config_hash: str, model: str = "svm_rbf") -> RunRecord:
    """Build a minimal record, enough to aggregate.

    Args:
        seed: The run's seed.
        config_hash: Fingerprint of the invocation that produced it.
        model: Model id.

    Returns:
        The record.
    """
    return RunRecord(
        run_id=f"{model}_seed{seed}",
        experiment="seeds_test",
        dataset="unit_fixture",
        model=model,
        task="classification",
        seed=seed,
        fold_id=0,
        protocol_version="1.1.0",
        window_length=8,
        metrics={"macro_f1": 0.5},
        n_train_windows=10,
        n_test_windows=4,
        train_specimens=["a"],
        test_specimens=["b"],
        duration_s=0.1,
        environment={},
        timestamp_utc="2026-01-01T00:00:00Z",
        config_hash=config_hash,
    )


@pytest.mark.unit
def test_the_two_seed_designs_are_not_pooled():
    """`seeds.extension_rule`: 10-seed and 20-seed results are reported apart.

    The two designs are distinguished by the configuration hash of the run that
    produced them, which is what a re-invocation with a different seed set
    changes. Pooling them into one mean would describe neither design.
    """
    ten = [_record(seed, "hash10") for seed in range(10)]
    twenty = [_record(seed, "hash20") for seed in range(20)]

    assert_seed_blocks_are_not_pooled(ten)
    assert_seed_blocks_are_not_pooled(twenty)
    assert not aggregate_records(ten).empty
    assert not aggregate_records(twenty).empty

    with pytest.raises(ValueError, match="pool 2 seed designs"):
        assert_seed_blocks_are_not_pooled(ten + twenty)
    with pytest.raises(ValueError, match="pool 2 seed designs"):
        aggregate_records(ten + twenty)


@pytest.mark.unit
def test_a_partial_extension_never_reaches_aggregation():
    """The block rule is re-checked here, because records can arrive without the runner."""
    with pytest.raises(ValueError, match="all-or-nothing"):
        assert_seed_blocks_are_not_pooled(
            [_record(seed, "hash_partial") for seed in list(range(10)) + [15]]
        )


@pytest.mark.unit
def test_a_seed_outside_the_declared_set_never_reaches_aggregation():
    with pytest.raises(ValueError, match="outside the declared seed set"):
        assert_seed_blocks_are_not_pooled([_record(0, "h"), _record(99, "h")])
