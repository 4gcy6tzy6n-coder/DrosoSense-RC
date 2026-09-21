"""Specimen-level split invariants.

The frozen protocol's red line is ``split_unit: specimen``. These tests assert
the partition properties directly, and assert that the constructors refuse to
produce a fold that violates them — an invalid split should be impossible to
build, not merely unlikely.
"""

from __future__ import annotations

import pytest

from drososense.data.leakage import LeakageError, audit_fold
from drososense.data.splits import (
    Fold,
    group_kfold,
    loso,
    make_folds,
    permute_specimens,
    time_block_holdout,
)

SPECIMENS = [f"sp{i:03d}" for i in range(12)]


@pytest.mark.unit
def test_group_kfold_partitions_specimens_exactly():
    """Every specimen lands in exactly one split, once."""
    for fold in group_kfold(SPECIMENS, n_splits=5, seed=0):
        union = set(fold.train) | set(fold.val) | set(fold.test)
        assert union == set(SPECIMENS)
        assert len(fold.train) + len(fold.val) + len(fold.test) == len(SPECIMENS)


@pytest.mark.unit
def test_group_kfold_has_no_specimen_overlap():
    """No specimen appears in two splits — the core protocol requirement."""
    for fold in group_kfold(SPECIMENS, n_splits=5, seed=3):
        assert not set(fold.train) & set(fold.val)
        assert not set(fold.train) & set(fold.test)
        assert not set(fold.val) & set(fold.test)
        audit_fold(fold, SPECIMENS)


@pytest.mark.unit
def test_group_kfold_is_deterministic_for_a_seed():
    """The same seed reproduces the identical partition."""
    first = group_kfold(SPECIMENS, n_splits=4, seed=11)
    second = group_kfold(SPECIMENS, n_splits=4, seed=11)
    assert [f.fingerprint for f in first] == [f.fingerprint for f in second]


@pytest.mark.unit
def test_group_kfold_differs_across_seeds():
    """Different seeds produce different partitions, so seeds are a real variance source."""
    a = [f.fingerprint for f in group_kfold(SPECIMENS, n_splits=4, seed=0)]
    b = [f.fingerprint for f in group_kfold(SPECIMENS, n_splits=4, seed=1)]
    assert a != b


@pytest.mark.unit
def test_loso_holds_out_each_specimen_once():
    """LOSO covers every specimen as test exactly once."""
    folds = loso(SPECIMENS, seed=0)
    tested = [s for fold in folds for s in fold.test]
    assert sorted(tested) == sorted(SPECIMENS)
    assert len(tested) == len(set(tested))
    for fold in folds:
        assert len(fold.test) == 1
        assert not set(fold.train) & set(fold.test)
        assert not set(fold.train) & set(fold.val)


@pytest.mark.unit
def test_validation_is_disjoint_from_test_in_every_strategy():
    """A validation specimen must never also be a test specimen."""
    for strategy in ("group_kfold", "loso"):
        for fold in make_folds(SPECIMENS, strategy, seed=2, n_splits=4):
            assert not set(fold.val) & set(fold.test)
            assert fold.val, f"{strategy}: validation must not be empty"


@pytest.mark.unit
def test_permutation_is_a_bijection():
    """Permuting preserves the specimen multiset exactly."""
    permuted = permute_specimens(SPECIMENS, seed=5)
    assert sorted(permuted) == sorted(SPECIMENS)
    assert len(set(permuted)) == len(SPECIMENS)


@pytest.mark.unit
@pytest.mark.parametrize(
    "strategy, kwargs",
    [
        ("group_kfold", {"n_splits": 5}),
        ("loso", {}),
        ("time_block_holdout", {"n_splits": 3}),
    ],
)
def test_too_few_specimens_is_rejected(strategy, kwargs):
    """Strategies refuse to build a split they cannot support."""
    with pytest.raises(ValueError):
        make_folds(["only_one"], strategy, seed=0, **kwargs)


@pytest.mark.unit
def test_unknown_strategy_is_rejected():
    """An unknown strategy name fails loudly rather than defaulting."""
    with pytest.raises(ValueError, match="unknown split strategy"):
        make_folds(SPECIMENS, "random_rows", seed=0)


@pytest.mark.unit
def test_fold_constructor_rejects_overlapping_splits():
    """A leaky fold cannot be constructed at all."""
    with pytest.raises(ValueError, match="both train and test"):
        Fold(fold_id=0, strategy="manual", train=("a", "b"), val=("c",), test=("b",), seed=0)


@pytest.mark.unit
def test_fold_constructor_rejects_empty_test():
    """A fold without a test set is a mistake, not a configuration."""
    with pytest.raises(ValueError, match="test is empty"):
        Fold(fold_id=0, strategy="manual", train=("a", "b"), val=(), test=(), seed=0)


@pytest.mark.unit
def test_audit_fold_detects_a_partition_that_loses_a_specimen():
    """An audit against the full specimen list catches a shrunken partition."""
    fold = Fold(fold_id=0, strategy="manual", train=("a", "b"), val=("c",), test=("d",), seed=0)
    with pytest.raises(LeakageError, match="does not cover"):
        audit_fold(fold, ["a", "b", "c", "d", "e"])


@pytest.mark.unit
def test_fold_split_of_labels_membership():
    """``split_of`` reports the owning split and rejects strangers."""
    fold = Fold(fold_id=0, strategy="manual", train=("a",), val=("b",), test=("c",), seed=0)
    assert fold.split_of("a") == "train"
    assert fold.split_of("b") == "val"
    assert fold.split_of("c") == "test"
    with pytest.raises(KeyError):
        fold.split_of("z")


@pytest.mark.unit
def test_no_row_level_splitter_is_importable_from_the_module():
    """The data layer never references a row-level splitter, by design.

    Checked against the parsed syntax tree rather than the raw text, so that a
    docstring which *names* ``train_test_split`` in order to forbid it does not
    register as a use of it. Column numbers are not inspected.
    """
    import ast

    import drososense.data.splits as splits_module

    source_path = splits_module.__file__
    assert source_path
    tree = ast.parse(open(source_path, encoding="utf-8").read())

    referenced: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.append(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.append(node.attr)
        elif isinstance(node, ast.alias):
            referenced.append(node.name.split(".")[-1])

    assert "train_test_split" not in referenced
    assert not hasattr(splits_module, "train_test_split")
