"""DATA-61 — fold_train_pool regression tests.

The pre-DATA-61 global-pool sampling in the E3 low-data runner could admit a
specimen that is the current fold's test (or val) specimen, emptying the
fold's train side and recording every unit of that fold as a failed run.
The D2 f10 batch (4 specimens, LOSO, f=0.10 ⇒ pool size 1) was 100% failed
for every seed — the batch that this issue closes.

``fold_train_pool`` (drososense.data.splits) makes ``pool ⊆ fold.train``
the invariant, so the admitted pool is always a subset of the fold's own
TRAIN side and never empty. These tests pin the regression.
"""

from __future__ import annotations

import pytest

from drososense.data.splits import (
    Fold,
    fold_train_pool,
    group_kfold,
    loso,
    nested_train_fraction,
)


# ---------------------------------------------------------------------------
# Invariant 1: pool ⊆ fold.train (no test/val leakage)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fold_train_pool_restricts_to_train_side():
    """The pool never admits a test or val specimen — the bug from DATA-60."""
    specimens = [f"sp{i:03d}" for i in range(10)]
    for seed in (0, 1, 2):
        folds = loso(specimens, seed=seed)
        for f in (0.10, 0.25, 0.50, 0.75):
            for fold in folds:
                admitted = fold_train_pool(fold, f)
                admitted_set = set(admitted)
                assert admitted_set <= set(fold.train), (
                    f"fold {fold.fold_id}: admitted {sorted(admitted_set)} not a subset of "
                    f"train {sorted(set(fold.train))}"
                )
                assert not (admitted_set & set(fold.test)), (
                    f"fold {fold.fold_id}: test specimen in admitted train pool"
                )
                assert not (admitted_set & set(fold.val)), (
                    f"fold {fold.fold_id}: val specimen in admitted train pool"
                )
                assert admitted, f"fold {fold.fold_id}: admitted pool is empty at f={f}"


@pytest.mark.unit
def test_fold_train_pool_group_kfold_restricts_to_train_side():
    """Same invariant for group_kfold (not just LOSO)."""
    specimens = [f"sp{i:03d}" for i in range(12)]
    folds = group_kfold(specimens, n_splits=5, seed=3)
    for f in (0.10, 0.25, 0.50, 0.75):
        for fold in folds:
            admitted = fold_train_pool(fold, f)
            assert set(admitted) <= set(fold.train)
            assert not (set(admitted) & (set(fold.test) | set(fold.val)))
            assert admitted, f"fold {fold.fold_id} f={f}: empty pool"


# ---------------------------------------------------------------------------
# Invariant 2: nesting within a fold (f=0.10 ⊆ 0.25 ⊆ 0.50 ⊆ 0.75 ⊆ 1.0)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fold_train_pool_nesting_per_fold():
    """Within a fold, the pool at each fraction is a nested prefix chain."""
    specimens = [f"sp{i:03d}" for i in range(10)]
    for seed in (0, 1):
        folds = loso(specimens, seed=seed)
        for fold in folds:
            admitted: list[set[str]] = []
            for f in (0.10, 0.25, 0.50, 0.75, 1.00):
                admitted.append(set(fold_train_pool(fold, f)))
            for smaller, bigger in zip(admitted, admitted[1:]):
                assert smaller <= bigger, (
                    f"fold {fold.fold_id}: pool not nested between consecutive "
                    f"fractions: {sorted(smaller - bigger)} missing at higher fraction"
                )
            # 100% is the full train side, byte-identical to E1/E2
            assert admitted[-1] == set(fold.train)


# ---------------------------------------------------------------------------
# Invariant 3: determinism (same fold + fraction → same tuple, no hidden RNG)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fold_train_pool_deterministic():
    """Same (fold, fraction) → same result on repeated calls (no hidden state)."""
    fold = Fold(
        fold_id=0,
        strategy="manual",
        train=("b", "d", "f", "h", "j"),
        val=("c",),
        test=("e",),
        seed=0,
    )
    p10 = fold_train_pool(fold, 0.10)
    p25 = fold_train_pool(fold, 0.25)
    p50 = fold_train_pool(fold, 0.50)
    # Sizes: ceil(f * n_train)
    assert len(p10) == 1  # ceil(0.10 * 5) = 1
    assert len(p25) == 2  # ceil(0.25 * 5) = 2
    assert len(p50) == 3  # ceil(0.50 * 5) = 3
    # Nested
    assert set(p10) <= set(p25) <= set(p50) <= set(fold.train)
    # Deterministic: same input → same output
    assert p10 == fold_train_pool(fold, 0.10)
    assert p25 == fold_train_pool(fold, 0.25)


@pytest.mark.unit
def test_fold_train_pool_deterministic_for_seed():
    """Same fold object, repeated calls → same result (no hidden RNG state)."""
    specimens = [f"sp{i:03d}" for i in range(10)]
    folds = loso(specimens, seed=42)
    for fold in folds[:3]:
        results = [fold_train_pool(fold, 0.25) for _ in range(3)]
        assert len(set(results)) == 1, (
            f"fold {fold.fold_id}: fold_train_pool not deterministic: {results}"
        )


# ---------------------------------------------------------------------------
# Invariant 4: 1.0 is the full train side (byte-for-byte E1/E2 anchor)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fold_train_pool_full_fraction_equals_train():
    """At fraction=1.0 the admitted pool is the entire fold.train."""
    specimens = [f"sp{i:03d}" for i in range(10)]
    folds = loso(specimens, seed=0)
    for fold in folds:
        admitted = fold_train_pool(fold, 1.0)
        assert set(admitted) == set(fold.train)
        assert len(admitted) == len(fold.train)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fold_train_pool_raises_on_bad_fraction():
    """A fraction outside (0, 1] raises a named error."""
    fold = Fold(
        fold_id=0,
        strategy="manual",
        train=("a", "b"),
        val=("c",),
        test=("d",),
        seed=0,
    )
    with pytest.raises(ValueError, match="train fraction must satisfy"):
        fold_train_pool(fold, 0.0)
    with pytest.raises(ValueError, match="train fraction must satisfy"):
        fold_train_pool(fold, 1.5)


# ---------------------------------------------------------------------------
# DATA-61 regression: the old global-pool bug
# ---------------------------------------------------------------------------


def test_old_global_pool_emptied_folds_in_d2_fixture():
    """The pre-DATA-61 global-pool sampling emptied a fold's train side when
    the (single) admitted specimen was that fold's test or val specimen.

    On a D2-like 4-specimen LOSO fixture at f=0.10 the global pool is a
    single specimen. For every seed that specimen is exactly one fold's
    test specimen (LOSO fold 0) and one fold's val specimen (LOSO fold 3,
    wrapping), so the old code recorded 2 of 4 folds as failed for EVERY
    seed — the 100%-failed DATA-60 batch.

    This test asserts that property explicitly, and that the new
    fold_train_pool succeeds on exactly those folds.
    """
    specimens = [f"sp{i:03d}" for i in range(4)]
    for seed in range(12):
        folds = loso(specimens, seed=seed, n_splits=4)
        global_pool = nested_train_fraction(specimens, 0.10, seed=seed)
        assert len(global_pool) == 1  # ceil(0.10 * 4) = 1

        # Old code behaviour: fold.train ∩ global_pool == ∅ ⇒ train side
        # emptied ⇒ tensor builder raises ⇒ unit recorded as failed.
        old_failed_folds = [
            fold.fold_id
            for fold in folds
            if not (set(global_pool) & set(fold.train))
        ]
        # The DATA-60 symptom: the old code failed at least one fold for
        # every seed (folds 0 and 3 — test and val of the same permutation
        # position — always lose the single admitted specimen).
        assert old_failed_folds, (
            f"seed {seed}: no fold had an empty train side under the old "
            f"global-pool code — this fixture does not reproduce the bug"
        )
        # New code: fold_train_pool never returns empty for these folds.
        for fold in folds:
            admitted = fold_train_pool(fold, 0.10)
            assert admitted, (
                f"fold {fold.fold_id} seed {seed}: new pool still empty "
                f"(train={list(fold.train)}, global_pool={list(global_pool)})"
            )
            assert set(admitted) <= set(fold.train)


def test_fold_train_pool_never_empties_train_side():
    """Sweep: for many (n, seed, fraction) combinations, the new pool is
    always non-empty whenever fold.train is non-empty."""
    for n in (4, 5, 6, 8, 10, 12):
        specimens = [f"sp{i:03d}" for i in range(n)]
        for seed in range(5):
            folds = loso(specimens, seed=seed, n_splits=n)
            for f in (0.10, 0.25, 0.50, 0.75):
                for fold in folds:
                    admitted = fold_train_pool(fold, f)
                    assert admitted, (
                        f"n={n} seed={seed} f={f} fold {fold.fold_id}: "
                        f"admitted pool empty"
                    )
                    assert set(admitted) <= set(fold.train)
