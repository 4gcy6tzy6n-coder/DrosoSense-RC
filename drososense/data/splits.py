"""Specimen-level splitting.

The frozen protocol (``split_unit: specimen``) forbids any split in which a
specimen contributes rows to more than one of train / validation / test.
Every function here partitions *specimens*, and a :class:`Fold` refuses to
exist if the three sets are not a disjoint partition.

Row-level splitting is deliberately not implemented — there is no function in
this module that accepts rows. ``sklearn.model_selection.train_test_split`` is
never imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

SplitStrategy = Literal["group_kfold", "loso", "time_block_holdout"]

# Minimum specimens for a strategy to yield a usable validation set.
_MIN_SPECIMENS_GROUP_KFOLD = 5
_MIN_SPECIMENS_LOSO = 3


@dataclass(frozen=True)
class Fold:
    """One train/validation/test partition over specimens.

    Attributes:
        fold_id: Index of this fold within its strategy.
        strategy: Name of the strategy that produced the fold.
        train: Specimens used for fitting the scaler, the model and the readout.
        val: Specimens used for hyperparameter and window-length selection.
        test: Specimens touched exactly once, for the reported metric.
        seed: Seed that produced the underlying specimen permutation.
    """

    fold_id: int
    strategy: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for name, members in sets.items():
            if len(members) != len(getattr(self, name)):
                raise ValueError(f"fold {self.fold_id}: duplicate specimen in {name}")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(
                    f"fold {self.fold_id}: specimens in both {a} and {b}: {sorted(overlap)}"
                )
        if not sets["train"]:
            raise ValueError(f"fold {self.fold_id}: train is empty")
        if not sets["test"]:
            raise ValueError(f"fold {self.fold_id}: test is empty")

    @property
    def specimens(self) -> tuple[str, ...]:
        """Every specimen in the fold, sorted."""
        return tuple(sorted(set(self.train) | set(self.val) | set(self.test)))

    @property
    def fingerprint(self) -> str:
        """Short deterministic identifier of this exact partition."""
        payload = f"{self.strategy}|{self.seed}|{'/'.join(sorted(self.train))}"
        payload += f"|{'/'.join(sorted(self.val))}|{'/'.join(sorted(self.test))}"
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def split_of(self, specimen: str) -> str:
        """Return which split a specimen belongs to.

        Args:
            specimen: Specimen identifier.

        Returns:
            ``"train"``, ``"val"`` or ``"test"``.

        Raises:
            KeyError: If the specimen is not part of this fold.
        """
        if specimen in set(self.train):
            return "train"
        if specimen in set(self.val):
            return "val"
        if specimen in set(self.test):
            return "test"
        raise KeyError(f"specimen {specimen!r} is not part of fold {self.fold_id}")


def _as_unique_sorted(specimens: Iterable[str]) -> tuple[str, ...]:
    """Normalise a specimen collection to a sorted, duplicate-free tuple.

    Args:
        specimens: Any iterable of specimen identifiers.

    Returns:
        Sorted tuple of unique identifiers.

    Raises:
        ValueError: If the collection is empty.
    """
    unique = sorted({str(s) for s in specimens})
    if not unique:
        raise ValueError("no specimens supplied")
    return tuple(unique)


def permute_specimens(specimens: Iterable[str], seed: int) -> tuple[str, ...]:
    """Deterministically permute specimens with a dedicated RNG.

    The permutation is the single source of split variance across seeds, so it
    must not depend on global RNG state.

    Args:
        specimens: Specimen identifiers.
        seed: Non-negative seed.

    Returns:
        The permuted identifiers.
    """
    unique = _as_unique_sorted(specimens)
    order = np.random.default_rng(seed).permutation(len(unique))
    return tuple(unique[i] for i in order)


def group_kfold(
    specimens: Iterable[str],
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[Fold, ...]:
    """Build specimen-disjoint K folds with a rotating validation block.

    The permuted specimens are cut into ``n_splits`` contiguous blocks. For
    block ``i``: test is block ``i``, validation is block ``i + 1`` (wrapping),
    and train is everything else.

    Args:
        specimens: Specimen identifiers.
        n_splits: Number of folds; must be at least 3 so that validation is
            always a distinct block.
        seed: Seed for the specimen permutation.

    Returns:
        One :class:`Fold` per block.

    Raises:
        ValueError: If there are too few specimens for the requested split.
    """
    unique = _as_unique_sorted(specimens)
    if n_splits < 3:
        raise ValueError(f"n_splits must be >= 3, got {n_splits}")
    if len(unique) < max(n_splits, _MIN_SPECIMENS_GROUP_KFOLD):
        raise ValueError(
            f"group_kfold needs >= {max(n_splits, _MIN_SPECIMENS_GROUP_KFOLD)} specimens, "
            f"got {len(unique)}; use loso or supply more specimens"
        )

    permuted = permute_specimens(unique, seed)
    blocks = [tuple(b) for b in np.array_split(np.array(permuted, dtype=object), n_splits)]

    folds: list[Fold] = []
    for i, test_block in enumerate(blocks):
        val_block = blocks[(i + 1) % n_splits]
        test_set, val_set = set(test_block), set(val_block)
        train = tuple(s for s in permuted if s not in test_set and s not in val_set)
        folds.append(
            Fold(
                fold_id=i,
                strategy="group_kfold",
                train=train,
                val=tuple(val_block),
                test=tuple(test_block),
                seed=seed,
            )
        )
    return tuple(folds)


def loso(
    specimens: Iterable[str],
    seed: int = 0,
    n_splits: int | None = None,
) -> tuple[Fold, ...]:
    """Build Leave-One-Specimen-Out folds.

    Each specimen is the test set exactly once; validation is the next specimen
    in the seeded permutation.

    Args:
        specimens: Specimen identifiers.
        seed: Seed for the specimen permutation.
        n_splits: Ignored. LOSO's fold count is the specimen count by
            definition; the parameter exists so every strategy is callable with
            the same signature through :func:`make_folds`.

    Returns:
        One :class:`Fold` per specimen.

    Raises:
        ValueError: If there are fewer than three specimens.
    """
    unique = _as_unique_sorted(specimens)
    if len(unique) < _MIN_SPECIMENS_LOSO:
        raise ValueError(
            f"loso needs >= {_MIN_SPECIMENS_LOSO} specimens, got {len(unique)}"
        )

    permuted = permute_specimens(unique, seed)
    folds: list[Fold] = []
    for i, test_specimen in enumerate(permuted):
        val_specimen = permuted[(i + 1) % len(permuted)]
        held_out = {test_specimen, val_specimen}
        train = tuple(s for s in permuted if s not in held_out)
        folds.append(
            Fold(
                fold_id=i,
                strategy="loso",
                train=train,
                val=(val_specimen,),
                test=(test_specimen,),
                seed=seed,
            )
        )
    return tuple(folds)


def time_block_holdout(
    specimens: Iterable[str],
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[Fold, ...]:
    """Build folds over contiguous time blocks instead of true specimens.

    This exists ONLY for datasets that publish no specimen identifier. Each
    "specimen" passed in must already be a time block produced by the loader.
    It is not protocol-compliant and results built on it are labelled
    accordingly by the runner.

    Args:
        specimens: Time-block identifiers, ordered by time.
        n_splits: Number of held-out blocks.
        seed: Unused; kept for a uniform strategy signature.

    Returns:
        One :class:`Fold` per block.
    """
    ordered = tuple(dict.fromkeys(str(s) for s in specimens))
    if len(ordered) < _MIN_SPECIMENS_GROUP_KFOLD:
        raise ValueError(
            f"time_block_holdout needs >= {_MIN_SPECIMENS_GROUP_KFOLD} blocks, got {len(ordered)}"
        )
    n_splits = min(n_splits, len(ordered) - 2)
    blocks = [tuple(b) for b in np.array_split(np.array(ordered, dtype=object), n_splits)]

    folds: list[Fold] = []
    for i, test_block in enumerate(blocks):
        val_block = blocks[(i + 1) % n_splits]
        held_out = set(test_block) | set(val_block)
        train = tuple(s for s in ordered if s not in held_out)
        folds.append(
            Fold(
                fold_id=i,
                strategy="time_block_holdout",
                train=train,
                val=tuple(val_block),
                test=tuple(test_block),
                seed=seed,
            )
        )
    return tuple(folds)


def nested_train_fraction(specimens: Iterable[str], fraction: float, seed: int) -> tuple[str, ...]:
    """Select a nested, deterministic fraction of the specimen set for training.

    Protocol ``E3_lowdata`` (``fractions: [0.10, 0.25, 0.50, 0.75, 1.00]``,
    ``sampling: nested``): for a fixed seed the specimen pool that survives a
    low-data subsample is a prefix of the pool for any higher fraction, so
    10% is nested inside 25% and so on up to 100%. Every specimen therefore
    has exactly one position in the order, independent of which fraction
    later consumes it; and at ``fraction == 1.0`` the pool is untouched, so a
    100% run is byte-identical to the full-data (E1/E2) batch.

    Args:
        specimens: Specimen identifiers (duplicates are collapsed).
        fraction: Fraction of the pool to keep; must satisfy ``0 < f <= 1``.
        seed: Non-negative seed for the specimen permutation. The same seed
            therefore yields the same nested chain.

    Returns:
        The ordered specimen pool of ``ceil(fraction * n)`` unique specimens.

    Raises:
        ValueError: If the collection is empty, or ``fraction`` is outside
            ``(0, 1]``.
    """
    unique = _as_unique_sorted(specimens)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train fraction must satisfy 0 < f <= 1, got {fraction!r}")
    if fraction >= 1.0:
        return unique
    pool_size = max(1, int(np.ceil(fraction * len(unique))))
    permuted = permute_specimens(unique, seed)
    return permuted[:pool_size]


def fold_train_pool(fold: Fold, fraction: float) -> tuple[str, ...]:
    """Build one fold's E3 low-data TRAIN pool (DATA-61 fix).

    The E3 low-data admitted pool is taken from each fold's TRAIN side, not
    from the global specimen set. The pre-DATA-61 runner passed
    ``nested_train_fraction(specimens, fraction, seed)`` — a prefix of the
    global permutation — straight into every fold's tensor builder; when
    that prefix contained this fold's test (or val) specimen and no
    train-side specimen, the builder's last-resort guard raised and every
    unit of the fold was recorded as failed (the 100%-failed D2 f10 batch).

    Semantics (option (b) of the DATA-61 fix — global nested ordering):

    1. The fold's own seeded permutation ``perm = permute_specimens(
       fold.train, fold.seed)`` is the order source. Because ``group_kfold``
       and ``loso`` both build their train sides by restriction of the same
       global seed permutation that ``make_folds`` used, this is the
       restriction of that global permutation to the fold's train side —
       i.e. option (b)'s "global permutation ∩ fold.train" order, which the
       fold's own ``seed`` carries (``Fold.seed`` is the seed that produced
       the partition, recorded on the fold object).
    2. ``pool = perm[:max(1, ceil(fraction * len(fold.train)))]``, so
       ``pool ⊆ fold.train`` always: a fold can never be admitted its own
       test/val specimen, and no fold's train side can be emptied by the
       sampling.
    3. Within one fold the chain is monotone: for the same fold
       ``pool(f=0.10) ⊆ pool(f=0.25) ⊆ … ⊆ pool(f=1.00) = fold.train``
       (byte-identical to the E1/E2 train side at 100%).
    4. Across folds the pools may differ (the train sides differ), but
       every order derives from the one seeded permutation, so results are
       reproducible across batches.

    Args:
        fold: The fold whose TRAIN side the pool is built from.
        fraction: Low-data fraction; must satisfy ``0 < f <= 1``. At
            ``f == 1.0`` the pool is the full train side (no subsampling).

    Returns:
        The ordered tuple of admitted TRAIN specimens for this fold: always
        non-empty when ``fold.train`` is non-empty, always a subset of
        ``fold.train``, and a nested prefix of the fold's order at any
        higher fraction.

    Raises:
        ValueError: If the fold's train side is empty, or ``fraction`` is
            outside ``(0, 1]``.
    """
    unique_train = _as_unique_sorted(fold.train)
    if not unique_train:
        raise ValueError(f"fold {fold.fold_id}: train side is empty; no pool to build")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train fraction must satisfy 0 < f <= 1, got {fraction!r}")
    if fraction >= 1.0:
        return unique_train
    permuted = permute_specimens(unique_train, fold.seed)
    pool_size = max(1, int(np.ceil(fraction * len(unique_train))))
    return permuted[:pool_size]


_STRATEGIES = {
    "group_kfold": group_kfold,
    "loso": loso,
    "time_block_holdout": time_block_holdout,
}


def make_folds(
    specimens: Iterable[str],
    strategy: SplitStrategy = "group_kfold",
    seed: int = 0,
    n_splits: int = 5,
) -> tuple[Fold, ...]:
    """Dispatch to a split strategy by name.

    Args:
        specimens: Specimen (or time-block) identifiers.
        strategy: One of ``group_kfold``, ``loso``, ``time_block_holdout``.
        seed: Seed for the specimen permutation.
        n_splits: Number of folds, where the strategy uses it.

    Returns:
        The folds produced by the requested strategy.

    Raises:
        ValueError: If ``strategy`` is unknown.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}; expected {sorted(_STRATEGIES)}")
    return _STRATEGIES[strategy](specimens, n_splits=n_splits, seed=seed)
