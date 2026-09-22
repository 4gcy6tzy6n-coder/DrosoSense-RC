"""Leakage audits — including the positive controls that prove they can fail.

An audit that has never been observed to fire is indistinguishable from an
audit that always passes. Every audit function therefore has two tests here:
one that it accepts clean input, and one that feeds it deliberately leaked input
and asserts it raises.

The final test in this file takes the opposite direction: it *demonstrates* the
cost of leakage by building the same model on a row-level split and on a
specimen-level split of the same fixture, and showing that the row-level split
scores far higher. That gap is the reason ``split_unit: specimen`` is a red line.
"""

from __future__ import annotations

import numpy as np
import pytest

from drososense.data.leakage import (
    LeakageError,
    audit_class_coverage,
    audit_no_row_reuse,
    audit_scaler,
    audit_window_split_membership,
    audit_windows,
)
from drososense.data.pipeline import build_fold_tensors, usable_specimens
from drososense.data.scaling import fit_standardizer
from drososense.data.schema import SPECIMEN_COLUMN
from drososense.data.splits import Fold, group_kfold
from drososense.data.windowing import make_windows

WINDOW_LENGTH = 8
EPS = 1e-8


# ---------------------------------------------------------------------------
# Split leakage
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_audit_fold_accepts_a_clean_partition(fixture_dataset):
    """A correctly built fold passes the partition audit."""
    from drososense.data.leakage import audit_fold

    specimens = fixture_dataset.specimens()
    for fold in group_kfold(specimens, n_splits=3, seed=0):
        audit_fold(fold, specimens)


# ---------------------------------------------------------------------------
# Scaler leakage
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_audit_scaler_accepts_a_train_only_scaler():
    """A scaler fitted on train rows alone passes."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 3))
    scaler = fit_standardizer(x_train, eps=EPS)
    audit_scaler(scaler.mean, scaler.std, x_train, EPS)


@pytest.mark.unit
def test_audit_scaler_detects_a_scaler_fitted_on_train_plus_test():
    """POSITIVE CONTROL: a scaler contaminated by test rows must be rejected."""
    rng = np.random.default_rng(1)
    x_train = rng.normal(loc=0.0, size=(50, 3))
    # Test rows come from a visibly different distribution.
    x_test = rng.normal(loc=8.0, size=(50, 3))
    contaminated = np.vstack([x_train, x_test])
    contaminated_scaler = fit_standardizer(contaminated, eps=EPS)

    with pytest.raises(LeakageError, match="does not match train-only"):
        audit_scaler(contaminated_scaler.mean, contaminated_scaler.std, x_train, EPS)


@pytest.mark.unit
def test_audit_scaler_rejects_a_contaminated_recomputation():
    """POSITIVE CONTROL: the audit must notice when it has no discriminating power."""
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(40, 2))
    scaler = fit_standardizer(x_train, eps=EPS)
    # Identical contamination means the audit cannot tell the two apart, and it
    # must say so rather than silently passing.
    with pytest.raises(LeakageError, match="no power"):
        audit_scaler(scaler.mean, scaler.std, x_train, EPS, X_contaminated=x_train.copy())


@pytest.mark.unit
def test_scaler_does_not_read_validation_or_test_rows(fixture_dataset):
    """Leaving test features untouched must not change the fitted scaler.

    This is the strongest form of the train-only guarantee: the scaler is built
    twice, once with the real test rows and once with test rows replaced by
    garbage, and the two scalers must be identical.
    """
    specimens = fixture_dataset.specimens()
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]

    clean = build_fold_tensors(fixture_dataset, fold, WINDOW_LENGTH, "/tmp/_leak_clean")

    corrupted_frame = fixture_dataset.frame.copy()
    features = list(fixture_dataset.schema.feature_columns)
    test_mask = corrupted_frame[SPECIMEN_COLUMN].astype(str).isin(set(fold.test)).to_numpy()
    corrupted_frame.loc[test_mask, features] = (
        corrupted_frame.loc[test_mask, features].to_numpy() * -1000.0 + 12345.0
    )

    from drososense.data.schema import Dataset

    corrupted = Dataset(schema=fixture_dataset.schema, frame=corrupted_frame)
    corrupted_tensors = build_fold_tensors(corrupted, fold, WINDOW_LENGTH, "/tmp/_leak_dirty")

    np.testing.assert_allclose(clean.scaler.mean, corrupted_tensors.scaler.mean, atol=1e-12)
    np.testing.assert_allclose(clean.scaler.std, corrupted_tensors.scaler.std, atol=1e-12)
    assert clean.scaler.n_fit_rows == corrupted_tensors.scaler.n_fit_rows


# ---------------------------------------------------------------------------
# Window leakage
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_audit_windows_accepts_clean_windows():
    """Windows built by the windowing module pass the audit."""
    row_specimens = np.array([["a"] * 4, ["b"] * 4], dtype=object)
    audit_windows(["a", "b"], row_specimens, 4)


@pytest.mark.unit
def test_audit_windows_detects_a_window_spanning_two_specimens():
    """POSITIVE CONTROL: a straddling window must be rejected."""
    row_specimens = np.array([["a", "a", "b", "b"]], dtype=object)
    with pytest.raises(LeakageError, match="spans multiple specimens"):
        audit_windows(["a"], row_specimens, 4)


@pytest.mark.unit
def test_audit_windows_detects_a_mislabelled_owner():
    """POSITIVE CONTROL: a window whose declared owner is wrong must be rejected."""
    row_specimens = np.array([["b", "b", "b", "b"]], dtype=object)
    with pytest.raises(LeakageError, match="declares specimen"):
        audit_windows(["a"], row_specimens, 4)


@pytest.mark.unit
def test_windows_never_cross_specimens_on_the_fixture(fixture_dataset):
    """Every produced window lies inside exactly one specimen."""
    windows = make_windows(
        fixture_dataset.frame, list(fixture_dataset.schema.feature_columns), WINDOW_LENGTH
    )
    audit_windows(windows.specimen_ids, windows.row_specimens, WINDOW_LENGTH)
    assert len(windows) > 0


@pytest.mark.unit
def test_window_count_matches_manual_per_specimen_tally(fixture_dataset):
    """The window count equals the sum of per-specimen windows, never a global count."""
    frame = fixture_dataset.frame
    windows = make_windows(frame, list(fixture_dataset.schema.feature_columns), WINDOW_LENGTH)
    expected = sum(
        max(0, len(group) - WINDOW_LENGTH + 1)
        for _, group in frame.groupby(SPECIMEN_COLUMN, observed=True)
    )
    assert len(windows) == expected


@pytest.mark.unit
def test_audit_window_split_membership_detects_foreign_specimens(fixture_dataset):
    """POSITIVE CONTROL: windows from another split must be rejected."""
    specimens = fixture_dataset.specimens()
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]
    audit_window_split_membership(list(fold.train), fold, "train")
    with pytest.raises(LeakageError, match="do not belong to it"):
        audit_window_split_membership(list(fold.test), fold, "train")


@pytest.mark.unit
def test_audit_no_row_reuse_detects_a_row_level_split():
    """POSITIVE CONTROL: rows shared between train and test must be rejected."""
    audit_no_row_reuse([1, 2, 3], [4, 5, 6])
    with pytest.raises(LeakageError, match="both train and test"):
        audit_no_row_reuse([1, 2, 3], [3, 4, 5])


# ---------------------------------------------------------------------------
# Fold tensors end to end
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_build_fold_tensors_produces_disjoint_specimens(fixture_dataset, tmp_path):
    """The assembled tensors obey every invariant at once."""
    specimens = usable_specimens(fixture_dataset, WINDOW_LENGTH)
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]
    tensors = build_fold_tensors(fixture_dataset, fold, WINDOW_LENGTH, tmp_path)

    train_specimens = set(tensors.train.specimen_ids)
    test_specimens = set(tensors.test.specimen_ids)
    assert not train_specimens & test_specimens
    assert train_specimens == set(fold.train)
    assert test_specimens == set(fold.test)

    audit_windows(tensors.train.specimen_ids, tensors.train.row_specimens, WINDOW_LENGTH)
    audit_windows(tensors.test.specimen_ids, tensors.test.row_specimens, WINDOW_LENGTH)


@pytest.mark.unit
def test_build_fold_tensors_persists_a_reloadable_scaler(fixture_dataset, tmp_path):
    """The frozen scaler survives a round-trip and reproduces its transform."""
    from drososense.data.scaling import Standardizer

    specimens = usable_specimens(fixture_dataset, WINDOW_LENGTH)
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]
    tensors = build_fold_tensors(fixture_dataset, fold, WINDOW_LENGTH, tmp_path)

    assert tensors.scaler_path.name == "scaler.pkl"
    reloaded = Standardizer.load(tensors.scaler_path.parent)
    np.testing.assert_allclose(reloaded.mean, tensors.scaler.mean)
    np.testing.assert_allclose(reloaded.std, tensors.scaler.std)


@pytest.mark.unit
def test_build_fold_tensors_rejects_a_specimen_too_short_to_window(fixture_dataset, tmp_path):
    """A fold that includes a too-short specimen fails rather than silently skipping it."""
    from drososense.data.schema import Dataset

    frame = fixture_dataset.frame.copy()
    victim = fixture_dataset.specimens()[0]
    victim_mask = frame[SPECIMEN_COLUMN].astype(str) == victim
    keep = frame.loc[victim_mask].index[: WINDOW_LENGTH - 1]
    frame = frame.drop(index=frame.loc[victim_mask].index.difference(keep))

    reduced = Dataset(schema=fixture_dataset.schema, frame=frame.reset_index(drop=True))
    specimens = reduced.specimens()
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]
    if victim not in fold.specimens:
        pytest.skip("the shortened specimen did not land in this fold")

    with pytest.raises(ValueError, match="too short"):
        build_fold_tensors(reduced, fold, WINDOW_LENGTH, tmp_path)


@pytest.mark.unit
def test_class_coverage_reports_absent_classes_without_fixing_them():
    """A class missing from a split is reported, not silently rebalanced."""
    report = audit_class_coverage(y_train=[0, 0, 1], y_test=[2, 3], n_classes=4)
    assert report["missing_in_train"] == [2, 3]
    assert report["missing_in_test"] == [0, 1]


# ---------------------------------------------------------------------------
# Why this matters: measured cost of leakage
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_row_level_split_scores_higher_than_specimen_level(fixture_dataset, tmp_path):
    """Quantify what a row-level split would buy, on the same data and model.

    The fixture gives each specimen its own sensor offset, exactly as a real
    e-nose does. A row-level split lets the model see each specimen's offset
    during training, so it can identify the specimen at test time; a
    specimen-level split cannot. The gap below is the leakage that
    ``split_unit: specimen`` exists to remove.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    frame = fixture_dataset.frame
    features = list(fixture_dataset.schema.feature_columns)
    windows = make_windows(frame, features, WINDOW_LENGTH)
    flat = windows.flat()
    labels = windows.y_class

    specimens = sorted(set(windows.specimen_ids))
    fold = group_kfold(specimens, n_splits=3, seed=0)[0]

    # Specimen-level split: the protocol's rule.
    train_mask = np.isin(windows.specimen_ids, list(fold.train))
    test_mask = np.isin(windows.specimen_ids, list(fold.test))
    honest = RandomForestClassifier(n_estimators=100, random_state=0)
    honest.fit(flat[train_mask], labels[train_mask])
    honest_f1 = f1_score(
        labels[test_mask], honest.predict(flat[test_mask]), average="macro", zero_division=0
    )

    # Row-level split: what the protocol forbids. Windows from the same specimen
    # appear on both sides.
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(flat))
    cut = int(0.7 * len(flat))
    leaky_train, leaky_test = shuffled[:cut], shuffled[cut:]
    leaky = RandomForestClassifier(n_estimators=100, random_state=0)
    leaky.fit(flat[leaky_train], labels[leaky_train])
    leaky_f1 = f1_score(
        labels[leaky_test], leaky.predict(flat[leaky_test]), average="macro", zero_division=0
    )

    assert leaky_f1 > honest_f1, (
        "expected the row-level split to inflate macro-F1 over the specimen-level split "
        f"(leaky={leaky_f1:.3f}, honest={honest_f1:.3f}); if this ever inverts, the fixture "
        f"has lost the specimen-specific structure that makes it a useful canary"
    )
