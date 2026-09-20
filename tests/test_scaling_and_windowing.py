"""Standardisation and windowing unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from drososense.data.scaling import DEFAULT_EPS, Standardizer, fit_standardizer
from drososense.data.schema import SPECIMEN_COLUMN, TIME_COLUMN
from drososense.data.windowing import make_windows


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_uses_only_the_rows_it_is_given():
    """Statistics are exactly the given rows' statistics."""
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    scaler = fit_standardizer(x)
    np.testing.assert_allclose(scaler.mean, x.mean(axis=0))
    np.testing.assert_allclose(scaler.std, x.std(axis=0) + DEFAULT_EPS)


@pytest.mark.unit
def test_transform_centres_and_scales():
    """Transformed training rows have approximately zero mean and unit variance."""
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
    scaler = fit_standardizer(x)
    transformed = scaler.transform(x)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-3)


@pytest.mark.unit
def test_inverse_transform_round_trips():
    """Scaling then unscaling returns the original values."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    scaler = fit_standardizer(x)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x, atol=1e-9)


@pytest.mark.unit
def test_zero_variance_channel_stays_finite():
    """A constant channel does not divide by zero."""
    x = np.column_stack([np.ones(20), np.arange(20.0)])
    scaler = fit_standardizer(x)
    transformed = scaler.transform(x)
    assert np.all(np.isfinite(transformed))
    np.testing.assert_allclose(transformed[:, 0], 0.0)


@pytest.mark.unit
def test_transform_preserves_extra_axes():
    """A window tensor can be transformed channel-wise in one call."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(7, 5, 3))
    scaler = fit_standardizer(x.reshape(-1, 3))
    transformed = scaler.transform(x)
    assert transformed.shape == x.shape


@pytest.mark.unit
def test_transform_rejects_a_channel_mismatch():
    """Feeding the wrong channel count fails rather than broadcasting."""
    scaler = fit_standardizer(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="channels"):
        scaler.transform(np.zeros((4, 2)))


@pytest.mark.unit
def test_scaler_round_trips_through_dict():
    """A scaler survives serialisation into a run record."""
    scaler = fit_standardizer(np.random.default_rng(3).normal(size=(30, 5)))
    restored = Standardizer.from_dict(scaler.to_dict())
    np.testing.assert_allclose(restored.mean, scaler.mean)
    np.testing.assert_allclose(restored.std, scaler.std)
    assert restored.eps == scaler.eps


@pytest.mark.unit
def test_fit_rejects_degenerate_input():
    """Too few rows, non-finite values and wrong rank are all rejected."""
    with pytest.raises(ValueError, match="at least 2 training rows"):
        fit_standardizer(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        fit_standardizer(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="2-D"):
        fit_standardizer(np.zeros(5))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def _toy_frame(n_specimens: int = 2, n_rows: int = 10, n_features: int = 2):
    import pandas as pd

    rows = []
    for s in range(n_specimens):
        for t in range(n_rows):
            row = {
                SPECIMEN_COLUMN: f"sp{s}",
                TIME_COLUMN: t,
                "freshness_class": t % 4,
                "tvc": 2.0 + 0.1 * t,
            }
            for f in range(n_features):
                row[f"f{f}"] = float(s * 100 + t)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_window_shapes_and_labels():
    """Window tensor, labels and provenance all line up."""
    frame = _toy_frame()
    features = ["f0", "f1"]
    windows = make_windows(frame, features, window_length=4, stride=1)

    assert windows.X.shape == (14, 4, 2)  # 2 specimens x (10 - 4 + 1)
    assert windows.y_class.shape == (14,)
    assert windows.row_specimens.shape == (14, 4)
    assert windows.window_length == 4
    assert set(windows.specimen_ids) == {"sp0", "sp1"}


@pytest.mark.unit
def test_last_label_rule_takes_the_final_timestep():
    """The default label is the freshness at the end of the window."""
    frame = _toy_frame(n_specimens=1, n_rows=6)
    windows = make_windows(frame, ["f0", "f1"], window_length=3, label_rule="last")
    expected = [frame["freshness_class"].iloc[i + 2] for i in range(4)]
    assert windows.y_class.tolist() == expected


@pytest.mark.unit
def test_majority_label_rule_is_accepted():
    """The alternative reduction is available and returns valid labels."""
    frame = _toy_frame(n_specimens=1, n_rows=8)
    windows = make_windows(frame, ["f0", "f1"], window_length=4, label_rule="majority")
    assert set(windows.y_class).issubset({0, 1, 2, 3})
    assert len(windows) == 5


@pytest.mark.unit
def test_stride_changes_the_window_count():
    """Stride is honoured exactly."""
    frame = _toy_frame(n_specimens=1, n_rows=10)
    assert len(make_windows(frame, ["f0", "f1"], 4, stride=1)) == 7
    assert len(make_windows(frame, ["f0", "f1"], 4, stride=2)) == 4
    assert len(make_windows(frame, ["f0", "f1"], 4, stride=4)) == 2


@pytest.mark.unit
def test_remainder_is_discarded_not_padded():
    """A specimen shorter than the window cannot silently become an empty window."""
    frame = _toy_frame(n_specimens=1, n_rows=5)
    assert len(make_windows(frame, ["f0", "f1"], 5, stride=1)) == 1
    with pytest.raises(ValueError, match="fewer than window_length"):
        make_windows(frame, ["f0", "f1"], 6)


@pytest.mark.unit
def test_no_padding_appears_in_the_tensor():
    """Every window contains only real observations from one specimen."""
    frame = _toy_frame()
    windows = make_windows(frame, ["f0", "f1"], window_length=4)
    for i in range(len(windows)):
        assert len(set(windows.row_specimens[i].tolist())) == 1
        source = frame.loc[windows.row_index[i]]
        np.testing.assert_allclose(source["f0"].to_numpy(), windows.X[i, :, 0])


@pytest.mark.unit
def test_window_boundaries_respect_time_order_within_a_specimen():
    """Windows advance monotonically in time inside each specimen."""
    frame = _toy_frame()
    windows = make_windows(frame, ["f0", "f1"], window_length=3)
    for specimen in set(windows.specimen_ids):
        mask = windows.specimen_ids == specimen
        ends = windows.end_time[mask]
        assert np.all(np.diff(ends) > 0)


@pytest.mark.unit
def test_flat_matches_the_tensor():
    """``flat`` is a pure reshape and shares the flattening order."""
    frame = _toy_frame()
    windows = make_windows(frame, ["f0", "f1"], window_length=4)
    np.testing.assert_allclose(
        windows.flat(), windows.X.reshape(len(windows), -1)
    )


@pytest.mark.unit
def test_subset_selects_windows():
    """Subsetting keeps provenance aligned."""
    frame = _toy_frame()
    windows = make_windows(frame, ["f0", "f1"], window_length=4)
    mask = np.zeros(len(windows), dtype=bool)
    mask[:3] = True
    subset = windows.subset(mask)
    assert len(subset) == 3
    np.testing.assert_allclose(subset.X, windows.X[:3])
    assert subset.specimen_ids.tolist() == windows.specimen_ids[:3].tolist()


@pytest.mark.unit
def test_invalid_parameters_are_rejected():
    """Bad window parameters fail loudly."""
    frame = _toy_frame()
    with pytest.raises(ValueError, match="window_length"):
        make_windows(frame, ["f0", "f1"], 0)
    with pytest.raises(ValueError, match="stride"):
        make_windows(frame, ["f0", "f1"], 4, stride=0)
    with pytest.raises(ValueError, match="label_rule"):
        make_windows(frame, ["f0", "f1"], 4, label_rule="median")
    with pytest.raises(ValueError, match="missing required column"):
        make_windows(frame, ["f0", "nope"], 4)
