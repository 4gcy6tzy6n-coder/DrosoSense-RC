"""Sliding-window construction that never crosses a specimen boundary.

Windows are built independently inside each specimen: the frame is grouped by
specimen, each group is sorted by time, and the sliding window is applied to
that group alone. A trailing remainder shorter than the window is discarded
rather than padded, exactly as the frozen protocol requires.

Every :class:`WindowSet` carries the per-row specimen labels and source row
indices of every window, which is what lets
:func:`drososense.data.leakage.audit_windows` and
:func:`drososense.data.leakage.audit_no_row_reuse` verify the invariant from
the produced data instead of trusting the code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from drososense.data.schema import CLASS_COLUMN, REGRESSION_COLUMN, SPECIMEN_COLUMN, TIME_COLUMN

LabelRule = Literal["last", "majority"]

SUPPORTED_LABEL_RULES: tuple[str, ...] = ("last", "majority")


@dataclass(frozen=True)
class WindowSet:
    """A model-ready window tensor plus full provenance.

    Attributes:
        X: Feature windows, shape ``(n_windows, window_length, n_channels)``.
        y_class: Integer freshness label per window.
        y_reg: Continuous reference value (log10 TVC) per window.
        specimen_ids: Owning specimen of each window, shape ``(n_windows,)``.
        row_specimens: Specimen of every timestep inside each window, shape
            ``(n_windows, window_length)``.
        row_index: Source frame row positions, shape ``(n_windows, window_length)``.
        end_time: The ``time_index`` of each window's last timestep.
        window_length: Length ``L`` shared by every window.
        stride: Step between consecutive window starts.
        label_rule: How the window label was reduced from its timesteps.
    """

    X: np.ndarray
    y_class: np.ndarray
    y_reg: np.ndarray
    specimen_ids: np.ndarray
    row_specimens: np.ndarray
    row_index: np.ndarray
    end_time: np.ndarray
    window_length: int
    stride: int
    label_rule: str

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (n, L, C), got shape {self.X.shape}")
        if self.X.shape[1] != self.window_length:
            raise ValueError(
                f"X second axis {self.X.shape[1]} != window_length {self.window_length}"
            )
        for name in ("y_class", "y_reg", "specimen_ids", "end_time"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} has {getattr(self, name).shape[0]} entries, expected {n}")
        expected_rows = (n, self.window_length)
        for name in ("row_specimens", "row_index"):
            if getattr(self, name).shape != expected_rows:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape}, expected {expected_rows}"
                )

    def __len__(self) -> int:
        """Number of windows."""
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of feature channels."""
        return int(self.X.shape[2])

    def flat(self) -> np.ndarray:
        """Return windows flattened to ``(n_windows, L * C)``.

        Classical estimators expect a 2-D design matrix; the flattening order
        (time-major, then channel) is fixed here so that it is identical for
        every model.

        Returns:
            The flattened design matrix.
        """
        return self.X.reshape(self.X.shape[0], -1)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        """Return a new set containing only the windows selected by ``mask``.

        Args:
            mask: Boolean array of length ``len(self)``.

        Returns:
            A new :class:`WindowSet`.

        Raises:
            ValueError: If ``mask`` has the wrong length.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(f"mask shape {mask.shape} does not match {len(self)} windows")
        return WindowSet(
            X=self.X[mask],
            y_class=self.y_class[mask],
            y_reg=self.y_reg[mask],
            specimen_ids=self.specimen_ids[mask],
            row_specimens=self.row_specimens[mask],
            row_index=self.row_index[mask],
            end_time=self.end_time[mask],
            window_length=self.window_length,
            stride=self.stride,
            label_rule=self.label_rule,
        )

    def label_summary(self) -> dict[str, object]:
        """Summarise window counts and label distribution.

        Returns:
            Mapping with ``n_windows``, ``class_counts`` and ``per_specimen``.
        """
        counts = pd.Series(self.y_class).value_counts().sort_index()
        per_specimen = pd.Series(self.specimen_ids).value_counts().sort_index()
        return {
            "n_windows": len(self),
            "class_counts": {str(int(k)): int(v) for k, v in counts.items()},
            "per_specimen": {str(k): int(v) for k, v in per_specimen.items()},
        }


def _window_one_specimen(
    group: pd.DataFrame,
    feature_columns: list[str],
    window_length: int,
    stride: int,
    label_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build windows for a single specimen's rows.

    Args:
        group: Rows of one specimen, pre-sorted by time.
        feature_columns: Feature columns in schema order.
        window_length: Window length ``L``.
        stride: Step between window starts.
        label_rule: ``last`` or ``majority``.

    Returns:
        Tuple ``(X, y_class, y_reg, row_specimens, row_index, end_time)``.
    """
    values = group.loc[:, feature_columns].to_numpy(dtype=np.float64)
    classes = group[CLASS_COLUMN].to_numpy()
    regression = group[REGRESSION_COLUMN].to_numpy(dtype=np.float64)
    specimens = group[SPECIMEN_COLUMN].astype(str).to_numpy()
    times = group[TIME_COLUMN].to_numpy()
    positions = group.index.to_numpy()

    n_rows = values.shape[0]
    starts = range(0, n_rows - window_length + 1, stride)
    n_windows = len(range(0, n_rows - window_length + 1, stride))
    if n_windows == 0:
        empty_x = np.empty((0, window_length, values.shape[1]), dtype=np.float64)
        empty_scalar = np.empty((0,), dtype=np.float64)
        empty_str = np.empty((0, window_length), dtype=object)
        return (
            empty_x,
            np.empty((0,), dtype=np.int64),
            empty_scalar,
            np.empty((0, window_length), dtype=object),
            np.empty((0, window_length), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )

    x_out = np.empty((n_windows, window_length, values.shape[1]), dtype=np.float64)
    y_class = np.empty(n_windows, dtype=np.int64)
    y_reg = np.empty(n_windows, dtype=np.float64)
    row_specimens = np.empty((n_windows, window_length), dtype=object)
    row_index = np.empty((n_windows, window_length), dtype=np.int64)
    end_time = np.empty(n_windows, dtype=np.float64)

    for w, start in enumerate(starts):
        stop = start + window_length
        x_out[w] = values[start:stop]
        row_specimens[w] = specimens[start:stop]
        row_index[w] = positions[start:stop]
        end_time[w] = times[stop - 1]
        if label_rule == "last":
            y_class[w] = int(classes[stop - 1])
            y_reg[w] = float(regression[stop - 1])
        else:  # majority — ties resolved toward the later (fresher-degraded) label
            labels, counts = np.unique(classes[start:stop], return_counts=True)
            top = counts.max()
            candidates = labels[counts == top]
            y_class[w] = int(candidates[-1])
            y_reg[w] = float(np.mean(regression[start:stop]))

    return x_out, y_class, y_reg, row_specimens, row_index, end_time


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    window_length: int,
    stride: int = 1,
    label_rule: LabelRule = "last",
) -> WindowSet:
    """Build sliding windows that never span two specimens.

    Args:
        frame: Tidy frame with canonical columns.
        feature_columns: Feature columns to stack as channels, in order.
        window_length: Window length ``L``; must be positive.
        stride: Step between consecutive window starts; must be positive.
        label_rule: ``last`` (label at the final timestep, the default — the
            task is to report freshness at the end of the observed window) or
            ``majority``.

    Returns:
        The assembled :class:`WindowSet`.

    Raises:
        ValueError: If inputs are malformed or a specimen has fewer rows than
            the window length.
    """
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if label_rule not in SUPPORTED_LABEL_RULES:
        raise ValueError(f"label_rule must be one of {SUPPORTED_LABEL_RULES}, got {label_rule!r}")

    feature_columns = list(feature_columns)
    for column in (SPECIMEN_COLUMN, TIME_COLUMN, CLASS_COLUMN, REGRESSION_COLUMN, *feature_columns):
        if column not in frame.columns:
            raise ValueError(f"frame is missing required column {column!r}")

    # Sorting is local to each specimen; the group key keeps specimens apart.
    ordered = frame.sort_values([SPECIMEN_COLUMN, TIME_COLUMN], kind="stable")

    parts = []
    for specimen, group in ordered.groupby(SPECIMEN_COLUMN, sort=True, observed=True):
        if len(group) < window_length:
            raise ValueError(
                f"specimen {specimen!r} has {len(group)} rows, fewer than window_length "
                f"{window_length}; it cannot contribute a window. Lower the window length or "
                f"the split must place this specimen in a split that tolerates it."
            )
        parts.append(
            _window_one_specimen(group, feature_columns, window_length, stride, label_rule)
        )

    if not parts:
        raise ValueError("no specimens produced windows")

    x = np.concatenate([p[0] for p in parts], axis=0)
    y_class = np.concatenate([p[1] for p in parts], axis=0)
    y_reg = np.concatenate([p[2] for p in parts], axis=0)
    row_specimens = np.concatenate([p[3] for p in parts], axis=0)
    row_index = np.concatenate([p[4] for p in parts], axis=0)
    end_time = np.concatenate([p[5] for p in parts], axis=0)

    return WindowSet(
        X=x,
        y_class=y_class,
        y_reg=y_reg,
        specimen_ids=row_specimens[:, 0].astype(str),
        row_specimens=row_specimens,
        row_index=row_index,
        end_time=end_time,
        window_length=window_length,
        stride=stride,
        label_rule=label_rule,
    )
