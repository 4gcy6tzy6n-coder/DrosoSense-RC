"""Tests for ``drososense.evaluation.e2_stats`` — the M4/E2 analysis plumbing.

The module is read-only by design (it must never evaluate a model), so the
tests exercise it entirely on synthetic record frames: the pairing arithmetic
on a hand-made frame, the family routing, and the honest "unpairable /
insufficient" paths that a reviewer must see reported, not imputed.

The records carry the protocol's primary metric for each task (classification:
``macro_f1``; regression: ``mae``) with a deliberate gap on some models, so
every "unpairable" / "insufficient" code path is exercised for real rather
than by skipping the call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from drososense.evaluation import e2_stats
from drososense.evaluation.stats import PairedSpec, fold_cluster_bootstrap
from drososense.utils.config import load_protocol, protocol_paired_spec


def _records() -> pd.DataFrame:
    """Hand-built frame: one dataset, classification task, seeds 0–1, folds 0–1.

    R0 and R2 have the same macro_f1 on every observation (0.5 vs 0.4), so
    the paired difference is a constant 0.1 — a degenerate (all-equal) delta
    distribution the pre-registered cluster sign test must handle honestly.
    R3 is present on only one seed, so its contrast with R0 is genuinely
    unpairable (never imputed).
    """
    rows: list[dict[str, Any]] = []
    for seed in (0, 1):
        for fold in (0, 1):
            for model, mf1 in (("R0_real_fly", 0.5), ("R2_degree_rewired", 0.4)):
                rows.append({
                    "dataset": "d2_fixture",
                    "model": model,
                    "task": "classification",
                    "seed": seed,
                    "fold_id": fold,
                    "window_length": 16,
                    "status": "ok",
                    "accuracy": 0.8,
                    "macro_f1": mf1,
                })
    # R3: only one of the two seeds, so it cannot pair with R0.
    rows.append({
        "dataset": "d2_fixture",
        "model": "R3_random_sparse",
        "task": "classification",
        "seed": 0,
        "fold_id": 0,
        "window_length": 16,
        "status": "ok",
        "accuracy": 0.7,
        "macro_f1": 0.3,
    })
    return pd.DataFrame(rows)


def test_seed_invariance_of_the_bootstrap_interval() -> None:
    """Protocol v1.3 `bootstrap.seed_invariance_required`: for a FIXED set of
    per-cluster values, the interval must not change when the same clusters
    are measured under more seeds. This is a property of the resampler the
    module routes through, not of the records — pinned here so a change of
    bootstrap plumbing cannot silently invalidate the interval a reviewer
    recomputes from the published table.
    """
    protocol = load_protocol()
    spec = protocol_paired_spec(protocol, "macro_f1", "classification")
    deltas = [0.3, -0.1, 0.2, 0.0, 0.4]  # one value per cluster
    folds = [0, 1, 2, 3, 4]

    b1 = fold_cluster_bootstrap(deltas, folds, spec)
    # The same clusters, measured under three more seeds: per-cluster means
    # unchanged, so the interval must be unchanged too.
    deltas_more = [v for v in deltas for _ in range(4)]
    folds_more = [f for f in folds for _ in range(4)]
    b2 = fold_cluster_bootstrap(deltas_more, folds_more, spec)
    tail = (1.0 - spec.ci_level) / 2.0
    lo1, hi1 = np.quantile(b1, tail), np.quantile(b1, 1.0 - tail)
    lo2, hi2 = np.quantile(b2, tail), np.quantile(b2, 1.0 - tail)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)


def test_paired_frame_lines_up_on_the_record_unit() -> None:
    frame = e2_stats.index_runs(_records())
    joined = e2_stats.paired_frame(
        frame, "R0_real_fly", "R2_degree_rewired", "macro_f1", "d2_fixture", "classification"
    )
    assert joined is not None
    # 2 seeds x 2 folds, one pair each; the constant delta is 0.5 - 0.4.
    assert len(joined) == 4
    assert np.allclose(joined["delta"].to_numpy(), 0.1)


def test_paired_frame_returns_none_when_a_side_is_missing() -> None:
    frame = e2_stats.index_runs(_records())
    # R5 never appears in the records, so its contrast with R0 is unpairable.
    assert e2_stats.paired_frame(
        frame, "R0_real_fly", "R5_small_world", "macro_f1", "d2_fixture", "classification"
    ) is None


def test_partial_overlap_pairs_the_common_units_only() -> None:
    """A contrast whose two sides do not cover the same seeds is NOT
    silently imputed: the row is built from the common scored units only, and
    the per-record unit is visible in n_pairs / n_folds. R3 on one seed vs
    R0 on two seeds yields exactly one pair.
    """
    frame = e2_stats.index_runs(_records())
    joined = e2_stats.paired_frame(
        frame, "R0_real_fly", "R3_random_sparse", "macro_f1", "d2_fixture", "classification"
    )
    assert joined is not None
    assert len(joined) == 1  # only (seed 0, fold 0) is common to both sides
    assert int(joined["seed"].iloc[0]) == 0


def test_rows_report_unpairable_contrasts_instead_of_dropping_them() -> None:
    protocol = load_protocol()
    frame = e2_stats.index_runs(_records())
    table = e2_stats.build_e2_table(protocol, frame)
    assert not table.empty
    unpairable = table[table["status"] == "unpairable"]
    # Models absent from the records (e.g. R5) must be reported, not gone.
    assert "R0_vs_R5" in set(unpairable["contrast_id"])
    # The primary contrast is fully paired and must be the "ok" one.
    ok = table[(table["contrast_id"] == "R0_vs_R2") & (table["status"] == "ok")]
    assert not ok.empty
    assert "R0_vs_R2" not in set(unpairable["contrast_id"])


def test_family_routing_uses_the_contrast_condition_pair() -> None:
    protocol = load_protocol()
    frame = e2_stats.index_runs(_records())
    table = e2_stats.build_e2_table(protocol, frame)
    primary = table[(table["contrast_id"] == "R0_vs_R2") & (table["status"] == "ok")]
    # F_primary is the single-pair family: no correction is claimed, so the
    # corrected p equals the raw p.
    assert not primary.empty
    for row in primary.itertuples():
        assert row.family == "F_primary"
        assert row.p_holm == pytest.approx(row.p_value)
        # Reachability floor is disclosed on every row (v1.3 §10 rule).
        assert row.minimum_achievable_p_over_clusters is not None
        # The decisive p is the cluster sign test, never the descriptive
        # Wilcoxon. Two clusters, one favouring R0 and one zero (constant
        # delta per cluster): p = 2 * C(1,1) / 2**1 = 0.5.
        assert row.p_value == pytest.approx(0.5)


def test_paired_stat_row_discloses_the_decisive_and_descriptive_tests() -> None:
    protocol = load_protocol()
    frame = e2_stats.index_runs(_records())
    joined = e2_stats.paired_frame(
        frame, "R0_real_fly", "R2_degree_rewired", "macro_f1", "d2_fixture", "classification"
    )
    row = e2_stats.paired_stat_row(
        protocol, joined, "R0_vs_R2", "macro_f1", "d2_fixture", "classification"
    )
    assert row is not None
    assert row["status"] == "ok"
    assert row["n_pairs"] == 4
    assert row["n_folds"] == 2
    # The decisive cluster-level p is present; the descriptive Wilcoxon p is
    # present separately and must NOT be the gate input.
    assert "p_value" in row
    assert "p_paired_wilcoxon" in row
    assert row["effect_size_name"] == "rank_biserial"  # classification task
    # Mean paired difference (constant 0.5 - 0.4).
    assert row["delta"] == pytest.approx(0.1)
    # With 10000 resamples on 2 clusters of a constant delta the
    # percentile CI is degenerate around 0.1 up to float precision.
    assert row["delta_ci_low"] == pytest.approx(0.1, abs=1e-9)
    assert row["delta_ci_high"] == pytest.approx(0.1, abs=1e-9)


def test_insufficient_data_is_reported_not_imputed() -> None:
    protocol = load_protocol()
    # No common scored unit at all: the module must say so, never fill a number.
    row = e2_stats.paired_stat_row(
        protocol, None, "R0_vs_R2", "macro_f1", "d2_fixture", "classification"
    )
    assert row is None
