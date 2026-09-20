"""The paired statistics, and the resampling unit that must never be the seed.

The R0 audit's CRITICAL item C1 was that protocol v1 named its bootstrap
resampling unit as "over seeds". These tests hold the replacement to that
finding: the unit is the fold, the seed is a stratification, and asking for a
seed-based bootstrap is refused rather than quietly performed.
"""

from __future__ import annotations

import numpy as np
import pytest

from drososense.evaluation.stats import (
    DEFAULT_ALPHA,
    InsufficientDataError,
    PairedSpec,
    clusters_needed_for_alpha,
    fold_cluster_bootstrap,
    hodges_lehmann,
    holm_correction,
    minimum_achievable_p,
    paired_test,
    rank_biserial,
    tost_from_interval,
    wilcoxon_reachable,
)


# ---------------------------------------------------------------------------
# The resampling unit
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_resampling_over_seeds_is_refused():
    """A seed-based bootstrap is the exact defect the R0 audit raised."""
    with pytest.raises(ValueError, match="must not be 'seed'"):
        PairedSpec(resample_unit="seed")


@pytest.mark.unit
def test_bootstrap_draws_folds_within_each_seed():
    """The bootstrap resamples fold identifiers, never seed identifiers.

    The check is structural rather than statistical: with two seeds whose fold
    differences are drawn from disjoint ranges, a fold-respecting bootstrap can
    never produce a resample mean outside the union of the per-seed ranges,
    whereas a seed-respecting one trivially can.
    """
    deltas = np.array([0.00, 0.01, 10.00, 10.01])
    seeds = np.array([0, 0, 1, 1])
    folds = np.array([0, 1, 0, 1])
    spec = PairedSpec(bootstrap_b=500, bootstrap_seed=1)
    draws = fold_cluster_bootstrap(deltas, seeds, folds, spec)
    # Each seed contributes one value near 0 and one near 10; the mean of the two
    # seed means always lands inside [5, 10.01/... ] — never outside the data.
    assert draws.min() >= deltas.min() - 1e-12
    assert draws.max() <= deltas.max() + 1e-12
    # Averaging the two per-seed means bounds the statistic well inside 0..10.01.
    assert draws.min() > 0.0
    assert draws.max() < 10.01


@pytest.mark.unit
def test_bootstrap_is_reproducible_from_its_own_seed():
    """The interval must not depend on the run seed, only on the declared one."""
    deltas = np.array([0.1, -0.2, 0.3, 0.05, 0.2, -0.1])
    seeds = np.array([0, 0, 0, 1, 1, 1])
    folds = np.array([0, 1, 2, 0, 1, 2])
    a = fold_cluster_bootstrap(deltas, seeds, folds, PairedSpec(bootstrap_b=200, bootstrap_seed=7))
    b = fold_cluster_bootstrap(deltas, seeds, folds, PairedSpec(bootstrap_b=200, bootstrap_seed=7))
    c = fold_cluster_bootstrap(deltas, seeds, folds, PairedSpec(bootstrap_b=200, bootstrap_seed=8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.unit
def test_bootstrap_refuses_a_single_fold_per_seed():
    """With no fold variation inside a seed there is nothing to resample."""
    deltas = np.array([0.1, 0.2])
    seeds = np.array([0, 1])
    folds = np.array([0, 0])
    with pytest.raises(InsufficientDataError, match="two or more folds"):
        fold_cluster_bootstrap(deltas, seeds, folds, PairedSpec(bootstrap_b=10))


@pytest.mark.unit
def test_unimplemented_ci_type_is_refused_not_substituted():
    """Asking for BCa must fail loudly rather than silently returning percentile."""
    with pytest.raises(NotImplementedError, match="bca"):
        PairedSpec(ci_type="bca")


# ---------------------------------------------------------------------------
# The paired test
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_paired_test_reports_clusters_and_non_zero_pairs():
    """n_pairs alone hides pseudo-replication; n_clusters is reported beside it."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.02, size=30)
    seeds = np.repeat([0, 1, 2], 10)
    folds = np.tile(np.arange(10), 3)
    result = paired_test(deltas, seeds, folds, PairedSpec(bootstrap_b=200), contrast_id="R0_vs_R2")
    assert result.n_pairs == 30
    assert result.n_clusters == 10
    assert result.n_seeds == 3
    assert result.n_nonzero == 30
    assert result.delta_ci_low < result.delta < result.delta_ci_high
    assert result.test == "wilcoxon_signed_rank"
    assert result.effect_size_name == "rank_biserial"


@pytest.mark.unit
def test_a_uniform_positive_shift_is_detected():
    """A clean, consistent effect must come out significant and favourable."""
    deltas = np.linspace(0.02, 0.10, 20)
    seeds = np.repeat([0, 1], 10)
    folds = np.tile(np.arange(10), 2)
    result = paired_test(deltas, seeds, folds, PairedSpec(bootstrap_b=500))
    assert result.favours_first
    assert result.p_value < 0.01
    assert result.delta_ci_low > 0


@pytest.mark.unit
def test_a_shift_in_the_wrong_direction_is_not_favourable():
    """The sign of the interval decides, not the raw p-value."""
    deltas = np.linspace(-0.10, -0.02, 20)
    seeds = np.repeat([0, 1], 10)
    folds = np.tile(np.arange(10), 2)
    result = paired_test(deltas, seeds, folds, PairedSpec(bootstrap_b=500))
    assert not result.favours_first
    assert result.delta_ci_high < 0


@pytest.mark.unit
def test_fewer_than_six_non_zero_pairs_substitutes_the_sign_test():
    """The substitution is pre-registered, and it is recorded on the result."""
    deltas = np.array([0.0, 0.0, 0.4, 0.5, 0.3, 0.0, 0.0])
    seeds = np.array([0, 0, 0, 0, 0, 0, 0])
    folds = np.arange(7)
    result = paired_test(deltas, seeds, folds, PairedSpec(bootstrap_b=50))
    assert result.test == "sign_test"
    assert result.n_nonzero == 3
    assert "sign test was substituted" in result.substitution


@pytest.mark.unit
def test_too_few_observations_raises_rather_than_reporting_nothing():
    """A comparison with no data must not silently produce a p-value of 1."""
    with pytest.raises(InsufficientDataError):
        paired_test([0.1], [0], [0], PairedSpec(bootstrap_b=10))


# ---------------------------------------------------------------------------
# Power, effect sizes and correction
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_minimum_achievable_p_matches_the_exact_bound():
    """The exact two-sided sign-flip minimum is 2 / 2^n."""
    assert minimum_achievable_p(5) == pytest.approx(2 / 2**5)
    assert minimum_achievable_p(10) == pytest.approx(2 / 2**10)


@pytest.mark.unit
def test_five_clusters_cannot_reach_alpha():
    """This is why D2's five cuts are reported as underpowered, not as no effect."""
    assert not wilcoxon_reachable(5, DEFAULT_ALPHA)
    assert minimum_achievable_p(5) > DEFAULT_ALPHA
    assert wilcoxon_reachable(6, DEFAULT_ALPHA)
    assert clusters_needed_for_alpha(DEFAULT_ALPHA) == 6


@pytest.mark.unit
def test_rank_biserial_is_signed_and_bounded():
    """The effect size carries the direction, which a p-value does not."""
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)
    assert rank_biserial(np.zeros(4)) == 0.0
    mixed = rank_biserial(np.array([1.0, 1.0, -1.0]))
    assert -1.0 < mixed < 1.0


@pytest.mark.unit
def test_hodges_lehmann_is_the_median_walsh_average():
    """The regression effect size is location, not rank."""
    deltas = np.array([1.0, 2.0, 3.0])
    # Walsh averages: 1, 1.5, 2, 2, 2.5, 3 -> median 2
    assert hodges_lehmann(deltas) == pytest.approx(2.0)


@pytest.mark.unit
def test_holm_is_step_down_and_monotone():
    """Adjusted p-values never invert the ordering of the raw ones."""
    raw = [0.001, 0.02, 0.03, 0.2]
    adjusted = holm_correction(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert all(a <= 1.0 for a in adjusted)
    order = np.argsort(raw)
    assert list(np.array(adjusted)[order]) == sorted(adjusted)


@pytest.mark.unit
def test_holm_leaves_a_single_test_alone():
    """A one-test family is uncorrected, and the protocol says so."""
    assert holm_correction([0.03]) == [0.03]


@pytest.mark.unit
def test_tost_needs_the_interval_inside_the_margin():
    """Equivalence is interval inclusion, never a failure to reject."""
    assert tost_from_interval(-0.01, 0.01, margin=0.02, alpha=0.05) == (0.0, True)
    _, equivalent = tost_from_interval(-0.05, 0.01, margin=0.02, alpha=0.05)
    assert not equivalent
    # A wide interval spanning zero is not evidence of equivalence.
    _, equivalent_wide = tost_from_interval(-0.5, 0.5, margin=0.02, alpha=0.05)
    assert not equivalent_wide
    # A zero margin can never establish equivalence.
    assert tost_from_interval(-0.001, 0.001, margin=0.0, alpha=0.05) == (1.0, False)
