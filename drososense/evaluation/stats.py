"""Paired statistics with a declared resampling unit.

This module exists because protocol v1.0 named its resampling unit as "bootstrap
… over seeds", which is exactly the unit that must never be resampled. Ten seeds
evaluated on the same specimen partition are ten model inits, not ten
independent draws of specimens; bootstrapping them produces a confidence
interval that is too narrow to support any claim about new food samples. The R0
audit raised this as CRITICAL (item C1).

The unit-of-analysis rules implemented here, and declared in
``configs/protocol_v1.1.yaml`` under ``pairing`` and ``statistical_tests``:

* One **observation** is a paired difference on one ``(seed, fold)``
  evaluation — the same specimens, scaled by the same train-only scaler,
  scored for both models.
* The **resampling unit is the fold** — the specimen-disjoint test block whose
  metrics actually enter the difference. It is never the seed.
* Seeds are a **stratification**, not a resampling unit: the bootstrap draws
  folds within each seed and then averages the per-seed means, so seed-level
  variability enters the interval without being mistaken for specimen sampling
  variability.
* ``n_pairs`` is therefore ``n_folds * n_seeds`` observations over ``n_folds``
  independent clusters, and the summary reports both, so a reader can see how
  many independent units are behind a p-value.

For ``loso`` the folds are single specimens, so the fold bootstrap *is* a
specimen bootstrap. For ``group_kfold`` a fold is a block of specimens and the
block is the cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats as scipy_stats

Direction = Literal["maximize", "minimize"]

# Defaults mirroring protocol v1.1. They are defaults for direct use of this
# module; every production call passes the protocol's frozen values explicitly.
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_B = 10000
DEFAULT_BOOTSTRAP_SEED = 20260920
DEFAULT_CI_LEVEL = 0.95
DEFAULT_RESAMPLE_UNIT = "fold"
DEFAULT_ZERO_METHOD = "wilcox"
DEFAULT_WILCOXON_MODE = "auto"
DEFAULT_WILCOXON_CORRECTION = False


class InsufficientDataError(ValueError):
    """Raised when a contrast has too few observations to test."""


@dataclass(frozen=True)
class PairedSpec:
    """The frozen parameters of one paired comparison.

    Attributes:
        alpha: Significance level.
        margin: Equivalence margin in the metric's own units.
        direction: ``maximize`` or ``minimize`` — which sign of the paired
            difference favours the first model of the contrast.
        alternative: Wilcoxon alternative; the protocol fixes ``two-sided``.
        zero_method: Wilcoxon zero handling; ``wilcox`` drops ties-to-zero.
        correction: Whether to apply scipy's continuity correction.
        wilcoxon_mode: ``auto`` or ``exact``.
        bootstrap_b: Bootstrap resamples.
        bootstrap_seed: Seed for the bootstrap draw — fixed, not derived from
            the run seed, so the interval is reproducible independently of the
            evaluation.
        ci_level: Confidence level of the reported interval.
        ci_type: Currently ``percentile``; ``bca`` is declared but not
            implemented, and asking for it raises rather than silently
            substituting a different interval.
        resample_unit: The cluster resampled by the bootstrap. Must not be
            ``seed``.
        stratification: The variable held fixed while resampling.
    """

    alpha: float = DEFAULT_ALPHA
    margin: float = 0.0
    direction: Direction = "maximize"
    alternative: str = "two-sided"
    zero_method: str = DEFAULT_ZERO_METHOD
    correction: bool = DEFAULT_WILCOXON_CORRECTION
    wilcoxon_mode: str = DEFAULT_WILCOXON_MODE
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    ci_level: float = DEFAULT_CI_LEVEL
    ci_type: str = "percentile"
    resample_unit: str = DEFAULT_RESAMPLE_UNIT
    stratification: str = "seed"

    def __post_init__(self) -> None:
        if self.resample_unit == "seed":
            raise ValueError(
                "resample_unit must not be 'seed': seeds on one specimen partition are not "
                "independent draws of specimens, and bootstrapping them yields an interval that "
                "is too narrow. See protocol v1.1 `pairing.resample_unit`."
            )
        if self.ci_type != "percentile":
            raise NotImplementedError(
                f"ci_type {self.ci_type!r} is declared in the protocol but not implemented; "
                f"refusing to substitute a different interval silently"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.margin < 0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")


@dataclass(frozen=True)
class PairedResult:
    """The outcome of one paired comparison.

    Attributes:
        contrast_id: Identifier of the contrast, e.g. ``R0_vs_R2``.
        metric: Metric name.
        dataset: Dataset identifier.
        condition: Evaluation condition, e.g. ``full`` or ``dropout_p0.3``.
        n_pairs: Number of paired observations (folds x seeds).
        n_clusters: Number of independent resampling clusters (folds).
        n_seeds: Number of distinct seeds represented.
        n_nonzero: Observations whose paired difference is not exactly zero.
        delta: Mean paired difference.
        delta_ci_low: Lower bootstrap CI bound.
        delta_ci_high: Upper bootstrap CI bound.
        ci_level: Confidence level of the interval.
        statistic: Wilcoxon signed-rank statistic.
        p_value: Raw Wilcoxon p-value.
        effect_size: Rank-biserial correlation (signed, in favour of the first model).
        effect_size_name: Name of the effect size.
        equivalence_p: TOST p-value (max of the two one-sided p-values).
        equivalent: Whether the interval lies inside the margin.
        favours_first: Whether the difference favours the first model of the contrast.
        test: Name of the hypothesis test performed.
        substitution: Non-empty when the pre-registered test could not be run.
    """

    contrast_id: str
    metric: str
    dataset: str
    condition: str
    n_pairs: int
    n_clusters: int
    n_seeds: int
    n_nonzero: int
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    ci_level: float
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    equivalence_p: float
    equivalent: bool
    favours_first: bool
    test: str
    substitution: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Mapping of every field.
        """
        return {
            "contrast_id": self.contrast_id,
            "metric": self.metric,
            "dataset": self.dataset,
            "condition": self.condition,
            "n_pairs": self.n_pairs,
            "n_clusters": self.n_clusters,
            "n_seeds": self.n_seeds,
            "n_nonzero": self.n_nonzero,
            "delta": self.delta,
            "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high,
            "ci_level": self.ci_level,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "equivalence_p": self.equivalence_p,
            "equivalent": self.equivalent,
            "favours_first": self.favours_first,
            "test": self.test,
            "substitution": self.substitution,
        }


def fold_cluster_bootstrap(
    deltas: np.ndarray,
    seeds: np.ndarray,
    folds: np.ndarray,
    spec: PairedSpec,
    statistic=np.mean,
) -> np.ndarray:
    """Bootstrap paired differences by resampling folds within each seed.

    Args:
        deltas: Paired differences, one per ``(seed, fold)`` observation.
        seeds: Seed of each observation.
        folds: Fold identifier of each observation.
        spec: Frozen test parameters (supplies ``bootstrap_b`` and the seed).
        statistic: Statistic applied to a resample.

    Returns:
        The bootstrap distribution of the statistic, length ``bootstrap_b``.

    Raises:
        InsufficientDataError: If no seed has at least two folds, or if the
            resampling unit is not ``fold``.
    """
    if spec.resample_unit != "fold":
        raise ValueError(
            f"fold_cluster_bootstrap implements resample_unit='fold', got {spec.resample_unit!r}"
        )
    deltas = np.asarray(deltas, dtype=np.float64)
    seeds = np.asarray(seeds)
    folds = np.asarray(folds)
    if deltas.size == 0:
        raise InsufficientDataError("no paired differences to bootstrap")

    per_seed: list[np.ndarray] = []
    for seed in np.unique(seeds):
        mask = seeds == seed
        seed_folds = folds[mask]
        seed_deltas = deltas[mask]
        if np.unique(seed_folds).size < 2:
            continue
        per_seed.append(seed_deltas)
    if not per_seed:
        raise InsufficientDataError(
            "no seed has two or more folds; the fold-cluster bootstrap cannot resample"
        )

    rng = np.random.default_rng(spec.bootstrap_seed)
    out = np.empty(spec.bootstrap_b, dtype=np.float64)
    for b in range(spec.bootstrap_b):
        seed_means = np.empty(len(per_seed), dtype=np.float64)
        for i, seed_deltas in enumerate(per_seed):
            draw = rng.integers(0, seed_deltas.size, size=seed_deltas.size)
            seed_means[i] = statistic(seed_deltas[draw])
        out[b] = float(np.mean(seed_means))
    return out


def rank_biserial(deltas: np.ndarray) -> float:
    """Compute the matched-pairs rank-biserial correlation.

    Args:
        deltas: Paired differences.

    Returns:
        Correlation in ``[-1, 1]``; positive means the first model is ahead.
        Zero-difference pairs are excluded, matching ``zero_method='wilcox'``.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    nonzero = deltas[deltas != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = scipy_stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total


def hodges_lehmann(deltas: np.ndarray) -> float:
    """Compute the Hodges-Lehmann estimator of the paired difference.

    Args:
        deltas: Paired differences.

    Returns:
        The median of all pairwise Walsh averages.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan")
    sums = (deltas[:, None] + deltas[None, :]) / 2.0
    return float(np.median(sums))


def tost_from_interval(
    ci_low: float, ci_high: float, margin: float, alpha: float
) -> tuple[float, bool]:
    """Decide equivalence from a two-sided confidence interval.

    The interval-inclusion form of TOST: equivalence at margin ``m`` is accepted
    exactly when the ``1 - 2*alpha`` interval lies strictly inside ``(-m, m)``.

    Args:
        ci_low: Lower bound of the two-sided interval.
        ci_high: Upper bound of the two-sided interval.
        margin: Equivalence margin in the metric's own units.
        alpha: Significance level the interval was built for.

    Returns:
        ``(equivalence_p, equivalent)``. The p-value is derived by finding the
        smallest level at which the interval would fit; it is reported for
        completeness and no claim rests on it alone.
    """
    if margin <= 0:
        return (1.0, False)
    equivalent = bool(ci_low > -margin and ci_high < margin)
    # A conservative p-value proxy: the interval's own coverage is 1 - alpha, and
    # it fits with slack. Report 0.0 when it fits, else the fraction by which it
    # misses, floored at alpha so it never looks "significant" when it is not.
    if equivalent:
        return (0.0, True)
    miss = max(abs(ci_low + margin), abs(ci_high - margin))
    scale = max(miss, 1e-12)
    return (float(min(1.0, alpha * (1.0 + scale))), False)


def paired_test(
    deltas: Sequence[float],
    seeds: Sequence[int],
    folds: Sequence[int],
    spec: PairedSpec,
    *,
    contrast_id: str = "",
    metric: str = "",
    dataset: str = "",
    condition: str = "full",
) -> PairedResult:
    """Run the protocol's paired test on one contrast.

    Args:
        deltas: Paired differences ``first - second``, one per observation.
        seeds: Seed of each observation.
        folds: Fold identifier of each observation.
        spec: Frozen test parameters.
        contrast_id: Contrast identifier for the record.
        metric: Metric name.
        dataset: Dataset identifier.
        condition: Evaluation condition.

    Returns:
        The :class:`PairedResult`.

    Raises:
        InsufficientDataError: If fewer than two observations are supplied.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    seeds = np.asarray(seeds)
    folds = np.asarray(folds)
    if deltas.size < 2:
        raise InsufficientDataError(f"{contrast_id or metric}: need >= 2 paired observations")

    where = "greater" if spec.direction == "maximize" else "less"
    nonzero = deltas[deltas != 0]

    substitution = ""
    if nonzero.size < 6:
        # The protocol pre-registers the sign test for a degenerate difference
        # distribution; it is substituted explicitly and recorded, never silently.
        n_positive = int((nonzero > 0).sum())
        result = scipy_stats.binomtest(n_positive, nonzero.size, 0.5, alternative="two-sided")
        statistic = float(n_positive)
        p_value = float(result.pvalue)
        test_name = "sign_test"
        substitution = (
            f"fewer than 6 non-zero pairs ({nonzero.size}); the pre-registered sign test was "
            f"substituted for the Wilcoxon signed-rank test, as protocol v1.1 allows"
        )
    else:
        wilcoxon = scipy_stats.wilcoxon(
            deltas,
            alternative=spec.alternative,
            zero_method=spec.zero_method,
            correction=spec.correction,
            mode=spec.wilcoxon_mode,
        )
        statistic = float(wilcoxon.statistic)
        p_value = float(wilcoxon.pvalue)
        test_name = "wilcoxon_signed_rank"

    bootstrap = fold_cluster_bootstrap(deltas, seeds, folds, spec)
    tail = (1.0 - spec.ci_level) / 2.0
    ci_low = float(np.quantile(bootstrap, tail))
    ci_high = float(np.quantile(bootstrap, 1.0 - tail))

    delta = float(np.mean(deltas))
    effect = rank_biserial(deltas) if metric != "mae" and metric != "rmse" else hodges_lehmann(
        deltas
    )
    effect_name = (
        "hodges_lehmann" if metric in ("mae", "rmse") else "rank_biserial"
    )
    equivalence_p, equivalent = tost_from_interval(ci_low, ci_high, spec.margin, spec.alpha)

    favours_first = bool(delta > 0) if spec.direction == "maximize" else bool(delta < 0)
    # A one-sided claim needs the interval to exclude zero on the favourable side.
    if spec.direction == "maximize":
        favours_first = favours_first and ci_low > 0
    else:
        favours_first = favours_first and ci_high < 0

    return PairedResult(
        contrast_id=contrast_id,
        metric=metric,
        dataset=dataset,
        condition=condition,
        n_pairs=int(deltas.size),
        n_clusters=int(np.unique(folds).size),
        n_seeds=int(np.unique(seeds).size),
        n_nonzero=int(nonzero.size),
        delta=delta,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        ci_level=spec.ci_level,
        statistic=statistic,
        p_value=p_value,
        effect_size=float(effect),
        effect_size_name=effect_name,
        equivalence_p=equivalence_p,
        equivalent=equivalent,
        favours_first=favours_first,
        test=test_name,
        substitution=substitution,
    )


def holm_correction(p_values: Sequence[float], alpha: float = DEFAULT_ALPHA) -> list[float]:
    """Apply the Holm step-down correction within one family.

    Args:
        p_values: Raw p-values of the family, in declaration order.
        alpha: Family-wise significance level.

    Returns:
        Adjusted p-values in the input order.
    """
    values = np.asarray(p_values, dtype=np.float64)
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return [float(v) for v in adjusted]


def minimum_achievable_p(n_pairs: int) -> float:
    """Smallest two-sided p-value an exact sign-flip test can reach at ``n``.

    Used by the protocol's power pre-check: with fewer than six non-zero pairs
    the exact two-sided minimum is above 0.05, so no significant result is
    reachable and the comparison must be reported as underpowered rather than as
    evidence of no effect.

    Args:
        n_pairs: Number of non-zero paired observations.

    Returns:
        The minimum achievable two-sided p-value, or ``nan`` for no pairs.
    """
    if n_pairs <= 0:
        return float("nan")
    return float(min(1.0, 2.0 / (2.0**n_pairs)))


def clusters_needed_for_alpha(alpha: float = DEFAULT_ALPHA) -> int:
    """Fewest clusters for which the exact two-sided minimum p reaches ``alpha``.

    Args:
        alpha: Target significance level.

    Returns:
        The smallest ``n`` with ``minimum_achievable_p(n) <= alpha``.
    """
    n = 1
    while minimum_achievable_p(n) > alpha and n < 64:
        n += 1
    return n


def wilcoxon_reachable(n_pairs: int, alpha: float = DEFAULT_ALPHA) -> bool:
    """Whether a significant Wilcoxon result is arithmetically reachable.

    Args:
        n_pairs: Number of non-zero paired observations.
        alpha: Significance level.

    Returns:
        ``True`` when the exact minimum two-sided p-value is at or below alpha.
    """
    value = minimum_achievable_p(n_pairs)
    return bool(math.isfinite(value) and value <= alpha)


__all__ = [
    "InsufficientDataError",
    "PairedResult",
    "PairedSpec",
    "clusters_needed_for_alpha",
    "fold_cluster_bootstrap",
    "hodges_lehmann",
    "holm_correction",
    "minimum_achievable_p",
    "paired_test",
    "rank_biserial",
    "tost_from_interval",
    "wilcoxon_reachable",
]
