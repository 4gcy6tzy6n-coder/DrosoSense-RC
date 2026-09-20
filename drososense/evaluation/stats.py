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
* Seeds are a **stratification**, not a resampling unit: the seeds are averaged
  inside each cluster before the clusters are resampled, so the interval is
  driven by cluster-to-cluster variation and does not narrow as the same
  specimens are re-measured under more seeds. (v1.1 resampled within each seed
  and averaged afterwards, which divided the interval's variance by the seed
  count. See :func:`fold_cluster_bootstrap`.)
* ``n_pairs`` is therefore ``n_folds * n_seeds`` observations over ``n_folds``
  independent clusters, and the summary reports both, so a reader can see how
  many independent units are behind a p-value.

For ``loso`` the folds are single specimens, so the fold bootstrap *is* a
specimen bootstrap. For ``group_kfold`` a fold is a block of specimens and the
block is the cluster.

The decisive test is the **cluster-level** one (protocol v1.2 §10). The R0.1
statistical review (DATA-18, item C2) found that a Wilcoxon over the
``n_folds * n_seeds`` paired differences reads a pseudoreplicated p-value: ten
seeds on one specimen partition are ten re-initialisations of the same five
specimens, so 50 pairs contain five independent units, and a p-value computed
as if there were 50 is not a p-value for any claim about new food samples. On
the delivered D2 table that showed up as ``p = 5.45e-13`` beside a
cluster-level exact p of ``0.0625``.

So the decision is taken on one mean per cluster:

* observations are collapsed to ``n_clusters`` cluster means (the mean paired
  difference of each held-out block, taken over the seeds);
* the pre-registered test is the exact two-sided sign test on those means, whose
  smallest attainable p is ``2 / 2**n_clusters`` — exactly
  :func:`minimum_achievable_p`, which is why that number is a *decision input*
  and not a footnote;
* the pair-level Wilcoxon is still computed and reported, because it is what the
  earlier version of the protocol published and a reader must be able to see the
  difference — but it is marked descriptive and no gate reads it.

With five clusters the exact floor is 0.0625, above alpha, so a significant
cluster-level result is not reachable on such a dataset. That is the honest
statement the protocol's own ``split_protocol.power_note`` already made, and it
is now enforced by the test rather than contradicted by it.
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
DEFAULT_PRIMARY_TEST = "cluster_sign_test"

# Below this a double-precision p-value carries no information: `1 - cdf` has
# cancelled every significant digit, and the number that survives is an artefact
# of float64 rather than of the data (R0.1 review item, LOW). Anything this small
# is reported as "<1e-12".
P_VALUE_FLOOR = 1.0e-12
_P_VALUE_FLOOR_TEXT = "<1e-12"


def format_p_value(value: float) -> str:
    """Render a p-value so that a float64 cancellation artefact is not published.

    Args:
        value: The p-value.

    Returns:
        The value formatted, or ``"<1e-12"`` when it is below the floor at
        which double precision still carries information.
    """
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    if float(value) < P_VALUE_FLOOR:
        return _P_VALUE_FLOOR_TEXT
    return f"{float(value):.6g}"


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
        primary_test: The test the decision is taken on. Only
            ``cluster_sign_test`` is implemented: a p-value computed over the
            ``n_folds * n_seeds`` pairs is pseudoreplicated and may not decide a
            gate (protocol v1.2 §10).
        effect_size_name: Which effect size to report. It is chosen by TASK, not
            by metric name — ``r2`` is a regression metric and takes the
            Hodges-Lehmann estimator even though its name does not look like
            ``mae`` or ``rmse`` (R0.1 review item M3).
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
    primary_test: str = DEFAULT_PRIMARY_TEST
    effect_size_name: str = "rank_biserial"

    def __post_init__(self) -> None:
        if self.primary_test != DEFAULT_PRIMARY_TEST:
            raise NotImplementedError(
                f"primary_test {self.primary_test!r} is not implemented. A pair-level test over "
                f"n_folds * n_seeds observations is pseudoreplicated and may not decide a gate "
                f"(protocol v1.2 §10); refusing to substitute one silently."
            )
        if self.effect_size_name not in ("rank_biserial", "hodges_lehmann"):
            raise ValueError(
                f"effect_size_name must be 'rank_biserial' or 'hodges_lehmann', "
                f"got {self.effect_size_name!r}"
            )
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
        n_clusters_nonzero: Clusters whose mean difference is not exactly zero —
            the effective n of the decisive sign test.
        n_seeds: Number of distinct seeds represented.
        n_nonzero: Pair-level observations whose difference is not exactly zero.
        delta: Mean paired difference.
        delta_ci_low: Lower bootstrap CI bound.
        delta_ci_high: Upper bootstrap CI bound.
        ci_level: Confidence level of the interval.
        statistic: Statistic of the DECISIVE test: clusters favouring the first
            model.
        p_value: p-value of the decisive test — the exact two-sided sign test
            over cluster means. This is the value gates read (through
            ``p_holm``).
        paired_statistic: Statistic of the descriptive pair-level test.
        p_paired_wilcoxon: p-value of the descriptive pair-level test. It is
            pseudoreplicated over ``(seed, fold)`` and is reported for
            continuity with the published tables only; no gate reads it.
        minimum_achievable_p_over_clusters: Smallest two-sided p the decisive
            test can reach at ``n_clusters_nonzero``. Above alpha means no
            significant result is reachable, whatever the data says.
        effect_size: The chosen effect size (signed, in favour of the first model).
        effect_size_name: Name of the effect size.
        tost_proxy_p: Conservative p-value PROXY from the interval-inclusion
            equivalence rule. It is not a TOST p-value: it is 0.0 when the
            interval fits inside the margin and an alpha-scaled miss otherwise.
            The decision rests on ``equivalent``; the number is reported for
            completeness only (R0.1 review item M4).
        equivalent: Whether the interval lies strictly inside the margin.
        favours_first: Whether the difference favours the first model of the contrast.
        test: Name of the decisive hypothesis test performed.
        paired_test: Name of the descriptive pair-level test.
        substitution: Non-empty when the pre-registered pair-level test could not
            be run.
    """

    contrast_id: str
    metric: str
    dataset: str
    condition: str
    n_pairs: int
    n_clusters: int
    n_clusters_nonzero: int
    n_seeds: int
    n_nonzero: int
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    ci_level: float
    statistic: float
    p_value: float
    paired_statistic: float
    p_paired_wilcoxon: float
    minimum_achievable_p_over_clusters: float
    effect_size: float
    effect_size_name: str
    tost_proxy_p: float
    equivalent: bool
    favours_first: bool
    test: str
    paired_test: str
    substitution: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Mapping of every field, plus the rendered p-values so that a value
            below the float64 floor is published as ``<1e-12`` rather than as a
            cancellation artefact.
        """
        return {
            "contrast_id": self.contrast_id,
            "metric": self.metric,
            "dataset": self.dataset,
            "condition": self.condition,
            "n_pairs": self.n_pairs,
            "n_clusters": self.n_clusters,
            "n_clusters_nonzero": self.n_clusters_nonzero,
            "n_seeds": self.n_seeds,
            "n_nonzero": self.n_nonzero,
            "delta": self.delta,
            "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high,
            "ci_level": self.ci_level,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_reported": format_p_value(self.p_value),
            "paired_statistic": self.paired_statistic,
            "p_paired_wilcoxon": self.p_paired_wilcoxon,
            "p_paired_wilcoxon_reported": format_p_value(self.p_paired_wilcoxon),
            "minimum_achievable_p_over_clusters": self.minimum_achievable_p_over_clusters,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "tost_proxy_p": self.tost_proxy_p,
            "equivalent": self.equivalent,
            "favours_first": self.favours_first,
            "test": self.test,
            "paired_test": self.paired_test,
            "substitution": self.substitution,
        }


def fold_cluster_bootstrap(
    deltas: np.ndarray,
    folds: np.ndarray,
    spec: PairedSpec,
    statistic=np.mean,
) -> np.ndarray:
    """Bootstrap paired differences by resampling the held-out clusters.

    The clusters — not the seeds — are what is resampled, and the seeds are
    averaged INSIDE each cluster first. That ordering is the whole point, and
    v1.2 changed it:

    v1.1 resampled folds within each seed and then averaged the per-seed means.
    Because each seed's resample was drawn independently, averaging ``S`` of them
    divided the interval's variance by ``S`` — so adding seeds narrowed the
    interval even though it added no specimens. Measured on ten clusters: the
    interval was 0.018 wide at one seed, 0.008 at five and 0.0039 at twenty. An
    interval that narrows as the same specimens are re-measured cannot support a
    claim about new specimens, and it is the interval side of the same
    pseudoreplication the decisive test was moved off.

    With the seeds averaged inside each cluster and the clusters resampled, the
    interval is driven by cluster-to-cluster variation alone and is invariant to
    the seed count. That makes it the natural companion to the cluster-level
    decisive test in :func:`cluster_sign_test`, and it is the interval that
    ``pairing.resample_unit_detail`` describes when it says the fold bootstrap is
    a specimen bootstrap.

    Args:
        deltas: Paired differences, one per ``(seed, fold)`` observation.
        folds: Fold identifier of each observation.
        spec: Frozen test parameters (supplies ``bootstrap_b`` and the seed).
        statistic: Statistic applied to a resample, called with ``axis=1`` over a
            ``(bootstrap_b, n_clusters)`` index array.

    Returns:
        The bootstrap distribution of the statistic, length ``bootstrap_b``.

    Raises:
        InsufficientDataError: If there are no observations, or fewer than two
            clusters to resample.
        ValueError: If the resampling unit is not ``fold``.
    """
    if spec.resample_unit != "fold":
        raise ValueError(
            f"fold_cluster_bootstrap implements resample_unit='fold', got {spec.resample_unit!r}"
        )
    deltas = np.asarray(deltas, dtype=np.float64)
    folds = np.asarray(folds)
    if deltas.size == 0:
        raise InsufficientDataError("no paired differences to bootstrap")

    means = cluster_means(deltas, folds)
    if means.size < 2:
        raise InsufficientDataError(
            f"{means.size} cluster(s); the cluster bootstrap cannot resample fewer than two"
        )

    rng = np.random.default_rng(spec.bootstrap_seed)
    draw = rng.integers(0, means.size, size=(spec.bootstrap_b, means.size))
    return np.asarray(statistic(means[draw], axis=1), dtype=np.float64)


def cluster_means(deltas: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Collapse the ``(seed, fold)`` observations to one mean per cluster.

    The seeds are a stratification, so the seeds inside a cluster are averaged
    rather than counted: with ten seeds and five folds, the cluster mean is the
    mean of the ten model re-initialisations evaluated on that held-out block.
    Averaging is what stops seed-level variability being read as specimen
    sampling variability.

    Args:
        deltas: Paired differences, one per ``(seed, fold)`` observation.
        folds: Fold identifier of each observation.

    Returns:
        One mean per distinct fold, in sorted fold order.

    Raises:
        InsufficientDataError: If there are no observations.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    folds = np.asarray(folds)
    if deltas.size == 0:
        raise InsufficientDataError("no paired differences to collapse into clusters")
    out = [float(np.mean(deltas[folds == fold])) for fold in np.unique(folds)]
    return np.asarray(out, dtype=np.float64)


def cluster_sign_test(means: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided sign test over cluster means.

    This is the decisive test of protocol v1.2 §10. Its smallest attainable
    p-value at ``n`` non-zero clusters is ``2 / 2**n``, which is exactly
    :func:`minimum_achievable_p` — so the reachability floor reported beside
    every line is the floor of the test that actually produced the p-value,
    rather than a footnote about a different test.

    Args:
        means: One mean paired difference per cluster.

    Returns:
        ``(p_value, n_nonzero_clusters, n_favouring_first)``.
    """
    means = np.asarray(means, dtype=np.float64)
    nonzero = means[means != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return (1.0, 0, 0)
    positive = int((nonzero > 0).sum())
    negative = n - positive
    k = max(positive, negative)
    p_value = float(scipy_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return (p_value, n, positive)


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

    The second return value is a PROXY, not a TOST p-value, and is named
    ``tost_proxy_p`` on the result for that reason (R0.1 review item M4): a real
    TOST p is ``max`` of two one-sided p-values computed from the data, whereas
    this is ``0.0`` whenever the interval fits and an alpha-scaled miss
    otherwise. Reporting it under the name ``equivalence_p`` invited a reader to
    treat it as a significance level. The decision rests on ``equivalent``,
    which is the strict interval inclusion.

    Args:
        ci_low: Lower bound of the two-sided interval.
        ci_high: Upper bound of the two-sided interval.
        margin: Equivalence margin in the metric's own units.
        alpha: Significance level the interval was built for.

    Returns:
        ``(tost_proxy_p, equivalent)``.
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

    nonzero = deltas[deltas != 0]

    # --- the descriptive pair-level test -----------------------------------
    # Pseudoreplicated by construction, and kept for continuity with the tables
    # the earlier protocol published. No gate reads this column.
    substitution = ""
    if nonzero.size == 0:
        # Every pair is exactly zero. There is no sign to count and nothing to
        # test; `binomtest` refuses n = 0, and reporting a p-value of 1.0 with no
        # test named is the honest outcome. This is the degenerate case the
        # pre-registered sign test exists for, taken to its limit.
        paired_statistic = 0.0
        p_paired_wilcoxon = 1.0
        paired_test_name = "sign_test"
        substitution = (
            "every paired difference is exactly zero; there is no non-zero pair to test"
        )
    elif nonzero.size < 6:
        # The protocol pre-registers the sign test for a degenerate difference
        # distribution; it is substituted explicitly and recorded, never silently.
        n_positive_pairs = int((nonzero > 0).sum())
        result = scipy_stats.binomtest(
            n_positive_pairs, nonzero.size, 0.5, alternative="two-sided"
        )
        paired_statistic = float(n_positive_pairs)
        p_paired_wilcoxon = float(result.pvalue)
        paired_test_name = "sign_test"
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
        paired_statistic = float(wilcoxon.statistic)
        p_paired_wilcoxon = float(wilcoxon.pvalue)
        paired_test_name = "wilcoxon_signed_rank"

    # --- the decisive cluster-level test -----------------------------------
    means = cluster_means(deltas, folds)
    p_value, n_clusters_nonzero, n_clusters_first = cluster_sign_test(means)
    minimum_p = minimum_achievable_p(n_clusters_nonzero)

    bootstrap = fold_cluster_bootstrap(deltas, folds, spec)
    tail = (1.0 - spec.ci_level) / 2.0
    ci_low = float(np.quantile(bootstrap, tail))
    ci_high = float(np.quantile(bootstrap, 1.0 - tail))

    delta = float(np.mean(deltas))
    effect = (
        hodges_lehmann(deltas)
        if spec.effect_size_name == "hodges_lehmann"
        else rank_biserial(deltas)
    )
    tost_proxy_p, equivalent = tost_from_interval(
        ci_low, ci_high, spec.margin, spec.alpha
    )

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
        n_clusters_nonzero=n_clusters_nonzero,
        n_seeds=int(np.unique(seeds).size),
        n_nonzero=int(nonzero.size),
        delta=delta,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        ci_level=spec.ci_level,
        statistic=float(n_clusters_first),
        p_value=p_value,
        paired_statistic=paired_statistic,
        p_paired_wilcoxon=p_paired_wilcoxon,
        minimum_achievable_p_over_clusters=minimum_p,
        effect_size=float(effect),
        effect_size_name=spec.effect_size_name,
        tost_proxy_p=tost_proxy_p,
        equivalent=equivalent,
        favours_first=favours_first,
        test=spec.primary_test,
        paired_test=paired_test_name,
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
    """Smallest two-sided p-value an exact sign test can reach at ``n``.

    This is the floor of the decisive cluster-level test, so it is a DECISION
    INPUT and not a footnote (protocol v1.2 §10): with fewer than six non-zero
    clusters the floor is above 0.05, no significant result is reachable, and
    the comparison must be reported as underpowered rather than as evidence of
    no effect. ``gates.sig`` reads it for exactly that reason.

    Args:
        n_pairs: Number of non-zero clusters.

    Returns:
        The minimum achievable two-sided p-value, or ``nan`` for no clusters.
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
    "P_VALUE_FLOOR",
    "InsufficientDataError",
    "PairedResult",
    "PairedSpec",
    "cluster_means",
    "cluster_sign_test",
    "clusters_needed_for_alpha",
    "fold_cluster_bootstrap",
    "format_p_value",
    "hodges_lehmann",
    "holm_correction",
    "minimum_achievable_p",
    "paired_test",
    "rank_biserial",
    "tost_from_interval",
    "wilcoxon_reachable",
]
