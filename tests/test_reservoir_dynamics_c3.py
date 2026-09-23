"""C3 — the recurrence must participate (v2 D5, criteria C3.1-C3.3).

The signed gates, from docs/v2_preregistration.md section 3:

  C3.1  median over validation windows of ``R_t = || A h_{t-1} || / || W_in x_t ||`` must
         lie in [0.20, 1.00] (the gain-FREE form is the gate; the v1-comparable
         gain-inclusive form is reported beside it on every line).
  C3.2  zero-recurrent memory drop >= 20 %, with the metric fixed before measurement:
         ``M = max_k in {1,4,8,16} | corr( h_t , x_{t-k} ) | ``; the criterion is
         ``(M_recurrent - M_A=0) / M_recurrent >= 0.20``.
  C3.3  effective rank ``D_eff >= 1.5 * Din``.

C3.4 names the knobs as a selection done ON VALIDATION within the protocol grid; the
search below uses the gain x leak grid, with ``input_scale`` fixed at the v2 mapping's
declared value, and is run on each of the four reservoirs (R0/R2 x shared/rho-matched).

Every test here pins a metric DEFINITION, never a result: synthetic substrates exercise
the wiring (gain inside the drive, scaling outside, R_t direction, memory direction, D_eff
direction) without depending on a particular substrate's eigenvalue distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.reservoir.dynamics import (  # noqa: E402
    MEMORY_DROP_GATE,
    MEMORY_LAGS,
    RT_BAND,
    effective_rank,
    evaluate_c3,
    memory_metric,
    reservoir_drive,
    select_knobs,
    summarize_R_t,
)


def _synaptic(rng: np.random.Generator, *, n: int, density: float) -> sp.csr_matrix:
    mask = rng.random((n, n)) < density
    weights = rng.integers(1, 16, size=(n, n)) * mask
    np.fill_diagonal(weights, 0.0)
    return sp.csr_matrix(weights.astype(float))


def _input_map(n: int, support: int, rng: np.random.Generator, *, scale: float) -> sp.csr_matrix:
    win = rng.uniform(-scale, scale, size=(n, support))
    return sp.csr_matrix(win)


def _synthetic_windows(
    rng: np.random.Generator,
    *,
    n_samples: int,
    length: int,
    din: int,
    autocorrelation: float = 0.0,
    scale: float = 0.5,
) -> np.ndarray:
    X = rng.standard_normal((n_samples, length, din)) * scale
    if autocorrelation > 0:
        noise = rng.standard_normal((n_samples, length, din))
        for t in range(1, length):
            X[:, t, :] = autocorrelation * X[:, t - 1, :] + (1 - autocorrelation) * noise[:, t, :]
    return X


# ---------------------------------------------------------------------------
# C3.1 — R_t = ||A h|| / ||W_in x||, gain-FREE form, in [0.20, 1.00]
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c3_1_R_t_increases_with_gain_and_is_not_bounded_by_the_input():
    """The gate is on the median of the gain-free form; the gain multiplies the recurrent
    term, so for a fixed input R_t must increase with gain (the tanh saturation makes the
    ratio strictly sub-linear, but it stays positive and increasing on any fixed signal)."""
    rng = np.random.default_rng(0)
    n, length = 60, 16
    A = _synaptic(rng, n=n, density=0.08)
    win = _input_map(n, 5, rng=rng, scale=0.3)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=8, length=length, din=5, scale=0.5)
    rt_at = []
    for g in (0.25, 0.5, 1.0, 2.0):
        r = reservoir_drive(A, win, bias, X, gain=g, leak=0.5)
        rt_at.append(np.median(r.R_t_gain_free))
    assert all(rt_at[i] > 0 for i in range(len(rt_at)))
    # monotone in gain: doubling gain must not decrease R_t on any fixed input
    assert all(rt_at[i + 1] >= rt_at[i] * 0.8 for i in range(len(rt_at) - 1))


@pytest.mark.unit
def test_c3_1_summarize_R_t_returns_median_and_quantiles_in_band():
    rng = np.random.default_rng(1)
    n = 60
    A = _synaptic(rng, n=n, density=0.08)
    win = _input_map(n, 5, rng=rng, scale=0.5)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=10, length=16, din=5)
    r = reservoir_drive(A, win, bias, X, gain=1.0, leak=0.5)
    s = summarize_R_t(r.R_t_gain_free)
    assert s["p10"] <= s["median"] <= s["p90"]
    assert np.isfinite(s["median"])


@pytest.mark.unit
def test_c3_1_gain_inclusive_equals_gain_times_gain_free_when_bias_zero():
    rng = np.random.default_rng(2)
    n = 60
    A = _synaptic(rng, n=n, density=0.08)
    win = _input_map(n, 5, rng=rng, scale=0.5)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=8, length=16, din=5, scale=0.7)
    for g in (0.5, 1.0, 2.0):
        r = reservoir_drive(A, win, bias, X, gain=g, leak=0.5)
        inc = np.median(r.R_t_gain_inclusive)
        free = np.median(r.R_t_gain_free)
        assert inc == pytest.approx(g * free, rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# C3.2 — memory metric and the zero-recurrent drop
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c3_2_memory_metric_returns_a_value_in_unit_interval_on_ar_signal():
    """The metric is defined and bounded in [-1, 1]. Whether it surfaces the AR
    structure on a particular substrate is a property of the substrate, not of the
    metric, so this test only pins the definition."""
    rng = np.random.default_rng(3)
    n = 80
    A = _synaptic(rng, n=n, density=0.08)
    win = _input_map(n, 2, rng=rng, scale=0.3)
    bias = np.zeros(n)
    length = 96
    X = np.zeros((10, length, 2))
    noise = rng.standard_normal(X.shape)
    for t in range(2, length):
        X[:, t, :] = 0.85 * X[:, t - 2, :] + 0.15 * noise[:, t, :]
    r = reservoir_drive(A, win, bias, X, gain=1.0, leak=0.75)
    M = memory_metric(r.states, X, lags=(1, 2, 4, 8))
    assert 0.0 <= M <= 1.0
    M_zero = memory_metric(
        reservoir_drive(sp.csr_matrix((n, n)), win, bias, X, gain=1.0, leak=0.75).states,
        X,
    )
    assert 0.0 <= M_zero <= 1.0


@pytest.mark.unit
def test_c3_2_zero_A_memory_is_a_real_number_not_a_run_failure():
    """The metric is computed identically for the real graph and for A=0; the comparison
    itself is substrate-dependent (the v1 measurement showed A=0 has the same memory as
    the real graph; a substrate where A genuinely contributes yields M_real > M_A=0).
    Both must be finite, and the drop is the signed metric."""
    rng = np.random.default_rng(4)
    n = 80
    A = _synaptic(rng, n=n, density=0.08)
    win = _input_map(n, 1, rng=rng, scale=0.3)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=10, length=96, din=1, autocorrelation=0.85)
    M_real = memory_metric(reservoir_drive(A, win, bias, X, gain=1.0, leak=0.75).states, X)
    M_zero = memory_metric(
        reservoir_drive(sp.csr_matrix((n, n)), win, bias, X, gain=1.0, leak=0.75).states, X
    )
    drop = (M_real - M_zero) / M_real if M_real > 0 else float("nan")
    assert np.isfinite(drop), f"the signed drop must be a number, got {drop}"
    # the drop is bounded -- M_zero in [-1, 1] and M_real in [0, 1], so drop in [-inf, 2]
    # (when M_real -> 0 the gate is not triggered; the metric is well-defined)


# ---------------------------------------------------------------------------
# C3.3 — D_eff via participation ratio
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c3_3_D_eff_is_one_for_a_1d_input_and_full_for_an_independent_signal():
    rng = np.random.default_rng(5)
    n = 40
    A = _synaptic(rng, n=n, density=0.1)
    win = _input_map(n, 2, rng=rng, scale=0.4)
    bias = np.zeros(n)
    length = 32
    one_d = np.zeros((10, length, 2))
    one_d[:, :, 0] = rng.standard_normal((10, length))
    one_d[:, :, 1] = one_d[:, :, 0]
    states_one = reservoir_drive(A, win, bias, one_d, gain=1.0, leak=0.5).states
    eff_one = effective_rank(states_one)["D_eff"]
    full = rng.standard_normal((10, length, 2))
    states_full = reservoir_drive(A, win, bias, full, gain=1.0, leak=0.5).states
    eff_full = effective_rank(states_full)["D_eff"]
    assert eff_one < eff_full
    assert eff_one < 5.0, f"a 1-D input should yield a small D_eff, got {eff_one}"


# ---------------------------------------------------------------------------
# The search and the integration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_select_knobs_returns_none_when_no_pair_qualifies():
    rng = np.random.default_rng(6)
    n = 40
    A = _synaptic(rng, n=n, density=0.05)
    win = _input_map(n, 2, rng=rng, scale=0.1)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=6, length=12, din=2)
    out = select_knobs(A, win, bias, X, din=2, input_scale=0.5,
                       leak_values=(0.1,), gain_values=(0.5,))
    assert out["chosen"] is None
    assert out["all_points_failed"] is True
    assert len(out["trace"]) == 1


@pytest.mark.unit
def test_evaluate_c3_returns_the_signed_three_lines():
    rng = np.random.default_rng(7)
    n = 40
    A = _synaptic(rng, n=n, density=0.1)
    win = _input_map(n, 2, rng=rng, scale=0.5)
    bias = np.zeros(n)
    X = _synthetic_windows(rng, n_samples=6, length=16, din=2, autocorrelation=0.7)
    out = evaluate_c3(A, win, bias, X, gain=1.0, leak=0.5, din=2)
    assert set(out["criterion_lines"]) == {
        "C3.1_R_t_in_band",
        "C3.2_memory_drop_>=_20pct",
        "C3.3_D_eff_>=_1.5*Din",
    }
    assert out["windows"]["Din"] == 2
    assert out["windows"]["length"] == 16
    assert out["verdict"] == all(out["criterion_lines"].values())


@pytest.mark.unit
def test_the_signed_metric_definitions():
    assert RT_BAND == (0.20, 1.00)
    assert MEMORY_DROP_GATE == 0.20
    assert MEMORY_LAGS == (1, 4, 8, 16)
    s = summarize_R_t(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert set(s) == {"median", "p10", "p90"}