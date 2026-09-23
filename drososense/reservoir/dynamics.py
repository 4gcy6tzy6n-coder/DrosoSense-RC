"""Reservoir dynamics metrics for the v2 C3 gate (recurrence/input ratio, memory, rank).

The three gates (D5, docs/v2_preregistration.md section 3):

    C3.1  median over validation windows of ``R_t = || A h_{t-1} ||_2 / || W_in x_t ||_2``
          (the GAIN-FREE form is the gate; the v1-comparable gain-inclusive form
          ``|| gain * A * h_{t-1} || / || W_in * x_t + b ||`` is reported beside it).

    C3.2  zero-recurrent memory drop >= 20 %, with the memory metric fixed BEFORE
          measurement: ``M = max_{k in {1,4,8,16}} | corr( h_t , x_{t-k} ) |`` over the
          validation windows, computed identically for the real graph and for ``A := 0``;
          the criterion is ``(M_recurrent - M_{A=0}) / M_recurrent >= 0.20``.

    C3.3  ``D_eff >= 1.5 * Din`` (effective rank of the state matrix).

The module is pure-numpy / pure-scipy: it consumes the matrices the C4 layer produces and
windows the user supplies. It carries no run-record state and fits no model -- it is a
metric, not a learner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: The memory horizon the pre-registration fixed before any measurement.
MEMORY_LAGS: tuple[int, ...] = (1, 4, 8, 16)

#: The gate thresholds. Quoted from D5.
RT_BAND = (0.20, 1.00)
MEMORY_DROP_GATE = 0.20
DEFF_FACTOR_GATE = 1.5


@dataclass(frozen=True)
class DynamicsResult:
    """The output of one reservoir drive on a batch of windows.

    Attributes:
        R_t_gain_free: ``|| A h_{t-1} ||_2 / || W_in x_t ||_2`` per time step.
        R_t_gain_inclusive: v1-comparable form, ``|| g * A h_{t-1} || / || W_in x_t + b ||``.
        states: ``(n_samples, length, N)`` -- the hidden states at every step.
        length: The window length the drive ran over.
    """

    R_t_gain_free: np.ndarray
    R_t_gain_inclusive: np.ndarray
    states: np.ndarray
    length: int


def reservoir_drive(
    A: sp.spmatrix,
    W_in: sp.spmatrix,
    bias: np.ndarray,
    X: np.ndarray,
    *,
    gain: float,
    leak: float,
    washout: int = 0,
    pool: str = "last",
) -> DynamicsResult:
    """Run the canonical leaky update on a batch of windows.

    Args:
        A: Reservoir adjacency (sparse CSR, shape (N, N)).
        W_in: Input projection (sparse CSR or dense, shape (N, Din)).
        bias: Bias vector (shape (N,)).
        X: Windows, shape (n_samples, length, Din).
        gain: Spectral gain ``g``.
        leak: Leaky integration rate ``alpha``.
        washout: Number of initial steps to discard before collecting ``R_t``.
        pool: Reserved for future pooling options; ignored.

    Returns:
        A :class:`DynamicsResult` carrying the two ``R_t`` forms and the full state
        trajectory.
    """
    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)
    W_in_csr = W_in.tocsr() if sp.issparse(W_in) else sp.csr_matrix(W_in)
    bias = np.asarray(bias, dtype=np.float64).ravel()
    if W_in_csr.shape[0] != A_csr.shape[0]:
        raise ValueError(
            f"W_in rows ({W_in_csr.shape[0]}) do not match A rows ({A_csr.shape[0]})"
        )
    n_samples, length, _ = X.shape
    N = A_csr.shape[0]
    states = np.zeros((n_samples, length, N), dtype=np.float64)
    rt_free = np.zeros((n_samples, length), dtype=np.float64)
    rt_inc = np.zeros((n_samples, length), dtype=np.float64)
    bias_term = np.linalg.norm(bias) if bias.size else 0.0
    for i in range(n_samples):
        state = np.zeros(N, dtype=np.float64)
        for t in range(length):
            x_t = X[i, t]
            # the canonical leaky update; R_t uses A h_{t-1}, so save h before the update
            recurrent = A_csr @ state
            input_drive = W_in_csr @ x_t
            drive = gain * recurrent + input_drive + bias
            new_state = (1.0 - leak) * state + leak * np.tanh(drive)
            ah_norm = float(np.linalg.norm(recurrent))
            w_norm = float(np.linalg.norm(input_drive))
            wb_norm = float(np.linalg.norm(input_drive + bias)) if bias_term > 0 else w_norm
            rt_free[i, t] = ah_norm / w_norm if w_norm > 0 else 0.0
            rt_inc[i, t] = (gain * ah_norm) / wb_norm if wb_norm > 0 else 0.0
            state = new_state
            states[i, t] = state
    return DynamicsResult(
        R_t_gain_free=rt_free,
        R_t_gain_inclusive=rt_inc,
        states=states,
        length=length,
    )


def summarize_R_t(R_t: np.ndarray, *, discard_washout: int = 0) -> dict[str, float]:
    """Median and quantiles of ``R_t`` over the flat (n_samples * steps) distribution."""
    flat = np.asarray(R_t).ravel()
    if discard_washout:
        flat = flat[discard_washout:]
    if flat.size == 0:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(flat)),
        "p10": float(np.quantile(flat, 0.10)),
        "p90": float(np.quantile(flat, 0.90)),
    }


def effective_rank(state_matrix: np.ndarray) -> dict[str, float]:
    """Effective rank via the participation ratio of the singular values.

    ``D_eff = ( sum sigma_i )^2 / sum sigma_i^2 ``, computed on the time-averaged
    state matrix ``states.mean(axis=0)`` so each window contributes one row (per the
    contract: ``states[i, t]`` is a state vector).
    """
    flat = np.asarray(state_matrix)
    flat = flat.reshape(flat.shape[0] * flat.shape[1], flat.shape[2]) if flat.ndim == 3 else flat
    if flat.size == 0:
        return {"D_eff": float("nan"), "D_eff_factor": float("nan")}
    # use the cross-window average so the rank is the rank of the typical state
    typical = flat.mean(axis=0) if flat.ndim == 2 else flat
    centered = flat - flat.mean(axis=0, keepdims=True) if flat.ndim == 2 else flat
    s = np.linalg.svd(centered, compute_uv=False)
    pos = s[s > 1e-12]
    if pos.size == 0:
        return {"D_eff": 0.0, "D_eff_factor": 0.0}
    d_eff = float(pos.sum() ** 2 / (pos ** 2).sum())
    return {"D_eff": d_eff, "D_eff_factor": float("nan")}


def memory_metric(
    states: np.ndarray,
    inputs: np.ndarray,
    *,
    discard_washout: int = 0,
    lags: Sequence[int] = MEMORY_LAGS,
) -> float:
    """``max_k in lags | corr( h_t , x_{t-k} ) |`` over validation windows.

    ``states`` shape ``(n_samples, length, N)``, ``inputs`` shape ``(n_samples, length, Din)``.
    The correlation is per-channel in ``x``, then the absolute values are averaged.
    """
    states = np.asarray(states)
    inputs = np.asarray(inputs)
    n_samples, length, N = states.shape
    if inputs.shape[:2] != (n_samples, length):
        raise ValueError("states and inputs must share their first two axes")
    Din = inputs.shape[2]
    best = 0.0
    for k in lags:
        if length - k <= discard_washout:
            continue
        h = states[:, discard_washout : length - k, :]
        x = inputs[:, discard_washout + k : length, :]
        per_channel = np.zeros(Din, dtype=np.float64)
        for c in range(Din):
            xc = x[:, :, c].ravel()
            hc = h[:, :, 0].ravel() if N == 1 else h.reshape(-1, N).mean(axis=1)
            if xc.std() == 0 or hc.std() == 0:
                continue
            per_channel[c] = abs(float(np.corrcoef(hc, xc)[0, 1]))
        if per_channel.size:
            best = max(best, float(per_channel.mean()))
    return best


@dataclass(frozen=True)
class GateVerdict:
    C3_1_R_t_band: bool
    C3_2_memory_drop: bool
    C3_3_D_eff_factor: bool
    C4_8_report: bool

    def all_pass(self) -> bool:
        return bool(self.C3_1_R_t_band and self.C3_2_memory_drop and self.C3_3_D_eff_factor)


def evaluate_c3(
    A: sp.spmatrix,
    W_in: sp.spmatrix,
    bias: np.ndarray,
    X_val: np.ndarray,
    *,
    gain: float,
    leak: float,
    din: int,
    pinned_R_t_band: tuple[float, float] = RT_BAND,
    memory_drop_gate: float = MEMORY_DROP_GATE,
    deff_factor_gate: float = DEFF_FACTOR_GATE,
    memory_lags: Sequence[int] = MEMORY_LAGS,
) -> dict[str, Any]:
    """Drive the reservoir on the validation windows and score C3.1-C3.3.

    The same ``X_val`` is driven through the real graph and through the same graph with
    ``A := 0``, so the memory metric is measured identically on the two and the
    signed ``(M - M_A=0) / M`` drop is comparable. ``din`` is used only for the
    ``D_eff >= 1.5 * Din`` check, not for any drive-time scaling.
    """
    if X_val.ndim != 3:
        raise ValueError(f"X_val must be (n, L, Din), got shape {X_val.shape}")
    if X_val.shape[2] != din:
        raise ValueError(
            f"X_val Din ({X_val.shape[2]}) does not match the declared Din ({din})"
        )
    result = reservoir_drive(A, W_in, bias, X_val, gain=gain, leak=leak)
    A_zero = sp.csr_matrix((A.shape[0], A.shape[1]))
    result_zero = reservoir_drive(A_zero, W_in, bias, X_val, gain=gain, leak=leak)
    rt_summary = summarize_R_t(result.R_t_gain_free)
    rt_inc_summary = summarize_R_t(result.R_t_gain_inclusive)
    deff = effective_rank(result.states)
    deff["D_eff_factor"] = float(deff["D_eff"] / din) if din > 0 else float("nan")
    M = memory_metric(result.states, X_val, lags=memory_lags)
    M_zero = memory_metric(result_zero.states, X_val, lags=memory_lags)
    drop = (M - M_zero) / M if M > 0 else float("nan")
    rt_ok = bool(pinned_R_t_band[0] <= rt_summary["median"] <= pinned_R_t_band[1])
    mem_ok = bool(drop >= memory_drop_gate) if not np.isnan(drop) else False
    deff_ok = bool(deff["D_eff_factor"] >= deff_factor_gate) if not np.isnan(deff["D_eff_factor"]) else False
    return {
        "R_t": {
            "gain_free": rt_summary,
            "gain_inclusive_v1_comparable": rt_inc_summary,
            "median_threshold": list(pinned_R_t_band),
            "C3_1_pass": rt_ok,
        },
        "memory": {
            "M": float(M),
            "M_A_equals_0": float(M_zero),
            "drop_fraction": float(drop),
            "drop_threshold": memory_drop_gate,
            "lags": list(memory_lags),
            "C3_2_pass": mem_ok,
        },
        "effective_rank": {
            "D_eff": float(deff["D_eff"]),
            "D_eff_factor": float(deff["D_eff_factor"]),
            "factor_threshold": deff_factor_gate,
            "C3_3_pass": deff_ok,
        },
        "windows": {
            "n_samples": int(X_val.shape[0]),
            "length": int(X_val.shape[1]),
            "Din": int(din),
        },
        "verdict": GateVerdict(
            C3_1_R_t_band=rt_ok,
            C3_2_memory_drop=mem_ok,
            C3_3_D_eff_factor=deff_ok,
            C4_8_report=False,
        ).all_pass(),
        "criterion_lines": {
            "C3.1_R_t_in_band": bool(rt_ok),
            "C3.2_memory_drop_>=_20pct": bool(mem_ok),
            "C3.3_D_eff_>=_1.5*Din": bool(deff_ok),
        },
    }


__all__ = [
    "DynamicsResult",
    "MEMORY_LAGS",
    "RT_BAND",
    "MEMORY_DROP_GATE",
    "DEFF_FACTOR_GATE",
    "GateVerdict",
    "evaluate_c3",
    "effective_rank",
    "memory_metric",
    "reservoir_drive",
    "summarize_R_t",
]

# ---------------------------------------------------------------------------
# Validation search: pick (gain, input_scale) on a small grid
# ---------------------------------------------------------------------------
def _drive_for_knobs(A, W_in, bias, X, gain, leak, n_workers=1):
    return reservoir_drive(A, W_in, bias, X, gain=gain, leak=leak)


def select_knobs(
    A: sp.spmatrix,
    W_in: sp.spmatrix,
    bias: np.ndarray,
    X_val: np.ndarray,
    *,
    din: int,
    input_scale: float,
    leak_values: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    gain_values: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0),
    rt_band: tuple[float, float] = RT_BAND,
    memory_drop_gate: float = MEMORY_DROP_GATE,
    deff_factor_gate: float = DEFF_FACTOR_GATE,
) -> dict[str, Any]:
    """Search (gain, leak) for the pair that puts C3 inside its gates.

    Selection criterion (declared before any measurement):

    1. median R_t (gain-free) must lie in ``rt_band``;
    2. memory drop >= ``memory_drop_gate``;
    3. D_eff_factor >= ``deff_factor_gate``;
    4. among the qualifying pairs, prefer the one with median R_t closest to the band
       centre, then the LARGEST leak (memory retention), then the SMALLEST gain.

    ``input_scale`` is FIXED for the search: the construct-phase C3 measurement names
    the gain/leak grid as the validation knobs and the input_scale is the v2 mapping's
    declared value. A wider search over input_scale would be a separate search and is not
    done here.
    """
    trace = []
    chosen = None
    centre = sum(rt_band) / 2.0
    for gain in gain_values:
        for leak in leak_values:
            r = evaluate_c3(A, W_in, bias, X_val, gain=gain, leak=leak, din=din,
                             pinned_R_t_band=rt_band, memory_drop_gate=memory_drop_gate,
                             deff_factor_gate=deff_factor_gate)
            point = {
                "gain": float(gain), "leak": float(leak), "input_scale": float(input_scale),
                "R_t_median": r["R_t"]["gain_free"]["median"],
                "R_t_gain_inclusive_median": r["R_t"]["gain_inclusive_v1_comparable"]["median"],
                "memory_drop": r["memory"]["drop_fraction"],
                "D_eff_factor": r["effective_rank"]["D_eff_factor"],
                "pass": bool(r["verdict"]),
            }
            trace.append(point)
            if r["verdict"]:
                distance = abs(r["R_t"]["gain_free"]["median"] - centre)
                key = (distance, -leak, gain)
                if chosen is None or key < chosen[0]:
                    chosen = (key, gain, leak, r)
    if chosen is None:
        return {
            "chosen": None, "trace": trace, "all_points_failed": True, "expanded_band": False,
            "rationale": (
                f"no (gain, leak) point in the grid put median R_t inside {list(rt_band)} "
                f"AND satisfied memory drop >= {memory_drop_gate} AND D_eff_factor >= "
                f"{deff_factor_gate}. C3 is reported as FAIL with this trace."
            ),
        }
    _, gain, leak, r = chosen
    return {
        "chosen": {"gain": float(gain), "leak": float(leak), "input_scale": float(input_scale)},
        "trace": trace,
        "all_points_failed": False,
        "expanded_band": False,
        "rationale": (
            f"closest-to-centre ({centre:.3f}) preferred, then largest leak, then smallest gain"
        ),
        "metrics_at_chosen": r,
    }
