"""A4 construct-validity audit -- does the A4 observable track real recurrence?

The pre-registration's A4 is declared as a *recurrence-derived memory* criterion:

```
A4:  (M_recurrent - M_{A:=0}) / M_recurrent  >=  0.20
```

Calibration showed the observable is small, sign-unstable, and non-monotone in recurrent
coupling. Small/unstable/monotone-in-the-wrong-direction are all symptoms of a construct
that does not measure what it claims. This module answers ONE question, on CALIBRATION
FAMILIES ONLY (C1-C7, ``resaudit.toys``), with no F family and no threshold tuning:

    As TRUE recurrence strength increases, does the A4 observable move monotonically in
    the direction A4's definition requires?

The recurrence ladder is a single, architecture-preserving knob: the spectral radius
``rho`` of the SAME wiring with the SAME input mapping. ``rho = 0`` reproduces the
``A := 0`` control exactly, so the ladder contains its own control and the direction test
is not confounded by a change of graph.

Pre-declared decision rule (fixed BEFORE the numbers are read):

* **directional**: for every calibration family, ``M(A)`` is non-decreasing in ``rho`` up
  to the frozen operating point, with a positive Spearman rank correlation, AND the
  observable at the frozen operating point exceeds its value at the control.
* **not directional**: any family where ``M(A)`` decreases as ``rho`` increases, or where
  the observable is negative at the operating point. Then A4 is **DESCRIPTIVE ONLY /
  RETIRED AS A GATE** by decision rule 6.2 -- reported, not repaired by moving the
  threshold.

Nothing here changes a threshold. A negative answer is the deliverable.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resaudit import _kernels as _dyn  # noqa: E402

from resaudit import toys  # noqa: E402
from resaudit.battery import (  # noqa: E402
    FROZEN_RHO_TARGET,
    PROBE_GAIN,
    PROBE_LEAK,
    PROBE_WASHOUT,
    frozen_conventions,
    probe_input_with_seed,
    probe_protocol,
    scale_to_spectral_radius,
    spectral_radius_of,
)

# ---------------------------------------------------------------------------
# The pre-declared recurrence ladder
# ---------------------------------------------------------------------------

#: The recurrence-strength ladder: spectral radii applied to the SAME wiring and input.
#: ``0.0`` IS the ``A := 0`` control, so the control is a point on the ladder rather than
#: a separate measurement.
RECURRENCE_LADDER: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.95)

#: The frozen operating point, taken from the battery's own conventions.
OPERATING_RHO = FROZEN_RHO_TARGET

#: Seeds for the repetition that separates a real trend from probe noise.
LADDER_SEEDS: tuple[int, ...] = (20260923, 11, 29, 47, 83)

#: Below this base spectral radius the ladder knob is undefined (a nilpotent substrate).
_RHO_FLOOR = 1e-9


@dataclass(frozen=True)
class LadderPoint:
    """One rung: the memory observable at a given recurrence strength."""

    rho: float
    m_recurrent: float
    m_feedforward: float
    observable: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho": float(self.rho),
            "M_recurrent": float(self.m_recurrent),
            "M_feedforward": float(self.m_feedforward),
            "observable": float(self.observable),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class FamilyValidity:
    """A4's directional behaviour for one family."""

    family_id: str
    label: str
    points: tuple[LadderPoint, ...]
    spearman: float
    monotone_nondecreasing: bool
    observable_at_operating: float
    observable_positive_at_operating: bool
    direction_ok: bool
    ladder_applicable: bool = True
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "label": self.label,
            "points": [p.as_dict() for p in self.points],
            "spearman_rho_vs_observable": float(self.spearman),
            "monotone_nondecreasing": bool(self.monotone_nondecreasing),
            "observable_at_operating_rho": float(self.observable_at_operating),
            "observable_positive_at_operating": bool(self.observable_positive_at_operating),
            "direction_ok": bool(self.direction_ok),
            "ladder_applicable": bool(self.ladder_applicable),
            "notes": list(self.notes),
        }


def _observable(m_recurrent: float, m_feedforward: float) -> float:
    """A4's declared form, with the same zero-guard the battery uses."""
    m = float(m_recurrent)
    if m <= 0.0:
        return 0.0
    return float((m - float(m_feedforward)) / m)


def _drive_memory(A: sp.spmatrix, B: np.ndarray, seed: int) -> tuple[float, float]:
    """``(M_recurrent, M_feedforward)`` on one probe realization."""
    n = A.shape[0]
    X = probe_input_with_seed(B.shape[1], seed)
    W_in = sp.csr_matrix(B)
    bias = np.zeros(n, dtype=np.float64)
    driven = _dyn.reservoir_drive(A, W_in, bias, X, gain=PROBE_GAIN, leak=PROBE_LEAK)
    feedforward = _dyn.reservoir_drive(
        sp.csr_matrix((n, n), dtype=np.float64), W_in, bias, X, gain=PROBE_GAIN, leak=PROBE_LEAK
    )
    m_rec = _dyn.memory_metric(driven.states, X, discard_washout=PROBE_WASHOUT)
    m_ff = _dyn.memory_metric(feedforward.states, X, discard_washout=PROBE_WASHOUT)
    return float(m_rec), float(m_ff)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, no scipy dependency and no ties handling beyond average."""
    def ranks(v: Sequence[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        order = np.argsort(arr, kind="mergesort")
        r = np.empty(len(arr), dtype=np.float64)
        r[order] = np.arange(len(arr), dtype=np.float64)
        # average ranks for ties
        for value in np.unique(arr):
            m = arr == value
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = ranks(xs), ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def audit_family_direction(
    family_id: str,
    label: str,
    A: sp.spmatrix,
    B: np.ndarray,
    *,
    ladder: Sequence[float] = RECURRENCE_LADDER,
    seeds: Sequence[int] = LADDER_SEEDS,
) -> FamilyValidity:
    """Run the ladder on ONE wiring, scaling ``rho`` and nothing else."""
    A = A.tocsr()
    rho0 = spectral_radius_of(A)
    # A nilpotent substrate has spectral radius 0, so "scale rho to t" is undefined and
    # the ladder knob does not exist. Saying so is the honest report; silently measuring
    # the same unscaled matrix five times and calling it a flat trend is not.
    if rho0 <= _RHO_FLOOR:
        return FamilyValidity(
            family_id=family_id,
            label=label,
            points=(),
            spearman=float("nan"),
            monotone_nondecreasing=False,
            observable_at_operating=float("nan"),
            observable_positive_at_operating=False,
            direction_ok=False,
            ladder_applicable=False,
            notes=(
                f"ladder NOT APPLICABLE: base spectral radius is {rho0:.3g}, so no "
                "recurrence-strength knob exists for this wiring. A4 cannot be tested "
                "for direction on it.",
            ),
        )
    points: list[LadderPoint] = []
    per_rho_rec: list[float] = []
    per_rho_obs: list[float] = []
    for rho in ladder:
        scaled = scale_to_spectral_radius(A, float(rho)) if rho0 > 0 else A.tocsr()
        recs, ffs = [], []
        for seed in seeds:
            m_rec, m_ff = _drive_memory(scaled, B, seed)
            recs.append(m_rec)
            ffs.append(m_ff)
            points.append(
                LadderPoint(
                    rho=float(rho),
                    m_recurrent=m_rec,
                    m_feedforward=m_ff,
                    observable=_observable(m_rec, m_ff),
                    seed=int(seed),
                )
            )
        per_rho_rec.append(float(np.mean(recs)))
        per_rho_obs.append(float(np.mean([_observable(r, f) for r, f in zip(recs, ffs)])))

    rho_axis = [float(r) for r in ladder]
    spearman = _spearman(rho_axis, per_rho_rec)
    mono = all(
        per_rho_rec[i + 1] >= per_rho_rec[i] - 1e-12 for i in range(len(per_rho_rec) - 1)
    )
    idx_op = min(
        range(len(rho_axis)), key=lambda i: abs(rho_axis[i] - OPERATING_RHO)
    )
    obs_op = per_rho_obs[idx_op]
    obs_pos = obs_op > 0.0
    direction_ok = bool(mono and not math.isnan(spearman) and spearman > 0 and obs_pos)

    notes: list[str] = []
    if math.isnan(spearman):
        notes.append("Spearman undefined: the recurrent memory was constant across rho")
    if not mono:
        notes.append("M_recurrent is NOT non-decreasing in rho: adding recurrence lowered it")
    if not obs_pos:
        notes.append(
            f"A4 observable is non-positive at the operating point rho={rho_axis[idx_op]}"
        )
    if min(per_rho_rec) < 0.02:
        notes.append("M_recurrent is O(0.01) across the ladder: the ratio is noise-dominated")
    return FamilyValidity(
        family_id=family_id,
        label=label,
        points=tuple(points),
        spearman=spearman,
        monotone_nondecreasing=bool(mono),
        observable_at_operating=float(obs_op),
        observable_positive_at_operating=bool(obs_pos),
        direction_ok=direction_ok,
        notes=tuple(notes),
    )


def _calibration_wirings() -> list[tuple[str, str, sp.csr_matrix, np.ndarray]]:
    """C1-C7: the calibration families, UNNORMALIZED so the ladder does its own scaling."""
    out: list[tuple[str, str, sp.csr_matrix, np.ndarray]] = []
    A, B, _ = toys.ring_with_chords(n=60, k=2, n_chords=30, din=4, seed=5)
    out.append(("C1", "ring+chords", A, B))
    A, B, _ = toys.two_coprime_cycles()
    out.append(("C2", "coprime cycles", A, B))
    A, B, _ = toys.hairline_fail_cycle()
    out.append(("C3", "single 2-cycle", A, B))
    A, B, _ = toys.zero_input_graph()
    out.append(("C4", "zero input", A, B))
    A, B, _ = toys.nilpotent_chain()
    out.append(("C5", "nilpotent chain", A, B))
    A, B, _ = toys.k_out_ring(n=20, k=1)
    out.append(("C6", "20-cycle (A2 parent)", A, B))
    A0, B0, _ = toys.k_out_ring(n=20, k=1)
    perm = (-np.arange(A0.shape[0])) % A0.shape[0]
    out.append(("C7", "20-cycle, reversal permutation", sp.csr_matrix(A0[perm][:, perm]), B0))
    return out


def run_audit() -> dict[str, Any]:
    """Run the full construct-validity audit and return the report."""
    results: list[FamilyValidity] = []
    for fid, label, A, B in _calibration_wirings():
        results.append(audit_family_direction(fid, label, A, B))

    applicable = [r for r in results if r.ladder_applicable]
    directional = [r for r in applicable if r.direction_ok]
    verdict = "DIRECTIONAL" if len(directional) == len(applicable) and applicable else "NOT_DIRECTIONAL"
    return {
        "report": "A4 construct-validity audit (calibration families only)",
        "question": (
            "As true recurrence strength increases, does the A4 observable move "
            "monotonically in the direction A4's definition requires?"
        ),
        "pre_declared_decision_rule": {
            "directional": (
                "every family: M_recurrent non-decreasing in rho, Spearman > 0, and the "
                "observable positive at the operating rho"
            ),
            "not_directional": (
                "any family failing that -> A4 is DESCRIPTIVE ONLY / RETIRED AS GATE by "
                "decision rule 6.2, reported and not repaired by moving the threshold"
            ),
            "declared_before_reading": True,
        },
        "ladder": {
            "rhos": list(RECURRENCE_LADDER),
            "seeds": list(LADDER_SEEDS),
            "operating_rho": OPERATING_RHO,
            "knob": "spectral radius of the SAME wiring with the SAME input mapping",
            "control": "rho = 0.0 reproduces the A := 0 control exactly",
        },
        "conventions": frozen_conventions(),
        "probe_protocol": probe_protocol(),
        "families": [r.as_dict() for r in results],
        "verdict": verdict,
        "n_directional": len(directional),
        "n_ladder_applicable": len(applicable),
        "n_families": len(results),
        "n_ladder_not_applicable": len(results) - len(applicable),
    }


def _render_table(report: Mapping[str, Any]) -> str:
    rows = [
        "| family | Spearman(rho, M) | monotone | A4 @ operating | positive | directional |",
        "|---|---|---|---|---|---|",
    ]
    for f in report["families"]:
        if not f.get("ladder_applicable", True):
            rows.append(
                "| {family_id} {label} | — | — | — | — | N/A (no ladder) |".format(
                    family_id=f["family_id"], label=f["label"]
                )
            )
            continue
        rows.append(
            "| {family_id} {label} | {sp:+.3f} | {mono} | {obs:+.4f} | {pos} | {ok} |".format(
                family_id=f["family_id"],
                label=f["label"],
                sp=f["spearman_rho_vs_observable"],
                mono="yes" if f["monotone_nondecreasing"] else "**no**",
                obs=f["observable_at_operating_rho"],
                pos="yes" if f["observable_positive_at_operating"] else "**no**",
                ok="yes" if f["direction_ok"] else "**NO**",
            )
        )
    return "\n".join(rows)


def _render_ladder(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for f in report["families"]:
        if not f.get("ladder_applicable", True):
            lines.append(f"* **{f['family_id']} {f['label']}** — " + "; ".join(f["notes"]))
            continue
        by_rho: dict[float, list[dict[str, Any]]] = {}
        for p in f["points"]:
            by_rho.setdefault(p["rho"], []).append(p)
        cells = []
        for rho in sorted(by_rho):
            m = float(np.mean([q["M_recurrent"] for q in by_rho[rho]]))
            o = float(
                np.mean([q["observable"] for q in by_rho[rho]])
            )
            cells.append(f"{rho:.2f}: M={m:.4f} A4={o:+.3f}")
        lines.append(f"* **{f['family_id']} {f['label']}** — " + " | ".join(cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results" / "audit" / "resaudit_a4_validity" / "A4_construct_validity.json",
    )
    args = ap.parse_args(argv)

    report = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("A4 construct-validity audit -- calibration families only\n")
    print(_render_table(report))
    print()
    print("Ladder detail (mean over seeds):")
    print(_render_ladder(report))
    print()
    print(
        f"VERDICT: {report['verdict']}  "
        f"({report['n_directional']}/{report['n_ladder_applicable']} ladder-applicable "
        f"families directional; {report['n_ladder_not_applicable']} not ladder-applicable)"
    )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
