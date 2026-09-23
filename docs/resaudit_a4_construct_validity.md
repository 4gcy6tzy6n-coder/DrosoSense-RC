# ResAudit-Food — A4 construct-validity audit

**Status: COMPLETED 2026-09-23, on CALIBRATION FAMILIES ONLY (C1–C7). No F family was
built or measured. No threshold was moved.**

**Verdict (pre-declared decision rule 6.2 fires):**

```
A4 = DESCRIPTIVE ONLY / RETIRED AS A QUALIFICATION GATE
```

The A4 observable does not track true recurrence strength in the direction its own
definition requires. It is retired as a gate. It is not repaired, and its threshold is not
adjusted.

Runner: `resaudit/a4_validity.py`. Report: `results/audit/resaudit_a4_validity/A4_construct_validity.json`.

---

## 1. The question, and the decision rule, fixed before the numbers

A4 is declared in the pre-registration as a *recurrence-derived memory* criterion:

```
A4:  (M_recurrent − M_{A:=0}) / M_recurrent  ≥  0.20
```

Calibration of the runner showed the observable is small, sign-unstable and non-monotone
in recurrent coupling. Those are symptoms of a construct that does not measure what it
claims, so the question asked here is:

> **As true recurrence strength increases, does the A4 observable move monotonically in
> the direction A4's definition requires?**

Pre-declared before reading any number:

* **directional** iff, for every calibration family, `M_recurrent` is non-decreasing in
  `rho`, the Spearman rank correlation between `rho` and `M_recurrent` is positive, AND the
  observable at the frozen operating point is positive;
* **not directional** otherwise → A4 is retired as a gate by decision rule 6.2, reported
  and **not** repaired by moving the threshold.

The goal was never to make A4 pass. It was to find out whether the observable has
directional validity.

## 2. The ladder

One architecture-preserving knob, so the direction test cannot be confounded by a change
of graph: the **spectral radius of the same wiring with the same input mapping**.

| element | value |
|---|---|
| ladder | `rho ∈ {0.00, 0.25, 0.50, 0.75, 0.95}` |
| operating point | `rho = 0.95` (the frozen convention, amendment 1 Section 0.1) |
| control | `rho = 0.00` reproduces the `A := 0` feed-forward control exactly |
| drive seeds | 5 realizations (20260923, 11, 29, 47, 83) |
| protocol | `leak = 1.0`, `gain = 1.0`, 48 windows × 256 steps, washout 16 — all frozen |

`rho = 0.00` being the control makes the ladder self-contained: the comparison "does
recurrence add memory" is a comparison between two rungs of the same measurement, not
between two experiments.

## 3. Result

```
| family                        | Spearman(rho, M) | monotone | A4 @ rho=0.95 | positive | directional |
| C1 ring+chords                | +0.400           | no       | -0.1237       | no       | NO          |
| C2 coprime cycles             | +0.700           | no       | +0.0493       | yes      | NO          |
| C3 single 2-cycle             | -0.900           | no       | -0.1610       | no       | NO          |
| C4 zero input                 | n/a              | n/a      | +0.0000       | no       | NO          |
| C5 nilpotent chain            | —                | —        | —             | —        | N/A         |
| C6 20-cycle (A2 parent)       | -0.900           | no       | -1.0235       | no       | NO          |
| C7 20-cycle, reversal perm.   | -0.900           | no       | -1.0235       | no       | NO          |

VERDICT: NOT_DIRECTIONAL — 0/6 ladder-applicable families directional.
```

`M_recurrent` across the ladder, mean over seeds:

```
C1 ring+chords   0.0098 | 0.0105 | 0.0111 | 0.0112 | 0.0099    A4: +0.000 +0.067 +0.105 +0.077 -0.124
C2 coprime       0.0094 | 0.0100 | 0.0109 | 0.0113 | 0.0103    A4: +0.000 +0.060 +0.132 +0.148 +0.049
C3 2-cycle       0.0164 | 0.0170 | 0.0163 | 0.0151 | 0.0149    A4: +0.000 +0.023 -0.034 -0.152 -0.161
C6 20-cycle      0.0164 | 0.0170 | 0.0162 | 0.0152 | 0.0109    A4: +0.000 +0.021 -0.048 -0.177 -1.024
```

Six observations, each independently disqualifying:

1. **`M_recurrent` falls as recurrence strengthens.** From `rho = 0.25` to `0.95` it
   declines on C1, C3, C6 and C7. The trend is *opposite* to the one A4's definition
   assumes. C6/C7 lose 36 % of the metric across that range (`0.0170 → 0.0109`).
2. **Magnitudes are O(0.01) and do not separate the extreme rungs.** `M` at `rho = 0.95`
   is statistically indistinguishable from `M` at `rho = 0.00` on C1 (0.0099 vs 0.0098)
   and C2 (0.0103 vs 0.0094), even though the wiring could not be more different.
3. **The observable is negative at the operating point on 5 of 6 families**, so A4 does
   not merely fail its 0.20 threshold — it reports that recurrence *subtracts* memory.
   C6/C7 reach −1.02, meaning the recurrent state correlates with lagged inputs at
   essentially zero while the feed-forward control still does.
4. **A2's fairness is confirmed by construction**: C7 is a reversal permutation of C6, and
   its ladder is bit-identical to C6's (Spearman −0.900, A4 = −1.0235 at the operating
   point). The measurement does not depend on node labelling.
5. **The trend that exists is non-monotone with a peak below the gate.** C2 — the
   most favourable family — peaks at `+0.148` at `rho = 0.75` and falls back to `+0.049`
   at the frozen operating point. Even its best rung is under the 0.20 threshold.
6. **The sign of Spearman carries no information here.** It is `-0.900` on three families
   and `+0.700` on another, over a metric whose total range is ~0.002. Ranking an
   O(0.01) noisy quantity produces ranks, not direction.

## 4. One caveat, recorded rather than hidden

**C5 (nilpotent chain) is not ladder-applicable.** Its base spectral radius is `0.0`, so
"scale `rho` to `t`" is undefined and no recurrence-strength knob exists for that wiring.
The audit reports `N/A (no ladder)` rather than measuring the same unscaled matrix five
times and calling the result a flat trend. The verdict is therefore `0/6` on the
ladder-applicable families, not `0/7`.

This is a property of the substrate, not a defect in the audit: a nilpotent graph has no
recurrence to strengthen.

## 5. Why the construct fails, in one paragraph

`memory_metric` computes the maximum absolute correlation between one state channel and
lagged inputs. On bounded random drive, a feed-forward layer already achieves that
correlation directly from the input projection, so the control `M_{A:=0}` is not a weak
baseline — it is roughly the whole signal. The recurrent path can only *change* the state,
and on these substrates it changes it in a way that reduces that correlation. Subtracting
the control therefore yields a quantity whose expected sign is not determined by the
property A4 claims to measure. The metric is a legitimate descriptive statistic; it is not
a recurrence-detection instrument at this scale, and no threshold on it can make it one.

## 6. Consequence, and what must not happen next

**Decision rule 6.2 fires, as the pre-registration anticipated.** Quoting it:

> *If F5 clears A3 but fails A4/A5, the battery's primary gate is mis-specified and that is
> reported as a battery finding before any task result is discussed.*

The finding is stronger than the rule anticipated: A4 is mis-specified independently of
F5, and it is mis-specified at the construct level, not at the threshold level.

Therefore:

* **A4 becomes a reported diagnostic, not a gate.** Stage 1 reports its number with the
  caveat that it is not directional. `construct_qualified` no longer treats an A4 failure
  as disqualifying; a family's qualification rests on A1/A2 (admission) and A3 (primary),
  with A5 as the remaining downstream gate.
* **The threshold stays at 0.20 and is not moved.** Moving it would be exactly the
  threshold-tuning the pre-registration forbids, and no value of it would fix a
  non-directional observable.
* **A4 may not be used as evidence for a memory claim anywhere in the paper.** The
  motivating negative (`100 % SCC + 100 % cycle coverage ⇏ useful reservoir dynamics`)
  still rests on A3's reachable-diversity measurement, which is unaffected by this audit.
* **Replacing A4 requires a new construct, not a new threshold.** If a recurrence-derived
  memory criterion is wanted for the paper, it needs an observable whose direction is
  established on a ladder like this one *before* it is applied to F families.

## 7. What the paper may therefore claim about memory

The pre-registration listed memory contribution as one of the checks that separate
substrates. That specific claim is now **withdrawn**:

> ~~the checks that do separate substrates are controllability, memory contribution and
> state expansion~~

becomes:

> the checks that separate substrates are **finite-horizon input-reachable state
> diversity** (A3) and **state expansion** (A5), both of which are measured properties of
> the substrate; the recurrent-memory criterion as originally formulated was found not to
> be directionally valid and is reported descriptively.

This is a narrowing of the framework, not a rescue of it, and it is recorded here — before
any F family exists — so that it cannot be presented later as a post-hoc adjustment.
