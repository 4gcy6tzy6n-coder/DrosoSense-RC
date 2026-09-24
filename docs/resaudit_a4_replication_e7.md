# E7 — A4 retirement replication under 30 seeds per cell

**Status: COMPLETE 2026-09-23. Frozen A4 definition, A4 threshold, Stage-1 logic and every
prior outcome unchanged. No A4b defined, no threshold modified, no replacement observable
searched for.**

Runner: `ops/audit/resaudit_e7_a4_replication.py`
Report: `results/audit/resaudit_a4_replication/E7_a4_replication.json`
Paper-ready table: `results/audit/resaudit_a4_replication/E7_a4_summary.csv`

## Primary question and answer

> **Does the preregistered A4 observable show reproducible directional validity under
> repeated calibration?**

**No. 0 of 6 families — reproducing the original 5-seed finding exactly at 30 seeds.** And the
failure is stronger than "no signal": the ladder is **monotonically decreasing** where the
declared expectation is **increasing**.

```
6 families x 5 recurrence levels x 30 seeds = 900 cells
DIRECTIONAL VALIDITY: 0/6
```

## 1. The frozen A4, recovered verbatim (not re-derived)

```
M_recurrent = memory_metric(states driven by A)      max_k |corr(h_t, u_{t-k})|,
                                                     k in {1,4,8,16}, washout 16
M_A:=0      = the SAME probe driven by A := 0        (zero recurrence)
A4          = (M_recurrent - M_A:=0) / M_recurrent   0.0 when M_recurrent <= 0
A4_MIN      = 0.20                                   unchanged; REPORTED, never applied
dynamics    leak = gain = 1.0
```

`A4_MIN` is reported in every cell but used to filter nothing: A4 is descriptive only, having
been retired as a gate by the construct-validity audit.

## 2. Machine-readable family specification

| family | construction rule | N | \|E\| | Din | ρ₀ (unscaled) | ladder applicable |
|---|---|---:|---:|---:|---:|---|
| C1 | ring+chords (k=2, 30 chords) | 60 | 134 | 4 | 2.46 | yes |
| C2 | coprime cycles (2,3,5,7,11) | 28 | 28 | 5 | 1.00 | yes |
| C3 | single 2-cycle | 2 | 2 | 1 | 1.00 | yes |
| C4 | recurrent graph, zero input | 12 | 12 | 2 | 1.00 | yes (degenerate) |
| C6 | 20-cycle (`k_out_ring` k=1) | 20 | 20 | 1 | 1.00 | yes |
| C7 | 20-cycle, reversal permutation | 20 | 20 | 1 | 1.00 | yes |

Common to all: recurrence-control parameter = **spectral radius of the same wiring with the
same input mapping**; weight rule = the toy builder's unit weights; `ρ = 0` reproduces the
`A := 0` control; the family's own `B`, never rescaled across `ρ`; 30 seeds.

**C5 (nilpotent chain) is excluded deliberately** — its base spectral radius is 0, so no
recurrence-strength knob exists for it. This exclusion was recorded before the run, and **no
family was added** to chase a prettier result.

## 3. The ladder: A4 DECREASES as recurrence strengthens

Panel (a) data — mean A4 by family and recurrence level (30 seeds per cell):

| family | ρ=0.00 | 0.25 | 0.50 | 0.75 | **0.95** |
|---|---:|---:|---:|---:|---:|
| C1 | +0.000 | −0.002 | −0.007 | −0.054 | **−0.196** |
| C2 | +0.000 | −0.008 | −0.007 | −0.047 | −0.071 |
| C3 | +0.000 | −0.007 | −0.082 | −0.094 | −0.154 |
| C4 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| C6 | +0.000 | −0.008 | −0.112 | −0.196 | **−0.562** |
| C7 | +0.000 | −0.008 | −0.112 | −0.196 | **−0.562** |

`A4 = 0` at ρ = 0 by construction (the zero-guard, since `A := 0` is the control there). From
there it goes **negative and progressively more so** in every non-degenerate family. The
declared expectation is `+`; the observed sign is `−` at every level of every family.

**The mechanism is visible in the two intermediates:**

| family | M_recurrent at ρ = 0 → 0.95 | M_A:=0 at ρ = 0 → 0.95 |
|---|---|---|
| C1 | 0.0105 → **0.0095** | 0.0105 → 0.0105 |
| C2 | 0.0101 → 0.0098 | 0.0101 → 0.0101 |
| C3 | 0.0146 → 0.0139 | 0.0146 → 0.0146 |
| C6 / C7 | 0.0146 → **0.0113** | 0.0146 → 0.0146 |

`M_A:=0` is constant, as it must be (the feed-forward path does not depend on `A`). But
`M_recurrent` **falls** as recurrence grows, so the subtraction can only become more negative.
Adding recurrence *reduces* the lagged-input correlation this metric measures: the recurrent
term mixes the state and destroys the simple lag-1 correlation the observable is built on.

## 4. Panel (b): directional validity with 95 % bootstrap CI

| family | Spearman(ρ, A4) | 95 % CI | monotonicity violations | expected | validity |
|---|---:|---|---:|---|---|
| C1 | −0.129 | [−0.299, +0.056] | 4 / 4 | + | **FAIL** |
| C2 | −0.011 | [−0.178, +0.157] | 3 / 4 | + | **FAIL** |
| C3 | +0.016 | [−0.149, +0.185] | 4 / 4 | + | **FAIL** |
| C4 | undefined | — | 0 / 4 | + | **FAIL** (degenerate) |
| C6 | **−0.228** | **[−0.387, −0.056]** | 4 / 4 | + | **FAIL** |
| C7 | **−0.228** | **[−0.387, −0.052]** | 4 / 4 | + | **FAIL** |

Two families (C6, C7) have CIs that **exclude zero on the negative side** — the relationship is
significantly *opposite* to the declared expectation. That is stronger than an absence of
validity: it is **anti-validity**.

**Monotonicity violations are near-total** (4/4 in four families, 3/4 in the fifth): advancing
along the recurrence ladder lowers A4 at almost every step.

C6 and C7 give **identical** Spearman (−0.228) and identical level means, consistent with C7
being a reversal permutation of C6 — the measurement is invariant to node relabelling, which
is a property one wants to see.

C4 is **degenerate, not merely failing**: zero input ⇒ zero state ⇒ `M_recurrent = M_A:=0 = 0`
⇒ A4 ≡ 0 by the documented zero-guard. No correlation is definable; this is an edge case
handled explicitly rather than a hidden NaN.

## 5. Panel (c): sign consistency

| family | fraction A4 > 0 | fraction A4 < 0 | `passes A4_MIN` by level (ρ = 0 → 0.95) |
|---|---:|---:|---|
| C1 | 0.37 | 0.43 | 0.00 / 0.03 / 0.13 / 0.27 / 0.20 |
| C2 | 0.39 | 0.41 | 0.00 / 0.00 / 0.20 / 0.30 / 0.20 |
| C3 | 0.43 | 0.37 | 0.00 / 0.00 / 0.17 / 0.23 / 0.23 |
| C4 | 0.00 | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| C6 | 0.36 | 0.44 | 0.00 / 0.00 / 0.13 / 0.20 / 0.13 |
| C7 | 0.36 | 0.44 | 0.00 / 0.00 / 0.13 / 0.20 / 0.13 |

A4 is **positive in only 36–43 % of cells** and negative in 37–44 %. If A4 were the
"recurrence contributes positive memory" quantity its name implies, it should be positive in
the large majority of cells. It is closer to a coin flip, and the frozen threshold of 0.20 is
met by at most 30 % of cells and by **0 %** at ρ = 0.

## 6. Conclusion

The A4 retirement is **reproduced and strengthened** at 6× the original seed count:

```
0/6 directional validity        (unchanged from the 5-seed audit)
ladder monotonically DECREASING where + was expected
2 of 6 CIs exclude zero, both NEGATIVE  (anti-validity, not just noise)
A4 > 0 in only 36-43% of cells
```

The observable does not measure "recurrence-derived memory"; it measures the *loss* of simple
lagged correlation caused by recurrence. Its retirement as a qualification gate —
recorded in `docs/resaudit_a4_construct_validity.md` and amendment 2 — is therefore not an
artefact of an underpowered calibration.

**Per instruction: no A4b, no threshold change, no replacement observable, no new family.**
E8 is not run; the experimental programme closes with E7.
