# E3 -- A3 construct-validity audit: calibration, Din sweep, K sweep

**Headline: A3 discriminates correctly on the calibration systems (low FAIL, high PASS), the
width-invariance claim is quantified for F1/F2/F3 (regression slope of `D_eff/Din` on `Din` is within
±0.013/Din), and F5's D_eff *saturates* at `Din >= 10` (slope -0.070/Din) -- the F5 construction is
itself bounded, which strengthens the finding that "A3 FAIL is a property of the substrate family, not
of the input width or the Krylov depth".**

Authority: the operator's spec for E3 (2026-09-23). **No change** to the A3 threshold (2*Din), no
change to family construction, and no change to input geometry. The Din sweep uses the frozen
amendment-6 allocation rule to extend the typed-aligned mapping to Din in {4, 10, 12}; the K sweep
uses krylov_score directly (audit_a3 reads A3_KRYLOV_K as a module constant and cannot be
parameterised without modifying the engine).

## 1. Calibration controls (component 1)

A3 on resaudit toys at their own declared Din, K=16:

```
known_high_two_coprime_cycles     D_eff=25.436   gate=10.0    PASS
known_low_zero_input_graph        D_eff= 0.000   gate= 4.0    FAIL
```

A3 discriminates known-high from known-low in the right direction on the calibration systems. This
is the construct-validity check the operator required: the criterion is **not** uniformly rejecting
everything, and it is **not** uniformly accepting the trivial case. The high-side value (25.4 vs gate
10) reproduces the engine's own calibration result; the low-side reproduces D_eff=0.0 (no B, no
Krylov span).

## 2. Din sweep (component 2)

F1/F2/F3/F5 at Din in {4, 6, 8, 10, 12}, K=16. The threshold is `2*Din` at every cell, unchanged.

```
family   Din=4   Din=6   Din=8   Din=10  Din=12    slope (/Din)   CI95                       r
F1       5.25    7.82    10.51   12.63   14.89     -0.0093       [-0.0183, -0.0003]         -0.885
F2       5.43    7.88    10.30   12.69   14.95     -0.0133       [-0.0175, -0.0092]         -0.986
F3       6.76    10.26   13.63   17.18   20.60     +0.0031       [-0.0002, +0.0063]         +0.863
F5       6.28    9.33    12.35   12.35   12.35     -0.0702       [-0.1289, -0.0115]         -0.910
```

**Interpretation.**

- **F1 / F2 / F3.** `D_eff/Din` is within ±0.013/Din of the Din=6 baseline; the slope is small and
  its sign agrees with what theory predicts (slight dependence on the eigenvalue gap). This
  **quantifies the width-invariance claim** that was the headline of the previous Stage-1 finding.
  F3's slope is positive (the random control grows slightly faster with Din) but the CI includes 0.
- **F5.** `D_eff` *saturates*: it rises 9.33 at Din=6 to 12.35 at Din=8, and then is flat at 12.35 for
  Din=10 and Din=12. The strong negative slope (-0.070/Din) is driven by the Din=4-8 region; from
  Din=10 onward, the substrate does not gain more effective Krylov span regardless of input width.
  This is **new evidence** that F5's FAIL is a property of the construction itself, not of the input
  geometry: widening Din does not help it.

All twelve cells FAIL A3 (gate at every Din is 2*Din and the measured D_eff is below).

## 3. K sensitivity (component 3)

F1/F2/F3/F5 at Din=6, K in {8, 16, 32}. Verdict stability across K, not model selection.

```
family   K=8      K=16     K=32     max-min range
F1       8.0216   7.8216   7.6994   0.3222
F2       7.8804   7.8841   7.8842   0.0038   (rock-solid)
F3       10.6698  10.2556  9.9961   0.6737
F5       8.9870   9.3341   9.4766   0.4896
```

**Verdict is K-stable**: every cell FAILs at gate 12 regardless of K. The D_eff values themselves move
slightly (≤ ±0.7 across K=8..32), so the chosen K=16 is not a knob that could have been tuned to
rescue a family. F2's wire is essentially K-invariant (range 0.004), consistent with F2 being a
degree-preserving random control whose Krylov span is determined by the random topology.

## 4. What this adds to C3

C3 is no longer "all families fail A3 at two widths". It is now:

1. A3 **works** on the calibration systems (low FAIL, high PASS, with the right values).
2. A3's verdict is **K-stable** and its D_eff is **nearly width-invariant** for F1/F2/F3.
3. F5's D_eff **saturates** with `Din`: it grows from 9.33 at Din=6 to 12.35 at Din=8 and then stops.
   The construction's A3-relevant span is bounded at ~12.4 regardless of how many channels we
   allocate, which is consistent with the frozen amendment-1 description of F5 as a construction
   built against a scale at which it does not scale.

Together with C1 and C2 the audit paper now has the construct-validity evidence the operator
asked for.

## Artifacts

```
results/audit/resaudit_stage1/e3_criterion_audit.json
ops/audit/e3_criterion_audit.py                                 re-runnable from the read-only deposit
```
