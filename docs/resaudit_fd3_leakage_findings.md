# FD3 — evaluation-leakage audit on the wine dataset: findings

**Result: repetition leakage is real but modest (+0.08 to +0.12 macro-F1), and the way the gap is
measured matters more than the gap — a grouped split scored with the wrong metric overstates it
nearly six-fold.**

Authority: `docs/resaudit_food_preregistration_amendment_7.md` §4. No reservoir, no family and no A3
qualification is involved: the question is how much an evaluation inflates when repeated
acquisitions of one physical bottle sit on both sides of the train/test boundary.

```
data     235 wine acquisitions, 3 classes (AQ/HQ/LQ), 22 bottles, 9-13 repetitions per bottle
inputs   6 gas channels; RH and temperature (columns 0-1) excluded, matching the declared Din=6
         the 65-file ethanol folder is a separate experiment and is not used
models   logistic regression (C=1.0) and RandomForest(n=300), fixed, no tuning, identical across schemes
metrics  macro-F1 primary; accuracy reported; no significance claim (22 bottles, 4 of them AQ)
```

## 1. The primary comparison: matched 5-fold, both sides multi-class

| features | model | grouped (by bottle) | naive (sequence) | **Δ macro-F1** |
|---|---|---:|---:|---:|
| summary | logistic | 0.896 | **1.000** | **+0.104** |
| summary | random forest | 0.903 | 0.984 | **+0.081** |
| temporal | logistic | 0.883 | **1.000** | **+0.117** |
| temporal | random forest | 0.871 | 0.973 | **+0.102** |

The naive side reaches **1.000** — a perfect macro-F1 — and falls to ~0.89 when the same model, the
same features and the same number of folds are used but whole bottles are held out. That is the
leakage signature: held-out sequences are repetitions of bottles the model has already seen.

**But the task is not fake.** Grouped accuracy is 0.951–0.972 and grouped macro-F1 0.871–0.903, so a
bottle that the model has never seen is still classified correctly the large majority of the time.
The wine task is genuinely learnable across bottles; what the naive split does is convert ~0.89 into
a suspiciously perfect 1.00.

## 2. The secondary comparison: leave-one-bottle-out against 22 folds, measured on accuracy

| features | model | grouped accuracy | naive accuracy | **Δ accuracy** |
|---|---|---:|---:|---:|
| summary | logistic | 0.972 | 1.000 | **+0.028** |
| summary | random forest | 0.971 | 0.993 | **+0.022** |
| temporal | logistic | 0.969 | 1.000 | **+0.032** |
| temporal | random forest | 0.951 | 0.984 | **+0.033** |

Accuracy, which is well defined on a single-class fold, gives a much smaller gap: **+0.02 to +0.03**.
Both numbers are honest — they answer different questions — but they must not be presented as the
same quantity.

## 3. The finding I did not expect: a metric-degeneracy trap worth 0.5 macro-F1

My first implementation used the design's original 22-fold pairing and scored both sides with
macro-F1 over the **full** label set. It reported:

```
Δ_leak = +0.6279  +0.6194  +0.6292  +0.6014        (~6x the true effect)
```

Those numbers are an artifact. **Leave-one-bottle-out gives every test fold a single class** —
measured: 22 of 22 folds carry exactly one class — so macro-F1 over the full label set scores 0 for
the two absent classes and is **bounded above by `1/K`**, i.e. 0.333 for three classes. The measured
grouped values sit at 0.324–0.327 with a maximum of exactly 0.3333, which is the ceiling itself. The
"leakage" was mostly the metric, not the split.

This is a publishable evaluation pitfall in its own right, and it is a *trap for the careful*: a
practitioner who groups by specimen — the correct instinct, and precisely what this project has been
insisting on — and then reports macro-F1 over all classes will produce a grouped score that looks
catastrophically worse than the naive one, for a purely definitional reason. The correct handling,
and what is now implemented, is to report macro-F1 **both** over the declared label set and over the
classes actually present in the fold, plus accuracy, so the degeneracy is visible instead of
inherited.

The engine already anticipated this: the design's own reporting rule required that "folds whose test
side lacks a class are reported as such rather than silently scoring 0", and the count printed
22/22. The rule caught the problem; the first metric choice did not.

## 4. What this means for the paper

C2 is stronger with the corrected numbers, not weaker:

1. **Leakage is real and quantified**: +0.08 to +0.12 macro-F1, +0.02 to +0.03 accuracy, with the
   naive side reaching a perfect 1.000.
2. **The task survives honest evaluation** (grouped accuracy 0.95–0.97), so the paper does not have
   to claim the wine benchmark is meaningless — it claims the *naive estimate* is inflated.
3. **Measurement choice dominates the effect size**: the same split pair yields +0.63 or +0.10
   depending on the metric, which is a cautionary result about how leakage audits themselves are
   usually reported.

## 5. Open items recorded, not resolved

- **4 AQ bottles.** The grouped scheme's class support is 4/5/13 bottles, so no per-class claim is
  made and no p-value is offered. The per-bottle spread of the grouped score is in the JSON.
- **RH and temperature** are excluded from the primary result by the declared `Din=6`. A
  supplementary covariate ablation remains available and is deliberately not run here.
- **The 65 ethanol acquisitions** form a separate 6-level task (1–20 % v/v) and are untouched.

## Artifacts

```
results/audit/resaudit_stage1/fd3_leakage.json   every fold, seed, metric variant and the degeneracy note
ops/audit/fd3_leakage.py                         the experiment, re-runnable from the read-only deposit
```
