# ResAudit-Food — pre-registration amendment 7: freeze of the positive-method track; the paper becomes a methodological audit

**Status: scope amendment, 2026-09-23. Written AFTER Stage 1 completed and BEFORE the leakage
experiment was run.** The original pre-registration and amendments 1–6 are immutable; this file
edits nothing and is added.

Operator decision recorded: *"Freeze ResAudit-Food positive-method track. Proceed with the
methodological audit paper. Do not change A3, add no new family, do not reopen the food-efficacy
stage. Keep the Stage 1 negative result, and write 'why we did not go on to train food models' into
the paper as evidence."*

## 1. What is frozen, and on what evidence

| Track | Status | Reason |
|---|---|---|
| Positive-method track (a qualification framework that improves food-sensing performance) | **FROZEN** | Stage 1 produced no qualifying family at either width |
| Amendment to the A3 criterion (`2·Din` → 1.5, 1.7, …) | **REFUSED** | it would be redrawing the pass mark after seeing the result |
| New families (F6, F7, …) or a different input geometry | **REFUSED** | the project pre-registered a stop-loss; reopening it is a new study |
| Food-efficacy stage on FD1 (tea) / FD2 (coffee) | **FROZEN**, decision rule 6.1 | no family is admissible to run on food data |
| FD3 (wine) leakage control | **PROCEEDS** | it asks a different question and needs no qualification |
| Stage 1 negative result | **RETAINED as a contribution** | see C3 |

Four measured facts are sufficient to fix the scope, and none of them is "the threshold was
slightly too high":

1. **`D_eff/Din` is width-invariant** (1.304/1.314, 1.314/1.287, 1.709/1.703, 1.556/1.544 across
   `Din = 6, 8`), so the failure is not a mis-chosen input width — the shortfall is a constant
   multiplicative factor against a gate that scales identically.
2. **F2 ≈ F1** (1.314 vs 1.304): randomising wiring while preserving the degree sequence changes
   essentially nothing.
3. **F3 ranks highest** (1.709): destroying the topology outright *raises* the measure above both the
   biological substrate and the family built to satisfy it.
4. **F5 fails the criterion it was constructed for, and fails A1** at F1's real edge budget.

## 2. The repositioning

The paper is no longer *"pre-training reservoir qualification for food sensing"*, because that claim
is not supported. It becomes an audit paper. Candidate titles, in order of preference:

1. **Files Are Not Samples: An Audit of Independent Units and Evaluation Validity in Food Sensor
   Learning**
2. **When Is a Food-Sensing Benchmark Actually Independent? Auditing Repetitions, Splits, and
   Reservoir Qualification**
3. **Auditing Before Modeling: Independent Units, Split Leakage, and Construct Validity in Food
   Sensor Evaluation**

Candidate 1 is preferred because a reader grasps the problem from the title alone.

**Venue is deliberately not decided here.** The audit framing fits TAFE's *AgriFood electronics*
scope less directly than the original algorithmic framing did. The version will be organised first
and the venue chosen from the finished abstract and contribution list, rather than defending a
target that the evidence no longer serves.

## 3. The three contributions

### C1 — Dataset-unit audit

A provenance-aware protocol that separates **files**, **measurements/sequences**, **repetitions**
and **independent experimental units**, applied to five public food-sensing datasets, exposing unit
inflation from 1× to 60×. Measured, already in hand:

```
Wine      300 files -> 235 wine sequences ->  22 bottles        (10.7x inflation over units)
Pork      420 files -> 420 sequences     ->   7 mixtures        (60x inflation)
Tea       234 sequences -> 78 unit ids (77 confirmed chops + 1 steam unit)
Coffee     58 files ->  58 samples        -> 58 units           (1x: measurement IS the unit)
Freshness  59 files ->  59 recordings     -> UNRESOLVED (no specimen identifier exists)
```

The taxonomy this yields is the point: recoverable replication structure (tea), measurement-equals-
unit with no batch field (coffee), excellent time axis with unrecoverable identity (freshness), file
count as pure inflation (pork), and legitimate replication that must be nested (wine).

### C2 — Evaluation-leakage audit

Using wine, which has real bottle identity and 9–13 repetitions per bottle, directly compare a
**naive sequence-random** evaluation against a **bottle-grouped** one, and quantify

```
Delta_leak = macroF1(naive sequence split) - macroF1(bottle-grouped split)
```

**This needs no reservoir and no A3 qualification**: the question is not which substrate is better
but how much a split that puts repeated acquisitions of one physical unit on both sides of the
train/test boundary inflates the estimate. Design pre-declared in §4.

### C3 — Construct-validity audit of the qualification criterion

Stage 1 is retained and reported, not buried:

| family | `D_eff` @`Din=6` | `D_eff` @`Din=8` | gate |
|---|---:|---:|---:|
| F1 connectome | 7.822 | 10.514 | 12 / 16 |
| F2 degree-preserving | 7.884 | 10.297 | 12 / 16 |
| F3 topology-destroyed | 10.256 | 13.628 | 12 / 16 |
| F5 A3-constructed | 9.334 | 12.349 | 12 / 16 |

The emphasised finding is **not** "everyone failed" but the ordering **F3 > F5 > F2 ≈ F1**: the
criterion does not separate the biological substrate from the topology-destroyed control in the
direction its introduction assumed. The framing is *criterion fails its own construct-validity
expectations*, not *the reservoirs were inadequate*.

The paper's honest claim about the framework is the inverse of the original one:

> The qualification criterion failed; the governance worked. The gate did the single most important
> thing a gate can do — it stopped an unsupported hypothesis from being carried into downstream food
> experiments.

## 4. The FD3 leakage experiment, pre-declared

Everything here is fixed before measurement, as the operator requires for the one remaining
experiment.

**Task.** Wine spoilage threshold, 3 classes (AQ/HQ/LQ), from the 6 gas channels. Each file is one
acquisition of a sample drawn from one bottle: 3330 steps at a uniform 0.1 s, no time column.

**Data.** The 235 wine sequences. **RH and temperature (columns 0–1) are excluded**, matching the
declared `Din=6`. The 65-file ethanol folder is a separate experiment and is **not** used here.

**Split schemes, matched at 22 folds each.**

| scheme | unit of the fold | folds | notes |
|---|---|---|---|
| **grouped (primary)** | bottle — leave-one-bottle-out | 22 | all 9–13 repetitions of a bottle always on the same side |
| **naive (control)** | sequence — random partition | 22 | 5 independent seeds; each partition assigns every sequence to one fold |

**Models — plain, fixed, non-novel.** No reservoir, no family, no tuning: (i) multinomial logistic
regression on standardised per-channel summary features; (ii) `RandomForestClassifier` at fixed
hyper-parameters; (iii) a temporal baseline, ridge classification on the downsampled series
(20 time points × 6 channels). Hyper-parameters are fixed in advance and identical across the two
split schemes; no per-scheme tuning, and no model selection by test score.

**Metric.** `macroF1` **primary**; accuracy secondary. Reported per fold, then aggregated.

**Reporting rules, fixed now.**

- `Delta_leak` is reported as a point estimate with the **per-bottle and per-seed spread**, never as
  a significance claim: with 22 bottles, of which only **4** are AQ, no p-value story is available
  and none will be written.
- The grouped scheme is deterministic; the naive scheme reports mean and range over its 5 seeds.
- Folds whose test side lacks a class are reported as such rather than silently scoring 0; the
  number of such folds is stated.
- **A negative or small `Delta_leak` is a publishable result** and will be reported as found.

## 5. The A2 finding, framed as an applicability conflict rather than a failure

Two different things must not be merged:

- **F3 — criterion/definition incompatibility, not a scientific failure.** A2 requires the parent's
  degree sequence to be preserved **exactly**, while F3's declared contrast *deliberately discards*
  the degree distribution; that is what makes it topology-destroyed rather than degree-matched. A2 is
  therefore **structurally inapplicable** to F3 under the frozen specification. This is a logical
  conflict between two frozen declarations.
- **F2 — a measured failure of a valid criterion.** `edge_overlap = 0.203709` against a ceiling of
  `0.20`, plus edge-weight multisets that differ while node strengths match to `5.8e-16`. It misses
  by 0.0037 and **stays FAIL**. The swap budget is frozen and is not topped up after the fact.

## 6. What the paper does not contain

No reservoir efficacy result, no FD1/FD2 task performance, no A3 threshold revision, no new family,
no connectome-substrate novelty claim, and no claim that A3 FAIL predicts poor food accuracy — that
prediction is untestable in this design, because A3-FAIL families are never run on food data.
