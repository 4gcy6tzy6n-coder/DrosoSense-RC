# ResAudit-Food — pre-registration amendment 1: what the battery's own calibration found

**Status: amendment, 2026-09-23. Written AFTER the audit runner was built and its
invariant tests passed, and BEFORE any real family (F1–F5) exists.** Nothing in
`resaudit_food_preregistration.md` is edited; per this repository's amendment discipline a
new file is added instead.

**Scope of this amendment.** The runner is implemented and its engine invariants hold
(`tests/test_resaudit_invariants.py`, 58 tests; full suite 695 passed / 8 skipped). While
calibrating the runner against synthetic graphs with known verdicts, four MEASUREMENT
properties of the declared battery surfaced. They are recorded here because they change
what a Stage 1 result will mean, and because discovering them after seeing family scores
would have made them indistinguishable from excuses.

**No threshold is changed by this amendment.** A1–A5 and their constants are untouched.

---

## 0. Status at the close of this amendment

```
Stage 1 runner          = IMPLEMENTED / VERIFIED
A3 semantics            = NARROWED          (Section 2 below)
A4 construct validity   = RESOLVED -> RETIRED AS A GATE
                          (docs/resaudit_a4_construct_validity.md; rule 6.2 fired)
F1-F5 generation        = HOLD
```

## 0.1 Pre-family conventions frozen by this amendment

These are **calibration-derived conventions, not pre-registration rules**. They did not
exist in `resaudit_food_preregistration.md`; they were derived by measuring the battery
against synthetic graphs with known verdicts, and they are frozen HERE, before any F
family exists, so that they cannot be adjusted in response to an F family's score.

| convention | value | why it is needed | may it change after seeing F scores? |
|---|---|---|---|
| `rho_target` | **0.95** | A3 is normalized-scale-dependent (Section 1) | **No** |
| `leak` | **1.0** | the drive protocol; pure tanh, so no leak can manufacture memory | **No** |
| `gain` | **1.0** | the drive protocol | **No** |
| probe windows / length / seed / washout | 48 / 256 / 20260923 / 16 | fixes a single shared drive for A4/A5 | **No** |
| `K` | 16 | already fixed by the pre-registration | — |
| A1–A5 thresholds | unchanged | signed criteria | — |

`rho_target = 0.95` is the only value tested that is below 1.0 (so the recurrence is
contractive, as a reservoir must be) and above 0.5 (measured: 0.5 pushes a well-connected
ER graph below the A3 gate). It is recorded in every report's `provenance`, beside the
git head, so a reader can see which convention a result was produced under.

**No parameter sweep is performed.** `leak` and `gain` are frozen at the values above and
are not searched. If a later stage wants a different drive, that is a new amendment with
its own justification, not a fallback when a family fails.


---

## 1. A3's verdict depends on the spectral-radius normalisation (material)

`substrate_scores.krylov_score` does not rescale `A`. The Frobenius norms of `A^k·B`
therefore grow like `rho^k`, so when `rho` is appreciably above 1 the Krylov block
`[B, AB, …, A¹⁶B]` is dominated by its highest powers and the participation ratio
`(Σs)²/Σs²` collapses toward its minimum.

Measured, same wiring and same `B`, differing only by a scalar:

| substrate | `rho` | `D_eff` | gate | A3 |
|---|---:|---:|---:|---|
| ER, n=100, m=400, Din=4 | 3.921 | **1.001** | 8 | FAIL |
| same, rescaled | 1.000 | 12.029 | 8 | PASS |
| same, rescaled | 0.900 | 11.061 | 8 | PASS |
| same, rescaled | 0.500 | 6.927 | 8 | FAIL |
| ER, n=1000, m=4000, Din=16 | 4.060 | **1.004** | 32 | FAIL |
| same, rescaled | 0.900 | 37.951 | 32 | PASS |

`power_norms` makes the mechanism visible in every report: unnormalized, `||A³B||_F` is
more than ten times `||B||_F`; normalized, it is smaller than `||B||_F`.

**Consequence.** A substrate that "FAILS A3" may be failing because nobody rescaled it.
A3 as written is a *normalized-scale-dependent* test of controllability, and the
pre-registration does not name a normalisation. The framework is only credible if the
rule is declared once and applied to every family. **This amendment declares it:**

> Every family is scaled to a spectral radius of `rho = 0.95` by a single scalar
> multiplication before A3/A4/A5 are measured, and the scaling factor is recorded in the
> family's construction record. `rho = 0.95` is chosen as the only value tested here that
> is below 1.0 (so the recurrence is contractive, as a reservoir must be) and above 0.5
> (measured: 0.5 pushes a well-connected ER graph below the gate).

This is a declared choice, not a signed threshold, and it is recorded in the Stage 1
report's `provenance`. It must not be re-chosen after seeing a family score.

## 2. A3's semantics are narrowed: finite-horizon input-reachable state diversity (material)

### 2.1 The name A3 has been using is too broad

A3 has been called "**Krylov controllability**" and treated as a proxy for the substrate
being a *useful reservoir*. The nilpotent-chain result in Section 4 shows that reading is
wrong. What

```
D_eff([B, A·B, …, A^K·B])
```

actually measures is: **how many independent state directions the input can reach within
a finite horizon of `K = 16` propagation steps.** That is a property of the reachable
subspace's *diversity at finite depth*. It is not a statement about recurrence, about
memory, or about whether the resulting dynamics are useful.

**A3 is hereby renamed and redefined, machine-readably:**

| before | after |
|---|---|
| `A3` "Krylov controllability" | `A3` **finite-horizon input-reachable state diversity** |

The criterion id, threshold (`>= 2·Din`), and measurement are UNCHANGED. Only the
claimed meaning is narrowed, because the broad meaning was not supported.

### 2.2 The explicit caveat, which the report now states

> **A3 PASS is neither sufficient evidence of recurrence nor of useful memory.**

This is not a hedge; it is a falsified claim being withdrawn. A pure feed-forward chain
passes A3 (Section 4), so A3 cannot be evidence of recurrence. Any figure, table or
sentence that presents an A3 PASS as showing that a substrate "has recurrence" or "has
memory" is now a misreading, and the Stage 1 report carries this caveat in its
`provenance` block so a downstream reader cannot miss it.

### 2.3 Consequence for what A3 can be used for

Permitted:

* as a **pre-training admissibility screen**: it is cheap, model-free, and it separates
  substrates that the usual structural intuitions (SCC, cycle coverage, edge count, mean
  out-degree) rank equally. The motivating case is a substrate that is structurally
  impeccable and has almost no reachable state diversity;
* as a **descriptive** quantity reported with its mechanism (`D_eff` against the gate, and
  `power_norms`, which show whether the block is scale-dominated).

Not permitted:

* as evidence that a substrate has recurrence or memory;
* as a prediction of food-stage accuracy (unobservable in this design — Section 5);
* as a quality score. It is a screen with a floor, not a ranking.

### 2.4 The structural result that survives, and is the paper's real engine

With A3 narrowed, the claim that carries the framework is the one the calibration made
measurable:

```
largest_SCC = 100 %   +   cycle coverage = 100 %   +   many edges   +   mean out-degree ~ 80
        ⇏   finite-horizon input-reachable state diversity
```

Note what this does NOT say, and must not be written as: it does not say that such a
substrate has no memory, and it does not say that structural connectivity is irrelevant.
It says structural connectivity did not, in the measured case, deliver reachable state
diversity — which is a necessary capacity, not a sufficient one.

### 2.1 A1's SCC term is nearly implied by its isolated term (minor, but it bounds the table)

Nodes with out-degree 0 are singleton SCCs, so they depress `largest_SCC/N`. With the
signed constants, clearing the isolated ceiling (≤ 0.02) caps that depression at 0.98,
and `0.98 > 0.90`. Therefore:

* for any graph that clears the isolated term, the SCC term passes UNLESS the graph has
  several large nontrivial components;
* "isolated fails while mean out-degree passes" is not reachable at small scale, because
  out-degree-0 nodes also drag the mean below 2.0.

Measured: a `k`-out ring with `k = 1` fails ONLY the mean-out-degree term; two disjoint
rings fail ONLY the SCC term; a 99-node `k = 3` ring plus 4 isolated nodes fails ONLY the
isolated term (at `N = 103`, where the isolated nodes are a negligible share of the
out-degree sum). The tests assert these reachable combinations rather than pretending the
three terms are independently reachable.

**No action.** This is a property of the signed thresholds, which are not changed.

## 2.2 A4 (recurrent memory) is small, sign-unstable, and non-monotone (material)

`memory_metric` correlates one state channel against lagged inputs; the `A := 0` control
is the same quantity for the feed-forward path. Measured on bounded uniform drive
(`windows = 48`, `length = 256`, washout 16):

| graph | leak | `M_recurrent` | `M_{A:=0}` | A4 |
|---|---:|---:|---:|---:|
| coprime cycles (N=28) | 1.0 | 0.0080 | 0.0093 | **−0.166** |
| 2-out ring (N=40) | 1.0 | 0.0052 | 0.0135 | **−1.583** |
| 1-out ring (N=1000, Din=16) | 1.0 | 0.0129 | 0.0113 | 0.129 |
| 2-out ring (N=1000, Din=16) | 1.0 | 0.0075 | 0.0113 | **−0.502** |
| 5-out ring (N=1000, Din=16) | 1.0 | 0.0088 | 0.0113 | **−0.278** |

Two properties stand out, and neither is fixed by scale:

1. **The metric is O(0.01) everywhere tested**, at `N = 40` and `N = 1000`. The signed
   contribution `(M − M_{A:=0})/M` is a ratio of two small numbers, so it is dominated by
   probe noise rather than by memory.
2. **It is NEGATIVE for graphs with genuine recurrence, and non-monotone in recurrent
   coupling**: at `N = 1000` the contribution moves from `+0.129` (k=1) to `−0.502` (k=2)
   to `−0.278` (k=5). Adding recurrence lowers the score.

The feed-forward control is not a weaker version of the recurrent one here: on these
substrates the recurrent path carries no more lagged-input correlation, so subtracting it
can only produce a non-positive number.

**Consequence.** A4 as presently measurable behaves like a criterion that most substrates
fail, for reasons that are about the metric's magnitude rather than about memory. This is
exactly the case the pre-registration's decision rule 6.2 anticipates ("if F5 clears A3
but fails A4/A5, the battery's primary gate is mis-specified"), and it is recorded here
BEFORE F1–F5 exist so that a later A4 result cannot be presented as a discovery.

**What this amendment does NOT do** is tune A4 into a pass. The declared protocol
(`leak = 1.0`, `gain = 1.0`, the frozen lags, the shared probe) stands, and A4 is reported
as measured.

**Outcome (completed in the same round).** The pre-declared recurrence-ladder audit in
`docs/resaudit_a4_construct_validity.md` ran on the calibration families and found A4
**not directional**: 0 of 6 ladder-applicable families satisfied the direction rule, the
recurrent memory *decreases* as recurrence strengthens, it is O(0.01) and does not
separate `rho = 0` from `rho = 0.95`, and the observable is negative at the operating
point on 5 of 6 families. **Decision rule 6.2 therefore fired and A4 is retired as a
qualification gate**: it is still measured and reported, `BLOCKING_CRITERIA` no longer
contains it, and the threshold was NOT moved. A5 is the only remaining downstream gate.

## 3. A nilpotent chain PASSES A3 (documents the gate's stringency)

The textbook uncontrollable example — a pure feed-forward lower shift — measures
`D_eff = 17.0` against a gate of `2·Din = 2.0` and **passes**, because each `A^k·B` is a
fresh standard basis vector and the Krylov columns are orthogonal.

This is not a bug; it calibrates how weak the gate is. `D_eff` of a block with `n` equal
singular values is exactly `n`, and the block has `(K+1)·Din = 17·Din` columns, so the
gate `2·Din` sits far below the dense ceiling and only asks that the input's reach survive
propagation with at least twice its own dimension. A short recurrence can clear it by
accident.

**Consequence.** A3-PASS on a small substrate is weak evidence on its own. The Stage 1
table must therefore be read with the mechanism (`D_eff` against the gate, and
`power_norms`) and not as a single yes/no.

---

## 4. The F5 circularity, and what the paper may claim

Recorded here as a design commitment so it cannot drift in the write-up.

F5 is **constructed** to satisfy A3 (pre-registration Section 4: "F5 is built to maximise
A3 subject to the same budget, by a declared, task-blind procedure"). Therefore:

* "F5 passes A3" is a **construction property**, not a result. It cannot be a
  contribution.
* "A3 predicts poor food performance" is **not testable in this design**, because A3-FAIL
  families are never run on food data (decision rule 6.1). The relationship is
  unobservable by construction, and the runner says so explicitly in its skipped-A4/A5
  reason string.

The claims the design licenses are narrower, and are the ones to write:

> A3 is a **pre-training admissibility screen** that excludes reservoirs lacking
> sufficient controllability capacity. The paper's evidence for its value is that it
> costs nothing to apply, that it separates substrates that the usual structural
> intuitions rank equally, and that the substrate it selects is the one taken to the food
> stage.

The paper must **not** claim that A3 FAIL predicts low food accuracy. Testing that
prediction would require a separate, strictly isolated study in which A3-FAIL families are
run on food tasks — which this design deliberately does not do, because not running them
is what makes the food stage affordable.

The strongest surviving main line remains the one the pre-registration already names:

> 100 % SCC + 100 % cycle coverage + many edges ⇏ a useful reservoir dynamics; and the
> checks that do separate substrates are controllability, memory contribution and state
> expansion, measured before training.

---

## 5. What is now in place, and what is next

Implemented and tested (no food data touched anywhere in the package):

| artefact | what it provides |
|---|---|
| `resaudit/criteria.py` | the frozen A1–A5 thresholds as constants, and the three-valued outcome |
| `resaudit/family.py` | the sealed `FamilySpec`; task-blindness enforced structurally, not by convention |
| `resaudit/battery.py` | the A1–A5 gate engine, the A3-first ordering rule, the Stage 1 table |
| `resaudit/toys.py` | calibration graphs with measured verdicts, including the four findings above |
| `tests/test_resaudit_invariants.py` | the seven engine invariants, plus one test per finding |
| `ops/audit/resaudit_stage1.py` | the task-blind Stage 1 entry point; writes one JSON report |
| `resaudit/a4_validity.py` | the A4 construct-validity ladder audit (calibration families only) |
| `docs/resaudit_a4_construct_validity.md` | the A4 verdict: DESCRIPTIVE ONLY / retired as a gate |

Next, unchanged from the pre-registration's order of work:

```
3. build F2/F3/F4 generators                                    next
4. build F5 to the declared task-blind rule
5. audit F1-F5, report A1-A5 with verdicts                      (no food data)
```

Two things the generators must respect, from this amendment: scale every family to
`rho = 0.95` before measurement and record the factor; and expect A4 to be the criterion
most likely to drive decision rule 6.2.
