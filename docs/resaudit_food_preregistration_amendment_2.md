# ResAudit-Food — pre-registration amendment 2: withdrawal of the memory-contribution claim

**Status: amendment, 2026-09-23. Written AFTER the A4 construct-validity audit completed
and BEFORE any formal family (F1–F5) was built or measured.**

**The original pre-registration is immutable.** This amendment does not edit
`docs/resaudit_food_preregistration.md`. That file's identity at the time of writing:

```
git rev-parse HEAD:docs/resaudit_food_preregistration.md
  350edc2b8cf88dbbc77b162ccd4a40a56ff5c71d
git hash-object docs/resaudit_food_preregistration.md
  350edc2b8cf88dbbc77b162ccd4a40a56ff5c71d     <- identical: unmodified
```

Everything below is a record of what was promised, what was measured, and what changed as
a result. The chain is kept intact so that a later reader can see the original commitment,
the falsification, and the timing — rather than reading a document that has been quietly
brought into line with its results.

---

## 1. What the original pre-registration promised about memory

The pre-registration listed five checks (Section 3) and named A4 as a **downstream gate**
that "a qualified reservoir must still clear":

| id | check | threshold |
|---|---|---|
| A4 | recurrent memory | `(M − M_{A:=0}) / M ≥ 0.20` |

and, in Section 2, described the battery's separating power as covering three dynamical
quantities:

> *the checks that do separate substrates are controllability, memory contribution and
> state expansion*

## 2. What was measured

`docs/resaudit_a4_construct_validity.md` (runner `resaudit/a4_validity.py`, report
`results/audit/resaudit_a4_validity/A4_construct_validity.json`) ran a pre-declared
recurrence-strength ladder — the spectral radius of the same wiring and input, over
`rho ∈ {0.00, 0.25, 0.50, 0.75, 0.95}`, five seeds, `rho = 0` reproducing the `A := 0`
control — on the seven calibration families C1–C7 only.

The pre-declared direction rule was: **directional** iff, for every applicable family,
`M_recurrent` is non-decreasing in `rho`, Spearman is positive, and the observable is
positive at the operating point.

**Result: `NOT_DIRECTIONAL`. 0 of 6 ladder-applicable families satisfied it.**

```
C1 ring+chords              Spearman +0.400   A4 @ rho=0.95 = -0.1237
C2 coprime cycles           Spearman +0.700   A4 @ rho=0.95 = +0.0493
C3 single 2-cycle           Spearman -0.900   A4 @ rho=0.95 = -0.1610
C4 zero input               n/a               A4 @ rho=0.95 = +0.0000
C5 nilpotent chain          ladder not applicable (base spectral radius = 0)
C6 20-cycle (A2 parent)     Spearman -0.900   A4 @ rho=0.95 = -1.0235
C7 20-cycle, reversal perm. Spearman -0.900   A4 @ rho=0.95 = -1.0235
```

The four facts that make this fatal rather than merely disappointing:

1. `M_recurrent` **decreases** as recurrence strengthens (`0.0170 → 0.0109` from
   `rho = 0.25` to `0.95` on C6/C7). The trend is opposite to what A4's definition assumes.
2. It is **O(0.01) and does not separate the extreme rungs** (`rho = 0` vs `rho = 0.95`
   differ by 0.0001 on C1). The ratio is a ratio of small numbers.
3. The observable is **negative at the operating point on 5 of 6** applicable families, so
   A4 does not merely miss 0.20 — it reports that recurrence *subtracts* memory.
4. The most favourable family peaks at **+0.148 at `rho = 0.75`** and falls to `+0.049` at
   the operating point. No rung of any ladder reached the threshold.

## 3. Item 1 — A4 is retired as a qualification gate

> **A4 is retired as a qualification gate because the frozen observable failed directional
> construct validation on the calibration families. The original threshold 0.20 is retained
> in the historical record but is no longer used for qualification.**

Precisely:

* the threshold constant `A4_MIN = 0.20` remains in `resaudit/criteria.py` and remains
  reported in every A4 result and in the report's `thresholds` block, so the original
  commitment stays visible;
* A4 is **still measured and reported** on every family whose A3 passes — it has become a
  descriptive statistic, not a silent omission;
* A4 is removed from the set that can disqualify a family;
* **the threshold was not moved.** No value of it would make a non-directional observable
  directional, and adjusting it after seeing this result is the tuning the pre-registration
  forbids.

This action is the pre-registration's own **decision rule 6.2** firing, not a deviation from
it. That rule says: *"If F5 clears A3 but fails A4/A5, the battery's primary gate is
mis-specified and that is reported as a battery finding before any task result is
discussed."* The finding here is stronger than the rule anticipated — A4 is mis-specified
independently of F5, and at the construct level rather than the threshold level — which is
why it is recorded as a withdrawal rather than a caveat.

**Timing, stated explicitly: the falsification occurred before F1–F5 existed.** No family
was built, no family was measured, and no food data was touched at any point in this
process. The audit's input was seven synthetic calibration graphs.

## 4. Item 2 — the paper's memory claim is withdrawn

The Section 2 sentence is **withdrawn** and replaced:

> ~~the checks that do separate substrates are controllability, memory contribution and
> state expansion~~

> **the checks that do separate substrates are finite-horizon input-reachable state
> diversity and state expansion**

With two explicit limits, so the replacement is not read as a silent upgrade:

* **A3 ≠ controllability in full.** A3 measures how many independent state directions the
  input can reach within a horizon of `K = 16` steps. It is not a controllability proof,
  not a rank test of the full reachable subspace, and not a statement about the system's
  behaviour at other horizons. This narrowing is recorded in amendment 1 Section 2.
* **A3 ≠ useful memory.** A pure feed-forward nilpotent chain passes A3 (measured:
  `D_eff = 17.0` against a gate of `2.0`), so A3 cannot be evidence of recurrence or of
  useful memory. Amendment 1 Section 2.2 carries this as a constant,
  `A3_CAVEAT`, into every report.

The motivating negative is unaffected by this withdrawal, because it never rested on A4:
`100 % SCC + 100 % cycle coverage + many edges ⇏ finite-horizon input-reachable state
diversity`, measured on the frozen project's own substrate.

## 5. Item 3 — the Stage-1 blocking logic is frozen

```
Blocking = {A1, A2, A3, A5}
```

with the qualifications that make it unambiguous:

| criterion | blocking? | condition |
|---|---|---|
| A1 | yes | always applicable (structure is always measurable) |
| A2 | **only when applicable** | `NOT_APPLICABLE` never blocks; it is not a soft failure |
| A3 | yes | the primary gate; also decides `food_eligible` (rule 6.1) |
| A4 | **no — permanently descriptive-only** | measured and reported, never disqualifying |
| A5 | yes | the only remaining downstream gate |

Frozen in code as `criteria.BLOCKING_CRITERIA = ("A1", "A2", "A3", "A5")`, with
`DESCRIPTIVE_ONLY_CRITERIA = ("A4",)` and `DOWNSTREAM_CRITERIA = ("A5",)`. A test asserts
that an A4 failure does not block qualification, so re-promoting A4 to a gate would be a
visible change rather than a silent one.

**Reporting rule, unchanged and permanent:** a Stage 1 report emits two separate fields and
never a merged pass flag.

```
food_eligible          cleared A3, so the family may enter the food stage
construct_qualified    cleared every ADMISSION and evaluated BLOCKING criterion
```

Neither field means "Stage 1 passed", and the phrase must not appear. A family can be
`food_eligible = yes` with `construct_qualified = no`; C5/C6/C7 are exactly that case.

## 6. Item 4 — why A4 is not redesigned here

> **No replacement memory metric is introduced before F1–F5. Doing so after observing the
> failure of A4 would constitute post-calibration construct substitution.**

This is the load-bearing sentence of this amendment, and it is deliberately restrictive.
After seeing A4 fail, a search over alternative memory observables — different lags,
different readouts, a nonlinear or information-theoretic measure, a trained readout,
a different control — would be free to select whichever candidate happens to pass on the
same calibration families that just falsified A4. The resulting criterion would carry no
independent evidence, because its selection and its validation would share the same data
and the same failure.

Therefore, for the duration of this project:

* **no A4b, A4′, or replacement memory criterion is introduced;**
* if a memory criterion is wanted later, it must be pre-declared as a **new construct with
  its own directional audit on data the current audit did not use**, and that audit must run
  before the construct is applied to any family;
* the paper reports the memory finding as a **negative result about the construct**, which
  is a legitimate and useful contribution: it documents that the obvious recurrence-derived
  memory statistic is not directionally valid at this scale.

## 7. Family-battery design rules (binding on the next stage)

Recorded here so the F1–F5 design is governed by rules fixed before the families exist.
These do not release the hold by themselves; they constrain what the design may do.

**F1–F5 are five ROLES, not five performance levels.** They must form a falsifiable
structural contrast:

| family | role |
|---|---|
| F1 | biological / reference structure |
| F2 | degree- and statistics-matched random control |
| F3 | topology-destroyed but density- and input-matched control |
| F4 | recurrence-preserving structural counterfactual |
| F5 | deliberately A3-qualified construction |

**Rule 1 — no food task may select a family parameter.** Every topology parameter,
normalisation factor, input geometry, and seed protocol is fixed before any food data is
read. The frozen conventions in amendment 1 Section 0.1 apply to every family.

**Rule 2 — families differ by as close to ONE structural factor as possible.** Otherwise the
only reportable result is `F_i ≠ F_j`, with degree, SCC, density, input reachability and
spectral scale all confounded. Each family's construction record must name the single
factor it is intended to vary and the factors it holds fixed.

**Rule 3 — `Stage 1 passed` is never reported.** Only `food_eligible` and
`construct_qualified`, separately, permanently.

**Rule 4 — F5's claim is bounded by its own construction.** F5 passes A3 **by
construction**, so it can only answer:

> *given that finite-horizon input-reachable diversity is guaranteed, how does downstream
> food performance behave?*

It **cannot** answer:

> *does A3 PASS cause better food performance?*

That causal question remains **unidentifiable** under the current gate design, because
A3-FAIL families are never run on food data (rule 6.1). Any sentence that presents F5's
result as validating A3's predictive power is a misreading; the runner already states this
in its own skipped-A4/A5 reason string.

## 8. Effect on the frozen status

```
Stage 1 runner          = IMPLEMENTED / VERIFIED
A3 semantics            = NARROWED (finite-horizon input-reachable state diversity)
A4 construct validity   = RESOLVED -> RETIRED AS A GATE (this amendment, Section 3)
A4 replacement          = NOT INTRODUCED BEFORE F1-F5 (Section 6)
Blocking logic          = FROZEN at {A1, A2, A3, A5} (Section 5)
Original preregistration = IMMUTABLE (hash 350edc2b..., unchanged)
F1-F5 generation        = next stage, under the rules in Section 7
```

Nothing here licenses a task result. No food data has been read, no family exists, and no
performance number in this project has been produced.
