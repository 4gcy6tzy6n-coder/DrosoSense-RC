# ResAudit-Food Stage 1 — decision record: pre-registration rule 6.4 invoked

**Status: DECISION RECORD, 2026-09-23. Stage 1 is formally CLOSED.**

```
ResAudit Stage-1 A3 result          = FROZEN_NEGATIVE_RESULT
Food stage                          = NOT_RELEASED_BY_STAGE1
M5_structural_dynamics historical   = INVALIDATED / RECOMPUTATION_REQUIRED
ledger_repair_E9 historical         = INVALIDATED / RECOMPUTATION_REQUIRED
```

This document fixes the four things the decision requires: that rule 6.4 was invoked, the
numbers that support it, that the food stage is **not released**, and that no
threshold/model/family rescue is permitted.

It does not renegotiate anything. The pre-registration remains immutable
(`docs/resaudit_food_preregistration.md`, blob `350edc2b…`); findings live in amendments 1–6,
the A4 construct-validity audit, the contrast matrix, the measurement-layer impact audit, the
V3 reselection, and the Stage-1 findings.

---

## 1. Rule 6.4 is invoked

The pre-registration's decision rule 6.4 states that if **every** family fails A3, the finding
is that the declared thresholds are too strict for this substrate size and edge budget, that
the thresholds are **not** lowered silently, and that the consequence is a paper about what a
lightweight reservoir cannot be — *"a weaker but honest outcome."*

That condition is met. Rule 6.4 is therefore invoked, and the honest outcome is taken rather
than a rescue.

## 2. The numbers that support it

Measured at HEAD `208475ed`, on F1 materialized to `VERIFIED_FOR_STAGE1`, at the battery's
declared operating radius `ρ = 0.95`. No food data was read; no label was seen; no task metric
exists anywhere in this stage.

| family | `Din` | `N` | `M` | A1 | A2 | `D_eff` | gate `2·Din` | A3 | `D_eff/Din` | gate fraction |
|---|---:|---:|---:|---|---|---:|---:|---|---:|---:|
| F1 connectome (biological reference) | 6 | 1000 | 80443 | PASS | n/a | 7.822 | 12 | **FAIL** | 1.304 | 0.65 |
| F2 degree-preserving control | 6 | 1000 | 80443 | PASS | FAIL | 7.884 | 12 | **FAIL** | 1.314 | 0.66 |
| F3 topology-destroyed control | 6 | 1000 | 80443 | PASS | FAIL | 10.256 | 12 | **FAIL** | 1.709 | 0.85 |
| F5 attempted A3-qualified construction | 6 | 1000 | 80443 | **FAIL** | FAIL | 9.334 | 12 | **FAIL** | 1.556 | 0.78 |
| F1 connectome | 8 | 1000 | 80443 | PASS | n/a | 10.514 | 16 | **FAIL** | 1.314 | 0.66 |
| F2 degree-preserving control | 8 | 1000 | 80443 | PASS | FAIL | 10.297 | 16 | **FAIL** | 1.287 | 0.64 |
| F3 topology-destroyed control | 8 | 1000 | 80443 | PASS | FAIL | 13.628 | 16 | **FAIL** | 1.703 | 0.85 |
| F5 attempted A3-qualified construction | 8 | 1000 | 80443 | **FAIL** | FAIL | 12.349 | 16 | **FAIL** | 1.544 | 0.77 |
| **F4** | — | — | — | — | — | — | — | **BLOCKED / NOT EVALUATED** | — | — |

A4 and A5 are absent by construction rather than missing: the engine evaluates them only
downstream of A3. F4 keeps an explicit blocked row because the pre-registration contained F4,
and a silent omission would misrepresent the design.

**Every family is below the gate at both widths.** The best case, F3, reaches
**0.85 of the gate** and still does not cross it.

### The width behaviour

`Din` increases by 33 % (6 → 8) while `D_eff/Din` changes by at most 2 %: F3 1.709 → 1.703,
F1 1.304 → 1.314, F2 1.314 → 1.287, F5 1.556 → 1.544. Since `D_eff` grows in proportion to
`Din` while the gate also grows in proportion to `Din`, the two lines are parallel and do not
cross within the tested range.

> **Within the preregistered input-width range, increasing input width does not close the
> finite-horizon input-reachable diversity deficit. The deficit is approximately multiplicative
> rather than additive under the frozen Stage-1 conventions.**

**Boundary, stated so it cannot be over-read.** This is established for `Din ∈ {6, 8}` only.
`N`, connectivity scale and input-coverage fraction were **not** varied, and any of them could
in principle move `D_eff/Din`. The claim that "a larger or differently connected substrate
could not clear the gate" is **not** supported by this experiment and must not be written.

## 3. What the result means, and what it does not

**Not** "the A3 threshold is set too high" — that would be a claim about the threshold, and the
threshold was neither derived nor revised here.

**Not** "the connectome fails as a reservoir" — that narrower statement was already the frozen
DrosoSense-RC finding, and it is not what this stage adds.

**It is:**

> **The preregistered construct requirement is unattained by the entire tested structural
> battery under the frozen operating conventions — including the biological reference, the
> matched random controls, the topology-destroyed control, and an attempted positive-control
> construction.**

F5's failure is the load-bearing part of that sentence. If only F1 had failed, "the fly's
structure is special" would remain available. If F2 and F3 had failed, "random controls are
simply worse" would remain available. F5 is the construction **built specifically to guarantee
A3**, and it fails once `N`, `m` and `ρ` are frozen jointly. That moves the question from
topology ranking to:

```
construct attainability under the specified reservoir regime
```

## 4. Stage 1 stops here — no rescue is permitted

Each of the following would convert a clean negative study into a post-hoc rescue, and each is
**prohibited** by this record:

| prohibited action | why |
|---|---|
| lowering the A3 gate | the pre-registration forbids silent threshold changes; rule 6.4 forbids them explicitly |
| widening `Din` to search for a crossover | a crossover search is a result-dependent design change |
| building a new F5 until one passes | positive-control engineering; the failed attempt is recorded as the result |
| unlocking A3-failing families for food | violates rule 6.1 and would confound the pre-registration logic |
| redefining A3 | the construct is the thing under test, not a free parameter |

The permitted action is to report the negative, and that has been done.

## 5. The food stage is NOT RELEASED

```
Food evaluation = NOT_RELEASED_BY_STAGE1
```

This is **not** "not yet run". It is that the pre-registration's own logic does not release it:
rule 6.1 states that no family runs food tasks unless it clears A3, and no family cleared A3.
Zero families are `food_eligible`.

The distinction matters for the write-up. Running food now would not merely be premature; it
would make any resulting performance number unattributable, because A3 — the structural
property the benchmark is supposed to be about — is demonstrably absent in every candidate.
A food result could then be read as evidence about a reservoir property that no substrate here
possesses.

## 6. What this says about the framework's value

The paper's main line has changed, and the change is the finding:

```
before:  connectome reservoir -> food task -> performance
after:   before any downstream benchmark, does the proposed reservoir family
         even satisfy the structural/dynamical construct it is assumed to possess?
         answer: no, under the preregistered operating regime
```

Which supports the methodological claim:

> **A task-blind audit prevented a downstream benchmark from attributing food-task performance
> to a reservoir property that the candidate substrates did not demonstrably possess.**

The scope of that claim is narrow and must stay narrow: it is **not** that these reservoirs are
useless for food sensing. It is that if the food stage were run now, its results could not be
interpreted as arising from the structural property A3 represents. That is precisely what the
audit exists to protect, and it is what the audit delivered.

## 7. M5 and E9 are a separate cleanup, not a reopening

`M5_structural_dynamics` and `ledger_repair_E9` remain
`INVALIDATED / RECOMPUTATION_REQUIRED` because they were produced through the directed-matrix
spectral-radius defect.

They are **not** blockers of this decision, and recomputing them **cannot reopen it**: A3 is
computed by `krylov_score` on the materialized families and never consults ρ, so neither
artifact enters the Stage-1 A3 verdict. Their recomputation is for the cleanliness of the
DrosoSense-RC historical record.

If a recomputation changes one of their conclusions, that is a **scientific correction to the
frozen project**, escalated as such — not a revision of this Stage-1 result.

## 8. Closure

```
Stage 1                              = CLOSED
Stage-1 primary result               = FROZEN NEGATIVE RESULT (construct-level)
A1-A5 thresholds                     = unchanged since pre-registration
Pre-registration                     = immutable, blob 350edc2b8cf88dbbc77b162ccd4a40a56ff5c71d
Food stage                           = NOT_RELEASED_BY_STAGE1
F4                                   = BLOCKED, never evaluated
F5                                   = attempted construction FAILED, not retried
M5 / E9                              = historical record cleanup, tracked separately
```

No food data has been read at any point in this project.
