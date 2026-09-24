# ResAudit-Food Stage 1 — findings

```text
scientific_status = FROZEN_NEGATIVE_RESULT
```

Both governing conditions are now met: (1) the upstream V3 selection was recomputed under the
corrected estimator and still selects nothing, so no alternative F1 was concealed by the
defect (`docs/resaudit_v3_reselection.md`); and (2) the full test suite runs clean
(871 passed, 8 skipped). The earlier `PROVISIONAL_PENDING_UPSTREAM_SELECTION_RECOMPUTATION`
status is superseded.

**Scope of the negative, stated narrowly.** It holds for the biological reference, the matched
random controls, the topology-destroyed control and the attempted positive-control
construction, at both preregistered input widths, under the frozen normalisation, input
geometry and A3 definition. It does **not** license the claim that a larger or differently
connected substrate of another kind could not clear the gate: `N`, connectivity scale and
input-coverage fraction were not varied, and any of them could move `D_eff/Din`. What is
established is that **within the preregistered input-width range, increasing `Din` does not
close the deficit; it behaves approximately multiplicatively rather than additively.**

Naming correction: F5 is reported as the **attempted** A3-qualified construction / the
positive-control construction **attempt**. It measured `D_eff/Din = 1.556` and `1.544`
(≈0.78 of the gate), so it is not A3-qualified and must not be labelled as if it were.

**Final decision: rule 6.4 invoked; Stage 1 closed. See
[`docs/resaudit_stage1_decision_rule_6_4.md`](resaudit_stage1_decision_rule_6_4.md).**

**Result: at both preregistered widths, no family clears the primary gate. Zero families are
`food_eligible`, so the food stage cannot start under decision rule 6.1.**

Measured at HEAD `208475ed` on F1 materialized to `VERIFIED_FOR_STAGE1`, with the audit run at the
battery's declared operating radius `rho = 0.95`. No food data was read, no label was seen, and no
task metric exists anywhere in this stage.

## 1. The matrix

| family | Din | N | M | A1 | A2 | A3 `D_eff` | gate `2·Din` | A3 | A4 | A5 | `food_eligible` | `construct_qualified` |
|---|---:|---:|---:|---|---|---:|---:|---|---|---|---|---|
| F1 connectome | 6 | 1000 | 80443 | PASS | n/a | 7.822 | 12 | **FAIL** | — | — | False | False |
| F2 ER / degree-preserving | 6 | 1000 | 80443 | PASS | FAIL | 7.884 | 12 | **FAIL** | — | — | False | False |
| F3 topology-destroyed | 6 | 1000 | 80443 | PASS | FAIL | 10.256 | 12 | **FAIL** | — | — | False | False |
| F5 attempted A3-qualified construction | 6 | 1000 | 80443 | **FAIL** | FAIL | 9.334 | 12 | **FAIL** | — | — | False | False |
| F1 connectome | 8 | 1000 | 80443 | PASS | n/a | 10.514 | 16 | **FAIL** | — | — | False | False |
| F2 ER / degree-preserving | 8 | 1000 | 80443 | PASS | FAIL | 10.297 | 16 | **FAIL** | — | — | False | False |
| F3 topology-destroyed | 8 | 1000 | 80443 | PASS | FAIL | 13.628 | 16 | **FAIL** | — | — | False | False |
| F5 attempted A3-qualified construction | 8 | 1000 | 80443 | **FAIL** | FAIL | 12.349 | 16 | **FAIL** | — | — | False | False |
| **F4** | — | — | — | — | — | — | — | **BLOCKED / NOT EVALUATED** | — | — | — | — |

`—` means **not evaluated**, because A3 (the primary gate) did not pass: the engine evaluates A4/A5
only downstream of A3, so the A4 and A5 columns are empty by construction rather than missing. F4
keeps an explicit blocked row because the pre-registration contained F4 and a silent omission would
misrepresent the design.

## 2. The structural finding: `D_eff / Din` is a width-invariant family property

The single most informative number in this run is not `D_eff` but its ratio to `Din`, because the
gate also scales with `Din`:

| family | `D_eff/Din` at Din=6 | at Din=8 | spread | fraction of the gate (2.0) |
|---|---:|---:|---:|---:|
| **F3** topology-destroyed | **1.709** | 1.703 | 0.3 % | 0.85 |
| **F5** A3-qualified | 1.556 | 1.544 | 0.8 % | 0.78 |
| **F2** degree-preserving | 1.314 | 1.287 | 2.0 % | 0.66 |
| **F1** connectome | 1.304 | 1.314 | 0.8 % | 0.65 |

Two consequences follow, and both are stronger than "the threshold is too strict":

1. **No choice of `Din` can close the gap.** The ratio is stable to within 2 % across an 8:6 change
   in width, so `D_eff` grows *proportionally* with `Din` while the gate also grows proportionally.
   The two lines are parallel and never cross; the shortfall is a **constant multiplicative factor**
   (0.65–0.85 of the gate), not a scale that could be tuned away by picking a different width or a
   larger substrate of the same kind. Lowering the gate to `1.7·Din` would admit only F3; `1.3·Din`
   would admit F1. The pre-registration forbids such a change being made silently, so it is recorded
   as a finding and not applied.

2. **The ordering is inverted relative to the design's premise.** Ranked by `D_eff/Din`:

   ```
   F3 (topology-destroyed random)  >  F5 (built to satisfy A3)  >  F2 ≈ F1 (connectome)
   ```

   The **topology-destroyed random control has the highest controllability ratio**, above both the
   biological substrate and the family whose only purpose is to be A3-qualified. And F2 ≈ F1
   (1.314 vs 1.304) means *preserving* degree structure while randomising wiring changes nothing,
   whereas *destroying* topology outright raises the measure. On this evidence A3 does not separate
   substrates in the direction the audit was introduced to support; if it separates them at all, it
   favours the random control.

## 3. F5 fails the gate it was constructed to pass, and fails A1 as well

F5 is a **construction**, not a competitor: it is built to maximise A3 by composing directed cycles
of pairwise-coprime length. It nonetheless scores 9.334 against a gate of 12, below F3's random
graph, **and it fails A1 (structure present)**. The generator already carried a warning about this
scale limit — the coprime mechanism was validated on small stand-ins, and at F1's real edge budget
(mean degree ≈ 80) it does not survive. This run confirms the warning at full scale.

This is the sharpest available statement of the F5 circularity already recorded in amendment 1 §4:
a family constructed to satisfy a criterion, measured against that criterion, fails it — so
"F5 passes A3" could never have been a result, and here there is not even a pass to report.

## 4. A2 fails for both controls — for two different reasons, one of them structural

A2 (counterfactual fair) requires the degree sequence, the global weight multiset and the per-source
weight multisets to match the declared parent **exactly**, plus `edge_overlap <= 0.20`. F1 is
`not_applicable` (it has no parent). **F2 and F3 both FAIL A2**, but the diagnostics show these are
not the same kind of failure:

```
F2  degree_exact = True    global_weights_exact = False   per_source_weights_exact = False
    edge_overlap = 0.203709  (ceiling 0.20)      median_in_strength_err = 5.8e-16
F3  degree_exact = False   global_weights_exact = False   per_source_weights_exact = False
    edge_overlap = 0.080827  (ceiling 0.20)      median_in_strength_err = 4.60
```

**F2 is a near miss on overlap and a real difference in edge weights.** It clears the degree
sequence exactly and preserves node *strengths* to machine precision (`5.8e-16`), but misses the
overlap ceiling by 0.0037 — 1.9 % of the threshold — and does not reproduce the weight multisets
exactly. The likely mechanism is aggregation: at mean degree ≈ 80 the graph is dense enough that a
804,430-accepted-swap rewire routinely collides on existing pairs, and a collision merges two edge
weights, changing the multiset while leaving total strength intact. That is a property of the
accepted-swap budget on a dense multigraph, and it points at the swap budget as the lever — but the
budget is frozen at `10|E|` and is explicitly "not adjusted for any reason after generation", so
per the design this FAIL is reported and not repaired.

**F3's failure is structural, not a measurement outcome.** A2 demands that the degree sequence match
the parent exactly, while F3's declared contrast *deliberately does not retain the degree
distribution* — that is what makes it a topology-destroyed control rather than a degree-matched one.
F3 declares `counterfactual_parent='F1'`, so A2 is applicable and must fail. **F3 can therefore never
be A2-admissible under the current definitions**: the admissibility condition and the contrast's own
definition are mutually inconsistent. This is an inconsistency between two frozen declarations, not
something a rerun or a budget change can fix, and it needs a decision rather than a repair.

Either way the consequence for interpretation is the same: until A2's status is resolved for the
controls, **no F1-versus-control difference in this design is admissible**, even if A3 later passes.

## 5. A defect I introduced and corrected, recorded in full

The first Stage 1 run at this HEAD produced **different F1 numbers** (8.119 / 10.698) and I did not
treat them as final. The cause was mine, in the runner:

- `audit_family` **does not normalise** its input; it calls `audit_a3(spec)` on `spec.A` as given.
- I passed `F1_A.npz`, which the materializer had written with the **legacy v3 scoring
  normalisation** targeting `rho_target = 0.9` on the *symmetrised* matrix. The resulting matrix sat
  at `rho = 0.8536`, not the battery's declared `FROZEN_RHO_TARGET = 0.95`.
- A3's verdict is normalisation-dependent — documented in amendment 1 §1 and pinned by
  `test_a3_verdict_depends_on_the_spectral_radius_normalization` — so those numbers were measured at
  the wrong scale and were invalid.

Fix, in two parts: the materializer now also writes **`F1_A_raw.npz`**, the raw induced
synapse-count adjacency, which is the substrate's `A` and what the edge-list hash identifies; and
the runner loads that raw matrix and normalises it once with the battery's own declared-seed
`scale_to_spectral_radius(A, 0.95)`, asserting `rho == 0.95` before auditing. `F1_A.npz` is retained
as a legacy scoring copy and the runner refuses to run without the raw file.

A useful consistency check fell out of this: **F2/F3/F5's values are byte-identical between the bad
run and the corrected run**, because those generators normalise themselves from whatever F1 they are
handed. Only F1 was affected. The generators were robust; the runner was not.

## 6. What is concluded, and what is explicitly not

**Concluded.** At N=1000, M=80443, `rho=0.95`, with the frozen S0 substrate and the frozen typed-
aligned input rule, no family in {F1, F2, F3, F5} reaches `D_eff >= 2·Din` at `Din ∈ {6, 8}`;
`D_eff/Din` is a stable family property (1.30–1.71) below the required 2.0; the ranking puts the
topology-destroyed random control first; F5's construction fails at this scale and also fails A1; and
A2 fails for both controls. Decision rule 6.1 therefore blocks the food stage, and decision rule 4's
outcome applies: the thresholds are **not** lowered, and the honest consequence is a study of what a
lightweight reservoir at this scale cannot be.

**Not concluded.** Nothing here is evidence about **food-task performance**. A3-FAIL families are
never run on food data in this design, so "A3 FAIL predicts poor accuracy" remains deliberately
untestable here and is **not** being claimed. The food datasets (FD1/FD2/FD3) are untouched: their
pre-registration stands, and no result in this document bears on them except that they cannot yet be
used, because no family is admissible to run on them.

## 7. What the next move has to be

Three options, none of which may be taken silently:

1. **Report the negative result.** Decision rule 4's declared outcome: a paper about what a
   lightweight reservoir cannot be at this scale, with the `D_eff/Din` invariance as the mechanism
   and the ranking inversion as the reason the audit's premise fails.
2. **Change the criterion through a new amendment.** Any relaxation must be justified by a stated
   principle rather than by the desire to admit a family, and must be applied to every family
   equally. The invariance in §2 is the fact that makes this a *criterion* question and not a
   parameter question.
3. **Change the substrate scale or budget.** Worth stating explicitly: because the ratio is
   width-invariant, a larger `N` of the *same kind* of construction is not predicted to help; what
   would have to change is the construction family or the input geometry.

## Artifacts

```
results/audit/resaudit_stage1/STAGE1_AUDIT_MATRIX.csv    the matrix above, machine-readable
results/audit/resaudit_stage1/FAMILY_PROVENANCE.json     per-cell spec, generation record, verdicts
results/audit/resaudit_stage1/A3_KRYLOV.json             A3 values, thresholds, expressions
results/audit/resaudit_stage1/A4_MEMORY.json             A4 (descriptive; empty here, A3 gated it)
results/audit/resaudit_stage1/A5_STATE_EXPANSION.json    A5 (empty here, A3 gated it)
results/audit/resaudit_stage1/STAGE1_VERDICT.md          generated verdict + carried caveats
results/audit/resaudit_stage1/f1/F1_materialization.json F1 provenance, VERIFIED_FOR_STAGE1
results/audit/resaudit_stage1/f1/F1_A_raw.npz            the raw substrate A (identity)
```
