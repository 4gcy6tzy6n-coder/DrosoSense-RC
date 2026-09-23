# ResAudit-Food — family contrast matrix (design gate)

**Status: DESIGN ONLY. No family instance exists and no food data has been read.**

```
Stage 1 runner               = IMPLEMENTED / VERIFIED
A3 semantics                 = NARROWED
A4                           = RETIRED AS GATE / DESCRIPTIVE ONLY
Amendment chain              = COMPLETE THROUGH AMENDMENT 2
Family-battery documentation = READY (this file)
F1-F5                        = RELEASED FOR DESIGN ONLY
Food evaluation              = HOLD
```

The original pre-registration is immutable (`docs/resaudit_food_preregistration.md`, blob
`350edc2b…`); findings live in amendments 1 and 2 and in this file.

**This matrix is a gate, not a description.** It is implemented in `resaudit/contrast.py`
and enforced by `tests/test_resaudit_contrast_matrix.py`. A declaration cannot be
constructed without a single varied factor, at least one held-fixed quantity, an
identifiable claim phrased as an association, and a non-empty list of forbidden stronger
claims. The matrix-level validator rejects duplicate varied factors, a missing contrast
parent, a reference that varies something, and redundant F2/F3 controls. The validator's
current output:

```
ready   = [F1, F2, F3, F5]
blocked = [F4]              <- varied factor NOT YET NAMED
instances_generated = 0
food_evaluation = HOLD
```

---

## 1. The matrix

| Family | Role | Contrast parent | Primary varied factor | Held fixed | Identifiable claim | Forbidden stronger claims | Status |
|---|---|---|---|---|---|---|---|
| **F1** | biological / reference | — (reference) | none (reference; varies nothing by construction) | N, edge count, mean degree, weight multiset, input geometry, ρ, leak/gain, probe protocol | Under matched measurement conventions, F1 supplies the reference substrate against which F2–F5 are contrasted. | F1 is a baseline to be beaten; F1's value is a performance benchmark | ready |
| **F2** | statistics-matched random control | F1 | wiring arrangement, with first-order statistics retained exactly | N, **edge count, mean degree, degree sequence**, weight multiset, input geometry, ρ, leak/gain, probe protocol | Under matched N, edge count, degree sequence, weight multiset, input geometry and spectral radius, any difference from F1 is associated with the specific wiring arrangement rather than with first-order structure. | biological topology causes better performance; biology is computationally superior; the connectome is the better reservoir | ready |
| **F3** | topology-destroyed control | F1 | higher-order structure (degree distribution, clustering, cycle profile), deliberately destroyed | N, **mean degree**, weight multiset, input geometry, ρ, leak/gain, probe protocol | Under matched N, mean degree, weight multiset, input geometry and spectral radius, any difference from F1 is associated with higher-order structure, since first-order degree statistics are deliberately NOT retained. | density causes the difference; more edges cause better performance; biological topology causes better performance; biology is computationally superior; the connectome is the better reservoir | ready |
| **F4** | recurrence-preserving structural counterfactual | F1 | **NOT YET NAMED** | N, edge count, degree sequence, mean degree, weight multiset, input geometry, ρ, scc profile, cycle-length profile, recurrence statistics, leak/gain, probe protocol | Under matched N, edge count, degree sequence, density, weight multiset, input geometry, spectral radius and recurrence statistics, any difference from F1 is associated with the named high-order organization factor. | biological topology causes better performance; biology is computationally superior; the connectome is the better reservoir; the named factor is the biological factor; recurrence statistics explain the difference | **BLOCKED** |
| **F5** | deliberately A3-qualified construction | F1 | finite-horizon input-reachable state diversity, guaranteed by construction | N, edge budget, mean degree, weight multiset, input geometry, ρ, leak/gain, probe protocol, readout | Under matched N, edge budget, input geometry, weight multiset, probe protocol and spectral radius, F5 is an A3-qualified construction within the same budget, so it shows what an A3-passing substrate looks like at this scale. | A3 causes better food performance; A3 PASS predicts food accuracy; guaranteed reachable diversity improves the task; biological topology causes better performance; biology is computationally superior; the connectome is the better reservoir | ready |

`ρ` in the held-fixed column is the frozen `rho_target = 0.95` from amendment 1 §0.1,
applied by a single scalar multiplication before any measurement.

## 2. The F1 → F2 → F3 gradient

The point of three families rather than two random baselines is a **structure-retention
ladder**, and the validator checks it explicitly — F3's structural-statistic set must be a
strict subset of F2's:

| family | structural statistics retained |
|---|---|
| **F1** | reference: everything as measured |
| **F2** | `degree_sequence`, `edge_count`, `mean_degree`, `weight_multiset` |
| **F3** | `mean_degree`, `weight_multiset` |

`F2 ⊃ F3` strictly. So a monotone trend F1 → F2 → F3 with respect to some measured
quantity is interpretable as *progressive loss of structure*, which no pair of random
graphs could show.

**One vocabulary decision, recorded because the gate forced it.** The first draft of this
matrix had F2 hold `edge_count` and F3 hold `density`. The validator rejected it: `density`
is not a subset of `degree_sequence`/`edge_count`, so the hierarchy was not expressible and
the two families could not be shown to be graded rather than merely different. Since
`density = mean_degree / N`, the same quantity was being named twice. F2 and F3 now both
declare `mean_degree`, and the real structural difference — the full degree sequence versus
its first moment — is what the subset check sees. A vocabulary difference was masquerading
as a structural one, and the gate caught it before any family was built.

## 3. F4 is blocked, and exactly why

The instruction was explicit: *do not generate F4 until its single varied structural factor
is explicitly named.* "A recurrence-preserving structural counterfactual" is a role, not a
factor, and building to it would change several things at once.

**Candidates, with what each would require:**

| candidate varied factor | preserved by | why the "before/after" is measurable |
|---|---|---|
| **cell-type connectivity organization** *(recommended)* | permuting which nodes carry which cell type while holding the degree sequence, then re-deriving edges within type-classes | the frozen package ships a cell-type annotation, and the (source-type, target-type) edge-count matrix is directly computable — no new statistic has to be invented |
| motif organization | rewiring that holds directed-path and feed-forward-loop counts | motif counts are computable, but preserving them under rewiring is a constraint-satisfaction problem with no primitive in the frozen package |
| edge-direction arrangement | reorienting edges while holding the undirected degree sequence | changes in/out-degree split, which is not in the held-fixed list without further work |
| modular organization | reassigning edges within and between communities | only meaningful for a graph with a detectable community structure, which must first be established for F1 |
| spatial / locality organization | permuting the physical layout that edges respect | the frozen substrate has no layout coordinate to permute |

**Two structural facts recorded before the choice, not after** (they are in the
declaration's `notes`, so they constrain the decision rather than rationalize it):

1. **The row sums of the type-pair matrix ARE the out-degrees by type.** Preserving the
   full degree sequence therefore already pins most of the type-pair marginals, so the
   varied factor must be the type *assignment* arrangement, not the marginals. A
   construction that tried to vary the marginals would move the degree statistics too.
2. **If F4 holds the degree sequence fixed, it is a different degree-preserving rewire of
   F1 than F2 is.** The two must be separated by what each *preserves* — F4 preserves
   recurrence statistics (SCC profile, cycle-length profile) where F2 randomizes them —
   and by their A2 overlap budgets. Without that separation the matrix validator would
   reject them as redundant controls, which is the correct behaviour.

**A third fact that must be checked before F4 is built, not discovered afterwards:**
"recurrence-preserving" is a constraint on a rewiring, and no primitive in the frozen
package currently enforces it. `weight_preserving_degree_rewire` preserves degrees and
per-source weight multisets but says nothing about SCCs or cycle profiles. So naming the
factor is necessary but not sufficient: the construction must also be *demonstrated* to hold
the recurrence statistics it claims to hold, on F1, before F4 is generated. If no such
construction is found within a bounded effort, that is itself a reportable finding about
the framework's reach — not something to patch by loosening the held-fixed list.

## 4. What each contrast may and may not license

The `identifiable claim` and `forbidden stronger claims` columns are enforced fields, not
prose. Restating them in one place for review:

**F1 vs F2 — may say:** under matched N, edge count, degree sequence, weight multiset,
input geometry and spectral radius, an observed difference is associated with the specific
wiring arrangement.

**F1 vs F2 — may not say:** that biological topology *causes* better performance; that
biology is computationally superior; that the connectome is the better reservoir. Matching
first-order statistics makes the contrast sharper than a naive random-graph comparison, but
it does not convert an association into a mechanism.

**F1 vs F3 — may say:** under matched N, mean degree, weight multiset, input geometry and
spectral radius, a difference is associated with higher-order structure.

**F1 vs F3 — may not say:** that density or edge count *causes* the difference. F3's
held-fixed list contains `mean_degree` precisely so a density explanation is excluded by
construction rather than argued away.

**F1 vs F5 — may say:** under matched budget and conventions, F5 is an A3-qualified
construction, so it shows what an A3-passing substrate looks like at this scale.

**F1 vs F5 — may not say:** that A3 *causes* better food performance, or that A3 PASS
predicts food accuracy. **A3 PASS is a construction property of F5.** The causal question
is unidentifiable under the current gate design, because A3-FAIL families are never run on
food data (rule 6.1). This is the boundary the F5 circularity analysis drew in amendment 1
§4, and it is now a `forbidden_claims` field rather than a paragraph.

## 5. Fixed before generation, per the three binding rules

**Rule 1 — no food task may select a family parameter.** Every topology parameter,
normalisation factor, input geometry and seed protocol is fixed before any food data is
read. Enforced structurally: `FamilySpec` has no field for a task metric and refuses
unknown attributes, and the construction record rejects task-named keys.

**Rule 2 — families differ by one structural factor.** Enforced by
`ContrastDeclaration.varied_factor` being a single value and unique across the matrix, with
held-fixed sets recorded and the F2/F3 subset relation checked.

**Rule 3 — `Stage 1 passed` is never reported.** Only `food_eligible` and
`construct_qualified`, separately. A test asserts no merged pass field can appear in a
Stage 1 report.

**Rule 4 — F5's claim is bounded by its construction** (§4 above).

## 6. Not yet decided, and deliberately so

These are open, and they are the substance of the review this matrix is submitted for:

1. **F4's single varied factor** (§3). Recommended: cell-type connectivity organization.
   Blocked until named, and additionally gated on demonstrating a recurrence-preserving
   construction.
2. **F2's swap budget.** F2's knob is the number of accepted degree-preserving swaps, which
   trades wiring randomization against the A2 overlap ceiling. That trade-off must be
   declared before generating, not tuned to taste, and it is the same primitive A2 audits —
   so F2 is the one family whose construction and whose audit share a measurement.
3. **F3's generator choice.** Mean-degree-matched Erdős–Rényi is the obvious reading of
   "coarse statistics retained", but it also destroys weak connectivity structure in a way
   F1 does not have. If F3 turns out to be disconnected or to contain few cycles, that is a
   finding about what "topology-destroyed" means at this scale, and it must be reported
   rather than re-parameterised away.
4. **Whether F4 is additive at all.** If F4 cannot be constructed to hold recurrence while
   varying one factor, the honest matrix is F1–F3 plus F5, and the recurrence-preserving
   counterfactual is dropped with the reason recorded.

No family instance will be generated, and no food evaluation performed, until this matrix
passes review.
