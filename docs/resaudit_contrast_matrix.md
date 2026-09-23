# ResAudit-Food — family contrast matrix (design gate)

**Status: GENERATORS IMPLEMENTED for F2/F3; F5 BLOCKED by a measured scale limit; F4 STILL
BLOCKED. No family instance has been produced at F1's real scale and no food data has been
read.** See Section 7 for the generation-round findings, including one real bug fixed in the
shared measurement layer.

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


---

## 7. Generation round: what the generators found

F2, F3 and F5 were released for generation. They are implemented in `resaudit/families.py`
with tests in `tests/test_resaudit_families.py`. Four findings, one of which is a bug in the
shared measurement layer that would have corrupted every family.

### 7.1 A real bug: the spectral-radius estimator was wrong on large graphs (FIXED)

`spectral_radius_of` used single-vector power iteration for `n > 500`, which assumes a
**simple** dominant eigenvalue. A graph that is a union of disjoint cycles has *degenerate*
dominant eigenvalues — all of them are 1 — so the iterate orbits between them forever and
`v @ (A @ v)` returns wherever in the orbit it happened to stop. Measured on 8 disjoint
directed cycles (true `rho = 1.0`, n = 1000):

| seed | reported rho |
|---|---|
| 0 | 0.00629 |
| 1 | **−0.01921** |
| 2 | **−0.02184** |
| 42 | 0.08438 |

It returned **negative radii**. This is not cosmetic: `scale_to_spectral_radius` divides by
this number, so every family built from a block-structured substrate was silently
mis-scaled — and A3's verdict is normalisation-dependent (amendment 1 Section 1). Fixed by
iterating a random 8-dimensional subspace, which accumulates every block's period and
returns 1.00000000 for every seed. A regression test pins the case.

### 7.2 The weight-multiset and `rho` invariants are mutually exclusive for F2/F3 (resolved)

Matching `rho = 0.95` exactly and preserving F1's weight multiset exactly cannot both hold in
general: normalisation multiplies every weight by a graph-dependent factor. The frozen
project had already recorded this incompatibility for R2. Faced with the choice, the **weight
multiset wins** for F2 and F3, because that is what their contrast is *defined* by. Both
families therefore share F1's single weight scale, and the resulting `rho` drift is reported
per family (`rho_gap_vs_target`) instead of hidden. Measured: F2 drifts 0.64 %.

**F5 is deliberately exempt**: it normalises to `rho = 0.95` with its own scale. It claims no
weight-multiset identity, and under F1's much smaller scale a block-union substrate has a
tiny spectral radius whose power norms decay away before the Krylov horizon (measured:
`D_eff ≈ 80` at `rho = 0.95` versus `D_eff ≈ 6.5` at F1's 0.128 scale).

### 7.3 F5 is BLOCKED at F1's budget — a measured scale limit, not a tuning problem

F5's declared mechanism is coprime-period blocks. It works at low degree and **does not
survive F1's edge budget**:

| configuration | mean degree | `D_eff` | gate | A3 |
|---|---:|---:|---:|---|
| coprime blocks, own `rho`=0.95 | ~1 | **80.0** | 10 | PASS |
| same, chords filling F1's budget | **80.4** | **7.9** | 10 | **FAIL** |

The blocks supply only ≈`N` cycle edges, so reaching `m = 80,443` requires ≈79k in-block
chords, which raise the raw spectral radius to **82**; normalising that back to 0.95 then
scales every weight down by ~1/86, so the power norms decay by ~14 orders of magnitude
across the horizon and the Krylov block collapses.

Two things must NOT be done about this: the edge budget must not be quietly dropped, and the
`rho` convention must not be bent to make the construction pass. Either the construction
changes to something that survives dense budgets, or F5 is reported as blocked. That
decision is not this round's.

### 7.4 F4's feasibility gate exists, and its recurrence term is vacuous (blocks F4)

`resaudit/f4_feasibility.py` implements the six conditions as a machine-checkable predicate.
It correctly catches the **label-only trap**: permuting cell types leaves the type-pair matrix
bit-identical, so `type_pair_organization_changed` is False and the candidate is rejected. A
genuine degree-preserving rewire moves type-pair organization while holding degrees, weights
and input geometry.

But the gate's fifth condition cannot fail. `_simple_cycle_lengths` reports which cycle
lengths are *present* via `trace(A^k) > 0`; on a dense directed graph every length 2..12 is
present, and a degree-preserving rewire keeps every one present. The term is therefore
constant across candidates:

| candidate | cycle lengths present | recurrence preserved |
|---|---|---|
| F1 | 2..12 | — |
| rewire, seed 1 | 2..12 | trivially |
| rewire, seed 2 | 2..12 | trivially |
| rewire, seed 3 | 2..12 | trivially |

**A gate term that cannot fail is not a gate term.** So F4 is *not* released on the strength
of a `feasible = True` verdict that rests on a vacuous check. Before F4 can be built, the
recurrence-preservation term must be replaced by a discriminating one (exact cycle *counts*,
or a motif profile). Until then F4 stays blocked even though the gate can return True — and
that is recorded rather than exploited.

### 7.5 F1 cannot be materialized here

F1 is the frozen S0 connectome substrate (N=1000, M=80443, S1=7.129) plus its typed-aligned
input mapping. Both live **outside this repository**: the raw connectome and induced
adjacency are gitignored large binaries under the external data root, and Git records only
their content hashes (`A_hash 3aa95745…`, `B_hash 3bd78eac…`). No machine without that root
can build F1, so the generators were verified against a 60-node **stand-in**, and a Stage 1
report may not substitute a stand-in without saying so. This is enforced as a module
constant (`F1_MATERIALIZATION_REQUIREMENT`) and asserted by a test.
