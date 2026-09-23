# DrosoSense-RC v2 — architecture pre-registration

**Status: SIGNED 2026-09-23.** D1–D7 are decided (§8). Every acceptance criterion
below is a property of the **construction**, measured validation-only, before any
formal experiment. The numbers were fixed before any v2 code exists, so none of
them can have been chosen to fit a result.

**Why v2 exists.** M4-Audit established that M4-v1.4 did not test the intended
hypothesis — not that the hypothesis failed. It is retained as an **invalid test of
the intended biological hypothesis due to construct-validity failure**; v1.5,
v1.5.1 and v1.5.2 repaired evidence identity and gate semantics only, never the
biological architecture. [`docs/audit_index.md`](audit_index.md) is the pinned
closure point; every "v1 measured" value quoted below was verified
programmatically against the evidence JSON in `results/audit/m4_audit/`.

**The three disciplines this document is bound by.**

1. Criteria are properties of the **construction**, never of a model's score. A
   criterion satisfiable by a favourable outcome would be tuning with extra steps.
2. If all criteria pass and R0 is still indistinguishable from R2, **that is the
   answer**, and it is the first *valid* negative result the project would have.
3. A criterion that fails is reported as a design limitation, never relaxed.
   Relaxing one after seeing model performance is a protocol amendment and is
   recorded as one.

---

## 1. Constraint 1 — ORN-only external input, ORN→PN front end (D1, D2)

**The defect it repairs.** `make_shared` draws a dense `W_in` of shape `(N, C)`;
every node receives every channel. A6 measured `corr(node degree, |W_in| mass) =
0.073` and a consistent relabelling of nodes leaves the dynamics **bit-identical**
(`max‖Δh‖ = 0.000e+00`). The e-nose's channels were therefore never routed through
anything olfactory, and the substrate contained **0 ORNs** at N ≤ 1000 (P1-3).

**The signed requirement.**

```
X  ->  ORN  ->  PN  ->  downstream
```

- **Only ORNs receive external sensor input.** PNs receive signal **only through
  real ORN→PN edges** of the connectome.
- A **PN-direct** variant exists as a declared **ablation**, never as the main
  model: injecting into PNs would let the readout path bypass the ORN layer and
  reintroduce the v1 construct failure with biological labels on it.
- The input map is named **"ORN-aligned sparse input mapping"**. It is **not**
  receptor-exact: no claim may be made that a particular MQ sensor corresponds to
  a particular olfactory receptor.

**Acceptance criteria (validation-only).** `Din` = number of input channels.

| id | criterion | v1 value | signed target |
|---|---|---|---|
| C1.1 | every node receiving external input has cell type ORN, and the receiving set equals the declared ORN population | no typed population | exact equality |
| C1.2 | input-map density `nnz(W_in)/(N·Din)` | 1.0 | **≤ 0.10** |
| C1.3 | **identity sensitivity**: reassigning which nodes form the input population, graph fixed, must change the states | 0 (impossible in v1) | `max‖Δh‖ > 0` strictly |
| C1.4 | population sizes present in the selected subgraph | ORN = 0 | **ORN ≥ max(20, 2·Din)**; **PN ≥ 20**; **KC ≥ max(50, 4·Din)**; **MBON + DAN + higher-order ≥ 20** |
| C1.5 | PNs receive signal only via real ORN→PN edges | n/a | no `W_in` support on PN; ≥ 1 real ORN→PN edge per PN that receives anything |
| C1.6 | the input mapping is a named, declared object on every run record | absent before v1.5 | present in `evidence_unit.components` |

C1.3 is load-bearing: it is the inverse of the v1 measurement, and it is what
"the input population is biological" has to mean operationally.

**Stated plainly, because a reviewer will ask.** These are **engineering
construction thresholds, not Drosophila cell-count ratios.** If the real candidate
pool in the selected subgraph cannot meet one, that is a **protocol amendment**,
never a silent threshold reduction.

---

## 2. Constraint 2 — grown, connected subgraph that keeps its biology (D3, D4)

**The defect it repairs.** At N=250 the induced subgraph is a **126-edge residue**
with **64.8 % isolated** nodes, largest weak component **14.8 %**, mean in-subgraph
out-degree **0.50**, and it retains **0.0009** of the selected nodes' out-edges
against a **0.0006** random-node baseline. So the "layer-stratified
degree-proportional" selection is no better than chance at keeping edges, and the
`(N,N)` square discarded 99.9 % of the wiring.

### 2.1 How the subgraph is built (D3, signed)

**Deterministic multi-source biological expansion**, not sampling:

1. Seeds = the declared **ORN** population.
2. Expansion follows the allowed layer order **ORN → PN → KC / LH →
   MBON / DAN / higher-order**.
3. The frontier is ordered by **cumulative synapse-count from already-selected
   nodes**, ties broken by **neuron id** — deterministic and reproducible.
4. After the node set is fixed, **its induced biological edges are kept**.

**Primary N = 1000.** It is a **registered engineering operating point, not a
biological constant**; the write-up must say so.

**Size study deferred.** v2's first round does **not** re-run
250/500/1000/2000/4000. After the phase gate passes, **500 / 1000 / 2000** are run
to answer *scaling only* — never to select the main model. Registering N=1000 now
is specifically to prevent choosing an N from results.

### 2.2 Quality criteria (D4, signed) — and the denominator correction

**The denominator is the eligible biological edge set induced by the selected node
set — never the whole-brain edge count.** With a whole-brain denominator the
number is dominated by subgraph size: v1's 0.0009 says almost nothing about
whether biology was kept, because the denominator grows with the full graph. The
signed definition:

```
eligible-edge retention  =  |E_kept ∩ E_induced_eligible| / |E_induced_eligible|
```

where `E_induced_eligible` is the eligible (declared layer/pathway rule) directed
edge set **among the selected nodes**, and `E_kept` is what the reservoir matrix
actually carries.

| id | criterion | v1 value at N=250 | signed target |
|---|---|---|---|
| C2.1 | input population present and typed | ORN = 0 | see C1.4 |
| C2.2 | **eligible-edge retention** (denominator = induced eligible edges) | not computable in v1 (v1 reported a whole-graph share, 0.0009) | **≥ 0.90** |
| C2.3 | largest weak component / N | 0.148 | **≥ 0.90** |
| C2.4 | isolated fraction | 0.648 | **≤ 0.02** |
| C2.5 | mean in-subgraph unweighted out-degree | 0.50 | **≥ 2.0** |
| C2.6 | deterministic, declared selection; `target_n`, seed and sha256 on the record | unrecorded | present |

**Reported alongside the gate, never substituted for it** (both were required):

- `Retention` as defined above, and
- `Enrichment = Retention_bio / Retention_random` for a random node set of the
  same size (v1's measured pair was 0.0009 vs 0.0006, i.e. ≈1.4× — which the
  corrected denominator makes interpretable),
- the subgraph's **induced share of the connectome's eligible edges**
  (`|E_induced_eligible| / |E_full|`), which grows with N and is why it is a
  reported diagnostic rather than a gate.

---

## 3. Constraint 3 — the recurrence must participate (D5)

**The defect it repairs.** With v1's pinned regime (`leak` 0.1, `gain` 0.5,
`input_scale` 0.1 — `PINNED_KNOBS`, "the first conservative candidate of each
list", i.e. chosen with no reference to the dynamics) the recurrent term supplied
**3 %** of the drive: `‖gain·A·h‖` = 0.054 against `‖W_in·x + b‖` = 2.151. Driving
with `A := 0` reproduced the state's memory **bit-identically**
(`max|corr(h_t, x_{t−1})| = 0.8396` both ways), so none of the state's memory came
from the graph, and the state's effective rank tracked the **channel count**, not N.

**The signed statistic and band.** The primary statistic is the **median over
validation windows** of the **gain-free** ratio:

```
R_t  =  || A h_{t-1} ||₂  /  || W_in x_t ||₂          [PINNED — the gate]
```

The band is `[0.20, 1.00]`, and it is a **validation-search domain, not a claimed
"biologically correct range"** — there is no evidence for a narrow optimum, and a
narrow band would disguise engineering tuning as a biological criterion.

**Definitional disclosure, because it changes the number.** v1's 0.026 was measured
in a *different* form: `‖gain·A·h‖ / ‖W_in·x + b‖`, which includes `gain` and the
bias. Both forms must be reported on every v2 reservoir result — the gate is on the
pinned gain-free form, and the v1-comparable gain-inclusive form is quoted beside
it so the two eras cannot be confused.

**Three conditions, all required.**

| id | criterion | v1 value | signed target |
|---|---|---|---|
| C3.1 | `median(R_t)` over validation windows, gain-free form | 0.026 (gain-inclusive) | **0.20 ≤ median(R_t) ≤ 1.00** |
| C3.2 | **memory benefit from the recurrence**: the zero-recurrent control's memory metric must be worse by a declared margin | bit-identical (0 % benefit) | **≥ 20 % drop** |
| C3.3 | effective rank `D_eff` of the state matrix | ≈ `Din` | **≥ 1.5 × `Din`** |
| C3.4 | the knobs (`gain`, `input_scale`, `leak`, spectral scaling) are selected **on validation** within the protocol grid, and the **selection trace is recorded** | pinned, unselected | present on every record |
| C3.5 | both `R_t` forms reported next to every reservoir result | never reported | mandatory |

**The metric in C3.2 is fixed here, before measurement:** memory metric `M` =
`max_k |corr(h_t, x_{t−k})|` maximised over `k ∈ {1, 4, 8, 16}` on the validation
windows, computed identically for the real graph and for `A := 0`; the criterion is
`(M_recurrent − M_{A=0}) / M_recurrent ≥ 0.20`. Substituting a different metric
after seeing results is a protocol amendment.

**The band is a phase gate, not a target to negotiate.** If the construct phase
lands at `median(R_t) = 0.18`, the phase gate has **failed**: stop, adjust `gain` /
`input_scale` **on validation**, record the selection trace, and do not proceed to
formal experiments because it is "close". The same applies to C3.2 and C3.3 — the
three are jointly necessary.

---

## 4. Constraint 4 — R2 as a true wiring-only counterfactual (D6)

**The defect it repairs.** `make_degree_rewired` rebuilds the matrix with
`np.ones(n_edges)` and rescales, so **R2 is a uniform-weight graph**: it destroys
the synapse-count weight structure *as well as* the wiring. R0-vs-R2 was therefore
a joint contrast, and the v1 conclusion "the wiring does not matter" is not what it
measured.

**The signed counterfactual.** Preserve all of:

```
same nodes                     same input-population membership (exactly)
same directed degree sequence  same global weight multiset
same per-source outgoing weight multiset / out-strength
```

and change only the wiring. **Exact per-node weighted in-strength preservation is
not required** — demanding it both constrains the swap space until mixing fails and
risks another 71-day build — but the weighted in-strength **distribution** must
stay inside a pre-registered tolerance.

| id | criterion | v1 value | signed target |
|---|---|---|---|
| C4.1 | directed in/out **unweighted** degree sequences preserved | True | exact (must not regress) |
| C4.2 | global weight multiset preserved | **False** (R2 uniform) | exact |
| C4.3 | per-**source** outgoing weight multiset / out-strength preserved | False | exact |
| C4.4 | weighted in-strength distribution deviation | not applicable | **median relative error ≤ 5 %** |
| C4.5 | input-population node set identical between R0 and R2 | n/a (no population) | exact equality |
| C4.6 | **mixing reached**: `|E_R0 ∩ E_R2| / |E_R0|` | 0.058 at N=1000 (fully-mixed expectation 0.055) | **≤ 0.20** |
| C4.7 | build wall time for one graph at N=1000 | projection 71 days at M = 14.8 M | **≤ 10 min**, hard cap 1 h |
| C4.8 | reported with every R2 result: swap count, original-edge retention, mixing figure | not reported | mandatory |

C4.6 exists because a degree-preserving swap chain that has not mixed can be
"degree-preserving" and still be almost R0; reporting the overlap is what makes
"rewired" checkable rather than asserted.

---

## 5. D7 — the cluster unit, aligned to the frozen text (SIGNED: specimen)

**The terminology that was contradictory in the draft, resolved by the frozen
protocol's own words.** `configs/protocol_v1.3.yaml`, `pairing`, verbatim:

> "The bootstrap resamples FOLD IDENTIFIERS — the specimen-disjoint test blocks
> whose metrics actually enter the difference. It never resamples seeds: **ten seeds
> on one specimen partition** are ten model initialisations, not ten draws of
> specimens... **For LOSO a fold is a single specimen, so the fold bootstrap IS a
> specimen bootstrap.**"

Three states, which the draft had collapsed into one phrase:

| | cluster unit | basis |
|---|---|---|
| **Frozen declaration** | the fold — **which is a specimen**, on the stated premise that the ten seeds share **one specimen partition** | `pairing.resample_unit_detail`, `n_clusters_formula: n_folds` |
| **Implementation, measured** | the fold **index** — and the premise fails, because a seeded LOSO permutation gives each seed a *different* partition, so all **62 of 62** D3 `fold_id`s hold a different specimen under each of the 10 seeds | A9, from the committed per-run records |
| **A9 amendment (D7, signed)** | the **specimen**, declared explicitly so the statistics no longer depend on that premise holding | D7 |

So the frozen text does not say "cluster by `fold_id`"; it says "cluster by the
fold, **because** a fold is a specimen". The implementation broke the *premise*,
which is a sharper statement of the defect than "the label is wrong", and it means
clustering by **specimen** is the faithful repair under either split design.

**Signed decision (D7 = YES).** Cluster unit = **specimen**, frozen as a statistics
amendment **before any formal v2 evaluation**. Construct-phase development may
proceed in parallel; **formal inference must not run first**, because an estimator
built on the wrong independent unit misstates every CI and p-value even if the
model is right. Expect wider CIs and less significance — that is the correct cost,
not a reason to defer.

**Required regression test**, of the form the owner asked for:

```
same specimen appearing in several folds/windows  ->  contributes exactly ONE cluster
n_clusters == number of distinct specimens tested
the cluster mean averages that specimen's seeds, not one observation per fold index
```

**Also to be reported, once the unit is fixed:** `n_clusters`, `n_clusters_nonzero`
and `minimum_achievable_p_over_clusters` on every line, and the D2 floor
(`2/2⁵ = 0.0625 > α`) restated so a D2 cluster-level claim is never read as
reachable.

---

## 6. Invariants inherited from v1.x — must not regress

- **Evidence-unit identity schema 2** (v1.5): one unit = `(dataset, task, fold,
  model, window, condition, reservoir_size, normalization, input_mapping,
  topology_variant, rewire_seed)`; the schema-1 fingerprint is a **fallback alias
  for pre-v1.5 records only**, so two substrates of one unit never collide.
- **Unit-addressable record paths** — a second substrate of one unit must not
  overwrite the first.
- **Parameter evidence scope** (v1.5.1/v1.5.2): matched rows inside a declared
  scope, **per dataset**, folded with AND, fail closed; no averaging, extrema
  selection, cross-dataset fallback or unscoped search.
- **§17**: only `status == "ok"` occupies the touch quota; skips carry both config
  hashes; no test split is re-scored.
- **Metrics**: fixed-label macro-F1 with `zero_division=0`; AUROC only when all four
  classes are present, with `n_auroc_defined` reported; D3's macro-F1 is **not**
  magnitude-comparable with D2's.
- **No test split is touched during v2 development.** Every criterion in §1–§5 is
  validation-only and says so in its own output.

---

## 7. Order, and the gate between phases

```
1. construct checks   C1.*, C2.*, C3.*, C4.* — validation-only, machine-checked report
        |
        v   (only if EVERY criterion passes; "close" is a failure)
2. formal experiments  with the specimen cluster unit (D7) frozen first
        |
        v
3. gates A/B/C, narrative rules, and whatever the result is
```

The gate between 1 and 2 is the whole point: a v2 experiment whose construct checks
failed would repeat v1's mistake at greater cost, with the failure
indistinguishable from a hypothesis failure.

**What v2 explicitly does not do.** It does not re-read, re-index or re-score any
v1.x result; it does not alter a frozen protocol file other than by adding a new
version; it does not touch a test split during development; it does not select N
from results; and it does not treat "R0 > R2" as its success condition.

---

## 8. Sign-off — SIGNED 2026-09-23

```
D1 = ORN-only external input; ORN->PN included in v2; PN-direct is an ablation only
D2 = density <= 0.10; ORN >= max(20, 2*Din); PN >= 20;
     KC >= max(50, 4*Din); MBON+DAN+higher-order >= 20
D3 = deterministic biological expansion; primary N = 1000;
     size study deferred to 500/1000/2000 after the phase gate
D4 = eligible-edge retention >= 0.90 (denominator: induced eligible edges of the
     selected node set, NOT the whole-brain edge count);
     WCC >= 0.90N; isolated <= 0.02; mean in-subgraph out-degree >= 2.0
D5 = median Rt in [0.20, 1.00] (gain-free form, pinned in §3);
     zero-recurrence memory benefit >= 20%; effective rank >= 1.5 * Din
D6 = preserve directed degree + global weight multiset +
     per-source outgoing weight multiset; in-strength median relative error <= 5%;
     edge overlap <= 0.20; build <= 10 min at N = 1000
D7 = YES -- specimen cluster unit, frozen before any formal v2 evaluation
```

**The three that decide what v2 is actually testing** (owner's ranking): **D1**,
**D3** and **D6**. Together they determine whether v2 tests real connectome wiring
with a biological input boundary, or merely re-runs experiments on a
biologically-labelled random reservoir.

**This set is deliberately conservative.** It does not make R0 beat R2 more likely;
it makes the construct phase *easier to fail*. That is the point. If C1–C4 all pass
and `R0 ≈ R2`, that result is finally entitled to be read as *"in a validly
implemented Drosophila olfactory reservoir, the real wiring conferred no detectable
advantage"*. And if R0 does show an advantage, the TAFE argument is far stronger
than v1's, because the input population, subgraph connectivity, recurrence
efficacy, R2 fairness and statistical unit have all been closed in advance.
