# DrosoSense-RC v2 — architecture pre-registration (DRAFT for owner sign-off)

**Status: draft. Nothing here is in force until the owner signs off the marked
decisions.** No v2 code exists yet, and this document is written before any v2
model has been run, so none of its criteria can have been chosen to fit a result.

**Why v2 exists.** M4-Audit established that M4-v1.4 did not test the intended
hypothesis — not that the hypothesis failed. It is retained as an **invalid test
of the intended biological hypothesis due to construct-validity failure**; v1.5,
v1.5.1 and v1.5.2 repaired evidence identity and gate semantics only, never the
biological architecture. See [`docs/audit_index.md`](audit_index.md) for the
pinned closure point and the measurements behind every "v1 defect" quoted below.

**The discipline this document is written under.**

1. Every acceptance criterion below is a property of the **construction** —
   measured on validation only, before any formal experiment — never of a model's
   score. A criterion that could be satisfied by a favourable outcome would be
   tuning with extra steps.
2. If all criteria pass and R0 is still indistinguishable from R2, **that is the
   answer** and it is reportable. v2 exists to make the hypothesis testable, not
   to make it true.
3. A criterion that cannot be met is reported as a design limitation. It is not
   relaxed. Relaxing one after seeing model performance is a protocol amendment
   and must be recorded as one.

---

## 0. The four hard constraints (decided) and their numeric bands (to sign off)

The four constraints are the owner's decision. The **numeric bands and floors are
proposals in this draft** and are marked `[SIGN-OFF]`; the requirement is not the
specific number but that a number is declared *before* the construct is measured.

| # | constraint | v1 measured | v2 requirement |
|---|---|---|---|
| 1 | **ORN/PN-aligned input population** | one dense random `W_in` over every node; dynamics bit-identical under relabelling | input enters through a declared, typed biological population; `W_in` sparse and supported only there |
| 2 | **Connected olfactory subgraph with preserved biological edges** | N=250: 126 edges, 64.8 % isolated, 0.0009 of out-edges retained, **0 ORNs** | contains the input population, retains a declared share of biological in-subgraph edges, largest weak component above a declared floor |
| 3 | **Recurrent/input ratio as an explicit design gate** | `R_t` = 0.026 mean / 0.056 final; memory survives `A := 0` bit-identically | `R_t` inside a declared band, and the memory of the past attributable to the graph rather than to the leak |
| 4 | **R2 as a true wiring-only counterfactual** | `np.ones(n_edges)` → R2 also flattens the weights; a global rewire ignoring the input boundary | destroys wiring while preserving the weight distribution, the input population membership and the degree sequences |

---

## 1. Constraint 1 — ORN/PN-aligned input population

**The defect it repairs.** `make_shared` draws `w_in ~ U(-s, s)` of shape
`(N, C)` and every node receives every channel; A6 measured
`corr(node degree, |W_in| mass) = 0.07` and a **consistent relabelling of nodes
leaves the dynamics bit-identical** (`max‖Δh‖ = 0.000e+00`). Node identity — cell
type, layer, pathway — therefore entered the computation only through the wiring,
and the wiring was 99.9 % discarded (constraint 2). The e-nose's channels were
never routed through anything olfactory.

**The requirement.** The channels enter through a node set identified by cell
type, declared in configuration, drawn from the connectome's own annotations
(`connectome/metadata/olfactory_v1_node_meta.csv`, `layer_mean` tiers as
`add_edge_masks.py` defines them). `W_in` is sparse and supported only on that
set: a node outside the population receives nothing.

**Acceptance criteria (validation-only).**

| id | criterion | v1 value | target |
|---|---|---|---|
| C1.1 | every node receiving input has a declared input cell type, and the receiving set equals the declared population | no typed population at all | exact equality |
| C1.2 | `nnz(W_in) / (N·C)` — input-map density | 1.0 | `[SIGN-OFF]` ≤ 0.10 |
| C1.3 | **identity sensitivity**: reassigning which nodes form the input population, holding the graph fixed, changes the states | 0 (impossible: every node received input) | `max‖Δh‖ > 0` strictly |
| C1.4 | the input population is present inside the selected subgraph | ORN count 0 at N≤1000 | `>= [SIGN-OFF]`, and strictly > 0 |
| C1.5 | the input mapping is a named, declared object recorded on every run record | absent (`input_mapping` was added in v1.5) | present in `evidence_unit.components` |

C1.3 is the load-bearing one: it is the inverse of the v1 measurement, and it is
what "the input population is biological" has to mean operationally.

**`[SIGN-OFF]` decisions.** Which cell types form the population (ORN only, or
ORN as a first stage feeding PNs); whether a two-stage olfactory front end
(ORN→PN) is in scope for v2 or deferred; the population's size relative to N.

---

## 2. Constraint 2 — connected olfactory subgraph with preserved biological edges

**The defect it repairs.** At N=250 the `(N,N)` induced subgraph of the real
connectome is a **126-edge residue** with **64.8 % isolated nodes**, retaining
**0.0009** of the selected nodes' out-edges (random-node baseline 0.0006 — i.e.
the "layer-stratified degree-proportional" selection is no better than chance at
keeping edges), largest weak component **14.8 %** of nodes, and **no ORNs**. The
pre-v1.5 comment claiming the selection keeps out-of-block fan-out small is false
as measured. An induced subgraph of a graph this sparse cannot be otherwise: the
expected in-block edge count is `M·(n/N)²`.

**The requirement.** Subgraph selection must be **grown** so that biological edges
survive, not sampled so that they mostly vanish. The selection rule is declared,
seeded, and its sha256 provenance recorded on every run record (a v1 gap).

**Acceptance criteria (machine-checked on the selected subgraph).**

| id | criterion | v1 value at N=250 | target |
|---|---|---|---|
| C2.1 | ORN (or declared input population) count in the subgraph | 0 | `>= [SIGN-OFF]`, strictly > 0 |
| C2.2 | edge retention: in-block out-edges / full-graph out-edges of selected nodes | 0.0009 | `>= [SIGN-OFF]` (orders of magnitude higher) |
| C2.3 | largest weak component / N | 0.148 | `>= [SIGN-OFF]` |
| C2.4 | isolated fraction | 0.648 | `<= [SIGN-OFF]` |
| C2.5 | mean unweighted out-degree in-block | 0.50 | `>= [SIGN-OFF]` |
| C2.6 | deterministic, declared selection; sha256 + target_n + seed on the record | unrecorded | present |

**`[SIGN-OFF]` decisions.** The growth rule (e.g. BFS/expansion outward from the
input population with a declared degree or edge-count cap, versus a declared
cell-type composition); N; whether the size study is re-run under v2 to show where
the retention curve stops being acceptable.

---

## 3. Constraint 3 — recurrent/input ratio as an explicit design gate

**The defect it repairs.** With the v1 regime (`leak` 0.1, `gain` 0.5,
`input_scale` 0.1 — `PINNED_KNOBS`, described as "the first conservative candidate
of each list", i.e. chosen with no reference to the dynamics) the recurrent term
supplies **`R_t` = 0.026** of the drive on average, `‖recurrent‖` = 0.054 against
`‖input‖` = 2.15. Driving with `A := 0` reproduces the state's memory
**bit-identically** (max|corr(h_t, x_{t−1})| = 0.8396 both ways), so none of the
state's memory came from the graph. Consistently, the state's effective rank
tracks the **channel count**, not N.

**The requirement.** The operating point is chosen so the recurrence participates,
and that is verified rather than assumed. The knobs that set it are selected on
**validation** within the protocol's declared grid and the choice is recorded on
every record — not pinned to a fixed conservative value.

**Acceptance criteria (fixed validation input, the A3 definition).**

| id | criterion | v1 value | target |
|---|---|---|---|
| C3.1 | `R_t = ‖gain·A·h‖ / ‖W_in·x + b‖`, mean and final step | 0.026 / 0.056 | inside a declared band, lower bound `[SIGN-OFF]` |
| C3.2 | **graph-attributable memory**: the zero-recurrent control must NOT reproduce the state's memory | bit-identical | `[SIGN-OFF]` margin on `max|corr(h_t, x_{t−k})|` |
| C3.3 | states are not a re-encoding of the input: effective rank of `H` must exceed the channel count by a declared margin | rank ≈ C | `>= [SIGN-OFF]` |
| C3.4 | the knobs are selected on validation and recorded, with the selection payload on the run record | pinned, unselected | present |
| C3.5 | `R_t` is reported next to every v2 reservoir result | never reported | mandatory |

**`[SIGN-OFF]` decisions.** The band for `C3.1` (the requirement is that it is
declared before measuring; a value on the order of the input term — e.g. ≥ 0.5 —
is the natural floor for "the recurrence participates", but the number is yours);
the margin for `C3.2`; whether C3.3's target is an absolute rank or a
multiple of the channel count.

---

## 4. Constraint 4 — R2 as a true wiring-only counterfactual

**The defect it repairs.** `make_degree_rewired` rebuilds the matrix with
`np.ones(n_edges)` and rescales, so R2 is a **uniform-weight** graph: it destroys
the synapse-count weight structure *as well as* the wiring. R0-vs-R2 is therefore a
joint contrast, and "the wiring does not matter" is not what it measured. The
rewire is also global and ignores the declared input population.

**The requirement.** R2 destroys wiring and nothing else: the weight distribution
(declared: global multiset, or per-node degree-weighted), the degree sequences
(in/out, weighted and unweighted as declared), and the membership of the input
population are all preserved, and the swap chain reaches mixing.

**Acceptance criteria.**

| id | criterion | v1 value | target |
|---|---|---|---|
| C4.1 | weight multiset preserved (declared: global or per-node) | **False** (R2 uniform) | exact, as declared |
| C4.2 | input population node set identical between R0 and R2 | not applicable (no population) | exact equality |
| C4.3 | unweighted in/out degree sequences preserved | True | exact (must not regress) |
| C4.4 | weighted degree preserved if declared | False | as declared |
| C4.5 | mixing reached: edge overlap with R0 within a declared distance of the configuration-model expectation | achieved | as declared |
| C4.6 | feasibility: one R2 build completes within a declared wall-clock at the target N | **71 days** projected at M = 14.8 M | `[SIGN-OFF]` |

C4.6 is a real constraint, not a footnote: the O(M)-per-attempt duplicate scan is
why the size study had to be run at small N, and a v2 counterfactual that cannot be
built at the target N would force the same compromise again.

**`[SIGN-OFF]` decisions.** Whether the weight multiset is preserved globally or
per-node; whether weighted degrees are required (a degree-preserving swap on a
weighted graph needs the declared semantics); the wall-clock ceiling.

---

## 5. Invariants inherited from v1.x — must not regress

- **Evidence-unit identity schema 2** (v1.5): one unit = `(dataset, task, fold,
  model, window, condition, reservoir_size, normalization, input_mapping,
  topology_variant, rewire_seed)`; the schema-1 fingerprint is a *fallback alias for
  pre-v1.5 records only*, so two substrates of one unit never collide.
- **Unit-addressable record paths** — a second substrate of one unit must not
  overwrite the first.
- **Parameter evidence scope** (v1.5.1/v1.5.2): counts from matched rows inside a
  declared scope, per dataset, folded with AND, fail closed; no averaging, extrema
  selection, cross-dataset fallback or unscoped search.
- **§17**: only `status == "ok"` occupies the touch quota; skips are disclosed with
  both config hashes; no test split is re-scored.
- **Metrics**: fixed-label macro-F1 with `zero_division=0`; AUROC only when all four
  classes are present, with `n_auroc_defined` reported; D3's macro-F1 is **not**
  magnitude-comparable with D2's.
- **No test split is touched during v2 development.** Every construct check in
  §1–§4 is validation-only, and says so in its own output.

**One prerequisite, still open, that decides v2's statistics.** A9: the decisive
test clusters by `fold_id`, and 62 of 62 D3 `fold_id`s map to a different specimen
under each seed, so the cluster unit is a fold index rather than a specimen. v2's
formal experiments inherit whatever is declared here, so **the cluster unit should
be re-declared as the specimen before v2's formal phase** (a statistics amendment;
no re-run needed). `[SIGN-OFF]`

---

## 6. Order, and the gate between phases

```
1. construct checks   C1.1-C4.6, validation-only, machine-checked report
        |
        v   (only if every criterion passes)
2. formal experiments  existing protocol discipline, v2 identity and scope
        |
        v
3. gates A/B/C, narrative rules, and whatever the result is
```

The gate between 1 and 2 is the whole point: a v2 experiment whose construct
checks failed would repeat v1's mistake at greater cost, and the failure would be
indistinguishable from a hypothesis failure.

**What v2 explicitly does not do.** It does not re-read, re-index or re-score any
v1.x result; it does not alter any frozen protocol file other than by adding a new
version; it does not touch a test split during development; and it does not treat
"R0 > R2" as its success condition.

---

## 7. What failure looks like

- If C1–C4 pass and R0 ≈ R2 on the formal experiments, that is the answer, and it
  is a **valid** negative result for the first time — reportable as such, unlike
  v1's.
- If a criterion cannot be met at the target N or budget, that is reported as a
  design limitation with the measured value, and the phase gate stays shut.
- If the owner relaxes a criterion after seeing model performance, the relaxation
  is recorded as a protocol amendment with its date and reason, and the earlier
  state remains on disk.

---

## 8. Open decisions for the owner — the sign-off list

| # | decision | where |
|---|---|---|
| D1 | which cell types form the input population; whether ORN→PN two-stage is v2 or deferred | §1 |
| D2 | `C1.2` input-map density ceiling; `C1.4` minimum population count in the subgraph | §1 |
| D3 | the subgraph growth rule; N; whether the size study is re-run under v2 | §2 |
| D4 | `C2.2`–`C2.5` floors and ceilings | §2 |
| D5 | the `R_t` band, the `C3.2` memory margin, the `C3.3` rank margin | §3 |
| D6 | R2 weight preservation: global multiset or per-node; weighted degrees required?; the `C4.6` wall-clock ceiling | §4 |
| D7 | whether the A9 statistics amendment (cluster unit = specimen) lands before v2's formal phase | §5 |

Until D1–D7 are settled, this document is a draft and no v2 code should be
written, because every one of them changes what the code has to do.
