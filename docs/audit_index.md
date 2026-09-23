# M4 Audit — index and closure point (v1.x)

**Read this first if you are picking the project up.** It pins the v1.x evidence
chain to a single state so that later commits, experiments and v2 work cannot
quietly move the baseline. Every number below is a quotation from a committed
artefact, and every digest is reproducible from the repository.

Frozen 2026-09-23. Branch `m4-audit/v1.5-evidence-identity`.

---

## Status line — the one sentence that must not be lost

> **M4-v1.4 is retained as an invalid test of the intended biological hypothesis
> due to construct-validity failure; v1.5–v1.5.3 repair evidence identity, gate
> semantics and the cluster unit only, not the biological architecture.**

The evidence for the construct-validity failure — not a "negative result":

| measurement | value | source |
|---|---|---|
| ORNs in the substrate | **0** at N<=1000, 0.47 % at N=4000 | P1-3, `A8_subgraph_degeneracy.json` |
| substrate structure at N=250 | **126 edges**, **64.8 % isolated**, 0.09 % of out-edges retained | A8 |
| recurrent share of the drive | **R_t = 0.026** mean / 0.056 final; memory survives A := 0 bit-identically | A3 |
| state's effective rank | tracks **channels**, not N (3.09/3.14/3.10 for C=3 at N=250/1000/4000) | A2/A6 |
| does the state beat the raw window | **no** — H 0.374 vs X 0.479, and X+real-H 0.392 < X+shuffled-H 0.490 | A2r/A4r |
| delivered R0-vs-R2 evidence | byte-identical metrics in **570/620** units; a dense random graph matches R0 in **561/620** | A10 |

Full argument: [`docs/m4_audit_reservoir_integrity.md`](m4_audit_reservoir_integrity.md)
(section *Locked wording*, and the verdict table at the top).

---

## 1. The closure backup — `full-verify-20260923T063242Z/`

Taken after v1.5.2 merged and Gate A was recomputed.

| | |
|---|---|
| On the GPU box | `/root/autodl-tmp/drososense/backups/full-verify-20260923T063242Z/` |
| On the delivery side | `results/audit/m4_audit/server_evidence/full-verify-20260923T063242Z/` |
| Code identity at capture | `52637fc986aa03c9460f4b880ea94ea59b37c1aa` — *fix(audit): report the per-dataset counts in the comparison* |
| Record counts | **21,860** run records (`record_counts.txt`) |
| Derived tables | **65** files, every one digested (`tables.sha256`) |
| Protocol digests | all seven, cross-checked against the committed sidecars (`protocols.sha256`) |

Contents: `HEAD.txt`, `protocols.sha256`, `record_counts.txt`, `tables.sha256`,
`repo-results-configs.tar.gz` (code + results + configs, 6.0 MB),
`results-raw.tar.gz` (all 21,860 records, 3.2 MB), `branch.bundle` (4.2 MB). The
archives are git-ignored; the four manifests are committed and are what a restored
copy is verified against.

**Verification state:** the seven protocol files on the box hash identically to
the committed sidecars (`v1.1` → `v1.5.2`, all SAME), and the local copy of
`results-raw.tar.gz` lists **21,860** `.json` members. The `21,860` figure is
`21,867` minus the 7 `e2_smoke` records — see §4.

---

## 2. Protocol chain v1.1 → v1.5.3 (frozen, sidecar-matched)

An amendment **adds** a version file; it never edits a frozen one. Every file below
is on disk byte-for-byte as frozen, with a `configs/<name>.sha256` sidecar whose
digest matches it. The **loadable** base protocol remains `protocol_v1.3.yaml`
(`drososense/utils/paths.py: PROTOCOL_PATH`); the later files are amendment records
layered on it, and the runners' `PROTOCOL_VERSION` label carries the newest one.

| version | sha256 | what it is |
|---|---|---|
| `protocol_v1.yaml` | `0eddb56033b53ce5cfa855b1212c3c91336282f1f59226141f6722d2d21f6c15` | v1 — predates the sidecar convention |
| `protocol_v1.1.yaml` | `b87ed6057962b00e41c175acd7c78f314e552c50e2eecf53174e6156d8f7c0b9` | fields, contrasts, gates, narrative rules |
| `protocol_v1.2.yaml` | `32c57f6efaa8f9f14584d3a153fddc2814c5be8829eb888fc0d6c27d536a0cc1` | decision machine (gate symbols, cluster bootstrap) |
| `protocol_v1.3.yaml` | `d85640e556db6c00c8217befda17c967772c30470be9fbe73b2051b3cbfe8357` | **active/loadable**: D3 LOSO(62) per OD1(a) |
| `protocol_v1.4.yaml` | `4f4504528f02df53bc7585f7bac4abe9f4e6bb66301ec9e74f7e36ce47c42d89` | §17: only `status == "ok"` occupies the touch quota |
| `protocol_v1.5.yaml` | `c220a54fabcb1e042d526224c2004d3a72241e8bf7b04057739431d0b9c63ae2` | **evidence-unit identity, schema 2** (§3) |
| `protocol_v1.5.1.yaml` | `8ed71ceeca7a4cf333b1148fd88b83ef1ef61ee319c424401de618550d91fb21` | **parameter evidence scope**, fail closed (§5) |
| `protocol_v1.5.2.yaml` | `f9bee020fdd14179e9cd0d6070a10d6758b0e78881c6c6df13254d59b9155027` | **dataset-conditioned parameter gate semantics** (§5) |
| `protocol_v1.5.3.yaml` | `85f64202e18d00dd025196f970d694c6844ab6a95fdeab0f2917175aeb901b67` | **the cluster unit is the specimen** (A9/D7, §5b) |

---

## 3. E9 identity migration — the disposition identity closes at 420

`results/evidence_migration/` — `e9_v14_to_v15_manifest.csv` (420 rows, exactly one
state each), `e9_adopted.jsonl` (350), `e9_rejected.jsonl` (70),
`e9_migration_audit.json`.

```
adopted=350 + superseded=0 + duplicate_test_touch=70 + invalid=0 + missing=0 = 420   [CLOSED]
```

- 420 records = 70 legacy fingerprints x 6 records; 350 evidence units = 70 x 5 substrates.
- The duplicated unit is **N=250, scored twice**: `e9_size_d2` (config `2b681f4040fc`, 06:07:31Z) was **kept**; `e9_size_d2_n250` (config `e2dbcef1dad6`, 06:52:35Z) is the **rejected duplicate test touch**.
- **Designation rule, frozen and result-blind:** earliest `timestamp_utc` wins, ties broken by the lexicographically smaller `config_hash`. Stated before any metric was examined.
- It is a migration of **identity only**: all 350 adopted records' metrics, duration, timestamp, status, config_hash and counts are byte-identical to the originals, each carrying the origin file's sha256. No model was re-fit, no test split touched.
- `fingerprint_collision_only` is **not** a record disposition (a collision is a property of the legacy *fingerprint*); it is carried as the boolean column `unit_identity_collided`.

**Do not re-run E9.** Its scientific result is already citable from the delivered
records: macro-F1 falls from 0.314 (N=250) to 0.242 (N=4000) and MAE rises from
1.69 to 6.26 — adding connectome is harmful on this design.

---

## 4. `e2_smoke` — permanent status

> **7 per-run records irrecoverably lost; surviving derived summary and
> data-contact evidence retained; no re-evaluation permitted under §17.**

Established, not assumed: `results/tables/e2_smoke_summary.csv` records all seven
rows as `dataset=d2_beef_uncontrolled`, **`evidence_class=real`**,
`protocol_compliant=True`, one specimen evaluation each; and
`data_contact_log.json` carries an `e2_smoke` entry with
**`counts_as_first_test_evaluation: true`**. It was a **real D2 test touch**, so
re-running it would re-score a real test split. Its derived summary is the
surviving evidence and is committed. A synthetic smoke test for CI, if wanted,
must be a **new** label on synthetic data — never this label, never on D2.

The loss itself, the near-miss and the recovery are recorded in
[`docs/incident_2026-09-23_run_records.md`](incident_2026-09-23_run_records.md),
including the correction that the first recovery pass checked only `results/raw`
and missed five derived tables.

---

## 5. Gate A — final closure, with the evolution chain

Computed by the frozen-expression gate engine, not written by hand
(`ops/audit/gate_a_scope_comparison.py`). Both terms and both resolutions come from
`scripts.analyze.evaluate_rules`.

| term | result |
|---|---|
| **A1** — R0 not meaningfully worse than R4 on D2 and D3 | **True** |
| **A2** — `∀ d ∈ {D2, D3}: params(R0, d) < params(GRU, d)` | **True** |
| **Gate A** (A1 and A2 combined by the frozen expression) | **True** |

| dataset | params(R0) | params(GRU) | A2 |
|---|---:|---:|---|
| D2 | 1004 | 4452 | PASS |
| D3 | 1004 | 4068 | PASS |
| aggregate | — | — | PASS iff both PASS |

**The evolution chain, which is the part that matters:**

| state | Gate A | why |
|---|---|---|
| M4-v1.4 (unscoped scan) | **False** | `params(R0) = 16004` — an **E9 N=4000** readout. The gate failed for a reason unrelated to the comparison, and its verdict depended on which unrelated experiments were on disk. |
| v1.5.1 (declared scope, no dataset conditioning) | **UNEVALUABLE** | `params(GRU)` is 4452 on D2 and 4068 on D3 within one scope, and the protocol's single dataset-agnostic term had no unique value. Reported, never resolved by a maximum. |
| v1.5.2 (dataset-conditioned semantics) | **True** | Per-dataset matched configurations, folded with AND over `evaluated_on`. |

Wording that must be used for A2 (not the old single-value story):

> **R0 has fewer trainable parameters than GRU under each registered evaluation
> configuration on both D2 and D3.**

---

## 5b. D7 closure — the cluster unit is the specimen (v1.5.3)

Committed by `configs/protocol_v1.5.3.yaml` (digest in §2), signed as **D7** in
[`v2_preregistration.md`](v2_preregistration.md) §5, and closed here with the item
**A9** it repairs. It changes the *independent unit* of the cluster-level statistics
and nothing else: alpha, the exact two-sided sign test over cluster means, the
bootstrap parameters, the equivalence margins, the multiplicity rule, the metrics,
the gates and the narrative rules do not move.

| | cluster unit | what a cluster mean averages |
|---|---|---|
| v1.4 / v1.x, measured | the fold **index** | ten **different** specimens (all 62 of 62 D3 `fold_id`s change specimen across the 10 seeds) |
| **v1.5.3, measured** | the **specimen** | that specimen's ten seed evaluations |

Measured on the **committed** evidence, before and after:

| evidence | v1.x reported | v1.5.3 reports |
|---|---|---|
| `e1_main_d2_per_run.csv` (5 folds, 5 specimens, 1 per fold) | 5 clusters | **5 clusters**, 50 pairs, floor `0.0625 > α` |
| `e1_main_d3_per_run.csv` (62 folds, 62 specimens, 1 per fold) | **5 clusters** (fold indices) | **62 clusters**, 620 pairs, `n_clusters_nonzero = 20`, floor `1.91e-06`, p = 0.263 |
| `m1_benchmark_per_run.csv` D3 (5 folds over 62 fillets, **12–13 specimens per fold**) | 5 clusters | **`unpairable`** with the reason: *a fold whose test set holds several specimens cannot be attributed to one specimen* |

Each cluster now carries a machine-checkable `cluster_provenance` naming the
independent unit: `specimen_id`, `source_folds`, `source_seeds`, `n_rows`,
`models_present`, `tasks_present`. For D3's `F1F1`, the ten seed evaluations sit
under **ten different fold ids** — which is precisely why a fold index cannot be an
identity.

**The refused split is a finding, not a limitation.** A grouped k-fold cannot
produce a cluster-level statement about specimens at all: it has no cross-seed-stable
unit. The v1.x D3 cluster-level numbers were produced by clustering those folds by
index and are retained as v1.x artefacts; they are **not comparable** with a v1.5.3
run, and `results/tables/m1_benchmark_statistics.csv` (which reports D3
`n_clusters = 5`) must be read that way.

**Pairing completeness** is part of the same amendment: a specimen observed on a
different set of `(seed, fold_id, window_length)` evaluations by the two sides — or
by only one of them — has no paired cluster mean, so the contrast is UNEVALUABLE
with the affected specimens named. Before v1.5.3 the inner join silently kept the
overlap and reported the specimen as a full cluster. **Expect wider CIs and less
significance; that is the correct cost of the fix, not a reason to defer.**

---

## 6. Open — carried forward, not dropped

These are **not** closed by v1.x and must not be lost when v2 starts:

1. **A9 — CLOSED by v1.5.3 (§5b).** The decisive test clustered by the fold
   **index**; the frozen text justifies the fold *because* "For LOSO a fold is a
   single specimen, so the fold bootstrap IS a specimen bootstrap"
   (`pairing.resample_unit_detail`), on the stated premise that the ten seeds share
   **one specimen partition**. Under a seeded LOSO permutation each seed gets a
   *different* partition — measured: all **62 of 62** D3 `fold_id`s hold a different
   specimen under each of the 10 seeds — so the premise failed. The repair (D7) is
   the declaration of the cluster unit as the **specimen**, frozen before any formal
   v2 evaluation. No data was re-run and no v1.x record was re-scored. What remains
   open from it is only the *reporting* convention on the v1.x tables, covered by the
   note in §5b: they are v1.x artefacts and are not comparable with a v1.5.3 run.
2. **Run-record field truthfulness** — `params.reservoir_size = 200` while
   `topology.n_nodes = 250`; node-selection provenance (`target_n`, seed, sha256)
   is not recorded; `class_coverage` lives only in the git-ignored raw records, so
   the macro-F1 ceiling cannot be verified from the delivery line.
3. **The biological architecture** — the whole of v2. See §7.

---

## 7. Where the rest lives

| artefact | what it holds |
|---|---|
| `docs/m4_audit_reservoir_integrity.md` | the audit: verdict table, A1–A11, the real-data half, the locked wording, the disposition table |
| `docs/incident_2026-09-23_run_records.md` | the `results/raw` loss, its recovery, and the rule it produced |
| `docs/merge_drop_policy.md` | why the GPU tree's DATA-52 `cpu_seconds` deletion was rejected rather than merged |
| `ops/audit/reservoir_integrity_audit.py` | A1/A7/A8 (graph level, real NPZ) |
| `ops/audit/reservoir_dynamics_audit.py` | A2–A6 (synthetic probe) |
| `ops/audit/reservoir_realdata_audit.py` | A2r/A4r/A5r (real D2/D3 windows) |
| `ops/audit/evidence_ledger_repair_audit.py` | the E9/E3 collision-vs-repeat classification |
| `ops/audit/e9_identity_migration.py` | the v1.4→v1.5 migration and its closure |
| `ops/audit/gate_a_scope_comparison.py` | the pre/post Gate A comparison |
| `results/audit/m4_audit/` | every evidence JSON, the ledger-repair outputs, `server_evidence/` |
| `results/evidence_migration/` | the E9 migration artefacts |

### Reproducing the closure in one command

```bash
python ops/audit/gate_a_scope_comparison.py     # Gate A, both terms, both resolutions
python -m pytest tests/ -m "not slow"           # 557 passed, 8 skipped (local)
```

On the GPU box the same suite reports **565 passed, 0 failed** (nothing skips there).

### Known local-suite qualifications

Removing the local connectome data (moved to the box, verified, then deleted —
12.2 GB freed) makes two real-NPZ tests **skip** with the reason *"olfactory_v1.npz
not provisioned in this environment"*: the local suite is 557/8 rather than
559/6. They run on the box. Three xgboost tests skip locally because `libomp` is
absent.

---

## 8. Next

```
audit index (this file)  ->  v2 preregistration  ->  validation-only construct checks  ->  formal experiments
```

**The D7 statistics amendment is now frozen (§5b), which is the gate the signed
decision put in front of the construct phase.** The order for v2 is fixed and
construct-validity-first: **C1 input mapping → C2 subgraph → C4 R2 counterfactual →
C3 dynamics**, each a property of the *construction*, measured validation-only, before
any formal inference. C3 depends on a valid R0 and a valid R2 graph, which is why it
comes last. Formal experiments run only when **every** criterion passes; "close" is a
failure.

**C1 is implemented and measured** —
[`docs/v2_construct_c1_input_mapping.md`](v2_construct_c1_input_mapping.md),
`results/audit/v2_construct/C1_input_mapping.json`. C1.1–C1.6 pass on the construct
substrate (the full delivered olfactory graph), including C1.3 — the strict inverse of
A6, where the v1 dynamics were bit-identical under relabelling. Two findings carry
forward into C2:

1. **C1.4 fails on every substrate the project actually ran**: the delivered v1
   selections hold 0/1/1/10/19 ORNs against a threshold of `max(20, 2·Din) = 20`, and
   at N ≤ 1000 the mapping refuses to build at all (fewer ORNs than channels). C2's
   selection is a **precondition**, not an optimisation.
2. **The cell-type vocabulary is derived, and two delivered tables of it disagree on
   8,679 of 124,185 nodes (7.0 %)** — which moves the admissible receiving-PN pool
   between 949 (declared tiers) and 2,091 (classified edge metadata). C1 is measured
   against the committed table; the choice is an open biological question.

**C2 is implemented and measured** —
[`docs/v2_construct_c2_subgraph.md`](v2_construct_c2_subgraph.md),
`results/audit/v2_construct/C2_subgraph.json`. At the primary N=1000, Din=5 the
deterministic ORN-seeded expansion passes **all six** criteria — retention 1.0000,
largest weak component 1.0000, isolated fraction 0.0000, mean out-degree 80.443,
14,305 real ORN→PN edges reaching 232 of 243 selected PNs — against the v1 selection
on the same graph at the same N: 2,343 induced edges, component 0.575, isolated 0.415,
Enrichment 2.29× vs **78.71×**. Three things are recorded rather than glossed:

- **C2.2 does not discriminate**: both selections retain 1.0000 of their induced
  eligible edges, so the criterion is a guard; the discriminating quantities are the
  induced edge count and Enrichment, and both are reported on every line.
- **Two defects were found by measuring, not by reading**: the layer-staged frontier
  starved KC and MBON/DAN to zero, and then a *group* floor left KC at 21 against a
  floor of 50. The allocation is now per class and per group, declared, with no tuning
  constant, and C1.2 + C1.4 together fix a derived minimum N of 110 at Din=5.
- **No formal inference has run**, and N=1000 is a registered engineering operating
  point. The composition (24.3 % ORNs against the connectome's 1.83 %) is a declared
  allocation, not a biological ratio.

**C3 is implemented and measured, and the gate FAILS** —
[`docs/v2_construct_c3_dynamics.md`](v2_construct_c3_dynamics.md),
`results/audit/v2_construct/C3_dynamics.json`. Both binding gates fail by the same
mechanism the v1 audit found: the substrate the C2 expansion produces (mean out-degree
80.4 on the 1000-node induced block) has a recurrent drive that is **orders of magnitude
below** the input drive through W_in. Tuned across a 36-point (gain x leak) grid, on
every reservoir (R0/R2 x shared-scale/rho-matched):

* **C3.2 (memory drop ≥ 20 %)** never passes — `M (real) ≤ M (A := 0)` everywhere; the
  recurrent graph contributes nothing to the autocorrelation the metric measures.
* **C3.3 (D_eff ≥ 1.5 · Din)** never passes — the state matrix's effective rank tops at
  ~1.1 · Din; the recurrent drive is too weak to spread the state space beyond the
  input dimensionality.

The C3 failure is correlated with the substrate's density, but the data does not
establish a causal link. The next round (M5 - Structural Dynamics Audit) separates
"structure failure" (graph has no directed cycles to speak of) from "dynamics failure"
(graph has cycles but the row-L1 normalization suppresses them).
* **C3.1 (R_t in [0.20, 1.00])** does pass — R0 at gain=2/leak=1, R2 at gain=3/leak=0.5
  (heavy right tail: P90 ≈ 7.5–14). C3.1 alone is not enough: the C3 gate is the AND.

This is the construct phase gate closing on a **measured finding**, not on a relaxed
threshold or a fixed defect: per the project rule, "close" is not a pass, and no
formal experiment will run on a construction that demonstrably cannot produce
recurrent dynamics. The next decision is the construction, not the gate.

**M5 - Structural Dynamics Audit** ([`docs/v2_construct_m5_structural_dynamics.md`](v2_construct_m5_structural_dynamics.md),
[`results/audit/m5_structural_dynamics/M5_structural_dynamics.json`](../results/audit/m5_structural_dynamics/M5_structural_dynamics.json))
separates the two structural causes the C3 failure leaves open:

* **"graph lacks recurrence"**: REJECTED -- the C2 substrate is one giant SCC with every
  node in a non-trivial cycle, every ORN-reachable node in the recurrent core. WCC = 100
  %, SCC = 100 %, cycle_edges = 100 %, orn_reach_core = 100 %.
* **"graph has cycles but row-L1 normalization suppresses them"**: REJECTED -- gamma-only
  rescaling (no row normalization) gives essentially the same Krylov D_eff as row-L1
  (5.68 vs 5.35 at K=16). The normalization is not the bottleneck.

The data surfaces a **third cause**: the linear Krylov `span(A^k · B)` is bounded by
`≈Din` on this substrate and this input geometry, for any preprocessing that keeps
tanh in its linear regime. The cycles exist; the algebraic collapse is because the
same weight pattern that fills the SCC also makes `span(A)` concentrate on the
directions `span(B)` already occupies. The next decision is the **input geometry /
substrate eigenstructure / drop-the-reservoir-claim** -- not a code change to the
construct check.

**C4 is implemented and measured, and the gate PASSES under AMENDMENT 1** —
[`docs/v2_preregistration_amendment_1.md`](v2_preregistration_amendment_1.md),
[`docs/v2_construct_c4_r2_counterfactual.md`](v2_construct_c4_r2_counterfactual.md),
`results/audit/v2_construct/C4_r2_counterfactual.json`. The four conservations are exact
(degree hashes, weight multiset hashes, per-source hashes, input pathway identical) and
overlap is 0.19998 in 13.95 s — but the amendment had to be signed first, and the failure
that produced it is preserved in the record because it is what made the owner decide the
amendment should exist.

The pre-amendment failure was a specification problem, not an algorithm defect:

- the counterfactual was built on R0's **normalized** matrix and asked to preserve its
  weight multiset exactly; but `n1_pre_l1` is the presynaptic L1 normalization (rows),
  measured against the delivered block, so the normalized weights are a function of the
  wiring and the swap space collapses to ~5 edges per weight class — overlap stalls **flat
  at 0.858** over 4.3 M swaps;
- C4.6's fully-mixed expectation is substrate-dependent: 0.0570 on the v1 substrate it
  was calibrated on, **0.2022** on the C2 substrate — i.e. AT the absolute 0.20 threshold.

**Amendment 1 declares** that C4.2/C4.3/C4.4/C4.6 are measured on the **raw** synapse-count
graph (the biological object) with the declared normalization applied identically to both
graphs under R0's scale factor, and that C4.6 is floor-relative (`overlap ≤ max(0.20, 1.10 ·
E[overlap])`). The amended result passes every line. Two disclosures on every R2 line: the
normalized in-strength profile differs by 0.455 (median) — the wiring's own signature,
not gated — and R2's spectral radius differs by ~6.7 % (the family convention matches
radius; the alternative is measured beside it at 0.453 in-strength difference).

**What the first run caught** — was that C4.6 was doing exactly its job. The earlier
description remains the record of how the finding came about.

**What C4 passing does and does not licence.** It says a fair wiring-only counterfactual
exists and is measurable. It says nothing about biology, and no formal inference has run.
C3 (`R_t`, memory benefit, effective rank) may now start — it needs a valid R0 AND a valid
R2, which is exactly what amendment 1 closes.

**C4 is implemented and measured, and the gate FAILS on mixing** —
[`docs/v2_construct_c4_r2_counterfactual.md`](v2_construct_c4_r2_counterfactual.md),
`results/audit/v2_construct/C4_r2_counterfactual.json`. The four conservations hold
**exactly** (degree-sequence hashes, global and per-source weight-multiset hashes, and the
ORN-aligned input mapping rebuilt independently on R0 and R2 agreeing on support rows and
`w_in` digest) and the in-strength error is 7.7e-08 — but `edge_overlap` is 0.858 against a
0.20 threshold and the 600 s budget is spent, so **C4 = FAIL**.

The failure is **a specification question, not an algorithm defect**, and it is put to the
owner rather than resolved here:

- C4.6's "fully-mixed expectation" is substrate-dependent. Measured: **0.0570** on the v1
  substrate it was calibrated on, **0.2022** on the C2 substrate — whose density is exactly
  what makes C2.3–C2.5 pass. The threshold sits *below* the substrate's own mixing floor.
- The multiset R2 must preserve is R0's **normalized, rescaled** weights, where the raw
  quantization is gone: 15,277 distinct values over 80,443 edges (mean class **5.3**)
  against 363 values (mean class 221.6) on the raw synapse counts. Exact weight matching —
  which is what makes C4.4 exact — leaves each edge a 5.3-edge reachable set, and the
  mixing curve is **flat** (1.0000 → 0.8608 → … → 0.8585 over 4.3 M swaps). Trapped, not
  slow.
- Three designs were measured (`C4_design_diagnostic.txt`): **A** (rewire the normalized
  matrix, exact weights) conserves exactly and cannot mix; **B** (±25 % partners) mixes to
  0.351 but drifts in-strength to 0.318; **C** (rewire the RAW matrix, then preprocess both
  identically) conserves exactly on the raw object and reaches **0.19998 in 14.4 s**, at
  the cost that the *normalized* pair no longer shares a weight multiset.

**Recommendation put to the owner:** declare the measurement object (reading R2 — the
criterion is about the biological graph, with the normalization applied identically to
both), which admits design C and satisfies C4.1–C4.7 together, and optionally make C4.6
floor-relative (`O ≤ min(0.20, 1.10 × E[overlap])`; v1's own numbers satisfy it at ratio
1.02). **Not proposed:** relaxing C4.6 to what the chain reached, or dropping C4.2/C4.3.
**No formal inference may run while a construct criterion fails, and C3 must not start
before C4 closes** — C3 measures the recurrence on a valid R0 *and* a valid R2.

v2 is the biological architecture redesign. Its pre-registration draft is
[`docs/v2_preregistration.md`](v2_preregistration.md) — written before any v2 code
exists, with every "v1 defect" it quotes verified against the evidence JSON in
`results/audit/m4_audit/`. Its four hard constraints are:

1. **ORN/PN-aligned input population** — not a dense random `W_in` over every node.
2. **Connected olfactory subgraph with preserved biological edges** — not a
   126-edge residue with 64.8 % isolated nodes.
3. **Recurrent/input contribution ratio as an explicit design gate** — not 3–5 % by
   accident.
4. **R2 as a true wiring-only counterfactual** — preserving weight statistics and
   the input/output boundary, rather than a counterfactual that also flattens the
   weights.

None of this is a re-tuning after seeing a test result: it repairs a
construct-validity failure established by an independent audit of the
implementation and the delivered records, before any v2 design decision was taken.
