# M4-Audit / Reservoir Integrity Audit — findings

> **Closure point.** The v1.x evidence chain is pinned to one state in
> [`docs/audit_index.md`](audit_index.md): the `full-verify-20260923T063242Z`
> backup and its verification state, the protocol chain v1.1 → v1.5.2 with
> digests, the E9 migration disposition (350 adopted / 70 duplicate test-touch),
> the permanent `e2_smoke` status, and Gate A's final closure with its evolution
> chain (v1.4 `False` → v1.5.1 `UNEVALUABLE` → v1.5.2 `True`). Start there.

**Status:** audit executed, verdicts recorded. No test split was touched, no
frozen protocol file was edited, no large batch was started, and every M4
result already on disk is left exactly as it is.

**The question this audit answers.** The project was about to read "R0 ≈ R2" as
*the biological topology does not matter for this task*. Before that reading is
admissible, one has to establish something weaker and prior: *did R0 and R2 ever
produce different, non-degenerate dynamics at all?* This document records the
measurement of exactly that, and nothing else.

**Why the audit was warranted at all** (not motivated reasoning — the project has
a track record of real implementation defects that were found the same way): the
run device fix (DATA-34) where every sequence model crashed on the wrong device;
the DATA-61 cross-experiment anchor rule that silently skipped units; the
`fcac719` merge that silently dropped ~2,000 lines; §17 counting crashed runs as
test touches (fixed in v1.4).

---

## The one-paragraph answer

**R0 ≈ R2 is not yet interpretable as a scientific result, because the
implementation confines the reservoir's state to a manifold whose dimension is
set by the number of input channels (≈ C + 1), not by the connectome.** With
`leak = 0.1`, `gain = 0.5`, `input_scale = 0.1` and `spectral_radius = 0.9`, the
recurrent term supplies **3–5 %** of the drive magnitude, the hidden state's
effective rank is **C** (measured 5.35 for a 5-channel input) **independently of
N = 250, 1000 or 4000**, and removing the connectome entirely (A := 0) changes
the state **less** than rewiring it does. The 8,680 delivered E2/D3 records
recovered from the GPU box confirm the consequence directly: R0 and R2 produce
**byte-identical metric dictionaries in 570 of 620 units**, and a *dense random*
graph reproduces R0 in 561 of 620 — the whole topology family is interchangeable
at the readout. Two secondary defects compound this: the induced subgraph the
reservoir actually sees retains **0.09 %** of the selected nodes' out-edges
(126 edges at N=250, 64.8 % of nodes isolated), and the statistics layer clusters
by `fold_id`, which under a seeded LOSO is **not** a specimen. A third, newly
found: the §17 test fingerprint omits the reservoir configuration, so the E9 size
study is **70 protocol violations** by the project's own rule and cannot be
loaded by its own pipeline — and that study shows performance *decreasing*
monotonically with N (macro-F1 0.314 → 0.242). The correct next step is a v1.5
statistics/identity amendment plus a v2 architecture re-operationisation —
**not** another sweep at the current design.

---

## Verdict table

| # | Audit | Verdict | Headline number |
|---|---|---|---|
| A1 | R0/R2 adjacency identity | **PASS** | distinct objects, well-mixed: overlap 0.058 vs fully-mixed 0.055 (N=1000) |
| A2 | Hidden-state divergence | **NOTE** | d(R0,R2) = 0.067 vs d(R0,R3) = 0.21, d(genuine difference) = 1.39 |
| A3 | Recurrent vs input drive | **FAIL** | R_t = 0.026 mean / 0.056 final; memory survives A := 0 identically |
| A4 | Readout ablation | **INCONCLUSIVE (fixture)** | all 5 arms identical; fixture is not learnable, so the arm cannot discriminate |
| A5 | Normalization sensitivity | **NOTE** | R_t 0.012–0.028 across n0_raw / n1_pre_l1 / n5_binary — not the dominant cause |
| A6 | Input-mapping audit | **FAIL (design)** | relabelling invariance max‖Δh‖ = 0.000e+00; every node receives input |
| A7 | R2 rewiring quality | **PASS + 2 notes** | degrees preserved exactly, 10/10 distinct graphs; but R2 flattens weights; full graph = 71 days |
| A8 | Subgraph degeneracy | **FAIL** | N=250 → 126 edges, 64.8 % isolated, 0.09 % of out-edges retained |
| A9 | Metric implementation | **PASS + 1 FAIL** | macro-F1 self-consistent (438/438); but the cluster unit is not a specimen |
| **A10** | **E2 delivered evidence** (server) | **CONFIRMS A2/A3/A8** | R0 vs R2 identical metrics in **570/620** units; R0 ≈ R6 (dense random) in 561/620 |
| **A11** | **E9 size study** (server) | **FAIL (new)** | performance *decreases* with N; **70 §17 violations** by the project's own rule |
| **A2r/A4r/A5r** | **real-data half** (D2/D3) | **FAIL** | H_only **0.374** vs X_only **0.479**; X+real-H 0.392 **worse** than X+shuffled-H 0.490 |
| **P1-3** | subgraph annotation composition | **FAIL** | the "olfactory subgraph" contains **0 ORNs** at N≤1000, 0.47 % at N=4000 |

---

## A8 — the reservoir is not looking at the connectome (FAIL)

The runner's `reservoir_size` default is `None`, which
`drososense/reservoir/runner.py` resolves to **the full graph**; the declared E9
size study and `connectome/select_neurons.py` use N ∈ {250, 500, 1000, 2000,
4000}, and every control is built as the **induced** subgraph
`matrix[idx, :][:, idx]`. Measured on the real
`data-root/connectome/adjacency/olfactory_v1.npz`:

| N | edges in the (N,N) block | isolated nodes | mean out-deg (unweighted) | largest weak component |
|---:|---:|---:|---:|---:|
| 250 | **126** | **64.8 %** | 0.50 | 14.8 % |
| 500 | 657 | 44.0 % | 1.31 | 44.4 % |
| 1000 | 2,343 | 30.5 % | 2.34 | 57.5 % |
| 2000 | 7,414 | 20.8 % | 3.71 | 61.8 % |
| 4000 | 19,819 | 8.7 % | 4.96 | 85.0 % |
| full | 14,828,657 | 0 % | 119.41 | 100 % |

The load-bearing number is not the edge count but the **retention**: at N=250
the (N,N) block keeps **0.0009** of the selected nodes' total out-edges
(random-node baseline **0.0006**). So 99.9 % of the connectome's wiring is
discarded by the sub-selection, and the selection function is *not* doing what
the code says it does:

> `connectome_reservoir.py`, on the sub-selection: *"The DATA-3 select_neurons
> function picks nodes whose out-of-block fan-out is small relative to M."*

That claim is false as measured: `select_neurons` performs **layer-stratified
degree-proportional** sampling, which actively *prefers* high-degree hubs, and a
hub's edges leave any 250-node block. Note also that `meta.json`'s "mean degree
≈ 430" is the **weighted** degree (sum of synapse counts); the unweighted mean
out-degree is M/N = 119.41, and it is the unweighted one that governs whether an
induced subgraph stays connected.

*Consequence:* at any declared E9 size, "R0" is a nearly edgeless residue, not
the olfactory connectome. A null result from it is a statement about a 126-edge
graph.

## A3 — the recurrence is a rounding error on the input (FAIL)

`R_t = ‖gain · A h_{t-1}‖ / ‖W_in x_t + b‖`, measured in the exact regime the
runner pins, on a seeded synthetic input with no dataset involved:

| N | M | R_t mean | R_t final | ‖recurrent‖ | ‖input‖ |
|---:|---:|---:|---:|---:|---:|
| 250 | 126 | 0.0256 | 0.0556 | 0.054 | 2.151 |
| 1000 | 2,343 | 0.0283 | 0.0540 | 0.121 | 4.282 |

The state update is therefore dominated by the instantaneous input projection.
The **zero-recurrent control** makes this unambiguous — driving with `A := 0`:

| memory probe | R0 | A := 0 |
|---|---|---|
| max abs corr(h_t, x_{t-1}) | 0.8396 | **0.8396** |
| max abs corr(h_t, x_{t-4}) | 0.8121 | **0.8121** |
| max abs corr(h_t, x_{t-8}) | 0.7311 | **0.7311** |
| d(R0, A:=0) final | — | 0.0360 |

Two conclusions: (i) **all** of the state's memory comes from the leak's
integration of the input, none from the graph; (ii) `d(R0, A:=0) = 0.036` is
**smaller** than `d(R0, R2) = 0.067`, i.e. deleting the connectome perturbs the
state less than rewiring it does. That is only possible because R0's own edge
mass (126) is smaller than the symmetric difference between R0 and R2 (≈226) — a
direct symptom of degeneracy.

## The core mechanism — state dimension = channels, not neurons (A2/A3/A6)

`W_in x_t` is a linear combination of the C columns of `W_in`, so the drive at
every step lives in a subspace of dimension ≤ C + 1 (the input columns plus the
bias); `h_t` is a leaky accumulation of `tanh` applied elementwise to that
drive, so the whole trajectory lies on a **(C + 1)-dimensional manifold** and
the recurrence can only wrinkle it by the 3–5 % measured in A3. Measured
effective rank (participation ratio of the state's singular values):

| N | M | C=3 | C=5 | C=8 | C=20 |
|---:|---:|---:|---:|---:|---:|
| 250 | 126 | 3.09 | 5.35 | 8.35 | 19.34 |
| 1000 | 2,343 | 3.14 | 5.36 | 8.43 | 19.92 |
| 4000 | 19,819 | 3.10 | 5.31 | 8.37 | 19.98 |
| 1000, **A := 0** | — | 3.05 | 5.24 | 8.25 | 19.80 |

**Effective rank ≈ C. It does not grow with N, and setting the connectome to
zero changes it by ~0.1 dimensions.** Adding neurons adds state-norm (‖h‖ scales
as √N) but no new *directions*. Since D2 has 5 channels
(`{MQ3, MQ5, MQ135, temperature, humidity}`), the reservoir's usable state space
is ~5–6 dimensional regardless of which graph — which is a complete and
sufficient explanation of R0 ≈ R2, and of R0 ≈ R3 ≈ R4 as well.

A6 closes the conceptual half: the input map is a **single dense random `W_in`
covering every node uniformly** — no ORN/PN input population, no cell-type
gating, no layer assignment — and the architecture is **exactly invariant to node
identity**: relabelling the graph and the input map consistently leaves the state
bit-identical (max‖Δh‖ = 0.000e+00). Node identity therefore enters the
computation only through the adjacency's wiring pattern, which A8 shows is 99.9 %
discarded. `corr(node degree, |W_in| mass) = 0.07` — i.e. the input map is, by
construction, unrelated to the connectome.

Also worth recording as a **PASS** on the user's P0-5: the readout is
`column_stack([h_L, ones])` — **H-only plus a bias; raw `x` is never a readout
feature**, so the readout cannot bypass the reservoir. The failure mode is not
bypass; it is that `h_L` itself is an almost memoryless function of the input.

## A9 — the decisive test clusters by the wrong unit (statistics-layer FAIL)

Verified from the committed `results/tables/e1_main_d{2,3}_per_run.csv` alone:

- **62 of 62 D3 fold_ids have a different test specimen under each of the 10
  seeds** (D2: 5 of 5). Within one seed every specimen is distinct, and each
  specimen is tested exactly 10 times — but its ten observations are scattered
  across ten *different* fold_ids.
- The protocol's decisive test is *"an exact two-sided sign test over cluster
  means"* with `n_clusters = n_folds`, and `stats.fold_cluster_bootstrap`
  averages the seeds **inside** each cluster.
- So a cluster mean is an average over ten *different* specimens, and the 62
  cluster means are 62 different **re-partitions of the same 62 specimens**
  rather than 62 independent specimens. A sign test over them assumes an
  independence that the design does not provide; conversely, clustering by
  specimen (each specimen = 10 repeated measures) is the unit the protocol's own
  language describes. Reachability (`2/2^62`) is unaffected — the count is still
  62 — but *what is being averaged inside a cluster* is not what the text says.

This is fixable in the statistics layer alone, with no data re-run.

> **CLOSED by protocol v1.5.3** (signed as D7, `docs/v2_preregistration.md` §5;
> closure record in [`docs/audit_index.md`](audit_index.md) §5b). The cluster unit is
> declared as the **specimen**, and the fix is measured on this same committed
> evidence: D3 goes from **5 clusters by fold index** to **62 clusters by specimen**
> (620 pairs, `n_clusters_nonzero = 20`, floor 1.91e-06, p = 0.263), each cluster
> carrying a `cluster_provenance` whose `source_folds` for `F1F1` are ten *different*
> fold ids. A split that cannot be attributed to one specimen — the committed
> `m1_benchmark_per_run.csv` D3 rows, 12–13 fillets per fold — is reported
> `unpairable` with that reason instead of being clustered on the fold. No model was
> re-fit and no v1.x record was re-scored.

Metric implementation itself is **clean**: `macro_f1 == 1.0` implies
`auroc_defined` in **438/438** records; `macro_f1` has a hard floor of exactly
0.25 on the 540 records that are perfect single-class folds; the fixed-label
`zero_division=0` ceilings hold; `balanced_accuracy` minimum 0.5 is consistent.
The accuracy ≈ 0.985 vs macro-F1 ≈ 0.63 gap is the documented fixed-label-set
arithmetic on one-fillet folds, **not** a bug. Caveat: the per-fold class support
is *not* recoverable from the delivery line, because `class_coverage` lives only
in `results/raw/**`, which is git-ignored — a reviewer cannot verify the ceiling
claim from committed evidence alone.

## A1 / A7 — what is *not* broken (PASS, with two notes)

R0 and R2 are genuinely different objects with different bytes; the rewiring
reaches the fully-mixed configuration-model expectation (edge overlap 0.058 vs
0.055 expected at N=1000) and the ten seeds produce **10 distinct graphs**
(seed-vs-seed overlap ≈ 0.049, i.e. as independent as fully-mixed). Unweighted
in/out degree sequences are preserved **exactly**, edge count is preserved, no
self-loops, no duplicate coordinates.

Two notes:

1. **R2 is not a wiring-only control.** `make_degree_rewired` rebuilds the
   matrix with `np.ones(n_edges)`, so R2 is a *uniform-weight* graph: it destroys
   the synapse-count weight structure at the same time as it rewires. R0-vs-R2 is
   therefore a joint contrast, and the pair (R1, R2) is needed to separate the
   two nulls.
2. **The runner's default size is computationally infeasible.** The duplicate
   scan is O(M) per attempt, so cost per attempt = c·M with c ≈ 2.8e-9 s/edge
   (fitted from N ≤ 2000). At the full graph (M = 14,828,657, 10 attempts/edge =
   148,286,570 attempts) one R2 build projects to **71 days**, and a 10-seed
   sweep to **711 days**. Memory is worse: `_drive` materialises
   `input_projection = x @ w_in.T` of shape (n_windows, 16, 124,185) float64,
   which is **102 GiB** for a D2-sized training fold (6,435 windows). So a bare
   `run_reservoir_e2.py` invocation cannot produce E2 evidence: it either OOMs or
   never returns. This is consistent with the absence of any E2 record on the
   delivery line.

---

## A10/A11 — the delivered evidence, recovered from the GPU box

The delivery line contains **no E2 record at all**, which made the audit look
predictive rather than retrospective. It is not: the GPU box holds the runs. The
following was recovered read-only from
`/root/autodl-tmp/drososense/repo/.git` HEAD `afbd0e1a` and copied into
`results/audit/m4_audit/server_evidence/`.

| experiment | records | note |
|---|---:|---|
| `e1_main_d2` / `e1_main_d3` | 900 / 11,160 | the committed baseline sweeps |
| `e2_main_d2` | 700 | R0–R6 × 5 folds × 10 seeds × 2 tasks |
| `e2_main_d3` | 8,680 | R0–R6 × 62 folds × 10 seeds × 2 tasks |
| `e9_size_d2` (+ n250…n4000) | 420 | 6 × 70 |
| `e2_smoke` | 7 | |
| **total** | **21,867** | matches the "≈21,790 records" figure the project was tracking |

**A10 — the substrate the E2 batch actually used is the one A8 measured.** From
`R0/classification_seed00_fold00.json`:
`topology.n_nodes = 250`, `n_edges = 126`, `density = 0.002016`,
`normalization = "n1_pre_l1"`, `spectral_radius = 0.9000` — byte-for-byte the
A8 row for N=250. `params` = `leak 0.1, gain 0.5, input_scale 0.1, state_pooling
last, washout 10`, i.e. the exact regime A3 measured. R0 and R2 carry the
**identical** `w_in_sha256` (`f390c19b…`) and `bias_sha256` (`99532ba6…`), so the
only difference between them is the adjacency — as designed.

**A10 — the delivered R0-vs-Rx result, computed from those 8,680 records**
(620 paired units per contrast, classification):

| contrast | mean Δ macro-F1 | median | sd | units with **byte-identical** metric dicts |
|---|---:|---:|---:|---:|
| R0 vs R1 (weight-shuffled) | −0.00004 | 0.00000 | 0.00581 | 572 / 620 |
| **R0 vs R2 (degree-rewired)** | **+0.00010** | 0.00000 | 0.00632 | **570 / 620** |
| R0 vs R3 (random sparse) | +0.00118 | 0.00000 | 0.01300 | 563 / 620 |
| R0 vs R4 (ER-ESN) | +0.00043 | 0.00000 | 0.01082 | 562 / 620 |
| R0 vs R5 (small-world) | +0.00055 | 0.00000 | 0.01139 | 565 / 620 |
| R0 vs R6 (dense random) | +0.00012 | 0.00000 | 0.00937 | 561 / 620 |

Read the last column. **A dense random graph reproduces the real Drosophila
olfactory connectome's predictions exactly in 561 of 620 units.** So this is not
a near-tie between R0 and its matched control; it is the whole topology family
being interchangeable at the readout. That is precisely what a 5-dimensional
state manifold and a 3 % recurrent contribution predict, and it is measured on
the delivered data, not on the synthetic probe.

Two record-keeping defects in the same files: `params.reservoir_size` is **200**
while `topology.n_nodes` is **250** (the record's `params` come from
`DEFAULT_RESERVOIR_PARAMS`, not from the selection that was actually used), and
`selection` is `{}` — **the node-selection provenance (`target_n`, seed,
`sha256_sorted_root_ids`) is not recorded at all**. A reader cannot tell which
250 nodes the delivered results used.

**A11 — the E9 size study (D2, seed 0, 5 folds) does the opposite of what the
substrate story needs.** Recovered from the six `e9_size_d2*` directories:

| N | R0 macro-F1 | R2 macro-F1 | R1 | R3 | R4 | R5 | R6 | R0 MAE | R2 MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | **0.3141** | 0.3084 | 0.3075 | 0.3125 | 0.2846 | 0.2882 | 0.2942 | **1.690** | 1.611 |
| 500 | 0.3015 | 0.2676 | 0.2988 | 0.2503 | 0.2652 | 0.2978 | 0.2987 | 3.598 | 3.705 |
| 1000 | 0.2709 | 0.2747 | 0.2660 | 0.2388 | 0.2558 | 0.2486 | 0.2292 | 5.975 | 5.547 |
| 2000 | 0.2355 | 0.2432 | 0.2349 | 0.2454 | 0.2372 | 0.2288 | 0.2677 | 5.544 | 5.464 |
| 4000 | **0.2418** | 0.2213 | 0.2164 | 0.2207 | 0.2330 | 0.2153 | 0.2346 | **6.260** | 6.455 |

**Performance decreases monotonically with N** — macro-F1 0.314 → 0.242,
regression MAE 1.69 → 6.26. Adding connectome is actively harmful on this
design, which is consistent with the state's effective dimension being set by
the channel count (A2) while the ridge readout's feature count grows with N: the
extra directions carry no generalisable signal and the fixed `ridge_lambda =
1e-3` cannot suppress them at four training specimens. On this evidence the
"more biological substrate" axis is not merely flat — it is negative. **This is
the strongest single reason not to launch E9-D3 (43,400 runs): the D2 size study
already answers the question, in the wrong direction, and it has never been
committed or read.**

**A11 — the E9 study is 70 §17 violations by the project's own rule.** Running
`drososense.evaluation.results.test_touched_once_report` over the 420 recovered
records:

```
n_records = 420   n_with_fingerprint = 420
n_distinct_test_fingerprints = 70   n_repeated_test_fingerprints = 70   n_violations = 70
violation example: fingerprint a53bad3aa75975fb carries 6 config_hashes
  [2b681f4040fc, 45dea923f5eb, 5ef9f26a4c53, aeaa183639c5, e2526bc8a3c8, e2dbcef1dad6]
  all six run_ids = d2_beef_uncontrolled|R0|classification|seed00|fold01
```

The cause is a definition, not a bug in a loop:
`make_test_fingerprint(fold_fingerprint, window_length, model, task)` **omits
`reservoir_size` and `normalization`**. Every legitimate within-split design
variation — a size study, a normalization comparison, a low-data fraction —
therefore reuses the same fingerprint under a new `config_hash`, which is
*exactly* the pattern §17 defines as a violation, and `load_evidence_bundle`
refuses any bundle with `n_violations != 0` (exit 2). So:

- the E9 size study **cannot be analysed by the M4 pipeline at all** as defined;
- and the DATA-60 "cross-experiment anchor skip" that had to be written for E3
  is a downstream patch for this same root cause — which is why the E9 commit
  message itself reports a batch where "N=500/1000/2000/4000 all 70 skipped
  prior_ok_different_config".

**A11 — and E3 produced no results at all.** There is no `e3_*` directory under
`results/raw` on the box. What exists is five skip-disclosure receipts, one per
low-data fraction, each with `n_skipped_units = 20` and `n_skipped_same_config =
0` / `n_skipped_different_config = 0` — i.e. **100 units, 100 % skipped**, every
one with `reason: "prior_ok_cross_experiment_anchor"`, anchored to
`d2_beef_uncontrolled|svm_rbf|classification|seed00|fold00` under
`prior_config_hash = c84d501b1a72`, which is the **E1 baseline** batch's config
hash. So E3's low-data sweep silently anchored itself against E1 and did nothing;
the receipt is the only trace, and it is the third independent manifestation of
the fingerprint-omits-the-configuration root cause above. Any future summary
that reports E3 as "run" is reporting a config hash and a skip count, not data.

**A11 — the server's working tree is dirty and produced these results.** HEAD is
`afbd0e1a`, with **10 uncommitted modifications, +1,236 / −337 lines**, including`drososense/reservoir/runner.py` (420 lines changed), `drososense/data/splits.py`,
`drososense/data/pipeline.py`, `drososense/evaluation/runner.py` and both
CLI scripts. The full diff is captured at
`results/audit/m4_audit/server_evidence/prov.txt`. So the E2/E3/E9 numbers on the
server are attributable to **no commit**, and the delivered E1 numbers were
likewise produced from a tree with no `.git` at all (`ops/e1/README.md`). Before
any of it is used, the tree that produced it must be committed or hashed.

---

## A2r / A4r / A5r — the same questions, asked on real e-nose data (FAIL)

The synthetic probe is the right instrument for an integrity question, but two of
the audit brief's items ask for a real-data answer: the state divergence under
real windows (A2r) and the readout ablation on a real specimen-level validation
split (A4r). D2 and D3 were re-acquired locally (`scripts/download_data.py`; D3
via HTTP range requests, 210 members SHA-256-verified) and both were run with
**only `fold.train` and `fold.val` ever built into features** — the script reads
`fold.test` once, to assert it is a distinct object it does not use.

**A4r — D2, LOSO(5), N=250, 11 channels, aggregated over all 5 folds:**

| arm | macro-F1 | accuracy |
|---|---:|---:|
| **X_only** (raw flattened window) | **0.4788** | 0.5102 |
| H_only (reservoir state) | 0.3737 | 0.3861 |
| X_plus_H | 0.3916 | 0.4062 |
| X_plus_**shuffled**_H | 0.4899 | 0.5155 |
| zero_H | 0.4901 | 0.5193 |
| majority-class baseline | — | 0.3916 |

Three things follow, and none of them is "the readout might be bypassing the
reservoir":

1. **H_only is 0.105 macro-F1 *worse* than X_only.** The reservoir state carries
   *less* usable information about the label than the raw 16-step window it was
   computed from. A random projection of the same window would not do that; this
   is a lossy transform in the sense that matters.
2. **Adding H to X makes X worse**: 0.3916 vs 0.4788. The 250 state dimensions
   actively degrade the readout — they inject directions that the fixed
   regularization cannot suppress at the specimen-level split.
3. The brief's decisive test, `F1(X+H) ≈ F1(X+shuffled H)`, holds — **and the
   sign is inverted**: destroyed H (0.4899) scores *higher* than real H (0.3916).
   There is no sense in which the real states carry information that shuffled
   states do not.

(The `zero_H` arm is arithmetically X_only plus 250 all-zero columns, so their
0.4901 vs 0.4788 gap is the optimiser's numerical path, not a real difference —
recorded here rather than presented as signal.)

**A5r — the state's effective rank on real data.** Mean over the 5 D2 folds:
**6.575** for 11 real channels (top-1 PC 43 %). The synthetic probe established
the *mechanism* (rank tracks the channel count and is flat in N: 3.09/3.14/3.10
for C=3 across N = 250/1000/4000). Real channels are correlated and the tanh
saturates, so the realised rank is *below* C rather than equal to it — but it is
still 6.6 dimensions of state, not 250, and it does not grow with the
connectome.

**A2r — the divergence result replicates on real windows.** Same three-way
comparison as the synthetic probe, driven this time by real D2 validation
windows: d(R0,R2) = **0.0671**, d(R0,R3) = **0.1904**, d(R0, A := 0) =
**0.0338**. Deleting the connectome still perturbs the state about half as much
as rewiring it, on real data.

**A4r — D3 confirms the ordering** (`group_kfold`, 2 folds — an *audit-only*
split choice, because LOSO(62) leaves a single fillet in validation and cannot be
scored; 7 declared feature channels, N=250):

| arm | macro-F1 | accuracy |
|---|---:|---:|
| X_only | **0.9737** | 0.9729 |
| H_only | 0.9597 | 0.9629 |
| X_plus_H | 0.9606 | 0.9634 |
| X_plus_shuffled_H | 0.9720 | 0.9714 |
| zero_H | 0.9759 | 0.9765 |

Identical ordering to D2: X_only beats H_only by 0.0140, and X+H (0.9606) is
*worse* than X+shuffled-H (0.9720). D3's effective state rank is **3.211 for 7
channels** (top-1 PC **74 %**) — more compressed than D2, because the trout
channels are strongly collinear and the tanh saturates.

The A2r divergences are the *same on both datasets* to within 1 % — d(R0,R2) =
0.0694 (D3) vs 0.0671 (D2); d(R0,R3) = 0.2120 vs 0.1904; d(R0, A := 0) = 0.0350
vs 0.0338. **These numbers are a property of the design, not of the food or the
sensor array**, which is the clearest available statement that what M4-v1.4
measured was its own architecture.

**P1-3 — the "olfactory subgraph" has no olfactory input layer.** Composition of
the selected node set, from the committed
`connectome/metadata/olfactory_v1_node_meta.csv` with the tier mapping
`add_edge_masks.py` itself defines:

| N=250 | selected | vs graph | enrichment | in largest CC | isolated |
|---|---:|---:|---:|---:|---:|
| **ORN** | **0** (0.00 %) | 1.83 % | **0.00×** | **0** | 0 |
| PN | 41 (16.40 %) | 4.31 % | 3.80× | 13 | 8 |
| KC | 18 (7.20 %) | 2.91 % | 2.47× | 4 | 13 |
| MBON | 27 (10.80 %) | 7.77 % | 1.39× | 4 | 16 |
| DAN | 12 (4.80 %) | 2.07 % | 2.32× | 3 | 5 |
| higher_order | 16 (6.40 %) | 2.55 % | 2.51× | 7 | 2 |
| other | 136 (54.40 %) | 78.56 % | 0.69× | 6 | 118 |

ORN count is **0 at N=250, 1 at N=1000, 19 (0.47 %) at N=4000**; at N=4000 every
class's enrichment collapses to ≈1.0× (the degree-proportional selection becomes
near-uniform), while ORN is still under-represented. The largest weak component at
N=250 holds **37 of 250 nodes**, and it too contains no ORNs. So the olfactory
input pathway is missing **twice**: at the input stage (A6 — a dense random
`W_in` over every node, no ORN/PN population) and in the substrate (here — the
one place biology enters, the adjacency, has had its input layer sampled away).



## v1.5 — Evidence Identity Fix (delivered)

**Scope statement, in the amendment's own words:** *v1.5 changes evidence-unit
identity only; no model, dataset, split, endpoint, or statistical decision rule
is altered.* It is a pipeline repair, not a model repair, and it is explicitly
**not** a licence to re-read any existing result.

### What changed

| Artefact | Change |
|---|---|
| `configs/protocol_v1.5.yaml` + `.sha256` | New frozen amendment record (`supersedes: 1.4.0`). v1.4, v1.3, v1.2, v1.1 and v1 stay byte-identical, each still matching its own sidecar. The loadable base protocol is still **v1.3**. |
| `drososense/evaluation/results.py` | The identity is now a versioned **named tuple** (schema 2): `dataset, task, fold_fingerprint, model, window_length, condition, reservoir_size, normalization, input_mapping, topology_variant, rewire_seed`, digested from canonical sorted JSON with the schema tag **inside** the digest. The schema-1 definition is kept verbatim as `legacy_test_fingerprint`. `RunRecord` gains an optional `evidence_unit` payload (`schema`, `components`, `id`, `legacy_id`). |
| `record_path` | Records are now **addressable by evidence unit** (the unit id is part of the filename). |
| Both runners | Use the new identity, carry it on every record, and consult the ledger through a new asymmetric lookup. `PROTOCOL_VERSION = "1.5.0"`. `ReservoirConfig` gains a `condition` field, part of `config_hash`. |
| `connectome_reservoir.py` | Declares `INPUT_MAPPING_DENSE_RANDOM = "dense_random_all_nodes"` as a named constant and reports it in `ReservoirShared.describe()`, so the v2 constrained-input mapping cannot collide with it. |

`condition` is the protocol's own experiment-condition axis (`full` /
`train10pct` / `dropout_p0.3` … — the label the multiplicity families are already
keyed on). It was added beyond the brief's list because it is the **E3 half** of
the defect: `E3_lowdata` declares that the test split stays fixed across training
fractions, so without the axis its four fractions are either skipped as a prior
touch (which is what actually happened) or recorded as §17 violations — neither
of which is what the protocol declares.

### Two things the tests caught, which changed the design

Both were found by the tests written for this change, not by review, and both are
recorded rather than quietly fixed:

1. **The legacy alias has to be asymmetric.** The first implementation registered
   every identity a record could be filed under, in one ledger. That made two
   *v1.5* substrates of one unit collide again through their shared schema-1
   alias — the fix would have changed nothing. The ledger is now split:
   **current** (schema-2 ids, populated only from v1.5 records) and **legacy**
   (schema-1 fingerprints, populated only from pre-v1.5 records, consulted only
   as a fallback). A v1.5 record occupies exactly one identity.
2. **Identity without addressability is not identity.** With the fingerprint
   fixed but the record path unchanged, scoring a second substrate of one unit
   silently **overwrote** the first record on disk. That is why the old size
   study had to invent one experiment label per size (`e9_size_d2_n250` …), which
   is itself what produced the colliding fingerprints. `record_path` now carries
   the unit, and one consequence is deliberate: a pre-v1.5 *failed* record is no
   longer clobbered by a later run (a failed record is not evidence), and the
   crashed-run test asserts the new behaviour instead of the old side effect.

### Guard tests — `tests/test_evidence_identity_v1_5.py` (26) + the synthetic ledger test

| Required case | Test |
|---|---|
| same config → same fingerprint | `test_same_components_give_the_same_identity`, `test_identity_is_independent_of_argument_order_and_of_extra_names` |
| `reservoir_size` 250 → 500 → different | `test_reservoir_size_change_changes_the_identity` (4 size pairs), `test_the_e9_size_study_no_longer_collides` |
| `normalization` n0_raw → n1_pre_l1 → different | `test_normalization_change_changes_the_identity` (4 pairs) |
| same unit re-run → §17 violation | `test_failed_run_does_not_occupy_the_touch_quota_and_ok_run_does` |
| `rewire_seed` 0 → 1 | `test_rewire_seed_change_changes_the_identity` + `test_reservoir_runner_pins_the_rewiring_seed_to_the_run_seed` |
| failed run does not occupy the quota (v1.4 not regressed) | same test, plus the updated `test_a_crashed_run_does_not_register_test_touched_once` |
| (added) `input_mapping` / `topology_variant` / `condition` are identity-bearing | `test_input_mapping_is_part_of_the_identity`, `test_topology_variant_is_part_of_the_identity`, and the condition axis in `test_v1_5_ledger_separates_substrates_and_still_blocks_true_repeats` |
| (added) a v1.5 re-run of an already-scored **legacy** unit is still refused | `test_rerunning_a_legacy_unit_under_v1_5_is_still_refused` |
| (added) the report does not re-interpret legacy rows | `test_report_separates_identity_schemas_without_reinterpreting_legacy_rows` |
| (added) v1.5 is frozen, sidecar-matched, and earlier files are untouched | `test_protocol_v1_5_is_a_frozen_amendment_with_a_matching_sidecar`, `test_earlier_protocol_files_are_byte_frozen`, `test_v1_5_does_not_change_the_loadable_base_protocol` |

**The synthetic ledger test** (`tests/test_reservoir_runner.py::test_v1_5_ledger_separates_substrates_and_still_blocks_true_repeats`)
drives the real runner on the synthetic fixture end to end and asserts the three
things at once: a different substrate (size / normalization / condition) is
scored as a **new** unit — not a violation, not a skip; an identical re-run is
still skipped; and the schema-1 fingerprints those units would have shared are
recorded so the defect cannot be quietly reintroduced. The fixture NPZ gained a
second stored normalization (`n5_binary`) so the normalization axis is exercised
on a real load path.

### Existing-record dry-run — `ops/audit/evidence_ledger_repair_audit.py`

Read-only over the recovered GPU-box evidence; starts no run.

| Experiment | Finding |
|---|---|
| **E9 size study** | 420 ok records → 70 legacy fingerprints → **350 evidence units**. Of those, **280 are a collision only** (`valid_but_unreadable_under_old_identity`: N = 500/1000/2000/4000 differ from R0's own substrate and from each other, which schema 1 could not name) and **70 are a true repeated test evaluation** (`invalid_experimental_evidence`: **N = 250 was scored twice under two different config hashes**, in `e9_size_d2` and `e9_size_d2_n250`). Zero same-config recomputations. |
| **E3 low-data** | **No raw records at all.** Five receipts covering 100 units, all skipped, all `prior_ok_cross_experiment_anchor`. Classification: `no_evidence_produced` — there is nothing here to call valid or invalid. |
| **The blocking dependency** | An E3-style v1.5 unit (`condition=train10pct`, N=250) is **still matched** by the legacy fallback against the pre-v1.5 E2/D2 records. So **v1.5 alone does not unblock E3**: the pre-v1.5 records must be classified and re-filed under their own components first. The conservative direction is deliberate — a re-run is refused, never silently re-scored. |

So the answer to "do the old E9/E3 records become valid evidence?" is **no, not as
a block**: 280 E9 units are recoverable by re-filing with their own components,
70 E9 units are permanently inadmissible (§17 violations that no re-indexing can
cure), and E3 has produced no evidence to admit. Nothing was re-indexed,
re-labelled or promoted by this amendment.

### Not done here, on purpose

No reservoir-architecture change, no re-run of E9 or E3, no edit to any frozen
protocol file, and no attempt to make anything look better. The audit's headline
finding is unchanged and unaffected by v1.5: **M4-v1.4 is not a valid
olfactory-connectome test**, and repairing the evidence system does not repair
that — it only makes the next test possible to believe.



## Locked wording: this is a construct-validity failure, not a negative result

M4-v1.4's outcome must **not** be described as a negative result. The evidence is
that the experiment did not test the hypothesis it was designed to test:

| evidence | measurement |
|---|---|
| the substrate's input layer | **ORN = 0** at N <= 1000, 0.47 % at N=4000 (P1-3) |
| the substrate's structure | 126 edges, **64.8 % isolated**, 0.09 % of out-edges retained (A8) |
| whether the topology enters the computation | recurrent term = **3-5 %** of the drive; memory survives A := 0 bit-identically (A3) |
| whether the state carries usable signal | **H worse than X** (0.374 vs 0.479), and X+real-H (0.392) worse than X+**shuffled**-H (0.490) (A4r) |
| whether the input pathway is biological | denserandom `W_in` over every node; dynamics invariant to relabelling nodes (A6) |

None of those is "the model did not win". Each is a failure to operationalise the
construct — *a biologically organised olfactory substrate* — so the correct
classification is:

> **M4-v1.4 is an invalid test of the intended biological hypothesis
> (construct-validity failure), not a negative result about it.**

This distinction matters for how the v2 redesign is reported. v2 is **not**
re-tuning after seeing a test result; it is repairing a construct-validity failure
that an independent audit established from the implementation and the delivered
records, before any v2 design decision was taken. That is the difference that has
to survive into the TAFE methodology and development-history sections.

## What this does and does not licence

**Does licence:** the statement that the delivered M4-v1.4 pipeline, at its
current design, cannot test the project's hypothesis — because the substrate it
evaluates is a 126-edge residue (A8), the recurrence contributes 3–5 % of the
drive (A3), the state space is ~5-dimensional by construction (A2/A6), and the
decisive test clusters the wrong unit (A9). **A future null on this design
would be a statement about the design, not about the Drosophila olfactory
connectome.**

**Does not licence:** any claim that the biological hypothesis is false, and any
claim that a fix will produce a positive result. If, after the fixes below, R0 is
still ≈ R2, that is the accepted answer.

## Recommended disposition

| Track | Content | Touches data? |
|---|---|---|
| **v1.5 (amendment)** | **DELIVERED — evidence identity, parameter scope, gate semantics and the cluster unit.** (a) the clustering unit is declared as the **specimen**, not `fold_id` (A9 / D7 / v1.5.3 — CLOSED, see the A9 section above); the remaining open sub-item is (c) record the node-selection provenance and the true `reservoir_size` in the run record (`params.reservoir_size = 200` while `topology.n_nodes = 250` is a false statement in the delivered evidence) and commit `class_coverage`. No dataset, split, endpoint or decision rule changes. | No re-run |
| **v2 (re-operationisation)** | Architecture: an input stage that maps e-nose channels onto a **biological input population** (ORN→PN→KC/higher-order) rather than a dense random `W_in` over all nodes; an induced subgraph that is **connected and edge-retaining** (or a sub-sampling scheme that preserves in-block edges); a knob regime in which the recurrence is not 3 % of the drive; R2 made weight-preserving so it is a pure wiring control. Re-pre-registered before any test is touched. | Validation/synthetic first |
| **Housekeeping** | Commit or hash the tree that produced the E2/E3/E9 numbers (10 uncommitted files on the GPU box, full diff preserved at `results/audit/m4_audit/server_evidence/prov.txt`), and bring the E2/E9 evidence onto the delivery line. | No re-run |

Both are honest to report in a TAFE submission as *"the v1 implementation did not
operationalise the hypothesis"* — which is a methods finding, not a
result-driven re-tuning, provided the v2 design is frozen before its test run.

**Preservation.** All existing M4 artefacts stay on disk under their own labels
(`protocol_version 1.4.0`, `M4-v1.4 / original implementation`). If a v1.5 is
frozen, v1.4 records are marked `superseded_by_v1.5`, are never pooled into new
statistics, and get an independent test-touch ledger. Nothing is deleted or
overwritten.

**Do not launch E3/E6/E9 until the above is adjudicated.** Every additional run
at the current design adds cost, not evidence.

---

## Reproducing this audit

```bash
# graph-level: A1, A7, A8  (real NPZ; resolves via $DROSOSENSE_DATA or ../data-root)
python ops/audit/reservoir_integrity_audit.py --only A1 A7 A8 \
    --sizes 250 500 1000 2000 4000

# dynamics: A2, A3, A4, A5, A6  (synthetic input; A4 uses the fixture's train+val only)
python ops/audit/reservoir_dynamics_audit.py --only A2 A3 A6 --sizes 250 1000
python ops/audit/reservoir_dynamics_audit.py --only A5 --sizes 250 1000
python ops/audit/reservoir_dynamics_audit.py --only A4 --a4-size 250
```

Evidence JSON: `results/audit/m4_audit/` — `A1_r0_r2_adjacency_identity.json`,
`A2_hidden_state_divergence.json`, `A3_recurrent_vs_input.json`,
`A4_readout_ablation.json`, `A5_normalization_sensitivity.json`,
`A6_input_mapping.json`, `A7_r2_rewiring_quality.json`,
`A8_subgraph_degeneracy.json`.

### Audit-discipline statement

- The test split is never read: A2/A3/A5/A6 use a seeded synthetic AR(1) input
  with no dataset at all; A4 asserts it reads only `fold.train` and `fold.val`.
- No file under `configs/` or `drososense/` was modified. The audit adds two new
  read-only scripts under `ops/audit/` and writes only under
  `results/audit/m4_audit/`.
- One **measurement** error was found and corrected mid-audit and is recorded
  here rather than quietly fixed: the first A7 seed-distinctness digest hashed
  only the row degrees (`indptr`/row sums), which a degree-preserving rewiring
  holds fixed by construction, and therefore reported "1 distinct graph over 10
  seeds". With a wiring-based digest (column indices) it is **10 of 10**. The
  corrected run is the one reported above.
- A4 is reported as **inconclusive** rather than as a result: all five arms
  scored identically (macro-F1 0.2259, accuracy 0.4190) with the classifier
  collapsing to two classes below the majority baseline (0.5048), because the
  synthetic fixture's four-class labels are not learnable from a single 16-step
  window under a specimen-level split. That arm must be re-run on real
  validation data before it can say anything about the readout.
