# M4 Phase 2 — statistics / gating pipeline manifest (DATA-56)

This page is the single manifest for the M4 Phase 2 pipeline. Every input,
command, exit code, output path and line count is recorded here so a re-reader
can reproduce every number without re-deriving them.

## What this pipeline does

Read-only statistics and gate evaluation over committed evidence bundles. The
pipeline never evaluates a model; it reads already-scored tidy per-run records
and applies the frozen protocol's own statistical machinery (the cluster-level
sign test as the decisive test, the fold-cluster bootstrap, Holm within the
declared multiplicity families, and the gate / narrative-rule expressions
verbatim from the frozen protocol file).

The one command:

```bash
python -m drososense.evaluation.evidence_stats --experiment e1_main_d2
```

produces six artefacts under `results/tables/e1_main_d2/` (one directory per
bundle). The D3 / E2 plug-in point is the `--experiment` label, not a code
change: the same command reads `e1_main_d3_*` when the D3 bundle files exist,
and `e2_topology_*` when the E2 / E1-reservoir half lands.

## Inputs (frozen evidence, on the delivery line)

| file | sha256 (first 16) | rows |
|---|---|---|
| `results/tables/e1_main_d2_summary.csv` | `c8369331a00c1cdb` | 19 (header + 18) |
| `results/tables/e1_main_d2_per_run.csv` | `1e796ccb85a5bb86` | 901 (header + 900) |
| `results/tables/e1_main_d2_fingerprints.csv` | `933f3850ca507e96` | 901 |
| `results/tables/e1_main_d2_test_touched_once.json` | `345f7fcaa74faca8` | 900 distinct / 0 violations |
| `results/tables/model_parameters.json` | (committed) | `params(...)` source for Gate_A |
| `configs/protocol_v1.3.yaml` (active, `PROTOCOL_PATH`) | `d85640e556db6c00` | every test parameter |

The pipeline verifies all four D2 files at intake (`load_evidence_bundle`):
existence, unique `(dataset, model, task, seed, fold)` key, and
`n_violations == 0` in the touched-once report. Any failure is a `ValueError`
or `FileNotFoundError` (exit 2), not a silent degradation.

## One command — exit codes and output

| command | exit | output |
|---|---|---|
| `python -m drososense.evaluation.evidence_stats --experiment e1_main_d2` | 0 | six files under `results/tables/e1_main_d2/` |
| `python -m drososense.evaluation.evidence_stats --experiment e1_main_d3` (bundle absent on this tree) | 2 | error on stderr naming the four missing files |
| `python -m drososense.evaluation.evidence_stats --experiment e1_main_d3 --tables-dir results/tables/stub_e1_main_d3_demo --bundle-prefix stub_e1_main_d3` | 0 | six files under `results/tables/e1_main_d3/` (STUB demo, see below) |
| `python scripts/evidence_audit.py --experiment e1_main_d2` | 0 | `results/tables/e1_main_d2/evidence_audit.csv` |

### D2 run — recorded output (this tree)

| artefact | data rows | key content |
|---|---|---|
| `results/tables/e1_main_d2/descriptive.csv` | 18 (9 models × 2 tasks) | decision metric + secondary columns, `n_auroc_defined`, fold-cluster bootstrap CI |
| `results/tables/e1_main_d2/contrast_statistics.csv` | 12 (6 contrasts × 2 tasks) | all 12 `unpairable` (the E1 baseline bundle carries no R0–R6 reservoir model; `R4` resolves to `esn` which IS present, but `R0` is not) |
| `results/tables/e1_main_d2/paired_differences.csv` | 1 note row | "no declared contrast pairs on the common scored units of this bundle; the reservoir family (R0–R6) has not run here yet — reported, not imputed" |
| `results/tables/e1_main_d2/gates.json` | 3 gates | all `UNEVALUABLE` with reasons; `unresolved_terms` lists the exact pending set |
| `results/tables/e1_main_d2/narrative.json` | 5 rules | N1–N4 unfired (contrast terms unresolved), N5 unfired (all datasets available per manifest) |
| `results/tables/e1_main_d2/audit.json` | — | `n_per_run=900`, `n_violations=0`, `n_distinct_config_hashes=4`, `r2_secondary_only=true` |
| `results/tables/e1_main_d2/evidence_audit.csv` | 26 | claim → evidence mapping for the frozen hypotheses / gates / narrative rules / contrast statuses |

The six output artefacts, one per bundle label:

| artefact | content |
|---|---|
| `descriptive.csv` | per-(dataset, model, task): decision metric (the task's PRIMARY — `macro_f1` maximize / `mae` minimize), mean / SD, `n_folds`, the fold-cluster bootstrap 95 % CI, and the secondary-metric columns the task declares (`r2` / `rmse` — reported, never decided on). `n_auroc_defined` / `n_auroc_undefined` accompany every AUROC mean (M4 hard constraint 1). |
| `contrast_statistics.csv` | every declared contrast of the frozen protocol, evaluated where both sides pair on the common `(seed, fold, window)` units; `unpairable` and `insufficient_data` rows are written out with a reason, never dropped. Under protocol v1.5.3 the pair key also carries the **test specimen**, which is the cluster unit (D7): a specimen the two sides did not score identically makes the contrast `unpairable` with the specimen named, and `insufficient_data` reports the cluster count as well as the pair count. The `family` column names the owning multiplicity family; `p_holm` is the Holm-corrected decisive p within that family. |
| `paired_differences.csv` | the per-`(seed, fold)` deltas for every pairable declared contrast, the audit trail the report needs next to the summary statistics. |
| `gates.json` | Gate A / B / C verdicts. A gate that names a contrast the bundle does not carry is recorded `UNEVALUABLE` with the reason (protocol §13 gate_rules) — never a pass, never a fail. `unresolved_terms` lists the exact pending set for the next bundle. |
| `narrative.json` | N1..N5 evaluation on the frozen triggers. More than one firing is reported as a conflict, never resolved by choice (protocol §14 evaluation_note). N5's dataset-availability term reads the dataset manifest, not whether this bundle carried the dataset. |
| `audit.json` | bundle sha256 prefixes, row counts, `n_distinct_config_hashes`, the decision-metric map, the `r2`-secondary-only note, `gates_unevaluable`, and `missing_required_models` when `--require-models` was given. |

## Direction constraints (verified by the tests, not the promise)

- decision metric = `tasks.<task>.metrics.primary`, read from the frozen
  protocol; `r2` appears only under `tasks.<task>.metrics.secondary` and is
  excluded from every `hypotheses.*.metric` / `contrasts[*].metric` (the three
  guard tests in `tests/test_protocol_and_manifests.py` stay green — 477 passed
  / 17 skipped on this tree).
- `r2` is never fed to the paired-contrast layer, which takes the task's
  primary metric only; it appears in `descriptive.csv` as a secondary column
  with its own mean / SD / n, and nowhere else.
- `summarize.py` and the committed E1 tables are untouched: the pipeline reads
  them and writes only under `results/tables/<experiment>/`, so the sha256
  anchors recorded in `ops/e1/README.md` are not rewritten.

## The D3 / E2 plug-in point (one label, no code change)

The pipeline is parameterised by the `--experiment` label. When D3 closes and
its four-file bundle (`e1_main_d3_summary.csv` / `_per_run.csv` /
`_fingerprints.csv` / `_test_touched_once.json`) lands under
`results/tables/`, the command

```bash
python -m drososense.evaluation.evidence_stats --experiment e1_main_d3
```

is the entire D3 switch-over: identical code path, no new module, no
per-dataset special case. The four hard constraints from the M4 Phase 1
manifest apply unchanged to the D3 output (AUROC coverage reported with
`n_auroc_defined`; no cross-dataset macro_f1 magnitude comparison; fold-level
paired comparisons only; `config_hash` is batch-scoped — the audit records
`n_distinct_config_hashes` so no reader writes "one config for all 11,160
records" when the batch scope is what the DATA-48 ruling says it is).

The stub demonstration bundle at
`results/tables/stub_e1_main_d3_demo/` (files marked `STUB` in `run_id`,
`test_specimens_joined` and the touched-once note) is the pre-registered
shape check for that plug-in point: it exercises the pairable-contrast,
Holm, gate-UNEVALUABLE and narrative paths with the same command, and it is
excluded from every delivery table and from any gate verdict.

```bash
python -m drososense.evaluation.evidence_stats \
  --experiment e1_main_d3 \
  --tables-dir results/tables/stub_e1_main_d3_demo \
  --bundle-prefix stub_e1_main_d3
```

### Stub run — recorded output (this tree)

Run on the delivery line, exit 0:

| artefact | rows / content |
|---|---|
| `results/tables/e1_main_d3/descriptive.csv` | 6 data rows: 3 models × 2 tasks |
| `results/tables/e1_main_d3/contrast_statistics.csv` | 12 data rows: 4 pairable (`R0_vs_R2` / `R0_vs_R4` × 2 tasks, `status=ok`) + 8 `unpairable` (the `R3` / `GRU` / `R5` / `R1` sides the stub does not carry) |
| `results/tables/e1_main_d3/paired_differences.csv` | 40 data rows: 4 pairable contrasts × 5 folds × 2 seeds, one row per `(seed, fold)` delta |
| `results/tables/e1_main_d3/gates.json` | Gate A / B / C all `UNEVALUABLE` with the reason (the stub carries no D2 dataset, and the reservoir contrasts it does carry are on D3 only) |
| `results/tables/e1_main_d3/narrative.json` | N1–N4 unfired (contrast terms unresolvable on the stub), N5 unfired (manifest says all datasets available) |
| `results/tables/e1_main_d3/audit.json` | `n_per_run=60`, `n_violations=0`, `n_distinct_config_hashes=4`, `bundle_prefix=stub_e1_main_d3`, `r2_secondary_only=true` |

> **v1.5.3 re-run note (the artefact above predates it and is retained).** The stub
> labels every row `test_specimens_joined = "STUB"`, so under protocol v1.5.3 (cluster
> unit = specimen) the `R0_vs_R2` / `R0_vs_R4` rows are **one cluster**, not five:
> they re-run as `insufficient_data` with *"1 cluster(s); the cluster bootstrap cannot
> resample fewer than two; 10 paired observation(s) in 1 specimen cluster(s)"*. The
> committed `ok` rows with `n_clusters = 5` are the v1.x **fold-index** clustering and
> are not comparable. The stub still exercises the paths it was written for; the
> demonstration that matters is the D2/D3 one in
> [`docs/audit_index.md`](audit_index.md) §5b, on real specimen labels.

Expected shape of the D3 output when the real bundle lands (values aside):
12 contrast rows (6 declared contrasts × 2 tasks × 1 dataset); `unpairable`
rows for any reservoir contrast whose two models the D3 bundle does not
carry; `p_value` reachable at `2/2^62` for the cluster sign test (D3 =
LOSO(62), so the `minimum_achievable_p_over_clusters` term of `sig` is
satisfied, unlike D2's five-cluster floor of 0.0625). The real D3 run is the
one command above; the stub run is the shape check only — its values are
synthetic and it is excluded from every delivery table and gate verdict.

## Pending (must land before this page's pending rows resolve)

- [ ] D3 bundle closure (`e1_main_d3_*` four files, 11,160 records) → run the
      one command with `--experiment e1_main_d3`;
- [ ] E2 / E1-reservoir formal batch (R0–R6, `n0_raw`, seeds 0..9, D2 first
      then D3; see DATA-50 runner and the e2_smoke_rerun void ruling) → the
      same command with `--experiment e2_topology` (or the label the batch
      uses) produces the Gate A / B / C and N1..N5 verdicts on the reservoir
      side;
- [ ] the `sig_any(...)` robustness terms of Gate B / C and of N2 read the
      E3 / E4 / E5 condition bundles (`dropout_p0.3`, `noise_s0.1`,
      `train10pct`, `train25pct`) — those condition bundles do not exist yet,
      so the `sig_any` clause reports its term unresolved until E3–E5 land.

## N1..N5 — what each rule is, what it needs, and what is computable today

The five narrative-adjustment rules are the frozen trigger → action pairs in
the protocol (§14). The pipeline evaluates each rule's trigger expression on
the bundle's own contrasts; a rule whose trigger names a contrast the bundle
does not carry is reported unfired with the reason, never silently skipped.

| rule | trigger | what it concludes | computable on `e1_main_d2` today? |
|---|---|---|---|
| N1 | `equiv(R0, R2, macro_f1, D3) and sig(R0, R3, macro_f1, D3)` | degree/statistical structure matters, not the specific Drosophila wiring; the title/abstract must be rewritten accordingly | No — needs the D3 contrast table (R0 vs R2 equivalence + R0 vs R3 significance on `d3_rainbow_trout`). Both contrasts are `unpairable` on the D2 E1 baseline bundle. |
| N2 | `equiv(R0, R4, macro_f1, D3) and sig_any(R0, R4, macro_f1, [D2, D3], [dropout_p0.3, noise_s0.1])` | reframe as a robust sparse biological reservoir contribution, drop the performance claim | No — needs the D3 equivalence term plus the E4/E5 condition bundles (`dropout_p0.3`, `noise_s0.1`), which do not exist yet. |
| N3 | `sig(R4, R0, macro_f1, D3)` | the negative-result branch: report that the biological connectome does not beat the matched ESN; reframe as a computational-neuroscience result | No — needs the D3 R0 vs R4 contrast table. |
| N4 | `(sig(R0, R2, macro_f1, D2) and sig(R2, R0, macro_f1, D3)) or (sig(R2, R0, macro_f1, D2) and sig(R0, R2, macro_f1, D3))` | cross-food consistency check: claim cross-food generalization only if the ordering holds on both foods; otherwise report the inconsistency | No — needs both the D2 and D3 R0 vs R2 contrast tables. Note: D2's `sig` term is arithmetically unreachable on its five clusters (floor 0.0625 > alpha), so N4's D2 clause can never fire; the rule is only reachable via the D3 clauses. |
| N5 | `unavailable(D1) or unavailable(D2) or unavailable(D3)` | report the blocker with the exact manual acquisition steps; mark the affected dataset UNAVAILABLE in every table | Yes — the pipeline reads dataset availability from the dataset manifest (not from whether this bundle carried the dataset, per the protocol's own §14 note), so N5 is evaluable on any bundle. On the current manifest state all three datasets are available, so N5 does not fire. |

The `narrative.json` artefact records, for each rule on each bundle, whether
it fired and the exact reason it did not, so a reviewer can see the pending
set per rule without re-deriving it.

## Red lines observed

- `configs/protocol_v1.{1,2,3,4}.yaml` untouched (byte-frozen; no in-place
  edit, no `as_dict()` change; `protocol_v1.3` is the active file the
  pipeline reads via `PROTOCOL_PATH`);
- no new D2 / D3 evaluation this round (read-only analysis of committed
  records; the stub is the only synthetic data and it is marked `STUB`);
- `r2` stays secondary-only;
- every number in this page is a function of the files in the table above,
  reproducible by the one command above on this tree.
