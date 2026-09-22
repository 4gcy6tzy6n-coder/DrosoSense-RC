# M4 Phase 1 — single-manifest pipeline (DATA-5)

Everything this page names is reproducible from this file alone. Phase 1 is
read-only over already-scored records: it never evaluates a model, and it must
not be read as "M4 complete" — the E2 formal batch, D3, and the narrative
gates are the pending items listed below.

## Inputs (frozen evidence, committed on the merge line)

| file | sha256 (first 16) | rows |
|---|---|---|
| `results/tables/e1_main_d2_summary.csv` | `c8369331a00c1cdb` | 19 (header + 18) |
| `results/tables/e1_main_d2_per_run.csv` | `1e796ccb85a5bb86` | 901 (header + 900) |
| `results/tables/e1_main_d2_fingerprints.csv` | `933f3850ca507e96` | 901 |
| `results/tables/e1_main_d2_test_touched_once.json` | `345f7fcaa74faca8` | 900 distinct / 0 violations |
| `results/tables/model_parameters.json` | (committed) | `params(...)` source |
| `results/tables/data_contact_log.json` | (committed) | freeze-time contact log |

The four D2 files are verified by the pipeline itself (`load_evidence_bundle`:
existence, unique `(dataset, model, task, seed, fold)` key, `n_violations == 0`).

## One command

```bash
python -m drososense.evaluation.e2_stats --experiment e1_main_d2
```

Produces, under `results/tables/e1_main_d2/`:

| artefact | content |
|---|---|
| `descriptive.csv` | per-(model, task): decision metric (`classification` → `macro_f1` maximize; `regression` → `mae` minimize), mean ± SD, 95% percentile CI over the 50 (seed, fold) record means (B = 10000, bootstrap seed 20260920 — the protocol's own spec, so the interval is reproducible without re-running the experiment), secondary columns (`r2` / `rmse` reported, never decided on) |
| `contrast_statistics.csv` | every declared contrast of the frozen protocol; rows the bundle cannot carry are reported `unpairable`, never imputed |
| `paired_differences.csv` | per-`(seed, fold)` deltas where pairable; a single stated row when nothing is pairable (the E1 baseline bundle carries no R-family). The exploratory baseline-vs-baseline deltas computed on this bundle are published in the reply, marked exploratory, and feed no gate |
| `gates.json` | gate evaluation: a gate naming a contrast the bundle does not carry is recorded **UNEVALUABLE** with the reason — never a pass, never a fail (protocol §13 gate_rules) |
| `pipeline.audit.json` | bundle sha256s, row counts, decision-metric map, `r2`-secondary-only note |

The same command with `--experiment e1_main_d3` (and the D3 bundle present)
is the entire D3 switch-over: no code change.

## Direction constraints (verified by the code, not the promise)

- decision metric = `tasks.<task>.metrics.primary`, read from the frozen
  protocol; `r2` appears only under `tasks.<task>.metrics.secondary` and is
  excluded from every `hypotheses.*.metric` / `contrasts[*].metric`
  (the three guard tests in `tests/test_protocol_and_manifests.py` stay green);
- `r2` is never fed to `paired_frame` / `paired_stat_row` — those take the
  task's primary metric only;
- `summarize.py` is untouched (its committed columns and the sha256 anchors
  above are not rewritten by this pipeline).

## Gate_A — D2-side computable part

From the committed bundle alone (read-only):

| item | state |
|---|---|
| Protocol freeze chain (v1.3 active `d85640e556db…`, v1.1/v1.2/v1.4 sidecars) | `python -m drososense.utils.protocol --check` → freeze OK |
| §17 on the D2 bundle | `n_violations == 0` over 900 distinct test fingerprints (verified from the JSON, rule quoted) |
| §17 prior-touch disclosure (D2 side) | two exploratory smoke batches (`e2_smoke` n0_raw, `e2_smoke_rerun` n5_binary) each touched 7 D2 units before v1.4 semantics landed; `e2_smoke_rerun` is declared **void** by Mika's 11:40Z ruling and excluded from every analysis/claim/gate here; the formal E2/M4 batch runs `n0_raw`, which per the v1.4 text is a same-config re-computation, not a second touch |
| D2 descriptive layer | `descriptive.csv` above (18 model×task rows, all ok) |
| `params(GRU)` | committed table: 4452 (maximum tuned configuration, selection rule recorded in `model_parameters.json`) |
| `ci_contains_zero(R0, R4, macro_f1, D2)` / `noninferior(...)` | **pending** — the `R0_vs_R4` contrast needs the formal E2 batch (`R0`/`R4` = `esn` are not in the E1 baseline bundle; `contrast_statistics.csv` reports them `unpairable`) |
| `params(R0)` | **pending** — R0 is not run; the committed `model_parameters.json` has no entry, and the pipeline's audit records `unresolved_params_terms` rather than inventing a count |

So the **computable part of Gate_A is done** (freeze chain, §17, prior-touch
disclosure, GRU side of the efficiency term); the R0-dependent terms stay
pending until the E2 formal batch runs. Gate_A PASS/CONDITIONAL is recorded in
the reply comment, not here.

## D3 switch-over — the four hard constraints

When the D3 bundle (`e1_main_d3_*`, written by `ops/e1/summarize_per_run_when_idle.sh`
after closure) lands, the pipeline ingests it with only the label changed,
subject to:

1. **AUROC coverage**: every AUROC number is reported together with
   `n_auroc_defined` (folds where all four classes are present;
   `auroc.empty_class_policy` is `require_all_classes`). The descriptive layer
   inherits the summary's `n_auroc_defined` column rather than pooling.
2. **No cross-dataset / cross-protocol macro_f1 magnitude comparison**: D3 is
   a fixed four-class label set with absent classes scored 0
   (`zero_division=0`), so D3 macro_f1 values are not comparable in magnitude
   with D2 (all four classes present per fold) or with the m1 D3
   GroupKFold(5) table; any side-by-side must carry the class-coverage
   distribution of both tables.
3. **Fold-level paired comparisons only**: decisions travel on the
   cluster-level paired test (one mean per held-out block; the fold is the
   resampling unit). No conclusion is drawn from a pooled mean across
   coverage groups.
4. **`config_hash` is batch-scoped**: each of D2/D3 carries exactly 4
   `config_hash` values (one per fan-out invocation, including the `as_dict()`
   model list), never "one config shared by all 900 / 11160 records"
   (DATA-48).

## Pending (not delivered by Phase 1)

- [ ] E2 formal batch (`R0 vs R2` primary, + R1/R3, D2 first then D3;
      `n0_raw`; seeds 0..9; all folds) — Gate_A R0 terms, Gate_B, Gate_C and
      every narrative rule N1..N5 hang on it;
- [ ] D3 bundle closure → run this manifest with `--experiment e1_main_d3`,
      subject to the four constraints above;
- [ ] E1 reservoir half (R0–R6 on D2/D3) and, where time allows, E3 / E7;
- [ ] narrative-gate table + evidence-audit table (each paper-claim candidate
      → experiment, statistic, CI, supported?) — Phase 2.

## Red lines observed

- `configs/protocol_v1.{1,2,3,4}.yaml` untouched (byte-frozen; no in-place
  edit, no `as_dict()` change);
- no new D2/D3 evaluation this round (only read-only analysis of committed
  records); no backfill, no imputation — missing quantities are
  `insufficient_data` / `unpairable` / `UNEVALUABLE` with a reason;
- `r2` stays secondary-only;
- committed artifacts: every number in this page is a function of the files in
  the table above, reproducible by the one command above on this tree.
