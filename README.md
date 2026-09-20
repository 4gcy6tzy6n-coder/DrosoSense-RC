# DrosoSense-RC

**Drosophila Sensory Connectome Reservoir Computing for Lightweight AgriFood Sensing**

A frozen sparse biological reservoir with a lightweight trained readout, investigated as a
computational substrate for electronic-nose food-quality monitoring.

---

## Status: M1 (revised) — protocol v1.2 frozen, connectome-free benchmark delivered

This repository is at **M1**. It contains:

- a frozen, machine-readable experimental protocol (`configs/protocol_v1.2.yaml`), with v1
  and v1.1 kept unchanged beside it and a digest sidecar per version that makes a
  post-freeze edit detectable;
- a leakage-audited data layer: specimen- **and session**-level splits, train-only
  standardisation, bounded windowing, and active leakage audits with positive controls;
- a nine-model baseline zoo behind one interface;
- traceable dataset manifests for D1/D2/D3, all three now acquired and checksum-pinned,
  including 210 sensor tables read out of a 21.1 GB archive by HTTP range request;
- paired statistics, multiplicity and **evaluable** gate and narrative rules.

### What M1 does NOT claim

There is **no connectome in this repository**. The biological reservoir (R0) and its
topology controls are M2/M3 work. Accordingly:

> We investigate whether a Drosophila olfactory connectome can serve as a fixed sparse
> computational substrate for lightweight food-quality sensing.

That is the only claim this stage supports. Any statement that a connectome reservoir
outperforms, or is competitive with, a baseline is **forbidden** here
(`configs/protocol_v1.2.yaml` → `scope_boundaries.current_claims_forbidden`).

The stronger claim — *"the biological topology provides a useful inductive bias under
limited-data and sensor-degraded conditions"* — may be used only after `Gate_B` evaluates
to true under the frozen expression.

---

## R0 rework — what changed and why

This revision closes the findings of the R0 audit (`DATA-11`). Each is a defect that would
have become unrecoverable once real connectome evaluation started:

| Item | Was | Now |
| --- | --- | --- |
| **X1** | D1 and D2 bound to each other's DOIs | D1 = `10.17632/n8mc3nspfn.1`, D2 = `10.17632/mwmhh766fc.3`, following the providers' own titles |
| **X2** | D2 used v1 (one continuous session) → "specimen split impossible" | D2 uses v3 (five beef cuts) → **LOSO(5)**, protocol-compliant |
| **X3** | D3 "BLOCKED — 21.1 GB archive" | 210 CSVs extracted by HTTP range request: **1.04 MB**, protocol-compliant |
| **X4** | D3 schema was an unverified expectation | Schema **measured** from the archive; four-level label derived at frozen trout thresholds and declared as derived |
| **X5** | Protocol said 4-class AUROC, implementation averaged 2–3 | Empty-class policy declared **and** implemented: four classes or `null`, never a partial average |
| **X6** | "36 runs each" | 18 per dataset, 36 in total |
| **C1** | Bootstrap resampling unit = *seeds* | Resampling unit = **fold**; seeds are a stratification and are never resampled |
| **C2** | Gates stated as prose (`~=`, `>`, "competitive") | Boolean expressions over a fixed predicate vocabulary, with numeric equivalence margins |
| **C3** | No freeze evidence | RFC3339 timestamp + digest sidecar + a live data-contact log |

---

## R0.1 review — what changed and why (protocol v1.2)

The R0.1 statistical review (`DATA-18`) re-derived the delivered v1.1 tables and reproduced
them exactly — 8/8 statistic rows, deltas, intervals and pair counts. It then returned
**Gate B = FAIL, Gate A / C = CONDITIONAL**, on the ground that the *decision machine* could
not be opened rather than that the numbers were wrong. No experiment was re-run.

| Item | Was | Now |
| --- | --- | --- |
| **C1** | Gate expressions name `D2`, the contrast table is keyed `d2_beef_uncontrolled`; the symbol table was built from the values the results happened to contain, so every gate resolved nothing and reported UNEVALUABLE under a reason that read like a missing contrast | Names come from the protocol's own `datasets` block; each short name is registered as an alias of its id. Unknown name and missing contrast are separate, distinguishable failures |
| **C1b** | `params(R0) < params(GRU)` — the evaluator was handed an empty parameter table, so Gate_A could never be evaluated even with every contrast present | Parameter counts are read from the run records' `n_trainable_parameters` |
| **C2** | `sig()` read a Wilcoxon p over `n_folds × n_seeds` pairs: 50 pairs from 5 specimens. D2 reported p = 5.45e−13 where the cluster-level exact p is 0.0625 | The decisive test is an exact sign test over cluster means, and `sig` additionally requires `minimum_achievable_p_over_clusters ≤ α` |
| **C2b** | The bootstrap resampled folds *within* each seed and averaged the per-seed means afterwards, so the interval was divided by the seed count — precision from a unit that carries no new specimens. Gate_A is decided entirely by interval predicates | Seeds are averaged inside each cluster *before* the clusters are resampled, so the interval is governed by the between-cluster spread. For a **fixed** set of per-cluster values it is exactly invariant to the seed count (`bootstrap.seed_invariance_required`) |
| **H1** | D2's cut grouping was described as specimen-level, and its 2016/2018 campaigns were unrecorded | `group_semantics_verified: false`; the campaign dates recorded; leave-TS1-out and leave-{TS2..TS5}-out decomposition is a mandatory E1 reporting item |
| **H2** | D3's endpoint is seven storage days; nothing said so | 7 endpoint levels, `df_cap: 7`, `label_stratum: day` declared, with the obligation to relate any D3 n / df / CI / resampling unit to them |
| **H4** | `--check` printed "first test evaluation: not started" while the live log had 24 entries | It prints the live log's value, and warns when the frozen placeholder and the live log disagree |
| **M1** | `n_specimens_tested` summed per-run counts (D2 reported 50 for 5 specimens) | `n_specimen_evaluations` + `n_distinct_specimens` (D2 = 5, D3 = 62) |
| **M2** | D3 stated 63 specimens in four places | 62, with the 63 raw filename tokens and the two aliases that reduce them recorded |
| **M3** | Effect size chosen by metric *name*, so `r2` fell through to rank-biserial | Chosen by **task**; `r2` takes Hodges–Lehmann |
| **M4** | `equivalence_p` was a proxy, not a TOST p | Renamed `tost_proxy_p` and labelled; the cluster count behind an equivalence verdict is reported |
| **M5** | D1's `data_verified.specimens: 12` described a split it no longer used | `specimens: null` + the time-block surrogate actually in use |
| Aux | Protocol claimed a 7-channel D1/D2/D3 intersection including `MQ2`/`MQ4`, which D3 does not have | Measured intersection: `{MQ3, MQ5, MQ135, temperature, humidity}` — 5 channels |

Two things this rework deliberately did **not** do. It did not change any
`split_protocol` value: `open_decisions[OD1]` in the protocol records that Gate_B and
Gate_C are unreachable at five clusters, states the arithmetic, and leaves the choice to
the project owner, because changing a split until a gate becomes reachable is choosing
the design after knowing which way the gate leans. And it did not start any connectome
experiment.

---

## Second review round — declared but not wired

The re-verification confirmed the geometry and found three more gaps, two of them the same
defect class as the first round: **the protocol declared something and no production code
read it.** That class had now appeared three times (gate symbols, `model_params`, and
`multiplicity.families`), so this round also delivers a standing inventory of every
declarative field and its reader — see [`docs/protocol_field_readers.md`](docs/protocol_field_readers.md),
regenerate with `python scripts/protocol_field_readers.py --write`.

682 declared leaves: 225 read by the production path, 2 read through a named exception, 189
free text, and 266 orphaned with a recorded disposition — 195 of them statements or
declarations whose operative copy lives elsewhere, 16 **awaiting human determination**, and
**55 real holes** where nothing reads the field and something should.

The reader criterion was tightened in DATA-21, and the numbers above are the tightened ones. A
reader is now a string literal equal to the field's NAME, parsed out of the module so that a
literal inside a docstring or a comment does not count, and never a match on the field's
declared VALUE. Both retired rules had been certifying coincidences: `tasks.*.label_column`
was "read" by the constant `schema.py` hardcodes, `gates.Gate_C.evaluated_on` by the `D2`
inside a docstring. Thirty-nine rows lost their reader; 17 were already recorded as gaps and
became holes, and the other 22 are listed under *awaiting human determination* with the match
the retired rule used printed beside them, because that match is what has to be adjudicated.
The hole count is a lower bound, and the point of the bucket is that it is now a stated one.

The holes are mostly dependencies on work not yet started (the E4–E6 robustness protocol, the
E9 size study, the experiment register), plus the three that have already diverged from the
code: `results.raw_dir`/`tables_dir`/`figures_dir`, whose paths are resolved through
`drososense/utils/paths.py`. DATA-21 closed ten of them: `hyperparameter_selection.grid.*`,
`selection_split` and `scope` are read by `drososense/evaluation/selection.py`, `seeds.*` by
`drososense/utils/seeding.py`, and `preprocessing.windowing.length_candidates` by the same
window-length check. `tests/test_protocol_field_readers.py` makes the inventory a gate: a newly
declared field with no reader and no disposition fails the suite.

| Item | Was | Now |
| --- | --- | --- |
| **N1** | `multiplicity.families` declared eight families; the correction grouped by `(metric, dataset)` instead, merging the primary comparison with the topology and baseline families and merging the five robustness families into one | The correction groups by the family the protocol enumerates, keyed on `(contrast, condition)` and stratified by dataset and metric. `family_rule` is enforced: two families for one pair stops the analysis, no family means uncorrected and relabelled exploratory |
| **N2** | `params()` read `results/raw/**`, which is git-ignored: on a clean clone Gate_A was UNEVALUABLE for lack of a parameter count | `results/tables/model_parameters.json` is committed, keyed by registry id **and** protocol id (`esn` → `R4`, `gru` → `GRU`), and is the base source. Model sizes vary per fold (the GRU spans 1265–4452), so the table records the spread and states the selection rule it uses |
| **N3** | The seed-invariance claim was written more broadly than it holds | Narrowed to the measured form: for a **fixed** set of per-cluster values the interval is exactly invariant (`bootstrap.seed_invariance_required`, now read by a test); in a real run the cluster values sharpen with seeds, so the width narrows with the between-cluster spread — measured 0.302 → 0.073 → 0.029 at 1/5/10 seeds on D2 |

### Reproducing the delivered artifacts needs the local data

`results/tables/m1_benchmark_gates.json` and `..._dataset_availability.json` are generated
against `data/raw/**`, which is git-ignored by design. On a clean clone D1/D2/D3 are
reported as not on disk and `N5` becomes `True`, where the committed files say `False`. That
is the data policy showing through, not a logic error — but a reader comparing the two will
otherwise have to work it out. The gate verdicts that do **not** depend on the data being
present are the ones covered by the tests in `tests/test_e2e_gate_evaluation.py`, including
one that evaluates the rules with `results/raw` made unavailable and asserts the outcome is
unchanged.

The same applies to the test suite, more quietly. Twelve of the tests in
`tests/test_dataset_bindings.py` load D1/D2/D3 from `data/raw/**` and **skip** when the raw
files are absent, so a clean clone runs a smaller suite than the author's checkout without
saying so in the pass count: the twelve move from passed to skipped, and the total is only
comparable between two environments that agree on whether the data is on disk. They are the
tests that pin the R0 audit's X1 and X2 — the DOI bindings, D2's five cuts, D3's fillet token,
and the rule that nothing reads a column by position — so a green run without them is not
evidence about the data layer.

---

## Quick start

```bash
# 1. Environment
conda env create -f environment.yml && conda activate drososense
#    or: python -m pip install -r requirements-dev.txt

# 2. Check the environment against the declaration (non-zero exit on any gap)
python -m drososense.utils.env_report

# 3. Verify the frozen protocol has not been edited since it was frozen
python -m drososense.utils.protocol --check

# 4. Run the test suite (hermetic; the real-data tests skip when data is absent)
python -m pytest tests/ -m "not slow"

# 5. Acquire the datasets (D3 pulls 210 members by range request, never the 21 GB)
python scripts/download_data.py --all
python scripts/download_data.py --all --verify-only

# 6. Smoke test the whole pipeline on the synthetic fixture
python scripts/make_fixture.py
python scripts/run_baselines.py --dataset synthetic_enose --experiment smoke \
    --models all --tasks classification regression --seeds 0 1 \
    --window-lengths 16 --max-folds 1 --smoke

# 7. Aggregate raw records into a summary table
python scripts/summarize.py --experiment smoke --print
```

> **Declared environment vs this machine.** `results/tables/environment_report.json` records
> which declared packages are present, missing or out of range *here*. Any statement about
> a test count or a benchmark applies to the environment it was produced in; the runner
> prints a caveat when that is not the declared one.

> **macOS / XGBoost.** XGBoost needs the OpenMP runtime. Install it with
> `brew install libomp`, or point `DYLD_FALLBACK_LIBRARY_PATH` at a `libomp.dylib` from
> another package. If it cannot load, `run_baselines.py` reports XGBoost as skipped **with
> the reason** — it never silently disappears from the comparison.

---

## The frozen protocol

`configs/protocol_v1.2.yaml` is the contract. Its digest is recorded in
`configs/protocol_v1.2.sha256`; `python -m drososense.utils.protocol --check` recomputes it
and fails on any drift. A frozen protocol is amended by **adding** a version file, never by
editing one — so `protocol_v1.yaml` and `protocol_v1.1.yaml` are still on disk, byte for
byte as frozen, and `protocol_v1.1.sha256` still matches.

| Item | Value |
| --- | --- |
| Split unit | `specimen` — **red line** |
| Session unit | `session_id` — a second boundary: no window spans two acquisition sessions |
| Seeds | `0..9`, with a declared RNG hierarchy (split / graph / input mapping / readout); extendable to 20 only as a whole block, and the two designs are never pooled |
| Classification primary / secondary | `macro_f1` / balanced accuracy, AUROC, accuracy |
| AUROC empty-class policy | `require_all_classes` — four classes or `null` |
| Regression primary / secondary | `mae` / RMSE, R² |
| Co-primary policy | both required, no alpha split |
| Contrasts | enumerated (`R0_vs_R2`, `R0_vs_R4`, `R0_vs_R3`, `R0_vs_GRU`, `R0_vs_R5`, `R0_vs_R1`) |
| **Primary (decisive) test** | **exact two-sided sign test over cluster means**, α = 0.05 |
| Descriptive pair-level test | Wilcoxon signed-rank, two-sided, `zero_method=wilcox` — published, **not decisive** |
| Effect size | rank-biserial (classification), Hodges–Lehmann (regression), chosen **by task** |
| Bootstrap | 10 000 resamples, percentile CI, **resampling unit = fold**, seed fixed, seeds averaged inside each cluster |
| Pairing | observation = `(seed, fold)`; `n_pairs = n_folds × n_seeds`; `n_clusters = n_folds` |
| Equivalence | TOST at Δmacro-F1 = 0.02, ΔMAE = 0.05 — interval inclusion, never "failed to reject"; the number published beside it is `tost_proxy_p`, a proxy |
| Multiplicity | Holm, families enumerated per `(contrast, condition)` |
| Window candidates | L ∈ {8, 16, 32, 64}, selected on **validation**, `test_touched_once: true` |

### Gate reachability, stated plainly

`sig(...)` requires `minimum_achievable_p_over_clusters <= alpha`, and an exact sign
test over `n` clusters cannot report below `2 / 2**n`. A dataset with **five** clusters
therefore has a floor of **0.0625 > 0.05**: no significant cluster-level result is
reachable there, whatever the data says.

D2 has five published groups and always will. D3's five blocks are a `split_protocol`
choice, and D3 has 62 fillets, so its cluster count can be raised. Under the current
split **Gate_B and Gate_C cannot be opened**, and that is a property of the design rather
than of any result. The remedy is recorded as `amendment_v1_2.open_decisions[OD1]` and is
deliberately **not** applied: changing a split until a gate becomes reachable would be
choosing the design after knowing which way the gate leans.
| Standardisation | `x' = (x − μ_train)/(σ_train + ε)`, fitted on **train only** |
| Gates | A (viable), B (TAFE-worthy), C (strong paper) — boolean expressions |
| Narrative rules | N1–N5, each with an evaluable trigger |
| Stopping / failure / exclusion / degradation | all declared; `seed_reduction_allowed: false` |

**Freeze evidence.** `freeze_evidence.data_contact_log` holds the freeze-time value (`null`);
the live record is written by the runner to `results/tables/data_contact_log.json`. Recording
the first test evaluation inside the frozen file would invalidate its digest exactly when the
freeze starts to matter, so it lives outside it and is asserted by a test.

**Gates are evaluable, not rhetorical.** `drososense/evaluation/gates.py` evaluates each
gate and each narrative trigger as a boolean expression over a fixed predicate vocabulary
(`sig`, `equiv`, `noninferior`, `ci_contains_zero`, `delta`, `ci_low`, `ci_high`, `params`,
`margin`, `unavailable`, …). The evaluator is a whitelisted AST walker — no attribute
access, no imports, no comprehensions — and an expression that references a contrast with
no result raises rather than quietly evaluating to `False`. A gate that cannot be evaluated
is reported as **UNEVALUABLE**, never as passed or failed.

**Narrative adjustment rules** are pre-registered, not improvised. For example, N1 fires on
`equiv(R0, R2, …) and sig(R0, R3, …)`: the conclusion becomes *degree / statistical
structure matters* — it is forbidden to claim the specific *Drosophila* wiring matters.

---

## Data

Three real datasets are in scope. Their status is reported honestly rather than worked
around; the project never substitutes a different dataset for a missing one.

| ID | Dataset | DOI | Rows | Specimens | License | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| D1 | Beef-Controlled | `10.17632/n8mc3nspfn.1` | 20 815 × 14 | 12 *(assumed blocks)* | CC BY 4.0 | Acquired — **non-compliant split** |
| D2 | Beef-Uncontrolled | `10.17632/mwmhh766fc.3` | 10 800 × 14 | **5 cuts** | CC BY 4.0 | Acquired — **protocol-compliant, LOSO(5)** |
| D3 | Rainbow Trout | `10.5281/zenodo.20649184` | 25 080 × 9 | **62 fillets / 210 sessions** | CC BY 4.0 | Acquired — **protocol-compliant, GroupKFold(5)** |

Full provenance, checksums, schemas and manual steps: `data/manifests/*.yaml`.

### D2 — the specimen split the project needed

`mwmhh766fc` **version 3** publishes five series, one per beef cut — the provider's own
grouping. `specimen.source: series_file` takes the file stem (`TS1`…`TS5`) as the specimen
id, so LOSO(5) is an exhaustive specimen-level split rather than a convenience.

Five specimens is few, and the protocol records why that matters **before** any result
exists: the exact two-sided minimum p over five clusters is 2/2⁵ = 0.0625, above α. A
significant specimen-level result is not reachable on D2 alone, so D2 is reported as
*underpowered* rather than as evidence of no effect.

Version 1 of the same DOI is **not used** — one continuous 4553-row session with no sample
grouping. Its digest is retained in the manifest so a stale local copy cannot be mistaken
for v3.

⚠️ **The five files do not share one column order.** `TS1` ends `(…, Humidity, Temperature)`;
`TS2`–`TS5` end `(…, Temperature, Humidity)`. Every column is addressed **by name** through
`raw.feature_map`; no column is read by position anywhere in this project, and a test asserts
that a positional read would be visibly wrong.

### D3 — acquired without the 21.1 GB download

The archive is a single 21.1 GB ZIP dominated by 1 640 CR2/JPG images. The sensor tables are
210 CSVs totalling **1 092 874 bytes**, read straight out of the archive by HTTP range
request (`drososense/data/remote_zip.py`, a self-contained ZIP64 central-directory parser).

**What is not claimed:** because the archive is never fetched whole, its publisher md5
**cannot** be verified. That is stated in the manifest rather than glossed. What *is*
verified on every acquisition is the archive's own CRC-32 for each member and a recorded
SHA-256 for every extracted file (`data/manifests/d3_sensor_files.yaml`, 210 entries).

The fillet token in each filename is a genuine repeated-measures identifier — the same
token reappears across storage days with a strictly larger TVC each time — giving 62
specimens over 210 sessions. Two filenames carry typos; `raw.specimen_aliases` records each
correction with the evidence, and the loader refuses to start if an alias names a file that
is not present.

### D1 — the honest blocker

The provider publishes **no sample identifier**: `sl no` and `minutes` are both monotonic
with zero resets, so the file is one continuous acquisition. `specimen.source: time_block`
cuts it into 150-minute blocks as a documented stand-in, and every record produced this way
carries `protocol_compliant: false`. **D1 is excluded from every gate.**

The stand-in is also *optimistic by construction*: D1's classes occupy contiguous stretches
(transitions at rows 2939, 5207, 7499), so a held-out block resembles its neighbours. High
scores under this split are an artefact of the assumption, not evidence of sensing
performance.

### Labels are derived, on every dataset

On all three datasets the class label is a **threshold of TVC**, so classification and
regression are two views of one variable and are never presented as two independent sources
of evidence. The provenance differs and is recorded per dataset:

- **D2** — the provider ships the words `excellent / good / acceptable / spoiled`. These are
  the dataset's own names.
- **D1** — the provider ships integers `1..4` and **no names**. The four names are the
  project's labels for TVC strata at the frozen thresholds 3/4/5, and must not be cited as
  D1's own class names.
- **D3** — the provider ships only `Fresh / Spoiled`, and the beef thresholds collapse the
  whole archive to two classes. Protocol v1.1 froze trout-specific thresholds
  **4.3 / 5.0 / 6.5** before any D3 model ran, giving four non-empty strata and making the
  cross-dataset macro-F1 a like-for-like quantity. The four names are the project's.

---

## Repository layout

```text
configs/          protocol_v1.yaml + protocol_v1.1.yaml (both frozen, unchanged)
                  + protocol_v1.2.yaml (active)
                  one .sha256 sidecar per version + dataset configs
data/
  raw/            acquired files (git-ignored)
  processed/      normalised frames (git-ignored)
  splits/         per-fold split.json + scaler.pkl (git-ignored)
  manifests/      provenance, checksums, licenses, blockers, D3 member listing (committed)
drososense/
  data/           schema, loaders, splits, scaling, windowing, leakage,
                  remote_zip, acquisition, synthetic
  baselines/      classic (SVM/RF/XGBoost/PCA+SVM) and sequence (GRU/LSTM/CNN/TCN)
  reservoir/      ESN — the R4 control, NOT a connectome
  evaluation/     metrics, paired statistics, gates, hyperparameter selection,
                  contact log, records, driver
  utils/          paths, seeding, config, protocol freeze checks, environment report
scripts/          download, fixture, preprocess, benchmarks, summarize
tests/            unit + integration; leakage audits carry positive controls
results/
  raw/            one JSON per (model, task, seed, fold) — git-ignored
  tables/         aggregated summaries, environment report, contact log — committed
```

**Raw data is never committed.** `.gitignore` excludes `data/raw`, `data/processed`,
`data/splits`, `connectome/raw`, `connectome/adjacency` and `results/raw`.

---

## Leakage control

Four channels are audited, and every audit has a **positive control** — a test that feeds it
deliberately leaked input and asserts it fires. An audit never observed to fail is
indistinguishable from one that cannot fail.

| Channel | Guarantee | Enforced by |
| --- | --- | --- |
| Split | No specimen in two of train/val/test | `audit_fold`, `Fold.__post_init__` |
| Preprocessing | Scaler statistics from train rows only | `audit_scaler` |
| Specimen window | No window spans two specimens | `audit_windows` on the specimen key |
| **Session window** | No window spans two acquisition sessions | `audit_windows` on the session key |
| Row reuse | No row on both sides | `audit_no_row_reuse` |

The session boundary is a second red line, added in this revision: D3 measures each fillet
on up to seven storage days with a different TVC each day, so a slide that crossed two of
them would carry channels from one occasion and a label from another. The specimen audit
alone cannot see that.

The sharpest test builds the same model on a **row-level** split and a **specimen-level**
split of the same fixture and asserts the row-level split scores higher — quantifying the
leakage that `split_unit: specimen` exists to remove
(`tests/test_leakage.py::test_row_level_split_scores_higher_than_specimen_level`).

There is **no row-level splitter anywhere in the data layer**; `drososense.data.splits`
never imports one, checked by an AST-based test.

---

## Models

All nine baselines implement one interface and receive identical tensors for a given
`(dataset, seed, fold, window length)`.

| ID | Family | Notes |
| --- | --- | --- |
| `svm_rbf` | classical | RBF kernel SVM/SVR |
| `random_forest` | classical | 300 trees |
| `xgboost` | classical | gradient-boosted trees |
| `pca_svm` | classical | PCA → RBF SVM |
| `gru` | sequence | torch |
| `lstm` | sequence | torch |
| `cnn1d` | sequence | torch |
| `tcn` | sequence | dilated **causal** TCN |
| `esn` | reservoir | leaky ESN, frozen reservoir + trained readout — **the R4 control** |

`--smoke` selects deliberately tiny hyperparameters. Smoke numbers are a pipeline check,
**not** tuned results, and are never reported as findings.

### Hyperparameters come from the declared grid

Both paths are checked against `hyperparameter_selection.grid` before any data is touched:

* **Supplied parameters** — every run, including every M1 run and every smoke run. A declared
  knob carrying an undeclared value is refused with an `OutOfGridError`, and so is a window
  length outside `grid.window_length`. The knob names are the protocol's; the binding to the
  names the models use (`reservoir_leak_alpha` → `leak`, `spectral_scaling` →
  `spectral_radius`, …) is declared in `drososense/evaluation/selection.py:KNOB_BINDINGS`, and
  a protocol knob that no model parameter answers to raises at load time.
* **Selection** — a run that passes a grid as `BenchmarkConfig.selection_grid` selects each
  model's parameters per fold on that fold's **validation** specimens, by the metric
  `selection_metric` names for the task. The test split is not read during selection. The
  spectral scaling is chosen once per `(dataset, seed, fold)` and reused across every reservoir
  in the group, so R0 and its control cannot end up on differently scaled graphs; the choice is
  recorded on every run record and in the committed fingerprint table.

M1 supplies its parameters and does not select. The selector exists so that M4 does not have to
invent one, and the grid gate exists so that neither path can tune past the frozen space.

`params(model)` in a gate means **trainable** parameters — the count updated by fitting, not
the reservoir's node count. Frozen reservoir weights are reported separately and never
folded into the trainable count.

---

## Results schema

Every run writes one record to
`results/raw/<experiment>/<dataset>/<model>/<task>_seed<k>_fold<i>.json`. Abridged from a
record produced by the compliant D2 benchmark:

```json
{
  "run_id": "d2_beef_uncontrolled|svm_rbf|classification|seed00|fold00",
  "experiment": "m1_benchmark",
  "dataset": "d2_beef_uncontrolled",
  "model": "svm_rbf",
  "task": "classification",
  "seed": 0, "fold_id": 0,
  "protocol_version": "1.1.0",   # v1.1 runs keep their label; M4 runs record 1.2.0
  "window_length": 16,
  "metrics": {"macro_f1": 0.6682, "balanced_accuracy": 0.6908, "accuracy": 0.7650,
              "auroc": 0.8987, "auroc_n_classes_scored": 4, "auroc_defined": true,
              "empty_class_policy": "require_all_classes"},
  "n_train_windows": 6435, "n_test_windows": 2145,
  "n_train_sessions": 3, "n_test_sessions": 1,
  "train_specimens": ["TS4", "TS1", "TS2"],
  "test_specimens": ["TS3"],
  "evidence_class": "real",
  "protocol_compliant": true,
  "test_fingerprint": "d46e3250d3d83766",
  "fold_fingerprint": "20a70c561f8e",
  "class_coverage": {"missing_in_train": [], "missing_in_test": []},
  "config_hash": "59a133a794a8",
  "status": "ok",
  "failure_reason": ""
}
```

Contrast that with the same metric on a fold where a class is absent: `auroc` becomes `null`,
`auroc_n_classes_scored` becomes `0`, and `auroc_defined` becomes `false` — rather than a
two-class average reported under a four-class name.

**AUROC policy.** `auroc` is `null` unless all four classes are present in the test split,
and `auroc_n_classes_scored` is then either `4` or `0` — never a partial average over two or
three classes wearing a four-class name. The number of runs that contributed to a mean is
reported as `n_auroc_defined`, so the lost coverage is visible instead of implied. Macro-F1
is scored over the fixed label set with `zero_division=0`, so it stays defined even when a
class is absent.

**Fields that make a number safe to quote**, all of which survive into the summary table:

- **`evidence_class`** — `real` or `synthetic_fixture`. Synthetic and real runs are never
  averaged together.
- **`protocol_compliant`** — whether the split satisfied `split_unit: specimen`.
- **`status`** — `ok`, `failed`, `skipped`, `oom` or `timeout`. A failed run stays in the
  table counted as failed; it is never silently dropped.
- **`test_fingerprint`** — identifies one `(test split, window length, model, task)`
  evaluation. Re-fitting on a split already touched under a different configuration raises
  rather than overwriting, which is `test_touched_once` in executable form.

---

## Regenerated results (M1, protocol v1.2)

The benchmark was regenerated on the corrected bindings. Counts are stated exactly, because
the previous revision overstated them ("36 runs each" for what was 18 per dataset):

| Experiment | Dataset | Split | Seeds | Runs | Status |
| --- | --- | --- | ---: | ---: | --- |
| `m1_benchmark` | D2 | LOSO(5) | 10 | **800** | 800 ok, `protocol_compliant: true` |
| `m1_benchmark` | D3 | GroupKFold(5) | 10 | **800** | 800 ok, `protocol_compliant: true` |
| `m1_real_validation` | D1 | time-block holdout(5) | 1 | **80** | 68 ok, **12 failed**, `protocol_compliant: false` |

800 runs = 10 seeds × 5 folds × 8 models × 2 tasks. (Nine models are declared; `xgboost` is
skipped in this environment — see Testing — and the skip is recorded in every summary.)

The 12 D1 failures are a real finding, not an incident: D1's four classes occupy contiguous
stretches of one continuous session, so a time-block training set can contain **one** class
and the estimator cannot be fitted. Each failure is written to `results/raw` with
`status: "failed"` and its reason, and is counted in the summary rather than dropped — which
is precisely what the protocol's `failure_handling` clause requires, and precisely what the
aborted-benchmark alternative would have hidden.

`results/tables/m1_benchmark_statistics.csv` holds the paired contrasts. Every declared
contrast is **UNEVEALUABLE**: the protocol's contrasts are connectome contrasts (R0–R5), and
an experiment containing only baselines has nothing to test against them. That is reported as
UNEVALUABLE rather than as a failed gate, because the two say different things.

Exploratory baseline comparisons (`--exploratory esn,gru`) exercise the machinery on real
data and are labelled exploratory — never corrected inside a protocol family, and unable to
make a gate true. Each line carries what the protocol demands beside the p-value:

| contrast | metric | dataset | n_pairs | n_clusters | Δ | 95% CI | effect | p (decisive) | p (pair-level, descriptive) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| esn vs gru | macro_f1 | D2 | 50 | **5** | −0.227 | [−0.240, −0.211] | −0.970 | **0.0625** | 5.5e−13 |
| esn vs gru | macro_f1 | D3 | 50 | **5** | −0.0005 | [−0.012, +0.007] | −0.058 | **0.375** | 0.73 |
| esn vs gru | mae | D3 | 50 | **5** | +0.049 | [+0.045, +0.054] | +0.043 | **0.0625** | <1e−12 |

The intervals moved when the bootstrap was fixed (v1.2 §10): every `delta` is bit-identical,
because a point estimate does not depend on how it is resampled, but the interval widths
changed — of the eight rows, six are narrower and two wider. That is what a change in the
variance decomposition looks like, and it is why the earlier intervals are not reproduced
verbatim.

Read `n_clusters` before the p-value. The decisive column is the exact sign test over the
five cluster means; its floor is 2/2⁵ = 0.0625, so a cluster-level claim at α = 0.05 is not
reachable on either dataset alone, and the table prints that floor
(`minimum_achievable_p_over_clusters`) beside every row. The far-right column is the
pair-level Wilcoxon over all 50 `(seed, fold)` pairs — the number the earlier protocol
published as *the* p-value (5.5e−13 above). It is still shown, because a reader must be able
to see what changed and why, but it is pseudoreplicated and **no gate reads it**.

What seed counts buy here is bounded, and it is worth being exact about it. The decisive
p-value is invariant to the seed count: it is computed from one mean per cluster, and a cluster
mean over ten seeds is the same point whichever way the seeds are counted. The interval is
invariant **for a fixed set of per-cluster values** — that is what the protocol's
`bootstrap.seed_invariance_required` claims, and it holds exactly. In a real run the cluster
values are themselves averages over the seeds, so they sharpen as seeds are added and the
interval does narrow: measured on the `esn vs gru / macro_f1 / D2` contrast it is 0.302 wide at
one seed, 0.073 at five and 0.029 at ten, tracking `1.96 · sd(cluster means) / √n_clusters`.
What changed is what governs it — the distance between clusters, not the seed count.

Every row also carries the multiplicity **family** that owns its `(contrast, condition)` pair
and the `p_holm` computed inside it. On this table the column is empty, because all eight rows
are exploratory baseline comparisons and an exploratory pair belongs to no pre-registered
family — a corrected p-value for one would imply a pre-registration that does not exist.

`results/tables/m1_benchmark_gates.json` records each gate and narrative rule with its
per-term values and, on every `sig` term, the cluster count and reachability floor behind it.
`results/tables/m1_benchmark_dataset_availability.json` records which datasets were obtained,
read from the manifests rather than from which contrasts this experiment happened to produce.

### Auditing the freeze from the repository alone

`results/raw/**` is git-ignored, so the per-run provenance the protocol relies on would
otherwise be checkable only on the machine that produced it. `results/tables/*_fingerprints.csv`
therefore commits the **identifiers and counts** of every run — `fold_fingerprint`,
`test_fingerprint`, `config_hash`, `class_coverage`, session and window counts, timestamp,
duration — and nothing observed. No sensor value is in it.

`results/tables/*_test_touched_once.json` turns that into the §17 check: a repeated
`test_fingerprint` with a *different* `config_hash` is a violation. Current state:

| Experiment | Records | With a fingerprint | Distinct fingerprints | Repeats | Violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m1_benchmark` | 1600 | 1600 | 1600 | 0 | **0** |
| `m1_real_validation` | 80 | 68 | 68 | 0 | **0** |
| `smoke` | 32 | 32 | 32 | 0 | **0** |

The `m1_real_validation` row shows the bookkeeping working: 12 of its 80 records carry no
`test_fingerprint` because they are the 12 failures described above — a run that never
scored a test split has no test evaluation to fingerprint, and the audit counts it as
absent rather than inventing one.

---

## Reproducibility

- A run is reproducible from `(dataset, model, task, seed, fold, window_length)` alone.
- Seeds 0–9 drive the specimen permutation, so seeds vary the *split*, not just weights.
- The RNG hierarchy is declared: `SeedSequence(root).spawn(4)` → split / graph / input
  mapping / readout, so re-running one stage does not perturb another.
- Torch CPU reductions depend on thread count, so the runner pins `torch.set_num_threads`
  (default 1). This trades speed for the ability to re-run a seed and get the same number.
- The ARPACK spectral-radius estimate for the ESN uses an explicitly seeded start vector;
  without this, two runs of the same seed diverge.
- The bootstrap seed is fixed independently of the run seed, so an interval is reproducible
  without re-running the experiment.
- Environments are recorded per run and compared against the declaration in
  `results/tables/environment_report.json`.

---

## Testing

```bash
python -m pytest tests/ -m "not slow"            # full suite
python -m pytest tests/ -m unit                  # fast, hermetic
python -m pytest tests/ --cov=drososense --cov-report=term-missing
```

Coverage is reported in the run log; the suite is required to stay above 80 %.

**Test counts are environment-qualified.** The suite's size and its pass/skip split depend on
which optional backends are installed. On the machine this revision was prepared on — macOS
arm64, CPython 3.12.13, `numpy` 2.4.4, `pandas` 3.0.2, `scikit-learn` 1.8.0, `torch` 2.11.0,
**`xgboost` not importable** (its native library needs `libomp`, which is not present) — and
with `data/raw/**` absent, the result is:

```text
340 passed, 15 skipped
```

Three of the skips are the xgboost model tests, and each skip names the missing backend. The
other twelve are `tests/test_dataset_bindings.py`, which skips when the raw data is not on
disk — see *Reproducing the delivered artifacts needs the local data* above. The same suite
in the environment declared by `environment.yml` — which pins xgboost — runs the first three
as well, and a checkout with `data/raw/**` populated runs all fifteen: **the pass count of
this suite is qualified by both the installed backends and the presence of the raw data**, and
a bare "N passed" is portable under neither. That is why the runner attaches
`environment_report.json` to every summary and prints a caveat when the local environment is
not the declared one.

---

## Known limitations

1. **No specimen-level split is possible for D1 as published.** D2 and D3 are compliant;
   D1 is not, and is excluded from every gate. Unblocking it requires an acquisition log
   from the dataset authors.
2. **D2 has five specimens.** Its exact minimum two-sided p over five clusters is 0.0625, so
   a significant specimen-level result is unreachable there. This is recorded in the
   protocol as a power limitation, before any result exists.
3. **D3's archive md5 is unverified** — verifying it would require the 21.1 GB download the
   range-request approach exists to avoid. Per-member CRC-32 and SHA-256 are verified instead.
4. **D3's four-level label is the project's, not the provider's**, and its thresholds are
   dataset-specific (4.3/5.0/6.5). Under the beef thresholds D3 collapses to two classes.
5. **Class labels are thresholds of TVC** in all three datasets, so classification is a
   discretisation of the regression target.
6. **Two D3 filenames carry typos.** Both corrections are declared with evidence in the
   dataset config; neither changes the extracted bytes.
7. **Smoke hyperparameters are untuned.** Reported comparisons require the full tuning and
   seeding procedure in the protocol.
8. **Probability values are not calibrated.** Where a classifier exposes only a decision
   function, AUROC consumes a softmax of it — monotone, so ranking is unaffected, but the
   values are not confidences.

---

## Roadmap

| Stage | Content | Depends on |
| --- | --- | --- |
| **M1** ✅ | Protocol v1.2 frozen (v1.1 superseded after the R0.1 review); leakage-audited benchmark; three datasets acquired | — |
| M2 | FlyWire/Codex olfactory subcircuit → `olfactory_v1.npz` + network report | M1 + **FlyWire CAVE/Codex token** |
| M3 | Frozen connectome reservoir + topology controls R0–R6 | M2 |
| M4 | E1–E12 experiments, paired statistics, Gates A/B/C | M3 |
| M5 | Edge replay, efficiency, real e-nose proof-of-system | M4 |

`scripts/build_connectome.py`, `run_reservoir.py` and `run_ablation.py` currently print their
plans and exit non-zero rather than half-implementing M2/M3.

**Open resource question.** FlyWire's own guidelines publish the data under CC BY-NC 4.0
while the Zenodo deposit of the same revision is marked CC BY 4.0. Which governs the derived
`olfactory_v1.npz` must be settled before any artifact is published.

### Wording constraint carried forward

Synapse counts are not synaptic strengths. The connectome weight matrix must be described as
a **synapse-count-informed structural weight**, never as true biological synaptic strength.

---

## License and attribution

Code in this repository is the project's own. Datasets retain their published licenses (all
CC BY 4.0); see each manifest for the required citation. Raw data is not redistributed here.
