# E1 sweep orchestration (D2 / D3)

These are the exact scripts used to run the M4/E1 baseline sweeps on the GPU box.
They are committed so the sweep can be reproduced and audited from the repo rather
than inferred from a chat log.

| file | role |
|---|---|
| `run_e1_d2_sweep.sh` | D2 (`d2_beef_uncontrolled`) fan-out: 4 parallel `scripts/run_baselines.py` invocations (cpu_trees / esn / gpu_rnn / gpu_conv), 10 seeds x 5 folds x 2 tasks x 9 models |
| `run_e1_d3_sweep.sh` | same shape for D3 (`d3_rainbow_trout`), LOSO(62) |
| `chain_d3_after_d2.sh` | waits for the D2 fan-out to exit, then starts D3 detached (`setsid nohup`) |
| `summarize_when_idle.sh` | waits for each sweep to go idle, then regenerates its summary from raw records |
| `d3_progress_sampler.sh` | server-side progress sampler: appends one line per interval (default 300 s) with all nine per-model record counts and the total to `logs/e1_d3_progress.log`. Exists because interactive ssh to the box flaps, which forced short, noisy rate windows; with the log, any >=10 min window is computable from one short command |
| `summarize_per_run_when_idle.sh` | sequenced follow-up: after D3 goes idle **and** the first watcher's marker artefact exists, also writes the tidy `<experiment>_per_run.csv` and logs the closure anchors (sha256 / bytes / line counts). Evidence-based sequencing so the two `summarize.py` runs never overlap and no process is killed |

## Known caveat: per-invocation summary clobbering

`write_summary_csv()` writes `results/tables/<experiment>_summary.csv` with mode `w`,
and all four fan-out invocations share one `--experiment` label, so the **last
invocation to finish overwrites the others**. The summary therefore never
accumulates all models by itself.

The per-run records under `results/raw/<experiment>/<dataset>/<model>/*.json` are
unaffected, and the summary is derived data, so `summarize_when_idle.sh` regenerates
it with the committed `scripts/summarize.py` (`load_records` -> `aggregate_records`
-> `write_summary_csv`, the same path `run_benchmark` uses).

Consequence for monitoring: **do not use the summary CSV row count as a
sweep-completion signal.** Use the per-model raw record counts
(`find results/raw/<experiment> -name '*.json' | wc -l` == 9 models x 100) plus the
absence of `run_baselines.py` processes.

## Evidence anchors (recorded 2026-09-21)

The sweeps were run on the GPU box from a checkout that is a **plain file copy with no `.git`**,
so provenance is established by content hash rather than by commit identity.

### Code provenance
The running tree is byte-identical to commit `6fac031`; the merged line `712cbef` differs from it
only by `results/tables/e1_main_d3_gpu_smoke_data34.md` (a results doc, zero code change).

| file | sha256 (first 16) |
|---|---|
| `drososense/evaluation/runner.py` | `55793bdcef7e077c` |
| `drososense/baselines/deep.py` | `ba32c5752d486cca` |
| `drososense/utils/paths.py` | `501ef4a49a2eaec6` |
| `configs/protocol_v1.4.yaml` | `4f4504528f02df53` |
| `configs/protocol_v1.3.yaml` | `d85640e556db6c00` |
| `scripts/summarize.py` | `4175a7aff6e0d390` |

### Orchestration provenance
Each script here was reconciled byte-for-byte against the copy that was running on the server
when it produced these results:

| script | sha256 (first 16) |
|---|---|
| `run_e1_d2_sweep.sh` | `e808b09da9032841` |
| `run_e1_d3_sweep.sh` | `45fe8894ea888514` |
| `chain_d3_after_d2.sh` | `85ac6fdc46bfc722` |
| `summarize_when_idle.sh` | `3db1ca0c54e65d0d` |

### Input data provenance
`tree_sha256` = `cd <dataset dir> && find . -type f | sort | xargs sha256sum | sha256sum`
(relative paths, so the digest is comparable across machines).

| dataset | resolved path | files | bytes | tree_sha256 (first 16) |
|---|---|---:|---:|---|
| `d1_beef_controlled` | `<data-root>/d1_beef_controlled` (symlink) | 1 | 1,247,016 | `346f289dc0ff5a0f` |
| `d2_beef_uncontrolled` | `<data-root>/d2_beef_uncontrolled` (symlink) | 6 | 1,064,540 | `9db0328153c8a88d` |
| `d3_rainbow_trout` | `repo/data/raw/d3_rainbow_trout` (**in-repo real dir**, see DATA-45) | 210 | 1,101,066 | `a0dca23970db229f` |

### Delivered artefacts
| file | sha256 (first 16) | rows |
|---|---|---|
| `results/tables/e1_main_d2_summary.csv` | `c8369331a00c1cdb` | 19 |
| `results/tables/e1_main_d2_fingerprints.csv` | `933f3850ca507e96` | 901 |
| `results/tables/e1_main_d2_test_touched_once.json` | `345f7fcaa74faca8` | 10 (`awk 'END{print NR}'`; `wc -l` reports 9 — no trailing newline) |
| `results/tables/e1_main_d2_per_run.csv` | `1e796ccb85a5bb86` | 901 (header + 900 tidy per-run rows; the frame M4's paired tests read) |

A summary is derived data: regenerate it from the raw records with
`python scripts/summarize.py --experiment <label> --fingerprints`. Because the four fan-out
invocations share one `--experiment` label and `write_summary_csv()` truncates, the CSV must be
regenerated after a sweep goes idle rather than read mid-flight (see the caveat above).

## E1 generation and metric semantics (recorded 2026-09-21)

These are the facts a reader needs to interpret the E1 tables without re-deriving them
from the run records. They are record semantics, not paper prose.

### Splits per dataset
| dataset | E1 split | fold count | note |
|---|---|---|---|
| `d2_beef_uncontrolled` | LOSO | 5 | one cut per fold; every fold's test set contains all four classes |
| `d3_rainbow_trout` | **LOSO(62)** | 62 | one fillet per fold; introduced by the v1.3 owner amendment (OD1 option (a)) because the exact two-sided sign test needs >= 6 clusters for alpha = 0.05 (floor `2/2^5 = 0.0625 > 0.05`; 62 clusters give `2/2^62`) |

The m1-era D3 benchmark used `GroupKFold(5)` under protocol v1.1 (folds of 13/13/12/12/12
specimens). It is **not** a like-for-like baseline for the E1 LOSO(62) numbers, and the v1.3
text states that the v1.1/v1.2 D3 rows are kept under their original labels and not
re-interpreted.

### `config_hash` is invocation-scoped
`BenchmarkConfig.as_dict()` includes the `models` list, so each fan-out invocation gets its own
`config_hash` (one per batch, not one per unit). E1/D2 produced exactly four:

| `config_hash` | models | records |
|---|---|---:|
| `c84d501b1a72` | pca_svm, random_forest, svm_rbf, xgboost | 400 |
| `d9a050a656ab` | gru, lstm | 200 |
| `a60d157027a1` | cnn1d, tcn | 200 |
| `d9528be0bd0a` | esn | 100 |

Consequences: "same config" statements are scoped to one invocation, a re-run must preserve the
batch grouping, and the §17 guard compares these hashes per test fingerprint (see DATA-48).

### Metric semantics that differ between D2 and D3
- **AUROC is defined only when all four classes are present in the test split.** On D3 LOSO(62)
  each fold holds one fillet, so most folds lack classes; the protocol requires reporting
  `n_auroc_defined` next to any AUROC mean, and a partial average is never reported.
- **macro-F1 is always scored over the fixed label set 0..3 with `zero_division=0`**, so a class
  absent from a fold contributes F1 = 0. Per-fold ceilings are therefore 0.25 / 0.50 / 0.75 / 1.00
  for 1 / 2 / 3 / 4 classes present. D3 macro-F1 is consequently **not magnitude-comparable** with
  D2 (four classes in every fold) or with the m1-era GroupKFold(5) D3 values.
- Conclusions must come from **fold-level paired comparisons** (the protocol's decisive test is
  the cluster-level paired sign test); pooled means across differently-covered folds are not a
  substitute.
- A class-complete-subset comparison may be used **only** as a pipeline-consistency check
  (`n_complete` and the partition difference must be stated); using it as evidence would require a
  protocol amendment.

### Known environment caveats affecting reproduction
- `DROSOSENSE_DATA` does not redirect raw data: `drososense/utils/paths.py` hard-codes
  `DATA_RAW_DIR = PROJECT_ROOT/"data"/"raw"` (DATA-45); the server bridges this with symlinks for D1/D2,
  while D3's raw sits **inside** the repo tree.
- `configs/datasets/d3_rainbow_trout.yaml` still declares `group_kfold(5)` while the active protocol
  requires LOSO(62), so `--split-strategy auto` (the default) would silently produce non-compliant D3
  records; the sweeps here pass `--split-strategy loso` explicitly (DATA-47).
