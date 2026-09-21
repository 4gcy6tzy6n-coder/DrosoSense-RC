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

A summary is derived data: regenerate it from the raw records with
`python scripts/summarize.py --experiment <label> --fingerprints`. Because the four fan-out
invocations share one `--experiment` label and `write_summary_csv()` truncates, the CSV must be
regenerated after a sweep goes idle rather than read mid-flight (see the caveat above).
