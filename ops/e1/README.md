# E1 sweep orchestration (D2 / D3)

These are the exact scripts used to run the M4/E1 baseline sweeps on the GPU box.
They are committed so the sweep can be reproduced and audited from the repo rather
inferred from a chat log.

| file | role |
|---|---|
| `run_e1_d2_sweep.sh` | D2 (`d2_beef_uncontrolled`) fan-out: 4 parallel `scripts/run_baselines.py` invocations (cpu_trees / esn / gpu_rnn / gpu_conv), 10 seeds x 5 folds x 2 tasks x 9 models |
| `run_e1_d3_sweep.sh` | same shape for D3 (`d3_rainbow_trout`), LOSO(62) |
| `run_e1_d3_sweep_v3.sh` | DATA-51 deep-model re-shard: 8 processes, one model x seed-half each (`gru` / `lstm` / `cnn1d` / `tcn`, seeds 0-4 and 5-9), logs per group to `$LOGDIR/e1_d3_deep_<STAMP>_<tag>.log` |
| `run_e9_size_d2.sh` | DATA-59 E9 size study, D2: 5 parallel `scripts/run_reservoir_e2.py` invocations, one per §19 size (250/500/1000/2000/4000), label `--experiment e9_size`, seed 0 smoke, `--skip-existing` default; committed NOT launched (awaiting D2 smoke go-ahead) |
| `run_e9_size_d3.sh` | DATA-59 E9 size study, D3: seed-half sharding over the §19 grid (DATA-51/58 pattern), 10 processes; heavy head (5 sizes x 7 families x 62 folds x 10 seeds x 2 tasks); committed NOT launched |
| `chain_d3_after_d2.sh` | waits for the D2 fan-out to exit, then starts D3 detached (`setsid nohup`) |
| `summarize_when_idle.sh` | waits for each sweep to go idle, then regenerates its summary from raw records |
| `d3_progress_sampler.sh` | server-side progress sampler: appends one line per interval (default 300 s) with all nine per-model record counts and the total to `logs/e1_d3_progress.log`. Exists because interactive ssh to the box flaps, which forced short, noisy rate windows; with the log, any >=10 min window is computable from one short command |
| `summarize_per_run_when_idle.sh` | sequenced follow-up: after D3 goes idle **and** the first watcher's marker artefact exists, also writes the tidy `<experiment>_per_run.csv` and logs the closure anchors (sha256 / bytes / line counts). Evidence-based sequencing so the two `summarize.py` runs never overlap and no process is killed |

## E9 pre-launch gate (DATA-59)

`run_e9_size_d{2,3}.sh` may only be launched after the protocol §19 nested
sampling property is asserted on the live NPZ:

    DROSOSENSE_DATA=<data-root> python -m pytest \
        connectome/tests/test_select_neurons_nested.py -v

The test pins `N=250 ⊂ N=500 ⊂ N=1000 ⊂ N=2000 ⊂ N=4000` on the frozen
DATA-3 selection (`connectome.select_neurons`, seed 20260920) — the exact
function the reservoir runner calls for every `--reservoir-size`. DATA-59
recorded all four inclusions OK, 0 leaked nodes.

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
