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
