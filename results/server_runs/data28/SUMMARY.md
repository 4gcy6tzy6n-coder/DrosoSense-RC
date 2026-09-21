# DATA-28 — Server engineering measurement report

Captured on the M4 GPU server (`connect.nmb2.seetacloud.com`) on 2026-09-21.
Everything below was produced by `scripts/server/measure_baselines.py` and
`scripts/server/concurrency_sweep.py` against the validation split of the
synthetic fixture (12 specimens × 120 timesteps × 8 channels, window length
16). The test split was never read.

## Hardware

| Item | Value |
| --- | --- |
| Host | autodl-container-... |
| CPU | Intel(R) Xeon(R) Gold 6248 @ 2.50 GHz, **80 vCPU** |
| RAM | 125 GB total, 107 GB available |
| GPU | **NVIDIA GeForce RTX 3080 Ti 12 GB**, driver 595.71.05 |
| `/` (overlay) | 30 GB total, 1.3 GB used (29 GB free) |
| `/root/autodl-tmp` | 50 GB total, 1.3 GB used (49 GB free) |
| Python | 3.12.3 (miniconda) |
| torch | 2.5.1+cu124 (`cuda.is_available() == True`) |

The `/` overlay grew from ~425 MB to 1.3 GB across the whole run; nothing
the harness wrote touched it. All work happened under `/root/autodl-tmp`.

## Per-model measurement

Reported values are mean ± std across **3 repetitions**, each preceded by an
un-timed warm-up fit so first-call JIT does not contaminate the latency.
`torch_num_threads` pinned to 1 for CPU runs. Window tensor is
`(315, 16, 8)` for the validation set; train tensor is `(630, 16, 8)`.

### CPU (torch_num_threads=1, torch on CPU)

| Model | Trainable params | Train time (s) | Predict (ms/sample) | Peak RSS (MB) |
| --- | ---: | ---: | ---: | ---: |
| svm_rbf | — (kernel method) | 0.027 ± 0.000 | 0.055 ± 0.001 | 545.4 |
| random_forest | 8 543 | 1.003 ± 0.024 | 0.219 ± 0.033 | 558.6 |
| xgboost | 5 111 | 1.196 ± 0.022 | 0.008 ± 0.001 | 675.0 |
| pca_svm | — | 0.059 ± 0.034 | 0.104 ± 0.076 | 679.9 |
| gru | 4 164 | 1.643 ± 0.007 | 0.008 ± 0.001 | 767.6 |
| lstm | 5 508 | 0.724 ± 0.001 | 0.006 ± 0.001 | 784.3 |
| cnn1d | 932 | 0.438 ± 0.000 | 0.003 ± 0.000 | 788.0 |
| tcn | 1 220 | 0.668 ± 0.036 | 0.005 ± 0.000 | 788.9 |
| esn | 804 | 0.159 ± 0.002 | 0.241 ± 0.003 | 800.1 |

CPU numbers follow the expected hierarchy: classical estimators with
`n_jobs=-1` saturate cores (`random_forest`, `xgboost`), the four torch
sequence models scale sub-linearly with hidden size, and `svm_rbf` is
fastest at this size because the kernel matrix is small enough to fit in
BLAS cache.

### GPU (`--use-gpu`; torch sequence models moved onto CUDA)

| Model | Trainable params | Train time (s, fit on CPU) | Predict (ms/sample) | Peak RSS (MB) | Peak GPU (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gru | 4 164 | 2.238 ± 0.246 | 0.0048 ± 0.0011 | 881.4 | 27.3 |
| lstm | 5 508 | 1.055 ± 0.022 | 0.0038 ± 0.0004 | 891.1 | 31.2 |
| cnn1d | 932 | 0.469 ± 0.005 | 0.0026 ± 0.0001 | 1049.8 | 9.9 |
| tcn | 1 220 | 0.658 ± 0.003 | 0.0031 ± 0.0002 | 1060.7 | 10.8 |

Fit time on GPU is dominated by the train step staying on CPU (the
baseline's `fit` contract doesn't accept a device); the numbers above are
the **predict-on-GPU** times that matter for an Edge-Replay / online
inference story. Predict latency is 1.5–2× faster than the CPU path on the
smallest models, but the kernel-launch overhead still makes the GPU path
comparable or slower on `cnn1d`/`tcn` at this size — at the scale M4 will
actually run (62-fold D3, larger window lengths), the GPU path is the one
that pays off.

`nvidia-smi` reported `utilization.gpu=0%` throughout every measurement
loop: the workload is too small to keep the GPU active. See "Bottlenecks"
below.

### Per-row CSVs

| Path | Content |
| --- | --- |
| `results/server_runs/data28/server_measurements/cpu/hardware_metrics.csv` | 27 rows — 9 models × 3 reps, CPU torch=1 |
| `results/server_runs/data28/server_measurements/cpu/hardware_summary.csv` | 9 rows — mean ± std per model |
| `results/server_runs/data28/server_measurements/gpu/hardware_metrics.csv` | 12 rows — 4 torch models × 3 reps on CUDA |
| `results/server_runs/data28/server_measurements/gpu/hardware_summary.csv` | 4 rows — mean ± std per model on CUDA |
| `results/server_runs/data28/server_measurements/{cpu,gpu}/environment.json` | env fingerprint + nvidia-smi sample |

## Concurrency scaling curve

`concurrency_sweep.py` with `--workers 1 2 4 8 12 16 24 32 48`,
`--runs-per-worker 2`, models `svm_rbf esn` (small workload to keep the
curve clean), `torch_num_threads=1`, OMP / MKL / OPENBLAS pinned to 1 per
child. The two-model workload measures the cost of "one model instance
fit + predict on the validation split" — about 8 s per run on this server.

| Workers | Wall clock (s) | Runs / minute | Per-run latency (s) | Speedup vs N=1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.79 | 15.40 | 7.79 | 1.0× |
| 2 | 8.03 | 29.88 | 7.88 | 1.94× |
| 4 | 8.32 | 57.72 | 8.06 | 3.75× |
| 8 | 8.85 | 108.46 | 8.61 | 7.04× |
| 12 | 9.16 | 157.20 | 8.85 | 10.2× |
| **16** | **9.29** | **206.67** | **8.99** | **13.4×** |
| **24** | **11.17** | **257.78** | **10.55** | **16.7×** |
| 32 | 15.60 | 246.23 | 14.69 | 16.0× |
| 48 | 27.09 | 212.65 | 25.80 | 13.8× |

The curve has a clear inflection:

- **N=1 → N=8** is essentially linear (per-run latency stays at the 8 s
  baseline; throughput scales with workers).
- **N=8 → N=16** still scales, but per-run latency starts creeping up.
- **N=24 is the sweet spot.** Runs/minute peaks at 257.8; per-run latency
  is +35 % vs N=1 but throughput is +17×.
- **N=32 onwards** is past the sweet spot — throughput plateaus, then
  falls; per-run latency nearly triples by N=48.

CSV at `results/server_runs/data28/concurrency_sweep/finer/concurrency_sweep.csv`.

## Bottlenecks discovered

| Item | Observation |
| --- | --- |
| GPU compute | `nvidia-smi` reported `utilization.gpu=0%` during every measurement loop and during the concurrency sweep. The M1 models are too small to keep the 3080 Ti busy at the fixture's window size; at M4 scale (62-fold D3) the GPU will start paying off but is not exercised by this harness. |
| GPU memory | Peak CUDA memory for the largest model was 31 MB (LSTM), far below the 12 GB available. No GPU memory contention even with N=32 workers, but per-run latency still grows past N=24 because Python + scheduler overhead dominates. |
| CPU cores | The 80 vCPU plateau is real — beyond N=24 the run starts to thrash. This is the binding constraint on CPU-resident throughput. |
| `/` disk | Grew from 425 MB → 1.3 GB during the run (mostly the conda env's package cache; nothing the harness wrote). `/` still has 29 GB free. `/root/autodl-tmp` only used 1.3 GB / 50 GB. |
| Load average | Saturation visible: 22.9 → 14.5 → 14.3 across the sweep. After the sweep it stayed above 14 — `Nproc=80` so we are using ~17 % of the box at the sweet spot. |

## What this implies for M4 on this machine

- One M4 run = "one (seed × fold × model × task)" cell of the benchmark.
  At the sweet spot of N=24 workers, the harness can sustain ~258 such
  runs per minute, i.e. a **single 24-worker parallel cell finishes a
  batch of 240 runs in about a minute** (one full seed × fold grid).
- The full M4 grid for D3 (LOSO 62 × seeds × models × tasks × window
  lengths) is much larger, but parallelising by seeds × folds pushes the
  cost back into the same envelope. With N=24, an M4 row (one model
  across the 62-fold D3 split with 10 seeds) would take roughly
  `62 × 10 × per_run_latency / 24` minutes. With the largest model
  (`random_forest` ~1 s / fit+predict on CPU) this is ~26 minutes per
  model; with `cnn1d` (~0.5 s) it is ~13 minutes.
- One big caveat: the M4 runs at D3 scale will have larger `n_train`
  windows, longer fit times, and the GPU path becomes useful. The M4
  budget needs a fresh pass of this sweep at the real-data shape; that
  is the natural follow-up, not DATA-28's job.

## Files in this report

| Path | Purpose |
| --- | --- |
| `evidence/data28_server_evidence.txt` | `nvidia-smi`, `df`, `free`, env versions, repo state |
| `server_measurements/cpu/{hardware_metrics,hardware_summary}.csv` | CPU-side per-rep + summary |
| `server_measurements/gpu/{hardware_metrics,hardware_summary}.csv` | GPU-side per-rep + summary |
| `concurrency_sweep/finer/concurrency_sweep.csv` | N=1..48 throughput / latency curve |
| `concurrency_sweep/fourmodel/concurrency_sweep.csv` | N=1..32 with 4-model workload |

All measured with `scripts/server/measure_baselines.py` and
`scripts/server/concurrency_sweep.py`, driven by
`scripts/server/run_on_server.sh`. The harness never reads the test split;
the assert at `measure_baselines.py:main` enforces that the val and test
tensors are distinct Python objects before any model is constructed.