# DATA-28 server harness

This directory holds the engineering-side measurement harness for the M1
baseline zoo. It produces two deliverables:

1. **`hardware_metrics.csv`** — per-model train time, predict latency, peak
   RSS, peak GPU memory, trainable parameter count.
2. **`concurrency_sweep.csv`** — how total throughput and per-run latency
   vary as the number of parallel workers grows.

Both numbers are measured on the **validation split only**. The test split
is never read; an assertion in `measure_baselines.py` enforces that the
validation and test tensors are distinct Python objects before any model
is constructed.

## Scripts

| File | Purpose |
| --- | --- |
| `lib_common.py` | RSS polling, GPU memory sampling, env fingerprint, CSV writer |
| `measure_baselines.py` | Per-model measurement loop; one CSV per model + one summary |
| `concurrency_sweep.py` | Worker-count sweep with multiprocessing.Pool |
| `run_on_server.sh` | Top-level shell driver: rsync + measure + sweep + rsync back |

## Running on the server

```bash
# From this repo on the Mac:
scripts/server/run_on_server.sh
```

That one call will:

1. rsync the project to `/root/autodl-tmp/drososense/repo/` (the server
   cannot reach GitHub).
2. Wait for any lingering `pip install` (DATA-24) to finish so concurrent
   installs cannot corrupt the conda env.
3. Print `nvidia-smi` and `df -h /root/autodl-tmp` for an evidence trail.
4. Run `measure_baselines.py` (9 models × 3 repetitions, CPU torch=1).
5. Run `concurrency_sweep.py` over N ∈ {1, 4, 8, 16, 32} workers.
6. Print `nvidia-smi` and `df -h` again.
7. rsync only the CSVs back into `results/server_runs/<run_id>/`.

Run from any subdirectory. The script never writes to `/`; everything lives
under `/root/autodl-tmp/drososense/` on the server.

## Measurement contract

Every reported number is the mean of **3 repetitions** with a fresh seed
each pass. Each repetition is preceded by an un-timed warm-up fit, so the
first-call JIT / import cost does not contaminate the latency numbers. The
hardware metrics CSV carries every per-rep row so a noisy sample cannot
masquerade as a number.

Reported columns:

- `train_time_s` — fit wall-clock
- `predict_time_s` — predict wall-clock on the validation set
- `ms_per_sample` — `predict_time_s / n_val_windows × 1000`
- `peak_rss_kb` — peak resident-set size across fit + predict
- `peak_gpu_mb` — `torch.cuda.max_memory_allocated` (only for the torch
  sequence models when `--use-gpu` is given)
- `n_trainable_parameters` — model-reported count after fit
- `device` — `cpu` or `cuda:0`
- `torch_num_threads` — torch thread count pinned in this run

A model whose backend is unavailable (e.g. xgboost without libomp) is
recorded with a `failure_reason` and contributes zero rows to the summary
— it never silently disappears from the table.

## Concurrency sweep contract

The driver spawns `N` child processes via `multiprocessing.Pool`. Each
child runs `K` independent `measure_baselines.py` invocations on its own
`--output-dir`. The driver records:

- `wall_clock_s` — pool finish time
- `runs_per_minute` — `N × K / wall_clock × 60`
- `per_run_latency_s_mean` and `_std` — across the successful children
- `n_failures` — children whose subprocess exited non-zero

The sweet spot is the smallest `N` at which `runs_per_minute` has plateaued.
Beyond that point, adding more workers buys latency on the parent but no
extra measured throughput, which is exactly the inflection point the issue
asks us to flag.

`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` are pinned per child to 1 so a too-broad thread cap
inherited from the parent cannot flatten the curve by accident.

## What this harness does NOT do

- It does not run the M4 scientific experiments (E1–E12).
- It does not run any connectome reservoir (R0–R6).
- It does not change `configs/protocol_v1*.yaml` or the D3 split count.
- It does not write large intermediate state to `/` (the 30 GB root).

Those are explicitly out of scope for DATA-28.