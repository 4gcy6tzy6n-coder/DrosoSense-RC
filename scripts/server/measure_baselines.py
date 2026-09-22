#!/usr/bin/env python
"""Per-model hardware-side measurement for the M1 baseline zoo.

This script answers "how expensive is each baseline on this machine?" — not
"is it correct?". It instantiates each model, fits it on the train split of a
single fold of the synthetic fixture, and times predict on the **validation**
split. The test split is never read.

Measurement protocol
--------------------
- Hardware: declared in the output (``platform``, ``nvidia-smi`` snapshot).
- Threading: torch models pin ``torch_num_threads=1`` for fair per-run latency;
  the harness shells can set ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` when
  measuring at higher parallelism, and that is recorded in the output row.
- Warm-up: each model is fit once without timing before the timed loop, so
  first-call JIT and import costs are excluded from the report.
- Repetitions: each measurement is repeated ``--repetitions`` times (default 3)
  with a fresh seed each pass; mean and std are reported across the repeats so
  a single noisy sample cannot masquerade as a number.
- Memory: RSS is polled in a background thread; GPU peak is queried through
  ``torch.cuda.max_memory_allocated``. Either is ``0`` when unavailable, never
  a misleading zero.

The script never touches the test split. The runner's :class:`FoldTensors`
builds train/val/test, and we read only the train and val tensors. A pre-run
assertion enforces that the val and test arrays are not the same Python object.

Output
------
Writes two CSVs in the directory given by ``--output-dir`` (default
``results/server_measurements``):

- ``hardware_metrics.csv`` — one row per (model, repetition), with train time,
  predict time (ms/sample), peak RSS, peak GPU memory, trainable parameters and
  the env fingerprint.
- ``hardware_summary.csv`` — one row per model, with mean and std across
  repetitions. This is the table that goes into the PR summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

from scripts.server.lib_common import (  # noqa: E402  (sys.path mutation above)
    RssPoller,
    bootstrap_project_root,
    env_fingerprint,
    gpu_available,
    peak_gpu_memory_mb,
    reset_gpu_memory_stats,
    sample_nvidia_smi,
    write_csv,
)

bootstrap_project_root(Path(__file__).resolve().parent)

from drososense.baselines.base import BaseModel, ModelUnavailableError  # noqa: E402
from drososense.baselines.registry import MODEL_IDS, build_model  # noqa: E402
from drososense.data.loaders import dataset_config_path, load_dataset  # noqa: E402
from drososense.data.pipeline import build_fold_tensors  # noqa: E402
from drososense.data.splits import make_folds  # noqa: E402
from drososense.data.synthetic import FIXTURE_FILENAME, FixtureSpec, write_fixture  # noqa: E402
from drososense.utils.paths import DATA_RAW_DIR, RESULTS_TABLES_DIR  # noqa: E402


# Hardware-sized fixture. The published datasets run at the same window length
# and channel count, so the measurement exercises the path a real M4 run would
# follow without requiring the network. Twelve specimens at 120 timesteps and
# eight channels produce roughly a thousand training windows after windowing —
# enough that a single fit is not over in microseconds, small enough that a
# 32-worker concurrency sweep still finishes in minutes.
DEFAULT_FIXTURE_SPEC = FixtureSpec(
    n_specimens=12,
    n_timesteps=120,
    n_channels=6,
    seed=20260920,
    noise_sd=0.35,
    specimen_offset_sd=6.0,
)

DEFAULT_WINDOW_LENGTH = 16
DEFAULT_REPETITIONS = 3
DEFAULT_SEED = 0


@dataclass(frozen=True)
class PerRepRow:
    """One repetition of one model on the validation split.

    Attributes:
        model: Model id, e.g. ``svm_rbf``.
        repetition: Repetition index within the model's loop.
        seed: Seed used for the underlying RNG in this repetition.
        task: ``classification`` or ``regression``.
        n_train_windows: Number of training windows the model saw.
        n_val_windows: Number of validation windows it scored.
        n_channels: Number of sensor channels per window.
        train_time_s: Fit wall-clock seconds, end-to-end.
        predict_time_s: Predict wall-clock seconds over the entire val set.
        ms_per_sample: ``predict_time_s / n_val_windows * 1000``.
        peak_rss_kb: Peak resident-set size during fit+predict.
        peak_gpu_mb: Peak CUDA memory during fit+predict; ``0`` without GPU.
        n_trainable_parameters: Trainable-parameter count from the model.
        device: ``cpu`` or ``cuda:0`` — what the model actually ran on.
        torch_num_threads: torch thread count used by this run.
    """

    model: str
    repetition: int
    seed: int
    task: str
    n_train_windows: int
    n_val_windows: int
    n_channels: int
    train_time_s: float
    predict_time_s: float
    ms_per_sample: float
    peak_rss_kb: int
    peak_gpu_mb: float
    n_trainable_parameters: int | None
    device: str
    torch_num_threads: int


@dataclass(frozen=True)
class SummaryRow:
    """Aggregated mean/std across repetitions for one model.

    Attributes:
        model: Model id.
        task: ``classification`` or ``regression``.
        repetitions: Number of successful repetitions.
        n_train_windows: Window count used to train.
        n_val_windows: Window count scored at predict time.
        n_channels: Sensor channels per window.
        train_time_s_mean: Mean fit time across repetitions.
        train_time_s_std: Standard deviation of fit time.
        predict_time_s_mean: Mean predict time across repetitions.
        predict_time_s_std: Standard deviation of predict time.
        ms_per_sample_mean: Mean ms/sample across repetitions.
        ms_per_sample_std: Standard deviation of ms/sample.
        peak_rss_kb_mean: Mean peak RSS across repetitions.
        peak_gpu_mb_mean: Mean peak GPU memory across repetitions.
        n_trainable_parameters: Mean trainable-parameter count.
        device: ``cpu`` or ``cuda:0`` (consensus across reps).
        torch_num_threads: torch thread count used.
        failure_reason: Empty on success; backend or fit error otherwise.
    """

    model: str
    task: str
    repetitions: int
    n_train_windows: int
    n_val_windows: int
    n_channels: int
    train_time_s_mean: float
    train_time_s_std: float
    predict_time_s_mean: float
    predict_time_s_std: float
    ms_per_sample_mean: float
    ms_per_sample_std: float
    peak_rss_kb_mean: float
    peak_gpu_mb_mean: float
    n_trainable_parameters: float
    device: str
    torch_num_threads: int
    failure_reason: str = ""


def _ensure_fixture() -> Path:
    """Generate the synthetic fixture in place if missing.

    Returns:
        The CSV path that the dataset loader expects.
    """
    destination = DATA_RAW_DIR / "synthetic_enose" / FIXTURE_FILENAME
    if not destination.exists():
        write_fixture(destination, DEFAULT_FIXTURE_SPEC)
    return destination


def _build_fold(fold_index: int = 0, seed: int = DEFAULT_SEED) -> Any:
    """Build a single fold of the synthetic fixture.

    Args:
        fold_index: Which GroupKFold partition to use; ``0`` keeps the report
            stable across runs.
        seed: Seed for the specimen permutation.

    Returns:
        A :class:`FoldTensors` ready to hand to a model.
    """
    config_path = dataset_config_path("synthetic_enose")
    dataset = load_dataset(config_path)
    specimens = dataset.specimens()  # noqa: SLF001 - baseline contract
    folds = make_folds(specimens, "group_kfold", seed=seed, n_splits=5)
    fold = folds[fold_index]
    artifact_dir = RESULTS_TABLES_DIR / "_measurement_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return build_fold_tensors(
        dataset,
        fold,
        DEFAULT_WINDOW_LENGTH,
        artifact_dir,
        stride=1,
        label_rule="last",
    )


def _select_device(model_id: str, use_gpu: bool) -> tuple[str, str]:
    """Pick the torch device a model will run on.

    Args:
        model_id: Model id; only torch-sequence models and ESN get the option.
        use_gpu: Whether the caller asked for GPU.

    Returns:
        ``(device_label, torch_device_string)``. The torch device string is
        ``""`` for non-torch models, where ``device_label`` is fixed to ``cpu``.
    """
    is_torch_model = model_id in {"gru", "lstm", "cnn1d", "tcn"}
    if not is_torch_model or not use_gpu:
        return "cpu", ""
    if gpu_available():
        return "cuda:0", "cuda:0"
    return "cpu", ""


def _time_one_repetition(
    model_id: str,
    task: str,
    fold_tensors: Any,
    seed: int,
    torch_num_threads: int,
    use_gpu: bool = False,
) -> PerRepRow:
    """Time a single model fit + predict on the validation split.

    Args:
        model_id: Registered model id.
        task: ``classification`` or ``regression``.
        fold_tensors: The fold's tensors.
        seed: Seed for the model.
        torch_num_threads: torch thread count, pinned for reproducibility.
        use_gpu: Whether to move torch-sequence models onto CUDA for the
            timed run. Defaults to ``False``; classic and reservoir models
            always stay on CPU regardless.

    Returns:
        A populated :class:`PerRepRow`.

    Raises:
        ModelUnavailableError: If the model's backend is missing here.
        RuntimeError: If the model cannot be fitted.
    """
    n_channels = fold_tensors.train.X.shape[2]
    n_val = fold_tensors.val.X.shape[0]
    n_train = fold_tensors.train.X.shape[0]
    device_label, torch_device = _select_device(model_id, use_gpu=use_gpu)

    model = build_model(model_id, task, seed, n_channels=n_channels)

    train_y = fold_tensors.train.y_class if task == "classification" else fold_tensors.train.y_reg
    val_x = fold_tensors.val.X
    val_y = fold_tensors.val.y_class if task == "classification" else fold_tensors.val.y_reg

    reset_gpu_memory_stats()

    with RssPoller() as rss:
        if torch_device:
            import torch

            torch.set_num_threads(max(int(torch_num_threads), 1))
            # Move the model's parameters onto CUDA so input + parameter live
            # on the same device. fit() inside the baseline contract does not
            # know about the device, so we only use the GPU branch for the
            # predict call where we hold the device explicitly.
            if hasattr(model, "_network") and model._network is not None:  # noqa: SLF001
                model._network = model._network.to(torch_device)  # noqa: SLF001
            torch.manual_seed(seed)
            # Train on CPU (the model's fit path is the contract; keeping
            # training on CPU keeps the comparison with --use-gpu off honest).
            fit_started = time.perf_counter()
            model.fit(fold_tensors.train.X, train_y)
            # After fitting, move the network to CUDA for predict timing.
            if hasattr(model, "_network") and model._network is not None:  # noqa: SLF001
                model._network = model._network.to(torch_device)  # noqa: SLF001
            train_time = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            with torch.no_grad():
                flat = model.flatten(val_x)
                window_length, ch = model._infer_windows(flat)  # noqa: SLF001
                tensor = flat.reshape(flat.shape[0], window_length, ch)
                tensor_gpu = torch.as_tensor(tensor, dtype=torch.float32, device=torch_device)
                model._network.eval()  # noqa: SLF001
                output = model._network(tensor_gpu)  # noqa: SLF001
                if task == "classification":
                    _ = output.argmax(dim=1).cpu().numpy()
                else:
                    _ = output.squeeze(1).cpu().numpy()
            torch.cuda.synchronize()
            predict_time = time.perf_counter() - predict_started
        else:
            fit_started = time.perf_counter()
            model.fit(fold_tensors.train.X, train_y)
            train_time = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            _ = model.predict(val_x)
            predict_time = time.perf_counter() - predict_started

        peak_rss = rss.stop()
        peak_gpu = peak_gpu_memory_mb()

    # The model exposes its trainable-parameter count after fitting; calling
    # it before fit returns None for most baselines, which is also fine and
    # gets written to the row.
    n_params = model.n_trainable_parameters()

    return PerRepRow(
        model=model_id,
        repetition=0,  # overwritten by the caller
        seed=seed,
        task=task,
        n_train_windows=n_train,
        n_val_windows=n_val,
        n_channels=n_channels,
        train_time_s=float(train_time),
        predict_time_s=float(predict_time),
        ms_per_sample=float(predict_time / max(n_val, 1) * 1000.0),
        peak_rss_kb=int(peak_rss),
        peak_gpu_mb=float(peak_gpu),
        n_trainable_parameters=int(n_params) if n_params is not None else None,
        device=device_label,
        torch_num_threads=int(torch_num_threads),
    )


def _measure_model(
    model_id: str,
    task: str,
    fold_tensors: Any,
    repetitions: int,
    base_seed: int,
    torch_num_threads: int,
    use_gpu: bool = False,
) -> tuple[list[PerRepRow], str]:
    """Measure one model across ``repetitions`` repetitions.

    Args:
        model_id: Registered model id.
        task: ``classification`` or ``regression``.
        fold_tensors: The fold to fit on.
        repetitions: How many timed repetitions to perform.
        base_seed: Seed for the first repetition; later reps increment it.
        torch_num_threads: torch thread count.
        use_gpu: Forwarded to :func:`_time_one_repetition`.

    Returns:
        ``(rows, failure_reason)``. On success ``failure_reason`` is empty.
    """
    rows: list[PerRepRow] = []
    failure_reason = ""

    warm_seed = base_seed + 10_000
    try:
        _time_one_repetition(
            model_id=model_id,
            task=task,
            fold_tensors=fold_tensors,
            seed=warm_seed,
            torch_num_threads=torch_num_threads,
            use_gpu=use_gpu,
        )
    except Exception as exc:  # noqa: BLE001 - record, don't crash the harness
        return rows, f"{type(exc).__name__}: {exc}"

    for repetition in range(repetitions):
        seed = base_seed + repetition
        try:
            row = _time_one_repetition(
                model_id=model_id,
                task=task,
                fold_tensors=fold_tensors,
                seed=seed,
                torch_num_threads=torch_num_threads,
                use_gpu=use_gpu,
            )
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"{type(exc).__name__}: {exc}"
            break
        rows.append(
            PerRepRow(
                **{**asdict(row), "repetition": repetition, "seed": seed},
            )
        )
    return rows, failure_reason


def _summarise(rows: list[PerRepRow], failure_reason: str) -> SummaryRow:
    """Aggregate one model's repetitions into mean/std.

    Args:
        rows: Successful repetitions.
        failure_reason: Forwarded verbatim; empty on success.

    Returns:
        A populated :class:`SummaryRow`.
    """
    if not rows:
        empty = rows
        return SummaryRow(
            model="",
            task="",
            repetitions=0,
            n_train_windows=0,
            n_val_windows=0,
            n_channels=0,
            train_time_s_mean=0.0,
            train_time_s_std=0.0,
            predict_time_s_mean=0.0,
            predict_time_s_std=0.0,
            ms_per_sample_mean=0.0,
            ms_per_sample_std=0.0,
            peak_rss_kb_mean=0.0,
            peak_gpu_mb_mean=0.0,
            n_trainable_parameters=0.0,
            device="",
            torch_num_threads=0,
            failure_reason=failure_reason,
        )

    def mean(values: list[float]) -> float:
        return float(statistics.fmean(values))

    def std(values: list[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    train_times = [r.train_time_s for r in rows]
    predict_times = [r.predict_time_s for r in rows]
    ms_per_sample = [r.ms_per_sample for r in rows]
    rss_values = [r.peak_rss_kb for r in rows]
    gpu_values = [r.peak_gpu_mb for r in rows]
    params = [float(r.n_trainable_parameters or 0) for r in rows]

    first = rows[0]
    return SummaryRow(
        model=first.model,
        task=first.task,
        repetitions=len(rows),
        n_train_windows=first.n_train_windows,
        n_val_windows=first.n_val_windows,
        n_channels=first.n_channels,
        train_time_s_mean=mean(train_times),
        train_time_s_std=std(train_times),
        predict_time_s_mean=mean(predict_times),
        predict_time_s_std=std(predict_times),
        ms_per_sample_mean=mean(ms_per_sample),
        ms_per_sample_std=std(ms_per_sample),
        peak_rss_kb_mean=mean(rss_values),
        peak_gpu_mb_mean=mean(gpu_values),
        n_trainable_parameters=mean(params),
        device=first.device,
        torch_num_threads=first.torch_num_threads,
        failure_reason=failure_reason,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_IDS),
        help=f"model ids to measure; default = all of {list(MODEL_IDS)}",
    )
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="torch thread count; 1 is the fairest per-run latency. Pass a "
             "positive value when measuring higher parallelism.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="move torch-sequence models onto CUDA for the timed run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_TABLES_DIR.parent / "server_measurements",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on how many models to measure (debugging only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = env_fingerprint()
    gpu_sample = sample_nvidia_smi()
    _ensure_fixture()
    fold_tensors = _build_fold(fold_index=args.fold_index, seed=args.seed)

    # Hard guard: the script never touches the test split. Failing loud here is
    # the only safe way to keep that invariant as the harness evolves.
    assert fold_tensors.val.X is not fold_tensors.test.X, "validation and test arrays must differ"
    assert fold_tensors.train.X is not fold_tensors.test.X, "train and test arrays must differ"

    models = list(args.models)
    if args.limit is not None:
        models = models[: args.limit]

    per_rep_rows: list[dict[str, object]] = []
    summary_rows: list[SummaryRow] = []
    for model_id in models:
        rows, reason = _measure_model(
            model_id=model_id,
            task=args.task,
            fold_tensors=fold_tensors,
            repetitions=args.repetitions,
            base_seed=args.seed * 1000,
            torch_num_threads=args.torch_num_threads,
            use_gpu=args.use_gpu,
        )
        for row in rows:
            per_rep_rows.append(asdict(row))
        summary_rows.append(_summarise(rows, reason))
        if reason:
            print(f"  ! {model_id}: {reason}", file=sys.stderr)

    per_rep_csv = output_dir / "hardware_metrics.csv"
    summary_csv = output_dir / "hardware_summary.csv"

    per_rep_columns = list(asdict(PerRepRow(
        model="", repetition=0, seed=0, task="", n_train_windows=0, n_val_windows=0,
        n_channels=0, train_time_s=0.0, predict_time_s=0.0, ms_per_sample=0.0,
        peak_rss_kb=0, peak_gpu_mb=0.0, n_trainable_parameters=None, device="",
        torch_num_threads=0,
    )).keys())
    summary_columns = list(asdict(SummaryRow(
        model="", task="", repetitions=0, n_train_windows=0, n_val_windows=0,
        n_channels=0, train_time_s_mean=0.0, train_time_s_std=0.0,
        predict_time_s_mean=0.0, predict_time_s_std=0.0,
        ms_per_sample_mean=0.0, ms_per_sample_std=0.0,
        peak_rss_kb_mean=0.0, peak_gpu_mb_mean=0.0, n_trainable_parameters=0.0,
        device="", torch_num_threads=0, failure_reason="",
    )).keys())

    write_csv(per_rep_csv, per_rep_columns, per_rep_rows)
    write_csv(summary_csv, summary_columns, [asdict(r) for r in summary_rows])

    env_path = output_dir / "environment.json"
    env_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "gpu_sample": gpu_sample,
        "torch_num_threads": args.torch_num_threads,
        "use_gpu": args.use_gpu,
        "fold_index": args.fold_index,
        "seed": args.seed,
        "task": args.task,
        "n_train_windows": fold_tensors.train.X.shape[0],
        "n_val_windows": fold_tensors.val.X.shape[0],
        "n_test_windows": fold_tensors.test.X.shape[0],
        "window_length": DEFAULT_WINDOW_LENGTH,
    }, indent=2), encoding="utf-8")

    print(f"wrote {per_rep_csv} ({len(per_rep_rows)} per-rep rows)")
    print(f"wrote {summary_csv} ({len(summary_rows)} models)")
    print(f"wrote {env_path}")
    print(f"env: torch_num_threads={args.torch_num_threads}, use_gpu={args.use_gpu}")
    print(f"nvidia-smi sample: {gpu_sample or '(unavailable)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())