#!/usr/bin/env python
"""Run the frozen benchmark over one dataset.

The split strategy defaults to ``auto``, which reads it from the dataset's own
config. That is deliberate: D2 can only do LOSO(5) with its five cuts, D3 has 62
fillets and uses GroupKFold, and D1 has no specimen id at all and must use the
explicitly non-compliant time-block stand-in. Defaulting at the call site would
let a run silently disagree with the dataset's declared protocol.

Examples
--------
Smoke test on the synthetic fixture (protocol-compliant split, every model whose
backend is installed)::

    python scripts/make_fixture.py
    python scripts/run_baselines.py --dataset synthetic_enose --experiment smoke \\
        --models all --tasks classification regression --seeds 0 1 \\
        --window-lengths 16 --max-folds 1 --smoke

The compliant M1 benchmark (D2 and D3 satisfy split_unit: specimen)::

    python scripts/run_baselines.py --dataset d2_beef_uncontrolled \\
        --experiment m1_benchmark --models all --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/run_baselines.py --dataset d3_rainbow_trout \\
        --experiment m1_benchmark --models all --seeds 0 1 2 3 4 5 6 7 8 9

D1's split is a documented time-block stand-in, so every record it produces is
flagged non-compliant and is excluded from every gate::

    python scripts/run_baselines.py --dataset d1_beef_controlled \\
        --experiment m1_real_validation --models all --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.baselines.registry import MODEL_IDS  # noqa: E402
from drososense.data.loaders import dataset_config_path  # noqa: E402
from drososense.evaluation.runner import BenchmarkConfig, describe_models, run_benchmark  # noqa: E402
from drososense.evaluation.selection import HyperparameterGrid, OutOfGridError  # noqa: E402
from drososense.utils.seeding import load_seed_policy  # noqa: E402

# Deliberately tiny hyperparameters for smoke runs, so the pipeline is exercised
# end to end in seconds rather than hours. These are NOT the tuning configs used
# for reported results.
SMOKE_PARAMS: dict[str, dict] = {
    "random_forest": {"n_estimators": 40},
    "xgboost": {"n_estimators": 40, "max_depth": 4},
    "pca_svm": {"n_components": 16},
    "gru": {"hidden_size": 16, "epochs": 5, "batch_size": 64},
    "lstm": {"hidden_size": 16, "epochs": 5, "batch_size": 64},
    "cnn1d": {"hidden_size": 16, "epochs": 5, "batch_size": 64},
    "tcn": {"hidden_size": 16, "num_layers": 2, "epochs": 5, "batch_size": 64},
    "esn": {"reservoir_size": 150, "density": 0.08, "washout": 2},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. synthetic_enose")
    parser.add_argument("--experiment", default="m1_benchmark", help="experiment label")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"'all' or a subset of {list(MODEL_IDS)}",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["classification", "regression"],
        choices=["classification", "regression"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--window-lengths", nargs="+", type=int, default=[16])
    parser.add_argument(
        "--split-strategy",
        default="auto",
        choices=["auto", "group_kfold", "loso", "time_block_holdout"],
        help="'auto' reads split.strategy from the dataset config; time_block_holdout is NOT "
        "protocol-compliant and exists only for D1's documented stand-in",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="ignored when --split-strategy auto, which takes the config's value",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--label-rule", default="last", choices=["last", "majority"],
        help="how a window's label is reduced from its timesteps",
    )
    parser.add_argument("--max-folds", type=int, default=None, help="cap folds per seed")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help=(
            "E3 low-data: fraction of each fold's TRAIN-side specimen pool admitted. "
            "Test and validation specimens are untouched, so the test set is "
            "identical across 10/25/50/75/100%. Sampling is nested within each fold "
            "(for a fixed seed, the 10% pool is contained in the 25% pool, contained in "
            "the 50% pool, up to 100% which is byte-for-byte the E1/E2 train set). "
            "The pool is always a subset of the fold's own TRAIN side, so no fold is "
            "ever admitted its test/val specimen and no train side is emptied. "
            "Epochs / hyperparameters are NOT scaled with the fraction. "
            "Must satisfy 0 < f <= 1; omit for the full E1/E2 behaviour."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="use deliberately tiny hyperparameters (pipeline check, not tuning)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="print model availability and exit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    # DATA-61 defect 3: E3 batches fan out (5 parallel subprocesses); without
    # pinned BLAS limits the v6 hang returns — 128 threads × 5 procs on an
    # 80-core box, with nothing in the record to evidence it. Validate the
    # threading knobs at start (before any worker could initialise BLAS);
    # when they are unset, pin them in-process so the record's ENV_* keys
    # evidence the limits the batch actually ran under.
    from ops import thread_limits

    try:
        thread_limits.validate_thread_limits(max_threads=8)
    except RuntimeError:
        thread_limits.set_thread_limits(thread_limits.DEFAULT_LIMIT)

    if args.list_models:
        print(json.dumps(describe_models(), indent=2))
        return 0

    config_path = dataset_config_path(args.dataset)
    if not config_path.is_file():
        print(f"no dataset config at {config_path}", file=sys.stderr)
        return 2

    models = tuple(MODEL_IDS) if args.models == ["all"] else tuple(args.models)
    unknown = [m for m in models if m not in MODEL_IDS]
    if unknown:
        print(f"unknown models {unknown}; registered: {list(MODEL_IDS)}", file=sys.stderr)
        return 2

    # The declared design is checked here as well as in run_benchmark, so a bad
    # argument is a message about the argument rather than a traceback from
    # three frames down. run_benchmark keeps its own check: it is the entry both
    # this script and the tests go through, and a rule enforced only at a call
    # site is a rule the next call site does not have.
    try:
        load_seed_policy().validate(args.seeds)
        HyperparameterGrid.from_protocol().check_window_lengths(args.window_lengths)
    except (ValueError, OutOfGridError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        print(
            f"declared seeds: {list(load_seed_policy().root_seeds)} (extendable to "
            f"{load_seed_policy().extension_to} as a whole block); declared window lengths: "
            f"{list(HyperparameterGrid.from_protocol().window_length_candidates)}",
            file=sys.stderr,
        )
        return 2

    # E3 low-data: the declared fractions are the protocol's; reject anything
    # else so a batch cannot silently run on an undeclared split. 1.0 is
    # accepted as a no-op alias for "no subsampling" (byte-identical E1).
    if args.train_fraction is not None and not 0.0 < args.train_fraction <= 1.0:
        print(f"--train-fraction must satisfy 0 < f <= 1, got {args.train_fraction}", file=sys.stderr)
        return 2
    if args.train_fraction == 1.0:
        args.train_fraction = None

    config = BenchmarkConfig(
        dataset_id=args.dataset,
        experiment=args.experiment,
        models=models,
        tasks=tuple(args.tasks),
        seeds=tuple(args.seeds),
        window_lengths=tuple(args.window_lengths),
        split_strategy=args.split_strategy,
        n_splits=args.n_splits,
        stride=args.stride,
        label_rule=args.label_rule,
        model_params=SMOKE_PARAMS if args.smoke else {},
        max_folds=args.max_folds,
        train_fraction=args.train_fraction,
    )

    summary = run_benchmark(config)

    skipped = summary.attrs.get("skipped_models", [])
    if skipped:
        print("models skipped (backend unavailable):", file=sys.stderr)
        for entry in skipped:
            print(f"  - {entry['model']}: {entry['reason']}", file=sys.stderr)

    # DATA-51 skip disclosure: units already held by a status == "ok" record
    # (same config re-shard, or a different-config prior ok record) were SKIPPED,
    # not re-scored and not aborted. The full list lands on disk next to the
    # summary as <experiment>_skip_disclosure.json; this block is the
    # operator-visible receipt so a re-shard that "ran" nothing still reports
    # exactly what it skipped and under which prior config hashes.
    skipped_units = summary.attrs.get("skipped_units", [])
    if skipped_units:
        n_same = summary.attrs.get("n_skipped_same_config", 0)
        n_diff = summary.attrs.get("n_skipped_different_config", 0)
        print(
            f"skipped units: {len(skipped_units)} total "
            f"({n_same} same-config re-computations, {n_diff} different-config prior ok records)",
            file=sys.stderr,
        )
        distinct_prior = sorted({d["prior_config_hash"] for d in skipped_units})
        if distinct_prior:
            print(f"  prior config_hash(es): {', '.join(distinct_prior)}", file=sys.stderr)
        for entry in skipped_units:
            print(
                f"  - {entry['dataset']} {entry['model']}/{entry['task']} "
                f"seed{entry['seed']:02d} fold{entry['fold_id']:02d} w{entry['window_length']} "
                f"[{entry['reason']}]: prior ok config {entry['prior_config_hash']} "
                f"(run {entry['prior_run_id']}), this batch {entry['run_config_hash']}",
                file=sys.stderr,
            )
        print(
            f"  full disclosure: results/tables/{args.experiment}_skip_disclosure.json",
            file=sys.stderr,
        )

    if summary.empty and not summary.attrs.get("skipped_units"):
        print("no results produced", file=sys.stderr)
        return 1

    if summary.empty:
        print(
            "no NEW results produced — every requested unit was skipped (already "
            "held by a status == 'ok' record). See the skip disclosure above.",
            file=sys.stderr,
        )
        return 0

    columns = [
        c
        for c in ("dataset", "model", "task", "window_length", "macro_f1_mean", "macro_f1_std",
                  "balanced_accuracy_mean", "auroc_mean", "n_auroc_defined", "mae_mean",
                  "rmse_mean", "r2_mean", "protocol_compliant", "n_seeds", "n_runs")
        if c in summary.columns
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nraw records: results/raw/{args.experiment}/")
    print(f"summary:     results/tables/{args.experiment}_summary.csv")
    environment = summary.attrs.get("environment_report", {})
    if environment and not environment.get("complete", True):
        print(
            f"\nNOTE — this run's environment is not the declared one "
            f"(missing: {', '.join(environment.get('missing', []))}). Install the missing "
            f"backends or report the result for this environment explicitly; see "
            f"`python -m drososense.utils.env_report`."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
