#!/usr/bin/env python
"""Run the frozen benchmark over one dataset.

Examples
--------
Smoke test on the synthetic fixture (protocol-compliant split, all nine models)::

    python scripts/make_fixture.py
    python scripts/run_baselines.py --dataset synthetic_enose --experiment smoke \\
        --models all --tasks classification regression --seeds 0 1 \\
        --window-lengths 16 --max-folds 1 --smoke

Real-data pipeline validation (D1 publishes no specimen ids, so the split is a
documented time-block stand-in and every record is flagged non-compliant)::

    python scripts/run_baselines.py --dataset d1_beef_controlled \\
        --experiment m1_real_validation --models all \\
        --tasks classification regression --seeds 0 --window-lengths 16 --max-folds 1
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
        default="group_kfold",
        choices=["group_kfold", "loso", "time_block_holdout"],
        help="time_block_holdout is NOT protocol-compliant; use it only for pipeline validation",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--label-rule", default="last", choices=["last", "majority"],
        help="how a window's label is reduced from its timesteps",
    )
    parser.add_argument("--max-folds", type=int, default=None, help="cap folds per seed")
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
    )

    summary = run_benchmark(config)

    skipped = summary.attrs.get("skipped_models", [])
    if skipped:
        print("models skipped (backend unavailable):", file=sys.stderr)
        for entry in skipped:
            print(f"  - {entry['model']}: {entry['reason']}", file=sys.stderr)

    if summary.empty:
        print("no results produced", file=sys.stderr)
        return 1

    columns = [
        c
        for c in ("dataset", "model", "task", "window_length", "macro_f1_mean", "macro_f1_std",
                  "balanced_accuracy_mean", "auroc_mean", "mae_mean", "rmse_mean", "r2_mean",
                  "protocol_compliant", "n_seeds")
        if c in summary.columns
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nraw records: results/raw/{args.experiment}/")
    print(f"summary:     results/tables/{args.experiment}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
