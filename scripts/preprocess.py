#!/usr/bin/env python
"""Build processed tables, specimen-level splits and frozen scalers.

Splits and scalers are produced here, once, and written to disk. Training code
never builds its own split and never fits its own scaler — that is what keeps
"specimen-level split" and "train-only standardisation" from being re-litigated
inside each model.

Outputs
-------
``data/processed/<dataset_id>/dataset.csv``
    The normalised frame in canonical columns.
``data/splits/<dataset_id>/<strategy>_seed<k>/fold<i>/``
    ``split.json`` describing the partition, and ``scaler.pkl`` fitted on that
    fold's training rows only.

Example
-------
    python scripts/preprocess.py --dataset synthetic_enose --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.data.leakage import leakage_report  # noqa: E402
from drososense.data.loaders import DatasetUnavailableError, dataset_config_path, load_dataset  # noqa: E402
from drososense.data.pipeline import build_fold_tensors, usable_specimens  # noqa: E402
from drososense.data.splits import make_folds  # noqa: E402
from drososense.utils.paths import DATA_PROCESSED_DIR, DATA_SPLITS_DIR, ensure_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument(
        "--split-strategy", default="group_kfold",
        choices=["group_kfold", "loso", "time_block_holdout"],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    try:
        dataset = load_dataset(dataset_config_path(args.dataset))
    except DatasetUnavailableError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2

    processed_dir = ensure_dir(DATA_PROCESSED_DIR / args.dataset)
    dataset.frame.to_csv(processed_dir / "dataset.csv", index=False)
    print(f"processed frame: {processed_dir / 'dataset.csv'} ({len(dataset.frame)} rows)")

    if not dataset.schema.protocol_compliant:
        print(
            f"[WARNING] {args.dataset}: specimen identifiers are "
            f"{dataset.schema.specimen_source.value}, not published sample ids.\n"
            f"          Every split built here is NON-COMPLIANT with split_unit: specimen "
            f"and may only be used for pipeline validation.",
            file=sys.stderr,
        )

    specimens = usable_specimens(dataset, args.window_length)
    print(f"specimens usable at window_length={args.window_length}: {len(specimens)}")

    written = 0
    for seed in args.seeds:
        folds = make_folds(specimens, args.split_strategy, seed=seed, n_splits=args.n_splits)
        for fold in folds:
            fold_dir = ensure_dir(
                DATA_SPLITS_DIR
                / args.dataset
                / f"{args.split_strategy}_w{args.window_length}_seed{seed:02d}"
                / f"fold{fold.fold_id:02d}"
            )
            tensors = build_fold_tensors(dataset, fold, args.window_length, fold_dir)

            record = {
                "dataset": args.dataset,
                "strategy": fold.strategy,
                "seed": seed,
                "fold_id": fold.fold_id,
                "fold_fingerprint": fold.fingerprint,
                "window_length": args.window_length,
                "protocol_compliant": dataset.schema.protocol_compliant,
                "train_specimens": list(fold.train),
                "val_specimens": list(fold.val),
                "test_specimens": list(fold.test),
                "n_train_windows": len(tensors.train),
                "n_val_windows": len(tensors.val),
                "n_test_windows": len(tensors.test),
                "scaler": {"n_fit_rows": tensors.scaler.n_fit_rows, "eps": tensors.scaler.eps},
                "class_coverage": tensors.class_coverage,
                "row_counts": leakage_report(dataset.frame, fold),
            }
            (fold_dir / "split.json").write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            written += 1

    print(f"wrote {written} fold directories under {DATA_SPLITS_DIR / args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
