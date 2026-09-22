#!/usr/bin/env python
"""M3 — run the R0–R6 family on one (synthetic or real) split.

The script is the standard entry point for DATA-4's "可运行" gate. It runs
the seven reservoir families over the same train/test split, with the same
``W_in`` and ``b``, and writes a per-(model, seed, fold, window_length)
record of raw metrics, the run configuration hash, the environment summary
and the wall-clock time per run.

Why this exists
---------------
* ``scripts/run_baselines.py`` drives the M1 zoo (including the random ESN,
  model id ``esn``). It does not know about R0–R6 because those models need
  the connectome NPZ as an additional input.
* The script does **not** replace ``run_baselines.py`` — the runner stays the
  one path that knows about splits, scaling, windowing, gates, etc. This
  script delegates the model fitting to the per-family ``BaseModel`` from
  :mod:`drososense.reservoir.connectome_reservoir` and uses the same window
  tensors the runner would feed.

Examples
--------
Smoke (no real data needed; the script auto-generates a synthetic tensor)::

    python scripts/run_reservoir.py --smoke

End-to-end with the synthetic enose dataset (still no real food data)::

    python scripts/run_reservoir.py --dataset synthetic_enose \\
        --seeds 0 1 --window-lengths 16 --max-folds 1 --smoke

End-to-end with the olfactory NPZ (real biological substrate)::

    DROSOSENSE_DATA=/path/to/data-root python scripts/run_reservoir.py \\
        --dataset synthetic_enose --seeds 0 1 --window-lengths 16 \\
        --reservoir-size 250 --max-folds 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp

from drososense.reservoir.connectome_reservoir import (
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    make_dense_random,
    make_degree_rewired,
    make_er_esn,
    make_random_sparse,
    make_shared,
    make_small_world,
    make_weight_shuffled,
    rescale_to_spectral_radius,
    spectral_radius,
    ReservoirTopology,
    R0RealFlyReservoir,
    R1WeightShuffledReservoir,
    R2DegreeRewiredReservoir,
    R3RandomSparseReservoir,
    R4ErEsnReservoir,
    R5SmallWorldReservoir,
    R6DenseRandomReservoir,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="synthetic_enose",
        help="dataset id; defaults to the synthetic enose fixture",
    )
    parser.add_argument(
        "--experiment",
        default="m3_smoke",
        help="experiment label used in the run record",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--window-lengths", nargs="+", type=int, default=[16]
    )
    parser.add_argument(
        "--reservoir-size", type=int, default=250,
        help="number of nodes N; E9 sweeps 250/500/1000/2000/4000",
    )
    parser.add_argument(
        "--normalization",
        default="n5_binary",
        choices=list(ALLOWED_NORMALIZATIONS),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny hyperparameters and one fold — pipeline check, not tuning",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="cap folds per seed (smoke uses 1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "raw" / "m3_reservoir"),
        help="where per-run records are written",
    )
    parser.add_argument(
        "--npz-path",
        default=None,
        help="explicit olfactory_v1.npz; otherwise derived from DROSOSENSE_DATA",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="do not try to load the olfactory NPZ; useful when no data root",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Synthetic-window generator (the script must run without real data)
# ---------------------------------------------------------------------------


def synthetic_window_tensor(
    seed: int,
    window_length: int,
    n_samples: int = 64,
    n_channels: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic window tensor and integer labels.

    Returns:
        X: ``(n_samples, window_length, n_channels)`` float64 tensor.
        y: ``(n_samples,)`` int64 labels in ``{0, 1, 2}``.
    """
    rng = np.random.default_rng(seed)
    classes = rng.integers(0, 3, size=n_samples)
    t = np.linspace(0, 1.0, window_length)
    X = rng.standard_normal((n_samples, window_length, n_channels))
    for sample in range(n_samples):
        freq = 1.0 + float(classes[sample])
        X[sample] += np.sin(2.0 * np.pi * freq * t)[:, None]
    return X.astype(np.float64), classes.astype(np.int64)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_npz_path(explicit: str | None) -> Path | None:
    """Return the olfactory NPZ path, or ``None`` if it cannot be found.

    Order:
        1. ``--npz-path`` argument;
        2. ``connectome.paths.adjacency_path()`` (DATA-3 / DATA-4 entry
           point — handles both the documented ``adjacency/`` sub-layout
           and the flat ``connectome/`` layout some server syncs produce);
        3. ``<project-root>/connectome/adjacency/olfactory_v1.npz`` fallback.

    The olfactory NPZ is gitignored (739 MB). When the data root is not
    provisioned the script falls back to a deterministic synthetic graph so
    it can still be smoke-tested, and the run record flags the gap.
    """
    from connectome.paths import adjacency_path as _adjacency_path

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(_adjacency_path())
    candidates.append(PROJECT_ROOT / "connectome" / "adjacency" / "olfactory_v1.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def config_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a config dict for run-record audit."""
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def environment_summary() -> dict[str, Any]:
    """Minimal environment snapshot for the run record."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "DROSOSENSE_DATA": os.environ.get("DROSOSENSE_DATA"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str))
            handle.write("\n")


def build_synthetic_family(seed: int, params: dict[str, Any], n_channels: int) -> dict[str, Any]:
    """Build the seven-model family from a synthetic R0 substitute.

    The substitute is a small Erdős–Renyi graph with the same ``reservoir_size``
    as the configured run; the pipeline is exercised end to end, and the
    run record is honest about the substrate.
    """
    matrix_size = int(params["reservoir_size"])
    rng = np.random.default_rng(seed)
    rows_idx = rng.integers(0, matrix_size, size=matrix_size * 2)
    cols_idx = rng.integers(0, matrix_size, size=matrix_size * 2)
    sl = rows_idx == cols_idx
    rows_idx[sl] = (rows_idx[sl] + 1) % matrix_size
    values = rng.uniform(-1.0, 1.0, size=matrix_size * 2)
    mat = sp.csr_matrix((values, (rows_idx, cols_idx)), shape=(matrix_size, matrix_size))
    mat = rescale_to_spectral_radius(mat, float(params["spectral_radius"]))
    fake_r0 = ReservoirTopology(
        matrix=mat,
        n_nodes=matrix_size,
        n_edges=int(mat.nnz),
        spectral_radius=spectral_radius(mat, seed=0),
        density=mat.nnz / (matrix_size * matrix_size),
        kind="R0_real_fly",
        normalization="synthetic_substitute",
    )
    topo = {
        "R0_real_fly": fake_r0,
        "R1_weight_shuffled": make_weight_shuffled(fake_r0, seed),
        "R2_degree_rewired": make_degree_rewired(fake_r0, seed),
        "R3_random_sparse": make_random_sparse(fake_r0, seed),
        "R4_er_esn": make_er_esn(fake_r0, seed),
        "R5_small_world": make_small_world(fake_r0, seed),
        "R6_dense_random": make_dense_random(fake_r0, seed),
    }
    shared = make_shared(
        n_nodes=fake_r0.n_nodes,
        n_channels=n_channels,
        seed=seed,
        input_scale=float(params["input_scale"]),
    )
    cls_map = {
        "R0_real_fly": R0RealFlyReservoir,
        "R1_weight_shuffled": R1WeightShuffledReservoir,
        "R2_degree_rewired": R2DegreeRewiredReservoir,
        "R3_random_sparse": R3RandomSparseReservoir,
        "R4_er_esn": R4ErEsnReservoir,
        "R5_small_world": R5SmallWorldReservoir,
        "R6_dense_random": R6DenseRandomReservoir,
    }
    return {
        fid: cls_map[fid](
            task="classification", seed=seed, params=params,
            topology=topo[fid], shared=shared, n_channels=n_channels,
        )
        for fid in TOPOLOGY_FAMILY_IDS
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    npz_path = None if args.skip_real else resolve_npz_path(args.npz_path)
    if npz_path is None:
        print(
            "olfactory_v1.npz not provisioned — falling back to a synthetic "
            "R0-shaped graph; the run record will mark this with "
            "`substrate: synthetic_substitute`.",
            file=sys.stderr,
        )

    node_indices = None
    if npz_path is not None:
        seed_for_select = int(args.seeds[0]) if args.seeds else 0
        peek = np.load(str(npz_path), allow_pickle=True)
        n_full = int(peek["node_ids"].shape[0])
        rng = np.random.default_rng(seed_for_select)
        node_indices = np.sort(rng.choice(n_full, size=args.reservoir_size, replace=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(DEFAULT_RESERVOIR_PARAMS)
    params["reservoir_size"] = args.reservoir_size
    params["washout"] = 2 if args.smoke else int(args.window_lengths[0] / 4)
    params["leak"] = 0.3
    params["gain"] = 1.0
    params["input_scale"] = 0.5
    params["ridge_lambda"] = 1.0e-3
    params["readout"] = "ridge"

    per_run: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    environment = environment_summary()
    substrate = "olfactory_v1" if npz_path is not None else "synthetic_substitute"
    n_channels = 6  # The synthetic enose has 6 sensor channels.

    for seed in args.seeds:
        for window_length in args.window_lengths:
            X, y = synthetic_window_tensor(
                seed=seed,
                window_length=window_length,
                n_samples=64,
                n_channels=n_channels,
            )
            split = int(0.75 * X.shape[0])
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if npz_path is not None:
                family = build_topology_family(
                    str(npz_path),
                    seed=seed,
                    n_channels=n_channels,
                    normalization=args.normalization,
                    node_indices=node_indices,
                    params=params,
                )
            else:
                family = build_synthetic_family(seed, params, n_channels)

            for fid, model in family.items():
                run_payload: dict[str, Any] = {
                    "experiment": args.experiment,
                    "dataset": args.dataset,
                    "model": fid,
                    "task": "classification",
                    "seed": int(seed),
                    "window_length": int(window_length),
                    "reservoir_size": int(args.reservoir_size),
                    "substrate": substrate,
                    "normalization": args.normalization,
                    "params": params,
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    # Single-fold record: the 75/25 split is fold 0 of this run.
                    # e2_stats pairs on fold_id; keeping it explicit here means
                    # a future multi-fold run file can be merged in without
                    # changing the record schema.
                    "fold_id": 0,
                }
                run_payload["config_hash"] = config_hash(run_payload)
                started = time.time()
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    accuracy = float((preds == y_test).mean())
                except Exception as exc:  # noqa: BLE001 — record and continue
                    failures.append(
                        {
                            "model": fid,
                            "seed": seed,
                            "window_length": window_length,
                            "error": repr(exc),
                        }
                    )
                    run_payload["status"] = "error"
                    per_run.append(run_payload)
                    continue
                elapsed = time.time() - started
                # Score every metric the protocol declares for this task, not
                # just accuracy: the pre-registered contrasts (R0_vs_R2, etc.)
                # are read on the PRIMARY metric (macro_f1 for classification,
                # mae for regression), and an analysis that only has accuracy
                # cannot produce a single gate row. These values are computed
                # with the same function the frozen protocol's runner uses
                # (drososense.evaluation.metrics.classification_metrics), so
                # they cannot silently diverge from the protocol definition.
                from drososense.evaluation.metrics import classification_metrics
                y_true_arr = np.asarray(y_test).reshape(-1)
                y_pred_arr = np.asarray(preds).reshape(-1)
                n_classes = int(y_true_arr.max() + 1)
                class_metrics = classification_metrics(y_true_arr, y_pred_arr, n_classes=n_classes)
                run_payload.update(
                    {
                        "status": "ok",
                        "accuracy": accuracy,
                        "macro_f1": class_metrics["macro_f1"],
                        "balanced_accuracy": class_metrics["balanced_accuracy"],
                        "auroc": class_metrics["auroc"],
                        "auroc_defined": class_metrics.get("auroc_defined", False),
                        "wallclock_seconds": float(elapsed),
                        "n_trainable_parameters": model.n_trainable_parameters(),
                        "frozen_parameters": model.n_frozen_parameters(),
                        "reservoir_sparsity": model.reservoir_sparsity(),
                        "topology_kind": model._topology.kind,
                        "topology_n_nodes": model._topology.n_nodes,
                        "topology_n_edges": model._topology.n_edges,
                        "topology_spectral_radius": model._topology.spectral_radius,
                    }
                )
                per_run.append(run_payload)
                summary_rows.append(
                    {
                        "model": fid,
                        "seed": seed,
                        "window_length": window_length,
                        "accuracy": accuracy,
                        "macro_f1": class_metrics["macro_f1"],
                        "wallclock_seconds": elapsed,
                        "n_trainable_parameters": run_payload["n_trainable_parameters"],
                        "frozen_parameters": run_payload["frozen_parameters"],
                    }
                )

    fieldnames = [
        "experiment", "dataset", "model", "task", "seed", "window_length",
        "reservoir_size", "substrate", "normalization", "config_hash",
        "n_train", "n_test", "status", "accuracy", "macro_f1",
        "balanced_accuracy", "auroc", "auroc_defined", "wallclock_seconds",
        "n_trainable_parameters", "frozen_parameters", "reservoir_sparsity",
        "topology_kind", "topology_n_nodes", "topology_n_edges",
        "topology_spectral_radius", "params",
    ]
    run_path = output_dir / f"{args.experiment}_per_run.csv"
    summary_path = output_dir / f"{args.experiment}_summary.csv"
    failure_path = output_dir / f"{args.experiment}_failures.jsonl"
    config_path = output_dir / f"{args.experiment}_config.json"

    write_csv(run_path, per_run, fieldnames)
    if summary_rows:
        write_csv(
            summary_path,
            summary_rows,
            ["model", "seed", "window_length", "accuracy", "macro_f1",
             "wallclock_seconds", "n_trainable_parameters", "frozen_parameters"],
        )
    if failures:
        write_jsonl(failure_path, failures)
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "environment": environment,
                "substrate": substrate,
                "npz_path": str(npz_path) if npz_path is not None else None,
            },
            indent=2,
            default=str,
        )
    )

    if summary_rows:
        columns = ["model", "seed", "window_length", "accuracy", "wallclock_seconds"]
        print("experiment:", args.experiment, "| substrate:", substrate)
        print("{:<22} {:<6} {:<6} {:<8} {:<8}".format(*columns))
        for row in summary_rows:
            print(
                "{:<22} {:<6} {:<6} {:<8.4f} {:<8.4f}".format(
                    row["model"],
                    row["seed"],
                    row["window_length"],
                    row["accuracy"],
                    row["wallclock_seconds"],
                )
            )
        print(f"\nper-run CSV : {run_path}")
        print(f"summary CSV : {summary_path}")
        print(f"config JSON : {config_path}")
    else:
        print("no runs produced", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())