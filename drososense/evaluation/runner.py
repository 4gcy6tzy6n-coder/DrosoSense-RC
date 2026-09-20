"""The benchmark driver.

Loops dataset -> window length -> seed -> fold -> (model, task), and writes one
run record per evaluation. The loop order matters: the fold is built once and
every model for that fold receives the identical tensor, which is what makes the
later paired statistical tests legitimate.

Models whose backend is unavailable are skipped with a recorded reason rather
than dropped silently, so a comparison table always states what is missing from
it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drososense.baselines.base import BaseModel, ModelUnavailableError
from drososense.baselines.registry import MODEL_IDS, build_model, model_availability, model_spec
from drososense.data.loaders import dataset_config_path, load_dataset
from drososense.data.manifest import load_manifest, manifest_path
from drososense.data.pipeline import build_fold_tensors, usable_specimens
from drososense.data.splits import make_folds
from drososense.evaluation.metrics import classification_metrics, regression_metrics
from drososense.evaluation.results import (
    RunRecord,
    aggregate_records,
    capture_environment,
    make_run_id,
    utc_now_iso,
    write_record,
    write_summary_csv,
)
from drososense.utils.config import config_hash
from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir

PROTOCOL_VERSION = "1.0.0"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Everything that defines one benchmark invocation.

    Attributes:
        dataset_id: Dataset to evaluate.
        experiment: Experiment label used in paths and tables.
        models: Model ids to run.
        tasks: Tasks to run.
        seeds: Seeds to evaluate.
        window_lengths: Window lengths to evaluate.
        split_strategy: ``group_kfold``, ``loso`` or ``time_block_holdout``.
        n_splits: Number of folds for strategies that use it.
        stride: Window stride.
        label_rule: ``last`` or ``majority``.
        model_params: Per-model hyperparameters.
        max_folds: Cap on folds per seed, for smoke runs.
        evidence_class: ``real`` or ``synthetic_fixture``; overridden from the
            manifest when one exists.
    """

    dataset_id: str
    experiment: str = "smoke"
    models: tuple[str, ...] = MODEL_IDS
    tasks: tuple[str, ...] = ("classification", "regression")
    seeds: tuple[int, ...] = (0,)
    window_lengths: tuple[int, ...] = (16,)
    split_strategy: str = "group_kfold"
    n_splits: int = 5
    stride: int = 1
    label_rule: str = "last"
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_folds: int | None = None
    evidence_class: str = "real"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the config fingerprint.

        Returns:
            Mapping of the configuration.
        """
        return {
            "dataset_id": self.dataset_id,
            "experiment": self.experiment,
            "models": list(self.models),
            "tasks": list(self.tasks),
            "seeds": list(self.seeds),
            "window_lengths": list(self.window_lengths),
            "split_strategy": self.split_strategy,
            "n_splits": self.n_splits,
            "stride": self.stride,
            "label_rule": self.label_rule,
            "model_params": self.model_params,
            "max_folds": self.max_folds,
        }


def _evaluate_one(
    model: BaseModel,
    fold_tensors: Any,
    task: str,
    n_classes: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Fit and score one model on one fold.

    Args:
        model: The constructed model.
        fold_tensors: The fold's tensors.
        task: ``classification`` or ``regression``.
        n_classes: Number of classes, for the AUROC denominator.

    Returns:
        ``(metrics, predictions, scores)``.
    """
    train = fold_tensors.train
    test = fold_tensors.test

    if task == "classification":
        model.fit(train.X, train.y_class)
        predictions = model.predict(test.X)
        scores = model.predict_proba(test.X)
        metrics = classification_metrics(test.y_class, predictions, scores, n_classes=n_classes)
    else:
        model.fit(train.X, train.y_reg)
        predictions = model.predict(test.X)
        scores = None
        metrics = regression_metrics(test.y_reg, predictions)
    return metrics, predictions, scores


def run_benchmark(
    config: BenchmarkConfig,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """Run a benchmark and write raw records plus an aggregated summary.

    Args:
        config: The benchmark configuration.
        raw_dir: Override for ``results/raw``.
        tables_dir: Override for ``results/tables``.

    Returns:
        The aggregated summary frame.

    Raises:
        ValueError: If no models or tasks are selected.
    """
    if not config.models:
        raise ValueError("no models selected")
    if not config.tasks:
        raise ValueError("no tasks selected")

    dataset = load_dataset(dataset_config_path(config.dataset_id))
    n_classes = dataset.schema.n_classes

    # A manifest, when present, is authoritative about whether the data is
    # observed. The fixture must never be describable as a real result.
    evidence_class = config.evidence_class
    if manifest_path(config.dataset_id).is_file():
        manifest = load_manifest(manifest_path(config.dataset_id))
        evidence_class = "synthetic_fixture" if manifest.synthetic else "real"

    protocol_compliant = dataset.schema.protocol_compliant
    notes = ""
    if not protocol_compliant:
        notes = (
            f"split_unit: specimen is NOT satisfied — the specimen identifiers are "
            f"{dataset.schema.specimen_source.value}, not published sample ids. "
            f"This result is pipeline validation only and must not be reported as a "
            f"protocol result."
        )

    availability = model_availability()
    environment = capture_environment()
    run_config_hash = config_hash(config.as_dict())
    records: list[RunRecord] = []
    skipped: list[dict[str, str]] = []

    for window_length in config.window_lengths:
        specimens = usable_specimens(dataset, window_length)
        if len(specimens) < 3:
            raise ValueError(
                f"{config.dataset_id}: only {len(specimens)} specimens are long enough for "
                f"window_length={window_length}; cannot build folds"
            )

        for seed in config.seeds:
            folds = make_folds(
                specimens, config.split_strategy, seed=seed, n_splits=config.n_splits
            )
            if config.max_folds is not None:
                folds = folds[: config.max_folds]

            for fold in folds:
                artifact_dir = (
                    ensure_dir(RESULTS_RAW_DIR)
                    / "_artifacts"
                    / config.dataset_id
                    / f"w{window_length}_seed{seed:02d}_fold{fold.fold_id:02d}"
                )
                fold_tensors = build_fold_tensors(
                    dataset,
                    fold,
                    window_length,
                    artifact_dir,
                    stride=config.stride,
                    label_rule=config.label_rule,
                )

                for model_id in config.models:
                    if not availability.get(model_id, {}).get("available", False):
                        reason = availability.get(model_id, {}).get("reason", "unknown")
                        entry = {"model": model_id, "reason": str(reason)}
                        if entry not in skipped:
                            skipped.append(entry)
                        continue

                    for task in config.tasks:
                        started = time.perf_counter()
                        params = dict(config.model_params.get(model_id, {}))
                        model = build_model(
                            model_id,
                            task,
                            seed,
                            params,
                            n_channels=dataset.schema.n_features,
                        )
                        metrics, _, _ = _evaluate_one(model, fold_tensors, task, n_classes)
                        duration = time.perf_counter() - started

                        records.append(
                            RunRecord(
                                run_id=make_run_id(
                                    config.dataset_id, model_id, task, seed, fold.fold_id
                                ),
                                experiment=config.experiment,
                                dataset=config.dataset_id,
                                model=model_id,
                                task=task,
                                seed=seed,
                                fold_id=fold.fold_id,
                                protocol_version=PROTOCOL_VERSION,
                                window_length=window_length,
                                metrics=metrics,
                                n_train_windows=len(fold_tensors.train),
                                n_test_windows=len(fold_tensors.test),
                                train_specimens=list(fold.train),
                                test_specimens=list(fold.test),
                                duration_s=duration,
                                environment=environment,
                                timestamp_utc=utc_now_iso(),
                                evidence_class=evidence_class,
                                protocol_compliant=protocol_compliant,
                                model_description=model.describe(),
                                fold_fingerprint=fold.fingerprint,
                                class_coverage=fold_tensors.class_coverage,
                                config_hash=run_config_hash,
                                notes=notes,
                            )
                        )

    for record in records:
        write_record(record, raw_dir)

    summary = aggregate_records(records)
    if not summary.empty:
        write_summary_csv(summary, config.experiment, tables_dir)
        summary.attrs["skipped_models"] = skipped
    return summary


def skipped_models(config: BenchmarkConfig) -> list[dict[str, str]]:
    """Report requested models whose backend is unavailable.

    Args:
        config: The benchmark configuration.

    Returns:
        List of ``{"model", "reason"}`` entries.
    """
    availability = model_availability()
    return [
        {"model": model_id, "reason": str(availability.get(model_id, {}).get("reason"))}
        for model_id in config.models
        if not availability.get(model_id, {}).get("available", False)
    ]


def describe_models() -> list[dict[str, Any]]:
    """Describe every registered model and its availability.

    Returns:
        One entry per registered model.
    """
    availability = model_availability()
    return [
        {
            "model_id": model_id,
            "family": model_spec(model_id).family,
            "note": model_spec(model_id).note,
            "available": availability[model_id]["available"],
            "unavailable_reason": availability[model_id]["reason"],
        }
        for model_id in MODEL_IDS
    ]


__all__ = [
    "BenchmarkConfig",
    "describe_models",
    "run_benchmark",
    "skipped_models",
    "ModelUnavailableError",
    "RESULTS_RAW_DIR",
    "RESULTS_TABLES_DIR",
]
