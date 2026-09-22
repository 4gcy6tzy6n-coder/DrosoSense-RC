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

import json
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
from drososense.data.splits import make_folds, nested_train_fraction
from drososense.evaluation.contact_log import record_contact
from drososense.evaluation.metrics import (
    EMPTY_CLASS_POLICY,
    classification_metrics,
    regression_metrics,
)
from drososense.evaluation.results import (
    RunRecord,
    aggregate_records,
    capture_environment,
    load_records,
    make_run_id,
    make_test_fingerprint,
    utc_now_iso,
    write_record,
    write_summary_csv,
)
from drososense.evaluation.selection import (
    HyperparameterGrid,
    SelectionSpec,
    select_hyperparameters,
    validate_run_hyperparameters,
)
from drososense.utils.config import config_hash
from drososense.utils.env_report import compare_environments
from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir
from drososense.utils.seeding import load_seed_policy

PROTOCOL_VERSION = "1.4.0"

# The metric fields a failed run still carries, so every run has the same
# columns and a failed run cannot be mistaken for a missing one.
NO_METRICS: dict[str, Any] = {
    "macro_f1": None,
    "balanced_accuracy": None,
    "accuracy": None,
    "auroc": None,
    "auroc_n_classes_scored": 0,
    "auroc_defined": False,
    "mae": None,
    "rmse": None,
    "r2": None,
}


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
        selection_grid: When supplied, the runner selects each model's
            hyperparameters on that fold's validation split, drawing only from
            this grid. ``None`` keeps the M1 behaviour of taking the supplied
            parameters; the declared grid is enforced either way.
        train_fraction: E3 low-data subsampling (protocol ``E3_lowdata``).
            When set to a value in ``(0, 1)``, only the nested prefix of the
            full specimen pool (seeded, deterministic) is allowed into each
            fold's TRAIN side; test (and validation) are untouched, so the
            test set is fixed across fractions. ``None`` (the default) keeps
            the full-pool, E1-identical behaviour — with which a 100% run is
            byte-for-byte the same split as any prior E1/E2 batch.
    """

    dataset_id: str
    experiment: str = "smoke"
    models: tuple[str, ...] = MODEL_IDS
    tasks: tuple[str, ...] = ("classification", "regression")
    seeds: tuple[int, ...] = (0,)
    window_lengths: tuple[int, ...] = (16,)
    split_strategy: str = "auto"
    n_splits: int = 5
    stride: int = 1
    label_rule: str = "last"
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_folds: int | None = None
    evidence_class: str = "real"
    enforce_test_touched_once: bool = True
    selection_grid: HyperparameterGrid | None = None
    train_fraction: float | None = None

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
            "selection_grid": None if self.selection_grid is None else self.selection_grid.as_dict(),
            "train_fraction": self.train_fraction,
        }


def dataset_split_strategy(config_path: Path) -> tuple[str, int]:
    """Read a dataset config's declared split strategy.

    The strategy is a property of the dataset — D2 can only do LOSO(5) with its
    five cuts, D3 has 62 fillets and uses GroupKFold, D1 has no specimen id at
    all — so it is declared in the config next to the data it describes rather
    than defaulted at the call site, where it could silently disagree.

    Args:
        config_path: Path to ``configs/datasets/*.yaml``.

    Returns:
        ``(strategy, n_splits)``; ``("group_kfold", 5)`` when undeclared.
    """
    from drososense.utils.config import load_yaml

    raw = load_yaml(config_path)
    split = raw.get("split", {})
    return str(split.get("strategy", "group_kfold")), int(split.get("n_splits", 5))


def _skip_disclosure_entry(
    dataset: str,
    model: str,
    task: str,
    seed: int,
    fold_id: int,
    window_length: int,
    test_fingerprint: str,
    prior_config_hash: str,
    run_config_hash: str,
    reason: str,
    prior_run_id: str,
    prior_status: str,
) -> dict[str, Any]:
    """Build the disclosure entry for one skipped unit.

    A skip is an explicit, auditable event: the reader must be able to see
    exactly which unit was skipped, why (same config re-computation vs a
    different-config prior ok record), which config hash the prior record
    was produced under, and which config hash this batch ran under.

    Args:
        dataset: Dataset identifier.
        model: Model identifier.
        task: Task name.
        seed: Seed.
        fold_id: Fold index.
        window_length: Window length used.
        test_fingerprint: The (partition, window, model, task) fingerprint.
        prior_config_hash: The config_hash of the prior ok record.
        run_config_hash: The config_hash of the current batch.
        reason: One of ``prior_ok_same_config`` or ``prior_ok_different_config``.
        prior_run_id: The run_id of the prior ok record.
        prior_status: Status of the prior record (always ``"ok"``).

    Returns:
        A JSON-serialisable mapping with the full skip disclosure.
    """
    return {
        "dataset": dataset,
        "model": model,
        "task": task,
        "seed": seed,
        "fold_id": fold_id,
        "window_length": window_length,
        "test_fingerprint": test_fingerprint,
        "reason": reason,
        "prior_config_hash": prior_config_hash,
        "run_config_hash": run_config_hash,
        "prior_run_id": prior_run_id,
        "prior_status": prior_status,
        # DATA-51 is an implementation clarification of §17 (a skip is not a
        # re-fit, so §17's intent is preserved). The protocol version label is
        # carried so the disclosure is auditable against the active protocol
        # without a live session.
        "protocol_version": PROTOCOL_VERSION,
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
        ValueError: If no models or tasks are selected, or if the requested
            seeds are not a set the protocol declares.
        OutOfGridError: If a requested window length or model parameter falls
            outside the grid the protocol pre-registered.
    """
    if not config.models:
        raise ValueError("no models selected")
    if not config.tasks:
        raise ValueError("no tasks selected")
    if config.train_fraction is not None and not 0.0 < config.train_fraction <= 1.0:
        raise ValueError(
            f"train_fraction must satisfy 0 < f <= 1, got {config.train_fraction!r}"
        )

    # The declared design, checked before any data is touched. The seed set was
    # previously whatever the caller passed — a run outside 0..9 would have been
    # accepted — and the grid was read by nothing at all, so a model tuned
    # outside it was indistinguishable from one tuned inside it.
    load_seed_policy().validate(config.seeds)
    declared_grid = (
        config.selection_grid
        if config.selection_grid is not None
        else HyperparameterGrid.from_protocol()
    )
    validate_run_hyperparameters(
        window_lengths=config.window_lengths,
        model_params=config.model_params,
        grid=declared_grid,
    )
    selection_spec = SelectionSpec.from_protocol() if config.selection_grid is not None else None

    config_path = dataset_config_path(config.dataset_id)
    dataset = load_dataset(config_path)
    n_classes = dataset.schema.n_classes

    split_strategy, n_splits = config.split_strategy, config.n_splits
    if split_strategy == "auto":
        split_strategy, n_splits_from_config = dataset_split_strategy(config_path)
        if config.n_splits == 5:
            n_splits = n_splits_from_config

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

    # §17's spectral-scaling rule: the value is chosen once per
    # (dataset, seed, fold) and applied identically to every reservoir, or the
    # primary contrast would compare two differently-scaled graphs. The run is
    # single-dataset, so (seed, fold) identifies the group; the first model in
    # the group selects and every later one is handed the same value.
    shared_choices: dict[tuple[int, int], dict[str, Any]] = {}

    # test_touched_once: a (test split, model, task) triple may be evaluated once.
    # Re-scoring saved predictions for another metric is not a new touch; fitting
    # the model again on the same test split under a different configuration is.
    # Protocol v1.4 §17: only a run that actually scored the split (status == ok)
    # occupies the (dataset, model, task, seed, fold) quota; a crashed/errored run
    # never produced test evidence and is not a touch. The filter mirrors
    # `test_touched_once_report` in drososense/evaluation/results.py.
    prior_touches: dict[str, str] = {}
    # The prior-touches registry also carries the run_id and status of the ok
    # record that owns each fingerprint, so a skip can be disclosed with the
    # full audit trail (which run, which status) rather than just a hash.
    prior_touch_meta: dict[str, tuple[str, str]] = {}
    if config.enforce_test_touched_once:
        for prior in load_records(raw_dir):
            if prior.test_fingerprint and prior.status == "ok":
                prior_touches.setdefault(prior.test_fingerprint, prior.config_hash)
                prior_touch_meta.setdefault(
                    prior.test_fingerprint, (prior.run_id, prior.status)
                )

    # Skip disclosure (DATA-51): units whose test_fingerprint is already held by
    # a status == "ok" record are SKIPPED — never re-fit and never re-scored —
    # and the skip is made explicit, not silent. Both directions are disclosed:
    #   - prior ok record has the SAME config_hash  -> re-computation, skip;
    #   - prior ok record has a DIFFERENT config_hash -> the pre-DATA-51 guard
    #     raised RuntimeError and aborted the whole batch here; now the unit is
    #     skipped so the re-shard (e.g. model-subset fan-out or a seed half)
    #     can finish, and the skip carries both config hashes so a reader can
    #     audit exactly which prior run the skip refers to. §17's intent
    #     ("a touched test split must not be re-fit") is preserved: skipping
    #     IS the not-re-fitting — nothing new is fitted on the split.
    skip_disclosures: list[dict[str, Any]] = []

    for window_length in config.window_lengths:
        specimens = usable_specimens(dataset, window_length)
        if len(specimens) < 3:
            raise ValueError(
                f"{config.dataset_id}: only {len(specimens)} specimens are long enough for "
                f"window_length={window_length}; cannot build folds"
            )

        for seed in config.seeds:
            folds = make_folds(specimens, split_strategy, seed=seed, n_splits=n_splits)
            if config.max_folds is not None:
                folds = folds[: config.max_folds]

            # E3 low-data (protocol E3_lowdata: test_set_fixed_across_fractions
            # true, sampling nested). Subsample the TRAIN side only: the test
            # (and validation) specimens never move, so the test set — and the
            # per-fold TEST partition fingerprint the E1/E2 records carry — is
            # identical across 10/25/50/75/100%. The pool is nested, so for
            # a fixed seed 10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%, and the 100% pool is
            # byte-for-byte the full pool (E1-identical train set).
            #
            # The admitted pool filters the TRAIN *window tensors* inside
            # ``build_fold_tensors`` — not the ``Fold`` object itself, which
            # would leak out of the disjoint-cover invariant or raise on an
            # empty train block. A fold whose train side loses every admitted
            # specimen records a run with ``failure_reason`` (one fold's
            # small pool must not abort the whole batch); the test fingerprint
            # is still byte-identical to the 100% fold's.

            for fold in folds:
                artifact_dir = (
                    ensure_dir(RESULTS_RAW_DIR)
                    / "_artifacts"
                    / config.dataset_id
                    / f"w{window_length}_seed{seed:02d}_fold{fold.fold_id:02d}"
                )
                # E3 low-data: pass the admitted specimen pool to the tensor
                # builder; it filters the TRAIN rows, leaving val/test
                # byte-identical to the full-data fold.
                train_pool = (
                    nested_train_fraction(specimens, config.train_fraction, seed)
                    if config.train_fraction is not None and config.train_fraction < 1.0
                    else None
                )
                # E3 low-data: when the admitted pool empties this fold's TRAIN
                # side the tensor builder raises a named, clear error. Record a
                # failed run on the fold's units (not aborted) so a small fold
                # in a 10% batch never blocks the whole run.
                fold_tensors = None
                fold_tensor_failure = ""
                try:
                    fold_tensors = build_fold_tensors(
                        dataset,
                        fold,
                        window_length,
                        artifact_dir,
                        stride=config.stride,
                        label_rule=config.label_rule,
                        train_specimen_pool=train_pool,
                    )
                except ValueError as exc:
                    fold_tensor_failure = str(exc)

                if fold_tensors is None:
                    for model_id in config.models:
                        if not availability.get(model_id, {}).get("available", False):
                            continue
                        for task in config.tasks:
                            test_fp = make_test_fingerprint(
                                fold.fingerprint, window_length, model_id, task
                            )
                            failure_reason = f"tensor build failed: {fold_tensor_failure}"
                            metrics_block = {**NO_METRICS, "failure_reason": failure_reason}
                            failure_record = RunRecord(
                                run_id=make_run_id(config.dataset_id, model_id, task, seed, fold.fold_id),
                                experiment=config.experiment,
                                dataset=config.dataset_id,
                                model=model_id,
                                task=task,
                                seed=seed,
                                fold_id=fold.fold_id,
                                protocol_version=PROTOCOL_VERSION,
                                window_length=window_length,
                                metrics=metrics_block,
                                n_train_windows=0,
                                n_test_windows=0,
                                train_specimens=list(fold.train),
                                test_specimens=list(fold.test),
                                duration_s=0.0,
                                environment=environment,
                                timestamp_utc=utc_now_iso(),
                                evidence_class=evidence_class,
                                protocol_compliant=protocol_compliant,
                                model_description={"model_id": model_id, "task": task},
                                status="failed",
                                failure_reason=failure_reason,
                                fold_fingerprint=fold.fingerprint,
                                class_coverage={},
                                config_hash=run_config_hash,
                                test_fingerprint=test_fp,
                                empty_class_policy=EMPTY_CLASS_POLICY,
                                n_train_sessions=0,
                                n_test_sessions=0,
                                notes=notes,
                                selection={},
                            )
                            records.append(failure_record)
                            write_record(failure_record, raw_dir)
                    continue

                for model_id in config.models:
                    if not availability.get(model_id, {}).get("available", False):
                        reason = availability.get(model_id, {}).get("reason", "unknown")
                        entry = {"model": model_id, "reason": str(reason)}
                        if entry not in skipped:
                            skipped.append(entry)
                        continue

                    for task in config.tasks:
                        test_fingerprint = make_test_fingerprint(
                            fold.fingerprint, window_length, model_id, task
                        )
                        prior = prior_touches.get(test_fingerprint)
                        if prior is not None:
                            # An ok record already occupies this (dataset, model,
                            # task, seed, fold) unit. Per §17 the split must not
                            # be re-fit, and re-computation adds nothing to the
                            # record set — so skip, and disclose the skip rather
                            # than either aborting the batch (different config)
                            # or silently overwriting the record (same config).
                            prior_run_id, prior_status = prior_touch_meta.get(
                                test_fingerprint, ("", "ok")
                            )
                            if prior == run_config_hash:
                                reason = "prior_ok_same_config"
                            else:
                                reason = "prior_ok_different_config"
                            skip_disclosures.append(
                                _skip_disclosure_entry(
                                    dataset=config.dataset_id,
                                    model=model_id,
                                    task=task,
                                    seed=seed,
                                    fold_id=fold.fold_id,
                                    window_length=window_length,
                                    test_fingerprint=test_fingerprint,
                                    prior_config_hash=prior,
                                    run_config_hash=run_config_hash,
                                    reason=reason,
                                    prior_run_id=prior_run_id,
                                    prior_status=prior_status,
                                )
                            )
                            continue

                        started = time.perf_counter()
                        params = dict(config.model_params.get(model_id, {}))
                        selection: dict[str, Any] = {}
                        status = "ok"
                        failure_reason = ""
                        model = None
                        metrics: dict[str, Any] = {}
                        group = (seed, fold.fold_id)
                        try:
                            if selection_spec is not None:
                                chosen = select_hyperparameters(
                                    model_id=model_id,
                                    task=task,
                                    seed=seed,
                                    fold_tensors=fold_tensors,
                                    n_channels=dataset.schema.n_features,
                                    n_classes=n_classes,
                                    grid=declared_grid,
                                    spec=selection_spec,
                                    shared=shared_choices.get(group),
                                )
                                shared_choices.setdefault(group, dict(chosen.shared))
                                params = {**params, **chosen.params}
                                selection = chosen.as_dict()
                            model = build_model(
                                model_id,
                                task,
                                seed,
                                params,
                                n_channels=dataset.schema.n_features,
                            )
                            metrics, _, _ = _evaluate_one(model, fold_tensors, task, n_classes)
                        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                            # Protocol v1.1 §20: a run that cannot be fitted is
                            # RECORDED as failed with its reason and counted, never
                            # silently dropped and never retried with different
                            # parameters. A whole benchmark aborting on one
                            # degenerate fold would be the other failure mode:
                            # it hides how many runs were affected.
                            status = "failed"
                            failure_reason = f"{type(exc).__name__}: {exc}"
                            metrics = {**NO_METRICS, "failure_reason": failure_reason}
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
                                model_description=model.describe() if model is not None else {},
                                status=status,
                                failure_reason=failure_reason,
                                fold_fingerprint=fold.fingerprint,
                                class_coverage=fold_tensors.class_coverage,
                                config_hash=run_config_hash,
                                test_fingerprint=test_fingerprint,
                                empty_class_policy=EMPTY_CLASS_POLICY,
                                n_train_sessions=fold_tensors.summarise()["n_train_sessions"],
                                n_test_sessions=fold_tensors.summarise()["n_test_sessions"],
                                notes=notes,
                                selection=selection,
                            )
                        )
                        # Write incrementally so a long-running parallel sweep is
                        # observable from disk and a worker crash does not lose the
                        # records already produced. Each record's path is unique in
                        # (model, task, seed, fold) so concurrent workers don't race.
                        write_record(records[-1], raw_dir)

    for record in records:
        write_record(record, raw_dir)

    # Skipped units (units already held by a status == "ok" record) are
    # disclosed here and in the summary so the re-shard audit trail is
    # explicit on disk, not just in a chat log. A skip is not a run: no new
    # record is written and the contact log is not touched by skipped units.
    if skip_disclosures:
        disclosure_counts = {
            "prior_ok_same_config": sum(
                1 for d in skip_disclosures if d["reason"] == "prior_ok_same_config"
            ),
            "prior_ok_different_config": sum(
                1 for d in skip_disclosures if d["reason"] == "prior_ok_different_config"
            ),
        }
        disclosure_payload = {
            "n_skipped_units": len(skip_disclosures),
            "n_skipped_same_config": disclosure_counts["prior_ok_same_config"],
            "n_skipped_different_config": disclosure_counts["prior_ok_different_config"],
            "run_config_hash": run_config_hash,
            "protocol_version": PROTOCOL_VERSION,
            "disclosures": skip_disclosures,
        }
        disclosure_path = (
            (Path(tables_dir) if tables_dir is not None else RESULTS_TABLES_DIR)
            / f"{config.experiment}_skip_disclosure.json"
        )
        ensure_dir(disclosure_path.parent)
        with disclosure_path.open("w", encoding="utf-8") as handle:
            json.dump(disclosure_payload, handle, indent=2, sort_keys=True)

    if records or skip_disclosures:
        # Record that the test splits were touched, so the freeze claim is backed
        # by a file the experiment wrote rather than by a note in the protocol.
        record_contact(
            experiment=config.experiment,
            dataset=config.dataset_id,
            split_strategy=split_strategy,
            protocol_compliant=protocol_compliant,
            evidence_class=evidence_class,
            n_models=len(records),
            base_dir=tables_dir,
        )

    summary = aggregate_records(records) if records else pd.DataFrame()
    # The summary frame only covers records NEWLY produced by this invocation
    # (skipped units are not re-scored and so contribute no new rows). The skip
    # disclosure travels alongside it, whether or not any new records exist, so
    # an invocation that did nothing but skip still reports it explicitly.
    summary.attrs["skipped_units"] = skip_disclosures
    summary.attrs["n_skipped_units"] = len(skip_disclosures)
    summary.attrs["n_skipped_same_config"] = sum(
        1 for d in skip_disclosures if d["reason"] == "prior_ok_same_config"
    )
    summary.attrs["n_skipped_different_config"] = sum(
        1 for d in skip_disclosures if d["reason"] == "prior_ok_different_config"
    )
    summary.attrs["skipped_models"] = skipped
    summary.attrs["split_strategy"] = split_strategy
    summary.attrs["environment"] = environment
    summary.attrs["environment_report"] = compare_environments().as_dict()
    if not summary.empty:
        write_summary_csv(summary, config.experiment, tables_dir)
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
    "_skip_disclosure_entry",
    "ModelUnavailableError",
    "RESULTS_RAW_DIR",
    "RESULTS_TABLES_DIR",
]
