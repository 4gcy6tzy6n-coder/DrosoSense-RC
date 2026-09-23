"""Protocol-compliant R0–R6 reservoir runner (E2 topology / E1 reservoir half).

Deliverable of DATA-50. The old ``scripts/run_reservoir.py`` evaluated
synthetic tensors with a row-level 75/25 split; this runner instead:

* reuses the E1 data pipeline — ``load_dataset`` -> ``usable_specimens`` ->
  ``make_folds(specimens, split_strategy, seed, n_splits)`` ->
  ``build_fold_tensors`` (train-only scaler, no window crosses a specimen), so
  every reservoir record sits on the same specimen-level folds and carries the
  same ``fold_fingerprint`` as an E1 record for the same dataset/seed/fold;
* builds R0–R6 from the real olfactory NPZ (``connectome.paths.
  adjacency_path()``) with the DATA-3 frozen deterministic node selection
  (SHA-256 provenance recorded in every record);
* trains ONLY the readout — ``A`` and ``W_in`` are frozen (M3 semantics);
* writes one JSON record per run to
  ``results/raw/<experiment>/<dataset>/<model>/<task>_seed<NN>_fold<NN>.json``
  with the same fields as an E1 record (``status`` / ``failure_reason`` /
  ``protocol_version`` / ``config_hash`` / ``test_fingerprint``);
* implements the §17 semantics the E1 runner lacked: only ``status == "ok"``
  records count as touches, a failed run is recorded and never silently
  dropped, and a unit that is already ``ok`` under the same config is SKIPPED
  (record written as ``status == "skipped"``), not re-computed and not
  refused.

The model id in records is the protocol's reservoir id (``R0`` … ``R6``),
which is what the frozen contrasts and gates name — the registry id
``R0_real_fly`` style is carried in ``model_description``.

``--select-hyperparameters`` turns on the protocol's §17 selector
(``drososense.evaluation.selection.select_hyperparameters``): the knob values
are drawn from the frozen grid, chosen on the fold's validation split, and
the spectral-scaling choice is shared identically across R0–R6 within one
(dataset, seed, fold) group. Without the flag the run takes the pinned knob
values declared in :data:`PINNED_KNOBS`. Neither path reads the test split.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from drososense.connectome_selection import NodeSelection, select_nodes
from drososense.data.loaders import dataset_config_path, load_dataset
from drososense.data.manifest import load_manifest, manifest_path
from drososense.data.pipeline import build_fold_tensors, usable_specimens
from drososense.data.splits import make_folds
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
    build_prior_ledgers,
    lookup_prior_unit,
    make_evidence_unit,
    make_run_id,
    record_path,
    utc_now_iso,
    write_record,
    write_summary_csv,
)
from drososense.evaluation.runner import NO_METRICS  # §17 failed-run metric block
from drososense.evaluation.selection import (
    HyperparameterGrid,
    OutOfGridError,  # noqa: F401 - re-exported for callers catching it
    SelectionSpec,
    select_hyperparameters,
)
from drososense.reservoir.connectome_reservoir import (
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    INPUT_MAPPING_DENSE_RANDOM,
    ReservoirShared,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    make_shared,
)
from drososense.utils.config import DEFAULT_CONDITION, config_hash, load_yaml
from drososense.utils.env_report import compare_environments
from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir
from drososense.utils.seeding import load_seed_policy

#: The protocol runner records under the clarified §17 semantics.
PROTOCOL_VERSION = "1.5.0"

#: Registry id -> protocol id. The frozen contrasts and gates name the
#: reservoirs R0..R6; records carry the protocol id and keep the registry id
#: in ``model_description``.
PROTOCOL_ID_BY_FAMILY: dict[str, str] = {
    family_id: family_id[:2] for family_id in TOPOLOGY_FAMILY_IDS
}

#: The reservoir knobs the runner can pin, in the protocol's own knob names.
RESERVOIR_KNOB_ALIASES: dict[str, str] = {
    "ridge_lambda": "readout_regularisation",
    "leak": "reservoir_leak_alpha",
    "gain": "reservoir_gain_g",
    "input_scale": "input_scale_gamma",
    "spectral_radius": "spectral_scaling",
}

#: The pinned knob values a bare invocation runs with. The protocol's
#: ``hyperparameter_selection.grid`` declares CANDIDATES, not a default, so
#: the values a default run carries are declared HERE, on the runner, and
#: carried in every run record. They are the first (conservative) candidate
#: of each list; all are inside the frozen grid, which the runner checks.
#: ``ridge_lambda`` is not pinned: the R0–R6 contrast is a TOPOLOGY contrast
#: and every family receives the identical readout, so its value does not
#: change which family wins.
PINNED_KNOBS: dict[str, float] = {
    "leak": 0.1,
    "gain": 0.5,
    "input_scale": 0.1,
}


def dataset_split_strategy(config_path: Path) -> tuple[str, int]:
    """Read a dataset config's declared split strategy and n_splits.

    Mirrors ``run_benchmark``'s behaviour: the strategy is a property of the
    dataset, so ``--split-strategy auto`` resolves from the dataset config.

    Args:
        config_path: Path to ``configs/datasets/*.yaml``.

    Returns:
        ``(strategy, n_splits)``; ``("group_kfold", 5)`` when undeclared.
    """
    raw = load_yaml(config_path)
    split = raw.get("split", {})
    return str(split.get("strategy", "group_kfold")), int(split.get("n_splits", 5))


@dataclass(frozen=True)
class ReservoirConfig:
    """Everything that defines one reservoir-runner invocation.

    Attributes:
        dataset_id: Dataset to evaluate.
        experiment: Experiment label used in paths and tables.
        tasks: Tasks to run.
        seeds: Seeds to evaluate.
        window_lengths: Window lengths to evaluate.
        split_strategy: ``group_kfold``, ``loso`` or ``time_block_holdout``.
        n_splits: Number of splits for strategies that use it.
        stride: Window stride.
        label_rule: ``last`` or ``majority``.
        max_folds: Cap on folds per seed, for smoke runs.
        npz_path: The olfactory NPZ; required — the runner has no synthetic
            substitute, by design.
        reservoir_size: Number of nodes N; the DATA-3 selection runs when N
            is below the full graph size.
        normalization: R0 normalization id (DATA-3 scheme n0–n5).
        spectral_radius: Target spectral radius every family rescales to.
        leak, gain, input_scale, ridge_lambda: Pinned knobs; the run refuses
            values outside the declared grid.
        family_ids: Subset of R0–R6 to run; the whole family by default.
        select_hyperparameters: Turn on the §17 per-fold grid selection on
            the validation split, sharing the spectral-scaling choice across
            the family within one (seed, fold).
        enforce_test_touched_once: The §17 guard.
        enforce_seed_policy: Validate the seed set against the protocol.
        enforce_declared_design: Window lengths must come from the declared
            grid; the seed policy and the grid are what make a run
            "protocol-compliant" in the first place.
        base_model_params: Per-model parameter overrides on top of the
            pinned knobs (rarely needed; kept for completeness).
    """

    dataset_id: str
    experiment: str = "e2_topology"
    tasks: tuple[str, ...] = ("classification", "regression")
    seeds: tuple[int, ...] = (0,)
    window_lengths: tuple[int, ...] = (16,)
    split_strategy: str = "auto"
    n_splits: int = 5
    stride: int = 1
    label_rule: str = "last"
    max_folds: int | None = None
    npz_path: Path | str | None = None
    reservoir_size: int | None = None
    normalization: str = "n1_pre_l1"
    spectral_radius: float = 0.9
    leak: float = PINNED_KNOBS["leak"]
    gain: float = PINNED_KNOBS["gain"]
    input_scale: float = PINNED_KNOBS["input_scale"]
    ridge_lambda: float | None = None
    #: The protocol's experiment-condition label (``full`` for E1/E2;
    #: ``train10pct``/``train25pct``/... for E3_lowdata; ``dropout_p0.3`` /
    #: ``noise_s0.1`` for the robustness batches). Part of the evidence-unit
    #: identity AND of ``config_hash``: a condition is a declared, pre-registered
    #: variation, so two conditions are two units, while an *undeclared* change
    #: of anything else still trips §17.
    condition: str = DEFAULT_CONDITION
    family_ids: tuple[str, ...] = TOPOLOGY_FAMILY_IDS
    select_hyperparameters: bool = False
    enforce_test_touched_once: bool = True
    enforce_seed_policy: bool = True
    enforce_declared_design: bool = True
    base_model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.normalization not in ALLOWED_NORMALIZATIONS:
            raise ValueError(
                f"normalization {self.normalization!r} is not one of "
                f"{list(ALLOWED_NORMALIZATIONS)}"
            )
        if not self.family_ids:
            raise ValueError("no reservoir families selected")
        unknown = [f for f in self.family_ids if f not in TOPOLOGY_FAMILY_IDS]
        if unknown:
            raise ValueError(
                f"unknown family ids {unknown}; declared: {list(TOPOLOGY_FAMILY_IDS)}"
            )
        for task in self.tasks:
            if task not in ("classification", "regression"):
                raise ValueError(f"unknown task {task!r}")

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view; its hash is the run's ``config_hash``."""
        return {
            "dataset_id": self.dataset_id,
            "experiment": self.experiment,
            "tasks": list(self.tasks),
            "seeds": list(self.seeds),
            "window_lengths": list(self.window_lengths),
            "split_strategy": self.split_strategy,
            "n_splits": self.n_splits,
            "stride": self.stride,
            "label_rule": self.label_rule,
            "max_folds": self.max_folds,
            "npz_path": str(self.npz_path),
            "reservoir_size": self.reservoir_size,
            "normalization": self.normalization,
            "spectral_radius": self.spectral_radius,
            "leak": self.leak,
            "gain": self.gain,
            "input_scale": self.input_scale,
            "ridge_lambda": self.ridge_lambda,
            "condition": self.condition,
            "family_ids": list(self.family_ids),
            "select_hyperparameters": self.select_hyperparameters,
            "base_model_params": self.base_model_params,
        }


def _resolve_npz(config: ReservoirConfig) -> Path:
    """Resolve the olfactory NPZ path; the runner requires the real graph."""
    if config.npz_path is not None:
        path = Path(config.npz_path)
    else:
        from connectome.paths import adjacency_path

        path = adjacency_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"olfactory NPZ not found at {path}. Provision the data root "
            f"(export DROSOSENSE_DATA=/path/to/data-root, with the NPZ under "
            f"connectome/adjacency/ or connectome/) or pass --npz-path."
        )
    return path


def _npz_fingerprint(npz_path: Path) -> dict[str, Any]:
    """Cheap provenance for the NPZ the run consumed."""
    import hashlib

    try:
        digest = hashlib.sha256()
        digest.update(npz_path.read_bytes())
        return {"npz_path": str(npz_path), "npz_sha256": digest.hexdigest()}
    except OSError:
        return {"npz_path": str(npz_path)}


def _params_for(params_this: dict[str, Any]) -> dict[str, Any]:
    """The knob overrides a record must state it ran with."""
    overrides = dict(params_this)
    overrides.pop("_normalization", None)
    return {k: v for k, v in overrides.items() if v is not None}


def _family_builder(family: dict[str, Any]):
    """A ``builder(model_id, task, seed, params, n_channels)`` for the §17
    selector, so it scores the family's own reservoirs rather than the
    registry's ESN. The protocol's reservoir id (``R0``…``R6``) is what the
    selector names; it maps back onto the registry family id."""
    alias_by_protocol_id = {v: k for k, v in PROTOCOL_ID_BY_FAMILY.items()}

    def _builder(
        model_id: str,
        task: str,
        seed_: int,
        params: dict[str, Any] | None,
        n_channels_: int | None = None,
    ) -> Any:
        fid = alias_by_protocol_id.get(model_id, model_id)
        model = family[fid]
        model.task = task  # type: ignore[attr-defined]
        model.params = {**DEFAULT_RESERVOIR_PARAMS, **model.params, **(params or {})}
        return model

    return _builder


def _record_exists_for(
    config: ReservoirConfig,
    model_id: str,
    task: str,
    seed: int,
    fold_id: int,
    raw_dir: Path | None,
    evidence_unit: dict[str, Any] | None = None,
) -> bool:
    """Whether the canonical record file for this unit already exists.

    A prior ok record shares this exact path (see :func:`record_path`), so
    re-writing a skip on top of it would clobber the ok status. The skip
    marker is written only when the slot is empty.
    """
    probe = RunRecord(
        run_id=make_run_id(config.dataset_id, model_id, task, seed, fold_id),
        experiment=config.experiment,
        dataset=config.dataset_id,
        model=model_id,
        task=task,
        seed=seed,
        fold_id=fold_id,
        protocol_version=PROTOCOL_VERSION,
        window_length=0,
        metrics={},
        n_train_windows=0,
        n_test_windows=0,
        train_specimens=[],
        test_specimens=[],
        duration_s=0.0,
        environment={},
        timestamp_utc="",
        evidence_unit=evidence_unit or {},
    )
    return record_path(probe, raw_dir).is_file()


def _write_skipped_record(
    report: "RunReport",
    config: ReservoirConfig,
    model_id: str,
    fid: str,
    task: str,
    seed: int,
    fold,
    window_length: int,
    fold_tensors: Any,
    family: dict[str, Any],
    npz_fingerprint: dict[str, Any],
    selection: NodeSelection,
    environment: dict[str, Any],
    evidence_class: str,
    protocol_compliant: bool,
    notes: str,
    run_config_hash: str,
    test_fingerprint: str,
    raw_dir: Path | None,
    split_strategy: str,
    evidence_unit: dict[str, Any] | None = None,
) -> None:
    """Persist the skip so the (unit, config) ledger stays complete."""
    model = family[fid]
    record = RunRecord(
        run_id=make_run_id(config.dataset_id, model_id, task, seed, fold.fold_id),
        experiment=config.experiment,
        dataset=config.dataset_id,
        model=model_id,
        task=task,
        seed=seed,
        fold_id=fold.fold_id,
        protocol_version=PROTOCOL_VERSION,
        window_length=window_length,
        metrics={},
        n_train_windows=len(fold_tensors.train),
        n_test_windows=len(fold_tensors.test),
        train_specimens=list(fold.train),
        test_specimens=list(fold.test),
        duration_s=0.0,
        environment=environment,
        timestamp_utc=utc_now_iso(),
        evidence_class=evidence_class,
        protocol_compliant=protocol_compliant,
        model_description={
            "model_id": model.model_id,
            "family_id": fid,
            "task": task,
            "skip_reason": "unit already ok under the same config; §17 skip-existing",
            "connectome": npz_fingerprint,
            "node_selection": selection.describe(),
            "split_strategy": split_strategy,
        },
        status="skipped",
        fold_fingerprint=fold.fingerprint,
        class_coverage=fold_tensors.class_coverage,
        config_hash=run_config_hash,
        test_fingerprint=test_fingerprint,
        empty_class_policy=EMPTY_CLASS_POLICY,
        notes=notes,
        evidence_unit=evidence_unit or {},
    )
    write_record(record, raw_dir)
    report.records.append(record)


@dataclass
class RunReport:
    """What one invocation did, in a form a summary can recompute from."""

    config_hash: str
    records: list[RunRecord] = field(default_factory=list)
    skipped_units: list[str] = field(default_factory=list)
    summary_frame: pd.DataFrame | None = None

    @property
    def ok_count(self) -> int:
        """How many units this invocation scored (excludes skips and fails)."""
        return sum(1 for r in self.records if r.status == "ok")

    @property
    def failed_units(self) -> list[str]:
        """The (unit, reason) pairs of every run that failed, never dropped."""
        return [
            f"{r.dataset}/{r.model}/{r.task}/seed{r.seed:02d}/fold{r.fold_id:02d}: "
            f"{r.failure_reason}"
            for r in self.records
            if r.status in ("failed", "error")
        ]


def run_reservoir_benchmark(
    config: ReservoirConfig,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> RunReport:
    """Run the reservoir family over the dataset's specimen-level folds.

    Args:
        config: The invocation configuration.
        raw_dir: Override for ``results/raw``.
        tables_dir: Override for ``results/tables``.

    Returns:
        The report: every record written (ok, failed and skipped), the list
        of skipped units, and a summary frame recomputable from the records.

    Raises:
        ValueError / OutOfGridError / FileNotFoundError: On an undeclared
            design, an undeclared knob value, or a missing NPZ.
    """
    # -- declared design checks, before any data is touched -----------------
    if config.enforce_seed_policy:
        load_seed_policy().validate(config.seeds)
    grid: HyperparameterGrid | None = None
    if config.enforce_declared_design:
        grid = HyperparameterGrid.from_protocol()
        grid.check_window_lengths(config.window_lengths)
        for alias, knob in RESERVOIR_KNOB_ALIASES.items():
            value = getattr(config, alias, None)
            if value is not None:
                grid.check(knob, value)

    npz_path = _resolve_npz(config)
    npz_fingerprint = _npz_fingerprint(npz_path)
    npz_fingerprint["normalization"] = config.normalization

    # -- data side: the E1 pipeline, byte-for-byte --------------------------
    config_path = dataset_config_path(config.dataset_id)
    dataset = load_dataset(config_path)
    n_classes = dataset.schema.n_classes
    n_channels = dataset.schema.n_features

    split_strategy, n_splits = config.split_strategy, config.n_splits
    if split_strategy == "auto":
        split_strategy, n_splits_from_config = dataset_split_strategy(config_path)
        if config.n_splits == 5:
            n_splits = n_splits_from_config

    evidence_class = "real"
    if manifest_path(config.dataset_id).is_file():
        manifest = load_manifest(manifest_path(config.dataset_id))
        evidence_class = "synthetic_fixture" if manifest.synthetic else "real"

    protocol_compliant = dataset.schema.protocol_compliant
    notes = ""
    if not protocol_compliant:
        notes = (
            "split_unit: specimen is NOT satisfied for this dataset; this "
            "result is pipeline validation only and must not be reported as a "
            "protocol result."
        )

    run_config_hash = config_hash(config.as_dict())

    # §17's ledger: only OK records count as test touches (v1.4 semantics).
    # The identity is now the v1.5 evidence unit — (dataset, task, fold, model,
    # window, reservoir_size, normalization, input_mapping, topology_variant,
    # rewire_seed) — so two substrates evaluated on the same split are two units,
    # not one collision. Every prior record is ALSO registered under its schema-1
    # fingerprint (recomputed from the record's own fields), because a record
    # written before v1.5 carries only that. Without the alias, a v1.5 re-run of
    # an already-scored legacy unit would match nothing and become a REAL §17
    # violation instead of today's false collision.
    prior_ok: dict[str, str] = {}
    prior_legacy_ok: dict[str, str] = {}
    if config.enforce_test_touched_once:
        relevant = [
            prior
            for prior in load_records(raw_dir)
            if prior.dataset == config.dataset_id
            and prior.experiment == config.experiment
            and prior.status == "ok"
        ]
        prior_ok, prior_legacy_ok = build_prior_ledgers(relevant)

    report = RunReport(config_hash=run_config_hash)
    environment = capture_environment()

    # The run params: pinned knobs over the DATA-3/4 defaults. R0's rescale
    # target comes from the config so the run record states the spectral
    # scaling it used.
    base_params = dict(DEFAULT_RESERVOIR_PARAMS)
    base_params["leak"] = config.leak
    base_params["gain"] = config.gain
    base_params["input_scale"] = config.input_scale
    if config.ridge_lambda is not None:
        base_params["ridge_lambda"] = config.ridge_lambda
    base_params["spectral_radius"] = config.spectral_radius

    # Node selection: the frozen DATA-3 deterministic function. Default
    # (reservoir_size None) = the full graph, which is the protocol's M2
    # substrate.
    import numpy as np

    selection_seed = 20260920  # RNG_SEED of the DATA-3 selection
    if config.reservoir_size is not None:
        target_n = int(config.reservoir_size)
    else:
        payload = np.load(str(npz_path), allow_pickle=True)
        target_n = int(payload["node_ids"].shape[0])
    selection = select_nodes(npz_path, target_n=target_n, seed=selection_seed)

    records: list[RunRecord] = []

    for window_length in config.window_lengths:
        specimens = usable_specimens(dataset, window_length)
        if len(specimens) < 3:
            raise ValueError(
                f"{config.dataset_id}: only {len(specimens)} specimens are long "
                f"enough for window_length={window_length}; cannot build folds"
            )

        for seed in config.seeds:
            folds = make_folds(specimens, split_strategy, seed=seed, n_splits=n_splits)
            if config.max_folds is not None:
                folds = folds[: config.max_folds]

            # One shared input map per seed, identical across R0–R6.
            family_shared = make_shared(
                n_nodes=int(selection.node_indices.size),
                n_channels=n_channels,
                seed=seed,
                input_scale=config.input_scale,
            )
            family: dict[str, Any] = {}
            family_params = dict(base_params)
            family_params["_normalization"] = config.normalization

            # §17 spectral-scaling sharing: chosen once per (dataset, seed,
            # fold) and applied identically to R0–R6.
            shared_choices: dict[tuple[int, int], dict[str, Any]] = {}

            for fold in folds:
                artifact_dir = (
                    ensure_dir(RESULTS_RAW_DIR)
                    / "_artifacts"
                    / config.dataset_id
                    / f"reservoir_w{window_length}_seed{seed:02d}_fold{fold.fold_id:02d}"
                )
                fold_tensors = build_fold_tensors(
                    dataset,
                    fold,
                    window_length,
                    artifact_dir,
                    stride=config.stride,
                    label_rule=config.label_rule,
                )

                if not family:
                    normalization = family_params.pop("_normalization")
                    family = build_topology_family(
                        str(npz_path),
                        seed=seed,
                        n_channels=n_channels,
                        normalization=normalization,
                        node_indices=selection.node_indices,
                        params={
                            **family_params,
                            "reservoir_size": int(selection.node_indices.size),
                        },
                    )

                for task in config.tasks:
                    selection_payload: dict[str, Any] = {}
                    params_this = dict(base_params)
                    params_this["_normalization"] = config.normalization

                    if config.select_hyperparameters and grid is not None:
                        chosen = select_hyperparameters(
                            model_id="R0",
                            task=task,
                            seed=seed,
                            fold_tensors=fold_tensors,
                            n_channels=n_channels,
                            n_classes=n_classes,
                            grid=grid,
                            spec=SelectionSpec.from_protocol(),
                            shared=shared_choices.get((seed, fold.fold_id)),
                            builder=_family_builder(family),
                        )
                        # §17 spectral-scaling rule: the chosen value is shared
                        # identically across R0–R6 within this (dataset, seed,
                        # fold) group — recorded on every run it applies to.
                        shared_choices.setdefault((seed, fold.fold_id), dict(chosen.shared))
                        params_this = {**params_this, **chosen.params}
                        selection_payload = chosen.as_dict()

                    for fid in [f for f in TOPOLOGY_FAMILY_IDS if f in config.family_ids]:
                        model_id = PROTOCOL_ID_BY_FAMILY[fid]
                        # Protocol v1.5: the unit of identity is the evidence
                        # unit, which includes the substrate. `n_nodes` is read
                        # from the topology the family actually built, and the
                        # rewiring seed is the run seed — the reservoir runner
                        # pins it there, so the two cannot drift apart silently.
                        evidence_unit = make_evidence_unit(
                            dataset=config.dataset_id,
                            task=task,
                            fold_fingerprint=fold.fingerprint,
                            model=model_id,
                            window_length=window_length,
                            condition=config.condition,
                            reservoir_size=int(family[fid]._topology.n_nodes),
                            normalization=config.normalization,
                            input_mapping=INPUT_MAPPING_DENSE_RANDOM,
                            topology_variant=fid,
                            rewire_seed=seed,
                        )
                        test_fingerprint = evidence_unit["id"]
                        prior_hit = lookup_prior_unit(
                            evidence_unit, prior_ok, prior_legacy_ok
                        )
                        prior_identity = prior_hit[0] if prior_hit else None

                        if prior_hit is not None:
                            if prior_hit[1] == run_config_hash:
                                # §17 skip-existing: this unit already has an ok
                                # record under the same config. The existing
                                # file IS the ledger entry — re-writing a
                                # `skipped` record on top of it would clobber
                                # the ok status (same path, same unit, and a
                                # skip that overwrites an ok is not a skip, it
                                # is a data loss). So: only write the skip
                                # marker when no record exists yet at the
                                # canonical path, and count the skip either
                                # way so the report states what happened.
                                report.skipped_units.append(
                                    f"{model_id}/{task}/seed{seed:02d}/fold{fold.fold_id:02d}"
                                )
                                if not _record_exists_for(
                                    config,
                                    model_id,
                                    task,
                                    seed,
                                    fold.fold_id,
                                    raw_dir,
                                    evidence_unit=evidence_unit,
                                ):
                                    _write_skipped_record(
                                        report,
                                        config,
                                        model_id,
                                        fid,
                                        task,
                                        seed,
                                        fold,
                                        window_length,
                                        fold_tensors,
                                        family,
                                        npz_fingerprint,
                                        selection,
                                        environment,
                                        evidence_class,
                                        protocol_compliant,
                                        notes,
                                        run_config_hash,
                                        test_fingerprint,
                                        raw_dir,
                                        split_strategy,
                                        evidence_unit=evidence_unit,
                                    )
                                continue
                            raise RuntimeError(
                                f"test_touched_once violated: {config.dataset_id} "
                                f"fold {fold.fold_id} seed {seed} was already evaluated "
                                f"for {model_id}/{task} under config "
                                f"{prior_hit[1]}, and is now being "
                                f"re-evaluated under {run_config_hash}. protocol v1.5 "
                                f"§17 forbids re-fitting on a test split already "
                                f"touched; a changed configuration requires a new "
                                f"protocol version file, not a re-run. "
                                f"(identity={prior_identity}, "
                                f"schema={evidence_unit['schema']})"
                            )

                        started = time.perf_counter()
                        cpu_started = resource.getrusage(resource.RUSAGE_SELF)
                        status = "ok"
                        failure_reason = ""
                        metrics: dict[str, Any] = {}
                        model = family[fid]
                        model.task = task  # type: ignore[attr-defined]
                        # Pinned (or selected) knob values, applied identically
                        # to every family so the contrast stays a topology
                        # contrast, never a knob contrast.
                        model.params = {**model.params, **_params_for(params_this)}
                        try:
                            if task == "classification":
                                model.fit(fold_tensors.train.X, fold_tensors.train.y_class)
                                predictions = model.predict(fold_tensors.test.X)
                                scores = model.predict_proba(fold_tensors.test.X)
                                metrics = classification_metrics(
                                    fold_tensors.test.y_class,
                                    predictions,
                                    scores,
                                    n_classes=n_classes,
                                )
                            else:
                                model.fit(fold_tensors.train.X, fold_tensors.train.y_reg)
                                predictions = model.predict(fold_tensors.test.X)
                                metrics = regression_metrics(
                                    fold_tensors.test.y_reg, predictions
                                )
                        except Exception as exc:  # noqa: BLE001 — recorded, not dropped
                            status = "failed"
                            failure_reason = f"{type(exc).__name__}: {exc}"
                            metrics = {**NO_METRICS, "failure_reason": failure_reason}
                        duration = time.perf_counter() - started
                        # CPU time consumed by the fit+predict window (user +
                        # system, RUSAGE_SELF delta). Appended to the record as
                        # cpu_seconds — a new-schema field that makes
                        # per-unit-CPU-second throughput and split-parallel
                        # speedup checkable from the record instead of
                        # back-solving wallclock x CPU%. Old records load
                        # fine because the field defaults to 0.0 and stays
                        # out of REQUIRED_RECORD_FIELDS.
                        cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
                        cpu_seconds = (
                            (cpu_usage.ru_utime - cpu_started.ru_utime)
                            + (cpu_usage.ru_stime - cpu_started.ru_stime)
                        )

                        model_description = {
                            "model_id": model.model_id,
                            "family_id": fid,
                            "task": task,
                            "params": dict(model.params),
                            "n_trainable_parameters": model.n_trainable_parameters(),
                            "frozen_parameters": model.n_frozen_parameters(),
                            "reservoir_sparsity": model.reservoir_sparsity(),
                            "topology": model._topology.describe(),
                            "shared": family_shared.describe(),
                            "connectome": npz_fingerprint,
                            "node_selection": selection.describe(),
                            "split_strategy": split_strategy,
                            "spectral_scaling_rule": (
                                "chosen once per (dataset, seed, fold) and applied "
                                "identically to R0..R6"
                                if config.select_hyperparameters
                                else f"pinned at {config.spectral_radius} for the whole run"
                            ),
                        }

                        record = RunRecord(
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
                            cpu_seconds=cpu_seconds,
                            environment=environment,
                            timestamp_utc=utc_now_iso(),
                            evidence_class=evidence_class,
                            protocol_compliant=protocol_compliant,
                            model_description=model_description,
                            status=status,
                            failure_reason=failure_reason,
                            fold_fingerprint=fold.fingerprint,
                            class_coverage=fold_tensors.class_coverage,
                            config_hash=run_config_hash,
                            test_fingerprint=test_fingerprint,
                            empty_class_policy=EMPTY_CLASS_POLICY,
                            n_train_sessions=int(fold_tensors.summarise()["n_train_sessions"]),
                            n_test_sessions=int(fold_tensors.summarise()["n_test_sessions"]),
                            notes=notes,
                            selection=selection_payload,
                            evidence_unit=evidence_unit,
                        )
                        records.append(record)
                        write_record(record, raw_dir)
                        report.records.append(record)
                        if status == "ok":
                            # Register this unit's touch under BOTH identities:
                            # the schema-2 id and the schema-1 fingerprint. A
                            # later v1.5 invocation looks up both, and so does a
                            # still-running pre-v1.5 process reading this ledger
                            # through `record_evidence_unit_aliases`. Skipped
                            # units keep their existing entry.
                            # A v1.5 record registers ONLY its schema-2 id: the
                            # schema-1 alias is reserved for pre-v1.5 records,
                            # so that two substrates of one unit stay distinct.
                            prior_ok[evidence_unit["id"]] = run_config_hash

    if records:
        record_contact(
            experiment=config.experiment,
            dataset=config.dataset_id,
            split_strategy=split_strategy,
            protocol_compliant=protocol_compliant,
            evidence_class=evidence_class,
            n_models=sum(1 for r in records if r.status == "ok"),
            base_dir=tables_dir,
        )

    # A summary is re-computable from the records — never read it mid-flight.
    # Scanned records (including prior ok records this invocation skipped) are
    # fed to the aggregator so the summary is regenerated from the full ledger
    # rather than only the records this call scored.
    if raw_dir is not None:
        relevant = [
            r
            for r in load_records(raw_dir)
            if r.experiment == config.experiment and r.dataset == config.dataset_id
        ]
    else:
        relevant = report.records
    # Records that are not the aggregator's own inputs: ok/failed records are
    # scored runs (their metrics are what a summary is about). Skipped records
    # carry an empty metrics block — aggregating them would produce a metric
    # column that is all NaN, and pandas 3 refuses that ("No objects to
    # concatenate") — so they are excluded here and stay in the raw ledger
    # where their status is what they document.
    scored = [r for r in relevant if r.status != "skipped"]
    if scored:
        summary = aggregate_records(scored)
        if not summary.empty:
            write_summary_csv(summary, config.experiment, tables_dir)
            summary.attrs["skipped_units"] = report.skipped_units
            summary.attrs["split_strategy"] = split_strategy
            summary.attrs["environment"] = environment
            summary.attrs["environment_report"] = compare_environments().as_dict()
            report.summary_frame = summary
    return report


__all__ = [
    "PINNED_KNOBS",
    "PROTOCOL_ID_BY_FAMILY",
    "PROTOCOL_VERSION",
    "RESERVOIR_KNOB_ALIASES",
    "RunReport",
    "ReservoirConfig",
    "OutOfGridError",
    "dataset_split_strategy",
    "run_reservoir_benchmark",
]
