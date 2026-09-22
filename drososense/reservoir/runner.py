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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from drososense.connectome_selection import NodeSelection, select_nodes
from drososense.data.loaders import dataset_config_path, load_dataset
from drososense.data.manifest import load_manifest, manifest_path
from drososense.data.pipeline import build_fold_tensors, usable_specimens
from drososense.data.splits import fold_train_pool, make_folds
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
    ReservoirShared,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    make_shared,
)
from drososense.utils.config import config_hash, load_yaml
from drososense.utils.env_report import compare_environments
from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir
from drososense.utils.seeding import load_seed_policy

#: The protocol runner records under the clarified §17 semantics.
PROTOCOL_VERSION = "1.4.0"

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
    family_ids: tuple[str, ...] = TOPOLOGY_FAMILY_IDS
    select_hyperparameters: bool = False
    enforce_test_touched_once: bool = True
    enforce_seed_policy: bool = True
    enforce_declared_design: bool = True
    base_model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    train_fraction: float | None = None

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
            "family_ids": list(self.family_ids),
            "select_hyperparameters": self.select_hyperparameters,
            "base_model_params": self.base_model_params,
            # E3 low-data (DATA-60): same fingerprint semantics as the E1
            # runner — None or 1.0 stays out of the config fingerprint so
            # e3_lowdata_d2_f100 records are byte-identical to the full E2
            # batches; a true subsample (0 < f < 1.0) requires a distinct
            # experiment label per fraction, because the record path
            # (results/raw/<experiment>/...) carries no fraction.
            "train_fraction": self.train_fraction if 0.0 < (self.train_fraction or 0.0) < 1.0 else None,
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
    config: ReservoirConfig, model_id: str, task: str, seed: int, fold_id: int, raw_dir: Path | None
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
    selection: NodeSelection | dict[str, Any],
    environment: dict[str, Any],
    evidence_class: str,
    protocol_compliant: bool,
    notes: str,
    run_config_hash: str,
    test_fingerprint: str,
    raw_dir: Path | None,
    split_strategy: str,
    e3_anchor: list[dict[str, Any]] | None = None,
    train_fraction: float | None = None,
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
            "node_selection": (
                selection.describe()
                if isinstance(selection, NodeSelection)
                else dict(selection)
            ),
            "split_strategy": split_strategy,
        },
        status="skipped",
        fold_fingerprint=fold.fingerprint,
        class_coverage=fold_tensors.class_coverage,
        config_hash=run_config_hash,
        test_fingerprint=test_fingerprint,
        empty_class_policy=EMPTY_CLASS_POLICY,
        notes=notes,
        # DATA-61 defect 5: readable off the record — None = full pool
        # (the E1/E2 anchor), (0, 1) = a true low-data subsample; and the
        # resolved full-pool anchor reference when this skip IS the anchor.
        train_fraction=train_fraction if train_fraction is not None else config.train_fraction,
        e3_anchor=e3_anchor if e3_anchor is not None else [],
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
    # The fingerprint identifies the (dataset, model, task, seed, fold,
    # window) unit; the config_hash distinguishes "this run re-scores a
    # unit it scored" from "this run re-scores a unit ANOTHER config
    # scored". Failed and skipped records carry fingerprints too — a
    # failed one must not occupy the quota (§17), a skipped one is the
    # runner's own continuation marker and must not block a later same-
    # config re-touch check.
    prior_ok: dict[str, str] = {}
    if config.enforce_test_touched_once:
        for prior in load_records(raw_dir):
            if (
                prior.dataset == config.dataset_id
                and prior.experiment == config.experiment
                and prior.test_fingerprint
                and prior.status == "ok"
            ):
                prior_ok.setdefault(prior.test_fingerprint, prior.config_hash)

    # E3 full-pool anchor ledger (DATA-61 defect 5): the cross-experiment
    # ok records that own each test fingerprint under ANY other experiment
    # label (the E1/E2 full batch, a sibling fraction). For a FULL-POOL run
    # (train_pool None — the f100 anchor) these are the anchor references
    # that must NOT be re-scored; for a true low-data run (0 < f < 1.0) they
    # are inert — the unit scores fresh under its own fraction label, the
    # way the E1 half does after the per-unit guard fix below.
    #
    # The ledger is scoped by FINGERPRINT, not by config identity: a
    # full-pool E3 run and a full E1/E2 batch may differ on config knobs
    # outside the split (the experiment label, enforcement flags, …), yet
    # the f100 unit is the full-batch record FOR THIS PARTITION by
    # construction — "which prior ok record owns this partition" is the
    # question that defines the reference, not whether the configs hash
    # equal. A true low-data run (train_pool set) is a DISTINCT unit and
    # this ledger is inert for it.
    anchor_prior_ok: dict[str, str] = {}
    if config.enforce_test_touched_once:
        for prior in load_records(raw_dir):
            if (
                prior.dataset == config.dataset_id
                and prior.experiment != config.experiment
                and prior.test_fingerprint
                and prior.status == "ok"
            ):
                anchor_prior_ok.setdefault(prior.test_fingerprint, prior.experiment)

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

            # E3 low-data (protocol E3_lowdata: test_set_fixed_across_fractions
            # true, sampling nested). Subsample the TRAIN side only: the test
            # (and validation) specimens never move, so the per-fold TEST
            # partition stays byte-for-byte identical across 10/25/50/75/100%
            # and the paired E1-vs-E3 fingerprint comparison stays legitimate.
            # The pool is nested, so for a fixed seed 10% ⊂ 25% ⊂ 50% ⊂ 75%
            # ⊂ 100%, and the 100% pool is byte-for-byte the full pool.
            #
            # The admitted pool filters the TRAIN *window tensors* inside
            # ``build_fold_tensors`` — not the ``Fold`` object itself — so
            # the fold fingerprint (test + val partition) is unchanged and
            # the disjoint-cover invariant on the partition itself holds.
            #
            # DATA-61: each fold's pool is ``fold_train_pool(fold,
            # config.train_fraction)`` — this fold's TRAIN side restricted to
            # its nested prefix — never a global-pool prefix. The old global
            # sampling could admit a specimen that is a given fold's test
            # specimen, emptying the fold's train side and recording every
            # unit as failed. Each fold's pool stays a nested prefix of its
            # TRAIN side ordered by the fold's seeded permutation, so pools
            # are nested per fold
            # (``pool(f=0.10) ⊆ pool(f=0.25) ⊆ … ⊆ fold.train``) and
            # reproducible across batches.

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
                # E3 low-data: pass the admitted specimen pool to the tensor
                # builder; it filters the TRAIN rows, leaving val/test
                # byte-identical to the full-data fold. DATA-61: the pool is
                # the nested prefix of this fold's TRAIN side, so
                # ``pool ⊆ fold.train`` always holds (no empty-train
                # failures); the tensor builder's empty-train guard remains
                # as a last-resort defence.
                train_pool = (
                    fold_train_pool(fold, config.train_fraction)
                    if config.train_fraction is not None and config.train_fraction < 1.0
                    else None
                )
                # E3 record fidelity (DATA-61, gate item d): the record must
                # state which specimens actually trained the model. The full
                # fold.train list stays in ``train_specimens`` (the partition
                # of record, byte-identical across fractions); the admitted
                # E3 pool is carried explicitly in model_description.
                e3_pool_payload = (
                    {
                        "train_fraction": float(config.train_fraction),
                        "e3_admitted_train_specimens": list(train_pool),
                    }
                    if train_pool is not None
                    else {}
                )
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
                    for task in config.tasks:
                        for fid in [f for f in TOPOLOGY_FAMILY_IDS if f in config.family_ids]:
                            model_id = PROTOCOL_ID_BY_FAMILY[fid]
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
                                model_description={
                                    "model_id": model_id,
                                    "family_id": fid,
                                    "task": task,
                                    "split_strategy": split_strategy,
                                },
                                status="failed",
                                failure_reason=failure_reason,
                                # DATA-61 defect 5: readable off the record
                                # — None = full pool, (0, 1) = a true
                                # low-data subsample.
                                train_fraction=config.train_fraction,
                                fold_fingerprint=fold.fingerprint,
                                class_coverage={},
                                config_hash=run_config_hash,
                                test_fingerprint=make_test_fingerprint(
                                    fold.fingerprint, window_length, model_id, task
                                ),
                                empty_class_policy=EMPTY_CLASS_POLICY,
                                n_train_sessions=0,
                                n_test_sessions=0,
                                notes=notes,
                                selection={},
                            )
                            records.append(failure_record)
                            report.records.append(failure_record)
                            write_record(failure_record, raw_dir)
                    continue

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
                        test_fingerprint = make_test_fingerprint(
                            fold.fingerprint, window_length, model_id, task
                        )

                        # DATA-61 defect 5: E3 FULL-POOL anchor. A full-pool
                        # reservoir run (train_pool None — the f100 anchor) is
                        # the full E2 batch for this partition by construction,
                        # so a prior ok record under a DIFFERENT experiment
                        # label that carries THIS run's exact config hash is
                        # the anchor itself: reference it at analysis time,
                        # do not re-score. A true low-data run (train_pool set,
                        # 0 < f < 1.0) is a DISTINCT unit — its config hash
                        # differs from every full-batch record, so this
                        # cross-experiment guard never swallows a f10/f25/
                        # f50/f75 reservoir unit (the same zero-out the E1
                        # per-unit guard previously caused).
                        cross_anchor = (
                            train_pool is None
                            and test_fingerprint in anchor_prior_ok
                        )
                        if cross_anchor:
                            prior_experiment = anchor_prior_ok[test_fingerprint]
                            # The anchor is recorded, not re-scored: a
                            # `skipped` record (written only when the unit has
                            # no record yet at its canonical path) carries
                            # the resolved reference so an audit join reads
                            # it off the record set.
                            report.skipped_units.append(
                                f"{model_id}/{task}/seed{seed:02d}/fold{fold.fold_id:02d} "
                                f"[anchor {prior_experiment}]"
                            )
                            if not _record_exists_for(
                                config, model_id, task, seed, fold.fold_id, raw_dir
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
                                    e3_anchor=[
                                        {
                                            "fingerprint": test_fingerprint,
                                            "experiment": prior_experiment,
                                            "config_hash": run_config_hash,
                                            "run_id": "",
                                            "status": "ok",
                                            "source": (
                                                "results/raw prior ok record "
                                                "(analysis-time join, full-pool anchor)"
                                            ),
                                        }
                                    ],
                                    train_fraction=config.train_fraction,
                                )
                            continue

                        if test_fingerprint in prior_ok:
                            if prior_ok[test_fingerprint] == run_config_hash:
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
                                    config, model_id, task, seed, fold.fold_id, raw_dir
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
                                    )
                                continue
                            elif config.train_fraction is not None and config.train_fraction < 1.0:
                                # E3 (DATA-60): cross-experiment anchor skip.
                                # The prior ok record was written under a
                                # DIFFERENT experiment label (e1_main_d2 /
                                # e2_main_d2 or a sibling fraction). The test
                                # partition is UNCHANGED across fractions
                                # (test_set_fixed_across_fractions true), so
                                # re-fitting on it would be a §17 violation
                                # for the SAME design — but for a true-subsample
                                # fraction label the disclosure is the correct
                                # outcome: the unit is NOT re-fit, the skip is
                                # disclosed, and the §17 RuntimeError is not
                                # raised because the prior record belongs to a
                                # DIFFERENT design (a different train
                                # fraction), not the same design re-fit.
                                report.skipped_units.append(
                                    f"{model_id}/{task}/seed{seed:02d}/fold{fold.fold_id:02d}"
                                )
                                continue
                            else:
                                raise RuntimeError(
                                    f"test_touched_once violated: {config.dataset_id} "
                                    f"fold {fold.fold_id} seed {seed} was already evaluated "
                                    f"for {model_id}/{task} under config "
                                    f"{prior_ok[test_fingerprint]}, and is now being "
                                    f"re-evaluated under {run_config_hash}. protocol v1.4 "
                                    f"§17 forbids re-fitting on a test split already "
                                    f"touched; a changed configuration requires a new "
                                    f"protocol version file, not a re-run."
                                )

                        started = time.perf_counter()
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
                            "node_selection": (
                selection.describe()
                if isinstance(selection, NodeSelection)
                else dict(selection)
            ),
                            "split_strategy": split_strategy,
                            "spectral_scaling_rule": (
                                "chosen once per (dataset, seed, fold) and applied "
                                "identically to R0..R6"
                                if config.select_hyperparameters
                                else f"pinned at {config.spectral_radius} for the whole run"
                            ),
                            **e3_pool_payload,
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
                            # DATA-61 defect 5: readable off the record —
                            # None = full pool, (0, 1) = a true low-data
                            # subsample; and a full-pool E3 unit (train_pool
                            # None ⇒ f100 anchor) references the prior ok
                            # record of this partition instead of re-scoring
                            # it; a true low-data record carries [].
                            e3_anchor=(
                                [
                                    {
                                        "fingerprint": test_fingerprint,
                                        "experiment": anchor_prior_ok.get(test_fingerprint, ""),
                                        "config_hash": run_config_hash,
                                        "run_id": "",
                                        "status": "ok",
                                        "source": (
                                            "results/raw prior ok record "
                                            "(analysis-time join, full-pool anchor)"
                                        ),
                                    }
                                ]
                                if status == "ok"
                                and train_pool is None
                                and test_fingerprint in anchor_prior_ok
                                else []
                            ),
                            train_fraction=config.train_fraction,
                        )
                        records.append(record)
                        write_record(record, raw_dir)
                        report.records.append(record)
                        if status == "ok":
                            # Register this unit's touch in the ledger a later
                            # invocation (and the §17 re-touch check below)
                            # reads. Skipped units keep their existing entry.
                            prior_ok[test_fingerprint] = run_config_hash

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
