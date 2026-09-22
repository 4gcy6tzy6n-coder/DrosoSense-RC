"""M4/E1-E2 bridge: the R0-R6 connectome reservoir family through the official runner.

Why this module exists
----------------------
The M3 deliverable (``scripts/run_reservoir.py``) scores a 75/25 window split
with **no** split protocol and writes only CSV/JSONL audit files.  It
therefore cannot support the frozen v1.3/v1.4 machinery:

* §17 ``test_touched_once`` — the ok-only guard that lives in
  :func:`drososense.evaluation.runner.run_benchmark` and keys every test
  evaluation on ``make_test_fingerprint(fold, window, model, task)``;
* the per-evaluation JSON records under ``results/raw`` that
  :mod:`drososense.evaluation.e2_stats` pairs on
  ``(dataset, task, seed, window_length, fold_id)``;
* the contact log, and the gate evaluator fed by the same records.

This module wires the R0-R6 classes into the runner's two extension points —
model construction and test-fingerprint computation — **without touching the
frozen runner loop**:

* :class:`ReservoirFingerprintRegistry` implements the exact
  :func:`drososense.evaluation.results.make_test_fingerprint` signature but
  keys on ``dataset`` as well, so a benchmark that spans several datasets
  (the M4 E1/E2 matrix) cannot collide two different D2/D3 folds into one
  fingerprint.
* :func:`make_family_constructor` returns a factory that matches
  ``runner.build_model``'s exact call site
  ``(model_id, task, seed, params, n_channels=...)``, building all seven
  families on one shared ``W_in`` / ``b`` and one per-seed node subgraph
  exactly as :func:`drososense.reservoir.connectome_reservoir.build_topology_family`
  does.
* :func:`run_family_benchmark` patches the two runner attributes for one
  benchmark call (restoring them in a ``finally``), and re-uses the runner's
  §17 ``prior_touches`` semantics: only a prior record with ``status == ok``
  occupies the ``(dataset, model, task, seed, fold)`` quota.

The module never writes its own result records; the raw JSON tree under
``results/raw`` remains the single source of truth and feeds
:mod:`drososense.evaluation.e2_stats` unchanged.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from drososense.baselines.base import BaseModel, TaskType
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark
from drososense.evaluation.results import (
    load_records,
    make_test_fingerprint,
    RunRecord,
    write_record,
)
from drososense.reservoir.connectome_reservoir import (
    build_topology_family,
    ReservoirTopology,
    TOPOLOGY_FAMILY_IDS,
)
from drososense.utils.paths import DATA_RAW_DIR, RESULTS_RAW_DIR

#: The record ``model`` ids used by the R0-R6 family, in the protocol's
#: comparison order (R0..R6).  ``R4_er_esn`` is the connectome ESN control
#: on the shared N — a distinct record id from the M1 ``esn`` zoo entry so a
#: pairing can never mix the two sources.
FAMILY_MODEL_IDS: tuple[str, ...] = TOPOLOGY_FAMILY_IDS


def _npz_path(explicit: str | Path | None) -> str:
    """Resolve the olfactory NPZ through the connectome entry point.

    Args:
        explicit: Override path; otherwise the connectome ``adjacency_path()``
            (the documented DATA-3 / DATA-4 entry point).

    Returns:
        A path that exists.

    Raises:
        FileNotFoundError: If no NPZ can be found.
    """
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"npz_path override does not exist: {path}")
    from connectome.paths import adjacency_path

    path = adjacency_path()
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        "olfactory_v1.npz is not provisioned; set --npz-path or DROSOSENSE_DATA"
    )


def _subgraph_indices(
    npz_file: str, reservoir_size: int, seed: int
) -> np.ndarray | None:
    """Draw the shared node subgraph for one seed (None = full graph).

    The subgraph is shared by ALL models of one seed — protocol §17 requires
    the same N for R0..R6 — so it is a pure function of (NPZ, N, seed).
    """
    n_full = int(np.load(npz_file, allow_pickle=True)["node_ids"].shape[0])
    if reservoir_size <= 0 or reservoir_size >= n_full:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_full, size=reservoir_size, replace=False))


def make_family_constructor(
    *,
    dataset_id: str,
    npz_path: str | Path | None = None,
    reservoir_size: int = 250,
    normalization: str = "n0_raw",
) -> "FamilyConstructor":
    """Build the family factory used by :func:`run_family_benchmark`.

    Args:
        dataset_id: The dataset the fold tensors came from (recorded on
            every run for §17 fingerprinting).
        npz_path: Olfactory NPZ location; derived from the connectome entry
            point when omitted.
        reservoir_size: Number of nodes N.
        normalization: R0 normalization (DATA-3 scheme).

    Returns:
        A :class:`FamilyConstructor` whose ``build`` method matches the
        runner's exact ``build_model`` call signature.

    Raises:
        FileNotFoundError: If the NPZ cannot be found.
    """
    npz_file = _npz_path(npz_path)
    return FamilyConstructor(
        dataset_id=dataset_id,
        npz_path=npz_file,
        reservoir_size=int(reservoir_size),
        normalization=normalization,
    )


class FamilyConstructor:
    """Builds one family on demand, keyed by the model id.

    Attributes:
        dataset_id: The dataset the fold tensors came from.
        npz_path: Resolved olfactory NPZ path.
        reservoir_size: Node count N.
        normalization: R0 normalization scheme.
        family_model_ids: The canonical record model ids (R0..R6).
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        npz_path: str,
        reservoir_size: int,
        normalization: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.npz_path = npz_path
        self.reservoir_size = reservoir_size
        self.normalization = normalization
        self.family_model_ids = list(FAMILY_MODEL_IDS)

    @property
    def meta(self) -> dict[str, Any]:
        """JSON-safe provenance record for the run's sidecar."""
        return {
            "dataset_id": self.dataset_id,
            "npz_path": self.npz_path,
            "reservoir_size": self.reservoir_size,
            "normalization": self.normalization,
            "family_model_ids": self.family_model_ids,
            "subgraph_rule": (
                "one shared node subgraph per seed; None when reservoir_size >= "
                "the full node count; np.random.default_rng(seed).choice over "
                "node_ids"
            ),
        }

    def build(
        self,
        model_id: str,
        task: TaskType,
        seed: int,
        params: dict[str, Any] | None = None,
        n_channels: int | None = None,
        *,
        fold_fingerprint: str = "",
    ) -> BaseModel:
        """Construct one member of the R0-R6 family.

        Args:
            model_id: One of :data:`FAMILY_MODEL_IDS` (the canonical record
                id the runner loop passes, e.g. ``R0_real_fly``).
            task: ``classification`` or ``regression``.
            seed: Run seed — shared by every family member, as §17 requires.
            params: Per-model overrides (reservoir_size, leak, gain, ...).
            n_channels: Input channel count, required for the shared W_in.
            fold_fingerprint: Fold identifier (accepted for signature
                compatibility; the family itself is not fold-dependent).

        Returns:
            A freshly built model of the requested family.

        Raises:
            ValueError: If ``model_id`` names an unknown family member.
        """
        merged = dict(params or {})
        node_indices = _subgraph_indices(self.npz_path, self.reservoir_size, seed)
        family = build_topology_family(
            self.npz_path,
            seed=int(seed),
            n_channels=int(n_channels),
            normalization=self.normalization,
            node_indices=node_indices,
            params={**merged, "reservoir_size": self.reservoir_size},
        )
        if model_id not in family:
            raise ValueError(
                f"unknown reservoir model id {model_id!r}; "
                f"available: {sorted(family)}"
            )
        model = family[model_id]
        # The runner record's model column must carry the canonical id, not
        # the class attribute (both happen to agree for the R family, but the
        # runner's registry is the source of truth).
        model.model_id = model_id
        return model


class ReservoirFingerprintRegistry:
    """§17 fingerprint store shared by every model of one dataset run.

    The runner computes each test fingerprint from ``(fold, window, model,
    task)`` of a *single* dataset; a reservoir benchmark spanning several
    datasets must fingerprint ``dataset`` too, or D2 and D3 would collide.
    This class keeps one dict for the whole run so the runner's
    single-fingerprint assumption still holds and §17 still bites.
    """

    def __init__(self) -> None:
        self._fingerprint_of: dict[str, str] = {}

    def record(
        self,
        dataset_id: str,
        fold_fingerprint: str,
        window_length: int,
        model_id: str,
        task: str,
    ) -> str:
        """Register one (dataset, fold, window, model, task) unit and return
        its fingerprint.

        Args:
            dataset_id: The dataset config id.
            fold_fingerprint: The fold's partition fingerprint.
            window_length: The window length.
            model_id: The model being scored (``R0_real_fly`` etc.).
            task: ``classification`` or ``regression``.

        Returns:
            The hex fingerprint of that exact unit.
        """
        payload = f"{dataset_id}|{fold_fingerprint}|w{int(window_length)}|{model_id}|{task}"
        digest = importlib.import_module("hashlib").sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]
        self._fingerprint_of[payload] = digest
        return digest

    def fingerprint(
        self,
        dataset_id: str,
        fold_fingerprint: str,
        window_length: int,
        model_id: str,
        task: str,
    ) -> str:
        """The fingerprint of a unit, computed lazily on first use."""
        return self.record(dataset_id, fold_fingerprint, window_length, model_id, task)

    def clear(self) -> None:
        """Drop all registered fingerprints (used between benchmark runs)."""
        self._fingerprint_of = {}

    def __len__(self) -> int:
        return len(self._fingerprint_of)


def _load_reservoir_records(raw_dir: Path | None = None) -> list[dict[str, Any]]:
    """Parse the reservoir JSON records the runner wrote under ``raw_dir``.

    Args:
        raw_dir: The ``results/raw`` tree (defaults to the repository's).

    Returns:
        Every record with a ``test_fingerprint`` field, as dicts.
    """
    records = load_records(raw_dir if raw_dir is not None else RESULTS_RAW_DIR)
    return [asdict(r) for r in records]


def run_family_benchmark(
    config: BenchmarkConfig,
    family: FamilyConstructor,
    fingerprints: ReservoirFingerprintRegistry | None = None,
    raw_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> Any:
    """Run the reservoir family through the official runner, with §17.

    The two runner extension points are patched for the duration of the call
    and restored afterwards (``try/finally``), even if the run raises:

    * ``runner.build_model`` — replaced by ``family.build`` so the runner's
      model loop constructs one of the seven ``_ReservoirBase`` instances
      instead of a zoo model.
    * ``runner.make_test_fingerprint`` — replaced by the shared
      :class:`ReservoirFingerprintRegistry` so the §17 check spans the
      whole run's records.

    Args:
        config: The benchmark configuration (models = the R0-R6 ids).
        family: The constructor from :func:`make_family_constructor`.
        fingerprints: A fresh registry when omitted.
        raw_dir: Override for ``results/raw``; ``results/raw`` when omitted.
        tables_dir: Override for ``results/tables``.

    Returns:
        The aggregated summary frame (``run_benchmark``'s return value),
        with ``summary.attrs["family_meta"]`` set.

    Raises:
        RuntimeError: On a §17 test-touched-once violation (the runner's
            own error, passed through unchanged).
    """
    import drososense.evaluation.runner as runner_module

    registry = fingerprints if fingerprints is not None else ReservoirFingerprintRegistry()
    resolved_raw = raw_dir if raw_dir is not None else RESULTS_RAW_DIR

    prior_touches: dict[str, str] = {}
    if config.enforce_test_touched_once:
        for record in _load_reservoir_records(resolved_raw):
            if record.get("test_fingerprint") and record.get("status") == "ok":
                prior_touches.setdefault(record["test_fingerprint"], record.get("config_hash"))

    original_builder = runner_module.build_model
    original_fingerprint = runner_module.make_test_fingerprint

    def patched_builder(
        model_id: str,
        task: TaskType,
        seed: int,
        params: dict[str, Any] | None = None,
        n_channels: int | None = None,
    ) -> BaseModel:
        return family.build(
            model_id, task, seed, params, n_channels, fold_fingerprint=""
        )

    def patched_fingerprint(
        fold_fingerprint: str,
        window_length: int,
        model: str,
        task: str,
    ) -> str:
        fp = registry.record(
            config.dataset_id, fold_fingerprint, window_length, model, task
        )
        # §17 ok-only: a prior ok-record under a different config hash is a
        # violation.  Same-hash re-runs are re-scores and allowed.
        if config.enforce_test_touched_once:
            prior = prior_touches.get(fp)
            if prior is not None and prior != _config_hash_of(config):
                raise RuntimeError(
                    f"test_touched_once violated: {config.dataset_id} "
                    f"fold {fold_fingerprint[:12]} w{window_length} "
                    f"{model}/{task} was already evaluated under config {prior}, "
                    f"and is now being re-evaluated under {_config_hash_of(config)}. "
                    f"protocol v1.4 §17 forbids re-fitting on a test split "
                    f"already touched; a changed protocol requires a new "
                    f"version file, not a re-run."
                )
        return fp

    runner_module.build_model = patched_builder
    runner_module.make_test_fingerprint = patched_fingerprint
    try:
        summary = run_benchmark(config, raw_dir=raw_dir, tables_dir=tables_dir)
    finally:
        runner_module.build_model = original_builder
        runner_module.make_test_fingerprint = original_fingerprint
    summary.attrs["family_meta"] = family.meta
    return summary


def _config_hash_of(config: BenchmarkConfig) -> str:
    """The run's own configuration hash, as the runner computes it."""
    from drososense.utils.config import config_hash

    return config_hash(config.as_dict())


__all__ = [
    "FamilyConstructor",
    "FAMILY_MODEL_IDS",
    "make_family_constructor",
    "run_family_benchmark",
    "ReservoirFingerprintRegistry",
]
