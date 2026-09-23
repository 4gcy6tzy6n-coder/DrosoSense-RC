"""Run records and aggregation.

Raw and summary results are kept apart by construction:

* ``results/raw/<experiment>/<dataset>/<model>/seed<seed>_fold<fold>.json`` holds
  one record per (model, seed, fold) with full provenance.
* ``results/tables/<experiment>_summary.csv`` holds the aggregated mean ± SD
  across seeds.

A summary is always regenerable from the raw records; the reverse is not true,
which is why the raw records are the source of truth.

Every record carries an ``evidence_class``. It is ``real`` only for observed
data; the synthetic fixture is ``synthetic_fixture``, and a split that does not
satisfy ``split_unit: specimen`` is flagged ``protocol_compliant: false``. Both
flags travel with the number all the way into the summary table, so a
non-compliant or synthetic number cannot be mistaken for a protocol result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drososense.utils.config import DEFAULT_CONDITION
from drososense.utils.paths import RESULTS_RAW_DIR, RESULTS_TABLES_DIR, ensure_dir
from drososense.utils.seeding import SeedPolicy

# Fields the frozen protocol requires on every record (section 14).
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "run_id",
    "experiment",
    "dataset",
    "model",
    "task",
    "seed",
    "fold_id",
    "protocol_version",
    "window_length",
    "metrics",
    "n_train_windows",
    "n_test_windows",
    "train_specimens",
    "test_specimens",
    "duration_s",
    "environment",
    "timestamp_utc",
)


@dataclass(frozen=True)
class RunRecord:
    """One model's result on one fold of one dataset at one seed.

    Attributes:
        run_id: Deterministic identifier of this exact evaluation.
        experiment: Experiment the run belongs to, e.g. ``smoke``.
        dataset: Dataset identifier.
        model: Model identifier.
        task: ``classification`` or ``regression``.
        seed: Seed for the split and the model.
        fold_id: Fold index within the seed's split.
        protocol_version: Protocol the run was produced under.
        window_length: Window length used.
        metrics: Computed metrics.
        n_train_windows: Training window count.
        n_test_windows: Test window count.
        train_specimens: Specimens used for training.
        test_specimens: Specimens held out for testing.
        duration_s: Wall-clock fitting plus prediction time.
        environment: Interpreter and library versions.
        timestamp_utc: When the run finished.
        evidence_class: ``real`` or ``synthetic_fixture``.
        protocol_compliant: Whether the split satisfied ``split_unit: specimen``.
        model_description: The model's own description of itself.
        fold_fingerprint: Identifier of the exact partition used.
        class_coverage: Classes absent from train or test.
        config_hash: Fingerprint of the run configuration.
        test_fingerprint: Identifier of the (test split, window length, model,
            task) evaluation this run represents. Protocol v1.1 requires each
            such evaluation to happen exactly once; a second record with the
            same fingerprint and a different configuration is a violation.
        empty_class_policy: The declared AUROC empty-class policy in force.
        n_train_sessions: Acquisition sessions in the training split.
        n_test_sessions: Acquisition sessions in the test split.
        status: ``ok``, ``failed``, ``skipped``, ``oom`` or ``timeout``.
        failure_reason: Populated whenever ``status`` is not ``ok``.
        notes: Free-text caveats attached to this run.
        selection: The hyperparameters chosen for this run, the split and metric
            they were chosen on, and the grid point they came from. Empty when
            the run supplied its own parameters instead of selecting.
        cpu_seconds: User+system CPU time for the fitted portion (``resource.
            getrusage`` delta, RUSAGE_SELF). Appended in the DATA-52 runner
            schema, default 0.0, so pre-DATA-52 records without the key still
            load; makes duration_s (wall) and CPU time separately auditable
            instead of back-solving one from the other via a CPU% sample.
        evidence_unit: The evidence-unit identity of this run under protocol
            v1.5 (schema 2): the components that define *which evaluation* was
            scored, and their digest. Optional and defaulted so every record
            written before v1.5 still loads unchanged.
    """

    run_id: str
    experiment: str
    dataset: str
    model: str
    task: str
    seed: int
    fold_id: int
    protocol_version: str
    window_length: int
    metrics: dict[str, Any]
    n_train_windows: int
    n_test_windows: int
    train_specimens: list[str]
    test_specimens: list[str]
    duration_s: float
    environment: dict[str, Any]
    timestamp_utc: str
    evidence_class: str = "real"
    protocol_compliant: bool = True
    model_description: dict[str, Any] = field(default_factory=dict)
    fold_fingerprint: str = ""
    class_coverage: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    test_fingerprint: str = ""
    empty_class_policy: str = ""
    n_train_sessions: int = 0
    n_test_sessions: int = 0
    status: str = "ok"
    failure_reason: str = ""
    notes: str = ""
    selection: dict[str, Any] = field(default_factory=dict)
    cpu_seconds: float = 0.0
    evidence_unit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_class not in ("real", "synthetic_fixture"):
            raise ValueError(
                f"evidence_class must be 'real' or 'synthetic_fixture', got {self.evidence_class!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to plain JSON-compatible types.

        Returns:
            Mapping of every field.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Mapping containing at least the required fields.

        Returns:
            The reconstructed record.

        Raises:
            ValueError: If a required field is missing.
        """
        missing = [name for name in REQUIRED_RECORD_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"run record is missing required fields: {missing}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def capture_environment() -> dict[str, Any]:
    """Capture interpreter and library versions for reproducibility.

    Returns:
        Mapping with the Python version, platform and key package versions.
    """
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost"):
        try:
            module = __import__(module_name)
            environment[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            environment[module_name] = None
    return environment


def record_path(record: RunRecord, base_dir: Path | None = None) -> Path:
    """Return the canonical location of a record's JSON file.

    The task is part of the filename because one model on one fold produces a
    classification record AND a regression record, and they must not collide.

    **The evidence unit is part of the filename too.** Two substrates of one unit
    — a different reservoir size, normalization, input mapping, topology variant,
    condition or rewiring seed — are distinct evidence under protocol v1.5, so
    they must be distinct *files*. Without the suffix the second substrate
    silently overwrote the first, which is why the pre-v1.5 size study had to
    invent one experiment label per size (``e9_size_d2_n250`` …) and why its
    records then collided on a single fingerprint. The suffix is derived from
    ``evidence_unit.id``; a record with no unit keeps the old, unsuffixed name so
    every pre-v1.5 tree still resolves.

    Args:
        record: The record.
        base_dir: Override for ``results/raw``.

    Returns:
        Path to the JSON file.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_RAW_DIR
    unit = (record.evidence_unit or {}).get("id") or ""
    suffix = f"_{unit}" if unit else ""
    return (
        base
        / record.experiment
        / record.dataset
        / record.model
        / f"{record.task}_seed{record.seed:02d}_fold{record.fold_id:02d}{suffix}.json"
    )


def write_record(record: RunRecord, base_dir: Path | None = None) -> Path:
    """Write a record to its canonical path.

    Args:
        record: The record to persist.
        base_dir: Override for ``results/raw``.

    Returns:
        Path written.
    """
    path = record_path(record, base_dir)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True, default=str)
    return path


def load_records(base_dir: Path | None = None) -> list[RunRecord]:
    """Load every run record under ``results/raw``.

    Args:
        base_dir: Override for ``results/raw``.

    Returns:
        Records sorted by experiment, dataset, model, seed and fold.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_RAW_DIR
    records: list[RunRecord] = []
    if not base.is_dir():
        return records
    for path in sorted(base.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(RunRecord.from_dict(json.load(handle)))
    records.sort(key=lambda r: (r.experiment, r.dataset, r.model, r.task, r.seed, r.fold_id))
    return records


def _metric_names(records: list[RunRecord]) -> list[str]:
    """Collect the union of metric names across records.

    Args:
        records: Records to scan.

    Returns:
        Sorted metric names.
    """
    names: set[str] = set()
    for record in records:
        names.update(record.metrics)
    return sorted(names)


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Flatten records into a tidy per-run frame.

    Args:
        records: Records to flatten.

    Returns:
        Frame with one row per run and one column per metric.
    """
    if not records:
        return pd.DataFrame()
    metric_names = _metric_names(records)
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "run_id": record.run_id,
            "experiment": record.experiment,
            "dataset": record.dataset,
            "model": record.model,
            "task": record.task,
            "seed": record.seed,
            "fold_id": record.fold_id,
            "window_length": record.window_length,
            "protocol_version": record.protocol_version,
            "evidence_class": record.evidence_class,
            "protocol_compliant": record.protocol_compliant,
            "status": record.status,
            "n_train_windows": record.n_train_windows,
            "n_test_windows": record.n_test_windows,
            "n_train_specimens": len(record.train_specimens),
            "n_test_specimens": len(record.test_specimens),
            # Carried so the aggregate can report how many DISTINCT specimens a
            # group covers, not just how many evaluations it performed. Summing
            # per-run counts across seeds counts the same specimen ten times.
            "test_specimens_joined": "|".join(sorted(str(s) for s in record.test_specimens)),
            "n_train_sessions": record.n_train_sessions,
            "n_test_sessions": record.n_test_sessions,
            "duration_s": record.duration_s,
            "cpu_seconds": record.cpu_seconds,
        }
        for name in metric_names:
            row[name] = record.metrics.get(name)
        rows.append(row)
    return pd.DataFrame(rows)


def assert_seed_blocks_are_not_pooled(
    records: list[RunRecord],
    policy: SeedPolicy | None = None,
) -> None:
    """Refuse to aggregate the two declared seed blocks into one number.

    ``seeds.extension_rule`` says results computed over 10 and over 20 seeds are
    reported separately and never pooled. Aggregation is where pooling would
    actually happen: the mean across seeds is taken over whatever seeds the
    records carry, so a set of records spanning both blocks would produce one
    number describing neither design. The rule is therefore enforced here, at
    the point of pooling, rather than left as a note in the protocol.

    Args:
        records: Records about to be aggregated.
        policy: The declared seed policy; loaded from the protocol when omitted.

    Raises:
        ValueError: If the records span the primary seeds and the extension
            block.
    """
    if not records:
        return
    resolved = policy if policy is not None else SeedPolicy.from_protocol()

    # One invocation per configuration hash, so the seed set each invocation
    # used is recoverable from the records themselves. A set that is not a
    # declared design is refused by validate(), which also catches a seed set
    # that took the extension block in part.
    seeds_by_run: dict[str, set[int]] = {}
    for record in records:
        seeds_by_run.setdefault(str(record.config_hash), set()).add(int(record.seed))

    blocks: dict[str, list[int]] = {}
    for config_hash, seeds in seeds_by_run.items():
        resolved.validate(sorted(seeds))
        blocks.setdefault(resolved.classify(sorted(seeds)), []).append(len(seeds))

    if len(blocks) > 1:
        described = ", ".join(
            f"{name} ({sizes[0]} seeds)" for name, sizes in sorted(blocks.items())
        )
        raise ValueError(
            f"these records pool {len(blocks)} seed designs: {described}. "
            f"seeds.extension_rule reports results over {resolved.primary_seed_count} and over "
            f"{resolved.extension_to} seeds separately and never pools them; aggregate each block "
            f"on its own and report them side by side."
        )


def aggregate_records(
    records: list[RunRecord],
    group_by: tuple[str, ...] = (
        "experiment",
        "dataset",
        "model",
        "task",
        "window_length",
        "protocol_version",
        "evidence_class",
        "protocol_compliant",
        "status",
    ),
) -> pd.DataFrame:
    """Aggregate per-run metrics into mean and SD across seeds.

    The grouping deliberately includes ``evidence_class``, ``protocol_compliant``
    and ``status``: aggregating a synthetic run together with real ones, a
    non-compliant split together with a compliant one, or a failed run together
    with a successful one would produce a number that describes none of them.

    Args:
        records: Records to aggregate.
        group_by: Columns forming an aggregate group.

    Returns:
        Frame with ``<metric>_mean``, ``<metric>_sd`` and ``n_seeds`` columns.

    Raises:
        ValueError: If the records span two declared seed designs; see
            :func:`assert_seed_blocks_are_not_pooled`.
    """
    assert_seed_blocks_are_not_pooled(records)
    frame = records_to_frame(records)
    if frame.empty:
        return frame

    # Failed runs stay in the frame under their own status group so the summary
    # can state how many runs were lost, rather than averaging over survivors and
    # implying complete coverage.
    metric_columns = [
        c
        for c in frame.columns
        if c
        not in {
            "run_id",
            "seed",
            "fold_id",
            "protocol_version",
            "evidence_class",
            "protocol_compliant",
            "status",
            "n_train_windows",
            "n_test_windows",
            "n_train_specimens",
            "n_test_specimens",
            "test_specimens_joined",
            "n_train_sessions",
            "n_test_sessions",
            "duration_s",
            *group_by,
        }
    ]

    # Only numeric metrics are averaged. A record's `metrics` mapping also
    # carries descriptive entries (the empty-class policy in force, for example);
    # taking a mean of those is meaningless, and silently coercing them would be
    # worse than leaving them out.
    metric_columns = [
        c for c in metric_columns if pd.api.types.is_numeric_dtype(frame[c])
    ]
    metric_columns = [c for c in metric_columns if c != "auroc_defined"]

    grouped = frame.groupby(list(group_by), dropna=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped["seed"].nunique()
    summary["n_runs"] = grouped.size()
    # `n_specimen_evaluations` counts test evaluations and therefore grows with
    # the seed count: D2 reports 50 (5 specimens x 10 seeds) and D3 reports 620.
    # It was previously called `n_specimens_tested`, which asserted a dataset
    # property that is ten times larger than the dataset's own specimen count and
    # was the obvious source of a Table I "Specimens" column (review item M1).
    # The dataset property is now reported separately, as a union over the group.
    summary["n_specimen_evaluations"] = grouped["n_test_specimens"].sum()
    summary["n_distinct_specimens"] = grouped["test_specimens_joined"].apply(
        lambda values: len(
            {specimen for value in values for specimen in str(value).split("|") if specimen}
        )
    )
    summary["mean_duration_s"] = grouped["duration_s"].mean()
    # DATA-52 runner schema: mean CPU time over the group so per-unit-CPU-
    # second throughput and split-parallel speedup are directly checkable.
    summary["mean_cpu_seconds"] = grouped["cpu_seconds"].mean()
    # AUROC is defined only on folds where every class is present, so the number
    # of runs that contributed to `auroc_mean` is reported next to it rather than
    # left for a reader to assume equals n_runs.
    if "auroc_defined" in frame.columns:
        summary["n_auroc_defined"] = grouped["auroc_defined"].sum()
    summary = summary.reset_index()

    # Pin the summary schema explicitly rather than relying on how the installed
    # pandas version happens to handle an all-NaN aggregation. A group with a
    # single run has an undefined SD; if that column were ever dropped, a
    # downstream table would silently imply that uncertainty was never computed
    # rather than showing it as undefined. Re-adding the pairs also guarantees
    # that a summary built from one seed and one built from ten have identical
    # columns, so they can be concatenated.
    expected = [f"{metric}_{stat}" for metric in metric_columns for stat in ("mean", "std")]
    ordered = [
        *group_by,
        *expected,
        "n_seeds",
        "n_runs",
        "n_specimen_evaluations",
        "n_distinct_specimens",
        "n_auroc_defined",
        "mean_duration_s",
        "mean_cpu_seconds",
    ]
    summary = summary.reindex(columns=ordered)
    for column in expected:
        summary[column] = summary[column].astype(float)
    for column in ("mean_duration_s", "mean_cpu_seconds"):
        summary[column] = summary[column].astype(float)
    return summary


def write_summary_csv(summary: pd.DataFrame, experiment: str, base_dir: Path | None = None) -> Path:
    """Write an aggregated summary to ``results/tables``.

    Args:
        summary: Frame produced by :func:`aggregate_records`.
        experiment: Experiment name used in the filename.
        base_dir: Override for ``results/tables``.

    Returns:
        Path written.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_TABLES_DIR
    ensure_dir(base)
    path = base / f"{experiment}_summary.csv"
    summary.to_csv(path, index=False)
    return path


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timestamp, e.g. ``2026-09-20T12:00:00+00:00``.
    """
    return datetime.now(timezone.utc).isoformat()


def make_run_id(dataset: str, model: str, task: str, seed: int, fold_id: int) -> str:
    """Build the deterministic identifier of a run.

    Args:
        dataset: Dataset identifier.
        model: Model identifier.
        task: Task name.
        seed: Seed.
        fold_id: Fold index.

    Returns:
        Identifier string.
    """
    return f"{dataset}|{model}|{task}|seed{seed:02d}|fold{fold_id:02d}"


def make_test_fingerprint(
    fold_fingerprint: str, window_length: int, model: str, task: str
) -> str:
    """Build the identifier of one test-set evaluation under **schema 1**.

    This is the pre-v1.5 definition and it is kept byte-for-byte: records on
    disk carry the value it produced, and re-deriving them under a new
    definition would silently rewrite history. New runs use
    :func:`make_evidence_unit_id`; this function survives as
    :func:`legacy_test_fingerprint`'s public name so that a re-run of an
    already-scored legacy unit is still recognised and still refused.

    Schema 1 identifies ``(partition, window length, model, task)`` and
    therefore does **not** distinguish two different substrates evaluated on the
    same split — which protocol v1.5 corrects.

    Args:
        fold_fingerprint: The fold's partition fingerprint.
        window_length: Window length used.
        model: Model identifier.
        task: Task name.

    Returns:
        Short hex identifier.
    """
    return legacy_test_fingerprint(fold_fingerprint, window_length, model, task)


# ---------------------------------------------------------------------------
# Evidence-unit identity (protocol v1.5, schema 2)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. §17 forbids scoring a test split twice under two
# configurations. Schema 1's fingerprint named only (partition, window length,
# model, task), so two runs that differed in the *substrate* — reservoir size,
# input normalization, the input mapping, the rewiring seed — collapsed onto one
# identity. Measured consequences: an E9 size study at N = 250/500/1000/2000/4000
# registered the same fingerprint under six configuration hashes (70 §17
# violations by the project's own rule, and `load_evidence_bundle` refuses such a
# bundle), while an E3 low-data sweep was skipped in full as
# `prior_ok_cross_experiment_anchor` against an E1 baseline unit it does not
# actually duplicate.
#
# Schema 2 makes the identity a *versioned, named* tuple: the components are
# recorded in the run record alongside the digest, so the identity is
# inspectable rather than opaque, and a future axis (v2's constrained input
# population) only has to be added to EVIDENCE_UNIT_COMPONENTS.

#: Version tag of the current identity definition. Bump only with a new
#: protocol amendment, and never silently: the tag is inside the digest.
EVIDENCE_UNIT_SCHEMA_VERSION = "2"

#: Record key holding the identity components + digest.
EVIDENCE_UNIT_FIELD = "evidence_unit"

#: The components that define one evidence unit under schema 2. Order is
#: documentation only — the digest sorts keys.
EVIDENCE_UNIT_COMPONENTS: tuple[str, ...] = (
    "dataset",
    "task",
    "fold_fingerprint",
    "model",
    "window_length",
    "condition",
    "reservoir_size",
    "normalization",
    "input_mapping",
    "topology_variant",
    "rewire_seed",
)

#: Components schema 1 did not carry — kept for the legacy reconstruction note.
EVIDENCE_UNIT_V1_ONLY_NOTE = (
    "schema 1 carried dataset/task/fold/model/window only; `condition` and every "
    "substrate component are unknown for a legacy record and are never guessed"
)

#: Components schema 1 did not carry. When a legacy record's identity is
#: reconstructed they are unknown by construction, and they must NOT be
#: guessed: see :func:`legacy_test_fingerprint`.
EVIDENCE_UNIT_V2_ONLY_COMPONENTS: tuple[str, ...] = (
    "condition",
    "reservoir_size",
    "normalization",
    "input_mapping",
    "topology_variant",
    "rewire_seed",
)

#: Marker for a component that genuinely does not apply (a baseline has no
#: reservoir) as opposed to one that was not recorded.
COMPONENT_NOT_APPLICABLE = "n/a"


def _component(value: Any) -> Any:
    """Normalise one identity component to a stable JSON scalar."""
    if value is None:
        return COMPONENT_NOT_APPLICABLE
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def evidence_unit_components(
    *,
    dataset: str,
    task: str,
    fold_fingerprint: str,
    model: str,
    window_length: int,
    condition: Any = DEFAULT_CONDITION,
    reservoir_size: Any = None,
    normalization: Any = None,
    input_mapping: Any = None,
    topology_variant: Any = None,
    rewire_seed: Any = None,
) -> dict[str, Any]:
    """Build the named component tuple of one evidence unit (schema 2).

    ``condition`` is the protocol's own experiment-condition axis
    (``full`` / ``train10pct`` / ``dropout_p0.3`` / ``noise_s0.1`` …) — the same
    label the multiplicity families are keyed on. It is part of the identity
    because the E3 low-data experiment declares that the test split stays fixed
    across training fractions: without the axis, its four fractions are either
    skipped as a prior touch or recorded as §17 violations, and neither is what
    the protocol declares.

    Returns:
        Mapping of every name in :data:`EVIDENCE_UNIT_COMPONENTS`.
    """
    return {
        "dataset": _component(dataset),
        "task": _component(task),
        "fold_fingerprint": _component(fold_fingerprint),
        "model": _component(model),
        "window_length": _component(window_length),
        "condition": _component(condition),
        "reservoir_size": _component(reservoir_size),
        "normalization": _component(normalization),
        "input_mapping": _component(input_mapping),
        "topology_variant": _component(topology_variant),
        "rewire_seed": _component(rewire_seed),
    }


def make_evidence_unit_id(**components: Any) -> str:
    """Digest an evidence unit's schema-2 identity.

    Args:
        **components: Keyword arguments accepted by
            :func:`evidence_unit_components`.

    Returns:
        Short hex identifier, distinct from a schema-1 fingerprint even for the
        same partition and model, because the schema tag is inside the digest.
    """
    payload = json.dumps(
        {
            "schema": EVIDENCE_UNIT_SCHEMA_VERSION,
            "components": evidence_unit_components(**components),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def legacy_test_fingerprint(
    fold_fingerprint: str, window_length: int, model: str, task: str
) -> str:
    """The schema-1 identity, frozen.

    Kept because old records' stored ``test_fingerprint`` values were produced
    by exactly this expression; computing it again for a *new* unit is how the
    ledger keeps an already-scored legacy unit protected after the definition
    changed. Without this alias a v1.5 run would happily re-score a split that
    schema 1 had already scored — turning a false collision into a real
    violation.

    Returns:
        Short hex identifier.
    """
    payload = f"{fold_fingerprint}|w{window_length}|{model}|{task}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_evidence_unit(
    *,
    dataset: str,
    task: str,
    fold_fingerprint: str,
    model: str,
    window_length: int,
    condition: Any = DEFAULT_CONDITION,
    reservoir_size: Any = None,
    normalization: Any = None,
    input_mapping: Any = None,
    topology_variant: Any = None,
    rewire_seed: Any = None,
) -> dict[str, Any]:
    """The record payload describing one evidence unit.

    Returns:
        Mapping with the schema tag, every component, the schema-2 ``id``, and
        the schema-1 ``legacy_id`` so a reader can match the unit against
        records written before v1.5 without re-deriving either.
    """
    components = evidence_unit_components(
        dataset=dataset,
        task=task,
        fold_fingerprint=fold_fingerprint,
        model=model,
        window_length=window_length,
        condition=condition,
        reservoir_size=reservoir_size,
        normalization=normalization,
        input_mapping=input_mapping,
        topology_variant=topology_variant,
        rewire_seed=rewire_seed,
    )
    return {
        "schema": EVIDENCE_UNIT_SCHEMA_VERSION,
        "components": components,
        "id": make_evidence_unit_id(**components),
        "legacy_id": legacy_test_fingerprint(
            fold_fingerprint, window_length, model, task
        ),
    }


def evidence_unit_aliases(
    *, fold_fingerprint: str, window_length: int, model: str, task: str, **components: Any
) -> tuple[str, ...]:
    """Every identifier this unit could be filed under.

    The ledger is keyed by identity, and identities changed in v1.5, so a lookup
    has to consider both. Returns the schema-2 id first, then the schema-1
    fingerprint.

    Returns:
        Tuple of identifiers, schema-2 first.
    """
    return (
        make_evidence_unit_id(
            fold_fingerprint=fold_fingerprint,
            window_length=window_length,
            model=model,
            task=task,
            **components,
        ),
        legacy_test_fingerprint(fold_fingerprint, window_length, model, task),
    )


def build_prior_ledgers(
    records: list["RunRecord"],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split prior ok records into the two ledgers §17 v1.5 needs.

    The alias has to be **asymmetric**, and the reason is the whole point of the
    amendment. A schema-1 fingerprint names only
    ``(partition, window, model, task)``, so if a v1.5 run also registered under
    it, two v1.5 substrates of the same unit would collide again — the fix would
    have changed nothing. But a record written *before* v1.5 has no schema-2 id,
    so its schema-1 fingerprint is the only name it has.

    Therefore:

    * **current** — keyed by the schema-2 evidence-unit id, populated only from
      records that carry one (v1.5 records);
    * **legacy** — keyed by the schema-1 fingerprint, populated only from
      records that do NOT carry one (pre-v1.5 records).

    A lookup consults *current* first and *legacy* only as a fallback.

    Args:
        records: Prior run records, any status.

    Returns:
        ``(current_ledger, legacy_ledger)``, each mapping identity -> config_hash.
    """
    current: dict[str, str] = {}
    legacy: dict[str, str] = {}
    for record in records:
        if record.status != "ok":
            continue
        unit = record.evidence_unit or {}
        unit_id = unit.get("id")
        if unit_id:
            current.setdefault(unit_id, record.config_hash)
            if record.test_fingerprint:
                current.setdefault(record.test_fingerprint, record.config_hash)
            continue
        name = record.test_fingerprint
        if not name and record.fold_fingerprint:
            name = legacy_test_fingerprint(
                record.fold_fingerprint, record.window_length, record.model, record.task
            )
        if name:
            legacy.setdefault(name, record.config_hash)
        # A pre-v1.5 record may also have been filed under its recomputed
        # schema-1 fingerprint when its stored value came from elsewhere.
        if record.fold_fingerprint and record.model and record.task:
            legacy.setdefault(
                legacy_test_fingerprint(
                    record.fold_fingerprint, record.window_length, record.model, record.task
                ),
                record.config_hash,
            )
    return current, legacy


def lookup_prior_unit(
    unit: dict[str, Any],
    current: dict[str, str],
    legacy: dict[str, str],
) -> tuple[str, str] | None:
    """Find the prior touch occupying an evidence unit, identity-first.

    The schema-2 id is authoritative; the schema-1 alias is a fallback that
    exists solely so a pre-v1.5 record — whose substrate components are unknown
    and must not be guessed — cannot be silently re-scored. A v1.5 record for a
    *different* substrate does not occupy this unit, so it must not be found
    through the shared alias.

    Returns:
        ``(identity, prior_config_hash)``, or ``None`` when the unit is free.
    """
    unit_id = unit.get("id")
    if unit_id and unit_id in current:
        return unit_id, current[unit_id]
    legacy_id = unit.get("legacy_id")
    if legacy_id and legacy_id in legacy:
        return legacy_id, legacy[legacy_id]
    return None


def record_evidence_unit_aliases(record: "RunRecord") -> tuple[str, ...]:
    """Every identifier a *stored* record could be filed under (diagnostic).

    NOTE: this is the union, for reporting. The §17 lookup must NOT use the
    union — see :func:`build_prior_ledgers` / :func:`lookup_prior_unit` for the
    asymmetric rule that keeps two v1.5 substrates of one unit distinct.

    A record written under v1.5 carries its components, so its schema-2 id is
    recomputable. A record written under schema 1 does not, and its v2-only
    components are unknown — so only its stored fingerprint and its recomputed
    schema-1 fingerprint are returned. Nothing is guessed.

    Returns:
        Tuple of identifiers, newest first, without duplicates.
    """
    out: list[str] = []
    if record.test_fingerprint:
        out.append(record.test_fingerprint)
    unit = record.evidence_unit or {}
    components = unit.get("components")
    if components:
        try:
            out.append(make_evidence_unit_id(**components))
        except TypeError:
            pass
    if record.fold_fingerprint and record.model and record.task:
        out.append(
            legacy_test_fingerprint(
                record.fold_fingerprint, record.window_length, record.model, record.task
            )
        )
    seen: set[str] = set()
    unique: list[str] = []
    for value in out:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)


# The per-run fields that make protocol v1.1 §17's `test_touched_once` rule
# auditable. They are fingerprints and counts — identifiers and integers — and
# carry no observed sensor value, which is why they can be committed while the
# records under results/raw stay gitignored.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "run_id",
    "experiment",
    "dataset",
    "model",
    "task",
    "seed",
    "fold_id",
    "window_length",
    "protocol_version",
    "evidence_class",
    "protocol_compliant",
    "status",
    "failure_reason",
    "timestamp_utc",
    "duration_s",
    "n_train_windows",
    "n_test_windows",
    "n_train_specimens",
    "n_test_specimens",
    "n_train_sessions",
    "n_test_sessions",
    "fold_fingerprint",
    "test_fingerprint",
    "config_hash",
    "empty_class_policy",
)


def fingerprint_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    """Reduce run records to the identifiers and counts that audit the freeze.

    The raw records are large and stay gitignored, so the recipe that produced
    each number cannot be re-derived from the repository. These fields are the
    part of a record that is provenance rather than observation: which partition
    was used, which evaluation it was, which configuration produced it, and how
    much data went in. Committing them costs kilobytes and makes
    ``test_touched_once`` checkable from the PR instead of from one machine's
    working copy (review item H3.3).

    Args:
        records: Run records to reduce.

    Returns:
        One mapping per record, in run-id order.
    """
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r.run_id):
        row: dict[str, Any] = {}
        for name in FINGERPRINT_FIELDS:
            value = getattr(record, name, "")
            if name == "timestamp_utc" and isinstance(value, str):
                # Second precision is enough to order runs and does not narrow
                # the acquisition window of the underlying data.
                value = value[:19] + "Z" if len(value) >= 19 else value
            row[name] = value
        coverage = record.class_coverage or {}
        row["class_coverage"] = json.dumps(coverage, sort_keys=True, default=str)
        # §17's spectral-scaling rule asks for the chosen value to be recorded on
        # every run so the sharing across R0..R6 is checkable. The raw records
        # are git-ignored, so the choice travels in the committed fingerprint
        # table too, for the same reason `params(...)` needed the model
        # parameter table.
        row["selection"] = json.dumps(record.selection or {}, sort_keys=True, default=str)
        rows.append(row)
    return rows


def model_parameter_table(records: list[RunRecord]) -> dict[str, Any]:
    """Reduce run records to each model's trainable-parameter count.

    A parameter count is a property of a model's ARCHITECTURE, not of an
    observation: it does not vary with the data, it is an integer, and it is what
    Gate_A's ``params(R0) < params(GRU)`` term compares. It used to be read from
    ``results/raw/**`` alone, which is git-ignored, so the delivered gate
    artifact could not be reproduced by anyone without the author's working copy
    (review item N2).

    The count is NOT unique per model. Hyperparameters are selected per fold, so
    a tuned model reports a different size on different folds — the GRU in this
    project spans eight values between 1265 and 4452 parameters, and the random
    forest spans 214 values. A single number therefore has to be chosen and
    stated: ``params()`` uses the model's LARGEST selected configuration, so the
    efficiency claim is made against the biggest GRU actually tuned rather than
    against a convenient small one. The spread is recorded beside it so the
    choice is visible and can be argued with.

    Args:
        records: Run records to reduce.

    Returns:
        Mapping with ``selection_rule``, ``source`` and ``models``, each model
        carrying ``n_trainable_parameters`` (the decisive value), ``min``,
        ``median``, ``max``, ``n_distinct`` and ``n_records``.
    """
    observed: dict[str, list[int]] = {}
    for record in records:
        description = record.model_description or {}
        value = description.get("n_trainable_parameters")
        if value is None:
            continue
        observed.setdefault(str(record.model), []).append(int(value))

    models: dict[str, Any] = {}
    for model_id, values in sorted(observed.items()):
        array = np.asarray(values, dtype=np.int64)
        models[model_id] = {
            # The decisive value. `params()` reads this one.
            "n_trainable_parameters": int(array.max()),
            "min": int(array.min()),
            "median": float(np.median(array)),
            "max": int(array.max()),
            "n_distinct": int(np.unique(array).size),
            "n_records": int(array.size),
        }
    return {
        "selection_rule": (
            "params(model) uses the model's largest selected configuration (the maximum "
            "n_trainable_parameters over its runs), so an efficiency claim is made against "
            "the largest control actually tuned. The min/median/max spread is recorded here "
            "because hyperparameters are selected per fold and the count is therefore not "
            "unique."
        ),
        "source": (
            "results/raw/** run records, field model_description.n_trainable_parameters. "
            "Committed because results/raw is git-ignored: without this file "
            "params(...) is unevaluable for anyone without the author's working copy."
        ),
        "models": models,
    }


def test_touched_once_report(records: list[RunRecord]) -> dict[str, Any]:
    """Check that no test evaluation was scored twice under two configurations.

    Protocol §17 defines a test evaluation as a ``test_fingerprint``: the
    (partition, window length, model, task) tuple. Scoring it twice with the same
    configuration is a re-computation; scoring it twice with a DIFFERENT
    configuration hash means the model saw the test set under two settings, which
    is what the rule forbids.

    Args:
        records: Run records to audit.

    Returns:
        Mapping with the number of distinct fingerprints, the number of repeated
        ones, and the violations, each naming both configuration hashes.
    """
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.test_fingerprint or record.status != "ok":
            continue
        entry = by_fingerprint.setdefault(
            record.test_fingerprint, {"runs": 0, "config_hashes": set(), "run_ids": []}
        )
        entry["runs"] += 1
        entry["config_hashes"].add(record.config_hash)
        entry["run_ids"].append(record.run_id)

    violations = [
        {
            "test_fingerprint": fingerprint,
            "config_hashes": sorted(entry["config_hashes"]),
            "run_ids": sorted(entry["run_ids"]),
        }
        for fingerprint, entry in by_fingerprint.items()
        if len(entry["config_hashes"]) > 1
    ]
    repeats = sorted(
        fingerprint for fingerprint, entry in by_fingerprint.items() if entry["runs"] > 1
    )
    # v1.5 diagnostics. The violation rule above is deliberately UNCHANGED: a
    # bundle written under schema 1 must keep reporting the collisions schema 1
    # actually has, and re-labelling them here would destroy the evidence that
    # the identity bug existed. What is added is only the ability to tell, from
    # the report, which identity definition produced each record.
    schema_counts: dict[str, int] = {}
    n_with_unit = 0
    for record in records:
        unit = record.evidence_unit or {}
        if unit.get("id"):
            n_with_unit += 1
            tag = f"schema_{unit.get('schema', '?')}"
        else:
            tag = "schema_1_legacy"
        schema_counts[tag] = schema_counts.get(tag, 0) + 1
    collisions_under_current_identity = 0
    for record in records:
        if not record.evidence_unit or record.status != "ok":
            continue
        unit_id = record.evidence_unit.get("id")
        if not unit_id:
            continue
        peers = {
            other.config_hash
            for other in records
            if other.status == "ok"
            and (other.evidence_unit or {}).get("id") == unit_id
        }
        if len(peers) > 1:
            collisions_under_current_identity += 1
    return {
        "n_records": len(records),
        "n_with_fingerprint": sum(1 for r in records if r.test_fingerprint and r.status == "ok"),
        "n_distinct_test_fingerprints": len(by_fingerprint),
        "n_repeated_test_fingerprints": len(repeats),
        "repeated_test_fingerprints": repeats,
        "n_violations": len(violations),
        "violations": violations,
        "identity_schema_counts": schema_counts,
        "n_records_with_evidence_unit": n_with_unit,
        "n_records_with_colliding_evidence_unit_id": collisions_under_current_identity,
        "rule": (
            "protocol §17: a second record with the same test_fingerprint and a different "
            "config_hash is a protocol violation. A repeat with the SAME config_hash is a "
            "re-computation and is reported but is not a violation. Under protocol v1.5 the "
            "identity is the schema-2 evidence-unit id; records written before v1.5 are "
            "counted under 'schema_1_legacy' and keep the collisions their own definition "
            "produced."
        ),
    }


def metric_float(record: RunRecord, name: str) -> float | None:
    """Read a metric as a float, tolerating nulls and non-finite values.

    Args:
        record: The record to read.
        name: Metric name.

    Returns:
        The value, or ``None`` when absent or not finite.
    """
    value = record.metrics.get(name)
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if np.isfinite(as_float) else None
