"""Configuration loading and fingerprinting.

Every run record stores the hash of the config it was produced from, so a
result can be traced back to the exact protocol and hyperparameters that
generated it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from drososense.utils.paths import PROTOCOL_PATH, PROTOCOL_SHA256_PATH

REQUIRED_PROTOCOL_KEYS: tuple[str, ...] = (
    "protocol_version",
    "frozen",
    "split_unit",
    "seeds",
    "tasks",
    "preprocessing",
    "primary_hypothesis",
    "secondary_hypotheses",
    "statistical_tests",
    "datasets",
    "gates",
    "narrative_adjustment_rules",
)

# Keys protocol v1.1 must carry in addition to the v1 set. They are what the R0
# audit found missing, so a file that claims to be v1.1 without them is rejected
# rather than trusted.
REQUIRED_V1_1_KEYS: tuple[str, ...] = (
    "freeze_evidence",
    "split_protocol",
    "contrasts",
    "pairing",
    "multiplicity",
    "equivalence",
    "hyperparameter_selection",
    "robustness_protocol",
    "size_study",
    "stopping_rules",
    "failure_handling",
    "excluded_specimen_criteria",
    "compute_degradation",
    "co_primary_policy",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed mapping. An empty file yields ``{}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping at the top level of {path}")
    return data


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the active protocol.

    The active protocol is ``configs/protocol_v1.1.yaml``; v1 stays on disk
    unchanged so earlier results remain attributable to the text that produced
    them. A v1.1 document is held to the additional keys the R0 audit required.

    Args:
        path: Override for the protocol location (used by tests).

    Returns:
        The protocol mapping.

    Raises:
        ValueError: If a required top-level key is missing.
    """
    protocol = load_yaml(path or PROTOCOL_PATH)
    missing = [key for key in REQUIRED_PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"protocol is missing required keys: {sorted(missing)}")

    version = str(protocol.get("protocol_version", ""))
    if version.startswith("1.1"):
        missing_v1_1 = [key for key in REQUIRED_V1_1_KEYS if key not in protocol]
        if missing_v1_1:
            raise ValueError(
                f"protocol {version} is missing v1.1 keys the R0 audit required: "
                f"{sorted(missing_v1_1)}"
            )
    return protocol


def protocol_sha256(path: str | Path | None = None) -> str:
    """Compute the SHA-256 of a protocol file's bytes.

    Args:
        path: Protocol file; defaults to the active one.

    Returns:
        Lower-case hex digest.
    """
    target = Path(path) if path is not None else PROTOCOL_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


def recorded_protocol_sha256(sidecar: str | Path | None = None) -> str | None:
    """Read the digest recorded in the protocol's sidecar file.

    Args:
        sidecar: Sidecar path; defaults to the active protocol's.

    Returns:
        The recorded digest, or ``None`` when no sidecar exists.
    """
    target = Path(sidecar) if sidecar is not None else PROTOCOL_SHA256_PATH
    if not target.is_file():
        return None
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.split()[0].lower()
    return None


def verify_protocol_freeze(
    path: str | Path | None = None, sidecar: str | Path | None = None
) -> dict[str, Any]:
    """Check a frozen protocol against its recorded digest.

    This is the enforcement for "freeze by adding a file, never by editing one".
    A post-freeze edit to the protocol changes its digest and makes this fail,
    which is the only way the rule can be more than a promise.

    Args:
        path: Protocol file; defaults to the active one.
        sidecar: Sidecar path; defaults to the active protocol's.

    Returns:
        Mapping with ``path``, ``actual``, ``recorded``, ``matches`` and
        ``ok``.
    """
    target = Path(path) if path is not None else PROTOCOL_PATH
    actual = protocol_sha256(target)
    recorded = recorded_protocol_sha256(sidecar)
    return {
        "path": str(target),
        "actual": actual,
        "recorded": recorded,
        "matches": recorded == actual,
        "ok": recorded is not None and recorded == actual,
    }


def task_metric_direction(task: Mapping[str, Any]) -> str:
    """Return the direction a task's metrics are read in.

    The protocol states a task's direction in up to three places: under
    ``metrics``, at the task level, and as ``primary_metric_direction``. Only the
    first was ever read, and ``tasks.regression`` does not carry it — so MAE and
    RMSE, which are minimised, were reported as metrics to maximise. The gate
    engine's ``favourable`` test and the selector's argmax/argmin both read this,
    so the wrong value is not cosmetic.

    All three are therefore read, and required to agree when more than one is
    present: a task that says two different things has to be resolved by
    amending the protocol, not by preferring whichever key the code happens to
    look at first.

    Args:
        task: A task entry from the protocol's ``tasks`` block.

    Returns:
        ``maximize`` or ``minimize``.

    Raises:
        ValueError: If the task declares no direction, or declares conflicting
            ones.
    """
    metrics = task.get("metrics", {})
    declared = {
        "tasks.*.metrics.direction": metrics.get("direction"),
        "tasks.*.direction": task.get("direction"),
        "tasks.*.primary_metric_direction": task.get("primary_metric_direction"),
    }
    present = {key: str(value) for key, value in declared.items() if value is not None}
    if not present:
        raise ValueError("task declares no metric direction in any of the three declared places")
    distinct = sorted(set(present.values()))
    if len(distinct) > 1:
        raise ValueError(
            f"task declares conflicting metric directions {present}; the protocol must state "
            f"one direction per task"
        )
    return distinct[0]


def protocol_metric_properties(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect each declared metric's direction and equivalence margin.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of metric name to ``{"direction", "margin", "alpha"}``.
    """
    margins = protocol["equivalence"]["margins"]
    alpha = float(protocol["statistical_tests"]["alpha"])
    properties: dict[str, dict[str, Any]] = {}
    for task in protocol["tasks"].values():
        metrics = task.get("metrics", {})
        direction = task_metric_direction(task)
        names = [metrics.get("primary"), *metrics.get("secondary", [])]
        for name in names:
            if not name:
                continue
            properties[str(name)] = {
                "direction": direction,
                "margin": float(margins.get(str(name), 0.0)),
                "alpha": alpha,
            }
    return properties


def protocol_dataset_entries(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return only the true dataset entries of the protocol's ``datasets`` block.

    That block also carries non-dataset policy keys (``acquisition_policy``,
    ``channel_intersection``). They are not datasets and must never enter a
    symbol table or be looked up as a manifest, so membership is decided by the
    entry declaring an ``id``.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of short name to the entry, for entries that declare an ``id``.
    """
    entries: dict[str, Mapping[str, Any]] = {}
    for short_name, entry in protocol.get("datasets", {}).items():
        if isinstance(entry, Mapping) and entry.get("id"):
            entries[str(short_name)] = entry
    return entries


def protocol_dataset_symbols(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Map every dataset the protocol declares to the config id it stands for.

    Gate expressions are written in the protocol's own vocabulary — ``D2``, not
    ``d2_beef_uncontrolled`` — while the contrast table is keyed by the config
    id. That mismatch is the R0.1 review's CRITICAL item C1: the symbol table was
    built from the dataset values that happened to appear in the results, so
    every gate that named a dataset resolved to nothing and the whole rule set
    reported UNEVALUABLE under a reason that read like a missing contrast.

    The fix is to take the names from the protocol's own ``datasets`` block,
    which is where they are declared, and to register each short name as an alias
    of its id. Both spellings then resolve, so an expression may use either.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of bare name to config id, containing each declared short name
        (``D1``) and each declared id (``d1_beef_controlled``, resolving to
        itself).
    """
    symbols: dict[str, str] = {}
    for short_name, entry in protocol_dataset_entries(protocol).items():
        dataset_id = str(entry["id"])
        symbols[short_name] = dataset_id
        symbols[dataset_id] = dataset_id
    return symbols


DEFAULT_CONDITION = "full"


def protocol_family_membership(protocol: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Map each declared ``(contrast, condition)`` pair to the family that owns it.

    ``multiplicity.family_rule`` states the contract this implements: the unit of
    correction is a ``(contrast, condition)`` pair, every declared pair belongs
    to exactly ONE family, and a pair in two families is a protocol error that
    stops the analysis rather than being resolved by a choice.

    v1.2 declared eight families and no production code read them: the analysis
    corrected inside a ``(metric, dataset)`` group instead, which merged the
    primary comparison with the topology and baseline families and merged the
    five robustness families into one. That is review item N1.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of ``(contrast_id, condition)`` to family id.

    Raises:
        ValueError: If a pair is declared in two different families, or if a
            family declares no tests at all.
    """
    membership: dict[tuple[str, str], str] = {}
    for family in protocol["multiplicity"]["families"]:
        family_id = str(family["id"])
        tests = [str(test) for test in family.get("tests", ())]
        if not tests:
            raise ValueError(f"multiplicity family {family_id!r} declares no tests")
        conditions = [str(c) for c in family.get("conditions", ())] or [DEFAULT_CONDITION]
        for contrast_id in tests:
            for condition in conditions:
                key = (contrast_id, condition)
                existing = membership.get(key)
                if existing is not None and existing != family_id:
                    raise ValueError(
                        f"multiplicity family rule violated: ({contrast_id}, {condition}) is "
                        f"declared in both {existing!r} and {family_id!r}. Every declared pair "
                        f"belongs to exactly one family and the analysis must stop rather than "
                        f"choose one."
                    )
                membership[key] = family_id
    return membership


def protocol_model_id_bindings(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Map each registered model id to the id the protocol's gates name it by.

    ``model_zoo.*.protocol_id`` records the binding — ``esn`` is written ``R4``
    in every contrast and gate, ``gru`` is written ``GRU``. A parameter count or
    a result keyed by the registry id has to be reachable under the protocol id
    too, or the frozen expressions resolve to nothing.

    Args:
        protocol: The parsed protocol.

    Returns:
        Mapping of registry id to protocol id, for entries that declare one.
    """
    bindings: dict[str, str] = {}
    for family in protocol.get("model_zoo", {}).values():
        if not isinstance(family, list):
            continue
        for entry in family:
            if isinstance(entry, Mapping) and entry.get("protocol_id"):
                bindings[str(entry["id"])] = str(entry["protocol_id"])
    return bindings


def protocol_model_symbols(protocol: Mapping[str, Any]) -> list[str]:
    """Return every model name a gate expression may name.

    The model zoo carries the registry ids (``esn``, ``gru``) while the contrasts
    and gates carry the protocol's reservoir ids (``R0``, ``R4``). Both are
    declared, so both resolve; the binding between them is recorded in
    ``model_zoo.*.protocol_id`` and used by the M4 runner, which must record the
    protocol id so that the frozen expressions keep resolving.

    Args:
        protocol: The parsed protocol.

    Returns:
        Sorted distinct model names.
    """
    names: set[str] = set()
    for family in protocol.get("model_zoo", {}).values():
        if isinstance(family, list):
            for entry in family:
                if isinstance(entry, Mapping):
                    names.add(str(entry.get("id", "")))
                    protocol_id = entry.get("protocol_id")
                    if protocol_id:
                        names.add(str(protocol_id))
    for contrast in protocol.get("contrasts", {}).get("list", ()):
        names.add(str(contrast["first"]))
        names.add(str(contrast["second"]))
    names.discard("")
    return sorted(names)


def protocol_condition_symbols(protocol: Mapping[str, Any]) -> list[str]:
    """Return every condition a gate expression may name.

    Conditions are declared by the multiplicity families, which enumerate the
    (contrast, condition) pairs the project is allowed to report. Deriving them
    from there means a condition that appears in a gate but in no family is
    caught as an unknown name rather than silently resolving.

    Args:
        protocol: The parsed protocol.

    Returns:
        Sorted distinct condition names, always including ``full``.
    """
    names: set[str] = {"full"}
    for family in protocol.get("multiplicity", {}).get("families", ()):
        names.update(str(condition) for condition in family.get("conditions", ()))
    return sorted(names)


def protocol_effect_size_name(protocol: Mapping[str, Any], task: str) -> str:
    """Return the declared effect size for a task.

    Chosen by TASK, never by metric name: ``r2`` is a regression metric and takes
    the Hodges-Lehmann estimator even though its name is not ``mae`` or ``rmse``,
    which is the R0.1 review's item M3.

    Args:
        protocol: The parsed protocol.
        task: ``classification`` or ``regression``.

    Returns:
        The declared effect-size name.

    Raises:
        KeyError: If the task declares no effect size.
    """
    declared = protocol["statistical_tests"]["effect_size"]
    if task not in declared:
        raise KeyError(
            f"task {task!r} declares no effect size; declared: "
            f"{sorted(k for k in declared if isinstance(declared[k], str))}"
        )
    return str(declared[task])


def protocol_metric_direction(protocol: Mapping[str, Any], metric: str) -> str:
    """Return the declared direction for a metric.

    Args:
        protocol: The parsed protocol.
        metric: Metric name.

    Returns:
        ``maximize`` or ``minimize``.

    Raises:
        KeyError: If the metric is not declared by any task.
    """
    properties = protocol_metric_properties(protocol)
    if metric not in properties:
        raise KeyError(f"metric {metric!r} is not declared by any task in the protocol")
    return str(properties[metric]["direction"])


def protocol_contrast_key(contrast: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the table key a contrast definition resolves to.

    Args:
        contrast: A contrast definition with ``id``, ``metric``, ``dataset`` and
            an optional ``condition``.

    Returns:
        ``(contrast_id, metric, dataset, condition)``.
    """
    return (
        str(contrast["id"]),
        str(contrast["metric"]),
        str(contrast["dataset"]),
        str(contrast.get("condition", "full")),
    )


def protocol_paired_spec(protocol: Mapping[str, Any], metric: str, task: str):
    """Build a :class:`~drososense.evaluation.stats.PairedSpec` from a protocol.

    The point of routing every test through one constructor is that the code
    cannot quietly use different parameters from the ones the protocol froze.

    Args:
        protocol: The parsed protocol.
        metric: Metric the contrast is evaluated on.
        task: ``classification`` or ``regression``; it selects the effect size
            (protocol v1.2 §10 declares one per task, not per metric name).

    Returns:
        The frozen :class:`~drososense.evaluation.stats.PairedSpec`.
    """
    from drososense.evaluation.stats import PairedSpec

    tests = protocol["statistical_tests"]
    bootstrap = tests["bootstrap"]
    margins = protocol["equivalence"]["margins"]
    direction = protocol_metric_direction(protocol, metric)
    primary = tests["primary_test"]
    pair_carried = tests.get("paired_descriptive_test", primary)
    return PairedSpec(
        alpha=float(tests["alpha"]),
        margin=float(margins.get(metric, 0.0)),
        direction=direction,
        alternative=str(pair_carried["alternative"]),
        zero_method=str(pair_carried["zero_method"]),
        correction=bool(pair_carried["correction"]),
        wilcoxon_mode=str(pair_carried["mode"]),
        bootstrap_b=int(bootstrap["n_resamples"]),
        bootstrap_seed=int(bootstrap["seed"]),
        ci_level=float(bootstrap["ci_level"]),
        ci_type=str(bootstrap["ci_type"]),
        resample_unit=str(bootstrap["resample_unit"]),
        stratification=str(bootstrap.get("stratification", "seed")),
        primary_test=str(primary["name"]),
        effect_size_name=protocol_effect_size_name(protocol, task),
    )


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable short fingerprint of a config mapping.

    Key order does not affect the digest, so re-serialising a config does not
    invalidate previously recorded runs.

    Args:
        config: Any JSON-serialisable mapping.

    Returns:
        First 12 hex characters of the SHA-256 of the canonical JSON form.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
