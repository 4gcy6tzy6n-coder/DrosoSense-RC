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
        names = [metrics.get("primary"), *metrics.get("secondary", [])]
        for name in names:
            if not name:
                continue
            properties[str(name)] = {
                "direction": str(metrics.get("direction", "maximize")),
                "margin": float(margins.get(str(name), 0.0)),
                "alpha": alpha,
            }
    return properties


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


def protocol_paired_spec(protocol: Mapping[str, Any], metric: str):
    """Build a :class:`~drososense.evaluation.stats.PairedSpec` from a protocol.

    The point of routing every test through one constructor is that the code
    cannot quietly use different parameters from the ones the protocol froze.

    Args:
        protocol: The parsed protocol.
        metric: Metric the contrast is evaluated on.

    Returns:
        The frozen :class:`~drososense.evaluation.stats.PairedSpec`.
    """
    from drososense.evaluation.stats import PairedSpec

    tests = protocol["statistical_tests"]
    bootstrap = tests["bootstrap"]
    margins = protocol["equivalence"]["margins"]
    direction = protocol_metric_direction(protocol, metric)
    wilcoxon = tests["primary_test"]
    return PairedSpec(
        alpha=float(tests["alpha"]),
        margin=float(margins.get(metric, 0.0)),
        direction=direction,
        alternative=str(wilcoxon["alternative"]),
        zero_method=str(wilcoxon["zero_method"]),
        correction=bool(wilcoxon["correction"]),
        wilcoxon_mode=str(wilcoxon["mode"]),
        bootstrap_b=int(bootstrap["n_resamples"]),
        bootstrap_seed=int(bootstrap["seed"]),
        ci_level=float(bootstrap["ci_level"]),
        ci_type=str(bootstrap["ci_type"]),
        resample_unit=str(bootstrap["resample_unit"]),
        stratification=str(bootstrap.get("stratification", "seed")),
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
