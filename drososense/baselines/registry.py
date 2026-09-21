"""Model registry: one lookup that resolves an id to a constructible model.

The registry also answers "is this model actually usable here?". That matters
because a missing optional backend (torch, or xgboost without an OpenMP
runtime) must show up as an explicit, reported gap in the comparison table — not
as a silently shorter table that looks like a complete result.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from drososense.baselines.base import BaseModel, ModelUnavailableError, TaskType


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registered model.

    Attributes:
        model_id: Identifier used in configs, results and tables.
        family: ``classical``, ``sequence`` or ``reservoir``.
        requires: Import name of a third-party backend, if any.
        note: What role this model plays in the study.
    """

    model_id: str
    family: str
    requires: str | None = None
    note: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "svm_rbf": ModelSpec("svm_rbf", "classical", note="SVM with an RBF kernel"),
    "random_forest": ModelSpec("random_forest", "classical", note="Random Forest"),
    "xgboost": ModelSpec("xgboost", "classical", "xgboost", note="Gradient-boosted trees"),
    "pca_svm": ModelSpec("pca_svm", "classical", note="PCA followed by an RBF SVM"),
    "gru": ModelSpec("gru", "sequence", "torch", note="Gated recurrent unit"),
    "lstm": ModelSpec("lstm", "sequence", "torch", note="Long short-term memory"),
    "cnn1d": ModelSpec("cnn1d", "sequence", "torch", note="1-D convolutional network"),
    "tcn": ModelSpec("tcn", "sequence", "torch", note="Dilated causal TCN"),
    "esn": ModelSpec(
        "esn",
        "reservoir",
        note="Standard echo-state network — the R4 control, NOT a connectome",
    ),
}

MODEL_IDS: tuple[str, ...] = tuple(MODEL_SPECS)


def model_spec(model_id: str) -> ModelSpec:
    """Return the spec for a model id.

    Args:
        model_id: Registered identifier.

    Returns:
        The :class:`ModelSpec`.

    Raises:
        KeyError: If the id is not registered.
    """
    if model_id not in MODEL_SPECS:
        raise KeyError(f"unknown model {model_id!r}; registered: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model_id]


def _module_available(name: str) -> bool:
    """Check whether a module can be found without importing it.

    Args:
        name: Import name.

    Returns:
        True if the module spec is findable.
    """
    return importlib.util.find_spec(name) is not None


def model_availability() -> dict[str, dict[str, Any]]:
    """Report which models can actually be built in this environment.

    Returns:
        Mapping of model id to ``{"available": bool, "reason": str|None}``.
    """
    report: dict[str, dict[str, Any]] = {}
    for model_id, spec in MODEL_SPECS.items():
        if spec.requires and not _module_available(spec.requires):
            report[model_id] = {
                "available": False,
                "reason": f"backend {spec.requires!r} is not installed",
            }
            continue
        if model_id == "xgboost":
            # Presence of the package is not enough: its native library can fail
            # to load (missing libomp on macOS). Probe it rather than assume.
            try:
                import xgboost  # noqa: F401
            except Exception as exc:
                report[model_id] = {
                    "available": False,
                    "reason": f"xgboost is installed but its native library failed to load: {exc}",
                }
                continue
        report[model_id] = {"available": True, "reason": None}
    return report


def build_model(
    model_id: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any] | None = None,
    n_channels: int | None = None,
) -> BaseModel:
    """Construct a registered model.

    Args:
        model_id: Registered identifier.
        task: ``classification`` or ``regression``.
        seed: Seed for every stochastic component.
        params: Hyperparameters for this model.
        n_channels: Input channel count, required by GRU/LSTM/CNN/TCN/ESN.

    Returns:
        The constructed model.

    Raises:
        KeyError: If the id is not registered.
        ModelUnavailableError: If its backend cannot be loaded.
    """
    spec = model_spec(model_id)

    if spec.family == "classical":
        from drososense.baselines.classical import CLASSICAL_MODELS

        return CLASSICAL_MODELS[model_id](task, seed, params)
    if spec.family == "sequence":
        from drososense.baselines.deep import DEEP_MODELS

        return DEEP_MODELS[model_id](task, seed, params, n_channels=n_channels)
    if spec.family == "reservoir":
        from drososense.reservoir.esn import EchoStateNetwork

        return EchoStateNetwork(task, seed, params, n_channels=n_channels)

    raise ModelUnavailableError(f"{model_id}: no builder registered for family {spec.family!r}")
