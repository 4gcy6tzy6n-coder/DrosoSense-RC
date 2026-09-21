"""Baseline model zoo.

Every baseline — classical, sequence and reservoir — implements the same small
interface (:class:`~drososense.baselines.base.BaseModel`) and receives the same
window tensors for a given ``(dataset, seed, fold, window_length)``. That is what
makes the comparison in E1 a like-for-like comparison rather than a comparison of
tuning effort.

M1 contains no connectome model. ``R0`` does not exist yet.
"""

from drososense.baselines.base import BaseModel, ModelUnavailableError
from drososense.baselines.registry import (
    MODEL_IDS,
    build_model,
    model_availability,
    model_spec,
)

__all__ = [
    "MODEL_IDS",
    "BaseModel",
    "ModelUnavailableError",
    "build_model",
    "model_availability",
    "model_spec",
]
