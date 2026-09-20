"""Deterministic seeding.

The protocol fixes seeds 0..9. A seed drives both the specimen permutation used
by the split and every stochastic model component, so a run is reproducible
from ``(dataset, model, seed, fold, window_length)`` alone.
"""

from __future__ import annotations

import os
import random

import numpy as np

# Torch is optional at import time so that the data-layer tests run on a
# minimal install (numpy + pandas + scikit-learn only).
try:  # pragma: no cover - exercised only by presence/absence of torch
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def make_rng(seed: int) -> np.random.Generator:
    """Return a dedicated generator, never the global numpy RNG.

    Using explicit generators keeps two concurrently evaluated models from
    consuming each other's random stream.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    """Seed every RNG a model might reach for.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
