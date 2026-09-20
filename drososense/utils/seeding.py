"""Deterministic seeding, and the declared seed set a run may draw from.

The protocol fixes seeds 0..9. A seed drives both the specimen permutation used
by the split and every stochastic model component, so a run is reproducible
from ``(dataset, model, seed, fold, window_length)`` alone.

The same block also declares how the set may be extended, and nothing read that
declaration: the runner took seeds from its arguments, so a run outside 0..9 was
accepted as readily as one inside it. :class:`SeedPolicy` is the reader. It also
carries the reporting rule that results over 10 seeds and results over 20 are
reported separately and never pooled, which
:func:`drososense.evaluation.results.assert_seed_blocks_are_not_pooled` enforces
at the point where pooling would happen.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PRIMARY_BLOCK = "primary"
EXTENDED_BLOCK = "extended"

# Torch is optional at import time so that the data-layer tests run on a
# minimal install (numpy + pandas + scikit-learn only).
try:  # pragma: no cover - exercised only by presence/absence of torch
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SeedPolicy:
    """The seeds a run may use, and the blocks it may report over.

    Protocol §4 declares ``root_seeds``, ``primary_seed_count`` and
    ``extension_to``. The extension is all-or-nothing — "seeds 10..19 may be
    added only as a whole block" — so a run may use any subset of the root seeds
    and then, if it extends at all, must take the extension whole. Anything else
    is refused rather than accepted and quietly reported as if it were the
    declared design.

    Attributes:
        root_seeds: The primary seed set.
        primary_seed_count: How many root seeds the protocol says there are.
        extension_to: The seed count an extended run reaches, exclusive of the
            seeds above it.
    """

    root_seeds: tuple[int, ...]
    primary_seed_count: int
    extension_to: int

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any] | None = None) -> "SeedPolicy":
        """Build the policy from the active protocol.

        Args:
            protocol: Parsed protocol; the active one when omitted.

        Returns:
            The declared seed policy.

        Raises:
            ValueError: If the block is missing, malformed, or contradicts
                itself.
        """
        if protocol is None:
            from drososense.utils.config import load_protocol

            protocol = load_protocol()

        declared = protocol.get("seeds")
        if not isinstance(declared, Mapping):
            raise ValueError("protocol declares no seeds block")

        root = tuple(int(seed) for seed in declared.get("root_seeds", ()))
        if not root:
            raise ValueError("seeds.root_seeds declares no seed")
        if len(set(root)) != len(root):
            raise ValueError(f"seeds.root_seeds repeats a seed: {list(root)}")

        count = int(declared.get("primary_seed_count", 0))
        extension_to = int(declared.get("extension_to", 0))
        if count != len(root):
            raise ValueError(
                f"seeds.primary_seed_count is {count} but seeds.root_seeds holds "
                f"{len(root)}; a protocol whose two statements of the same thing disagree "
                f"cannot be validated against."
            )
        if extension_to <= count:
            raise ValueError(
                f"seeds.extension_to is {extension_to}, which does not extend "
                f"seeds.primary_seed_count={count}; the extension rule would be vacuous."
            )
        return cls(root_seeds=root, primary_seed_count=count, extension_to=extension_to)

    @property
    def extension_block(self) -> tuple[int, ...]:
        """The seeds the extension adds, as one indivisible block."""
        return tuple(range(self.primary_seed_count, self.extension_to))

    @property
    def extended_seeds(self) -> tuple[int, ...]:
        """The root seeds followed by the extension block."""
        return self.root_seeds + self.extension_block

    def classify(self, seeds: Iterable[int]) -> str:
        """Name the reporting block a seed set belongs to.

        Args:
            seeds: The seeds a run used.

        Returns:
            ``primary`` for a subset of the root seeds, ``extended`` for the
            full extended set.

        Raises:
            ValueError: If the set is not one the protocol declares.
        """
        self.validate(seeds)
        chosen = set(int(seed) for seed in seeds)
        return EXTENDED_BLOCK if chosen == set(self.extended_seeds) else PRIMARY_BLOCK

    def validate(self, seeds: Iterable[int]) -> None:
        """Refuse a seed set the protocol does not declare.

        Args:
            seeds: The seeds a run asked for.

        Raises:
            ValueError: If the set is empty, repeats a seed, names a seed
                outside the declared set, or takes part of the extension block.
        """
        chosen = [int(seed) for seed in seeds]
        if not chosen:
            raise ValueError(
                f"no seeds requested; the protocol declares {list(self.root_seeds)}"
            )
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"the seed set repeats a seed: {sorted(chosen)}")

        allowed = set(self.extended_seeds)
        outside = sorted(set(chosen) - allowed)
        if outside:
            raise ValueError(
                f"seed(s) {outside} are outside the declared seed set {sorted(allowed)}. "
                f"seeds.root_seeds declares {list(self.root_seeds)} and seeds.extension_to "
                f"extends it to {list(self.extended_seeds)}. A run outside that set is not the "
                f"pre-registered design; extend the protocol with a new version file instead."
            )

        block = set(self.extension_block)
        taken = block & set(chosen)
        if not taken:
            return
        if taken != block or set(chosen) != set(self.extended_seeds):
            missing = sorted(set(self.extended_seeds) - set(chosen))
            raise ValueError(
                f"the extension block is all-or-nothing: seeds {sorted(block)} must be added "
                f"whole and only on top of the full primary set, but this run is missing "
                f"{missing}. seeds.extension_rule adds seeds {self.primary_seed_count}.."
                f"{self.extension_to - 1} only as a whole block, so results over "
                f"{len(self.root_seeds)} and over {self.extension_to} seeds stay comparable."
            )


def load_seed_policy(protocol: Mapping[str, Any] | None = None) -> SeedPolicy:
    """Load the declared seed policy.

    Args:
        protocol: Parsed protocol; the active one when omitted.

    Returns:
        The :class:`SeedPolicy`.
    """
    return SeedPolicy.from_protocol(protocol)


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
