"""Reservoir models.

M1 contains :class:`~drososense.reservoir.esn.EchoStateNetwork` — the standard
random echo-state network that serves as the R4 control in M4. It is NOT a
connectome reservoir and carries no biological claim.

M3 (DATA-4) adds :mod:`drososense.reservoir.connectome_reservoir` with the
frozen connectome reservoir and the seven matched topology controls (R0–R6).
"""

from drososense.reservoir.connectome_reservoir import (
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    PRIMARY_CONTROL,
    ReservoirShared,
    ReservoirTopology,
    R0RealFlyReservoir,
    R1WeightShuffledReservoir,
    R2DegreeRewiredReservoir,
    R3RandomSparseReservoir,
    R4ErEsnReservoir,
    R5SmallWorldReservoir,
    R6DenseRandomReservoir,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    constraint_match,
    load_reservoir_topology_from_npz,
    make_dense_random,
    make_degree_rewired,
    make_er_esn,
    make_random_sparse,
    make_shared,
    make_small_world,
    make_weight_shuffled,
    rescale_to_spectral_radius,
    spectral_radius,
)
from drososense.reservoir.esn import EchoStateNetwork

__all__ = [
    "ALLOWED_NORMALIZATIONS",
    "DEFAULT_RESERVOIR_PARAMS",
    "EchoStateNetwork",
    "PRIMARY_CONTROL",
    "R0RealFlyReservoir",
    "R1WeightShuffledReservoir",
    "R2DegreeRewiredReservoir",
    "R3RandomSparseReservoir",
    "R4ErEsnReservoir",
    "R5SmallWorldReservoir",
    "R6DenseRandomReservoir",
    "ReservoirShared",
    "ReservoirTopology",
    "TOPOLOGY_FAMILY_IDS",
    "build_topology_family",
    "constraint_match",
    "load_reservoir_topology_from_npz",
    "make_dense_random",
    "make_degree_rewired",
    "make_er_esn",
    "make_random_sparse",
    "make_shared",
    "make_small_world",
    "make_weight_shuffled",
    "rescale_to_spectral_radius",
    "spectral_radius",
]