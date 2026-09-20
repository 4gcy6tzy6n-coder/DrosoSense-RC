"""DrosoSense-RC: Drosophila Sensory Connectome Reservoir Computing.

A frozen sparse connectome reservoir with a lightweight trained readout for
electronic-nose food-quality monitoring.

M1 scope
--------
This package currently contains the connectome-INDEPENDENT food benchmark:
leakage-safe specimen-level data handling, the classical/sequence/reservoir
baseline zoo, and the evaluation harness. The connectome reservoir itself
(``drososense.reservoir.connectome``) belongs to M2/M3 and is not implemented
here. Nothing in this package supports a claim about connectome advantage.
"""

__version__ = "0.1.0"

PROTOCOL_VERSION = "1.0.0"

__all__ = ["__version__", "PROTOCOL_VERSION"]
