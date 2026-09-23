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

#: The newest frozen amendment whose semantics the runners implement.
#: v1.5 = evidence-unit identity schema 2 (see configs/protocol_v1.5.yaml);
#: the loadable base protocol is still configs/protocol_v1.3.yaml.
PROTOCOL_VERSION = "1.5.0"

__all__ = ["__version__", "PROTOCOL_VERSION"]
