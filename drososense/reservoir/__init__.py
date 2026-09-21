"""Reservoir models.

M1 contains only :class:`~drososense.reservoir.esn.EchoStateNetwork` — the
standard random echo-state network that serves as the R4 control in M4. It is
NOT a connectome reservoir and carries no biological claim.

The biological reservoir (``R0``, real olfactory connectome) is M2/M3 work and
does not exist in this package yet. ``scripts/build_connectome.py`` states the
plan and refuses to run until M2 is scoped.
"""

from drososense.reservoir.esn import EchoStateNetwork

__all__ = ["EchoStateNetwork"]
