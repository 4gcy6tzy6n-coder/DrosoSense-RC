#!/usr/bin/env python
"""M2 — connectome extraction. NOT IMPLEMENTED IN M1.

This script exists so the M2 entry point is a known location rather than a
surprise, and so that the reason it does nothing is recorded next to the plan
instead of in a commit message.

Running it exits non-zero and prints the plan. It does not produce a graph, and
it does not write anything into ``connectome/``.
"""

from __future__ import annotations

import argparse
import sys

PLAN = """
M2 — Connectome engine (NOT STARTED)

Scope
  Build a biologically defined olfactory subcircuit from FlyWire/Codex rather
  than loading all ~139k neurons. Target N ~ 500-5000, spanning:
      olfactory sensory neurons -> projection neurons ->
      mushroom body / Kenyon cells -> MB output neurons.

Planned steps
  1. Acquire the FlyWire/Codex annotation and connectivity tables
     (https://flywire.ai/apps). Record the dataset version and licence in a
     manifest under connectome/metadata/, exactly as data/manifests/ does for
     the food datasets.
  2. Restrict to the olfactory subcircuit by cell type, and record the
     inclusion rule so the subcircuit is reproducible.
  3. Define S_ij = synapse count from i to j. Store as connectome/adjacency/
     olfactory_v1.npz together with a network report (N, edge count, density,
     degree distribution, spectral radius).
  4. Normalise with D^-1/2 A D^-1/2 (or another explicitly stated rule), and
     apply the SAME normalisation to every topology control, so that a
     difference in M4 cannot come from a difference in preprocessing.

Wording constraint carried from the parent issue
  Synapse counts are not synaptic strengths. Any paper text must call the
  result a "synapse-count-informed structural weight" and must not describe it
  as true biological synaptic strength.

Why this is gated
  M4's core comparison is R0 (real) vs R2 (degree-preserving rewired). Building
  the reservoir before the food benchmark would let the connectome work be tuned
  against results that do not yet exist. M1 is deliberately connectome-free.
"""


def main(argv: list[str] | None = None) -> int:
    """Report the M2 plan and refuse to run.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code, always non-zero.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="print the plan (the default)")
    parser.parse_args(argv)
    print(PLAN, file=sys.stderr)
    print("M2 is not implemented; nothing was written.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
