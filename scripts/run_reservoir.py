#!/usr/bin/env python
"""M3 — connectome reservoir experiments. NOT IMPLEMENTED IN M1.

The standard echo-state network IS available in M1 and is run through
``scripts/run_baselines.py --models esn``. What is missing here is the
biological reservoir R0 and its topology controls, because the connectome
itself (M2) does not exist yet.

Running this exits non-zero and prints the plan. It does not produce a result.
"""

from __future__ import annotations

import argparse
import sys

PLAN = """
M3 — DrosoSense-RC reservoir (NOT STARTED)

Available now
  The standard leaky ESN (R4 control) is implemented in drososense/reservoir/esn.py
  and runs through scripts/run_baselines.py --models esn. It carries no
  biological claim.

Planned steps
  1. Implement the frozen-connectome reservoir with the M3 dynamics:
         h_t = (1 - alpha) h_{t-1} + alpha * tanh(g A_hat h_{t-1} + W_in x_t + b)
     with A_hat the normalised connectome from M2, and A_hat, W_in and b frozen.
  2. Train ONLY the readout. Classification: logistic / ridge classifier.
     Regression: ridge, solved as (H^T H + lambda I) W = H^T Y.
  3. Build the reservoir family, all matched on nodes, edge count, spectral
     scaling, readout, split and input mapping:
         R0 real fly      R1 weight-shuffled   R2 degree-rewired (core control)
         R3 random sparse R4 ER-ESN            R5 small-world   R6 dense random
  4. Run a separate ablation on input mapping (random vs biologically informed)
     so that an R0 effect cannot be confused with where the inputs land.

The comparison that decides the paper
  R0 vs R2. R2 preserves node count, edge count and the degree distribution and
  breaks only the specific wiring. If R0 does not beat R2, no topology claim is
  available, and the pre-registered narrative rules in
  configs/protocol_v1.yaml (section 12) apply.
"""


def main(argv: list[str] | None = None) -> int:
    """Report the M3 plan and refuse to run.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code, always non-zero.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="print the plan (the default)")
    parser.parse_args(argv)
    print(PLAN, file=sys.stderr)
    print(
        "For the available R4 control, run: "
        "python scripts/run_baselines.py --models esn ...",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
