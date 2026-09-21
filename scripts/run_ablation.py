#!/usr/bin/env python
"""E2-E6 ablation experiments. NOT IMPLEMENTED IN M1.

E2 (topology) and E5 (efficiency) require the connectome reservoir from M3.
E3 (low-data) and E4 (robustness) are defined against R0/R2/R4 and are therefore
also blocked on M3, even though their mechanics are model-agnostic.

The registry below records what each experiment will do and what it depends on,
so the experimental design is reviewable now rather than reconstructed later.
Running this exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys

EXPERIMENTS: dict[str, dict[str, object]] = {
    "E2_topology": {
        "question": "Does performance come from the real wiring, or from degree structure?",
        "design": "R0 real vs R2 degree-rewired vs R3 random sparse vs R4 ESN, "
                  "matched on nodes, edge count, spectral scaling, readout and split.",
        "primary_metric": "macro_f1",
        "depends_on": ["M2 connectome", "M3 reservoir"],
    },
    "E3_lowdata": {
        "question": "Does the biological reservoir degrade more slowly with less data?",
        "design": "Training specimens at 10 / 25 / 50 / 75 / 100%, sampled by specimen.",
        "primary_metric": "macro_f1 retention relative to the 100% condition",
        "depends_on": ["M3 reservoir"],
    },
    "E4_robustness": {
        "question": "Is the biological reservoir more robust to degraded sensing?",
        "design": "Channel dropout at p_drop in 0.1..0.5; per-sensor Delta F1; input "
                  "noise at sigma in 0.01..0.2; gain/offset/combined drift.",
        "primary_metric": "macro_f1 degradation curve",
        "depends_on": ["M3 reservoir"],
    },
    "E5_efficiency": {
        "question": "Does the frozen sparse substrate lower training cost?",
        "design": "Report TRAINABLE parameters (not total nodes), frozen parameters, "
                  "training time, ms/sample inference, peak memory, reservoir sparsity, "
                  "and MACs/FLOPs where computable.",
        "primary_metric": "trainable parameter count and inference latency",
        "depends_on": ["M3 reservoir"],
    },
    "E6_analysis": {
        "question": "What mechanism, if any, explains the differences?",
        "design": "Effective dimensionality (participation ratio), state separability, "
                  "linear memory capacity, and graph statistics correlated with "
                  "performance. Correlations are never written up as causal mechanisms.",
        "primary_metric": "n/a (analysis)",
        "depends_on": ["M3 reservoir"],
    },
}


def main(argv: list[str] | None = None) -> int:
    """List the planned ablations and refuse to run.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code, always non-zero.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--plan", action="store_true", help="print the plan (the default)")
    args = parser.parse_args(argv)

    selected = {args.experiment: EXPERIMENTS[args.experiment]} if args.experiment else EXPERIMENTS
    print(json.dumps(selected, indent=2), file=sys.stderr)
    print(
        "\nE2-E6 are blocked on M2/M3 and are not implemented in M1. "
        "Nothing was run and nothing was written.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
