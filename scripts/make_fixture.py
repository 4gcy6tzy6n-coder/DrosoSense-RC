#!/usr/bin/env python
"""Generate the synthetic e-nose fixture used for pipeline smoke tests.

The fixture is the only dataset in this repository with genuine specimen
identifiers, so it is the only one on which a protocol-compliant specimen-level
split can be demonstrated today. It is NOT evidence: every result derived from
it is written with ``evidence_class: synthetic_fixture``.

Example
-------
    python scripts/make_fixture.py --specimens 12 --timesteps 120
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.data.synthetic import FIXTURE_FILENAME, FixtureSpec, write_fixture  # noqa: E402
from drososense.utils.paths import DATA_RAW_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--specimens", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=120)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--noise-sd", type=float, default=0.35)
    parser.add_argument("--offset-sd", type=float, default=6.0)
    args = parser.parse_args(argv)

    spec = FixtureSpec(
        n_specimens=args.specimens,
        n_timesteps=args.timesteps,
        n_channels=args.channels,
        seed=args.seed,
        noise_sd=args.noise_sd,
        specimen_offset_sd=args.offset_sd,
    )

    destination = DATA_RAW_DIR / "synthetic_enose" / FIXTURE_FILENAME
    write_fixture(destination, spec)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"wrote {destination}")
    print(f"  {spec.n_specimens} specimens x {spec.n_timesteps} timesteps "
          f"x {spec.n_channels + 2} channels")
    print(f"  generator seed: {spec.seed}")
    print(f"  sha256: {digest}")
    print("  NOT observed data — smoke tests only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
