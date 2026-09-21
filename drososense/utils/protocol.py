"""Protocol freeze bookkeeping.

The freeze rule is only real if a post-freeze edit is detectable. This module
computes the digest of the active protocol, records it in a sidecar file, and
compares the two — which is what turns "the protocol was frozen before the data
was touched" from a claim into a check that can fail.

    python -m drososense.utils.protocol --check      # verify, non-zero on drift
    python -m drososense.utils.protocol --write      # (re)record the digest
    python -m drososense.utils.protocol --show       # print the current state

``--write`` is only legitimate immediately after authoring a NEW protocol
version. Editing a frozen file and re-running ``--write`` would defeat the check,
which is why the command prints what it is doing and the digest is also recorded
in the project README and the PR that introduced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from drososense.utils.config import (
    load_protocol,
    protocol_sha256,
    recorded_protocol_sha256,
    verify_protocol_freeze,
)
from drososense.utils.paths import PROTOCOL_PATH, PROTOCOL_SHA256_PATH


def write_digest(path: str | Path | None = None, sidecar: str | Path | None = None) -> Path:
    """Record a protocol's digest in its sidecar file.

    Args:
        path: Protocol file; defaults to the active one.
        sidecar: Sidecar path; defaults to the active protocol's.

    Returns:
        The sidecar path written.
    """
    target = Path(path) if path is not None else PROTOCOL_PATH
    out = Path(sidecar) if sidecar is not None else PROTOCOL_SHA256_PATH
    digest = protocol_sha256(target)
    header = (
        "# SHA-256 of the frozen protocol file named below.\n"
        "# Regenerate ONLY when authoring a new protocol version, never to silence\n"
        "# a failing freeze check.\n"
    )
    out.write_text(f"{header}{digest}  {target.name}\n", encoding="utf-8")
    return out


def freeze_status(
    path: str | Path | None = None, sidecar: str | Path | None = None
) -> dict[str, Any]:
    """Return the full freeze state of a protocol.

    The protocol file itself is REQUIRED to keep
    ``freeze_evidence.data_contact_log.first_test_evaluation_at`` null (§1), and
    the live record lives in ``results/tables/data_contact_log.json``. Reporting
    only the frozen placeholder made ``--check`` print "first test evaluation:
    not started" while the live log already held two dozen contacts, which a
    reader would take as the opposite of the truth (review item H4). Both values
    are therefore reported, and ``first_test_evaluation_at`` is the LIVE one.

    Args:
        path: Protocol file; defaults to the active one.
        sidecar: Sidecar path; defaults to the active protocol's.

    Returns:
        Mapping combining the digest comparison, the protocol's own declared
        freeze evidence, and the live contact log.
    """
    from drososense.evaluation.contact_log import contact_log_path, load_contact_log

    report = verify_protocol_freeze(path, sidecar)
    target = Path(path) if path is not None else PROTOCOL_PATH
    protocol = load_protocol(target)
    evidence = protocol.get("freeze_evidence", {})
    declared = evidence.get("data_contact_log", {})

    live_log = load_contact_log()
    live_first = live_log.first_test_evaluation_at
    live_entries = list(live_log.entries)
    live_datasets = list(live_log.datasets_touched)
    declared_first = declared.get("first_test_evaluation_at")

    if live_first:
        contact_status = f"{live_first} (live log)"
    elif live_entries:
        contact_status = "log has entries but none counts as a first test evaluation"
    else:
        contact_status = "not started"

    report.update(
        {
            "protocol_version": protocol.get("protocol_version"),
            "frozen": protocol.get("frozen"),
            "frozen_at": protocol.get("frozen_at"),
            "data_contact_log": declared,
            "declared_first_test_evaluation_at": declared_first,
            "live_contact_log_path": str(contact_log_path()),
            "live_entries": len(live_entries),
            "live_datasets_touched": live_datasets,
            "live_first_test_evaluation_at": live_first,
            "first_test_evaluation_at": live_first,
            "contact_status": contact_status,
            "test_evaluation_started": live_first is not None,
            "declared_matches_live": (
                declared_first is None or declared_first == live_first
            ),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify the recorded digest")
    group.add_argument("--write", action="store_true", help="(re)record the digest")
    group.add_argument("--show", action="store_true", help="print the freeze state")
    parser.add_argument("--protocol", default=None, help="protocol file to inspect")
    parser.add_argument("--sidecar", default=None, help="digest sidecar to use")
    args = parser.parse_args(argv)

    if args.write:
        written = write_digest(args.protocol, args.sidecar)
        print(f"recorded {protocol_sha256(args.protocol)} in {written}")
        return 0

    status = freeze_status(args.protocol, args.sidecar)
    if args.show:
        print(json.dumps(status, indent=2, sort_keys=True, default=str))
        return 0

    if not status["ok"]:
        print(
            f"FREEZE CHECK FAILED for {status['path']}\n"
            f"  recorded {status['recorded']}\n"
            f"  actual   {status['actual']}\n"
            f"The file changed after it was frozen. A change to a frozen protocol requires a new\n"
            f"version file (protocol_v1.2.yaml) that supersedes this one; re-recording the digest\n"
            f"of the edited file is not an amendment.",
            file=sys.stderr,
        )
        return 1
    print(
        f"freeze OK: {status['protocol_version']} frozen at {status['frozen_at']}, "
        f"sha256 {status['actual'][:12]}…"
    )
    print(
        f"  first test evaluation: {status['contact_status']}\n"
        f"    live log {status['live_contact_log_path']}: "
        f"{status['live_entries']} entr(ies), datasets {status['live_datasets_touched'] or '[]'}\n"
        f"    the frozen protocol's own field is null by design (§1) and is NOT the live state"
    )
    if not status["declared_matches_live"]:
        print(
            f"    WARNING: the protocol declares first_test_evaluation_at="
            f"{status['declared_first_test_evaluation_at']!r} but the live log says "
            f"{status['live_first_test_evaluation_at']!r}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
