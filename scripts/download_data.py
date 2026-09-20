#!/usr/bin/env python
"""Acquire dataset files described by their manifests, and verify integrity.

Nothing is downloaded that is not described in a manifest, and nothing is
accepted that does not match the manifest's SHA-256. A dataset whose manifest
says it is not automatically available is never silently skipped: the script
prints the blocker and the manual steps, and exits non-zero.

Examples
--------
    python scripts/download_data.py --dataset d1_beef_controlled
    python scripts/download_data.py --all --verify-only
    python scripts/download_data.py --dataset d3_rainbow_trout --accept-large-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.data.manifest import (  # noqa: E402
    AvailabilityStatus,
    DatasetManifest,
    load_all_manifests,
    load_manifest,
    manifest_path,
    sha256_of,
    verify_manifest,
)
from drososense.utils.paths import DATA_RAW_DIR, ensure_dir  # noqa: E402

# Downloads above this size need an explicit opt-in flag.
LARGE_DOWNLOAD_BYTES = 1 << 30  # 1 GiB


def download_file(url: str, destination: Path, expected_size: int | None) -> None:
    """Stream a URL to disk.

    Args:
        url: Direct download URL.
        destination: Where to write the file.
        expected_size: Expected byte count, used to sanity-check the transfer.

    Raises:
        RuntimeError: If the transfer fails or is short.
    """
    import requests

    ensure_dir(destination.parent)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        written = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
    if expected_size is not None and written != expected_size:
        raise RuntimeError(
            f"{destination.name}: downloaded {written} bytes, manifest declares {expected_size}"
        )


def process_manifest(
    manifest: DatasetManifest,
    verify_only: bool,
    accept_large: bool,
) -> tuple[bool, str]:
    """Fetch or verify one dataset according to its manifest.

    Args:
        manifest: The dataset manifest.
        verify_only: Skip downloading; only check what is on disk.
        accept_large: Permit downloads above :data:`LARGE_DOWNLOAD_BYTES`.

    Returns:
        ``(ok, message)``.
    """
    base = DATA_RAW_DIR / manifest.dataset_id

    if verify_only:
        report = verify_manifest(manifest, base)
        if report["ok"]:
            return True, f"{manifest.dataset_id}: verified ({len(report['present'])} file(s))"
        return False, f"{manifest.dataset_id}: {report['missing']} missing, {report['corrupt']}"

    if manifest.status is not AvailabilityStatus.AUTO:
        lines = [f"{manifest.dataset_id}: NOT AUTOMATICALLY AVAILABLE ({manifest.status.value})."]
        if manifest.blockers:
            lines.append("  blockers:")
            lines.extend(f"    - {b}" for b in manifest.blockers)
        if manifest.manual_steps:
            lines.append("  manual steps:")
            lines.extend(f"    {i + 1}. {s}" for i, s in enumerate(manifest.manual_steps))
        return False, "\n".join(lines)

    lines = []
    for entry in manifest.files:
        if not entry.download_url:
            if entry.required:
                return False, f"{manifest.dataset_id}: {entry.name} is required but has no URL"
            continue

        if entry.size_bytes and entry.size_bytes > LARGE_DOWNLOAD_BYTES and not accept_large:
            return False, (
                f"{manifest.dataset_id}: {entry.name} is "
                f"{entry.size_bytes / 1e9:.1f} GB, above the "
                f"{LARGE_DOWNLOAD_BYTES / 1e9:.0f} GB threshold. Re-run with "
                f"--accept-large-download to fetch it."
            )

        destination = base / entry.name
        already_ok = (
            destination.is_file()
            and entry.sha256 is not None
            and sha256_of(destination) == entry.sha256.lower()
        )
        if already_ok:
            lines.append(f"  {entry.name}: already present and verified")
            continue

        if destination.exists():
            # Re-download rather than trust a file that failed its checksum.
            destination.unlink()

        download_file(entry.download_url, destination, entry.size_bytes)
        if entry.sha256:
            actual = sha256_of(destination)
            if actual != entry.sha256.lower():
                destination.unlink()
                return False, (
                    f"{manifest.dataset_id}: {entry.name} checksum mismatch\n"
                    f"    expected {entry.sha256.lower()}\n    actual   {actual}"
                )
        lines.append(f"  {entry.name}: downloaded and verified")

    # Confirm the whole dataset matches the manifest, not just what we fetched.
    report = verify_manifest(manifest, base)
    if not report["ok"]:
        return False, f"{manifest.dataset_id}: post-download verification failed: {report}"
    return True, f"{manifest.dataset_id}: OK\n" + "\n".join(lines)


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", default=[], help="dataset id (repeatable)")
    parser.add_argument("--all", action="store_true", help="process every manifest")
    parser.add_argument("--verify-only", action="store_true", help="check without downloading")
    parser.add_argument(
        "--accept-large-download",
        action="store_true",
        help=f"permit downloads over {LARGE_DOWNLOAD_BYTES / 1e9:.0f} GB",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset <id> (repeatable) or --all")

    if args.all:
        manifests = load_all_manifests()
    else:
        manifests = {}
        for dataset_id in args.dataset:
            path = manifest_path(dataset_id)
            if not path.is_file():
                print(f"no manifest for {dataset_id!r} at {path}", file=sys.stderr)
                return 2
            manifests[dataset_id] = load_manifest(path)

    failures = 0
    for dataset_id, manifest in manifests.items():
        ok, message = process_manifest(manifest, args.verify_only, args.accept_large_download)
        stream = sys.stdout if ok else sys.stderr
        print(f"[{'ok' if ok else 'BLOCKED'}] {message}", file=stream)
        if not ok:
            failures += 1

    if failures:
        print(
            f"\n{failures} dataset(s) unavailable. This is reported, not worked around: "
            f"the project does not substitute another dataset for a missing one.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
