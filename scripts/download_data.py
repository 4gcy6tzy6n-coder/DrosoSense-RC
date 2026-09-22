#!/usr/bin/env python
"""Acquire dataset files described by their manifests, and verify integrity.

Nothing is downloaded that is not described in a manifest, and nothing is
accepted that does not match the manifest's SHA-256. A dataset whose manifest
says it is not automatically available is never silently skipped: the script
prints the blocker and the manual steps, and exits non-zero.

Three acquisition shapes exist:

``download``
    A plain HTTP fetch of a whole file, verified against its SHA-256.

``local_archive_member``
    A file extracted from an archive the script just downloaded (D2's five beef
    cuts inside the 188 KB v3 zip).

``remote_archive_member``
    A member read out of a remote ZIP by HTTP range request, without downloading
    the archive (D3: 210 CSVs, 1.04 MB, from a 21.1 GB archive). The extracted
    SHA-256 set is written to the manifest's member listing, which is what makes
    the extraction reproducible rather than a one-off.

Examples
--------
    python scripts/download_data.py --dataset d2_beef_uncontrolled
    python scripts/download_data.py --dataset d3_rainbow_trout
    python scripts/download_data.py --all --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.data.acquisition import (  # noqa: E402
    AcquisitionError,
    extract_remote_archive_members,
)
from drososense.data.manifest import (  # noqa: E402
    ORIGIN_DOWNLOAD,
    ORIGIN_REMOTE_ARCHIVE_MEMBER,
    AvailabilityStatus,
    DatasetManifest,
    load_all_manifests,
    load_manifest,
    manifest_path,
    sha256_of,
    verify_manifest,
)
from drososense.utils.paths import DATA_MANIFESTS_DIR, DATA_RAW_DIR, ensure_dir  # noqa: E402

# Downloads above this size need an explicit opt-in flag. It applies only to
# whole-file downloads; a remote archive's members are read by range request and
# never trip the threshold.
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
        if entry.origin != ORIGIN_DOWNLOAD:
            continue
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

        destination = base / entry.resolved_local_name
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

    if manifest.archive is not None and manifest.archive.kind == "remote_zip":
        ok, message = _materialise_remote_archive(manifest, base, verify_only=False)
        if not ok:
            return False, message
        lines.extend(message.splitlines())

    ok, message = _extract_local_archive_members(manifest, base)
    if not ok:
        return False, message
    lines.extend(message.splitlines())

    # Confirm the whole dataset matches the manifest, not just what we fetched.
    report = verify_manifest(manifest, base)
    if not report["ok"]:
        return False, f"{manifest.dataset_id}: post-download verification failed: {report}"
    return True, f"{manifest.dataset_id}: OK\n" + "\n".join(lines)


def _materialise_remote_archive(
    manifest: DatasetManifest, base: Path, verify_only: bool
) -> tuple[bool, str]:
    """Extract a remote archive's declared members using HTTP range requests.

    Args:
        manifest: The dataset manifest, which must declare an archive.
        base: The dataset's raw directory.
        verify_only: Check the members already on disk against the member
            manifest instead of fetching them again.

    Returns:
        ``(ok, message)``.
    """
    archive = manifest.archive
    if archive is None:  # pragma: no cover - guarded by the caller
        return True, ""
    member_manifest_path = DATA_MANIFESTS_DIR / archive.member_manifest
    if verify_only and member_manifest_path.is_file():
        from drososense.data.manifest import load_member_manifest

        members = load_member_manifest(member_manifest_path)
        missing = [m for m in members if not (base / m.resolved_local_name).is_file()]
        if missing:
            return False, f"  {len(missing)} member(s) missing from {base}"
        return True, f"  {len(members)} archive member(s) present (verify-only)"

    if not member_manifest_path.is_file() and archive.member_glob:
        return _regenerate_member_manifest(manifest, base)

    if not member_manifest_path.is_file():
        return False, (
            f"{manifest.dataset_id}: no member manifest at {member_manifest_path}; "
            f"cannot verify the extraction"
        )

    from drososense.data.manifest import load_member_manifest

    members = load_member_manifest(member_manifest_path)
    expected = {m.name: (m.sha256 or "") for m in members}
    to_fetch = [
        m
        for m in members
        if not (
            (base / m.resolved_local_name).is_file()
            and m.sha256
            and sha256_of(base / m.resolved_local_name) == m.sha256.lower()
        )
    ]
    if not to_fetch:
        return True, f"  {len(members)} archive member(s) already present and verified"

    try:
        acquired = extract_remote_archive_members(
            archive.url,
            member_names=[m.name for m in to_fetch],
            destination=base,
            expected_sha256=expected,
        )
    except AcquisitionError as exc:
        return False, f"{manifest.dataset_id}: archive extraction failed: {exc}"
    return True, f"  {len(acquired)} archive member(s) extracted and verified by SHA-256"


def _regenerate_member_manifest(manifest: DatasetManifest, base: Path) -> tuple[bool, str]:
    """Discover a remote archive's members from its central directory and fetch them.

    Used the first time a dataset is acquired, when no member listing exists yet.
    The listing written here is the record of what was pulled, so it is only ever
    written from bytes that passed the archive's own CRC-32.

    Args:
        manifest: The dataset manifest.
        base: The dataset's raw directory.

    Returns:
        ``(ok, message)``.
    """
    archive = manifest.archive
    if archive is None:  # pragma: no cover - guarded by the caller
        return True, ""
    import fnmatch

    try:
        acquired = extract_remote_archive_members(
            archive.url,
            member_predicate=lambda name: fnmatch.fnmatch(name, archive.member_glob),
            destination=base,
        )
    except AcquisitionError as exc:
        return False, f"{manifest.dataset_id}: archive extraction failed: {exc}"

    payload = {
        "dataset_id": manifest.dataset_id,
        "archive_member_glob": archive.member_glob,
        "member_count": len(acquired),
        "total_bytes": sum(a.size_bytes for a in acquired),
        "members": [
            {
                "member": a.name,
                "local_name": a.local_name,
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
                "crc32": f"{a.crc32:08x}" if a.crc32 is not None else None,
            }
            for a in sorted(acquired, key=lambda a: a.name)
        ],
    }
    path = DATA_MANIFESTS_DIR / archive.member_manifest
    ensure_dir(path.parent)
    path.write_text(_render_member_manifest(payload), encoding="utf-8")
    return True, (
        f"  {len(acquired)} archive member(s) extracted; member listing written to {path.name}"
    )


def _render_member_manifest(payload: dict) -> str:
    """Render a member manifest as YAML with a stable, readable layout.

    Args:
        payload: The member listing.

    Returns:
        The YAML document.
    """
    lines = [
        f"# {payload['dataset_id']} — generated member listing. DO NOT EDIT BY HAND.",
        "#",
        "# Produced by scripts/download_data.py when the sensor tables were extracted",
        "# from the published archive by HTTP range request. One entry per member that",
        "# was pulled, with the SHA-256 of the bytes actually written.",
        "#",
        "# This file is the integrity anchor for the dataset: the archive is never",
        "# downloaded whole, so its publisher digest cannot be checked. What IS checked",
        "# is the archive's own CRC-32 for each member (verified during extraction) and",
        "# the SHA-256 below (re-verified on every `--verify-only` run).",
        "",
        f"dataset_id: {payload['dataset_id']}",
        f"archive_member_glob: \"{payload['archive_member_glob']}\"",
        f"member_count: {payload['member_count']}",
        f"total_bytes: {payload['total_bytes']}",
        "members:",
    ]
    for entry in payload["members"]:
        lines.append(f"  - member: \"{entry['member']}\"")
        lines.append(f"    local_name: \"{entry['local_name']}\"")
        lines.append(f"    sha256: \"{entry['sha256']}\"")
        lines.append(f"    size_bytes: {entry['size_bytes']}")
        lines.append(f"    crc32: \"{entry['crc32']}\"")
    return "\n".join(lines) + "\n"


def _extract_local_archive_members(manifest: DatasetManifest, base: Path) -> tuple[bool, str]:
    """Extract members of an archive that was downloaded whole.

    Args:
        manifest: The dataset manifest.
        base: The dataset's raw directory.

    Returns:
        ``(ok, message)``.
    """
    members = [f for f in manifest.files if f.origin == "series_member"]
    if not members:
        return True, ""
    archives = [f for f in manifest.files if f.origin == ORIGIN_DOWNLOAD and f.name.endswith(".zip")]
    if not archives:
        return False, f"{manifest.dataset_id}: series_member entries but no archive to extract"

    pending = [
        m
        for m in members
        if not (
            (base / m.resolved_local_name).is_file()
            and m.sha256
            and sha256_of(base / m.resolved_local_name) == m.sha256.lower()
        )
    ]
    if not pending:
        return True, f"  {len(members)} series file(s) already present and verified"

    archive_path = base / archives[0].resolved_local_name
    if not archive_path.is_file():
        return False, f"{manifest.dataset_id}: archive {archive_path.name} is not on disk"
    written = 0
    with zipfile.ZipFile(archive_path) as bundle:
        by_basename = {Path(n).name: n for n in bundle.namelist()}
        for member in pending:
            source = by_basename.get(member.resolved_local_name)
            if source is None:
                return False, (
                    f"{manifest.dataset_id}: {member.resolved_local_name} is not in "
                    f"{archive_path.name}"
                )
            (base / member.resolved_local_name).write_bytes(bundle.read(source))
            if member.sha256 and sha256_of(base / member.resolved_local_name) != member.sha256.lower():
                return False, (
                    f"{manifest.dataset_id}: {member.resolved_local_name} does not match its "
                    f"recorded digest after extraction"
                )
            written += 1
    return True, f"  {written} series file(s) extracted from {archive_path.name} and verified"


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
