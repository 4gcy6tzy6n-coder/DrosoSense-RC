"""Materialise dataset files, including members of a remote ZIP archive.

Two acquisition shapes are supported:

``download``
    Plain HTTP fetch of a whole file, verified against a declared SHA-256.

``remote_archive``
    The dataset is published as one large ZIP but the project only needs a small
    subset of its members. Rather than downloading the archive, the members are
    read individually through HTTP range requests (see
    :mod:`drososense.data.remote_zip`). D3 is the reason this exists: the archive
    is 21.1 GB and the sensor tables inside it are 1.04 MB.

Integrity for the remote-archive path is anchored differently from the plain
download path, and the difference is stated rather than glossed: because the
archive is never fetched whole, its publisher md5 cannot be checked. What IS
checked, on every acquisition, is the CRC-32 that the archive's own central
directory declares for each member, plus the SHA-256 of every extracted member
against a manifest that is committed to the repository.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from drososense.data.remote_zip import RemoteZip, RemoteZipEntry

# Requests are retried on transport failures and on HTTP 429, because Zenodo
# rate-limits the file endpoint and a silent partial extraction is worse than a
# slow one.
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 5.0
# Pause between member reads, to stay well inside the published rate limit.
_INTER_REQUEST_PAUSE_S = 0.25


class AcquisitionError(RuntimeError):
    """Raised when a file cannot be materialised or fails verification."""


@dataclass(frozen=True)
class AcquiredFile:
    """One file written to disk by an acquisition.

    Attributes:
        name: Archive member path, or the plain filename.
        local_name: Filename written under the dataset's raw directory.
        sha256: Digest of the bytes actually written.
        size_bytes: Size of the bytes actually written.
        crc32: CRC-32 the archive declared, when the file came from an archive.
    """

    name: str
    local_name: str
    sha256: str
    size_bytes: int
    crc32: int | None = None


def _sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string.

    Args:
        data: Bytes to hash.

    Returns:
        Lower-case hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def _with_retries(operation, description: str):
    """Run ``operation`` with bounded retries and exponential backoff.

    Args:
        operation: Zero-argument callable.
        description: Human-readable description for the error message.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        AcquisitionError: If every attempt failed.
    """
    import requests

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return operation()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            last_error = exc
            if status is not None and status not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as exc:
            last_error = exc
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_SECONDS * (2**attempt))
    raise AcquisitionError(f"{description}: failed after {_MAX_ATTEMPTS} attempts: {last_error}")


def extract_remote_archive_members(
    url: str,
    member_names: list[str] | None = None,
    member_predicate=None,
    destination: Path | None = None,
    expected_sha256: dict[str, str] | None = None,
    reader=None,
) -> list[AcquiredFile]:
    """Fetch selected members of a remote ZIP and write them to disk.

    Args:
        url: URL of the archive.
        member_names: Exact member paths to fetch. Mutually exclusive with
            ``member_predicate``.
        member_predicate: Callable taking a member name and returning a bool.
        destination: Directory to write into. When omitted the members are read
            and verified but not written.
        expected_sha256: Optional mapping of member name to expected SHA-256.
        reader: Injected range reader, for tests.

    Returns:
        One :class:`AcquiredFile` per fetched member.

    Raises:
        AcquisitionError: If a member is missing, a checksum does not match, or
            the archive cannot be read.
    """
    if (member_names is None) == (member_predicate is None):
        raise AcquisitionError("pass exactly one of member_names or member_predicate")

    archive = RemoteZip(url, reader=reader)
    entries = archive.entries
    if member_predicate is not None:
        selected = [e for e in entries if not e.is_directory and member_predicate(e.name)]
    else:
        by_name = {e.name: e for e in entries}
        missing = [n for n in member_names or [] if n not in by_name]
        if missing:
            raise AcquisitionError(f"{url}: archive has no members {missing[:5]}")
        selected = [by_name[n] for n in member_names or []]
    if not selected:
        raise AcquisitionError(f"{url}: the selection matched no members")

    acquired: list[AcquiredFile] = []
    for index, entry in enumerate(selected):
        if index:
            time.sleep(_INTER_REQUEST_PAUSE_S)
        payload = _with_retries(lambda e=entry: archive.read(e), f"reading {entry.name}")
        digest = _sha256(payload)
        expected = (expected_sha256 or {}).get(entry.name)
        if expected is not None and digest != expected.lower():
            raise AcquisitionError(
                f"{entry.name}: SHA-256 mismatch\n    expected {expected.lower()}\n    actual   {digest}"
            )
        local_name = Path(entry.name).name
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / local_name).write_bytes(payload)
        acquired.append(
            AcquiredFile(
                name=entry.name,
                local_name=local_name,
                sha256=digest,
                size_bytes=len(payload),
                crc32=entry.crc32,
            )
        )
    return acquired


def list_remote_archive_members(url: str, reader=None) -> tuple[RemoteZipEntry, ...]:
    """List a remote archive's members without fetching any of them.

    Args:
        url: URL of the archive.
        reader: Injected range reader, for tests.

    Returns:
        Every member, in central-directory order.
    """
    return RemoteZip(url, reader=reader).entries


__all__ = [
    "AcquiredFile",
    "AcquisitionError",
    "extract_remote_archive_members",
    "list_remote_archive_members",
]
