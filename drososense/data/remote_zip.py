"""Read individual members of a remote ZIP/ZIP64 archive over HTTP Range.

D3 is published as a single 21.1 GB ``dataset.zip`` of which only ~1 MB is the
sensor tables this project uses. Downloading the whole archive to read 0.005 % of
it is wasteful and was, incorrectly, recorded as a resource blocker. This module
reads the archive's own central directory through ``Range`` requests and fetches
only the members that were asked for.

The implementation is deliberately self-contained (``struct`` + ``zlib``) rather
than depending on a third-party remote-zip package: the archive layout is
load-bearing for a reproducibility claim, so it is better to have the parser in
the repository where it can be read and tested than behind an unpinned
dependency.

ZIP64 is not optional here — the archive is 21.1 GB, so the classic 32-bit
central-directory offset is saturated (``0xFFFFFFFF``) and the real values live in
the ZIP64 end-of-central-directory record.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

# Signatures, little-endian, as they appear on disk.
_SIG_LOCAL = 0x04034B50
_SIG_CENTRAL = 0x02014B50
_SIG_EOCD = 0x06054B50
_SIG_ZIP64_EOCD = 0x06064B50
_SIG_ZIP64_LOCATOR = 0x07064B50

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_EOCD = struct.Struct("<IHHHHIIH")
_ZIP64_EOCD = struct.Struct("<IQHHIIQQQQ")
_ZIP64_LOCATOR = struct.Struct("<IIQI")

# Default number of trailing bytes scanned for the end-of-central-directory
# record. The EOCD comment field is at most 65535 bytes, so 64 KiB + slack always
# contains it.
_TAIL_SCAN_BYTES = 1 << 16
_TAIL_SCAN_SLACK = 1024

# ZIP compression methods this module can decode.
_METHOD_STORED = 0
_METHOD_DEFLATE = 8

# Bit 3 of the general-purpose flags: sizes and CRC are written into a trailing
# data descriptor instead of the local header. Central-directory values are still
# authoritative, so this only matters when validating the local header.
_FLAG_DATA_DESCRIPTOR = 1 << 3


class RemoteZipError(RuntimeError):
    """Raised when the archive cannot be parsed or a member cannot be read."""


@dataclass(frozen=True)
class RemoteZipEntry:
    """One member of a remote archive.

    Attributes:
        name: Member path inside the archive, as stored.
        method: ZIP compression method (0 stored, 8 deflate).
        crc32: Expected CRC-32 of the uncompressed bytes.
        compressed_size: Size of the stored (possibly deflated) bytes.
        uncompressed_size: Size after decompression.
        header_offset: Offset of this member's local file header.
    """

    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    header_offset: int

    @property
    def is_directory(self) -> bool:
        """Whether the entry is a directory marker rather than a file."""
        return self.name.endswith("/")


class HttpRangeReader:
    """Random-access reader over an HTTP resource that supports ``Range``.

    Args:
        url: URL of the resource.
        session: Optional ``requests.Session``; one is created when omitted.
        timeout: Per-request timeout in seconds.

    Raises:
        RemoteZipError: If the server does not advertise byte-range support.
    """

    def __init__(self, url: str, session: object | None = None, timeout: float = 120.0) -> None:
        import requests

        self.url = url
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()
        self._size: int | None = None

    @property
    def size(self) -> int:
        """Total length of the resource in bytes.

        Returns:
            Content length as reported by the server.

        Raises:
            RemoteZipError: If the server answers without a content length.
        """
        if self._size is None:
            response = self._session.head(
                self.url, timeout=self.timeout, allow_redirects=True
            )
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length is None:
                raise RemoteZipError(f"{self.url}: server did not report a content length")
            self._size = int(length)
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``offset``.

        Args:
            offset: Byte offset from the start of the resource.
            length: Number of bytes to read.

        Returns:
            The bytes read.

        Raises:
            RemoteZipError: If the server ignores the range request or returns
                fewer bytes than requested.
        """
        if length <= 0:
            return b""
        end = offset + length - 1
        response = self._session.get(
            self.url,
            headers={"Range": f"bytes={offset}-{end}"},
            timeout=self.timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RemoteZipError(
                f"{self.url}: expected HTTP 206 for a range request, got "
                f"{response.status_code}. The server does not support byte ranges."
            )
        payload = response.content
        if len(payload) != length:
            raise RemoteZipError(
                f"{self.url}: requested {length} bytes at {offset}, received {len(payload)}"
            )
        return payload


class RemoteZip:
    """A ZIP64 archive that is read member-by-member over HTTP.

    Args:
        url: URL of the archive.
        reader: Optional pre-built reader (used by tests to avoid a network).
    """

    def __init__(self, url: str, reader: HttpRangeReader | None = None) -> None:
        self.url = url
        self.reader = reader if reader is not None else HttpRangeReader(url)
        self._entries: tuple[RemoteZipEntry, ...] | None = None

    @property
    def entries(self) -> tuple[RemoteZipEntry, ...]:
        """Every member of the archive, in central-directory order.

        Returns:
            The archive's members.
        """
        if self._entries is None:
            self._entries = self._read_central_directory()
        return self._entries

    def names(self) -> tuple[str, ...]:
        """Member names, in archive order.

        Returns:
            Tuple of member paths.
        """
        return tuple(entry.name for entry in self.entries)

    def find(self, predicate) -> tuple[RemoteZipEntry, ...]:
        """Return members whose name satisfies ``predicate``.

        Args:
            predicate: Callable taking a member name and returning a bool.

        Returns:
            Matching members.
        """
        return tuple(e for e in self.entries if not e.is_directory and predicate(e.name))

    def read(self, entry: RemoteZipEntry) -> bytes:
        """Fetch and decompress one member.

        The member's CRC-32 is verified against the central directory, so a
        truncated or corrupted transfer cannot be mistaken for good data.

        Args:
            entry: The member to read.

        Returns:
            The decompressed bytes.

        Raises:
            RemoteZipError: On an unsupported method, a short read, or a CRC
                mismatch.
        """
        local = self.reader.read(entry.header_offset, _LOCAL_HEADER.size)
        (signature, _version, flags, _method, _time, _date, _crc, _csize, _usize,
         name_len, extra_len) = _LOCAL_HEADER.unpack(local)
        if signature != _SIG_LOCAL:
            raise RemoteZipError(f"{entry.name}: no local file header at {entry.header_offset}")

        data_offset = entry.header_offset + _LOCAL_HEADER.size + name_len + extra_len
        payload = self.reader.read(data_offset, entry.compressed_size)

        if entry.method == _METHOD_STORED:
            raw = payload
        elif entry.method == _METHOD_DEFLATE:
            try:
                raw = zlib.decompress(payload, -zlib.MAX_WBITS)
            except zlib.error as exc:  # pragma: no cover - only on a corrupt archive
                raise RemoteZipError(f"{entry.name}: deflate stream is corrupt: {exc}") from exc
        else:
            raise RemoteZipError(
                f"{entry.name}: compression method {entry.method} is not supported"
            )

        if len(raw) != entry.uncompressed_size:
            raise RemoteZipError(
                f"{entry.name}: decompressed to {len(raw)} bytes, central directory "
                f"declares {entry.uncompressed_size}"
            )
        actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
        if actual_crc != entry.crc32:
            raise RemoteZipError(
                f"{entry.name}: CRC-32 mismatch (expected {entry.crc32:08x}, got {actual_crc:08x})"
            )
        return raw

    def _read_central_directory(self) -> tuple[RemoteZipEntry, ...]:
        """Parse the archive's central directory.

        Returns:
            Every member, in archive order.

        Raises:
            RemoteZipError: If the end-of-central-directory records are absent or
                inconsistent.
        """
        size = self.reader.size
        tail_len = min(size, _TAIL_SCAN_BYTES + _TAIL_SCAN_SLACK)
        tail = self.reader.read(size - tail_len, tail_len)

        eocd_index = tail.rfind(struct.pack("<I", _SIG_EOCD))
        if eocd_index < 0:
            raise RemoteZipError(f"{self.url}: no end-of-central-directory record in the last "
                                 f"{tail_len} bytes")
        (_sig, _disk, _cd_disk, _n_disk, n_total, cd_size, cd_offset, _comment_len) = (
            _EOCD.unpack_from(tail, eocd_index)
        )

        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:
            locator_index = tail.rfind(struct.pack("<I", _SIG_ZIP64_LOCATOR), 0, eocd_index)
            if locator_index < 0:
                raise RemoteZipError(f"{self.url}: 32-bit EOCD is saturated but no ZIP64 locator")
            _sig, _disk, zip64_offset, _total_disks = _ZIP64_LOCATOR.unpack_from(
                tail, locator_index
            )
            zip64_record = self.reader.read(zip64_offset, _ZIP64_EOCD.size)
            (
                _sig,
                _record_size,
                _version_made,
                _version_needed,
                _disk,
                _cd_disk,
                _n_disk,
                n_total,
                cd_size,
                cd_offset,
            ) = _ZIP64_EOCD.unpack_from(zip64_record, 0)

        directory = self.reader.read(cd_offset, cd_size)
        return self._parse_entries(directory, n_total)

    @staticmethod
    def _parse_entries(directory: bytes, expected_count: int) -> tuple[RemoteZipEntry, ...]:
        """Parse central-directory bytes into entries.

        Args:
            directory: Raw central-directory bytes.
            expected_count: Member count from the end-of-central-directory record.

        Returns:
            The parsed members.

        Raises:
            RemoteZipError: If an entry header is malformed.
        """
        entries: list[RemoteZipEntry] = []
        cursor = 0
        while cursor + _CENTRAL_HEADER.size <= len(directory):
            fields = _CENTRAL_HEADER.unpack_from(directory, cursor)
            if fields[0] != _SIG_CENTRAL:
                break
            (signature, _vmade, _vneed, _flags, method, _time, _date, crc, csize, usize,
             name_len, extra_len, comment_len, _disk_start, _internal, _external,
             header_offset) = fields
            name_start = cursor + _CENTRAL_HEADER.size
            name = directory[name_start:name_start + name_len].decode("utf-8", errors="replace")
            extra = directory[name_start + name_len:name_start + name_len + extra_len]
            usize, csize, header_offset = _apply_zip64_extra(
                extra, usize, csize, header_offset, name
            )
            entries.append(
                RemoteZipEntry(
                    name=name,
                    method=method,
                    crc32=crc,
                    compressed_size=csize,
                    uncompressed_size=usize,
                    header_offset=header_offset,
                )
            )
            cursor = name_start + name_len + extra_len + comment_len

        if len(entries) != expected_count:
            raise RemoteZipError(
                f"central directory declares {expected_count} members, parsed {len(entries)}"
            )
        return tuple(entries)


def _apply_zip64_extra(
    extra: bytes, usize: int, csize: int, header_offset: int, name: str
) -> tuple[int, int, int]:
    """Replace saturated 32-bit central-directory values with their ZIP64 values.

    Args:
        extra: The entry's extra field bytes.
        usize: 32-bit uncompressed size from the central directory.
        csize: 32-bit compressed size from the central directory.
        header_offset: 32-bit local-header offset from the central directory.
        name: Member name, for error messages.

    Returns:
        ``(uncompressed_size, compressed_size, header_offset)`` with ZIP64 values
        substituted wherever the 32-bit fields are saturated.

    Raises:
        RemoteZipError: If the ZIP64 extra field is absent when it is required.
    """
    needs_zip64 = 0xFFFFFFFF in (usize, csize, header_offset)
    if not needs_zip64:
        return usize, csize, header_offset
    if len(extra) < 4 or struct.unpack_from("<HH", extra, 0)[0] != 0x0001:
        raise RemoteZipError(
            f"{name}: a 32-bit size field is saturated but the ZIP64 extra field is missing"
        )
    cursor = 4
    if usize == 0xFFFFFFFF:
        usize = struct.unpack_from("<Q", extra, cursor)[0]
        cursor += 8
    if csize == 0xFFFFFFFF:
        csize = struct.unpack_from("<Q", extra, cursor)[0]
        cursor += 8
    if header_offset == 0xFFFFFFFF:
        header_offset = struct.unpack_from("<Q", extra, cursor)[0]
        cursor += 8
    return usize, csize, header_offset


__all__ = ["HttpRangeReader", "RemoteZip", "RemoteZipEntry", "RemoteZipError"]
