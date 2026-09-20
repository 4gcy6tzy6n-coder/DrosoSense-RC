"""Dataset manifests: provenance, integrity and availability.

A manifest is the contract between the project and its data. It records where
each file came from, what it is licensed under, the SHA-256 that identifies the
exact revision used, and — when automated download is not possible — the manual
steps and the blocker.

The project's rule is that a dataset with no verifiable manifest is reported as
unavailable. It is never silently replaced with something else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from drososense.utils.config import load_yaml
from drososense.utils.paths import DATA_MANIFESTS_DIR, DATA_RAW_DIR


class AvailabilityStatus(str, Enum):
    """How obtainable a dataset currently is.

    Attributes:
        AUTO: Downloadable by ``scripts/download_data.py`` from a public URL.
        MANUAL: Public, but requires a human step the script cannot perform
            (account, licence acceptance, very large archive).
        BLOCKED: Public in principle, but an identified obstacle currently
            prevents acquisition.
        UNAVAILABLE: Cannot be obtained; the dataset must not be substituted.
    """

    AUTO = "auto"
    MANUAL = "manual"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ManifestFile:
    """One file belonging to a dataset.

    Attributes:
        name: Filename as published.
        sha256: Expected SHA-256 hex digest, when the publisher provides one.
        size_bytes: Expected size in bytes, when known.
        download_url: Direct download URL, when automated retrieval is allowed.
        required: Whether the dataset is usable without this file.
        note: Free-text caveat about this file.
    """

    name: str
    sha256: str | None = None
    size_bytes: int | None = None
    download_url: str | None = None
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class DatasetManifest:
    """Everything the project records about one dataset's provenance.

    Attributes:
        dataset_id: Stable identifier, matching the dataset config.
        display_name: Human-readable name.
        status: Current availability.
        synthetic: Whether the data is generated rather than observed.
        license: License string as published.
        license_url: Link to the license text.
        source_url: Canonical landing page.
        doi: DOI, when assigned.
        version: Published version identifier.
        citation: Citation the project must reproduce.
        files: Individual files with integrity information.
        schema_notes: What the columns are and what is missing.
        specimen_note: What is known about specimen identification.
        manual_steps: Exact steps when ``status`` is not ``auto``.
        blockers: Known obstacles, stated plainly.
        notes: Any remaining context.
    """

    dataset_id: str
    display_name: str
    status: AvailabilityStatus
    synthetic: bool = False
    license: str = ""
    license_url: str = ""
    source_url: str = ""
    doi: str = ""
    version: str = ""
    citation: str = ""
    files: tuple[ManifestFile, ...] = ()
    schema_notes: str = ""
    specimen_note: str = ""
    manual_steps: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """Whether the dataset can currently be evaluated."""
        return self.status in {AvailabilityStatus.AUTO, AvailabilityStatus.MANUAL}


def manifest_path(dataset_id: str) -> Path:
    """Resolve a dataset id to its manifest file.

    Args:
        dataset_id: Dataset identifier.

    Returns:
        Path to ``data/manifests/<dataset_id>.yaml``.
    """
    return DATA_MANIFESTS_DIR / f"{dataset_id}.yaml"


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a dataset manifest.

    Args:
        path: Path to a manifest YAML.

    Returns:
        The parsed manifest.

    Raises:
        ValueError: If required fields are missing or ``status`` is unknown.
    """
    path = Path(path)
    raw = load_yaml(path)
    for key in ("dataset_id", "display_name", "status"):
        if key not in raw:
            raise ValueError(f"{path}: manifest missing required field {key!r}")

    try:
        status = AvailabilityStatus(str(raw["status"]))
    except ValueError as exc:
        allowed = [s.value for s in AvailabilityStatus]
        raise ValueError(f"{path}: status must be one of {allowed}") from exc

    files = tuple(
        ManifestFile(
            name=entry["name"],
            sha256=entry.get("sha256"),
            size_bytes=entry.get("size_bytes"),
            download_url=entry.get("download_url"),
            required=bool(entry.get("required", True)),
            note=entry.get("note", ""),
        )
        for entry in raw.get("files", [])
    )

    if status is not AvailabilityStatus.AUTO and not raw.get("blockers") and not raw.get(
        "manual_steps"
    ):
        raise ValueError(
            f"{path}: status is {status.value!r}, so the manifest must state either "
            f"manual_steps or blockers — an unexplained unavailability is not acceptable"
        )

    return DatasetManifest(
        dataset_id=str(raw["dataset_id"]),
        display_name=str(raw["display_name"]),
        status=status,
        synthetic=bool(raw.get("synthetic", False)),
        license=str(raw.get("license", "")),
        license_url=str(raw.get("license_url", "")),
        source_url=str(raw.get("source_url", "")),
        doi=str(raw.get("doi", "")),
        version=str(raw.get("version", "")),
        citation=str(raw.get("citation", "")),
        files=files,
        schema_notes=str(raw.get("schema_notes", "")),
        specimen_note=str(raw.get("specimen_note", "")),
        manual_steps=tuple(raw.get("manual_steps", ())),
        blockers=tuple(raw.get("blockers", ())),
        notes=str(raw.get("notes", "")),
        extra=dict(raw.get("extra", {})),
    )


def sha256_of(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_bytes: Read size; the file is streamed, never fully loaded.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: DatasetManifest, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Check the on-disk state of a dataset against its manifest.

    Missing files do not raise — the caller decides whether a partial dataset is
    fatal. Integrity mismatches ARE reported prominently, because a checksum
    mismatch means the project is not using the revision it thinks it is.

    Args:
        manifest: Manifest to verify.
        base_dir: Directory holding the files; defaults to ``data/raw/<id>``.

    Returns:
        Report with ``present``, ``missing``, ``corrupt`` and ``ok`` entries.
    """
    base = Path(base_dir) if base_dir is not None else DATA_RAW_DIR / manifest.dataset_id
    present: list[str] = []
    missing: list[str] = []
    corrupt: list[dict[str, str]] = []

    for entry in manifest.files:
        candidate = base / entry.name
        if not candidate.is_file():
            matches = sorted(base.rglob(entry.name)) if base.is_dir() else []
            if not matches:
                missing.append(entry.name)
                continue
            candidate = matches[0]
        present.append(entry.name)
        if entry.sha256:
            actual = sha256_of(candidate)
            if actual.lower() != entry.sha256.lower():
                corrupt.append(
                    {"name": entry.name, "expected": entry.sha256.lower(), "actual": actual}
                )

    return {
        "dataset_id": manifest.dataset_id,
        "base_dir": str(base),
        "present": present,
        "missing": missing,
        "corrupt": corrupt,
        "ok": not missing and not corrupt,
    }


def load_all_manifests(directory: str | Path | None = None) -> dict[str, DatasetManifest]:
    """Load every manifest in the manifests directory.

    Args:
        directory: Override for ``data/manifests``.

    Returns:
        Mapping of dataset id to manifest.
    """
    directory = Path(directory) if directory is not None else DATA_MANIFESTS_DIR
    manifests: dict[str, DatasetManifest] = {}
    for path in sorted(directory.glob("*.yaml")):
        manifest = load_manifest(path)
        manifests[manifest.dataset_id] = manifest
    return manifests
