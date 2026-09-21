"""The data-contact log: when a test split was first touched, and on what.

Protocol v1.1 freezes with ``data_contact_log.first_test_evaluation_at: null``.
That value cannot live in the protocol itself, because recording the first
contact would edit a frozen file and invalidate its digest — the freeze rule
would then be unenforceable exactly when it starts to matter. So the protocol
declares the rule and this module keeps the live record:

    results/tables/data_contact_log.json

The runner appends to it whenever it evaluates a model on a test split, so the
claim "the protocol was fixed before any test result existed" can be checked
against a file the experiment itself wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from drososense.utils.paths import RESULTS_TABLES_DIR, ensure_dir

CONTACT_LOG_NAME = "data_contact_log.json"

# Splits that do NOT count as touching a test set for the purpose of the freeze.
# The synthetic fixture is generated locally and the time-block stand-in for D1
# is not a specimen-level split; neither can produce a protocol result, so
# neither may be used to claim "the frozen metrics were already known".
NON_COUNTING_SPLIT_STRATEGIES = ("time_block_holdout",)
NON_COUNTING_EVIDENCE_CLASSES = ("synthetic_fixture",)


@dataclass(frozen=True)
class ContactEntry:
    """One recorded test evaluation.

    Attributes:
        at: UTC timestamp of the run.
        experiment: Experiment label.
        dataset: Dataset identifier.
        split_strategy: Strategy the fold came from.
        protocol_compliant: Whether the split satisfied ``split_unit: specimen``.
        evidence_class: ``real`` or ``synthetic_fixture``.
        n_models: How many model/task evaluations were recorded.
        counts_as_first_test_evaluation: Whether this run is the first contact
            the freeze rule cares about.
    """

    at: str
    experiment: str
    dataset: str
    split_strategy: str
    protocol_compliant: bool
    evidence_class: str
    n_models: int
    counts_as_first_test_evaluation: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Mapping of every field.
        """
        return {
            "at": self.at,
            "experiment": self.experiment,
            "dataset": self.dataset,
            "split_strategy": self.split_strategy,
            "protocol_compliant": self.protocol_compliant,
            "evidence_class": self.evidence_class,
            "n_models": self.n_models,
            "counts_as_first_test_evaluation": self.counts_as_first_test_evaluation,
        }


@dataclass(frozen=True)
class ContactLog:
    """The accumulated record of test-set contact.

    Attributes:
        first_test_evaluation_at: First contact that counts, or ``None``.
        datasets_touched: Datasets evaluated in counting runs, sorted.
        entries: Every recorded run, in the order it happened.
        note: What does and does not count, restated for a reader.
    """

    first_test_evaluation_at: str | None = None
    datasets_touched: tuple[str, ...] = ()
    entries: tuple[ContactEntry, ...] = ()
    note: str = field(default="")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Mapping of the log.
        """
        return {
            "first_test_evaluation_at": self.first_test_evaluation_at,
            "datasets_touched": list(self.datasets_touched),
            "note": self.note,
            "entries": [e.as_dict() for e in self.entries],
        }

    @property
    def started(self) -> bool:
        """Whether a counting test evaluation has happened."""
        return self.first_test_evaluation_at is not None


def contact_log_path(base_dir: str | Path | None = None) -> Path:
    """Resolve the contact log's location.

    Args:
        base_dir: Override for ``results/tables``.

    Returns:
        Path to the log file.
    """
    base = Path(base_dir) if base_dir is not None else RESULTS_TABLES_DIR
    return base / CONTACT_LOG_NAME


def load_contact_log(base_dir: str | Path | None = None) -> ContactLog:
    """Read the contact log, returning an empty one when absent.

    Args:
        base_dir: Override for ``results/tables``.

    Returns:
        The :class:`ContactLog`.
    """
    path = contact_log_path(base_dir)
    if not path.is_file():
        return ContactLog(note=_NOTE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ContactLog(
        first_test_evaluation_at=payload.get("first_test_evaluation_at"),
        datasets_touched=tuple(payload.get("datasets_touched", ())),
        entries=tuple(
            ContactEntry(**{k: v for k, v in entry.items() if k in ContactEntry.__dataclass_fields__})
            for entry in payload.get("entries", [])
        ),
        note=payload.get("note", _NOTE),
    )


def counts_as_contact(split_strategy: str, protocol_compliant: bool, evidence_class: str) -> bool:
    """Decide whether a run counts as touching a test set.

    Args:
        split_strategy: The fold strategy used.
        protocol_compliant: Whether the split satisfied ``split_unit: specimen``.
        evidence_class: ``real`` or ``synthetic_fixture``.

    Returns:
        ``True`` when the run must be recorded as test-set contact.
    """
    if evidence_class in NON_COUNTING_EVIDENCE_CLASSES:
        return False
    if split_strategy in NON_COUNTING_SPLIT_STRATEGIES:
        return False
    return bool(protocol_compliant)


def record_contact(
    experiment: str,
    dataset: str,
    split_strategy: str,
    protocol_compliant: bool,
    evidence_class: str,
    n_models: int,
    at: str | None = None,
    base_dir: str | Path | None = None,
) -> ContactLog:
    """Append a run to the contact log and return the updated log.

    Args:
        experiment: Experiment label.
        dataset: Dataset identifier.
        split_strategy: The fold strategy used.
        protocol_compliant: Whether the split satisfied ``split_unit: specimen``.
        evidence_class: ``real`` or ``synthetic_fixture``.
        n_models: Number of model/task evaluations in this run.
        at: Timestamp override; defaults to now.
        base_dir: Override for ``results/tables``.

    Returns:
        The updated :class:`ContactLog`.
    """
    existing = load_contact_log(base_dir)
    timestamp = at or datetime.now(timezone.utc).isoformat()
    counting = counts_as_contact(split_strategy, protocol_compliant, evidence_class)
    entry = ContactEntry(
        at=timestamp,
        experiment=experiment,
        dataset=dataset,
        split_strategy=split_strategy,
        protocol_compliant=bool(protocol_compliant),
        evidence_class=evidence_class,
        n_models=int(n_models),
        counts_as_first_test_evaluation=counting,
    )
    entries: Iterable[ContactEntry] = (*existing.entries, entry)
    touched = sorted(
        {e.dataset for e in entries if e.counts_as_first_test_evaluation}
    )
    first = existing.first_test_evaluation_at
    if counting and first is None:
        first = timestamp
    updated = ContactLog(
        first_test_evaluation_at=first,
        datasets_touched=tuple(touched),
        entries=tuple(entries),
        note=_NOTE,
    )
    path = contact_log_path(base_dir)
    ensure_dir(path.parent)
    path.write_text(json.dumps(updated.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return updated


_NOTE = (
    "Records when a test split was first evaluated. Runs on the synthetic fixture and runs "
    "under a time-block stand-in split do not count: neither can produce a protocol result, so "
    "neither may be used to claim the frozen metrics were already known. "
    "Protocol v1.1 freeze_evidence.data_contact_log states the rule; this file is the evidence."
)


__all__ = [
    "CONTACT_LOG_NAME",
    "ContactEntry",
    "ContactLog",
    "contact_log_path",
    "counts_as_contact",
    "load_contact_log",
    "record_contact",
]
