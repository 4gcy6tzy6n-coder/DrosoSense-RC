"""Every declarative protocol field has a reader, or a recorded reason it has none.

"Declared but not wired" reached the project three times — the gate symbol
namespace, the model parameter table behind ``params(...)``, and
``multiplicity.families``. Each was caught by a reviewer, not by the code. The
first two were found by the author and the third was not, which is the point:
finding them one at a time does not scale.

`scripts/protocol_field_readers.py` walks the protocol and reports, for every
declarative leaf, whether the production path mentions it. These tests make that
report a gate: a new protocol field with no reader and no recorded disposition
fails the suite, so the next one is caught by CI rather than by a review round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol_field_readers import (  # noqa: E402
    DOC_PATH,
    disposition,
    inventory,
    is_hole,
)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return inventory()


@pytest.mark.unit
def test_the_inventory_covers_every_declared_leaf_once(records):
    """A field that is not walked cannot be triaged."""
    assert len(records) > 400, "the protocol carries far more fields than this"
    paths = [record["path"] for record in records]
    assert len(paths) == len(set(paths)), "a path is reported twice"
    assert any(record["path"].startswith("multiplicity.families") for record in records)


@pytest.mark.unit
def test_no_orphaned_field_is_left_untriaged(records):
    """The standing check: no protocol field may be both unread and unaccounted for.

    An orphan is fine — a claim guard, a provenance record or a hypothesis
    statement is written for a human. What is not fine is an orphan nobody has
    looked at, because that is exactly what ``multiplicity.families`` was.
    """
    untriaged = [
        record["path"]
        for record in records
        if record["orphan"] and "NOT TRIAGED" in disposition(record)
    ]
    assert not untriaged, (
        "these protocol fields are declared, read by nothing, and have no recorded "
        "disposition. Either wire them, test them, or record why they are declarative:\n  "
        + "\n  ".join(untriaged)
    )


@pytest.mark.unit
def test_the_fields_three_review_rounds_wired_have_readers(records):
    """The regressions this inventory exists to prevent, pinned by name.

    Each of these was declared, unread, and found by a review rather than by the
    code. If one of them ever slides back to "no reader", this fails.

    Two of them — the family membership and the effect-size binding — now have a
    production reader. The other two are read by a test on purpose: the protocol
    says the seed-invariance property is a checkable rule, and the test is the
    checker. The assertion is therefore "not a hole", with the reader pinned
    where one exists.
    """
    by_path = {record["path"]: record for record in records}
    for path in (
        "multiplicity.families[0].tests",
        "multiplicity.families[3].conditions",  # F_lowdata: the first family with conditions
        "statistical_tests.effect_size.classification",
        "statistical_tests.effect_size.regression",
        "statistical_tests.bootstrap.seed_invariance_required",
        "split_protocol.per_dataset.D2.group_semantics_verified",
    ):
        assert path in by_path, f"{path} disappeared from the protocol"
        assert not is_hole(by_path[path]), f"{path} is a hole again"

    # The family membership rule is read by the production path, not only by a test.
    assert by_path["multiplicity.families[0].tests"]["readers"], (
        "multiplicity.families[0].tests has no production reader: the correction would be "
        "back to grouping by (metric, dataset)"
    )
    assert by_path["statistical_tests.effect_size.classification"]["readers"]
    assert by_path["statistical_tests.effect_size.regression"]["readers"]


@pytest.mark.unit
def test_the_holes_are_named_and_counted(records):
    """The real holes are listed rather than buried among the statements.

    A ``HOLE`` is a field nothing reads whose reader would have to exist before
    the thing it declares could be relied on — an M4 dependency, or a
    declaration that has silently diverged from the code.
    """
    holes = sorted(record["path"] for record in records if record["orphan"] and is_hole(record))
    assert holes, "if there are no holes, delete this test rather than the check"

    # Each hole must be a *declarative* field, never a sentence.
    for path in holes:
        record = next(r for r in records if r["path"] == path)
        assert not record["prose"], f"{path} is prose and cannot be a hole"

    # The ones that are dependencies on work not yet started, declared as such.
    assert any(path.startswith("robustness_protocol") for path in holes)
    assert any(path.startswith("hyperparameter_selection.grid") for path in holes)
    # And the ones that have already diverged: the code resolves these elsewhere.
    assert "results.raw_dir" in holes


@pytest.mark.unit
def test_the_document_is_committed_and_current(records):
    """A stale inventory is worse than none: it reports readers that have moved."""
    assert DOC_PATH.is_file(), f"{DOC_PATH} must be committed"
    committed = DOC_PATH.read_text(encoding="utf-8")
    assert "Generated by `python scripts/protocol_field_readers.py --write`" in committed
    for record in records:
        assert f"`{record['path']}`" in committed, f"{record['path']} is missing from the document"
