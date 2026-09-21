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
    awaiting_determination,
    determination_reason,
    disposition,
    inventory,
    is_hole,
    render,
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
    assert any(path.startswith("experiments.") for path in holes)
    # And the ones that have already diverged: the code resolves these elsewhere.
    assert "results.raw_dir" in holes


@pytest.mark.unit
def test_the_wired_families_are_no_longer_holes(records):
    """The fields this issue wired have readers, and stayed wired.

    Each of these was a HOLE before: nothing read the grid, nothing read the
    declared seed set. A regression here means the protocol section is
    decorative again.
    """
    by_path = {record["path"]: record for record in records}
    still_holes = sorted(
        path
        for path, record in by_path.items()
        if record["orphan"]
        and is_hole(record)
        and (
            path.startswith("hyperparameter_selection.grid")
            or path.startswith("seeds.")
            or path in ("hyperparameter_selection.scope", "hyperparameter_selection.selection_split")
            or path == "preprocessing.windowing.length_candidates"
        )
    )
    assert not still_holes, f"these went back to having no reader: {still_holes}"


@pytest.mark.unit
def test_the_retired_reader_rules_no_longer_certify_a_reader(records):
    """A declared value, a docstring and a comment are not readers.

    Each of these three rows was in the "read" bucket for a reason that was not
    a reader: a hardcoded constant that happens to equal the declared value, a
    value that appears in a docstring, and a value coincidence beside siblings
    that were correctly listed as gaps. The tightening has to hold, or the hole
    count stops being a lower bound.
    """
    by_path = {record["path"]: record for record in records}

    label_column = by_path["tasks.classification.label_column"]
    assert not label_column["readers"], (
        "tasks.classification.label_column is read again by a literal: the reader has to be a "
        "reader of the FIELD, not the same string written twice in schema.py"
    )

    common = by_path["datasets.channel_intersection.common_to_D1_D2_D3"]
    assert not common["readers"]
    assert is_hole(common), (
        "common_to_D1_D2_D3 is the third of its family; channel_count and measured_on are holes, "
        "so a value coincidence in synthetic.py must not keep it out of the same bucket"
    )

    evaluated_on = by_path["gates.Gate_C.evaluated_on"]
    assert not evaluated_on["readers"], (
        "gates.Gate_C.evaluated_on was certified by the `D2` inside a docstring"
    )


@pytest.mark.unit
def test_rows_awaiting_determination_name_what_matched(records):
    """The demoted rows are adjudicable, not merely demoted.

    A row the tightened test cannot certify goes to neither bucket: it is not a
    reader, and calling it a hole would be a determination nobody made. What
    makes it actionable is that the retired test's match is printed beside it.
    """
    awaiting = [record for record in records if record["orphan"] and awaiting_determination(record)]
    assert awaiting, (
        "if no row awaits determination, delete this test rather than the check — the tightened "
        "rule is then certifying everything it used to"
    )
    for record in awaiting:
        assert not record["readers"], f"{record['path']} has a reader and is not awaiting anything"
        reason = determination_reason(record)
        assert reason.startswith("awaiting determination")
        assert "`" in reason, f"{record['path']} names no location to adjudicate"
        assert not is_hole(record), f"{record['path']} is a hole and has a disposition already"

    # The rule that replaced the retired one has to be recorded, not implied.
    assert any(
        record["path"] == "tasks.classification.label_column" for record in awaiting
    ), "the hardcoded-constant case is the one the tightening exists for"


@pytest.mark.unit
def test_the_document_is_committed_and_current(records):
    """A stale inventory is worse than none: it reports readers that have moved."""
    assert DOC_PATH.is_file(), f"{DOC_PATH} must be committed"
    committed = DOC_PATH.read_text(encoding="utf-8")
    assert "Generated by `python scripts/protocol_field_readers.py --write`" in committed
    for record in records:
        assert f"`{record['path']}`" in committed, f"{record['path']} is missing from the document"


@pytest.mark.unit
def test_the_known_limitations_persist():
    """The DATA-23 blocks survived into the generator.

    DATA-23 was registered with two pieces hand-written into the document at
    commit ``79751e2`` (a long ``known inconsistency`` disposition for
    ``gates.Gate_C.evaluated_on`` and a ``## Known limitations`` section).
    The next regeneration at ``9947b6e`` silently deleted both pieces. The
    fix is to make the generator own them, so any future ``--write`` cannot
    drop them silently. This test asserts they are present on disk.
    """
    committed = DOC_PATH.read_text(encoding="utf-8")

    # The Known limitations section must exist with all three subsections.
    assert "## Known limitations (carried forward, not fixed in this freeze)" in committed, (
        "the `## Known limitations` section disappeared — it lives in "
        "scripts/protocol_field_readers.py:_known_limitations_section so a "
        "`--write` cannot drop it silently"
    )
    assert "### Bucket math is self-consistent" in committed, (
        "the `Bucket math` subsection disappeared from `## Known limitations`"
    )
    assert "### r2 direction is task-scoped" in committed, (
        "the `r2 direction` subsection disappeared from `## Known limitations` — "
        "r2 inheriting `minimize` from the regression task is a documented "
        "DATA-23 pre-flight item that must stay on the record"
    )
    assert "### Gate_C evaluated_on vs expression" in committed, (
        "the `Gate_C evaluated_on vs expression` subsection disappeared from "
        "`## Known limitations` — the D2 mismatch is a documented DATA-23 "
        "pre-flight item that must stay on the record"
    )

    # The Gate_C disposition must carry the long known-inconsistency text.
    gate_c_disposition = disposition(
        {
            "path": "gates.Gate_C.evaluated_on",
            "key": "evaluated_on",
            "value": ["D3", "E3", "E4", "E5"],
            "readers": [],
            "prose_hits": [],
            "value_hits": [],
            "known_reader": None,
            "orphan": True,
            "prose": False,
        }
    )
    assert gate_c_disposition.startswith("known inconsistency"), (
        f"gates.Gate_C.evaluated_on disposition regressed to "
        f"{gate_c_disposition!r}; the D2 mismatch must remain on the record"
    )
    assert "sig_any(..., [D2, D3], ...)" in gate_c_disposition, (
        "the disposition must name the offending expression so a reader does "
        "not have to re-derive it"
    )
    assert "Known limitations" in gate_c_disposition, (
        "the disposition must point at the section that holds the full reading"
    )


@pytest.mark.unit
def test_the_render_output_carries_known_limitations(records):
    """The generator's own output contains the section — no hand-edit path.

    Asserting this on ``render(records)`` rather than on the committed file
    guards the second failure mode: a maintainer who hand-edits the doc
    back to a clean state still gets caught, because the next ``--write``
    would emit the section and the diff would re-introduce it.
    """
    text = render(records)
    assert "## Known limitations" in text
    assert "### Bucket math is self-consistent" in text
    assert "### r2 direction is task-scoped" in text
    assert "### Gate_C evaluated_on vs expression" in text
    assert "known inconsistency" in text
    # The bucket-math sums are computed live, so the rendered sums must match
    # what the inventory itself reports — guards against the section drifting
    # from the table it summarises.
    orphans = [r for r in records if r["orphan"]]
    holes = [r for r in orphans if is_hole(r)]
    awaiting = [r for r in orphans if awaiting_determination(r)]
    prose = [r for r in records if r["prose"]]
    read = [r for r in records if r["readers"]]
    named = [r for r in records if not r["prose"] and not r["readers"] and r["known_reader"]]
    statement_orphans = [r for r in orphans if r not in holes and r not in awaiting]
    assert (
        f"`{len(records)} = {len(read)} + {len(named)} + {len(prose)} + {len(orphans)}`"
        in text
    ), "the top-line bucket sum is missing from the Known limitations section"
    assert (
        f"`{len(orphans)} = {len(statement_orphans)} + {len(awaiting)} + {len(holes)}`"
        in text
    ), "the orphan-bucket sum is missing from the Known limitations section"
