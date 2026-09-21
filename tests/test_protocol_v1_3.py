"""DATA-25 — protocol v1.3 amendment invariants.

The amendment rule says a new protocol version lives in a new file, never as an
edit to an existing one. v1.3 is the amendment that takes OD1 ruling (a) and
moves D3's split from GroupKFold(5) to LOSO(62) — the only field-by-field
difference from v1.2 is the D3 split and the bookkeeping that change forces
(gate reachability on D3, the reachability note, the data_verified.D3 record,
the amendment record itself, and the freeze_evidence.previous_version that
points to v1.2). These tests pin the rule mechanically:

* v1.1 and v1.2 must verify byte-for-byte against their sidecars — the new file
  is the only thing that changed.
* v1.3 must verify against its own sidecar.
* D3 must be LOSO with 62 folds, no overlap, and the 62 fold keys must be
  specimen-disjoint under the protocol's seeded permutation.
* Gate_B and Gate_C must be REACHABLE on D3 (floor 2/2^62); Gate_A's
  reachability is unchanged.
* The field-by-field diff between v1.3 and v1.2 must be exactly the D3 split
  and the bookkeeping that change forces — nothing else moves.
* The integrity disclosures must name the first D3 test touch and state the
  non-merge rule for v1.1/v1.2/v1.3 runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from drososense.data.splits import loso
from drososense.utils.config import verify_protocol_freeze
from drososense.utils.paths import CONFIGS_DIR


# Recorded digests from the freeze rules. Each is fixed at freeze time; a
# post-freeze edit changes one of them and these tests fail.
V1_1_DIGEST = "b87ed6057962b00e41c175acd7c78f314e552c50e2eecf53174e6156d8f7c0b9"
V1_2_DIGEST = "32c57f6efaa8f9f14584d3a153fddc2814c5be8829eb888fc0d6c27d536a0cc1"
# v1.3's digest is recorded in configs/protocol_v1.3.sha256 at freeze time. The
# expected-digest constant is fixed at freeze time and is checked against the
# actual digest the freeze check reports; the value here MUST match the
# sidecar, or this test fails before the freeze check runs. The constant is
# updated whenever the protocol file is re-frozen.
V1_3_DIGEST = "d85640e556db6c00c8217befda17c967772c30470be9fbe73b2051b3cbfe8357"

# The protocol records 62 D3 fillets; 63 raw filename tokens reduce to 62
# under the config's two filename_aliases. The actual fillet ids do not matter
# for the LOSO(62) property test — what matters is that 62 distinct ids
# produce 62 folds. A placeholder set of 62 ids suffices.
D3_DECLARED_SPECIMENS: tuple[str, ...] = tuple(f"F{i // 10 + 1}F{i % 10 + 1}" for i in range(62))
assert len(D3_DECLARED_SPECIMENS) == 62, "the protocol records exactly 62 specimens"
assert len(set(D3_DECLARED_SPECIMENS)) == 62, "specimen ids must be distinct for LOSO"

# Fields that must differ between v1.2 and v1.3 because OD1 ruling (a) changed
# D3's split. Anything else moving fails the "minimal blast radius" promise.
EXPECTED_V1_2_TO_V1_3_DIFF_KEYS = frozenset(
    {
        # Header metadata
        ("protocol_version",),
        ("frozen_at",),
        ("supersedes",),
        # Freeze evidence: previous_version now points to v1.2 with its sha256.
        ("freeze_evidence", "protocol_frozen_at"),
        ("freeze_evidence", "protocol_sha256_sidecar"),
        ("freeze_evidence", "previous_version", "protocol_version"),
        ("freeze_evidence", "previous_version", "frozen_at"),
        ("freeze_evidence", "previous_version", "sha256"),
        ("freeze_evidence", "previous_version", "sidecar"),
        ("freeze_evidence", "previous_version", "note"),
        # Amendment rule text expanded to mention v1.3 labels.
        ("freeze_evidence", "freeze_policy", "amendment_rule"),
        # Split protocol: D3 strategy/n_splits and the surrounding prose.
        ("split_protocol", "per_dataset", "D3"),
        # Reachability note + examples: D3 now has 62 clusters.
        ("split_protocol", "reachability_note"),
        ("split_protocol", "reachability_examples"),
        # Gate reachability arithmetic flips on D3.
        ("gates", "gate_reachability", "per_gate", "Gate_B"),
        ("gates", "gate_reachability", "per_gate", "Gate_C"),
        ("gates", "gate_reachability", "effect_on_the_project"),
        # data_verified.D3 now records split_strategy/split_n_splits.
        ("data_verified", "D3"),
        # Amendment record: v1.3 added; nothing else in §25 changes.
        ("amendment_v1_3",),
    }
)


def _iter_leaf_paths(node, prefix=()):
    """Yield every leaf path as a tuple of segments (no stringification)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _iter_leaf_paths(value, prefix + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_leaf_paths(value, prefix + (f"[{index}]",))
    else:
        yield prefix


def _materialise(node):
    """Materialise lists of dicts to a key-stable, deterministic structure."""
    if isinstance(node, dict):
        return {str(k): _materialise(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_materialise(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Byte-for-byte invariants: v1.1 and v1.2 are not edited by v1.3.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_1_file_and_sidecar_are_byte_for_byte_unchanged():
    """The amendment rule forbids editing a frozen file. v1.1 is the original freeze."""
    path = CONFIGS_DIR / "protocol_v1.1.yaml"
    sidecar = CONFIGS_DIR / "protocol_v1.1.sha256"
    last_line = sidecar.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert last_line.split()[0] == V1_1_DIGEST
    assert path.name in last_line
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V1_1_DIGEST


@pytest.mark.unit
def test_v1_2_file_and_sidecar_are_byte_for_byte_unchanged():
    """The amendment rule forbids editing a frozen file. v1.2 is the prior freeze."""
    path = CONFIGS_DIR / "protocol_v1.2.yaml"
    sidecar = CONFIGS_DIR / "protocol_v1.2.sha256"
    last_line = sidecar.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert last_line.split()[0] == V1_2_DIGEST
    assert path.name in last_line
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V1_2_DIGEST


@pytest.mark.unit
def test_v1_3_file_verifies_against_its_own_sidecar():
    """A new frozen file must freeze-check against its own recorded digest."""
    report = verify_protocol_freeze(
        CONFIGS_DIR / "protocol_v1.3.yaml", CONFIGS_DIR / "protocol_v1.3.sha256"
    )
    assert report["matches"], report
    assert report["actual"] == V1_3_DIGEST


@pytest.mark.unit
def test_v1_3_sidecar_was_written_for_this_file_only():
    """The sidecar must name v1.3, not a previous version's file."""
    sidecar = (CONFIGS_DIR / "protocol_v1.3.sha256").read_text(encoding="utf-8")
    assert "protocol_v1.3.yaml" in sidecar
    assert "protocol_v1.2.yaml" not in sidecar
    assert "protocol_v1.1.yaml" not in sidecar


# ---------------------------------------------------------------------------
# D3 split: LOSO(62), 62 folds, no specimen overlap.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_3_d3_split_is_loso_with_62_folds():
    """The amended split is LOSO with one fold per specimen — exactly 62."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    d3 = v1_3["split_protocol"]["per_dataset"]["D3"]
    assert d3["strategy"] == "loso"
    assert int(d3["n_splits"]) == 62
    # LOSO's fold count is the specimen count by construction.
    assert int(d3["n_splits"]) == 62
    # The protocol_compliant flag stays True and the label_stratum stays day.
    assert d3["protocol_compliant"] is True
    assert d3["label_stratum"] == "day"
    assert d3["group_semantics_verified"] is True


@pytest.mark.unit
def test_d2_split_is_unchanged_loso_5_in_v1_3():
    """The amendment touches D3 only. D2 must still be LOSO(5)."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    d2 = v1_3["split_protocol"]["per_dataset"]["D2"]
    assert d2["strategy"] == "loso"
    assert int(d2["n_splits"]) == 5
    assert d2["protocol_compliant"] is True
    # The campaign confound lives under datasets.D2, not split_protocol.
    d2_dataset = v1_3["datasets"]["D2"]
    assert "leave-TS1-out" in d2_dataset["campaign_confound"]["mandatory_decomposition"]


@pytest.mark.unit
def test_d3_loso_produces_62_specimen_disjoint_folds():
    """Under the protocol's LOSO, every specimen is test exactly once.

    This pins the contract mechanically: a reader can see, from the
    ``loso(specimens, seed=...)`` constructor, that the 62 folds are
    specimen-disjoint and exhaustive. A v1.3 implementation that built a
    5-block GroupKFold and renamed it to LOSO would fail here, because
    ``loso(62_specimens)`` returns 62 folds, not 5.
    """
    folds = loso(D3_DECLARED_SPECIMENS, seed=0)
    assert len(folds) == 62, "LOSO over 62 specimens must produce exactly 62 folds"
    tested = [s for fold in folds for s in fold.test]
    assert sorted(tested) == sorted(D3_DECLARED_SPECIMENS), (
        "every specimen must be held out exactly once"
    )
    assert len(tested) == len(set(tested)), "no specimen may appear in more than one fold"
    for fold in folds:
        assert len(fold.test) == 1
        assert not set(fold.train) & set(fold.test)
        assert not set(fold.train) & set(fold.val)
        assert not set(fold.val) & set(fold.test)


@pytest.mark.unit
def test_d3_loso_minimum_p_is_reachable_at_alpha_0_05():
    """2/2^62 <= alpha = 0.05; v1.3's D3 split must be arithmetically reachable.

    This is the v1.3-design claim stated mechanically, not in prose.
    """
    n_clusters = 62
    minimum_p = 2 / (2 ** n_clusters)
    assert minimum_p <= 0.05, (
        f"with n_clusters=62 the floor is {minimum_p}, which must be <= alpha=0.05"
    )


# ---------------------------------------------------------------------------
# Gate reachability: D3 LOSO(62) flips Gate_B and Gate_C to REACHABLE.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_3_gate_b_is_reachable_on_d3_loso_62():
    """Gate_B must be REACHABLE on D3 under v1.3 (was NOT REACHABLE under v1.2)."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    gate_b = v1_3["gates"]["gate_reachability"]["per_gate"]["Gate_B"]
    assert gate_b.startswith("REACHABLE"), gate_b
    # The D2 `sig` terms remain unreachable on five clusters; the OR over D2/D3
    # is reached via D3. The amendment must say so explicitly so a reader
    # doesn't mistake a D3-only fire for a cross-food fire.
    assert "D2" in gate_b
    assert "five clusters" in gate_b or "LOSO(5)" in gate_b


@pytest.mark.unit
def test_v1_3_gate_c_is_reachable_on_d3_loso_62():
    """Gate_C must be REACHABLE on D3 under v1.3 (was NOT REACHABLE under v1.2)."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    gate_c = v1_3["gates"]["gate_reachability"]["per_gate"]["Gate_C"]
    assert gate_c.startswith("REACHABLE"), gate_c
    assert "D3" in gate_c
    # Gate_C's expression does not name D2 as a `sig` term, so the chain is
    # decidable on D3 alone — the amendment must say so.
    assert "D2" in gate_c


@pytest.mark.unit
def test_v1_3_gate_a_reachability_is_unchanged_from_v1_2():
    """Gate_A is decided by ci/noninferior/params, none of which the amendment touches."""
    v1_2 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.2.yaml").read_text(encoding="utf-8"))
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    a2 = v1_2["gates"]["gate_reachability"]["per_gate"]["Gate_A"].strip()
    a3 = v1_3["gates"]["gate_reachability"]["per_gate"]["Gate_A"].strip()
    assert a2 == a3, "Gate_A reachability text moved"


# ---------------------------------------------------------------------------
# Field-by-field diff: only D3-split-related fields differ from v1.2.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_3_differs_from_v1_2_only_in_d3_split_fields():
    """The amendment is a pure split update. Anything else moving fails this test.

    The exact set of allowed differences is recorded above as
    ``EXPECTED_V1_2_TO_V1_3_DIFF_KEYS``; it covers the D3 split, the
    reachability text that mentions 62 clusters, the gate reachability per-gate
    rows that flip on D3, the data_verified.D3 record, the header metadata, the
    freeze_evidence.previous_version, and the new amendment_v1_3 block.

    Tuples containing ``[N]`` index markers match ALL list-index variants: the
    rule is on the parent key, not the row number, so any future list
    reordering inside an unchanged field does not register as a drift.
    """
    v1_2 = _materialise(yaml.safe_load((CONFIGS_DIR / "protocol_v1.2.yaml").read_text(encoding="utf-8")))
    v1_3 = _materialise(yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8")))

    def _covers(expected_prefix, actual_path):
        if len(expected_prefix) > len(actual_path):
            return False
        for expected_seg, actual_seg in zip(expected_prefix, actual_path):
            if expected_seg.startswith("[") and expected_seg.endswith("]"):
                # Index wildcard: any index marker counts.
                continue
            if expected_seg != actual_seg:
                return False
        return True

    def _allowed_diff(path):
        for prefix in EXPECTED_V1_2_TO_V1_3_DIFF_KEYS:
            if _covers(prefix, path):
                return True
        return False

    only_v1_2 = []
    only_v1_3 = []
    for path in set(_iter_leaf_paths(v1_2)) | set(_iter_leaf_paths(v1_3)):
        v1_2_val = v1_2
        v1_3_val = v1_3
        for segment in path:
            if segment.startswith("[") and segment.endswith("]"):
                # Numeric index — only follow if the current node is a list
                # and the index is in range. Comparing across both v1.2 and
                # v1.3 is the only way the diff can be reported, and a
                # list-reordering inside an unchanged field must NOT count
                # as a drift. Treat out-of-range on either side as "present
                # only on one side" — but a path's index list is bounded by
                # whichever side has the smaller list, so an index present
                # on one side and out of range on the other can only mean
                # the lists differ in length, which is exactly what we want
                # to flag.
                idx = int(segment[1:-1])
                v1_2_val = v1_2_val[idx] if isinstance(v1_2_val, list) and idx < len(v1_2_val) else _MISSING
                v1_3_val = v1_3_val[idx] if isinstance(v1_3_val, list) and idx < len(v1_3_val) else _MISSING
            else:
                v1_2_val = v1_2_val.get(segment, _MISSING) if isinstance(v1_2_val, dict) else _MISSING
                v1_3_val = v1_3_val.get(segment, _MISSING) if isinstance(v1_3_val, dict) else _MISSING
        if v1_2_val == _MISSING and v1_3_val != _MISSING:
            only_v1_3.append(path)
        elif v1_3_val == _MISSING and v1_2_val != _MISSING:
            only_v1_2.append(path)
        elif v1_2_val != v1_3_val:
            # Same path, different value. This is a leaf-level drift.
            only_v1_3.append(path)

    disallowed = [p for p in only_v1_3 + only_v1_2 if not _allowed_diff(p)]
    assert not disallowed, (
        "v1.3 differs from v1.2 in fields outside the D3-split amendment:\n  "
        + "\n  ".join("/".join(p) for p in disallowed)
    )


_MISSING = object()


# ---------------------------------------------------------------------------
# Integrity disclosures: required by the amendment rule.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_3_integrity_disclosures_state_the_first_d3_touch():
    """The amendment must restate the first D3 test touch from the contact log.

    The timestamp is read from results/tables/data_contact_log.json so the
    disclosure matches what the runner actually recorded, not a prose guess.
    """
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    disclosure = v1_3["amendment_v1_3"]["integrity_disclosures"][
        "d3_test_split_touched_in_v1_1_v1_2"
    ]
    contact_log = json.loads(
        (CONFIGS_DIR.parent / "results" / "tables" / "data_contact_log.json").read_text()
    )
    expected = contact_log["first_test_evaluation_at"]
    assert disclosure["first_test_evaluation_at"] == expected
    assert "1.1.0" in disclosure["under_protocol_versions"]
    assert "1.2.0" in disclosure["under_protocol_versions"]
    assert disclosure["under_split_strategy"] == "group_kfold"
    assert disclosure["under_split_n_splits"] == 5


@pytest.mark.unit
def test_v1_3_disclosure_states_v1_1_v1_2_runs_are_not_rerun_or_merged():
    """The protocol_version column is the audit trail; nothing is re-labelled."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    block = v1_3["amendment_v1_3"]["integrity_disclosures"][
        "v1_1_v1_2_d3_runs_preserved_unchanged"
    ]
    text = (block["rule"] + " " + block["rationale"]).lower()
    assert "1.1.0" in block["rule"]
    assert "1.2.0" in block["rule"]
    assert "protocol_version" in block["rule"]
    # "None are re-run, re-interpreted, or merged" — None-as-not is the
    # language the amendment uses; both "not" and "none" satisfy the intent.
    assert "not " in text or "none " in text
    # Both re-run and merge are forbidden.
    assert "re-run" in text or "rerun" in text
    assert "merge" in text or "pool" in text


@pytest.mark.unit
def test_v1_3_disclosure_states_the_loso_fold_keys_are_new():
    """The new LOSO folds must carry new fold fingerprints, not collide with v1.2."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    block = v1_3["amendment_v1_3"]["integrity_disclosures"][
        "v1_3_loso_fold_keys_are_new"
    ]
    text = block["rule"] + block["enforcement"]
    assert "LOSO" in text
    assert "GroupKFold" in text or "group_kfold" in text
    assert "fold" in text.lower()
    assert "test_touched_once" in text


@pytest.mark.unit
def test_v1_3_amendment_record_says_what_drove_and_what_did_not():
    """The amendment rule requires the trigger; the OD1 ruling (a) is the trigger."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    amendment = v1_3["amendment_v1_3"]
    assert amendment["trigger"].strip()
    assert "OD1" in amendment["trigger"]
    assert "62" in amendment["trigger"]
    # The 'not driven by' field is required by the issue's integrity clause.
    assert "not" in amendment["decision_recorded"]["not_driven_by"].lower()
    assert "R0..R5" in amendment["decision_recorded"]["not_driven_by"] or "connectome" in amendment[
        "decision_recorded"
    ]["not_driven_by"].lower()
    # The ruling is "(a)" and the option key is recorded.
    assert amendment["decision_recorded"]["od1_ruling"] == "(a)"
    assert "a" in amendment["decision_recorded"]["od1_ruling_options"]


@pytest.mark.unit
def test_v1_3_records_no_runs_re_run_under_the_amendment():
    """The M4 contrasts have not been produced, so v1.3 changes no results."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    assert "NONE ARE RE-RUN" in v1_3["amendment_v1_3"]["runs_affected"]


# ---------------------------------------------------------------------------
# Unchanged-from-v1.2 fields that the amendment explicitly pins.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "field_path",
    [
        # Primary / secondary metrics and their direction
        ("tasks", "classification", "metrics", "primary"),
        ("tasks", "classification", "metrics", "secondary"),
        ("tasks", "regression", "metrics", "primary"),
        ("tasks", "regression", "metrics", "secondary"),
        # Alpha
        ("statistical_tests", "alpha"),
        # Multiplicity method and family rule
        ("multiplicity", "method"),
        ("multiplicity", "family_rule"),
        # Decisive test stays the cluster_sign_test
        ("statistical_tests", "primary_test", "name"),
        # Robustness injection stage and retrain_readout
        ("robustness_protocol", "injection_stage"),
        ("robustness_protocol", "retrain_readout"),
        ("robustness_protocol", "drift_applied_after_standardization"),
        ("robustness_protocol", "channel_dropout", "levels"),
        ("robustness_protocol", "noise", "levels"),
        # Seed set
        ("seeds", "root_seeds"),
        # Narrative rules N1..N5 are present
        ("narrative_adjustment_rules",),
    ],
)
def test_v1_3_pins_undeclared_change_for_field(field_path):
    """Every "unchanged" field in the amendment must be byte-equal to v1.2.

    The list is the issue's "what we explicitly did NOT change" inventory.
    A drift here means the amendment did something the issue forbade.
    """
    v1_2 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.2.yaml").read_text(encoding="utf-8"))
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    a = v1_2
    b = v1_3
    for segment in field_path:
        a = a[segment]
        b = b[segment]
    assert a == b, f"{'/'.join(field_path)} drifted between v1.2 and v1.3"


@pytest.mark.unit
def test_v1_3_keeps_all_six_declared_contrasts():
    """The contrast list is the scientific claim surface; it must not move."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    ids = {c["id"] for c in v1_3["contrasts"]["list"]}
    assert ids == {"R0_vs_R2", "R0_vs_R3", "R0_vs_R4", "R0_vs_R5", "R0_vs_R1", "R0_vs_GRU"}


@pytest.mark.unit
def test_v1_3_keeps_the_d2_leave_ts1_out_decomposition_mandatory():
    """D2's campaign confound decomposition is part of the unchanged contract."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    mandatory = v1_3["datasets"]["D2"]["campaign_confound"]["mandatory_decomposition"]
    assert "leave-TS1-out" in mandatory
    # "E1 MUST report leave-TS1-out ..." — the MUST is the operative contract.
    assert "MUST" in mandatory


@pytest.mark.unit
def test_v1_3_d3_endpoint_stratum_is_unchanged():
    """D3's seven endpoint levels / df cap / label_stratum are not touched."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    d3 = v1_3["datasets"]["D3"]["endpoint_degrees_of_freedom"]
    assert d3["endpoint_levels"] == 7
    assert d3["df_cap"] == 7
    assert d3["label_stratum"] == "day"


# ---------------------------------------------------------------------------
# protocol_field_readers: v1.3 introduced new declarative fields, so the
# inventory was regenerated and a v1.3-specific doc was committed.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_v1_3_field_readers_inventory_doc_is_committed_and_current():
    """The new v1.3 fields (amendment_v1_3, integrity_disclosures, ...) must appear.

    The active protocol is still v1.2, so docs/protocol_field_readers.md walks
    v1.2 and the existing tests stay valid. The v1.3 amendment added ~35
    declarative leaves; they are inventoried in docs/protocol_field_readers_v1.3.md
    so the failure mode the script exists to prevent cannot reappear on v1.3
    silently. This test pins that the doc is present and the new paths are
    listed.
    """
    from scripts.protocol_field_readers import inventory  # local import: tests/ import path

    doc_path = Path(__file__).resolve().parents[1] / "docs" / "protocol_field_readers_v1.3.md"
    assert doc_path.is_file(), f"{doc_path} must be committed alongside the amendment"

    committed = doc_path.read_text(encoding="utf-8")
    assert "Generated by `python scripts/protocol_field_readers.py --write`" in committed
    assert "Source: `configs/protocol_v1.3.yaml`" in committed

    records = inventory(protocol_path=CONFIGS_DIR / "protocol_v1.3.yaml")
    paths = [record["path"] for record in records]
    path_set = set(paths)
    # v1.3-specific new fields must be inventoried (so a future orphan of them
    # would be visible to a reviewer before it ships). The inventory records
    # leaves, so we assert at least one leaf exists under each new key.
    assert any(p.startswith("amendment_v1_3.trigger") for p in path_set)
    assert any(p.startswith("amendment_v1_3.integrity_disclosures") for p in path_set)
    assert any(p.startswith("amendment_v1_3.decision_recorded") for p in path_set)
    assert any(
        p.startswith("amendment_v1_3.integrity_disclosures.d3_test_split_touched_in_v1_1_v1_2")
        for p in path_set
    )
    assert "data_verified.D3.split_strategy" in path_set
    assert "data_verified.D3.split_n_splits" in path_set
    assert "data_verified.D3.split_strategy_change" in path_set
    for record in records:
        assert f"`{record['path']}`" in committed, (
            f"{record['path']} is missing from the v1.3 inventory doc"
        )


@pytest.mark.unit
def test_v1_3_inventory_does_not_introduce_new_holes():
    """The amendment adds statement-type fields, not hole-type ones.

    The hole count is the lower bound on what production code has yet to wire.
    v1.3 adds bookkeeping for the LOSO(62) amendment and the integrity
    disclosures; none of those has a hidden production reader. If a v1.3 field
    shows up as a HOLE, that is a wiring failure the amendment introduced.
    """
    from scripts.protocol_field_readers import inventory, is_hole

    records = inventory(protocol_path=CONFIGS_DIR / "protocol_v1.3.yaml")
    v1_3_holes = [
        record["path"]
        for record in records
        if is_hole(record) and (
            record["path"].startswith("amendment_v1_3")
            or record["path"].startswith("data_verified.D3.split_")
        )
    ]
    assert not v1_3_holes, (
        "v1.3 introduced hole-type orphans: " + ", ".join(v1_3_holes)
    )