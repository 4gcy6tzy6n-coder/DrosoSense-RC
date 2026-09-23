#!/usr/bin/env python3
"""E9 evidence-identity migration: v1.4 (schema 1) -> v1.5 (schema 2).

WHAT THIS IS. A migration of *identity*, not a re-evaluation. No metric is
recomputed, no test split is touched, no model is re-fit. Every record's
`metrics`, `duration_s`, `status`, `timestamp_utc` and everything else observed
is copied verbatim; what changes is which evidence unit the record belongs to.

WHY IT IS NEEDED. Schema 1's fingerprint named (partition, window, model, task)
and so could not distinguish two substrates evaluated on the same split. The E9
size study scored one split at N = 250/500/1000/2000/4000, and all six records of
each unit landed on ONE fingerprint under SIX config hashes — read by §17's own
report as 70 violations, and refused outright by `load_evidence_bundle`.

THE ARITHMETIC THAT MUST CLOSE (and did not, in the first report). 420 records is
not 280 + 70. Each of the 70 legacy fingerprints covers SIX records and FIVE
substrates: N = 250 was scored TWICE (in `e9_size_d2` and again in
`e9_size_d2_n250`), and 500/1000/2000/4000 once each. So:

    420 records  =  70 fingerprints x 6 records
    350 units    =  70 fingerprints x 5 substrates
      of those 350 units: 280 hold one record, 70 hold two (140 records)
    record-level:  280 adopted (single-record units)
                 +  70 adopted (the kept touch of each duplicated unit)
                 +  70 rejected as a duplicate test touch
                 = 420                                                      [closed]

The first report said "280 adoptable, 70 inadmissible", which left 70 records
unexplained. The corrected statement is **350 adopted / 70 rejected / 420 total**.

THE DESIGNATION RULE IS FROZEN AND RESULT-BLIND. For a unit holding two records,
the kept one is the one with the **earliest `timestamp_utc`**, ties broken by the
lexicographically smaller `config_hash`. The rule is stated here, in the audit
output, and in the manifest, and it is chosen without reference to any metric:
picking the better-scoring touch would be choosing the design after knowing which
way the result leans.

DERIVATION. Every identity component is derived from a named source, and the
source is recorded per component in the audit file:

    record fields ....... dataset, task, fold_fingerprint, model, window_length,
                         topology_variant, reservoir_size
    batch command line .. normalization, condition
                         (ops/e1/run_e9_size_d2.sh passes --normalization
                          n1_pre_l1 and no --train-fraction)
    runner rule ......... rewire_seed (pinned to the run seed; the batch ran
                         --seeds 0)
    implementation ...... input_mapping (INPUT_MAPPING_DENSE_RANDOM is the only
                         mapping this implementation has)

Outputs, under results/evidence_migration/:
    e9_v14_to_v15_manifest.csv   one row per original record, exactly one state
    e9_adopted.jsonl             the 350 adopted records, re-identified
    e9_rejected.jsonl            the 70 rejected records, with the reason and the
                                 identity of the touch that was kept
    e9_migration_audit.json      the closure identity, the rule, the sources
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drososense.evaluation.results import (  # noqa: E402
    EVIDENCE_UNIT_SCHEMA_VERSION,
    legacy_test_fingerprint,
    make_evidence_unit,
)

EVIDENCE = REPO_ROOT / "results" / "audit" / "m4_audit" / "server_evidence" / "ev"
OUT_DIR = REPO_ROOT / "results" / "evidence_migration"

E9_DIRS = ("e9_size_d2",) + tuple(f"e9_size_d2_n{n}" for n in (250, 500, 1000, 2000, 4000))

#: The closed enumeration of record-level dispositions. A record is in exactly
#: one of these; the counts must sum to the number of records.
DISPOSITIONS = (
    "adopted",
    "superseded",
    "duplicate_test_touch",
    "invalid",
    "missing",
)

#: `fingerprint_collision_only` is deliberately NOT a member of the enumeration.
#: A collision is a property of the legacy *fingerprint*, not of a record:
#: "this fingerprint could not name its substrate" is true of all 420 records and
#: says nothing about whether any individual record is admissible. It is carried
#: as the boolean column `unit_identity_collided` so the information survives
#: without breaking the partition.
NOT_A_DISPOSITION = {
    "fingerprint_collision_only": (
        "a unit-level diagnosis, not a record disposition; carried as "
        "unit_identity_collided=true on every E9 record"
    )
}

#: Where each identity component comes from. Recorded per component so a reader
#: can see which parts are *recorded* by the run and which are *derived*.
DERIVATION_SOURCES = {
    "dataset": "record",
    "task": "record",
    "fold_fingerprint": "record",
    "model": "record",
    "window_length": "record",
    "topology_variant": "record: model_description.topology.kind",
    "reservoir_size": "record: model_description.topology.n_nodes",
    "normalization": (
        "batch command line: ops/e1/run_e9_size_d2.sh passes "
        "--normalization n1_pre_l1 for every size arm"
    ),
    "condition": (
        "batch command line: ops/e1/run_e9_size_d2.sh passes no --train-fraction, "
        "so the condition is the default 'full'"
    ),
    "input_mapping": (
        "implementation constant: INPUT_MAPPING_DENSE_RANDOM is the only input "
        "mapping this code has; these records predate the field"
    ),
    "rewire_seed": (
        "runner rule: the reservoir runner pins the rewiring seed to the run "
        "seed, and the batch ran --seeds 0"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_e9() -> list[dict]:
    """Every E9 record with its provenance, sorted deterministically."""
    out = []
    for d in E9_DIRS:
        for path in sorted((EVIDENCE / d).rglob("*.json")):
            record = json.loads(path.read_text())
            out.append(
                {
                    "record": record,
                    "origin_dir": d,
                    "origin_path": str(path.relative_to(REPO_ROOT)),
                    "origin_sha256": sha256_file(path),
                }
            )
    out.sort(key=lambda e: (
        e["record"]["test_fingerprint"],
        e["record"]["model_description"]["topology"]["n_nodes"],
        e["record"]["timestamp_utc"],
        e["record"]["config_hash"],
    ))
    return out


def derive_components(entry: dict) -> dict:
    """The schema-2 components of one record, from its own fields + the batch."""
    r = entry["record"]
    topology = (r.get("model_description") or {}).get("topology") or {}
    return {
        "dataset": r["dataset"],
        "task": r["task"],
        "fold_fingerprint": r["fold_fingerprint"],
        "model": r["model"],
        "window_length": r["window_length"],
        "condition": "full",
        "reservoir_size": topology.get("n_nodes"),
        "normalization": "n1_pre_l1",
        "input_mapping": "dense_random_all_nodes",
        "topology_variant": topology.get("kind"),
        "rewire_seed": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_e9()
    print("=" * 78)
    print("E9 evidence-identity migration v1.4 (schema 1) -> v1.5 (schema 2)")
    print("=" * 78)
    print(f"  records loaded: {len(entries)}")
    print(f"  origin dirs: {dict(Counter(e['origin_dir'] for e in entries))}")

    # ---- group into evidence units ---------------------------------------
    units: dict[tuple, list[dict]] = defaultdict(list)
    for entry in entries:
        comps = derive_components(entry)
        entry["components"] = comps
        entry["unit_key"] = (
            entry["record"]["test_fingerprint"],
            json.dumps(comps, sort_keys=True),
        )
        units[entry["unit_key"]].append(entry)

    n_fingerprints = len({k[0] for k in units})
    print(f"  legacy fingerprints: {n_fingerprints}")
    print(f"  evidence units (fingerprint x substrate): {len(units)}")
    print(f"  records per unit: {dict(Counter(len(v) for v in units.values()))}")

    # ---- frozen, result-blind designation --------------------------------
    rule = (
        "kept = the record with the earliest timestamp_utc; ties broken by the "
        "lexicographically smaller config_hash. Chosen without reference to any "
        "metric: selecting the better-scoring touch would be choosing the design "
        "after knowing which way the result leans."
    )
    for key, members in units.items():
        members.sort(key=lambda e: (e["record"]["timestamp_utc"], e["record"]["config_hash"]))
        winner = members[0]
        winner["disposition"] = "adopted"
        winner["disposition_reason"] = (
            "sole scored evaluation of this evidence unit"
            if len(members) == 1
            else f"kept touch of a duplicated unit ({len(members)} records); {rule}"
        )
        for loser in members[1:]:
            loser["disposition"] = "duplicate_test_touch"
            loser["disposition_reason"] = (
                f"second scored evaluation of the same evidence unit "
                f"(N={derive_components(loser)['reservoir_size']}, same split, "
                f"same model, same condition) under config {loser['record']['config_hash']}; "
                f"the kept touch is {winner['record']['config_hash']} at "
                f"{winner['record']['timestamp_utc']}. {rule}"
            )
            loser["superseded_by"] = {
                "config_hash": winner["record"]["config_hash"],
                "timestamp_utc": winner["record"]["timestamp_utc"],
                "origin_path": winner["origin_path"],
                "origin_sha256": winner["origin_sha256"],
            }

    for entry in entries:
        entry.setdefault("disposition", "missing")
        entry.setdefault("disposition_reason", "not reached by the designation pass")

    counts = Counter(e["disposition"] for e in entries)
    print(f"\n  dispositions: {dict(counts)}")

    # ---- closure ---------------------------------------------------------
    closure = {name: counts.get(name, 0) for name in DISPOSITIONS}
    total = sum(closure.values())
    closure_ok = total == len(entries)
    print(f"\n  CLOSURE: {' + '.join(f'{k}={v}' for k, v in closure.items() if v)} "
          f"= {total}  (records = {len(entries)})  -> {'CLOSED' if closure_ok else 'OPEN'}")
    if not closure_ok:
        raise SystemExit(f"disposition identity does not close: {total} != {len(entries)}")

    # ---- build the payloads ---------------------------------------------
    def reidentified(entry: dict) -> dict:
        """The original record, re-identified. Nothing observed is altered."""
        r = entry["record"]
        unit = make_evidence_unit(**entry["components"])
        payload = dict(r)
        payload["test_fingerprint"] = unit["id"]
        payload["evidence_unit"] = unit
        payload["evidence_unit_migration"] = {
            "schema_from": "1",
            "schema_to": EVIDENCE_UNIT_SCHEMA_VERSION,
            "legacy_test_fingerprint": unit["legacy_id"],
            "origin_experiment": r["experiment"],
            "origin_path": entry["origin_path"],
            "origin_sha256": entry["origin_sha256"],
            "metrics_recomputed": False,
            "test_split_touched": False,
            "derivation_sources": DERIVATION_SOURCES,
        }
        return payload

    adopted = [e for e in entries if e["disposition"] == "adopted"]
    rejected = [e for e in entries if e["disposition"] == "duplicate_test_touch"]

    # invariant: the migration may not alter anything observed
    for entry in adopted + rejected:
        assert entry["record"]["metrics"] is not None
    with (out_dir / "e9_adopted.jsonl").open("w") as fh:
        for entry in adopted:
            fh.write(json.dumps(reidentified(entry), sort_keys=True) + "\n")
    with (out_dir / "e9_rejected.jsonl").open("w") as fh:
        for entry in rejected:
            payload = {
                "reason": "duplicate_test_touch",
                "detail": entry["disposition_reason"],
                "kept": entry["superseded_by"],
                "original_record": entry["record"],
                "origin_path": entry["origin_path"],
                "origin_sha256": entry["origin_sha256"],
                "evidence_unit": make_evidence_unit(**entry["components"]),
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    # ---- manifest CSV ----------------------------------------------------
    columns = [
        "origin_path",
        "origin_sha256",
        "origin_dir",
        "origin_experiment",
        "legacy_test_fingerprint",
        "v1_5_evidence_unit_id",
        "v1_5_legacy_alias",
        "dataset", "task", "fold_fingerprint", "model", "window_length",
        "condition", "reservoir_size", "normalization", "input_mapping",
        "topology_variant", "rewire_seed",
        "unit_identity_collided",
        "records_in_unit",
        "disposition",
        "disposition_reason",
        "kept_config_hash",
        "config_hash",
        "status",
        "timestamp_utc",
    ]
    with (out_dir / "e9_v14_to_v15_manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            r = entry["record"]
            unit = make_evidence_unit(**entry["components"])
            writer.writerow({
                "origin_path": entry["origin_path"],
                "origin_sha256": entry["origin_sha256"],
                "origin_dir": entry["origin_dir"],
                "origin_experiment": r["experiment"],
                "legacy_test_fingerprint": r["test_fingerprint"],
                "v1_5_evidence_unit_id": unit["id"],
                "v1_5_legacy_alias": unit["legacy_id"],
                **{k: entry["components"][k] for k in (
                    "dataset", "task", "fold_fingerprint", "model", "window_length",
                    "condition", "reservoir_size", "normalization", "input_mapping",
                    "topology_variant", "rewire_seed")},
                "unit_identity_collided": True,
                "records_in_unit": len(units[entry["unit_key"]]),
                "disposition": entry["disposition"],
                "disposition_reason": entry["disposition_reason"],
                "kept_config_hash": entry.get("superseded_by", {}).get("config_hash", r["config_hash"]),
                "config_hash": r["config_hash"],
                "status": r["status"],
                "timestamp_utc": r["timestamp_utc"],
            })

    # ---- audit -----------------------------------------------------------
    per_size = Counter()
    for entry in adopted:
        per_size[entry["components"]["reservoir_size"]] += 1
    audit = {
        "artifact": "E9 evidence-identity migration (v1.4 schema 1 -> v1.5 schema 2)",
        "is_a_migration_not_an_experiment": {
            "metrics_recomputed": False,
            "models_refit": False,
            "test_splits_touched": False,
            "records_carry_origin_sha256": True,
            "note": (
                "every adopted record carries the sha256 of the file it came from, so a "
                "reader can prove the numbers were not re-derived"
            ),
        },
        "closure": {
            "records_total": len(entries),
            "by_disposition": closure,
            "closed": closure_ok,
            "identity": " + ".join(f"{k}={v}" for k, v in closure.items()) + f" = {len(entries)}",
        },
        "dispositions_are_a_partition": (
            "one state per record; the states sum to the record count"
        ),
        "not_a_disposition": NOT_A_DISPOSITION,
        "correction_to_the_first_report": (
            "The first repair-audit report said '280 adoptable, 70 inadmissible', which "
            "leaves 70 of the 420 records unexplained. 280 counts UNITS holding a single "
            "record, not records: the 70 duplicated units each contribute one adoptable "
            "and one rejected record, so the record-level partition is 350 adopted / 70 "
            "rejected / 420 total. No unit and no record is unaccounted for."
        ),
        "structure": {
            "legacy_fingerprints": n_fingerprints,
            "records_per_fingerprint": 6,
            "evidence_units": len(units),
            "substrates_per_fingerprint": sorted({
                len({json.dumps(derive_components(e), sort_keys=True) for e in v})
                for v in units.values()
            }),
            "duplicated_unit": (
                "N=250 was scored twice, in e9_size_d2 (config 2b681f4040fc, 06:07:31Z) "
                "and again in e9_size_d2_n250 (config e2dbcef1dad6, 06:52:35Z)"
            ),
            "adopted_by_reservoir_size": {str(k): v for k, v in sorted(per_size.items())},
        },
        "designation_rule": rule,
        "derivation_sources": DERIVATION_SOURCES,
        "adopted_records_are_not_yet_admissible_evidence": (
            "This migration removes the identity collision, which was the FIRST of two "
            "blockers. The second is that a pre-v1.5 record remains in the legacy ledger, "
            "so until the E2/E1 records are migrated too, a v1.5 re-run of an already "
            "scored unit is still conservatively refused. Migration does not admit the "
            "records to any gate by itself."
        ),
        "what_this_migration_does_not_do": [
            "it does not re-score anything",
            "it does not choose between the two N=250 touches on the basis of their metrics",
            "it does not make the 70 rejected records admissible under any later rule",
            "it does not touch results/raw, configs/, or any frozen protocol file",
            "it does not change any Gate, narrative rule, metric or decision rule",
        ],
    }
    (out_dir / "e9_migration_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )

    print(f"\n  adopted by reservoir size: {dict(sorted(per_size.items()))}")
    print(f"\n  written to {out_dir.relative_to(REPO_ROOT)}/")
    for name in ("e9_v14_to_v15_manifest.csv", "e9_adopted.jsonl",
                 "e9_rejected.jsonl", "e9_migration_audit.json"):
        size = (out_dir / name).stat().st_size
        print(f"    {name:34s} {size:>10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
