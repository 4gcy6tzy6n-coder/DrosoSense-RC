#!/usr/bin/env python3
"""M4-Audit — evidence-ledger repair audit (protocol v1.5, does NOT re-run anything).

WHY THIS EXISTS. v1.5 fixes the identity definition, but that fix must not, by
itself, promote any existing record to valid evidence. The recovered GPU-box
records have to be classified per unit first:

  * ``valid_but_unreadable_under_old_identity`` — the old fingerprint collided,
    but the records differ in a substrate/condition component, so they are
    genuinely different evaluations that schema 1 could not name;
  * ``invalid_experimental_evidence`` — the same unit was scored twice under two
    configurations. That is a true §17 violation and stays one;
  * ``no_evidence_produced`` — nothing was ever scored.

The distinction is decided from the record's own fields, never from the
fingerprint alone: a colliding fingerprint is precisely the thing under audit.

READ-ONLY. This script reads the evidence recovered into
``results/audit/m4_audit/server_evidence/`` and writes one report. It starts no
run, touches no test split, and edits no protocol file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drososense.evaluation.results import (  # noqa: E402
    RunRecord,
    build_prior_ledgers,
    legacy_test_fingerprint,
    lookup_prior_unit,
    make_evidence_unit,
    test_touched_once_report,
)

EVIDENCE = REPO_ROOT / "results" / "audit" / "m4_audit" / "server_evidence" / "ev"
OUT_DIR = REPO_ROOT / "results" / "audit" / "m4_audit"

#: The substrate-defining components of a recorded run, read from the record's
#: own ``model_description``. A schema-1 record did record these — it just did
#: not put them in its identity — so a dry-run reconstruction is possible for
#: the reservoir families and impossible for a baseline (which has none).
def substrate_from_record(record: RunRecord) -> dict:
    md = record.model_description or {}
    topology = md.get("topology") or {}
    shared = md.get("shared") or {}
    return {
        "reservoir_size": topology.get("n_nodes"),
        "normalization": topology.get("normalization"),
        "input_mapping": shared.get("input_mapping"),
        "topology_variant": topology.get("kind"),
        "rewire_seed": None,  # not recorded by any pre-v1.5 run
    }


def load_experiment(name: str) -> list[RunRecord]:
    """Every record under one experiment directory, tolerating odd files."""
    out: list[RunRecord] = []
    for path in sorted((EVIDENCE / name).rglob("*.json")):
        try:
            out.append(RunRecord.from_dict(json.loads(path.read_text())))
        except Exception:
            continue
    return out


def classify(records: list[RunRecord]) -> tuple[list[dict], dict]:
    """Per-EVIDENCE-UNIT classification of one experiment's records.

    The meaningful granularity is ``(legacy fingerprint, substrate)``: a schema-1
    fingerprint cannot name the substrate, so one fingerprint covering several
    substrates is a collision, while one fingerprint covering the SAME substrate
    twice is a genuine repeated evaluation. Deciding between the two from the
    fingerprint alone is impossible — which is the whole point — so the substrate
    is read from each record's own ``model_description``.
    """
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.status != "ok" or not record.test_fingerprint:
            continue
        groups.setdefault(record.test_fingerprint, []).append(record)

    rows: list[dict] = []
    counts = {
        "collision_only": 0,
        "true_repeat_different_config": 0,
        "recomputation_same_config": 0,
        "single_scored_evaluation": 0,
    }
    n_colliding_fingerprints = 0
    n_fingerprints_with_true_repeat = 0

    for fingerprint, members in sorted(groups.items()):
        by_substrate: dict[str, list[RunRecord]] = {}
        for record in members:
            key = json.dumps(substrate_from_record(record), sort_keys=True)
            by_substrate.setdefault(key, []).append(record)

        if len(by_substrate) > 1:
            n_colliding_fingerprints += 1
        fingerprint_has_true_repeat = False

        for substrate_key, group in sorted(by_substrate.items()):
            substrate = json.loads(substrate_key)
            scored = [
                r for r in group if r.n_test_windows > 0 and r.metrics
            ]
            hashes = {r.config_hash for r in group}
            unit = make_evidence_unit(
                dataset=group[0].dataset,
                task=group[0].task,
                fold_fingerprint=group[0].fold_fingerprint,
                model=group[0].model,
                window_length=group[0].window_length,
                reservoir_size=substrate.get("reservoir_size"),
                normalization=substrate.get("normalization"),
                input_mapping=substrate.get("input_mapping"),
                topology_variant=substrate.get("topology_variant"),
                rewire_seed=substrate.get("rewire_seed"),
            )

            if len(scored) <= 1:
                classification = "single_scored_evaluation"
            elif len(hashes) == 1:
                classification = "recomputation_same_config"
            else:
                classification = "true_repeat_different_config"
                fingerprint_has_true_repeat = True

            counts[classification] += 1
            rows.append(
                {
                    "legacy_fingerprint": fingerprint,
                    "dataset": group[0].dataset,
                    "model": group[0].model,
                    "task": group[0].task,
                    "seed": group[0].seed,
                    "fold_id": group[0].fold_id,
                    "window_length": group[0].window_length,
                    "reservoir_size": substrate.get("reservoir_size"),
                    "normalization": substrate.get("normalization"),
                    "input_mapping": substrate.get("input_mapping"),
                    "topology_variant": substrate.get("topology_variant"),
                    "rewire_seed": substrate.get("rewire_seed"),
                    "v1_5_identity_id": unit["id"],
                    "n_ok_records_on_this_substrate": len(group),
                    "n_scored_on_this_substrate": len(scored),
                    "n_config_hashes_on_this_substrate": len(hashes),
                    "config_hashes": "|".join(sorted(hashes)),
                    "experiments": "|".join(sorted({r.experiment for r in group})),
                    "n_substrates_sharing_this_fingerprint": len(by_substrate),
                    "classification": classification,
                }
            )
        if fingerprint_has_true_repeat:
            n_fingerprints_with_true_repeat += 1

    summary = {
        "n_records": len(records),
        "n_ok_records": sum(1 for r in records if r.status == "ok"),
        "n_legacy_identity_records": sum(
            1 for r in records if not (r.evidence_unit or {}).get("id")
        ),
        "n_legacy_fingerprints": len(groups),
        "n_fingerprints_with_more_than_one_substrate": n_colliding_fingerprints,
        "n_fingerprints_with_a_true_repeat": n_fingerprints_with_true_repeat,
        "n_evidence_units": len(rows),
        **{f"units_{k}": v for k, v in counts.items()},
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-limit", type=int, default=6)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"audit": "evidence_ledger_repair", "schema": "protocol_v1.5"}

    print("=" * 78)
    print("EVIDENCE-LEDGER REPAIR AUDIT (read-only; no run is started)")
    print("=" * 78)

    # ---------------------------------------------------------------- E9
    e9 = []
    for name in (
        "e9_size_d2",
        "e9_size_d2_n250",
        "e9_size_d2_n500",
        "e9_size_d2_n1000",
        "e9_size_d2_n2000",
        "e9_size_d2_n4000",
    ):
        e9.extend(load_experiment(name))
    print(f"\n--- E9 size study: {len(e9)} records ---")
    rows, summary = classify(e9)
    touched = test_touched_once_report(e9)
    print(f"  records={summary['n_records']}  ok={summary['n_ok_records']}  "
          f"legacy fingerprints={summary['n_legacy_fingerprints']}")
    print(f"  §17 report under the STORED (schema 1) fingerprints: "
          f"{touched['n_violations']} violations "
          f"({touched['n_repeated_test_fingerprints']} repeated units)")
    print(f"  fingerprints: {summary['n_legacy_fingerprints']}  "
          f"of which >1 substrate (collisions): "
          f"{summary['n_fingerprints_with_more_than_one_substrate']}  "
          f"with a TRUE repeat: {summary['n_fingerprints_with_a_true_repeat']}")
    print(f"  evidence units (fingerprint x substrate): {summary['n_evidence_units']}")
    print(f"    single scored evaluation ......... {summary['units_single_scored_evaluation']}")
    print(f"    same-config recomputation ....... {summary['units_recomputation_same_config']}")
    print(f"    TRUE REPEAT, different config ... {summary['units_true_repeat_different_config']}")
    print(f"  sample (first {args.print_limit} units):")
    print(f"  {'legacy fp':18s} {'N':>5s} {'norm':>10s} {'recs':>5s} {'substr':>7s} "
          f"{'cfgs':>5s}  classification")
    for row in rows[: args.print_limit]:
        print(f"  {row['legacy_fingerprint']:18s} {str(row['reservoir_size']):>5s} "
              f"{str(row['normalization']):>10s} "
              f"{row['n_ok_records_on_this_substrate']:5d} "
              f"{row['n_substrates_sharing_this_fingerprint']:7d} "
              f"{row['n_config_hashes_on_this_substrate']:5d}  {row['classification']}")
    if len(rows) > args.print_limit:
        print(f"  ... {len(rows) - args.print_limit} more units "
              f"(full table in ledger_repair_E9_units.csv)")
    report["e9"] = {
        "summary": summary,
        "units": rows,
        "dry_run_caveats": {
            "normalization_for_controls": (
                "The dry run reads each record's own `model_description.topology."
                "normalization`. That field documents R0's normalization and is "
                "null for R1..R6 (a control is built from the already-normalized "
                "R0 matrix), whereas the reservoir runner passes "
                "config.normalization into the identity for EVERY family. The "
                "dry run is therefore faithful for R0 and conservative for the "
                "controls. It does not affect either verdict: collisions and "
                "repeats here are decided by `reservoir_size`, which every "
                "family does record."
            ),
            "rewire_seed": (
                "No pre-v1.5 record stored the rewiring seed. The dry run leaves "
                "it unknown and never guesses it; a re-filed record must carry it "
                "from the run's own seed."
            ),
        },
        "touched_once": {
            k: touched[k]
            for k in (
                "n_violations",
                "n_repeated_test_fingerprints",
                "n_distinct_test_fingerprints",
            )
        },
    }

    # ---------------------------------------------------------------- E3
    print("\n--- E3 low-data: what exists? ---")
    e3_records = [r for name in ("e3_lowdata_d2_f10", "e3_lowdata_d2_f25",
                                 "e3_lowdata_d2_f50", "e3_lowdata_d2_f75",
                                 "e3_lowdata_d2_f100")
                  for r in load_experiment(name)]
    disclosures = []
    for path in sorted((EVIDENCE / "tables").glob("e3_lowdata_*_skip_disclosure.json")):
        payload = json.loads(path.read_text())
        disclosures.append(
            {
                "receipt": path.name,
                "n_skipped_units": payload.get("n_skipped_units"),
                "n_skipped_same_config": payload.get("n_skipped_same_config"),
                "n_skipped_different_config": payload.get("n_skipped_different_config"),
                "run_config_hash": payload.get("run_config_hash"),
                "reasons": sorted({d.get("reason") for d in payload.get("disclosures", [])}),
                "prior_run_ids": sorted(
                    {d.get("prior_run_id") for d in payload.get("disclosures", [])}
                )[:3],
            }
        )
    n_skipped = sum(d["n_skipped_units"] or 0 for d in disclosures)
    print(f"  raw records: {len(e3_records)}  <- nothing was scored")
    print(f"  skip receipts: {len(disclosures)} covering {n_skipped} units, all "
          f"reason(s)={sorted({r for d in disclosures for r in d['reasons']})}")
    for d in disclosures:
        print(f"    {d['receipt']:42s} skipped={d['n_skipped_units']:3d} "
              f"same={d['n_skipped_same_config']} diff={d['n_skipped_different_config']}")
    print(f"  -> classification: no_evidence_produced ({n_skipped} units, 0 scored). "
          f"There is nothing here to call valid or invalid.")
    report["e3"] = {
        "n_records": len(e3_records),
        "n_skipped_units": n_skipped,
        "receipts": disclosures,
        "classification": "no_evidence_produced",
    }

    # ------------------------------------------- legacy fallback implication
    print("\n--- the legacy fallback: is an E3 re-run unblocked by v1.5 alone? ---")
    e2 = load_experiment("e2_main_d2")
    ok_e2 = [r for r in e2 if r.status == "ok"]
    current, legacy = build_prior_ledgers(ok_e2)
    probe = make_evidence_unit(
        dataset="d2_beef_uncontrolled",
        task="classification",
        fold_fingerprint=ok_e2[0].fold_fingerprint if ok_e2 else "",
        model="R0",
        window_length=16,
        condition="train10pct",
        reservoir_size=250,
        normalization="n1_pre_l1",
        input_mapping="dense_random_all_nodes",
        topology_variant="R0_real_fly",
        rewire_seed=0,
    )
    hit = lookup_prior_unit(probe, current, legacy)
    print(f"  legacy ok records on D2 (E2): {len(ok_e2)}  "
          f"current-identity keys={len(current)}  legacy-identity keys={len(legacy)}")
    if hit:
        print(f"  an E3-style v1.5 unit (condition=train10pct, N=250) is STILL "
              f"matched by the legacy fallback: identity={hit[0]} "
              f"prior config={hit[1]}")
        print("  -> v1.5 alone does not unblock E3. The pre-v1.5 records must be")
        print("     classified and re-filed under their own components first, or a")
        print("     re-run would be conservatively refused (never silently re-scored).")
    else:
        print("  an E3-style v1.5 unit does NOT match any legacy key -> the E3 "
              "re-run is unblocked by the identity fix alone.")
    report["legacy_fallback"] = {
        "n_legacy_ok_records_d2": len(ok_e2),
        "current_identity_keys": len(current),
        "legacy_identity_keys": len(legacy),
        "e3_style_probe_id": probe["id"],
        "matched": bool(hit),
        "matched_identity": hit[0] if hit else None,
        "matched_config_hash": hit[1] if hit else None,
        "interpretation": (
            "v1.5 does not by itself unblock E3; the pre-v1.5 records must be "
            "classified and re-filed under their own components first."
            if hit
            else "the identity fix alone unblocks the E3 re-run"
        ),
    }

    # ---------------------------------------------------------------- write
    (OUT_DIR / "ledger_repair_E9.json").write_text(
        json.dumps(report["e9"], indent=2, sort_keys=True)
    )
    with (OUT_DIR / "ledger_repair_E9_units.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ledger_repair_E3.json").write_text(
        json.dumps(report["e3"], indent=2, sort_keys=True)
    )
    (OUT_DIR / "ledger_repair_summary.json").write_text(
        json.dumps(
            {
                "e9_summary": report["e9"]["summary"],
                "e9_violations_under_old_identity": report["e9"]["touched_once"],
                "e3": {k: report["e3"][k] for k in ("n_records", "n_skipped_units",
                                                    "classification")},
                "legacy_fallback": report["legacy_fallback"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nEvidence written under {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("  ledger_repair_E9.json / ledger_repair_E9_units.csv")
    print("  ledger_repair_E3.json / ledger_repair_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
