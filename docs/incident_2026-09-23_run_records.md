# Incident — `results/raw` deleted on the GPU box, 2026-09-23, and its recovery

**Status: fully recovered except 7 `e2_smoke` records. Nothing in the delivered
evidence set was lost.** This document exists because the loss was caused by this
session's own operator step, and because the near-miss is instructive: the
recovery came from an unrelated stray tarball, not from a backup that was
supposed to exist.

## What happened

While verifying that a server-only test failure was data-dependent rather than
code-dependent, the operator wanted the test's premise reproduced twice: once
with `results/raw` populated and once with it absent. The command was

```bash
mv results/raw results/raw.audit-hold
mkdir -p results/raw
mv results/raw.audit-hold/_artifacts results/raw/
rm -rf results/raw.audit-hold          # <-- the mistake
```

The `rm -rf` released the hold directory **before** its contents were moved back,
and the "restore" line that followed addressed a path that did not exist
(`/root/autodl-tmp/drososense/records-hold/`). `results/raw/` was left empty.

**Impact: 21,867 run-record JSON files** — E1 (900 + 11,160), E2 (700 + 8,680),
E9 (6 × 70), `e2_smoke` (7).

**Why the existing backup did not help.** The backup taken minutes earlier for the
v1.5 deployment deliberately excluded `results/raw`
(`--exclude=./results/raw --exclude=./results/_artifacts`), on the reasoning that
raw records are large and reproducible. That reasoning was wrong for this
directory: `results/raw/**` is git-ignored, so it is exactly the data with no
other copy.

## How it was recovered

| Source | Contents | Records |
|---|---|---|
| `/root/autodl-tmp/drososense/m4_bundle_pull/raw4.tar` (82 MB, 2026-09-22 12:35) | `e1_main_d2`, `e1_main_d3`, `e2_main_d2`, `e2_main_d3`, `_artifacts` | 21,440 |
| this repository, `results/audit/m4_audit/server_evidence/ev/e9_size_d2*` (pulled during the audit) | the six E9 size-study directories | 420 |
| **total restored** | | **21,860** |

Verified against the pre-loss counts, per experiment:

| experiment | restored | pre-loss |
|---|---:|---:|
| `e1_main_d2` | 900 | 900 |
| `e1_main_d3` | 11,160 | 11,160 |
| `e2_main_d2` | 700 | 700 |
| `e2_main_d3` | 8,680 | 8,680 |
| `e9_size_d2`, `_n250`, `_n500`, `_n1000`, `_n2000`, `_n4000` | 70 each | 70 each |
| **`e2_smoke`** | **0** | **7** ← the residual loss |

A restored record was inspected field by field (`31` keys, `status: ok`,
`metrics` intact) rather than counted only.

The `raw4.tar` had never been documented; it was found by searching the box for
archives after the loss. It predates the E9 runs, which is why E9 is absent from
it and had to come from the audit's own copy.

## Residual loss, stated exactly

**7 `e2_smoke` records.** It is a smoke experiment (pipeline check, not a
finding); its derived summary `results/tables/e2_smoke_summary.csv` survives, so
no reported number depends on the missing records. No E1, E2 or E9 record is
missing.

## What changed as a result

1. A backup that **includes `results/raw`** now exists in two places:
   `/root/autodl-tmp/drososense/backups/results-with-raw-20260923T022731Z.tgz`
   (21,908 JSON files) and, pulled to the delivery side,
   `results/audit/m4_audit/server_evidence/results-with-raw-20260923T022731Z.tgz`
   (git-ignored, like `ev/`).
2. **Rule, for the next operator:** a destructive command is never run against a
   directory that is git-ignored unless a *verified* copy of that exact directory
   exists elsewhere. `results/raw/**`, `data/raw/**`, `data/processed/**`,
   `data/splits/**`, `connectome/adjacency/**` are all in that class — the
   repository's own `.gitignore` is a list of the things that have no second
   copy.
3. Move-and-restore is never expressed as `mv X X.hold; ...; rm -rf X.hold` in one
   unattended line. If a directory must be temporarily absent, it is *copied*
   aside, or renamed and renamed back in a command whose failure mode is "nothing
   happens".

## Correction to the recovery claim: the loss was larger than 7 records

The first recovery pass checked **only `results/raw`** and concluded "recovered in
full except 7 `e2_smoke` records". That claim was too narrow. A whole-`results/`
comparison against the backup taken at 10:27 found **five more files missing from
`results/tables/`**:

    e2_main_d2_summary.csv
    e2_main_d3_summary.csv
    e2_main_d3_skip_disclosure.json
    e2_smoke_summary.csv
    e9_size_d2_skip_disclosure.json

All five survived in the audit's own evidence copy
(`results/audit/m4_audit/server_evidence/ev/tables/`) and in the on-box staging
directory `/tmp/ev/tables/`; their sha256 matched between those two copies
(808ce88a…, a340e8cf…, 12f2d745…, bbe6bd7d…, cf39637f…), and they have been
restored to the server and committed to the delivery line together with the other
eleven server-only E2/E3/E9 tables.

The whole-`results/` check is now closed rather than narrowed: the tree holds
23,319 files against the backup's 23,312, and the only differences are eight
additions — the five restored tables plus three `scaler.pkl` artefacts written by
running the test suite on the server — and one empty line, which is an artefact of
the tar listing and not a file. **Nothing is missing.**

**Revised residual loss: the 7 `e2_smoke` per-run JSON records.** Their derived
summary survives, so no reported number depends on them.

## `e2_smoke` was NOT synthetic — do not "reconstruct" it by re-running

The follow-up plan proposed re-running the 7 `e2_smoke` records on the grounds
that the experiment was synthetic, touched no test split, and fed no reported
number. **The first two are false**, and the evidence is unambiguous:

* `results/tables/e2_smoke_summary.csv` records all seven rows as
  `dataset=d2_beef_uncontrolled`, **`evidence_class=real`**,
  `protocol_compliant=True`, `status=ok`, `protocol_version=1.4.0`, one specimen
  evaluation each;
* `results/tables/data_contact_log.json` carries an `e2_smoke` entry with
  `evidence_class: real`, `dataset: d2_beef_uncontrolled`, `split_strategy: loso`,
  `n_models: 7` and **`counts_as_first_test_evaluation: true`** — i.e. the project
  itself counted these runs as a real D2 test touch.

So re-running them would **re-score a real D2 test split**, which is exactly what
§17 exists to prevent and what the standing instruction forbids. The derived
summary is the surviving evidence of that touch, and it is now committed.

If a synthetic smoke test is wanted for the CI/reproducibility loop, it must be a
**new** experiment label on synthetic data (for example `e2_smoke_ci`) that is
explicitly *not* a reconstruction of the lost runs — never the same label, and
never on D2.

## Follow-on: the local connectome data was moved to the server, deliberately

Separately from this incident, the user asked to free local disk space by
removing the ~12 GB `data-root/connectome` tree. It was first established that the
server held **none** of it — a whole-filesystem search found no `flywire_synapses_783`,
no `proofread_connections_783`, no edge CSVs — and that the repository records no
download URL, so those bytes are not re-acquirable from anything committed. The
tree was therefore pushed to
`/root/autodl-tmp/drososense/data-root/connectome/` and **verified before
deletion**: the six raw inputs were sha256-compared against the hashes committed
in `connectome/metadata/olfactory_v1_meta.json` (6/6 match on both sides), the two
edge CSVs — which had **no hash recorded anywhere** — were hashed independently on
both sides and matched, the adjacency npz matched (`ae86cbb9…`), and the 16
neuron-class-ranking tables matched 16/16. Only then were the local copies
removed; 12.2 GB was freed (the disk had been at 98 %).

The full per-file evidence, including the two edge-CSV hashes now recorded for
the first time, is `results/audit/m4_audit/server_data_verification.json`.

One consequence, stated rather than discovered later: two real-NPZ guard tests in
`tests/test_reservoir_topology.py` now **skip** locally with the reason
"olfactory_v1.npz not provisioned in this environment" (the local suite is 515
passed / 8 skipped instead of 517 / 6). They run on the server, where the NPZ
lives.

## The test failure that prompted it — separate, real, and not fixed here

The failure was
`tests/test_e2e_gate_evaluation.py::test_the_gate_outcome_does_not_depend_on_the_git_ignored_run_records`.
It is **not** a v1.5 regression: the same code passes with `results/raw` held
aside and fails with it populated, and no v1.5 commit touches `scripts/analyze.py`
or the parameter table. The mechanism it exposes is a real defect of its own:

`model_parameter_counts(load_records())` derives `params(model)` from **every**
record under `results/raw`, with no experiment scope — `load_records()` rglobs the
whole tree. With the E2/E9 evidence present it therefore reports
`R0 = 16004` (the N=4000 readout from an E9 size-study record) and
`R4 = 16004`, against the committed table's `R4 = 804`, and `random_forest =
192606` against the committed `355623`. So **`params(R0)` in Gate_A can be set by
an E9 record that no E2 contrast used**, and the committed
`results/tables/model_parameters.json` is only equal to the records-derived table
on a tree whose `results/raw` holds the M1 benchmark and nothing else.

That is a parameter-table scope defect, not an identity defect, so it is reported
here rather than folded into the v1.5 evidence-identity commit — the two changes
must not travel together. It needs its own decision: either `params()` is scoped
to the experiment and configuration that produced the contrast, or the table
declares which experiment it is the parameter source for.
