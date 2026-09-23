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
