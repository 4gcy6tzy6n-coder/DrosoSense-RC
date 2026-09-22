# DATA-28 per-item adjudication (doc written by the DATA-53 landing round, 2026-09-21)

Scope: every deletion `fcac719` (the DATA-5 merge, "resolve replay conflict —
user's DATA-43/46 line wins, keep ES side's r2 guards + reservoir scoring
fields") performed against the then-current tree, and what DATA-53 / the
restored `data-52` branch (commits `01da3bd` → `af480cf` → `b3aec08` →
`2c4f156`, cherry-picked onto the delivery line) did about each one.
Disposition per item is one of **restore / no-restore-needed / deferred**,
with the reason. "Restore source" below always means tree of `d627186`
(`agent/jenifer/data-34`), verified byte-identical to the original commit
`57637c2` (DATA-28's `feat(DATA-28): server engineering harness + per-model
metrics + concurrency sweep`) via empty `git diff`.

Method (how each line was derived, all reproducible):

- `git log --all --follow -- <path>` to find the commit that deleted it
  (`fcac719`) and the commits that last touched it before that.
- `git diff fcac719^ fcac719 --stat` restricted to `scripts/server/`,
  `tests/server/`, `results/server_runs/` for the deletion set.
- Byte-identity check `git diff d627186 57637c2 -- scripts/server tests/server results/server_runs` → empty.
- Downstream check: grep for "retired/deprecated/removed" notes about the
  server harness in `README.md`, `docs/`, `configs/`, and test files → none
  found; `tests/server/` is the only place exercising `scripts/server/`.

## 1. `scripts/server/` (DATA-28 deliverable code)

| file | lines deleted by `fcac719` | disposition | reason |
|---|---|---|---|
| `README.md` | 98 | **restore** (`af480cf`) | Documents the harness; no retirement note anywhere. |
| `__init__.py` | 7 | **restore** (`af480cf`) | Required for the package import. |
| `concurrency_sweep.py` | 343 | **restore** (`af480cf`) | DATA-28's core sweep; still referenced by `tests/server/test_concurrency_sweep.py`. |
| `lib_common.py` | 325 | **restore** (`af480cf`) | Shared helpers for the sweep + baselines. |
| `measure_baselines.py` | 621 | **restore** (`af480cf`) | Baseline timing deliverable (80 vCPU + RTX 3080 Ti). |
| `run_on_server.sh` | 251 | **restore** (`af480cf`) | Launch script; no replacement exists on the delivery line. |

## 2. `tests/server/`

| file | lines deleted | disposition | reason |
|---|---|---|---|
| `__init__.py` | 0 | **restore** (`af480cf`) | Package marker; empty file. |
| `conftest.py` | 36 | **restore** (`af480cf`) | Fixture setup the 3 test files depend on. |
| `test_concurrency_sweep.py` | 72 | **restore** (`af480cf`) | `pytest tests/server` → 28 passed (verified inside `af480cf`'s message). |
| `test_lib_common.py` | 110 | **restore** (`af480cf`) | Same run. |
| `test_measure_baselines.py` | 144 | **restore** (`af480cf`) | Same run. |

## 3. `results/server_runs/data28/` (measured evidence, not re-runnable cheaply)

| file | lines deleted | disposition | reason |
|---|---|---|---|
| `SUMMARY.md` | 158 | **restore** (`af480cf`) | The only on-record summary of DATA-28's measured concurrency results. |
| `concurrency_sweep/finer/concurrency_environment.json` | 24 | **restore** | Provenance of the sweep. |
| `concurrency_sweep/finer/concurrency_sweep.csv` | 10 | **restore** | Measured data; re-running costs a full server session. |
| `concurrency_sweep/fourmodel/concurrency_environment.json` | 22 | **restore** | Same. |
| `concurrency_sweep/fourmodel/concurrency_sweep.csv` | 6 | **restore** | Same. |
| `evidence/data28_server_evidence.txt` | 47 | **restore** | The issue-level evidence artifact DATA-28 itself referenced. |
| `server_measurements/cpu/environment.json` | 27 | **restore** | Hardware provenance. |
| `server_measurements/cpu/hardware_metrics.csv` | 28 | **restore** | Measured values. |
| `server_measurements/cpu/hardware_summary.csv` | 10 | **restore** | Measured values. |
| `server_measurements/gpu/environment.json` | 27 | **restore** | Same. |
| `server_measurements/gpu/hardware_metrics.csv` | 13 | **restore** | Same. |
| `server_measurements/gpu/hardware_summary.csv` | 5 | **restore** | Same. |

No item in sections 1–3 was ever marked retired: the only deletion of these
files anywhere in the repo's history is `fcac719` itself (a replay-conflict
resolution whose commit message accounts only for the reservoir files, not
for the DATA-28 harness). The restore commit `af480cf` states this explicitly
and records it as a **new** commit, no history rewrite.

## 4. `connectome/select_neurons.py` — argparse side-effect fix

**Disposition: not a `fcac719` deletion — a forward fix bundled into `2c4f156`.**

`fcac719` did not touch `connectome/select_neurons.py`. The change in
`2c4f156`'s tree is the *new* `drososense/connectome_selection.py` module
(138 lines, new file) which must import `connectome/select_neurons.py`;
that import failed because `select_neurons.py` parsed its CLI args at module
level. The fix (move parsing under `if __name__ == "__main__"`) is a
prerequisite for the new module to be importable, not a recovery of anything
`fcac719` removed. It was applied because it is in `2c4f156`'s diff, and the
delivery line's `connectome/select_neurons.py` already had no module-level
argparse side-effect that the DATA-51 merge had reintroduced — so on the
delivery line this hunk is effectively a no-op / already-consistent; no
additional action beyond accepting `2c4f156`'s version of the file.

## 5. `scripts/run_reservoir_e2.py` and `drososense/reservoir/runner.py`

**Disposition: already landed via DATA-50's PR; `2c4f156` is a no-op for
these two on the delivery line (verified byte-identical to `2c4f156`'s own
versions, see the landing comment's blob-sha check for `runner.py` =
`241ad561...`).** `fcac719` predates DATA-50, so it could not have dropped
these — they simply didn't exist on `fcac719`'s base. No DATA-28
adjudication needed; recorded here only to close the loop on why
`2c4f156`'s stat line shows large insertions for `runner.py` that are NOT
new work on the delivery line.

## 6. Items explicitly **not** restored, and why

- **No `deferred` item in this doc.** Every `fcac719`-deleted file under
  `scripts/server/`, `tests/server/`, `results/server_runs/data28/` was
  restored by `af480cf`. Nothing was found in the deletion set that is both
  (a) deleted only by `fcac719` and (b) not needed — hence no "deferred"
  or "no-restore-needed" lines above; the empty result itself is the
  adjudication, and this section exists so that result is explicit rather
  than implied.

## Provenance

- Deletion commit: `fcac719` (DATA-5 merge).
- Original creation commit: `57637c2` (DATA-28).
- Restore source tree: `d627186` (`agent/jenifer/data-34`), byte-identical
  to `57637c2` for `scripts/server/` + `tests/server/` +
  `results/server_runs/` (verified `git diff` empty).
- Restore commit: `af480cf` (original `data-52` branch) → cherry-picked
  onto the delivery line as `93602f5` during this landing round.
