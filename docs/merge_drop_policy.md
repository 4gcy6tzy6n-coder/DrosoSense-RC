# Merge-Drop Policy (recorded 2026-09-21, triggered by `fcac719` / DATA-52)

This rule exists because a real merge (`fcac719`, DATA-5) performed a
silent **functional** drop — it left the tree advertising
`normalization="n0_raw"` while the loader had lost its NPZ-key routing for
that entry, and it dropped the DATA-28 server harness (~2,000 lines,
including its tests and its measured run evidence) with no per-file
disposition in the commit message. `fcac719` did step ① of the rule below
(it narrated the reservoir resolution), but not ② or ③; this document is
the ③-for-① backfill, and the guard that `fcac719`'s own class of bug can
no longer ship silently.

## The rule

**Any merge that drops, overwrites, or "took theirs/ours" a chunk of code
must, in the merge round itself:**

1. **① Narrate** — the commit message accounts for the drop file by file
   (which side won, what was lost). `fcac719` did this for the reservoir
   file; it did *not* do it for `scripts/server/` + `tests/server/` +
   `results/server_runs/data28/`, which is exactly why the drop needed a
   post-hoc audit (DATA-52).
2. **② File a recovery / disposition issue** — the drop does not stay an
   implicit byproduct of the merge. If the lost code is still needed, the
   issue's acceptance is *restoring it*; if it is being retired, the
   issue records the reason and where the retirement is noted
   (`docs/`/`README`). Either outcome is explicit; **silence is not an
   allowed disposition.**
3. **③ Add a guard test** — at least one test must fail if the dropped
   code path is re-dropped by a future merge. The guard's job is to make
   the specific regression *catchable by CI*, not merely by a human
   re-reading a diff. For a functional loader path (DATA-52's case), that
   is a test that exercises the path end to end against the real input
   layout; see
   `tests/test_reservoir_topology.py::test_all_allowed_normalizations_load_from_repo_npz`.

## Why ③ is the non-obvious part

① and ② are process: they only exist on the commit and the issue tracker,
and either can be forgotten. ③ is the only step that *outlives* the merge
round — it is the one that catches a future, even more careful merge that
introduces the same class of bug in a new place. A merge with perfect
commit-message narration and a filed disposition issue, but no tripwire,
has shipped `fcac719`'s regression again; the narration and the issue are
both absent from the next reader's view by the time the bug matters.

## Scope of this rule

- Applies to code (`drososense/`, `scripts/`, `tests/`, `connectome/`).
  Documentation, fixtures, and config-YAML drops follow the same three
  steps, but the guard test for a config/protocol YAML drop is typically a
  *freeze-check* test (digest sidecar) rather than a functional one — the
  `configs/protocol_v1.*.yaml` files are already covered by that mechanism
  and this rule does not extend or modify it.
- "Drop" means the merged tree no longer *executes* or *tests* something
  the losing side had working. A plain conflict where the winning side's
  version is a strict superset (the losing side's unique lines were
  already present, just not in that exact position) is not a drop and
  needs no issue — but the merge author must state that conclusion in ①
  so a reader doesn't have to re-derive it.
