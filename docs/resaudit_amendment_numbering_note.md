# Amendment numbering anomaly — recorded, not backfilled

**Status: provenance note, 2026-09-23. Contains NO rule, NO threshold and NO measurement.**

The amendment chain is expected to be append-only and contiguous. It is contiguous in Git,
but two workstreams were writing to this repository concurrently, so one number was consumed
by a different author than the one who had planned to use it.

| amendment | subject | issued by |
|---|---|---|
| 1 | calibration findings; frozen conventions; A3 narrowed | workstream A |
| 2 | withdrawal of the memory-contribution claim; blocking logic | workstream A |
| 3 | the probe's input realization | workstream B |
| **4** | **F1 structural identity and its dataset-conditioned input** | workstream B |
| 5 | the `A_hash` reproduction criterion is defective | workstream B |

**Workstream A had planned the number 4 for the amended weight/scale rule. Workstream B
issued its own amendment 4 before that rule was written, so the number was taken.**

## The rule applied

The new weight and scale rule is **not** written into amendment 4. Backfilling substantive
content into a number lower than amendments that already exist would fabricate a timeline: a
reader would conclude the weight rule preceded the F1-identity and `A_hash` decisions, when it
in fact followed them. Chronology is the point of this chain, so the rule takes the next free
number (6) and amendment 4 is left exactly as issued.

No file was rewritten to produce this note. `docs/resaudit_food_preregistration_amendment_4.md`
is unmodified; its SHA-256 is `9992d79f4ed524da0b13babd62345a6c2452b03c4b7f9ca9c0eb9bdc18ed9ac1`,
matching its `.sha256` sidecar.

An earlier attempt during this round DID overwrite amendment 4 with a tombstone. It was
restored from Git in the same round and verified by hash before this note was written. The
fact is recorded because a silently-corrected mistake is worse provenance than a stated one.

```
thresholds changed                none
candidates admitted or excluded   none
measurements reported             none
files of other workstreams edited none (after restoration)
```
