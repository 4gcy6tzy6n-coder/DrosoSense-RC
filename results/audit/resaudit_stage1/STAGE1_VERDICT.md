# ResAudit-Food Stage 1 verdict — the family x dataset admissibility matrix

Status: generated 2026-09-23T17:01:48.388537+00:00 at HEAD `208475edf8593da327d77eef8e3be54ad0ff8c73`. **No food data was read, no label was seen, and no task metric is computed anywhere in this stage.**

## What was measured

F1's wiring is the provenance-verified frozen S0 substrate (node_list_sha256 `77ec9bfd90d6ec72…`, edge_list_sha256 `a840f45bd5c96ce6…`, N=1000, M=80443), status `VERIFIED_FOR_STAGE1`. F2/F3/F5 were generated against it at each width, and each width was audited in its own pass so the declared `counterfactual_parent='F1'` resolves.

Two input widths are audited because `Din` varies by dataset (amendment 1 §4 of amendment 2; amendment 6): **Din=6 for FD1/FD3** and **Din=8 for FD2**. The input support is 473 nodes at both widths, so `Din` is the only thing that varies.

### Din = 6  (used by FD1, FD3)

| Family | A1 | A2 | A3 Krylov | A4 Memory | A5 State | Food eligible | Construct qualified |
|---|---|---|---|---|---|---|---|
| F1 Drosophila olfactory connectome (frozen S0) | PASS | N/A | FAIL | N/A | N/A | no | no |
| F2 statistics-matched random control (10|E| accepted swaps) | PASS | FAIL | FAIL | N/A | N/A | no | no |
| F3 topology-destroyed directed G(n, m) control | PASS | FAIL | FAIL | N/A | N/A | no | no |
| F5 audit-qualified sparse construction (coprime-period blocks) | FAIL | FAIL | FAIL | N/A | N/A | no | no |

`Food eligible` = cleared A3, so it may enter the food stage. `Construct qualified` = cleared every ADMISSION and evaluated criterion. They are independent: a family can be food-eligible without being construct-qualified. Neither column means "Stage 1 passed".

### Din = 8  (used by FD2)

| Family | A1 | A2 | A3 Krylov | A4 Memory | A5 State | Food eligible | Construct qualified |
|---|---|---|---|---|---|---|---|
| F1 Drosophila olfactory connectome (frozen S0) | PASS | N/A | FAIL | N/A | N/A | no | no |
| F2 statistics-matched random control (10|E| accepted swaps) | PASS | FAIL | FAIL | N/A | N/A | no | no |
| F3 topology-destroyed directed G(n, m) control | PASS | FAIL | FAIL | N/A | N/A | no | no |
| F5 audit-qualified sparse construction (coprime-period blocks) | FAIL | FAIL | FAIL | N/A | N/A | no | no |

`Food eligible` = cleared A3, so it may enter the food stage. `Construct qualified` = cleared every ADMISSION and evaluated criterion. They are independent: a family can be food-eligible without being construct-qualified. Neither column means "Stage 1 passed".

## F4

**BLOCKED / NOT EVALUATED** — varied factor *cell-type connectivity (named)*. the construction-feasibility gate does not pass; no stand-in is substituted, because amendment 1's design commitment forbids it and the pre-registration already contained F4

F4 is kept in the matrix above as an explicit blocked row rather than deleted: the pre-registration contained F4, so a silent omission would misrepresent the design.

## The two verdicts are separate and must stay separate

```
food_eligible        = A3 PASS                    (pre-registration section 6.1)
construct_qualified  = A1 and A2 and A3 and A5    (BLOCKING_CRITERIA)
```

A4 is **not** in `BLOCKING_CRITERIA`: amendment 2 retired it to descriptive status because the observable does not track recurrence strength in the direction its own definition requires (more recurrent coupling scores *lower*). A4 values are reported in `A4_MEMORY.json` and in the matrix, and they never gate anything.

A family can therefore be `food_eligible = True` while `construct_qualified = False` — it cleared the primary gate but failed an admission or downstream criterion. The two columns are emitted separately and must never be collapsed into one "Stage 1 passed" flag.

## Caveats carried with these numbers

- **A3 PASS is not evidence of recurrence.** Its name was narrowed to *finite-horizon input-reachable state diversity* because calibration falsified the broader reading: a feed-forward nilpotent chain passes it.
- **F5's A3 status is a construction property, not a result.** F5 is built to satisfy A3, so "F5 passes A3" cannot be a finding.
- **A3 FAIL does not predict poor task performance** in this design, and that claim is deliberately untestable here: A3-FAIL families are never run on food data.
- **A4/A5 were measured on the frozen white-noise probe** (amendment 1 §0.1, amendment 3), whose spectral character differs from the autocorrelated food drives, so neither may be presented as a predictor of task performance.
