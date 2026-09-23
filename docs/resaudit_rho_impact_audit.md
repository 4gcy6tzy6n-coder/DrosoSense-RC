# Measurement-layer impact audit: the spectral-radius defect

**Status: COMPLETE 2026-09-23. Read-only on data; no historical output was overwritten.**
**All real family generation and food evaluation remain on HOLD.**

```
Infrastructure                 = READY
Measurement-layer provenance   = REVALIDATED (this document)
F2/F3 generators               = IMPLEMENTED, not scientifically instantiated
F4                             = BLOCKED
F5 current construction        = BLOCKED
Food evaluation                = HOLD
```

Runner: `ops/audit/resaudit_rho_impact.py`.
Report: `results/audit/resaudit_rho_impact/rho_impact_audit.json`.

---

## 1. There were TWO defects, not one

The defect I fixed in the previous round was mine, in `resaudit/battery.py`. Auditing it
surfaced a second, **independent and more serious** one in the frozen layer.

| layer | estimator | 4-cycle (ρ=1) | 8 disjoint cycles (ρ=1) | realistic dense graph |
|---|---|---|---|---|
| frozen `drososense` | `eigsh(matrix, k=1)` | **1.5909** | **1.9808** | **+1.4 % to +26.8 %** |
| resaudit pre-fix | single-vector power iteration | 1.0000 | **0.0063** | 0.00 % |
| resaudit current | 8-dim subspace iteration | 1.0000 | 1.0000 | 0.00 % |
| **oracle** | dense eigendecomposition | 1.0000 | 1.0000 | — |

**The frozen defect:** `drososense/reservoir/connectome_reservoir.spectral_radius` calls
`scipy.sparse.linalg.eigsh`, which is ARPACK for **symmetric** problems. Applied to a
directed adjacency it does not compute the spectral radius at all. On a 4-node directed
cycle — radius exactly 1 — it returns **1.5909**, varying with the start vector
(1.4499–1.5909 across seeds). On realistic dense directed graphs it is biased high by
**1.4 % to 26.8 %**.

This is a *different failure mode* from mine. Mine was a non-convergent iteration that
returned near-zero or negative values on degenerate spectra. The frozen one returns
**plausible-looking but wrong** numbers everywhere, including on ordinary graphs where mine
happened to be right. The second is worse: a wrong-but-plausible ρ silently mis-scales, and
nothing looks broken.

**Scope of the frozen defect, by call site:** `rescale_to_spectral_radius` (8 call sites),
`rescale_to_rewired_spectral_radius`, `constraint_match`'s ρ term, and therefore
`results/audit/v3_selection/V3_substrate_selection.json`,
`results/audit/m5_structural_dynamics/M5_structural_dynamics.json`,
`ops/audit/c4_design_diagnostic.py`, `scripts/run_reservoir.py`, and
`results/audit/m4_audit/ledger_repair_E9.json`.

**`resaudit` does not import the frozen estimator**, so this round's `resaudit` results are
unaffected by the frozen defect specifically — but they *were* affected by mine, which is
what the revalidation below addresses.

## 2. Solver validation (independent oracles, not self-consistency)

The corrected estimator was validated against dense eigendecomposition, which is exact:

| family | worst absolute error |
|---|---|
| known-spectrum degenerate cases | **1.13e-14** |
| dense random graphs (n = 60–200) | **0.00e+00** |
| the property that actually matters: ρ(αA) ≈ 0.95 after scaling | **6.33e-15** |

So the estimator is now correct on both regimes, and — importantly — the check is
`ρ(scaled) ≈ target`, not merely "the function does not return a negative number".

Independent cross-checks recorded in the report: ARPACK `eigs(which='LM')` **agrees** with
dense truth where it converges (4-cycle, 3 coprime cycles) and **fails to converge** on the
degenerate unions, which is why dense eigendecomposition was made the oracle of record. A
solver that is right where it converges and silent where it does not is not usable as the
primary oracle for this project's structured substrates.

## 3. Blast radius, and the honest verdict per artifact

| artifact | in scope? | status |
|---|---|---|
| A3 Krylov score | **no** | `UNAFFECTED` — `krylov_score` never rescales `A`; A3 is scale-*sensitive* but not scale-*computed*, so the estimator can only corrupt it by supplying a wrong scale to a caller |
| A4 construct-validity ladder | **yes** | `INVALIDATED_BY_MEASUREMENT_BUG` → recomputed in §4; historical JSON left in place, superseded by this report |
| Stage-1 calibration table | **yes** | `INVALIDATED_BY_MEASUREMENT_BUG` → recomputed; conclusions in §4 |
| F2/F3/F5 generators | **yes** | `INVALIDATED_BY_MEASUREMENT_BUG`; generators now call the corrected estimator, and no F2/F3/F5 instance has been scientifically instantiated, so nothing downstream inherits it |
| F4 feasibility gate | **no** | `UNAFFECTED` — imports no spectral-radius function |
| frozen `V3_substrate_selection.json`, `M5_structural_dynamics.json`, `ledger_repair_E9.json` | **yes, via the frozen defect** | `INVALIDATED_BY_MEASUREMENT_BUG` pending its own recomputation |

Historical outputs were **not overwritten**. Each suspect file is listed in the report with
its reason and a `superseded_by` pointer.

**One frozen conclusion is directly threatened.** `R2.spectral.rho.note` asserts that
*preserving the global weight multiset (C4.2) and rescaling to a target radius are mutually
exclusive*. That note was a conclusion drawn from measurements made with the defective
solver. The tension is real — §5 below re-derives it from exact arithmetic — but the frozen
note's specific numbers are not trustworthy and should be recomputed before being cited.

## 4. Conclusion survival: A4 is NOT_DIRECTIONAL, and it never depended on ρ

The A4 ladder was rebuilt from the original seeds and the original wirings. Recomputed:

| family | monotone in ρ | observable @ ρ=0.95 | directional |
|---|---|---|---|
| C1 ring+chords | no | **−0.1237** | no |
| C2 coprime cycles | no | **+0.0493** | no |
| C3 single 2-cycle | no | **−0.1610** | no |
| C6 20-cycle | no | **−1.0235** | no |

```
VERDICT: NOT_DIRECTIONAL  (0/4 recomputed families)
```

**The observable values are bit-identical to the pre-fix run** (−0.1237, +0.0493, −0.1610,
−1.0235 — the same four numbers recorded before the fix). That is not a coincidence to be
enjoyed but a fact to be explained: the A4 observable is a lagged-input correlation of the
driven state, and on this probe it is **insensitive to ρ**. The ladder was structurally
incapable of moving it. So the A4 retirement stands on two independent grounds — the
observable is non-directional, *and* it does not respond to the recurrence knob the ladder
varies.

A3's semantic observations are `UNAFFECTED` by construction (§3): the motivating
`D_eff ≈ 4.7–7.1` against a gate of 10 comes from `krylov_score` on the frozen substrate,
which never consults ρ.

## 5. Amendment 4 is required, and is NOT written yet

The F2/F3 ρ-vs-weight conflict is real and is settled by exact arithmetic rather than by
the defective measurements. For a fixed wiring with raw radius `ρ_raw`, preserving F1's
weights exactly forces a scale `α_F1 = 0.95/ρ_F1`, giving the candidate a realized radius
`0.95·ρ_cand/ρ_F1 ≠ 0.95`; matching `0.95` exactly forces a different scale and breaks
weight equality.

The review's resolution — **weight multiset matched up to ONE global normalisation scalar**,
i.e. keep the relative weights and multiply the whole family by `α = 0.95/ρ_raw` — makes
dynamical scale exact and identical across families while leaving the weight *shape* and the
topology contrast untouched. That is the better definition, and it supersedes the
"weight multiset wins, 0.64 % ρ drift permitted" rule currently recorded in the contrast
matrix.

**It is not applied in this round.** It changes a held-fixed semantic in the contrast matrix,
so per the amendment discipline it needs **Amendment 4**; a generator note is not sufficient
authority. Amendment 4 must also correct §7.2 of `docs/resaudit_contrast_matrix.md`, which
currently records the superseded rule. Awaiting the operator's approval to write it.

## 6. What stays blocked, and what is next

* **F5 stays blocked.** What is established is that the *current* coprime-block construction
  cannot guarantee A3 once `N`, `m` and `ρ` are frozen — not that no F5 is possible.
  Rebuilding it now, re-tuning chord weights, or lowering the effective edge budget would be
  positive-control engineering. Unless an analytic or fully pre-declared construction can be
  stated that needs no search over F1 or over A3 scores, no F5 is preferable to a
  manufactured one.
* **F4's `trace(A^k) > 0` test is deprecated.** It has no discriminating power at this
  density. The replacement must be a **quantitative recurrence profile** — closed-walk
  statistics `[tr(A²), tr(A³), tr(A⁴), …]` plus an SCC/feedback summary — with a **frozen
  tolerance**, and it must be demonstrated to both PASS and FAIL on calibration/stand-in
  inputs before F4 is reconsidered. If no construction satisfies it, `F4 = OMITTED`.
* **The 60-node stand-in is not a scientific result.** It validated code paths and
  constraints only. Real Stage 1 requires materializing the data root behind
  `A_hash 3aa95745…` and `B_hash 3bd78eac…` and verifying those hashes. Only then does
  F1/F2/F3 Stage 1 produce science rather than infrastructure.
