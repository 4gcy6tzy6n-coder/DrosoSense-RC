# ResAudit-Food — pre-registration amendment 6: exact ρ, weight matching up to one global scalar, and the frozen Din apportionment

**Status: amendment, 2026-09-23. Written AFTER the measurement-layer impact audit and BEFORE
any real Stage-1 family result.**

**The original pre-registration is immutable**, and so is every earlier amendment. This file
edits nothing; it is added. (Its number is 6, not 4, for the concurrency reason recorded in
`docs/resaudit_amendment_numbering_note.md`.)

No threshold of A1–A5 is changed here. No family is admitted or excluded. No food data is read.

## 1. Exact ρ for every dynamical family

> **`rho(A) = 0.95` EXACTLY for every dynamical family.**

The previous rule — that F2/F3 share F1's single weight scale, accepting a measured 0.64 % ρ
drift — is **superseded**. That rule was written before the spectral-radius defect was
understood, and it traded away the dynamical scale to buy absolute weight equality. A3's
verdict is normalized-scale dependent (amendment 1 §1), so the scale is the invariant that
must hold exactly, and the weight claim is the one that narrows.

## 2. Weight structure matched up to ONE global scalar

> **Weight structure is matched up to a single global normalization scalar.**

If the reference weight multiset is `W = {w_1, …, w_m}`, then a family is permitted

```
W' = {alpha * w_1, …, alpha * w_m},      alpha = 0.95 / rho(A_raw)
```

and **not** a claim that the released weights are absolutely identical to the reference's.

What this preserves:

| property | preserved |
|---|---|
| weight ranking | yes |
| pairwise weight ratios | yes |
| distribution shape | yes |
| topology-independent dynamical scale | **yes, exactly** |
| absolute released weight values | no — by construction, and no longer claimed |

The amended semantics replace the previous "weight multiset preserved exactly" held-fixed
entry in the contrast matrix. `docs/resaudit_contrast_matrix.md` §7.2, which records the
superseded rule, is corrected by this amendment.

Implementation: `resaudit/contracts.py`
(`global_normalisation_scalar`, `weights_up_to_global_scalar`).

## 3. Frozen Din apportionment (typed-alignment extension)

`Din = 6` and `Din = 8` could not be built, because
`DEFAULT_TYPED_ALIGNED_DIN_PER_LAYER = {ORN:2, PN:2, KC:1}` sums to 5 and the rule declared
no allocation for any other width. Filling a table per width by hand would be a result-adjacent
choice; instead the frozen ratio is extended by a deterministic rule.

> **Rule.** Apportion `Din` channels across `ORN:PN:KC` by the frozen ratio `2:2:1` using
> **largest-remainder (Hamilton) apportionment**. Compute the quotas `Din * (2,2,1)/5`, take
> the floors, and award the remaining channels in descending order of fractional remainder.
> Ties break by the permanently fixed order **`ORN -> PN -> KC`**.

> **The allocation is a deterministic extension of the previously declared 2:2:1
> typed-alignment ratio, not a result-dependent input search.**

Verified output (`resaudit.contracts.apportion_din`):

| `Din` | allocation |
|---:|---|
| 1 | 1/0/0 |
| 2 | 1/1/0 |
| 3 | 1/1/1 |
| 4 | 2/1/1 |
| **5** | **2/2/1** ← reproduces the frozen allocation exactly |
| **6** | **3/2/1** |
| 7 | 3/3/1 |
| **8** | **3/3/2** |

`Din = 5` reproducing `2:2:1` is the consistency requirement, and it is met.

**What is NOT permitted, and is fixed by this amendment:** the neuron selection for `B`, the
seed, and the within-layer sampling rule stay exactly as declared. `Din = 6` and `Din = 8`
may **not** each choose a different set of neurons — no per-width neuron search, and above all
no search over which neurons make A3 pass.

## 4. `A_hash` retired as a provenance identity

The historical `A_hash` was produced along a path that normalises with an **unseeded** and
additionally **incorrect** spectral-radius estimate (`eigsh` applied to a non-symmetric
directed matrix; see `docs/resaudit_rho_impact_audit.md` §1). It is therefore
**non-identifying**: it can differ between two builds of the same graph.

> **`old A_hash = non-identifying historical field.`**

F1's identity is taken instead from its content:

```
node-list hash  +  edge-list hash  +  deterministic build specification
                +  canonical adjacency content hash
```

The canonical content hash is computed over a canonicalised CSR serialisation —
`(n_rows, n_cols, nnz, dtype)` followed by `indptr`, `indices`, `data` in a fixed little-endian
layout, after sorting indices, summing duplicates and dropping explicit zeros. It involves **no
spectral-radius estimator, no normalisation and no floating-point accumulation order**, so it is
reproducible across builds and machines.

Implementation: `resaudit/contracts.py` (`canonical_adjacency_hash`,
`canonical_adjacency_bytes`). The function additionally refuses to hash a matrix containing
non-finite weights, because NaN would let two runs disagree without either being wrong.

## 5. What this amendment does NOT do

* It does **not** unblock F4. F4 remains blocked; its `trace(A^k) > 0` term is deprecated as
  non-discriminating, and any replacement must first be shown to both PASS and FAIL on
  calibration/stand-in inputs.
* It does **not** rescue F5. The current coprime-block construction is recorded as a **failed
  attempt** at `N`, `m`, `rho` frozen jointly. The effective edge budget is not lowered, chord
  weights are not re-tuned, and no search over A3 scores is permitted.
* It does **not** license a real Stage-1 run by itself. The frozen studies affected by the
  directed-matrix spectral-radius defect must be recomputed first (§6).
* It does **not** touch food evaluation, which stays on **HOLD**.

## 6. Outstanding before the first real Stage 1

Recompute, with the repaired measurement layer and the **original configurations, seeds and
inputs** — never overwriting the originals:

| historical artifact | why in scope |
|---|---|
| `results/audit/v3_selection/V3_substrate_selection.json` | `rescale_to_spectral_radius` / `constraint_match` ρ terms |
| `results/audit/m5_structural_dynamics/M5_structural_dynamics.json` | mirrors `rescale_to_spectral_radius` |
| `results/audit/m4_audit/ledger_repair_E9.json` | ledger repair over ρ-scaled topologies |

Each must emit `historical_result` with
`status = INVALIDATED_BY_SPECTRAL_RADIUS_DEFECT` and a `replacement_result` carrying
`supersedes = <historical id>`. **If any historical conclusion changes direction, it is
escalated as a scientific correction, not filed as an implementation fix.**

## 7. Authorised Stage-1 scope once §6 is complete

```
F1, F2, F3        permitted
F4                blocked
F5                blocked (current construction failed)
food evaluation   HOLD
```
