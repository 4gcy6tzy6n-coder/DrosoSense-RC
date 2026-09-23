# ResAudit-Food — pre-registration amendment 6: extension of the typed-aligned input allocation to the preregistered widths

**Status: amendment, 2026-09-23. Written BEFORE B at `Din=6` and `Din=8` was built and BEFORE any
Stage-1 family score existed.**

**The original pre-registration is immutable**, and so is every earlier amendment. This file edits
nothing; it is added.

## 1. The decision

> The frozen typed-aligned mapping originally specified the allocation `{ORN:2, PN:2, KC:1}` only at
> `Din=5`. For the preregistered native widths `Din ∈ {6, 8}`, the allocation is extended
> deterministically by preserving the frozen 2:2:1 layer ratio and applying Hamilton / largest-remainder
> apportionment, with the fixed tie order **ORN > PN > KC**. This yields `(3,2,1)` at `Din=6` and
> `(3,3,2)` at `Din=8`. **No reservoir score, food label, downstream performance, or audit outcome may
> alter this allocation.**
>
> At `Din=5`, the extension must reproduce the historical mapping **bit-for-bit**; failure to reproduce
> it blocks Stage 1.

Operator decision this amendment records: *"Approved option 1. Extend the typed-aligned mapping with
the frozen 2:2:1 ratio + largest-remainder + fixed ORN>PN>KC tie-break; Din=5 must reproduce exactly,
Din=6/8 are fixed at 3:2:1 and 3:3:2. Then materialize B6/B8 and enter Stage 1, without reopening the
input geometry."*

## 2. Why this option, and not the alternative

The rejected alternative was to use the C1 ORN-aligned mapping, which does accept `Din ∈ {5,6,8}`
natively. It was rejected because it would replace amendment 2's declared **typed-aligned ORN+PN+KC
input geometry** with a different one. Stage 1 would then vary two things at once — `Din` *and* the
input support — which is exactly what the audit is supposed to hold fixed. Measured difference:
support 473 nodes at density 0.0946 under the typed-aligned rule, versus 243 ORN-only nodes at
density 0.0486 under the ORN-aligned rule at the same width.

## 3. The rule, which is what is frozen

The three resulting allocations are **consequences** of the rule, not constants. The rule is what a
reviewer must be able to check, so that `(3,2,1)` and `(3,3,2)` cannot look like an allocation chosen
to make `Din=6`/`8` clear A3.

```
weights        w = (ORN:2, PN:2, KC:1)          frozen, unchanged from the Din=5 default
total weight   W = 5
quota          q_l(D) = D * w_l / W
floor          n_l = floor(q_l)
remainders     r_l = q_l - n_l
surplus        D - sum(n_l) channels, assigned in descending remainder order
tie-break      remainder equal -> ORN > PN > KC            fixed, not by layer size or node count
```

Evaluation:

| `D` | quotas `q_l` | floors | remainders | surplus | **allocation** |
|---|---|---|---|---|---|
| 5 | 2.0, 2.0, 1.0 | 2, 2, 1 | .0, .0, .0 | 0 | **(2,2,1)** |
| 6 | 2.4, 2.4, 1.2 | 2, 2, 1 | .4, .4, .2 | 1 | **(3,2,1)** — surplus to ORN |
| 8 | 3.2, 3.2, 1.6 | 3, 3, 1 | .2, .2, .6 | 1 | **(3,3,2)** — surplus to KC |

`Din=5` reproduces the frozen default exactly, which is why this reads as a generalisation of the
existing rule rather than a replacement of it.

## 4. What is explicitly NOT changed

| Item | Effect |
|---|---|
| `Din=5` typed-aligned mapping | **none — must be bit-identical**, and failure to reproduce it blocks Stage 1 |
| A1–A5 thresholds | **none** |
| The probe / drive protocol (amendments 1 §0.1, 3) | **none** |
| `gain`, `input_scale`, `density_limit`, `receiving_fraction` | **none** |
| The family selection algorithm, F1's node set, edge set, weights | **none** |
| The typed-aligned builder's channel ordering | **none — see §5** |
| Candidates admitted or excluded | **none** |

## 5. Nestedness is a desirable invariant that is NOT pursued

Amendment 4 §7 asked whether `B_6` could be a prefix of `B_8` (`B_6 == B_8[:, :6]`). It is recorded
here as **not required**, and **no second algorithmic change may be introduced to obtain it**: the
typed-aligned builder assigns each layer a contiguous channel window in the order ORN, PN, KC
(channels `[0, n_ORN)`, then PN, then KC), so the window boundaries shift when the allocation
changes, and nesting does not hold in general. Since the allocation change already alters the
per-layer widths, the prefix property cannot be recovered without re-ordering channels, which is a
change to the frozen mapping that the operator expressly declined.

What must hold instead, per width `D`:

- the allocation is **uniquely determined** by the rule in §3 (no fallback, no dynamic assignment);
- the **seed is unchanged** (`seed = 0`, the frozen selection seed);
- the **node-selection rule is unchanged**;
- the **input scale is unchanged**;
- **no family score may influence** the allocation.

## 6. Pre-conditions for advancing F1's status to `VERIFIED_FOR_STAGE1`

Four checks, all required, run before any family is measured:

```
1.  Din=5  -> allocation (2,2,1)  AND  B_hash == 3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968
2.  Din=6  -> allocation (3,2,1),  no fallback, no dynamic allocation
3.  Din=8  -> allocation (3,3,2),  no fallback, no dynamic allocation
4.  each width, materialized twice -> B^(1) == B^(2) byte-identical
```

On success, F1's materialization status advances from `PARTIAL` to **`VERIFIED_FOR_STAGE1`**. The
legacy stochastic `A_hash` mismatch is **not** carried as an open problem: per amendment 5 it is a
property of a legacy run's scaled-float digest and not part of substrate identity, and substrate
identity is established by the deterministic content proof — **node-list hash + edge-list hash + raw
weights/topology provenance**. Amendments 5 and 6 close it; the defect's scope is not widened.
