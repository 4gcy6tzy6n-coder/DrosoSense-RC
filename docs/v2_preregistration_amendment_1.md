# v2 pre-registration — AMENDMENT 1 (SIGNED): the measurement object of C4

**Signed 2026-09-23, before any formal experiment.** It amends
[`docs/v2_preregistration.md`](v2_preregistration.md) §4 (D6, criteria C4.1–C4.8) and
nothing else. The pre-registration is not edited: this file is the amendment, per the
project's rule that an amendment ADDS a version and never edits a frozen one.

**Why it exists.** C4 was measured and **failed**, and the failure was traced to the
criterion rather than to the algorithm — with the diagnosis in
[`docs/v2_construct_c4_r2_counterfactual.md`](v2_construct_c4_r2_counterfactual.md):

- C4.2/C4.3/C4.4 are stated over "the graph", but the reservoir never sees the graph. It
  sees the matrix after the declared normalization, and `n1_pre_l1` divides each source's
  outgoing weights by that source's total outgoing mass — so the normalized weights are a
  **function of the wiring**. Preserving the *normalized* weight multiset therefore pins
  the wiring: measured, the swap space collapses to **~5 edges per weight class**, the
  overlap stalls **flat at 0.858** across 4.3 M swaps, and the 600 s budget is spent.
- C4.6's "fully-mixed expectation" is substrate-dependent. Measured with the declared
  configuration-model estimator: **0.0570** on the v1 substrate C4.6 was calibrated on,
  **0.2022** on the C2 substrate — i.e. at the absolute 0.20 threshold.

Both are **measured degenerations of the criterion on the substrate the other criteria
forced**, not a wish for a pass. C2.3–C2.5 pass precisely because the C2 substrate is 34×
denser than v1's (mean out-degree 80.4 against 2.3), and density is what raises the mixing
floor.

---

## A1.1 The measurement object is the biological graph

**Declared.** The R2 counterfactual is built on the **raw synapse-count graph** of the
selected substrate, and the declared normalization is applied to **both** graphs
identically, with **R0's scale factor** shared by both.

**Consequences, all measured**
(`results/audit/v2_construct/C4_r2_counterfactual.json`):

| quantity | value |
|---|---|
| directed in/out degree sequences | **exact** (`max|Δ| = 0`, hashes equal) |
| raw global weight multiset | **exact** (hashes equal over 80,443 edges) |
| raw per-source outgoing weight multiset | **exact** (hashes equal) |
| raw in-strength median relative error (C4.4) | **0.0** |
| overlap (C4.6) | **0.19998**, reached in **13.95 s** with 574,532 swaps |
| normalized per-source **out**-strength | **exact** (row normalization + preserved multisets) |
| normalized **in**-strength median relative difference | **0.455** — reported, NOT gated |
| spectral radius R2/R0 | **0.933** |

**Two disclosures this amendment requires on every R2 line.**

1. **The normalized in-strength profile differs by ~45 % (median) between R0 and R2**, and
   that difference is the *wiring's own signature*: dividing each source's weights by that
   source's outgoing mass turns "how much weight arrives at a node" into "what share of
   each source's output arrives there", which rewiring changes by definition. It is
   reported as a diagnostic; the gated in-strength criterion (C4.4) is measured on the raw
   object, where it is exactly 0.
2. **R2's spectral radius is not matched** (ratio 0.933). Matching the radius and sharing
   one weight scale are mutually exclusive; the family convention (protocol §17) matches
   the radius, and the alternative is measured and reported beside it —
   `rho_matched_alternative.in_strength_median_relative_error = 0.453`, i.e. the same
   in-strength difference with the radius matched instead of the scale.

## A1.2 C4.6 becomes floor-relative

**Declared.** With `E[overlap]` the configuration-model expectation for a random graph with
R0's directed degree sequences,

```
E[overlap] = (1/M) · Σ_{(u→v) ∈ E_R0} (out_u · in_v) / M          (computed from R0 alone)
overlap    ≤ max(0.20, 1.10 · E[overlap])                          (C4.6, amended)
```

The absolute 0.20 **stays operative wherever the substrate's floor is below it** — a sparse
substrate must still mix to 0.20, and this amendment must not make that easier — and the
relative form takes over only where the floor reaches it (0.2224 on the C2 substrate).

**Discrepancy, recorded.** The option text put to the owner wrote `min` while describing
`max` semantics. `min` would have *tightened* the criterion on sparse substrates (v1's
floor of 0.0570 would have demanded 0.0627 instead of 0.20), the opposite of the approved
intent, so the implementation follows the described semantics. **No past verdict changes:**
both forms accept v1's measured 0.058 against its floor of 0.0570. Checked against v1: the
amended criterion would not have altered that result.

## A1.3 What the amendment does NOT change

- **C4.1, C4.3, C4.5, C4.7, C4.8** are unchanged, including their exactness.
- **The conservations are not weakened**: all four are still exact equalities, published as
  hashes over sorted values.
- **C4.2 is not reinterpreted away**: it still demands the *global* weight multiset exactly,
  and it is still the criterion that v1's R2 fails by construction (uniform weights).
- **No threshold was moved to fit a result.** C4.6's absolute 0.20 is retained as the
  operative value everywhere it is reachable; the amendment adds a floor for substrates
  where it is not, and the floor is computed from R0 alone, before any chain runs.
- **explicitly not done**: relaxing C4.6 to whatever the chain happened to reach, or
  dropping C4.2/C4.3 so the first (failing) design would pass.

## A1.4 Gate result after the amendment

```
degree_exact                 True      PASS
global_weights_exact         True      PASS
per_source_weights_exact     True      PASS
input_population_same        True      PASS
median_in_strength_err       0.0       PASS   (<= 0.05)
edge_overlap                 0.19998   PASS   (<= 0.2224, = 1.10 x E[overlap])
build_time                   13.95 s   PASS   (<= 600 s)
mixing_declared              True      PASS
C4                           PASS
```

Mixing curve (accepted swaps → overlap): `1.0000 → 0.4340 → 0.2710 → 0.2094 → 0.2065 →
0.2000`, i.e. a monotone approach to the target rather than the flat line the first design
produced. Acceptance rate 0.997; deterministic on repeat; the v1 control's weight multiset
is still a constant (`global_weight_multiset_preserved = False`), which is the defect D6
repairs.

**What C4 passing does and does not licence.** It says a fair wiring-only counterfactual
exists and is measurable. It says nothing about biology, and no formal inference has run.
C3 (`R_t`, memory benefit, effective rank) may now start — it needs a valid R0 **and** a
valid R2, which is exactly what this amendment closes.
