# v2 construct phase — C4, R2 as a wiring-only counterfactual

**Status: IMPLEMENTED AND MEASURED — AND C4 NOW PASSES, UNDER
[AMENDMENT 1](v2_preregistration_amendment_1.md) (signed by the owner).**

The first measurement FAILED, and the failure was a specification problem rather than an
algorithm defect: §2–§4 below are that diagnosis, kept because it is what produced the
amendment. §1 states the final, passing result.

The conservations hold exactly. What fails is **mixing** (C4.6 + C4.7), and the cause is
now measured precisely: the weight multiset R0 carries is the **normalized and rescaled**
one, and that destroys the raw graph's quantization, so an exact-weight rewrite of it has
almost no swap partners. The counterfactual is therefore caught by C4.6 doing exactly
what C4.6 exists to do — telling us the graph was not really rewired — while the two
designs that *would* mix each break a different criterion.

Artefacts: [`results/audit/v2_construct/C4_r2_counterfactual.json`](../results/audit/v2_construct/C4_r2_counterfactual.json)
(the gate run), `results/audit/v2_construct/C4_design_diagnostic.txt` (the design comparison).
Implementation: [`drososense/reservoir/r2_counterfactual.py`](../drososense/reservoir/r2_counterfactual.py).
No model was fitted, no test split was read, no formal inference ran.

---

## 1. The final gate result (after amendment 1)

`ops/audit/v2_construct_check.py --only C4 --din 5 --target-n 1000 --c4-time-budget 600`:

```
degree_exact                 True      PASS
global_weights_exact         True      PASS
per_source_weights_exact     True      PASS
input_population_same        True      PASS
median_in_strength_err       0.0       PASS     (<= 0.05)
edge_overlap                 0.19998   PASS     (<= 0.2224 = 1.10 x E[overlap])
build_time                   13.95 s   PASS     (<= 600 s)
mixing_declared              True      PASS
C4                           PASS
swaps accepted=574,532 attempted=576,001 overlap=0.2000 seconds=13.95 stopped=target_overlap
```

Mixing curve (accepted swaps → overlap): `1.0000 → 0.4340 → 0.2710 → 0.2094 → 0.2065 →
0.2000` — a monotone approach to the target, against the **flat** `0.8585` line the first
design produced (§2). Acceptance rate **0.997**; deterministic on repeat.

The amended measurement object is the **raw synapse-count graph** of the selected
substrate, with the declared normalization and R0's scale applied identically to both
graphs (amendment 1, A1.1). Disclosed with it: the normalized **in**-strength profile
differs by 0.455 (median) between R0 and R2 — the wiring's own signature, since normalizing
by each source's outgoing mass turns "how much arrives" into "what share of each source's
output arrives" — and R2's spectral radius is 0.933× R0's, because a shared weight scale
and a matched radius are mutually exclusive (the alternative is measured beside it at 0.453
in-strength difference). The normalized per-**source** out-strength is **exact**.

The pre-amendment failure, kept for the record:

```
degree_exact                    True      PASS
global_weights_exact            True      PASS
per_source_weights_exact        True      PASS
input_population_same           True      PASS
median_in_strength_err      7.7215e-08    PASS
edge_overlap                  0.858459    FAIL     (threshold 0.20)
build_time                    600.038     FAIL     (threshold 600 s)
mixing_declared                 False     FAIL
C4                                        FAIL
swaps accepted=4,345,068 attempted=14,704,001 seconds=600.04 stopped=time_budget
```

The four conservations are **exact**, as hashes over sorted values:
in/out degree-sequence hashes equal, global weight-multiset hashes equal, per-source
outgoing weight-multiset hashes equal, and the ORN-aligned input mapping built
independently on R0 and on R2 gives identical support rows and an identical `w_in` digest.
The v1 control is measured on the same R0 beside it: `make_degree_rewired` writes
`np.ones(n_edges)` and rescales, so **v1's R2 fails C4.2 and C4.3 by construction** — the
defect D6 exists to repair, now pinned by a test rather than left in prose.

## 2. Why the chain cannot mix, in one number

C4.6 was written against the v1 substrate, quoting v1's `0.058` at N=1000 against a
"fully-mixed expectation" of `0.055`. **That expectation is not a constant.** For a
uniformly random graph with the same directed degree sequences,

```
E[overlap] = (1/M) · Σ_{(u→v) ∈ E_R0} (out_u · in_v) / M
```

| substrate | M | mean out-degree | **E[overlap]** | C4.6 threshold |
|---|---:|---:|---:|---:|
| v1 selection (what C4.6 was calibrated on) | 2,343 | 2.3 | **0.0570** | 0.20 |
| **C2 expansion (what v2 must use)** | **80,443** | **80.4** | **0.2022** | 0.20 |

C2.3–C2.5 pass *because* the substrate is 34× denser than v1's, and density is exactly
what raises the mixing floor. Independently confirmed by an edge-shuffle redraw with the
same degree sequences: 0.1692 on the C2 substrate, 0.0531 on v1's.

**And the binding constraint is tighter than that floor.** The weight multiset R2 must
preserve is R0's **normalized + rescaled** weights, in which the raw quantization is gone:

| weight object | edges | distinct values | mean class | edges with an exact-weight partner |
|---|---:|---:|---:|---:|
| raw synapse counts | 80,443 | **363** | **221.6** | **99.87 %** |
| R0's matrix (after `n1_pre_l1` + spectral rescale) | 80,443 | **15,277** | **5.3** | **90.2 %** |

With (i) the weight staying with its source (C4.3) and (ii) exact-weight partners required
for in-strength to be preserved at all (C4.4), each edge's reachable target set is the
weight class it belongs to — a **5.3-edge** class on average. And the mixing curve shows
the chain is **trapped, not slow**: sampled at 0.5·|E|, |E|, 2·|E|, 5·|E|, 10·|E| accepted
swaps it reads

```
1.0000 → 0.8608 → 0.8599 → 0.8597 → 0.8605 → 0.8592 → 0.8585   (4,345,068 swaps, 600 s)
```

flat to within 0.003 across 4.3 million accepted swaps. It reaches its floor within the
first 0.5·|E| swaps and then does nothing, which is what a 5.3-edge class means: there is
no room to rewire inside it. (Contrast design B, whose curve descends monotonically —
`1.000 → 0.711 → 0.664 → 0.612 → 0.535 → 0.468 → 0.351` — because a ±25 % partner pool is
large; it pays for that with in-strength drift.)

## 3. The three designs, measured

Every number below is from a run on the delivered connectome; the full transcript is in
`results/audit/v2_construct/C4_design_diagnostic.txt`.

| | **A** rewire R0's normalized matrix, exact weights | **B** same, near weights (±25 %) | **C** rewire the RAW quantized matrix, then preprocess both identically |
|---|---|---|---|
| degree sequences | **exact** | exact | **exact (raw)** |
| global weight multiset | **exact** | exact | **exact (raw)**; differs after normalization |
| per-source out-weight multiset | **exact** | exact | **exact (raw)**; differs after normalization |
| in-strength median rel. error | **7.7e-08** | **0.318** ✗ | **0.495** ✗ (normalized pair) |
| edge overlap | **0.858** ✗ | 0.351 ✗ | **0.19998** ✓ |
| build time | **600 s** ✗ | 311 s (still descending) | **14.4 s** ✓ |
| accepted swaps | 4,345,068 | 3,105,655 | 574,532 |

Design C, measured on the RAW object, satisfies **every conservation exactly** and reaches
the mixing target in **14.4 s** — because on the raw graph the weight classes hold 221.6
edges on average instead of 5.3, so exact-weight swaps have partners.

**Where design C costs something, stated precisely.** Rewiring changes the column sums, so
after the declared normalization the two matrices no longer share a weight multiset
(median relative weight shift 0.132) and, because each is rescaled to the same spectral
radius, their total weight differs — which is why its in-strength error on the *normalized*
pair is 0.495 rather than 0. **Both of those are consequences of measuring on the
normalized object.** With one additional declared rule — apply R0's rescale factor to both
graphs, so the two share a scale — the normalized in-strength is preserved exactly, at the
cost of R2's spectral radius differing from R0's (which design A already reports: ratio
0.955).

## 4. The decision (owner) — RESOLVED

The signed rule is that a failed criterion is reported, never relaxed, and that relaxing
one after seeing a result is a **protocol amendment**. So this is put to the owner. The
question is a *definition* question: **C4.2/C4.3/C4.4 (and C4.6's floor) are stated over
"the graph", but the reservoir never sees the graph — it sees the normalized, rescaled
matrix, and normalization is a function of the wiring.**

- **Reading R1 — the criterion is about the matrix the reservoir uses.** Then design A is
  the only admissible design, and it fails: preserved weights leave 5.3-edge classes, the
  chain cannot mix, and C4.6 correctly refuses to call the result a rewiring. **Under this
  reading the v2 phase cannot proceed past C4 on the C2 substrate** — the construct, not
  the algorithm, is what fails.
- **Reading R2 — the criterion is about the biological graph; the normalization is
  preprocessing applied identically to both graphs.** Then design C (+ the common-scale
  rule) satisfies C4.1–C4.7 simultaneously: exact raw conservation, exact normalized
  in-strength, mixing 0.200 in 14.4 s. The price is a disclosure: the *normalized*
  per-source weight multisets differ, because the columns are wired differently.

**Signed by the owner: Reading R2**, together with the floor-relative C4.6. The amendment
that records it, with its measured consequences and its two required disclosures, is
[`docs/v2_preregistration_amendment_1.md`](v2_preregistration_amendment_1.md). The analysis
that produced it follows.

**Recommendation: Reading R2, declared as a measurement-object clarification.** It matches
D6's own words ("same directed degree sequence / same global weight multiset / same
per-source outgoing weight multiset" — the graph's weights, not a derivative of them),
and it matches the defect D6 repairs (`np.ones(n_edges)`: the weights destroyed *at the
source*). It is also the only reading under which a counterfactual that satisfies all of
C4.1–C4.7 exists at all, and it comes with a checkable disclosure rather than a caveat.

**A second, separable question — also resolved (floor-relative C4.6).** C4.6's absolute 0.20 is below the C2 substrate's own
mixing floor (0.2022). Even with design C, a future denser substrate would hit this again.
Making C4.6 floor-relative — `O ≤ min(0.20, 1.10 × E[overlap])`, with `E[overlap]` computed
by the declared estimator above — would keep the criterion's intent (*has the chain
mixed?*) and be substrate-independent. Checked against v1: `E = 0.0570`, measured `0.058`,
ratio **1.02** → it would not have changed v1's verdict. Under design C the measured 0.200
already passes the absolute form, so this is a robustness amendment, not a rescue.

**What is NOT proposed:** relaxing C4.6 to whatever the chain reached, or dropping C4.2/
C4.3 to make design C pass. Both would be exactly the "relax it after seeing the result"
move the discipline forbids.

## 5. Two implementation defects found by measuring (both fixed)

**5.1 In-strength drift from the near-weight fallback.** (Design B's 0.318 above.) The first chain let a proposal
without an exact-weight partner fall back to weights within 25 %, and 11.2 % of proposals
did. Because the weight stays with its source, a swap makes the two *targets* exchange
weights, so the drift accumulated over millions of swaps: median relative error **0.318**.
The chain now refuses such a proposal by default; the fallback is opt-in and the report
states how many *accepted* swaps exchanged different weights. In-strength error is now
exactly 0 (design A) — which is what let the failure be attributed to mixing rather than
to conservation.

**5.2 61 % of proposals were wasted.** Measured: 9.78 M of 16.1 M proposals were discarded
because the two edges shared an endpoint and 1.55 M because the swap would duplicate an
edge. The proposal now examines up to 16 candidates from the weight class and takes the
first *legal* one — acceptance conditions unchanged, every illegal candidate still counted.
On a synthetic 80k-edge substrate the acceptance rate goes 0.19 → 1.00 and the chain
reaches 0.194 overlap in 0.6 s instead of 311 s. This mattered: C4.7's budget is part of
the construct, so a chain that "would" mix given an hour is a FAIL, and the earlier run
could not distinguish *slow* from *cannot*.

## 6. Not claimed

- **No formal inference has run**, and none may run while a construct criterion fails.
- **C4's failure is not evidence about biology.** It is a statement about a threshold, a
  substrate density, and a measurement object.
- **The type-pair matrix is a diagnostic, not a gate.** R2 preserves node identities,
  degree and source weights but not ORN→PN / PN→KC type-level connectivity — that is what
  a wiring-only counterfactual means. Making R2 type-pair preserving would change the
  frozen primary counterfactual; it belongs to a separately pre-registered secondary.
- **R2 does not rescale** (design A). Both radii and their ratio (0.955) are reported.
