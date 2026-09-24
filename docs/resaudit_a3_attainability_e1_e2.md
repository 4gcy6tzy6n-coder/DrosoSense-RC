# E1 + E2 — A3 attainability ensemble and family replication (A3 line)

**Status: COMPLETE 2026-09-23. Case B. A3's discriminative validity fails at the frozen gate.**

> **Naming note.** These E1/E2 are the **A3 attainability** experiments of the ResAudit gate
> line. They are unrelated to `docs/resaudit_e1_e2_findings.md`, which reports a different
> E1/E2 series (wine/tea evaluation-leakage audits). The labels collided because two
> workstreams number their experiment series independently. This file and
> `resaudit_a3_attainability/` are the A3 line's.

Runner: `ops/audit/resaudit_e1_e2_ensemble.py`
Report: `results/audit/resaudit_a3_attainability/E1_E2_ensemble.json`

Real materialized F1 throughout: **N = 1000, M = 80443**, read from
`results/audit/resaudit_stage1/f1/F1_A.npz`. **The toy stand-in's N=100, E=482 was not used
anywhere.** No food data was read. No A1–A5 threshold was changed.

---

## 1. The headline

```
410 distinct substrate instances, four construction families, two widths
A3 PASSES:  0
maximum fraction of the gate reached ANYWHERE:  0.8734
```

| ensemble | n | `D_eff/Din` mean | sd | range | gate frac (max) | pass rate |
|---|---:|---:|---:|---|---:|---:|
| E1 `F2` degree-preserving rewires (40/Din) | 80 | 1.375 / 1.380 | 0.013 | 1.349–1.411 | 0.706 | **0.000** |
| E1 `F3` directed `G(n,m)` (40/Din) | 80 | 1.225 / 1.263 | 0.003 | 1.220–1.266 | 0.633 | **0.000** |
| E1 directed configuration model (40/Din) | 80 | 1.216 / 1.230 | 0.008 | 1.190–1.237 | 0.618 | **0.000** |
| E1 `F5` coprime-block attempt, full scale (20/Din) | 40 | 1.376 / 1.351 | 0.011 | 1.325–1.397 | 0.698 | **0.000** |
| E2 `F2` frozen generator, 50 seeds (Din=5) | 50 | 1.130 | 0.004 | 1.123–1.141 | 0.571 | **0.000** |
| E2 `F3` frozen generator, 50 seeds (Din=5) | 50 | 1.431 | 0.013 | 1.405–1.456 | **0.728** | **0.000** |
| E2 `F5` frozen generator, 30 seeds (Din=5) | 30 | 1.310 | 0.012 | 1.287–1.331 | 0.666 | **0.000** |

The gate demands `D_eff ≥ 2·Din`. The **highest `D_eff/Din` observed across every family and
every width is 1.456**; the highest *gate fraction* is **0.873**.

## 2. F1 is not specially bad — which is the load-bearing observation

F1 sits at the **57.1st percentile** of the E1 ensemble at **both** widths. At Din=6 its ratio
(1.291) falls between the configuration-model family (1.216) and the degree-preserving rewire
family (1.375). Against the frozen generators it is likewise mid-distribution.

So Stage-1's *"F1 fails A3"* was **true but uninformative**. It is not a statement about the
connectome: F1 is an ordinary member of a population in which **nothing** reaches the gate.
What survives is a statement about the gate.

The per-family spreads are also very small (sd 0.002–0.015 on ratios of 1.2–1.4), so the
families sit in a tight cluster **far** from the gate. This is not a near-miss that more
samples, a luckier seed, or a slightly different construction would close.

## 3. Case B is confirmed

E1's two outcomes were fixed before the run:

* **Case A** — `0 < P(A3 PASS) < 1`: the gate discriminates, and the Stage-1 negative has force.
* **Case B** — `P(A3 PASS) ≈ 0`: the gate has little discriminative power here, and
  "F1 failed A3" cannot be the paper's main claim.

**`P(A3 PASS) = 0` over 410 instances → Case B.**

> Under the frozen Stage-1 conventions (ρ = 0.95, `leak = gain = 1.0`, `K = 16`, the frozen
> typed input geometry, the frozen A3 definition with its **absolute** singular-value
> tolerance), the gate `D_eff ≥ 2·Din` is attained by **none** of 410 task-blind candidates
> drawn from degree-preserving rewiring, directed `G(n,m)`, a directed configuration model,
> and a full-scale coprime-block positive-control construction, at either width.

This does **not** claim that no reservoir could clear it, and does **not** claim the threshold
is wrong in absolute terms. It claims the frozen gate does not separate substrates in the
region the framework operates in.

## 4. The F5 attempt was rebuilt at full scale, and still fails

Stage-1's attempt used 6 small coprime blocks and could not reach F1's edge budget (≈338 of
80,443 edges), so its failure was confounded with a construction that never reached scale. E1
rebuilds it properly:

```
block lengths (101,103,107,109,113,127,131,209)
  pairwise coprime: True      sum: exactly 1000
  in-block chord capacity Σ L(L−2) = 130,880  ≥  needed M − N = 79,443
```

Chords are enumerated exactly and subsampled, so the construction reaches `M = 80,443`
exactly and terminates in 0.14 s. It measures `D_eff/Din = 1.376` at Din=6 → **0.698 of the
gate**.

**F5's failure is therefore not an artifact of the earlier construction being unable to reach
scale. The scale-matched construction fails too.**

## 5. What changes, and what does not

**Changes — the primary claim.** "The audited battery failed the construct requirement" is no
longer defensible as the headline: E1 shows that formulation is uninformative because
*everything* fails it. The defensible primary result is:

> **The preregistered gate failed discriminative validation.** A gate whose pass rate is zero
> across a 410-instance ensemble spanning four construction families separates nothing, so a
> negative result stated against it carries no information about the substrates.

**Does not change — the Stage-1 record.** `FROZEN_NEGATIVE_RESULT` remains an accurate
description of what was *measured*; what changes is its **interpretation** — evidence about the
gate's reach, not about the connectome's adequacy. The `rule 6.4` record also stands: the food
stage is NOT RELEASED, because no family is `food_eligible`, and that follows from the
measurements however the gate is later judged.

**Does not license — a threshold change.** `2·Din` is not lowered, and E1 is not used to argue
that it should be. Per the pre-registration, a threshold is not moved because the data dislike
it; it is reported. `A3_FACTOR` remains 2.0 in `resaudit/criteria.py`, and no rescue route
listed in the rule 6.4 decision record is taken.

## 6. Consequence for E3–E7

E1 was run first because it could render the remaining experiments unnecessary or redirect
them. It redirected them.

| experiment | status after E1 |
|---|---|
| **E3 sensitivity grid** | **now highest value.** It asks *why* the gate saturates: `K` ∈ {8,16,32}, ρ ∈ {0.50…0.99}, τ = {0.01…100}×τ₀, β ∈ {0.25…4}. If saturation is driven by the **absolute** singular-value tolerance (a scale artifact) rather than topology, that is the mechanism behind Case B and becomes the paper's analytical core |
| E4 scale-invariant dimensions | pairs with E3: compare frozen A3 against participation ratio, squared-spectrum PR and entropy effective rank on the *same* spectra. Directly tests whether the saturation is the estimator's |
| E5 nonlinear construct validation | still needed for the Reviewer-2 objection, but now secondary: correlating a saturated instrument with nonlinear state richness is a question about the instrument |
| E6 post-hoc external validity | method unaffected, framing changes: a zero-variance gate verdict cannot be correlated with task performance — only the **continuous** `D_eff` can |
| E7 A4 replication | independent of all the above; proceeds on its own schedule |

**Not yet run: E3–E7.** Awaiting the decision on whether to proceed with the Audit-the-Audit
reframing or to halt the A3 line here.
