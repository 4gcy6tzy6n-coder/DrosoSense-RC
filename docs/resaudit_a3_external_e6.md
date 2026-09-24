# E6 — external computational validity of frozen A3, on synthetic tasks

**Status: COMPLETE 2026-09-23. Stage-1 untouched. `A3_FACTOR` unchanged. Food release
unchanged (NOT RELEASED). No threshold altered. No replacement gate defined; no winning
metric selected. No food data accessed.**

Runner: `ops/audit/resaudit_e6_external.py`
Report: `results/audit/resaudit_a3_external/E6_external.json`

Post-hoc exploratory external-validity study, exactly the 181-substrate pool of E5
(F1 ×1, F2 ×50, F3 ×50, configuration-model ×50, F5 ×30), primary ρ = 0.95 with a
robustness check at ρ ∈ {0.50, 0.95, 1.05}.

## Primary question, and the answer

> **Does A3 have external computational validity, and is it stable across operating regimes?**

**Yes for memory capacity and NARMA-10 — and both are sign-stable across ρ. Not established
for delayed XOR, because no substrate solves that task.**

| task | Spearman(A3, ·) at ρ=0.95, n=181 | 95 % CI | sign across ρ = {0.5, 0.95, 1.05} |
|---|---:|---|---|
| **`MC_total`** (delayed recall, τ=1..50) | **+0.874** | [+0.84, +0.90] | **+0.51 / +0.85 / +0.62 — stable** |
| **`NARMA_NRMSE`** (lower = better) | **−0.372** | [−0.50, −0.24] | **−0.32 / −0.42 / −0.32 — stable** |
| `XOR_3_7_acc` | +0.318 | [+0.18, +0.44] | +0.35 / +0.21 / +0.10 — weakening |
| `XOR_5_15_acc` | +0.190 | [+0.05, +0.34] | **−0.16 / +0.15 / +0.18 — FLIPS** |
| `MC_long` (τ=26..50) | **undefined** | — | identically 0 (see §3) |

## 1. Memory capacity: A3's strongest external association in the whole audit

`MC_total` mean 2.711, sd 0.349, range [2.051, 3.433]. F1 is the **best substrate in the
pool** on this task (3.433), ahead of F2 (3.023), F5 (2.936), F3 (2.762) and CM (2.197).

A3 tracks it at **+0.874**, the largest correlation measured anywhere in this project, and it
stays positive at every tested ρ.

**Circularity caveat, stated plainly.** A3 is a *linear* Krylov-block quantity and delayed
recall is a *linear* memory task, both governed by propagation under `A`. A high correlation
here is therefore partly mechanical — it is close to testing A3 against a task built from the
same linear structure. This is why NARMA and XOR matter more for the Reviewer-2 question.

## 2. NARMA-10: moderate, stable, and the more independent test

NRMSE ranges 0.810–0.837 across families, i.e. the reservoir carries real but limited
nonlinear memory. A3 correlates **−0.372** (higher A3 → lower error), sign-stable at all three
ρ values. Moderate in magnitude, but it is a *nonlinear* task, so this is the most meaningful
positive evidence for A3's nonlinear validity — and it is consistent with E5's finding that
A3 tracks linear memory capacity (MC +0.805 there).

## 3. `MC_long` is identically zero — a real limitation, not a missing number

Long-delay recall (τ ≥ 26) yields R² ≤ 0 for **every one of the 181 substrates** (with R²
clamped at 0). With zero variance the correlation is undefined, and the report records it as
NaN rather than a spurious zero.

The interpretation: under the frozen dynamics (`leak = gain = 1.0`, ρ = 0.95) the influence of
an input decays like ρ^τ, so recall beyond ≈25 steps is unattainable at this operating point.
`MC_total` ≈ 2.7 of a possible 50 (mean R² ≈ 0.054 per lag) is consistent with weak, short
memory. **Any claim about long-delay memory capacity is out of reach for this reservoir at this
operating point**, and cannot be correlated with anything.

## 4. Delayed XOR is unsolved, so it provides no evidence either way

| pair | accuracy range across families |
|---|---|
| (3, 7) | 0.506 – 0.523 |
| (5, 15) | **0.478 – 0.505** |

Chance for balanced XOR is 0.5. Group (5, 15) sits **at or below chance** for four of five
families. A correlation computed against a near-chance task measures readout noise, not
computational capacity — so the `XOR_5_15` sign flip across ρ carries no information, and the
`XOR_3_7` result (+0.318, weakening to +0.10) should be read as very weak at best.

## 5. The other descriptors on the same tasks (ρ = 0.95)

| descriptor | `MC_total` | `NARMA_NRMSE` | `XOR_3_7_acc` | `XOR_5_15_acc` |
|---|---:|---:|---:|---:|
| `frozen_A3` | **+0.874** | **−0.372** | +0.318 | +0.190 |
| `kernel_erank` | +0.407 | +0.344 | +0.213 | +0.261 |
| `state_entropy_rank` | −0.116 | **−0.523** | −0.138 | −0.253 |
| `state_PReff` | −0.126 | −0.374 | −0.076 | −0.284 |
| `state_separation` | **−0.535** | −0.162 | −0.180 | −0.269 |

Note the tension: `state_separation` is the **best** descriptor of NARMA-adjacent structure in
E5's positive direction yet **anti-correlates** with `MC_total` (−0.535); `state_entropy_rank`
is the strongest NARMA correlate (−0.523) while being near-zero on memory. No single
descriptor dominates across tasks — which is itself an argument against selecting any one of
them as a replacement. **Per instruction, none is selected.**

## 6. Answer, and the limits of it

**External computational validity is established for the linear memory task (strongly,
stably) and for NARMA-10 (moderately, stably); it is not established for delayed XOR, because
the task is not solved.** This is the most favourable evidence for A3 in the audit series — it
contrasts with E5, where three of five internal observables reversed sign across ρ, and with
E1/E3/E4, where the gate admitted nothing and ranked unstably.

**It does not rehabilitate the gate.** E6 measures the *continuous* `D_eff` against task
performance. It says nothing about whether the frozen threshold `2·Din` separates substrates —
E1 showed 0 of 410 clear it, and E3 showed the ordering is operating-point dependent. A
quantity can be task-predictive and still be a bad gate, and that is exactly the position the
evidence now supports:

> frozen A3 is a **real but mis-calibrated** instrument: it tracks memory-related capability,
> its threshold is unattainable, and its ranking is unstable.

E7 remains unrun. Food stage remains NOT RELEASED.
