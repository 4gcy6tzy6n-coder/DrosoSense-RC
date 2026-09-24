# E3 + E4 — mechanism audit of frozen A3, and scale-invariant measurement comparison

**Status: COMPLETE 2026-09-23. `A3_FACTOR = 2.0` unchanged. The frozen Stage-1 result is not
redefined. No sweep selects a replacement threshold.**

Runner: `ops/audit/resaudit_e3_e4_mechanism.py`
Report: `results/audit/resaudit_a3_mechanism/E3_E4_mechanism.json`

Real materialized F1 (N = 1000, M = 80443); substrate pool of 25 (5 families × 5 instances,
6 per family for the control families), all at the frozen ρ = 0.95, Din = 6, gate = 12.0.
No food data.

## E3.1 The frozen estimator, recovered exactly

```
D_eff(blocks) = (Σσ)² / Σσ²          over singular values σ > τ₀
blocks        = [B, A·B, …, A^K·B]   →  K + 1 blocks
K frozen      = 16                   →  17 blocks, (K+1)·Din = 102 columns at Din = 6
τ₀            = 1e-10                ABSOLUTE (NUMERICAL_TOLERANCE)
gate          = 2·Din                unchanged; diagnostic only in this run
```

## E3.2–3 The tolerance and input-scaling sweeps: NO effect

| sweep | range | `D_eff/Din` | A3 pass rate |
|---|---|---|---|
| τ/τ₀ | 1e-4 … 1e4 (τ = 1e-14 … 1e-6, factor **1e10**) | **constant 1.2868** | **0.000** |
| β | 1e-2 … 1e2 (factor **1e4**) | **constant 1.3774** | **0.000** |

The tolerance is **active but inconsequential**. Measured directly (E3.6): τ₀ prunes
**82.7 % of singular values by count** (58–82 of 390 survive), yet `D_eff` does not move —
because the participation ratio is dominated by the largest σ, so removing the tail changes
neither Σσ nor Σσ² materially. `pr_all` (the same estimator with no pruning) equals
`frozen_A3` to within 1e-9, confirming this independently.

**This refutes the hypothesis recorded in the E1 findings document**, which proposed that
saturation might be driven by the absolute tolerance being a scale artifact. It is not. The
estimator is effectively scale-invariant at this operating point (β spread 1.1e-9) *despite*
the absolute τ₀, for the same reason.

## E3.4 The spectral-radius sweep: the family ranking FULLY REVERSES

| ρ | ranking (best → worst) | family-mean ratio |
|---|---|---|
| 0.25 | **F3** > F2 > F5 > F1 > CM | 1.0365 |
| 0.50 | **F3** > F2 > F5 > F1 > CM | 1.1321 |
| 0.75 | F2 > F5 > **F3** > F1 > CM | 1.2584 |
| 0.90 | F2 > F5 > **F1** > F3 > CM | 1.3536 |
| **0.95** (frozen) | **F5** > F2 > F1 > F3 > CM | 1.3755 |
| 0.99 | **F1** > F5 > F2 > CM > F3 | 1.3899 |
| 1.05 | **F1** > F5 > F2 > CM > F3 | 1.3104 |

**F3 moves from rank 1 to rank 5; F1 moves from rank 4 to rank 1** across the tested ρ range.
The topology ranking A3 produces is therefore **an artifact of the operating point**, not a
property of the substrates. The maximum ratio rises to 1.3899 (ρ = 0.99) — still only 0.69 of
the gate.

## E3.5 The horizon sweep: ranking moves, and the ratio saturates

| K | blocks | ranking | family-mean ratio |
|---|---|---|---|
| 4 | 5 | F2 > F5 > F1 > F3 > CM | 1.3616 |
| 8 | 9 | F2 > F5 > F1 > F3 > CM | 1.3748 |
| 16 (frozen) | 17 | F5 > F2 > F1 > F3 > CM | 1.3755 |
| 32 | 33 | F1 > F5 > F2 > CM > F3 | 1.3764 |
| 64 | 65 | F1 > F5 > F2 > CM > F3 | 1.3774 |

F1 rises from rank 3 to rank 1; F3 falls from rank 4 to rank 5. The ratio itself is
**saturated by K = 16** (1.3755 → 1.3774 from K = 16 to 64, +0.14 %), so extending the horizon
cannot close a deficit of this size.

## E3.7 The ceiling holds under every sweep

```
A3 pass rate at EVERY sweep point (36 points):  0.0000
maximum family-mean D_eff/Din at ANY sweep point:  1.3899      (gate needs 2.0)
distinct family rankings observed across all sweeps:  5
```

So no setting of τ, β, ρ or K within the tested ranges admits any substrate, and five
different family orderings are produced depending on the setting. The gate is simultaneously
**unsatisfiable** and **unstable in what it ranks**.

## E4 Scale-invariant comparison

Four estimators on the *same* Krylov spectra:

| metric | definition |
|---|---|
| `frozen_A3` | PR over σ > τ₀ (the frozen estimator) |
| `stable_rank` | Σσ² / σ_max² |
| `pr_all` | PR over all σ > 0 (= frozen A3 with τ → 0) |
| `entropy_effective_rank` | exp(−Σ pᵢ log pᵢ), pᵢ = σᵢ/Σσ |

**Invariance to global B scaling** (relative spread of the mean across β ∈ [1e-2, 1e2]):

| metric | relative spread |
|---|---|
| `frozen_A3` | 1.15e-09 |
| `stable_rank` | 0.0 |
| `pr_all` | 2.30e-16 |
| `entropy_effective_rank` | 1.91e-16 |

All four are scale-invariant at this operating point — including the frozen estimator, whose
absolute τ₀ turns out not to bite (E3.2). **The scale-sensitivity objection does not apply
here**; the frozen estimator's problem is not its tolerance.

**F1's percentile within each control ensemble** depends on the metric:

| metric | F2 | F3 | CM | F5 |
|---|---|---|---|---|
| `frozen_A3` | 83 % | 100 % | 100 % | 50 % |
| `stable_rank` | **0 %** | 100 % | 100 % | **0 %** |
| `pr_all` | 83 % | 100 % | 100 % | 50 % |
| `entropy_effective_rank` | 50 % | 100 % | 100 % | **100 %** |

`frozen_A3` and `pr_all` agree (the pruning is inconsequential); `stable_rank` and
`entropy_effective_rank` **move F1 from the top of a distribution to the bottom**, i.e. the
metric choice materially changes whether F1 looks comparatively good or bad.

**Within-family variance** is small for every metric (CV 0.0012–0.0303), so all four are
precise; the disagreement is about *what they measure*, not about noise.

**Per instruction, no alternative metric was compared against the frozen 2·Din gate, and no
replacement threshold was derived.**

## What E3/E4 establish

1. The gate's failure to admit anything is **not** a tolerance artifact, **not** a scaling
   artifact, and **not** fixable by extending the horizon: it is intrinsic to the
   participation-ratio definition at this operating point, whose ceiling is ≈0.69 of the gate.
2. The **ranking** A3 produces is not stable: 5 distinct orderings arise across τ/β/ρ/K, with
   F3 moving rank 1 → 5 and F1 rank 4 → 1. A gate that ranks differently at a different ρ is
   not measuring a substrate property.
3. Together with E1/E2 (0 passes in 410 instances; F1 at the 57th percentile), this makes the
   Audit-the-Audit reframing the defensible one: **the preregistered gate failed
   discriminative validation**, and its ordering is operating-point dependent.

**Correction recorded:** the E1 findings document's conjecture that the absolute tolerance
might be the mechanism is **withdrawn** by this measurement. The tolerance prunes 83 % of
singular values by count and still changes nothing.

## Not done, deliberately

No threshold was changed (`A3_FACTOR` asserted 2.0 in the runner). The frozen Stage-1 result
stands unmodified. E5/E6/E7 not run. The food stage remains NOT RELEASED.
