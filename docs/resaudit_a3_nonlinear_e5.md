# E5 — nonlinear construct validation of frozen A3

**Status: COMPLETE 2026-09-23. `A3_FACTOR` unchanged. ρ convention unchanged. Stage-1 outcomes
untouched. No gate defined on any nonlinear metric; no replacement metric selected.**

Runner: `ops/audit/resaudit_e5_nonlinear.py`
Report: `results/audit/resaudit_a3_nonlinear/E5_nonlinear.json`

## Primary question, and the answer

> **Does frozen A3 have empirical construct validity with respect to nonlinear reservoir state
> richness and memory?**

**At the frozen operating point: yes for memory capacity and state separability, no for raw
state diversity — and the relationships are not operating-point stable.**

| observable | Spearman(A3, ·) at ρ = 0.95 | 95 % bootstrap CI | sign-stable across ρ? |
|---|---:|---|---|
| **linear memory capacity `MC`** | **+0.805** | [+0.735, +0.852] | **yes** (always positive) |
| **kernel effective rank** | **+0.721** | [+0.624, +0.787] | no — **reverses** (−0.29 → +0.88) |
| state separation | +0.549 | [+0.449, +0.621] | no — **reverses** (−0.21 → +0.66) |
| state entropy rank | +0.389 | [+0.263, +0.501] | **yes** (always positive) |
| **state participation ratio** | **−0.667** | [−0.729, −0.578] | no — **reverses** (−0.72 → +0.87) |

Two of the five exceed the ρ_s > 0.6 bar that was set for "construct validity": `MC` (+0.805)
and `kernel_erank` (+0.721). But only `MC` and `state_entropy_rank` keep their sign across the
whole operating-point ladder.

## Setup (all frozen, asserted in the runner)

```
dynamics   x_{t+1} = (1-leak) x_t + leak * tanh(gain * A x_t + W_in u_t)   leak = gain = 1.0
probe      48 windows x 256 steps, seed 20260923, washout 16  (battery.probe_input)
A3         (sum sigma)^2 / sum sigma^2 over sigma > 1e-10, blocks [B, ..., A^16 B]
pool       F1 x1, F2 x50, F3 x50, configuration-model x50, F5 x30  =  181 substrates
           all at the frozen rho = 0.95, real F1 (N=1000, M=80443), Din=5
```

`MC` = Σ_{τ=1..20} R²_τ from an OLS readout of the delayed input off the top-128 state
components, on the task-blind probe. Bootstrap: 2000 substrate-level resamples.

## The negative result is informative: A3 moves *against* raw state diversity

`state_PReff` correlates **−0.667** with A3 at the frozen point. So A3 is **not** a proxy for
"the state occupies many dimensions" — it is anti-correlated with it. The positive associations
are specifically with **memory** and **separability**: substrates that A3 favours retain more
linear memory and produce more distinguishable window states, while their raw state-covariance
participation ratio is *lower*.

That is a coherent mechanism, not a paradox: a reservoir whose state spreads over many
directions can be *less* readably memory-bearing, and A3's Krylov-block measure rewards
propagated, structured reach rather than isotropic spread.

## Operating-point stability: three of five observables reverse sign

| ρ | state_PReff | state_entropy | kernel_erank | separation | MC |
|---|---:|---:|---:|---:|---:|
| 0.25 | +0.41 | +0.65 | **−0.29** | **−0.21** | +0.16 |
| 0.50 | +0.49 | +0.69 | **−0.24** | **−0.20** | +0.52 |
| 0.75 | +0.04 | +0.48 | +0.10 | +0.01 | **+0.90** |
| 0.90 | **−0.56** | +0.14 | +0.61 | +0.46 | +0.79 |
| **0.95** (frozen) | **−0.72** | +0.29 | +0.78 | +0.50 | +0.74 |
| 0.99 | +0.37 | +0.79 | +0.88 | +0.66 | +0.46 |
| 1.05 | **+0.87** | +0.89 | +0.81 | +0.65 | +0.54 |

`state_PReff` runs −0.72 → +0.87, `kernel_erank` −0.29 → +0.88, `state_separation` −0.21 →
+0.66. **Any claim of the form "A3 tracks nonlinear state richness" is therefore
operating-point dependent and cannot be stated unconditionally.**

`MC` is the most stable association (always positive, range +0.16 to +0.90, strongest near the
frozen ρ), and `state_entropy_rank` is likewise always positive (+0.14 to +0.89).

## Family distributions

| metric | F1 | F2 | F3 | CM | F5 |
|---|---:|---:|---:|---:|---:|
| `frozen_A3` | 6.964 | 7.196 ± 0.092 | 6.402 ± 0.014 | 6.366 ± 0.039 | 7.072 ± 0.061 |
| `state_PReff` | 5.225 | 5.088 ± 0.045 | 5.218 ± 0.017 | 5.195 ± 0.026 | 5.142 ± 0.028 |
| `state_entropy_rank` | 5.925 | 5.891 ± 0.060 | 5.917 ± 0.018 | 5.835 ± 0.023 | 5.948 ± 0.025 |
| `kernel_erank` | 1.237 | 2.254 ± 0.102 | 1.366 ± 0.007 | 1.705 ± 0.034 | 2.098 ± 0.077 |
| `state_separation` | 1.400 | 1.416 ± 0.003 | 1.404 ± 0.001 | 1.412 ± 0.002 | 1.416 ± 0.001 |
| `MC` | 3.670 | 3.462 ± 0.222 | 3.211 ± 0.007 | 2.843 ± 0.203 | 3.209 ± 0.064 |

Note the spread in `kernel_erank` (F2 2.254 vs F3 1.366 vs F1 1.237) — the families separate
far more strongly on kernel quality than on A3, where all five means sit within 6.366–7.196.
**`state_separation` is nearly constant across families (1.400–1.416) and is therefore a poor
discriminator despite its positive correlation.**

## What this does and does not establish

**Establishes.** At the frozen operating point A3 is a genuinely non-trivial correlate of
linear memory capacity (+0.805) and of kernel/state separability (+0.721), both with CIs
excluding zero over 181 substrates. The Reviewer-2 objection — *"you used a linear Krylov
measure to gate nonlinear RC"* — is **partially answered**: empirically, at this operating
point, A3 does track memory and separability. It is **not** answered in general, because three
of five observables reverse sign across ρ.

**Does not establish.** That A3 is a validated nonlinear-construct proxy. It does not track raw
state diversity (it anti-correlates), the associations are operating-point dependent, and the
E4/E1 results already showed the gate admits nothing and ranks unstably. This is a
**partial-validity** result, not a vindication.

**No gate was defined on any nonlinear metric, no replacement metric was selected, and
`A3_FACTOR` was asserted at 2.0 in the runner.** The frozen Stage-1 result is untouched.

## Consequence

The Audit-the-Audit framing survives, with a sharpened mechanism: A3 is not measuring nothing
(it tracks memory), but it is measuring something **operating-point dependent and
unsatisfiable at the frozen gate**. E6 and E7 remain unrun. Food stage remains NOT RELEASED.
