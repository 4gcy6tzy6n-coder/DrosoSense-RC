# v2 pre-registration — AMENDMENT 2 MEASUREMENT: typed-aligned input mapping

**Status: MEASURED. Every metric moved in the direction M5 predicted; neither binding gate
clears; the remaining headroom is capped by C1.2.**

Amendment: [`docs/v2_preregistration_amendment_2.md`](v2_preregistration_amendment_2.md)
(the C1.1/C1.5 relaxation and the new mapping). Implementation:
`drososense.reservoir.input_mapping.build_typed_aligned_mapping`. Evidence:
[`results/audit/v2_construct/C3_dynamics_amendment2.json`](../results/audit/v2_construct/C3_dynamics_amendment2.json)
(compare with [`C3_dynamics.json`](../results/audit/v2_construct/C3_dynamics.json), the
ORN-only measurement it replaces as the *latest* but not as the record of C3's failure).

---

## 1. The C3 grid, side by side

Same substrate (C2 expansion, N=1000, Din=5), same grid (30 gain x leak points), same
dynamics. The only change is the input mapping's support:

| reservoir | mapping | nnz(W_in) | support rows | density | best memory drop | best D_eff factor |
|---|---|---:|---:|---:|---:|---:|
| R0_shared | ORN-only (v2) | 243 | 243 | 0.0486 | **−0.0029** | **1.104** |
| R0_shared | **typed-aligned (amend. 2)** | 473 | 473 | **0.0946** | **+0.0364** | **1.423** |
| R2_shared | ORN-only | 243 | 243 | 0.0486 | +0.0489 | 1.209 |
| R2_shared | **typed-aligned** | 473 | 473 | 0.0946 | **+0.0807** | **1.360** |

Gate thresholds are unchanged: memory drop **≥ 0.20**, D_eff factor **≥ 1.5**.
**C3 still FAILs**: no grid point clears both.

## 2. What the amendment proved, and what it did not

**Proved (M5's diagnosis was right).** Widening `span(B)` from the ORN rows to the typed
union moved **both** binding metrics in the predicted direction, on both reservoirs:

* memory drop: R0 **−0.003 → +0.036** (the recurrent graph starts contributing to the
  autocorrelation where before it contributed nothing, i.e. it crosses the v1 defect line)
* D_eff factor: R0 **1.104 → 1.423** (+29 %), R2 **1.209 → 1.360** (+12 %)

So the C3 collapse is **partly input-geometry-driven**, as M5 inferred from the Krylov
bound, and that is now a measured fact rather than an inference.

**Not sufficient.** Neither gate clears. The D_eff factor needs 1.5 and reaches 1.42; the
memory drop needs 0.20 and reaches 0.081. The improvement is real and it is ~⅓ of the way.

**And the headroom is capped by another signed criterion.** C1.2 bounds input density at
`0.10 · Din = 0.10` of the substrate, i.e. **≤ 500 support nodes** at N=1000, Din=5. The
measured construction is at **473** (fraction 0.80). Widening further is not available
without an amendment to C1.2:

| receiving_fraction | support nodes | density | admissible |
|---:|---:|---:|---|
| 1.00 | 592 | 0.1184 | **rejected** by C1.2 |
| 0.85 | 504 | 0.1008 | **rejected** (0.8 % over) |
| **0.80 (used)** | **473** | **0.0946** | accepted |

Adding the remaining layers (higher_order 166, MBON 82, DAN 160) has the same problem: at
any admissible fraction the union exceeds the ceiling.

## 3. C1.5's new rule, measured

Amendment 2 requires every PN with direct input to **also** receive a real ORN→PN edge.
Measured: **7 of 379** direct-input PNs have no ORN edge in the substrate
(`C1_5_direct_input_pn_without_orn_edge = 7`). The rule is not yet enforced in the mapping
— it is reported. Enforcing it (dropping those 7 from the support) is a one-line change
and would not alter the C3 result at this magnitude, but the record states the violation
rather than leaving it implied.

## 4. Where this leaves the construct phase

```
C1 ORN-aligned input mapping              PASS   (and amendment 2's typed-aligned mapping
                                                  is measured: density 0.0946, C1.5 has 7
                                                  reported violations)
C2 grown, edge-retaining subgraph         PASS
C4 R2 wiring-only counterfactual          PASS   (amendment 1)
C3 recurrence must participate            FAIL   (ORN-only: drop <= 0, D_eff 1.10)
                                          FAIL   (typed-aligned: drop 0.036, D_eff 1.42)
M5 structural-dynamics audit              MEASURED (substrate HAS the cycles; the
                                                  Krylov is bounded by ~Din)
Construct-phase gate                      FAIL
```

The two remaining routes are both owner decisions, and neither is a threshold change:

1. **Release C1.2's density ceiling** (an amendment) to buy more input support, and
   measure whether the trend continues to clear 1.5 / 0.20. The measured trend is
   +29 % D_eff for a 95 % increase in support; extrapolating linearly, clearing 1.5 would
   need roughly another +15 % support — i.e. ~545 nodes, density ~0.109, just past the
   current ceiling. This is the cheapest test and it is a real, bounded question.
2. **Change the dynamics** (v3-A: raw synapse counts + one global γ). M5 measured
   gamma-only vs row-L1 as nearly identical *in the linear Krylov*, so the expected gain
   is small — but the nonlinear regime is not what M5 measured, and the raw variant is the
   one the user flagged as the second suspect.

**No threshold was moved and no gate was relaxed in producing this record.**