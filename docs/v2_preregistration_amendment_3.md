# v2 pre-registration — AMENDMENT 3 (SIGNED): the C1.2 density ceiling

**Signed 2026-09-23, after amendment 2's measurement.** It amends
[`docs/v2_preregistration.md`](v2_preregistration.md) §1 criterion **C1.2 only**. The
pre-registration is not edited; this file is the amendment.

**Why it exists.** Amendment 2 (typed-aligned input mapping) moved both binding C3 metrics
in the direction M5 predicted — memory drop R0 −0.003 → **+0.036**, D_eff factor R0
1.104 → **1.423** — and C3 still failed (gates 0.20 / 1.5). The measured headroom was
capped by C1.2:

| receiving_fraction | support nodes | density = nnz/(N·Din) | admissible under 0.10 |
|---:|---:|---:|---|
| 1.00 | 592 | 0.1184 | rejected |
| 0.85 | 504 | 0.1008 | rejected (0.8 % over) |
| **0.80 (used)** | **473** | **0.0946** | accepted |

The trend is +29 % D_eff for a +95 % increase in support. Clearing the 1.5 gate by linear
extrapolation needs roughly **545 support nodes**, i.e. density **≈ 0.109** — just past the
ceiling. **The criterion that was written to keep the input map sparse is now the binding
constraint on a bounded, answerable question**, so it is amended — not because a result was
displeasing, but because the question it blocks ("does the input-geometry trend continue?")
has a declared, bounded answer either way.

**The amendment.**

```
C1.2 (v2)      nnz(W_in) / (N · Din) <= 0.10
C1.2 (am. 3)   nnz(W_in) / (N · Din) <= 0.16
```

**Rationale for 0.16, stated before the experiment.** The criterion exists so the input map
is not the v1 dense map (`density = 1.0`). At 0.16, **84 % of the (node, channel) matrix is
still exactly zero** and no node receives more than one channel — the map remains a sparse
declared injection on typed populations, not a dense projection. The value was chosen as a
declared round number that admits the whole typed union at N=1000 (592 nodes → 0.1184) plus
headroom (800 nodes → 0.16), **not** fitted to any observed metric.

**The bounded experiment, declared in advance.** With the ceiling at 0.16, the
input-geometry curve is measured at the support sizes

```
support = 243 (ORN only, the v2 construction)
          ~350, ~473 (amendment 2), ~600 (full typed union), ~800 (best-effort)
```

Each support size is built by the typed-aligned mapping at the fraction that realises it,
and measured on the same (gain x leak) grid, the same substrate and the same dynamics as
amendment 2. Reported per support size: the best achievable memory drop and D_eff factor,
and how many grid points put median R_t inside [0.20, 1.00].

**Decision rule, declared before the data:**

* if some support size **clears both gates**, the C3 failure was input-geometry-driven, C3
  is re-run at that construction, and the construct phase re-opens;
* if the curve **plateaus below both gates**, input geometry is **excluded as sufficient**,
  the M5 diagnosis narrows to the dynamics, and the next step is the v3-A dynamics change
  (raw synapse counts + one global γ) rather than more input support.

**What is NOT changed.** Every other criterion, including C1.1/C1.4/C1.5 (as amended by
amendment 2), C2, and C3's thresholds (memory drop ≥ 0.20, D_eff ≥ 1.5 · Din) — untouched.
**No gate is relaxed by this amendment**; only the construction's declared density budget.