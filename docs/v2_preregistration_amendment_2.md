# v2 pre-registration — AMENDMENT 2 (SIGNED): the typed-aligned input mapping

**Signed 2026-09-23, between C3's failure and the input-geometry fix.** It amends
[`docs/v2_preregistration.md`](v2_preregistration.md) §1 (D1, D2, criteria C1.1 and C1.5),
nothing else. The amendment follows the project's rule that an amendment ADDS a file
and never edits a frozen one; the pre-registration is not touched.

**Why it exists.** C3 failed on the C2-expansion substrate with both binding gates (memory
drop, D_eff) below threshold. M5 (the structural-dynamics audit) showed:

* the substrate is **one giant strongly connected component** (SCC = 100 %, cycle_edges
  = 100 %, ORN-reachable recurrent core = 100 %); the two "graph lacks recurrence"
  branches were rejected by direct measurement;
* the linear Krylov `span(A^k · B)` is bounded at **≈ Din** on this substrate and this
  W_in, regardless of normalization — the normalization is not the bottleneck.

The remaining cause is **algebraic**: `span(B) ≈ Din` because the current input mapping
puts exactly one channel per ORN row (and zero on every other cell type), and the
recurrent expansion `A^k · B` saturates on the directions `span(B)` already occupies.
The owner's instruction is to extend W_in to the PN and KC rows, so `span(B)` widens to
the PN-channel and KC-channel subspaces and the recurrent expansion can carry more
independent state.

**Why C1.1 and C1.5 must be relaxed.** The current signed rules are:

* **C1.1** -- *every node receiving external input has cell type ORN, and the receiving
  set equals the declared ORN population*.
* **C1.5** -- *PNs receive signal only through real ORN→PN edges*.

Under amendment 2 they read:

* **C1.1** -- *every node receiving external input has a cell type in the declared
  receiving set, and the receiving set equals the union of the declared populations
  in that set* (default: ORN ∪ PN ∪ KC; subsets like {ORN, PN} or {ORN} are allowed).
* **C1.5** -- *every PN with direct external input must also receive signal through
  at least one real ORN→PN edge in the substrate; the recurrent path is preserved on
  every PN that has a direct input*.

The construction still keeps PNs downstream of ORN in the recurrent graph; the
amendment only adds a *direct* sensor injection alongside the recurrent path, so the
comparison R0-vs-R2 stays a fair counterfactual of wiring (amendment 1's measurement-
object rule continues to apply).

**What the amendment does NOT change.**

* C1.2 (≤ 0.10 · Din) -- unchanged. The amendment extends the receiving set, not the
  per-node entry count, so density is checked the same way and the rule still fails
  closed when the union's size pushes density above the bound.
* C1.3 (identity sensitivity), C1.4 (population minima), C1.6 (named object) --
  unchanged.
* C2, C3, C4, amendment 1 -- unchanged.
* **No threshold was moved.** This is a construction change, not a criterion change.

**Why not just expose more nodes by themselves.** The algebraic collapse is not
"more edges" -- the substrate already has cycles and ORN-reachability everywhere.
The collapse is because `span(B)` is small, so the recurrent expansion can only
walk a few independent directions. The amendment widens `span(B)` at the input, not
the substrate.

**Gate after the amendment** (the measurement, not a decision):

* **C1.1 / C1.5 (amended)**: every input-receiving node has a declared type; every
  direct-input PN also has a real ORN→PN edge. Measured at N=1000, Din=5.
* **C3**: re-run on the new substrate + new W_in. The branch point is whether the
  linear Krylov `D_eff(K=16)` grows above `≈ Din` under the wider input geometry --
  if yes, the algebraic collapse was input-driven and C3 should re-evaluate; if no,
  the collapse is downstream of W_in and a v3-A-style dynamics change is the
  next step.