# DrosoSense-RC — FROZEN: a completed negative construct study

**Status: FROZEN 2026-09-23. No v4 will be opened. The repository is not deleted and is not
continued; it stands as a completed negative construct study, and its methods and
diagnostics carry forward into the new project
[`docs/resaudit_food_preregistration.md`](resaudit_food_preregistration.md).**

Do not add experiments to this project. The construct phase's stop-loss fired for a
mechanism-level reason, and the pre-registered rule that produced it is explicit: no v4.

---

## 1. What this project established

The question DrosoSense-RC asked was whether the *Drosophila* olfactory connectome can serve
as a recurrent reservoir for e-nose food sensing. The answer this work supports is **no for
this construction family**, for a reason that is now measured rather than assumed.

```
C1  ORN-aligned / typed-aligned input mapping      PASS   (amendments 2, 3)
C2  grown, edge-retaining olfactory subgraph       PASS
C4  R2 wiring-only counterfactual                  PASS   (amendment 1)
C3  the recurrence must participate                FAIL
M5  structural-dynamics audit                      the substrate HAS the cycles
M5b input-geometry curve                           input support excluded as sufficient
v3  five biologically-defined candidate substrates STOP-LOSS: none reaches S1 >= 2*Din
```

The mechanism, in one line:

```
D_eff([B, A·B, …, A¹⁶·B]) ≈ 4.7 – 7.1   against a gate of 10
```

even though the input actually reaches ~40 of `A`'s eigenmodes in every candidate. The modes
are reached; they are not **independent under propagation**. And this held across five
biologically distinct substrates, including variants built specifically to add recurrence:

| substrate | M | S1 |
|---|---:|---:|
| S0 the v2 substrate | 80,443 | **7.129** |
| S1 + higher_order feedback → PN/KC | 82,139 | 6.963 |
| S2 MB/KC-centred | 85,212 | 4.713 |
| S3 + DAN/MBON feedback → KC | 79,474 | 7.005 |
| S4 S1 ∪ S3 | 80,594 | 6.890 |

**More edges, more feedback, and a different centre of the circuit all failed to move it.** The
MB/KC-centred candidate had the *most* edges and the *lowest* score.

## 2. Why the stop-loss is the result, not a failure to keep trying

The project's own discipline says a criterion that fails is reported and never relaxed, and
that "close" is not a pass. Two rounds of substrate and input work had already produced a
precise negative; the v3 stage existed to ask, once, whether a differently-chosen
biologically-defined substrate would clear a gate that the v2 substrate did not. Five
candidates answered no. Continuing from here would convert the construct gate from a check
into a search — which is exactly what a stop-loss is for.

Three findings deserve to survive this project as findings, not as incident:

1. **Structural connectivity does not imply computational capacity.** 100 % SCC, 100 % cycle
   coverage, 80,443 edges, mean out-degree 80 — and the substrate is still not a reservoir.
2. **Input geometry is bounded by the substrate's eigenstructure.** Widening W_in from 243 to
   592 nodes peaked the Krylov rank at 1.42·Din and then *reduced* it.
3. **Biological plausibility and computational suitability are different axes.** The
   candidates that were most biologically motivated (feedback closures) scored *below* the
   plain expansion. Nothing in this project licenses reading a higher score as "more
   biological".

## 3. What is reusable

| artefact | what it gives the next project |
|---|---|
| `drososense/connectome_cycles.py` | SCC (Tarjan), cycle-edge fraction, cycle-length distribution, ORN-reachable recurrent core |
| `drososense/substrate_scores.py` | S1 Krylov controllability, S2 left-eigenvector mode participation (Schur fallback), S3 spectral/non-normality diagnostics |
| `drososense/reservoir/dynamics.py` | the C3 metrics: gain-free and gain-inclusive `R_t`, the memory metric with the `A := 0` control, effective rank |
| `drososense/evaluation/` | evidence identity (schema 2), parameter-evidence scoping, the gate engine, the specimen cluster unit |
| `ops/audit/` | the audit pattern: read-only, writes one JSON per criterion, states its own provenance |
| `docs/` | the amendment discipline (a new file per amendment, never an edit) and the construct-phase record |
| the connectome substrate | the **negative biological control** in the next project's comparison |

## 4. What this project does NOT claim

* **Not** that a *Drosophila* connectome could never serve as a reservoir. What is
  established is the negative for the declared construction family.
* **Not** that biology is computationally uninteresting. The finding is narrower: strong
  structural connectivity did not produce a controllable, memory-bearing state under a
  faithful input geometry.
* **Not** a food-sensing result. No formal food experiment was ever run, because the phase
  gate legitimately stopped before one. There is no macro-F1, no MAE, no TVC number from this
  project, and none may be quoted as if there were.
* **Not** evidence about fly olfaction. Nothing here speaks to how *Drosophila* computes; it
  speaks to what this substrate does inside a reservoir-formalism.

## 5. The final status line

> **DrosoSense-RC is a completed negative construct study: through C1→C2→C4→C3, M5, M5b and a
> five-candidate biologically-defined substrate selection, the *Drosophila* olfactory
> connectome did not produce the controllability, memory or state expansion a reservoir
> requires — and the reason is mechanism-level, measured, and independent of which
> biologically-motivated variant was chosen.**

The question this project could not answer — *how should a lightweight reservoir be validated
and designed for food sensing?* — is now the new project's to answer, with this connectome as
its motivating counterexample.