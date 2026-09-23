# ResAudit-Food — pre-registration (new project)

**A construct-validity-driven framework for auditing and designing lightweight reservoir
computers before deployment in AgriFood sensor systems.**

**Status: PRE-REGISTERED 2026-09-23, BEFORE ANY RESERVOIR IS BUILT OR SCORED.** Everything
below — the audit battery, the reservoir families, the held-constant design, the qualification
rule and the stage gate — is fixed here so that no reservoir can be chosen, and no criterion
moved, after seeing a task result.

**Provenance.** This project derives from
[`docs/FROZEN_negative_construct_study.md`](FROZEN_negative_construct_study.md)
(DrosoSense-RC, frozen). Its motivating case study is that project's measured result:

```
100 % strongly-connected + 100 % cycle coverage + 80,443 edges + mean out-degree 80
        ⇏   an effective reservoir
```

— and, across five biologically-distinct candidates, `D_eff([B, A·B, …, A¹⁶·B]) ≈ 4.7–7.1`
against a requirement of 10.

The old project's stop-loss said *"the connectome-reservoir main line stops; no v4"*. This is
**not** v4: it is a new scientific question with the connectome demoted to a control. The
freeze is respected — nothing here re-opens DrosoSense-RC's design loop.

---

## 1. The question

> Can a candidate reservoir be judged — **before any expensive training and before any
> formal test** — to have the computational capacity that food sensing requires?

The immediate follow-on question, which the framework must also answer:

> Given that judgement, what does a reservoir that *passes* it look like, and how does it
> compare with the standard choices (random ESN, degree-matched random, small-world) and with
> a biologically-derived substrate that fails?

Both are AgriFood-engineering questions, not biology questions. The deliverable is a
**measurable, pre-deployment audit** plus the design that it selects.

## 2. The three contributions

**C1 — The audit battery.** A small, cheap, model-free set of pre-deployment checks, applied
to a candidate reservoir *before* task training:

```
C_structural      structure: SCC/cycle coverage, WCC, isolated fraction, degree distribution,
                  and the induced-edge retention of the substrate
C_counterfactual  the wiring-only counterfactual exists and is fair: degree sequence,
                  weight multiset and per-source weight multisets preserved exactly, and a
                  mixing criterion that the counterfactual is genuinely rewired
D_Krylov          controllability: D_eff([B, A·B, …, A^K·B]) — does the input's reach
                  survive propagation?
M_recurrence      memory: (M_recurrent − M_{A:=0}) / M_recurrent — does the recurrence carry
                  memory the feed-forward path does not?
D_state           state expansion: D_eff of the state matrix — does the state leave the
                  input's own dimension?
```

The claim to be tested is that **spectral radius and connectivity are not sufficient** and that
these five are: the motivating case passes the usual sanity checks and fails the battery.

**C2 — The connectome as a compelling failure case.** The *Drosophila* olfactory connectome
becomes the paper's sharpest figure: a substrate that is structurally impeccable and
computationally inadequate, with the mechanism measured (modes are reached but not independent
under propagation) rather than asserted. DrosoSense-RC's C1–C4, M5, M5b and v3 work becomes the
battery's validation evidence.

**C3 — An audit-qualified lightweight sparse reservoir.** Designed against the battery's own
pre-declared targets — not by search over task scores — and then taken through the food tasks
like every other family. This is the constructive half of the paper.

**Explicitly not a contribution.** "The connectome fails" is not a paper on its own for this
venue: it is a computational-neuroscience or RC-methods result. It earns its place as the case
study that motivates and validates the audit.

## 3. The audit battery, with thresholds declared now

Every family is audited on the same substrate size and the same input geometry.

| id | check | threshold | source of the threshold |
|---|---|---|---|
| A1 | structure present | SCC largest ≥ 0.90·N, isolated ≤ 0.02, mean out-degree ≥ 2.0 | DrosoSense-RC C2.3–C2.5 |
| A2 | counterfactual fair | degree sequence, weight multiset, per-source multisets **exact**; overlap ≤ 0.20 (floor-relative where the mixing floor reaches it) | DrosoSense-RC C4, amendment 1 |
| A3 | **Krylov controllability** | **`D_eff([B, A·B, …, A¹⁶·B]) ≥ 2·Din`** | DrosoSense-RC v3, pre-registered |
| A4 | **recurrent memory** | **`(M − M_{A:=0}) / M ≥ 0.20`** | DrosoSense-RC C3.2 |
| A5 | **state expansion** | **`D_eff(state) ≥ 1.5·Din`** | DrosoSense-RC C3.3 |

**A3 is the primary qualification gate**; A1/A2 are admission requirements and A4/A5 are the
downstream gates that a qualified reservoir must still clear. A family that fails A3 is
**not run on food data** — that is the framework's entire point, and it is also what makes the
food stage affordable.

**A4 and A5 are measured on validation windows only.** The test splits of D2 and D3 are not
read until a family is qualified and the food stage begins.

## 4. Stage 1 — the reservoir families (no food data)

Five families, all with **N = 1000**, the **same edge budget**, the same input dimension
`Din`, and the same readout:

| id | family | role |
|---|---|---|
| **F1** | *Drosophila* olfactory connectome substrate (the frozen project's S0) | **negative biological control** |
| **F2** | Erdős–Rényi random reservoir | the standard RC baseline |
| **F3** | degree-matched random rewiring of F1 | does degree alone explain anything? |
| **F4** | small-world reservoir | a structured, non-biological control |
| **F5** | **audit-qualified sparse reservoir** | the design contribution |

**Held constant across F1–F5**: `N`, the edge budget `M`, `Din`, the input-mapping rule
(one channel per receiving node, density within the declared ceiling), the readout family and
its training rule, and the audit measurement rule. **What varies is the wiring and the input
placement** — that is the comparison.

**F5's design rule is declared now, before it is built.** F5 is constructed to maximise A3
subject to the same budget, by a declared, task-blind procedure: build a sparse directed graph
whose eigenstructure admits many independent propagated directions from `B` (e.g. by
composing blocks whose directed cycles have incommensurate lengths, and placing the input
across those blocks), and verify it against A3 before any task is run. **The procedure may not
consult food labels, macro-F1, MAE or any test data.** If no such construction is found within
a bounded effort, that is reported as a finding about the battery's stringency, not patched by
loosening A3.

**Stage 1 output**: one JSON per family with A1–A5, the qualification verdict, and the
supporting diagnostics (spectrum, mode participation, power norms, structure). Every family
that fails A3 is reported with its failure, not dropped.

## 5. Stage 2 — the food stage (qualified families only)

Run on the frozen v1.x data and splits (D2 beef-uncontrolled, D3 rainbow trout), with the
project's existing evidence-integrity and cluster-unit discipline unchanged.

* **Primary metrics**: macro-F1 (classification) and MAE (regression), per the frozen protocol.
* **Data-efficiency**: 10 / 25 / 50 / 100 % of the training side, nested pools, test partition
  byte-identical across fractions.
* **Robustness**: sensor dropout (channels zeroed at test time) and additive noise / drift at
  declared levels.
* **Cost**: latency (inference time per window), memory, and trainable parameter count.
* **Comparison**: F1–F5 on identical folds, with the paired cluster-level statistics the frozen
  protocol specifies (specimen cluster unit, exact sign test over cluster means).

**The headline comparison is not "biology vs random".** It is:

> does an **audit-qualified** lightweight reservoir deliver the AgriFood task performance that
> the usual structural heuristics (connectivity, spectral radius, biological plausibility) do
> not predict?

F1's role is to make the audit's necessity concrete; F5 is what the paper offers the field.

## 6. Decision rules, declared in advance

1. **No family runs food tasks unless it clears A3.** This is the framework's claim, and the
   food stage is the test of it.
2. **If F5 clears A3 but fails A4/A5**, the battery's primary gate is mis-specified and that is
   reported as a battery finding before any task result is discussed.
3. **If F1 clears A3** — contrary to the frozen project's measurement — the audit is re-derived
   from scratch, because the motivating case would have evaporated.
4. **If every family fails A3**, the finding is that the declared thresholds are too strict for
   this substrate size and edge budget; the thresholds are not lowered silently, and the
   consequence is a paper about what a lightweight reservoir cannot be, which is a weaker but
   honest outcome.
5. **Negative task results are results.** If the audit-qualified reservoir does not beat the
   baselines on food tasks, that is reported as the finding; the audit's validity is judged by
   whether its predictions held, not by whether the design won.

## 7. What this project will not do

* **Not** reopen DrosoSense-RC. The frozen study stays frozen; the connectome appears here as
  F1, a control.
* **Not** claim biological fidelity anywhere. F1 is described as "the *Drosophila* olfactory
  connectome substrate"; no result here is a statement about fly olfaction.
* **Not** tune the audit thresholds to admit a candidate. A3–A5 are the frozen project's
  signed criteria, reused as-is.
* **Not** select any family, hyperparameter or input placement using task performance. Every
  selection decision in this project is made on validation-only structural and dynamical
  criteria, and the food stage is the first place a test split is read.

## 8. Order of work

```
1. freeze DrosoSense-RC                                        DONE (docs/FROZEN_negative_construct_study.md)
2. write this pre-registration                                DONE (this file)
3. build the audit runner + F2/F3/F4 generators                 next
4. build F5 to the declared task-blind rule
5. audit F1-F5, report A1-A5 with verdicts                      (no food data)
6. food stage on the qualified families                         (first test-split access)
7. robustness, data-efficiency and cost sweeps
8. write-up: the battery, the failure case, the design
```