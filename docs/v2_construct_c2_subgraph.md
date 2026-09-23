# v2 construct phase — C2, the grown connected subgraph

**Status: IMPLEMENTED AND MEASURED (validation-only).** Every number below comes from
[`results/audit/v2_construct/C2_subgraph.json`](../results/audit/v2_construct/C2_subgraph.json),
produced by [`ops/audit/v2_construct_check.py`](../ops/audit/v2_construct_check.py)
`--only C2` on the GPU box against the delivered connectome. No model was fitted, no
test split was read, no record was written, and no formal inference ran.

Implementation: `drososense.connectome_selection.expand_from_orns` +
`subgraph_quality`. Commits `e7af8715`, `13d2d78f`, `f9c53e8b`.

---

## 1. What was required

Signed as D3/D4 (`docs/v2_preregistration.md` §2): a **deterministic multi-source
biological expansion** — seeds = the declared **ORN** population, expansion through
the layer order **ORN → PN → KC/LH → MBON/DAN/higher-order**, frontier ordered by
**cumulative synapse count from the already-selected nodes** with ties by neuron id,
and the induced biological edges **kept**. Primary **N = 1000**, a registered
engineering operating point, not a biological constant.

| id | criterion | v1 (at N=250, the audit's size) | signed target | **v2 at N=1000** | verdict |
|---|---|---|---|---|---|
| C2.1 | input population present and typed | ORN = 0 | see C1.4 | ORN 243, PN 243, KC 106, MBON+DAN+HO 408 | **PASS** |
| C2.2 | eligible-edge retention (denominator = induced eligible edges) | not computable in v1 | ≥ 0.90 | **1.0000** before *and* after the declared normalization | **PASS** |
| C2.3 | largest weak component / N | 0.148 | ≥ 0.90 | **1.0000** | **PASS** |
| C2.4 | isolated fraction | 0.648 | ≤ 0.02 | **0.0000** | **PASS** |
| C2.5 | mean in-subgraph unweighted out-degree | 0.50 | ≥ 2.0 | **80.443** (median 65) | **PASS** |
| C2.6 | deterministic, declared selection with `target_n`, seed, sha256 | unrecorded | present | `orn_seeded_biological_expansion (v2 D3)`, sha256 `a9b7b3f5…`, consumes no randomness | **PASS** |

Composition of the substrate: **ORN 243 · PN 243 · KC 106 · higher_order 166 ·
MBON 82 · DAN 160** — 1,000 nodes, 80,443 induced eligible edges, and **14,305 real
ORN→PN edges reaching 232 of the 243 selected PNs (95.5 %)**. C1.2 on this substrate:
ORN fraction 0.243, density `243/(1000·5) = 0.0486 ≤ 0.10`.

## 2. The same graph and the same N, both selections

The v1 selection is measured on the identical graph at the identical N, so the
comparison is like for like:

| | v1 selection (N=1000) | **v2 expansion (N=1000)** |
|---|---:|---:|
| induced eligible edges | 2,343 | **80,443** (34×) |
| largest weak component / N | 0.575 | **1.000** |
| isolated fraction | 0.415 | **0.000** |
| mean unweighted out-degree | 2.343 | **80.443** |
| Enrichment (vs random of the same size) | 2.29× | **78.71×** |
| eligible-edge retention | 1.0000 | 1.0000 |
| ORN-aligned input mapping | **refused**: 1 ORN for 5 channels | buildable, 243 ORN |

Note where v1 actually fails **at this N**: C2.3 and C2.4. It *passes* C2.5 at N=1000
(2.343 ≥ 2.0); the audit's 0.50 was measured at N=250, where the substrate is a
126-edge residue. Reporting v1's numbers at the size they were measured at, and v2's at
the primary N, is why both columns state their N.

## 3. C2.2 does **not** discriminate — and saying so is part of the record

Under the corrected denominator **both** selections retain **1.0000** of their induced
eligible edges. That is expected: any construction that keeps its induced block retains
all of it, and a *random* node set does too. What v1 lacked was not retention but
**induced edges at all**: 2,343 against 80,443.

So C2.2 is a **guard**, not a discriminator: it catches a construction that sparsifies
or drops wiring biology provides (v1's whole-graph share of 0.0009 was the mislabelled
number this denominator replaced). The measurements that carry the biology-keeping
information, both reported on every line, are the **induced eligible edge count** and
the **Enrichment** (2.29× → 78.71×).

**Enrichment's form is declared, not chosen.** The signed text defines it as
`Retention_bio / Retention_random`, but under the corrected denominator that ratio is
≈1 for any node set and therefore carries no information. The report therefore defines
Enrichment on the **connectome-share** form — `induced_share_of_connectome(bio) /
induced_share_of_connectome(random of the same size)` — states that in the JSON
(`enrichment_form`), and publishes both shares so a reader can re-derive either
reading. Picking the flattering form silently would have been the alternative.

## 4. Two defects found by measuring, not by reading

Both were caught by running the check against the real connectome. Neither would have
surfaced from unit tests on synthetic graphs, because both are properties of the real
degree distribution.

**4.1 The layer-staged frontier starved the downstream layers.** The first run gave
`{ORN: 500, PN: 500}` at N=1000 — **zero Kenyon cells, zero MBON/DAN** — because the
ORN group took the whole C1.2 budget and the frontier then filled PN to exhaustion
before reaching the last two groups. C2.1 failed: the substrate did not contain the
populations the criteria name.

**4.2 A group floor is not a class floor.** After allocating per group, the
`(KC, higher_order)` group held 272 nodes of which only **21 were Kenyon cells**:
higher-order/lateral-horn neurons carry more synapses and won every frontier tie.
C1.4 asks for `KC ≥ max(50, 4·Din) = 50`, so C2.1 failed again, for a different reason.

The allocation is now declared and contains no tuning constant:

1. every **class** C1.4 names individually (ORN, PN, KC) is filled to its floor
   **first**, by frontier score;
2. every **layer group** gets at least its floor before any group takes more;
3. the remainder is shared **equally** among groups that still have room, capped by
   what each group can supply and — for ORN — by C1.2's ceiling;
4. a graph that does not contain a population is refused **by name**
   (`KC+higher_order needs 50 nodes for C1.4 and the graph offers 30`), never padded
   with untyped nodes.

C1.2 and C1.4 together fix a **minimum admissible N** that is derived rather than
hoped for: `max(Σ floors, 200/Din)` = **110 at Din=5**, and the expansion refuses below
it. At the primary N=1000 it is satisfied with room to spare.

## 5. What is deliberately NOT claimed

- **No formal inference has run.** C2 passing means the substrate is *constructible*;
  it is not evidence for or against the biological hypothesis.
- **N = 1000 is an engineering operating point**, registered so that no N is chosen
  from results. The size study (500/1000/2000) runs only after the phase gate, and
  only to answer scaling.
- **The composition is not a biological ratio.** 24.3 % ORNs against the connectome's
  1.83 % is a declared allocation driven by C1.4's floor and C1.2's ceiling, not a
  claim about fly olfactory anatomy. (The real ratio would give ~18 ORNs at N=1000,
  just under C1.4's floor of 20 — so the criterion itself forces a slightly
  super-biological ORN fraction at this N.)
- **The substrate is drawn only from typed populations**; the 78.6 % of the connectome
  classified `other` is never added, which is what makes C1.4 measurable on it and is
  itself a declared design choice.

## 6. Reproducing

```bash
# on the GPU box, where the connectome lives
python ops/audit/v2_construct_check.py --only C2 \
  --data-root /root/autodl-tmp/drososense/data-root --din 5 --target-n 1000
```

Tests: `tests/test_subgraph_expansion_c2.py` — 15 tests pinning the layer order, the
per-class floors, the derived minimum N, the named refusals, the frontier rule, the
retention denominator, the self-loop handling, the component/isolation/degree metrics
and the digest convention (identical to the v1 `sha256_of_array`, so v1 and v2
selection digests are comparable).

## 7. Next — C4

C4 is the **weight-preserving R2 counterfactual** (D6): preserve directed degree, the
global weight multiset and the per-source outgoing weight multiset, in-strength median
relative error ≤ 5 %, edge overlap ≤ 0.20, build ≤ 10 min at N=1000. It runs before C3
because C3 measures the recurrence/input ratio on a valid R0 **and** a valid R2.
