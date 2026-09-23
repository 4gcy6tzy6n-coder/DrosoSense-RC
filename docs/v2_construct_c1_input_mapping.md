# v2 construct phase — C1, the ORN-aligned sparse input mapping

**Status: IMPLEMENTED AND MEASURED (validation-only).** Every number below comes
from a committed artefact, produced by
[`ops/audit/v2_construct_check.py`](../ops/audit/v2_construct_check.py) on the GPU
box against the delivered connectome and written to
[`results/audit/v2_construct/C1_input_mapping.json`](../results/audit/v2_construct/C1_input_mapping.json).
No model was fitted, no test split was read, no record was written.

Implementation: [`drososense/reservoir/input_mapping.py`](../drososense/reservoir/input_mapping.py)
(`orn_aligned_sparse_input_mapping`), vocabulary:
[`connectome/cell_types.py`](../connectome/cell_types.py). Commits `bd35c215`
(implementation + tests) and the C2-phase work that follows it.

---

## 1. What the criterion set required, and what was measured

The signed requirement (D1, `docs/v2_preregistration.md` §1) is
`X -> ORN -> PN -> downstream`: **only ORNs receive external sensor input**, PNs
receive signal **only through real ORN→PN edges**, the input map is named the
**"ORN-aligned sparse input mapping"** and is a **declared object** on every run
record, and reassigning which nodes form the input population **must change the
states**.

Measured on the **construct substrate** = the full delivered olfactory graph
(124,185 nodes, 14,828,657 edges — the only substrate where the real populations
and the real ORN→PN edges are all present):

| id | criterion | observed | threshold | verdict |
|---|---|---|---|---|
| C1.1 | receiving set == declared ORN population | 2,267 rows, `support_is_orn_only = True` | exact equality | **PASS** |
| C1.2 | `nnz(W_in)/(N·Din)` | **0.00365** | ≤ 0.10 | **PASS** |
| C1.3 | reassigning the input population moves the states | `max‖Δh‖ > 0` strictly | `> 0` | **PASS** |
| C1.4 ORN | ORN population | **2,267** | ≥ max(20, 2·Din) = 20 | **PASS** |
| C1.4 PN | PN population | **5,355** | ≥ 20 | **PASS** |
| C1.4 KC | KC population | **3,613** | ≥ max(50, 4·Din) = 50 | **PASS** |
| C1.4 downstream | MBON + DAN + higher_order | **15,392** | ≥ 20 | **PASS** |
| C1.5a | `W_in` support on PN rows | **0** | 0 | **PASS** |
| C1.5b | PNs receive only through real ORN→PN edges | **83,093** real edges reaching **949 of 5,355** PNs | every receiving PN has ≥ 1 | **PASS** |
| C1.6 | declared object + reproducible digest | `orn_aligned_sparse_input_mapping`, `w_in_sha256`, annotation digest | named + on the record | **PASS** |

C1.3 is the load-bearing one: it is the **strict inverse of the v1 measurement**.
On v1, a consistent relabelling of the nodes left the dynamics bit-identical
(A6, `max‖Δh‖ = 0.000e+00`). Here, two declarations over the *same graph and the
same nodes* that differ only in **which** nodes are ORNs produce different states,
while the v1 dense map remains annotation-blind on the same two declarations —
both halves are asserted in `tests/test_input_mapping_c1.py`.

**What C1.2 actually constrains.** One channel per ORN means exactly one non-zero
per ORN, so `nnz(W_in) = n_ORN` and the criterion *is* the ORN-fraction bound

```
n_ORN / N  <=  0.10 · Din
```

At Din = 3 an olfactory population may be at most **30 % of the selected nodes**.
This is a constraint on the **substrate**, not a free parameter of the mapping, and
the mapping reports the failure instead of thinning its own support. C2 has to
respect it.

---

## 2. The substrates the project actually ran cannot carry the pathway

Measured on the five delivered v1 selections
(`connectome.select_neurons`, seed `20260920`). All five reproduce **locally** from
the committed node metadata — the selection digests match the delivered records
exactly — so the substrate analysis needs neither the data root nor the 775 MB
adjacency.

| N | ORN | PN | KC | MBON+DAN+HO | mapping buildable | C1.4 ORN |
|---:|---:|---:|---:|---:|---|---|
| 250 | **0** | 41 | 18 | 55 | no — *"0 ORN row(s) for 5 channel(s)"* | FAIL (0 < 20) |
| 500 | **1** | 78 | 36 | 116 | no — same refusal | FAIL (1 < 20) |
| 1000 | **1** | 146 | 64 | 237 | no — same refusal | FAIL (1 < 20) |
| 2000 | **10** | 238 | 117 | 386 | yes | FAIL (10 < 20) |
| 4000 | **19** | 238 | 117 | 533 | yes | FAIL (19 < 20) |

Two statements, both load-bearing:

1. **C1.4's ORN threshold fails at every delivered size**, including the largest:
   19 ORNs against a threshold of 20. The audit's "0 ORNs at N ≤ 1000" was very
   slightly wrong — it is 0 at N=250 and **1** at N=500/1000 — and the direction of
   the finding is unchanged.
2. At N ≤ 1000 the mapping **refuses to build at all**, because one channel per ORN
   needs at least one ORN per channel. That refusal is the honest outcome: the
   alternative — densifying `W_in` or letting one ORN carry several channels to make
   the numbers work — would be the v1 construct failure with a biological label.

So **C2's selection is not an optimisation, it is a precondition.** The v1 substrate
could not have hosted an olfactory input pathway at any size it ran.

---

## 3. The vocabulary is derived, and two tables of it disagree

`ORN`/`PN`/`KC`/`MBON`/`DAN`/`higher_order` are **derived labels**: no committed file
carries a `cell_type` column. The rule is the `layer_mean` tier table, now declared
once in [`connectome/cell_types.py`](../connectome/cell_types.py) and imported by
`connectome/add_edge_masks.py` instead of being copied there (`layer_mean` is a
per-neuron olfactory-modality **score**; the boundaries are convention, not
measurement, and no claim about a specific receptor is licensed).

**Declared table** (committed `olfactory_v1_node_meta.csv`, the table
`select_neurons` already stratifies by):

| ORN | PN | KC | MBON | DAN | higher_order | other |
|---:|---:|---:|---:|---:|---:|---:|
| 2,267 | 5,355 | 3,613 | 9,650 | 2,572 | 3,170 | 97,558 |

**Delivered classified edge metadata** (`olfactory_v1_edge_meta_classified.csv`,
858 MB, remote-only):

| ORN | PN | KC | MBON | DAN | higher_order | other |
|---:|---:|---:|---:|---:|---:|---:|
| 2,299 | 5,363 | 3,594 | 9,582 | 2,563 | 3,212 | 97,572 |

They are **not** the same classification: **8,679 of 124,185 nodes disagree (7.0 %)**,
spread over every class pair (`MBON→other` 1,462; `other→MBON` 1,373; `PN→other`
803; `other→PN` 806; …). The earlier single-number comparison (ORN 2,267 vs 2,299)
was the visible tip, not the size of the disagreement.

**This is not cosmetic.** The admissible receiving-PN pool is vocabulary-dependent:

| vocabulary | real ORN→PN edges | PNs reached |
|---|---:|---:|
| declared tiers (committed metadata) | 83,093 | **949** of 5,355 (17.7 %) |
| delivered classified metadata | 55,335 | **2,091** of 5,363 (39.0 %) |

The criteria above are therefore measured **against the declared table**, and the
report records both tables, both digests and the disagreement. The declared table is
the one C1 uses because it is **committed to the repository** (so a reader can
reproduce every number without the data root and without a 858 MB file) and because
it is the classification the v1 selection already used. **Open:** which table is
*better* is a biological question this bundle cannot settle, and until it is settled
the PN pool for C2 must be stated with the vocabulary that produced it.

---

## 4. What is deliberately NOT claimed

- **No formal inference has run, and none may run** until every construct criterion
  passes. C1 passing on the construct substrate does not make the substrate usable:
  the formal substrate is C2's deliverable and must itself satisfy C1.4.
- **"Receptor-exact" is not claimed.** One ORN receives one channel, chosen by a
  seeded permutation. A statement that a particular MQ sensor corresponds to a
  particular olfactory receptor would need receptor-level data this bundle does not
  carry, and the report says `receptor_exact: false` at every use site.
- **The full graph is not a reservoir.** It is the construct substrate for
  measurement only; the report states `is_formal_substrate: false` in its own JSON.
- **C1 passing is not evidence for the biological hypothesis.** It is evidence that
  the input pathway is now what the protocol says it is.

---

## 5. Where the code lives, and how to re-run it

| artefact | what it holds |
|---|---|
| `drososense/reservoir/input_mapping.py` | the mapping: `orn_aligned_sparse_input_mapping`, `pn_direct_ablation`, the v1 map retained; the criteria as properties; every refusal |
| `connectome/cell_types.py` | the single declared vocabulary + `layer_to_class` + `load_node_classes` |
| `ops/audit/v2_construct_check.py` | the C1 report generator (READ-ONLY); `--only C1 --din 5 --data-root … --classes-csv …` |
| `results/audit/v2_construct/C1_input_mapping.json` | the measured report, one row per criterion with observed/threshold/verdict/source |
| `tests/test_input_mapping_c1.py` | 21 tests, one per criterion plus the refusals and the root-id precision regression |

```bash
# on the GPU box, where the connectome lives
python ops/audit/v2_construct_check.py \
  --data-root /root/autodl-tmp/drososense/data-root --din 5 \
  --classes-csv /root/autodl-tmp/drososense/data-root/connectome/metadata/olfactory_v1_edge_meta_classified.csv
```

---

## 6. Next — C2

C2 must deliver a subgraph that (a) contains the populations C1.4 requires, (b)
respects C1.2's ORN-fraction bound at the declared `Din`, (c) keeps the ORN→PN
structure that C1.5b measures (949 PNs under the declared vocabulary), and (d)
satisfies D4's retention/connectivity criteria — which the audit measured the
current selection failing at every size (0.09 % of out-edges retained at N=250).
The order after that is C4 (weight-preserving R2) then C3 (recurrence/input ratio),
because C3 depends on a valid R0 **and** a valid R2 graph.
