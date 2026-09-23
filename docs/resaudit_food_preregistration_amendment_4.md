# ResAudit-Food — pre-registration amendment 4: F1's structural identity and its dataset-conditioned input

**Status: amendment, 2026-09-23. Written BEFORE F1 was materialized and BEFORE any Stage-1
family score existed.**

**The original pre-registration is immutable**, and so is every earlier amendment. Per this
repository's amendment discipline this file edits nothing; it is added.

Operator decision this amendment records: *"YES: F1 wiring materialized per the frozen S0; YES: F1
instantiated with the Din=6 and Din=8 input mappings per the ResAudit-Food data plan. Din=5 is
retained as the v3 historical measurement context, not as Stage-1 F1 structural identity."*

## 1. The decision

> **F1 structural identity is defined by the frozen S0 substrate (node set, induced wiring,
> weights, N=1000, M=80,443). The historical `Din=5` belongs to the v3 measurement context and is
> not part of the Stage-1 structural identity. For ResAudit-Food, the frozen F1 substrate is
> instantiated with the dataset-conditioned input widths declared in `DATA_PREREGISTRATION.md`:
> `Din=6` for FD1/FD3 and `Din=8` for FD2. Only `B` may change with `Din`; the F1 node set and `A`
> must remain byte-identical across widths.**
>
> **No food labels, model performance, or Stage-1 family scores may influence the input
> re-instantiation.**

## 2. Why this is the only reading consistent with the frozen record

The frozen v3 selection record (`results/audit/v3_selection/V3_substrate_selection.json`) declares
candidate 0 — the substrate this project calls F1 — with `Din = 5`, and
`resaudit/families.py` transcribed that as F1's identity:

```
F1 identity: N=1000, Din=5, K=16   (frozen V3_substrate_selection.json settings)
"These are the values F2/F3/F5 must match; they are NOT re-derived here."
```

Read as a *structural* identity that would freeze F1 at `Din=5` while FD1/FD3 require 6 and FD2
requires 8, every family comparison would become unfair in a way the pre-registration already
forbids: the siblings would run at a different input width from F1. `DATA_PREREGISTRATION.md` has
already frozen the opposite rule —

> *Within each dataset, F1–F5 must use exactly the same `Din` and the same input-mapping rule.
> Different datasets may use their native sensor dimensionality.*

— so the two documents can only be reconciled by splitting F1's identity in two. That is what this
amendment does, and it is why `Din = 5` is reclassified rather than treated as an obstacle.

## 3. The identity split

```
STRUCTURAL IDENTITY   (may not change with Din, may not change per dataset)
    A_F1                  the frozen S0 induced wiring
    node set              the frozen S0 node list
    weights               the frozen S0 weight multiset
    N = 1000
    M = 80,443
    provenance            the frozen S0 selection rule and its admission record

DATASET-CONDITIONED INPUT INSTANTIATION   (may change with Din)
    B_F1(Din)
```

Stage 1 therefore contains two distinct F1 instantiations, not two substrates:

```
F1^(6) = (A_F1, B_6)      used for FD1 and FD3
F1^(8) = (A_F1, B_8)      used for FD2
```

with the invariants:

| Property | Required across `F1^(6)` and `F1^(8)` |
|---|---|
| `A_hash` | **identical** |
| node list | **identical** |
| edge list | **identical** |
| weight multiset | **identical** |
| `N`, `M` | **identical** (1000, 80443) |
| `B_hash` | may differ |
| `Din`, input support | may differ |

`Din = 5` is retained in the record as a **historical v3 measurement context** — it is what the
frozen v3 selection happened to be run at — and it is **not** part of Stage-1 F1 structural
identity.

## 4. What this amendment does not change

| Item | Effect |
|---|---|
| A1–A5 thresholds | **none** |
| Probe / drive protocol (amendment 1 §0.1, amendment 3) | **none** |
| The frozen S0 selection rule, `target_n`, frontier rule, cap rule | **none** — F1 is materialized *by* them, not redefined |
| The typed-aligned input-mapping rule (`receiving_fraction 0.80`, `density_limit 0.16`) | **none** — only `Din` is re-parameterized, by the rule's own declared width |
| F1–F5 family definitions, `N=1000`, edge budget | **none** |
| Candidates admitted or excluded | **none** |

## 5. Provenance verification required before F1 may be called F1

Materialization is permitted against the local raw adjacency
(`DrosoSense-RC/connectome/adjacency/olfactory_v1.npz`) using the frozen selection rule. The
artifacts may be **named F1 only if** the regenerated substrate reproduces the frozen S0
provenance exactly:

```
N                 = 1000
M                 = 80443
node_list_sha256  == the frozen S0 value
A_sha256          == 3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20
weight_multiset   == the frozen S0 value
```

`B` is verified the same way at the historical width before any re-parameterization:
`B_hash == 3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968` at `Din=5`.

**If any hash disagrees, materialization STOPS.** Agreement of `N` and `M` alone is **not**
sufficient and must not be accepted as reproduction: if the frozen S0 cannot be rebuilt exactly
from the current raw adjacency plus the frozen selection procedure, that is a **provenance
failure to be explained**, not a rounding difference to be tolerated or a stand-in to be
substituted.

## 6. Artifacts this amendment authorizes

```
F1_A.npz                  the frozen S0 induced adjacency
F1_B_Din6.npz             dataset-conditioned input for FD1/FD3
F1_B_Din8.npz             dataset-conditioned input for FD2
F1_materialization.json   raw_connectome_sha256, selection_rule_version,
                          node_list_sha256, A_sha256, N, M, B6_sha256, B8_sha256,
                          input_mapping_rule, protocol/amendment hashes
```

`F1_materialization.json` records the exact `HEAD` the measurement was taken at, so a Stage-1
result can be traced to a stable code state.

## 7. Channel-level provenance for `B`

Amendment 3 established the probe's nested property (`probe(6) == probe(8)[:, :, :6]`). Where the
typed-aligned input-mapping rule permits it without changing the algorithm, the same discipline is
wanted for `B`:

```
B_6  ==  B_8[:, 0:6]      on the first six channels
```

so that a difference between FD1/FD3 and FD2 cannot be contaminated by a re-randomised input
assignment on the shared channels. **If the current mapping rule does not guarantee this, the
algorithm is not modified to make it so.** In that case the channel-level provenance is recorded
and verified instead, and the absence of the nesting property is reported as a property of the
mapping rule.
