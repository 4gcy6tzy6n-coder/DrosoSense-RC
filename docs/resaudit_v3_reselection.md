# V3 selection recomputation under the corrected spectral-radius estimator

**Status: COMPLETE 2026-09-23. Answers the upstream-provenance blocker. Data read-only; the
frozen artifact was NOT overwritten.**

Runner: `ops/audit/resaudit_v3_reselection.py`.
Report: `results/audit/resaudit_v3_reselection/V3_reselection.json`.

## The question

The Stage-1 negative is only freezable if F1's *selection* provenance is sound. Even with F1's
current measurement verified by content hash, a defective ρ estimator upstream could have
selected a **different substrate**:

```
wrong rho estimator -> possibly a different selected F1
```

so `current F1 measurement is valid` and `current F1 selection provenance may be invalid` had
to be settled separately.

## 1. V3's selection IS ρ-dependent — and that had to be established, not assumed

`score_matrix` normalises each candidate to a common `rho_target` by one global scalar, and
`krylov_score` prunes singular values with an **absolute** tolerance (`1e-10`). The scalar
therefore changes how many Krylov directions survive pruning. Measured on one matrix: `D_eff`
moved from **1.02 to 13.44** under a 0.5× rescale. S1 is **not** scale-invariant, so the
selection really can be reordered by a wrong ρ. The concern was legitimate.

## 2. But the frozen `score_matrix` is NOT affected by the directed-`eigsh` defect

`score_matrix` **symmetrises** before solving: it estimates the radius of `(A + Aᵀ)/2`, which is
a genuine symmetric problem, so `eigsh` is the correct tool there. V3's selection path is
therefore not corrupted by the defect found in
`connectome_reservoir.spectral_radius`. That distinction is why this recomputation compares two
*conventions* rather than asserting a defect.

**Setup validated by exact reproduction.** My frozen-convention run reproduces the historical
`S1` values to the digit they were stored with, so the pipeline is the frozen one:

| candidate | frozen artifact `S1` | reproduced `S1_sym` | match |
|---|---:|---:|---|
| S0 | 7.129071758407175 | 7.1291 | ✓ |
| S1 | 6.9629 (6.963) | 6.9629 | ✓ |
| S2 | 4.713 | 4.7134 | ✓ |
| S3 | 7.005 | 7.0054 | ✓ |
| S4 | 6.89 | 6.8897 | ✓ |

## 3. Both conventions agree: no candidate is eligible, and the stop-loss fires

```
$ PYTHONPATH=. python3 ops/audit/resaudit_v3_reselection.py

  S0: N=1000 M=80443 rho_sym=2808.47 rho_dir=2663.67 S1_sym=7.1291 S1_dir=7.1624
  S1: N=1000 M=82139 rho_sym=2805.33 rho_dir=2660.87 S1_sym=6.9629 S1_dir=6.9515
  S2: N=1000 M=85212 rho_sym=1551.51 rho_dir=1495.80 S1_sym=4.7134 S1_dir=4.8979
  S3: N=1000 M=79474 rho_sym=2797.79 rho_dir=2653.96 S1_sym=7.0054 S1_dir=6.9692
  S4: N=1000 M=80594 rho_sym=2796.32 rho_dir=2652.65 S1_sym=6.8897 S1_dir=6.7846

frozen convention (symmetrised rho): selected=None stop_loss=True
corrected convention (directed rho): selected=None stop_loss=True
SELECTION CHANGED: False
```

The directed/symmetrised radius ratio is a near-constant **0.9486–0.9642** across all five
candidates, so the two conventions are close to a single global rescaling of the whole
candidate set. And every `S1` sits far below the gate of **10.0** under *both*:

| candidate | max `S1` (either convention) | gate | fraction |
|---|---:|---:|---:|
| S0 | 7.1624 | 10.0 | 0.72 |
| S1 | 6.9629 | 10.0 | 0.70 |
| S2 | 4.8979 | 10.0 | 0.49 |
| S3 | 7.0054 | 10.0 | 0.70 |
| S4 | 6.8897 | 10.0 | 0.69 |

## 4. Consequence

**F1's selection provenance is confirmed.** The corrected estimator does not change which
substrate is chosen, because the frozen rule chose **none** — the v3 stop-loss fired, and it
fires identically under the corrected measurement. There is no alternative F1 that the defect
concealed.

Two independently true statements now:

```
current F1 measurement          = VALID   (content hashes; amendment 5)
current F1 selection provenance = VALID   (this document)
```

The Stage-1 negative therefore does **not** have to be withheld for upstream-provenance
reasons, and its status can be upgraded from
`PROVISIONAL_PENDING_UPSTREAM_SELECTION_RECOMPUTATION` to `FROZEN_NEGATIVE_RESULT` once the
remaining scoped corrections are recorded.

## 5. Still outstanding

`M5_structural_dynamics` and `ledger_repair_E9` are in scope via the frozen defect and have not
yet been recomputed. Neither feeds the Stage-1 family battery's A3 verdict — A3 is computed
from `krylov_score` on the materialized families and never consults ρ — but both must be
recomputed before their historical conclusions are cited.

**Note on this document's own scope:** it establishes selection provenance, not a scientific
result. No food data was read and no A1–A5 threshold was touched.
