# ResAudit-Food — pre-registration amendment 5: the `A_hash` reproduction criterion is defective

**Status: corrective amendment, 2026-09-23. Triggered by Amendment 4 §5's STOP condition firing
during F1 materialization. No threshold changed, no candidate admitted, no family measured.**

Amendment 4 §5 required F1 materialization to reproduce the frozen S0 including
`A_sha256 == 3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20`, and instructed that
any disagreement STOP and be explained rather than waived. The STOP fired. This amendment records
the explanation, and corrects the criterion that produced it.

## 1. What was reproduced, and what was not

Materializing the frozen S0 from the local raw adjacency with the frozen selection settings
(`din=5`, `target_n=1000`, `seed=0`) gave:

| Identity | Frozen value | Reproduced | Result |
|---|---|---|---|
| `n_nodes` | 1000 | 1000 | **OK** |
| `n_edges` | 80443 | 80443 | **OK** |
| `node_list_sha256` | `77ec9bfd90d6ec7216f642e1860c815ca972fd7638f1bb464766b9053faccb1e` | identical | **OK** |
| `edge_list_sha256` | `a840f45bd5c96ce654f8ff9c5957b0ec3eb158527a417ff635b462e03815740a` | identical | **OK** |
| `B_hash` at `Din=5` | `3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968` | identical | **OK** |
| `A_hash` | `3aa95745aea38ea85b4750c51c5262bd3d79c64e3450cca9a07fa8ecb33dac20` | `d3ae4a4d…` (run-dependent) | **MISMATCH** |

The two wiring-level hashes agree **byte for byte**, and they are stable when S0 is generated
repeatedly. `B_hash` agreeing is independently strong: `B` is built from `cand.root_ids`, so an
identical `W_in` requires an identical node set. **The frozen S0 wiring is therefore reproduced
exactly.** N and M were not the basis of that conclusion — the two line-level hashes were.

## 2. Why `A_hash` disagrees: it hashes a stochastic quantity

`A_hash` is not a hash of the substrate. It is `_matrix_sha256(score_matrix(cand.adjacency,
rho_target=0.9))`, and `score_matrix` estimates the spectral radius with

```python
vals = scipy.sparse.linalg.eigsh(sym, k=min(6, n - 2), which="LM", return_eigenvectors=False)
rho = float(np.abs(vals).max())
return (matrix * (rho_target / rho)).tocsr()
```

`eigsh` starts from a **random** vector because no `v0` is supplied. The estimate therefore varies,
and because the whole matrix is then multiplied by the float64 scalar `rho_target / rho`, a change
in the last bits of `rho` rewrites **every byte** of `A.data`, and with it the digest.

Measured, on one and the same substrate with one and the same code:

```
rho across 5 eigsh calls   2808.471257603490
                           2808.471257603497
                           2808.471257603492
                           2808.471257603495
                           2808.471257603496
                           -> 5 distinct values, relative spread 2.3e-15

A_hash across 5 runs       f5f07d3771a62c4b...   dad9a5bb63a64e75...
                           8ffacf6517c36ad2...   3aa95745aea38ea8...   <- the frozen value
                           bc9d9420c491185a...
                           -> 5 distinct values
```

**One of the five is the frozen `A_hash`.** The frozen value is therefore a single sample of a
stochastic process, not a reproducible identity of the substrate. No one — including whoever
produced the v3 record — can reproduce it deterministically; a reproduction attempt succeeds only
by chance, at roughly the rate of drawing the same float64 tail twice.

## 3. The correction

**`A_hash` is reclassified: it is not part of F1's structural identity and it may not be used as a
reproduction criterion.** F1's identity check, all of whose terms reproduced exactly, is:

```
n_nodes             == 1000
n_edges             == 80443
node_list_sha256    == 77ec9bfd90d6ec7216f642e1860c815ca972fd7638f1bb464766b9053faccb1e
edge_list_sha256    == a840f45bd5c96ce654f8ff9c5957b0ec3eb158527a417ff635b462e03815740a
B_hash at Din=5     == 3bd78eaca10cad865749a6bb1d04871d83a48af48b83ee5d1000e9ca39e7d968
```

Amendment 4 §5's `A_sha256` line is superseded by this list. The mismatch is **recorded, not
waived**: it is carried in `F1_materialization.json` together with this explanation, so a reader sees
the disagreement and its cause rather than a silently dropped check.

## 4. The latent defect this exposes, and its scope

`F1_MATERIALIZATION_REQUIREMENT` in `resaudit/families.py` cites the substrate's identity as *"the
substrate's A/B are recorded in Git only as content hashes (A_hash …, B_hash …)"*. Under this
finding, **the `A_hash` half of that sentence cannot serve as an identity**, because it hashes a
`eigsh`-normalized matrix. The requirement's intent — that F1 not be silently replaced by a
stand-in — is fully served by the wiring-level hashes, which are line-level and deterministic. The
defect is recorded here rather than edited into the frozen requirement.

**Scope: this does not propagate to Stage 1 measurements.** The resaudit battery normalizes with its
own declared-seed block power iteration, and that path was verified deterministic before this
amendment was written:

```
battery.spectral_radius_of(A)      4 calls -> 1 distinct value (3.047504526370813)
battery.scale_to_spectral_radius(A) 4 calls -> 1 distinct hash  -> deterministic
```

The instability is confined to the v3 `score_matrix` used to produce the old selection record. A3,
A4 and A5 in Stage 1 are computed on `FROZEN_RHO_TARGET = 0.95` via the deterministic path, so no
Stage 1 number is affected.

## 5. Effect

| Item | Effect |
|---|---|
| A1–A5 thresholds | **none** |
| Probe / drive | **none** |
| F1 structural identity | **narrowed** — wiring-level hashes only; `A_hash` removed as a criterion |
| F1 materialization | **authorized to proceed**, on the reproduced wiring |
| Candidates admitted or excluded | **none** |
| Stage 1 numbers | **unaffected** — the battery's normalization is deterministic |
