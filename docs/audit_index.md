# M4 Audit — index and closure point (v1.x)

**Read this first if you are picking the project up.** It pins the v1.x evidence
chain to a single state so that later commits, experiments and v2 work cannot
quietly move the baseline. Every number below is a quotation from a committed
artefact, and every digest is reproducible from the repository.

Frozen 2026-09-23. Branch `m4-audit/v1.5-evidence-identity`.

---

## Status line — the one sentence that must not be lost

> **M4-v1.4 is retained as an invalid test of the intended biological hypothesis
> due to construct-validity failure; v1.5–v1.5.2 repair evidence identity and gate
> semantics only, not the biological architecture.**

The evidence for the construct-validity failure — not a "negative result":

| measurement | value | source |
|---|---|---|
| ORNs in the substrate | **0** at N<=1000, 0.47 % at N=4000 | P1-3, `A8_subgraph_degeneracy.json` |
| substrate structure at N=250 | **126 edges**, **64.8 % isolated**, 0.09 % of out-edges retained | A8 |
| recurrent share of the drive | **R_t = 0.026** mean / 0.056 final; memory survives A := 0 bit-identically | A3 |
| state's effective rank | tracks **channels**, not N (3.09/3.14/3.10 for C=3 at N=250/1000/4000) | A2/A6 |
| does the state beat the raw window | **no** — H 0.374 vs X 0.479, and X+real-H 0.392 < X+shuffled-H 0.490 | A2r/A4r |
| delivered R0-vs-R2 evidence | byte-identical metrics in **570/620** units; a dense random graph matches R0 in **561/620** | A10 |

Full argument: [`docs/m4_audit_reservoir_integrity.md`](m4_audit_reservoir_integrity.md)
(section *Locked wording*, and the verdict table at the top).

---

## 1. The closure backup — `full-verify-20260923T063242Z/`

Taken after v1.5.2 merged and Gate A was recomputed.

| | |
|---|---|
| On the GPU box | `/root/autodl-tmp/drososense/backups/full-verify-20260923T063242Z/` |
| On the delivery side | `results/audit/m4_audit/server_evidence/full-verify-20260923T063242Z/` |
| Code identity at capture | `52637fc986aa03c9460f4b880ea94ea59b37c1aa` — *fix(audit): report the per-dataset counts in the comparison* |
| Record counts | **21,860** run records (`record_counts.txt`) |
| Derived tables | **65** files, every one digested (`tables.sha256`) |
| Protocol digests | all seven, cross-checked against the committed sidecars (`protocols.sha256`) |

Contents: `HEAD.txt`, `protocols.sha256`, `record_counts.txt`, `tables.sha256`,
`repo-results-configs.tar.gz` (code + results + configs, 6.0 MB),
`results-raw.tar.gz` (all 21,860 records, 3.2 MB), `branch.bundle` (4.2 MB). The
archives are git-ignored; the four manifests are committed and are what a restored
copy is verified against.

**Verification state:** the seven protocol files on the box hash identically to
the committed sidecars (`v1.1` → `v1.5.2`, all SAME), and the local copy of
`results-raw.tar.gz` lists **21,860** `.json` members. The `21,860` figure is
`21,867` minus the 7 `e2_smoke` records — see §4.

---

## 2. Protocol chain v1.1 → v1.5.2 (frozen, sidecar-matched)

An amendment **adds** a version file; it never edits a frozen one. Every file below
is on disk byte-for-byte as frozen, with a `configs/<name>.sha256` sidecar whose
digest matches it. The **loadable** base protocol remains `protocol_v1.3.yaml`
(`drososense/utils/paths.py: PROTOCOL_PATH`); the later files are amendment records
layered on it, and the runners' `PROTOCOL_VERSION` label carries the newest one.

| version | sha256 | what it is |
|---|---|---|
| `protocol_v1.yaml` | `0eddb56033b53ce5cfa855b1212c3c91336282f1f59226141f6722d2d21f6c15` | v1 — predates the sidecar convention |
| `protocol_v1.1.yaml` | `b87ed6057962b00e41c175acd7c78f314e552c50e2eecf53174e6156d8f7c0b9` | fields, contrasts, gates, narrative rules |
| `protocol_v1.2.yaml` | `32c57f6efaa8f9f14584d3a153fddc2814c5be8829eb888fc0d6c27d536a0cc1` | decision machine (gate symbols, cluster bootstrap) |
| `protocol_v1.3.yaml` | `d85640e556db6c00c8217befda17c967772c30470be9fbe73b2051b3cbfe8357` | **active/loadable**: D3 LOSO(62) per OD1(a) |
| `protocol_v1.4.yaml` | `4f4504528f02df53bc7585f7bac4abe9f4e6bb66301ec9e74f7e36ce47c42d89` | §17: only `status == "ok"` occupies the touch quota |
| `protocol_v1.5.yaml` | `c220a54fabcb1e042d526224c2004d3a72241e8bf7b04057739431d0b9c63ae2` | **evidence-unit identity, schema 2** (§3) |
| `protocol_v1.5.1.yaml` | `8ed71ceeca7a4cf333b1148fd88b83ef1ef61ee319c424401de618550d91fb21` | **parameter evidence scope**, fail closed (§5) |
| `protocol_v1.5.2.yaml` | `f9bee020fdd14179e9cd0d6070a10d6758b0e78881c6c6df13254d59b9155027` | **dataset-conditioned parameter gate semantics** (§5) |

---

## 3. E9 identity migration — the disposition identity closes at 420

`results/evidence_migration/` — `e9_v14_to_v15_manifest.csv` (420 rows, exactly one
state each), `e9_adopted.jsonl` (350), `e9_rejected.jsonl` (70),
`e9_migration_audit.json`.

```
adopted=350 + superseded=0 + duplicate_test_touch=70 + invalid=0 + missing=0 = 420   [CLOSED]
```

- 420 records = 70 legacy fingerprints x 6 records; 350 evidence units = 70 x 5 substrates.
- The duplicated unit is **N=250, scored twice**: `e9_size_d2` (config `2b681f4040fc`, 06:07:31Z) was **kept**; `e9_size_d2_n250` (config `e2dbcef1dad6`, 06:52:35Z) is the **rejected duplicate test touch**.
- **Designation rule, frozen and result-blind:** earliest `timestamp_utc` wins, ties broken by the lexicographically smaller `config_hash`. Stated before any metric was examined.
- It is a migration of **identity only**: all 350 adopted records' metrics, duration, timestamp, status, config_hash and counts are byte-identical to the originals, each carrying the origin file's sha256. No model was re-fit, no test split touched.
- `fingerprint_collision_only` is **not** a record disposition (a collision is a property of the legacy *fingerprint*); it is carried as the boolean column `unit_identity_collided`.

**Do not re-run E9.** Its scientific result is already citable from the delivered
records: macro-F1 falls from 0.314 (N=250) to 0.242 (N=4000) and MAE rises from
1.69 to 6.26 — adding connectome is harmful on this design.

---

## 4. `e2_smoke` — permanent status

> **7 per-run records irrecoverably lost; surviving derived summary and
> data-contact evidence retained; no re-evaluation permitted under §17.**

Established, not assumed: `results/tables/e2_smoke_summary.csv` records all seven
rows as `dataset=d2_beef_uncontrolled`, **`evidence_class=real`**,
`protocol_compliant=True`, one specimen evaluation each; and
`data_contact_log.json` carries an `e2_smoke` entry with
**`counts_as_first_test_evaluation: true`**. It was a **real D2 test touch**, so
re-running it would re-score a real test split. Its derived summary is the
surviving evidence and is committed. A synthetic smoke test for CI, if wanted,
must be a **new** label on synthetic data — never this label, never on D2.

The loss itself, the near-miss and the recovery are recorded in
[`docs/incident_2026-09-23_run_records.md`](incident_2026-09-23_run_records.md),
including the correction that the first recovery pass checked only `results/raw`
and missed five derived tables.

---

## 5. Gate A — final closure, with the evolution chain

Computed by the frozen-expression gate engine, not written by hand
(`ops/audit/gate_a_scope_comparison.py`). Both terms and both resolutions come from
`scripts.analyze.evaluate_rules`.

| term | result |
|---|---|
| **A1** — R0 not meaningfully worse than R4 on D2 and D3 | **True** |
| **A2** — `∀ d ∈ {D2, D3}: params(R0, d) < params(GRU, d)` | **True** |
| **Gate A** (A1 and A2 combined by the frozen expression) | **True** |

| dataset | params(R0) | params(GRU) | A2 |
|---|---:|---:|---|
| D2 | 1004 | 4452 | PASS |
| D3 | 1004 | 4068 | PASS |
| aggregate | — | — | PASS iff both PASS |

**The evolution chain, which is the part that matters:**

| state | Gate A | why |
|---|---|---|
| M4-v1.4 (unscoped scan) | **False** | `params(R0) = 16004` — an **E9 N=4000** readout. The gate failed for a reason unrelated to the comparison, and its verdict depended on which unrelated experiments were on disk. |
| v1.5.1 (declared scope, no dataset conditioning) | **UNEVALUABLE** | `params(GRU)` is 4452 on D2 and 4068 on D3 within one scope, and the protocol's single dataset-agnostic term had no unique value. Reported, never resolved by a maximum. |
| v1.5.2 (dataset-conditioned semantics) | **True** | Per-dataset matched configurations, folded with AND over `evaluated_on`. |

Wording that must be used for A2 (not the old single-value story):

> **R0 has fewer trainable parameters than GRU under each registered evaluation
> configuration on both D2 and D3.**

---

## 6. Open — carried forward, not dropped

These are **not** closed by v1.x and must not be lost when v2 starts:

1. **A9 — the decisive test clusters by the wrong unit.** 62 of 62 D3 `fold_id`s
   map to a different specimen under each seed, so the cluster unit is a fold index,
   not a specimen. A statistics-layer amendment (declare the cluster unit as the
   **specimen**) is still owed; it needs no re-run. Recorded in the audit report's
   disposition table and in §A9.
2. **Run-record field truthfulness** — `params.reservoir_size = 200` while
   `topology.n_nodes = 250`; node-selection provenance (`target_n`, seed, sha256)
   is not recorded; `class_coverage` lives only in the git-ignored raw records, so
   the macro-F1 ceiling cannot be verified from the delivery line.
3. **The biological architecture** — the whole of v2. See §7.

---

## 7. Where the rest lives

| artefact | what it holds |
|---|---|
| `docs/m4_audit_reservoir_integrity.md` | the audit: verdict table, A1–A11, the real-data half, the locked wording, the disposition table |
| `docs/incident_2026-09-23_run_records.md` | the `results/raw` loss, its recovery, and the rule it produced |
| `docs/merge_drop_policy.md` | why the GPU tree's DATA-52 `cpu_seconds` deletion was rejected rather than merged |
| `ops/audit/reservoir_integrity_audit.py` | A1/A7/A8 (graph level, real NPZ) |
| `ops/audit/reservoir_dynamics_audit.py` | A2–A6 (synthetic probe) |
| `ops/audit/reservoir_realdata_audit.py` | A2r/A4r/A5r (real D2/D3 windows) |
| `ops/audit/evidence_ledger_repair_audit.py` | the E9/E3 collision-vs-repeat classification |
| `ops/audit/e9_identity_migration.py` | the v1.4→v1.5 migration and its closure |
| `ops/audit/gate_a_scope_comparison.py` | the pre/post Gate A comparison |
| `results/audit/m4_audit/` | every evidence JSON, the ledger-repair outputs, `server_evidence/` |
| `results/evidence_migration/` | the E9 migration artefacts |

### Reproducing the closure in one command

```bash
python ops/audit/gate_a_scope_comparison.py     # Gate A, both terms, both resolutions
python -m pytest tests/ -m "not slow"           # 540 passed, 8 skipped (local)
```

On the GPU box the same suite reports **548 passed, 0 failed, 0 skipped**.

### Known local-suite qualifications

Removing the local connectome data (moved to the box, verified, then deleted —
12.2 GB freed) makes two real-NPZ tests **skip** with the reason *"olfactory_v1.npz
not provisioned in this environment"*: the local suite is 540/8 rather than
542/6. They run on the box. Three xgboost tests skip locally because `libomp` is
absent.

---

## 8. Next

```
audit index (this file)  ->  v2 preregistration  ->  validation-only construct checks  ->  formal experiments
```

v2 is the biological architecture redesign, and its four hard constraints are
pre-registered before any v2 code exists:

1. **ORN/PN-aligned input population** — not a dense random `W_in` over every node.
2. **Connected olfactory subgraph with preserved biological edges** — not a
   126-edge residue with 64.8 % isolated nodes.
3. **Recurrent/input contribution ratio as an explicit design gate** — not 3–5 % by
   accident.
4. **R2 as a true wiring-only counterfactual** — preserving weight statistics and
   the input/output boundary, rather than a counterfactual that also flattens the
   weights.

None of this is a re-tuning after seeing a test result: it repairs a
construct-validity failure established by an independent audit of the
implementation and the delivered records, before any v2 design decision was taken.
