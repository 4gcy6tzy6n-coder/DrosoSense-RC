# DrosoSense-RC

**Drosophila Sensory Connectome Reservoir Computing for Lightweight AgriFood Sensing**

A frozen sparse biological reservoir with a lightweight trained readout, investigated as a
computational substrate for electronic-nose food-quality monitoring.

---

## Status: M1 — protocol frozen, connectome-free benchmark delivered

This repository is at **M1**. It contains:

- a frozen, machine-readable experimental protocol (`configs/protocol_v1.yaml`);
- a leakage-audited data layer: specimen-level splits, train-only standardisation,
  specimen-bounded windowing, and active leakage audits with positive controls;
- a nine-model baseline zoo behind one interface;
- traceable dataset manifests for D1/D2/D3 with acquisition scripts and checksums;
- a working benchmark pipeline, verified end to end on a synthetic fixture and on
  both acquirable real datasets.

### What M1 does NOT claim

There is **no connectome in this repository**. The biological reservoir (R0) and its
topology controls are M2/M3 work. Accordingly:

> We investigate whether a Drosophila olfactory connectome can serve as a fixed sparse
> computational substrate for lightweight food-quality sensing.

That is the only claim this stage supports. Any statement that a connectome reservoir
outperforms, or is competitive with, a baseline is **forbidden** here
(`configs/protocol_v1.yaml` → `scope_boundaries.m1_claims_forbidden`).

The stronger claim — *"the biological topology provides a useful inductive bias under
limited-data and sensor-degraded conditions"* — may be used only after Gate B passes on a
real dataset with the primary test.

---

## Quick start

```bash
# 1. Environment
conda env create -f environment.yml && conda activate drososense
#    or: python -m pip install -r requirements-dev.txt

# 2. Run the test suite (hermetic, offline, no datasets required)
python -m pytest tests/ -m "not slow"

# 3. Acquire the real datasets that are publicly downloadable
python scripts/download_data.py --all

# 4. Smoke test the whole pipeline on the synthetic fixture
python scripts/make_fixture.py
python scripts/run_baselines.py --dataset synthetic_enose --experiment smoke \
    --models all --tasks classification regression --seeds 0 1 \
    --window-lengths 16 --max-folds 1 --smoke

# 5. Aggregate raw records into a summary table
python scripts/summarize.py --experiment smoke --print
```

> **macOS / XGBoost.** XGBoost needs the OpenMP runtime. Install it with
> `brew install libomp`, or point `DYLD_FALLBACK_LIBRARY_PATH` at a `libomp.dylib` from
> another package. If it cannot load, `run_baselines.py` reports XGBoost as skipped with
> the reason — it never silently disappears from the comparison.

---

## The frozen protocol

`configs/protocol_v1.yaml` is the contract. It fixes, before any result was seen:

| Item | Value |
| --- | --- |
| Split unit | `specimen` — **red line** |
| Seeds | `0..9` |
| Classification primary / secondary | `macro_f1` / balanced accuracy, AUROC, accuracy |
| Regression primary / secondary | `mae` / RMSE, R² |
| Primary hypothesis | H1: R0 > R2 (degree-rewired) |
| Primary test | paired Wilcoxon signed-rank, two-sided, α = 0.05 |
| Multiplicity | Holm, within dataset |
| Window candidates | L ∈ {8, 16, 32, 64}, selected on **validation** |
| Standardisation | `x' = (x − μ_train)/(σ_train + ε)`, fitted on **train only** |
| Gates | A (viable), B (TAFE-worthy), C (strong paper) |
| Narrative rules | N1–N5, decided in advance |

**Freeze rule.** The primary metric and primary hypothesis may not be changed after
observing test results. A change requires a *new* file, `protocol_v2.yaml`, stating the
trigger and the change; runs completed under v1 stay reported under v1.

**Narrative adjustment rules** are pre-registered, not improvised. For example:
if `R0 ≈ R2` and both beat random sparse, the conclusion becomes *degree / statistical
structure matters* — it is forbidden to claim the specific *Drosophila* wiring matters.

---

## Data

Three real datasets are in scope. Their status is reported honestly rather than worked
around; the project never substitutes a different dataset for a missing one.

| ID | Dataset | Task | Size | License | Status |
| --- | --- | --- | --- | --- | --- |
| D1 | Beef-Controlled | 4-class freshness + TVC | 4553 × 15 | CC BY 4.0 | **Acquired**, checksum verified |
| D2 | Beef-Uncontrolled | 4-class freshness + TVC | 20815 × 14 | CC BY 4.0 | **Acquired**, checksum verified |
| D3 | Rainbow Trout | cross-food validation | 21.1 GB archive | CC BY 4.0 | **BLOCKED** — not fetched |

Full provenance, checksums, schemas and manual steps: `data/manifests/*.yaml`.

### ⚠️ The specimen-identifier blocker (D1 and D2)

**Neither beef dataset publishes a specimen identifier.** This is the single most
important finding of M1, and it blocks the protocol's red line on the two datasets the
project most depends on.

Verified on the downloaded files:

- **D1** — `minute` runs 1…4553, strictly increasing, 4553 unique values, **zero resets**.
- **D2** — both `sl no` and `minutes` strictly increasing, **zero resets**.

A dataset whose time axis never restarts is one continuous measurement, not a
concatenation of independent samples. There is no column identifying which beef sample a
row came from, and no session boundary to recover one from.

Two further consequences, both verified:

1. **The class label is derived, not observed.** In both datasets `class` is a
   deterministic threshold of `TVC` at 3.0 / 4.0 / 5.0 log₁₀ CFU/g. Classification and
   regression are two views of the *same* variable, and must never be presented as two
   independent sources of evidence.
2. **D1 is heavily imbalanced** — 85 % of rows are `spoiled`. Macro-F1 is the primary
   metric precisely because accuracy would be dominated by the majority class.

**What was done instead of pretending.** The configs use
`specimen.source: assumed_time_block`, which cuts each series into contiguous time blocks
as a documented stand-in grouping. Every run record produced this way carries
`protocol_compliant: false` and a note saying so. These numbers are **pipeline
validation, not results**, and are labelled as such in every table.

The time-block stand-in is also *optimistic by construction*: D2's classes occupy
contiguous time blocks (transitions at rows 2939, 5207, 7499), so a held-out block
resembles its neighbours. High scores under this split are an artefact of the assumption,
not evidence of sensing performance. This is exactly why the split is not reported as a
result.

**To unblock:** obtain the per-sample acquisition log from the dataset authors (which
rows belong to which sample). Until then, the honest options for D1/D2 are to report them
under an explicitly non-compliant split, or to drop them from protocol results — both
stated in the manifests.

---

## Repository layout

```text
configs/          protocol_v1.yaml + dataset, reservoir and experiment configs
data/
  raw/            acquired files (git-ignored)
  processed/      normalised frames (git-ignored)
  splits/         per-fold split.json + scaler.pkl (git-ignored)
  manifests/      provenance, checksums, licenses, blockers (committed)
drososense/
  data/           schema, loaders, splits, scaling, windowing, leakage, synthetic
  baselines/      classic (SVM/RF/XGBoost/PCA+SVM) and sequence (GRU/LSTM/CNN/TCN)
  reservoir/      ESN — the R4 control, NOT a connectome
  evaluation/     metrics, run records, aggregation, the driver
scripts/          download, fixture, preprocess, benchmarks, summarize
tests/            unit + integration; leakage audits carry positive controls
results/
  raw/            one JSON per (model, task, seed, fold) — git-ignored
  tables/         aggregated summaries — committed
```

**Raw data is never committed.** `.gitignore` excludes `data/raw`, `data/processed`,
`data/splits`, `connectome/raw`, `connectome/adjacency` and `results/raw`.

---

## Leakage control

Three channels are audited, and every audit has a **positive control** — a test that feeds
it deliberately leaked input and asserts it fires. An audit never observed to fail is
indistinguishable from one that cannot fail.

| Channel | Guarantee | Enforced by |
| --- | --- | --- |
| Split | No specimen in two of train/val/test | `audit_fold`, `Fold.__post_init__` |
| Preprocessing | Scaler statistics from train rows only | `audit_scaler` |
| Window | No window spans two specimens | `audit_windows` |
| Row reuse | No row on both sides | `audit_no_row_reuse` |

The sharpest test builds the same model on a **row-level** split and a **specimen-level**
split of the same fixture and asserts the row-level split scores higher — quantifying the
leakage that `split_unit: specimen` exists to remove
(`tests/test_leakage.py::test_row_level_split_scores_higher_than_specimen_level`).

There is **no row-level splitter anywhere in the data layer**; `drososense.data.splits`
never imports one, checked by an AST-based test.

---

## Models

All nine baselines implement one interface and receive identical tensors for a given
`(dataset, seed, fold, window length)`.

| ID | Family | Notes |
| --- | --- | --- |
| `svm_rbf` | classical | RBF kernel SVM/SVR |
| `random_forest` | classical | 300 trees |
| `xgboost` | classical | gradient-boosted trees |
| `pca_svm` | classical | PCA → RBF SVM |
| `gru` | sequence | torch |
| `lstm` | sequence | torch |
| `cnn1d` | sequence | torch |
| `tcn` | sequence | dilated **causal** TCN |
| `esn` | reservoir | leaky ESN, frozen reservoir + trained readout — **the R4 control** |

`--smoke` selects deliberately tiny hyperparameters. Smoke numbers are a pipeline check,
**not** tuned results, and are never reported as findings.

---

## Results schema

Every run writes one record to `results/raw/<experiment>/<dataset>/<model>/<task>_seed<k>_fold<i>.json`.
Abridged from a real record produced by the D1 validation run:

```json
{
  "run_id": "d1_beef_controlled|svm_rbf|classification|seed00|fold00",
  "experiment": "m1_real_validation",
  "dataset": "d1_beef_controlled",
  "model": "svm_rbf",
  "task": "classification",
  "seed": 0, "fold_id": 0,
  "protocol_version": "1.0.0",
  "window_length": 16,
  "metrics": {"macro_f1": 0.2442, "balanced_accuracy": 0.4774, "auroc": 0.9758,
              "accuracy": 0.7883, "auroc_n_classes_scored": 2},
  "n_train_windows": 2585, "n_test_windows": 940,
  "train_specimens": ["block_0005", "block_0007", "block_0017", "… (11 total)"],
  "test_specimens": ["block_0002", "block_0003", "block_0010", "block_0014"],
  "duration_s": 0.070,
  "environment": {"python": "3.12.13", "numpy": "…", "sklearn": "…"},
  "timestamp_utc": "…",
  "evidence_class": "real",
  "protocol_compliant": false,
  "model_description": {"n_trainable_parameters": null,
                        "proba_source": "softmax(decision_function)"},
  "fold_fingerprint": "68ff40326c30",
  "class_coverage": {"missing_in_train": [], "missing_in_test": [0, 1]},
  "config_hash": "bfe17b84180b",
  "notes": "split_unit: specimen is NOT satisfied — …"
}
```

Note `auroc_n_classes_scored: 2` and `missing_in_test: [0, 1]` in that record: with
85 % of D1's rows labelled `spoiled`, a held-out block can contain only a subset of the
classes. AUROC is then averaged over the classes actually present and says how many it
scored, rather than inventing values for classes with no test samples. Macro-F1, by
contrast, is always averaged over all four declared classes, so folds stay comparable.

Two fields make a number safe to quote, and both survive into the summary table:

- **`evidence_class`** — `real` or `synthetic_fixture`. Synthetic and real runs are never
  averaged together.
- **`protocol_compliant`** — whether the split actually satisfied `split_unit: specimen`.
  Compliant and non-compliant runs are never averaged together.

Aggregation additionally keeps the SD column present even when a group has one run, so an
undefined SD reads as `NaN` rather than vanishing.

---

## Reproducibility

- A run is reproducible from `(dataset, model, task, seed, fold, window_length)` alone.
- Seeds 0–9 drive the specimen permutation, so seeds vary the *split*, not just weights.
- Torch CPU reductions depend on thread count, so the runner pins `torch.set_num_threads`
  (default 1). This trades speed for the ability to re-run a seed and get the same number.
- The ARPACK spectral-radius estimate for the ESN uses an explicitly seeded start vector;
  without this, two runs of the same seed diverge.
- Environments are recorded per run and summarised in `results/tables/environment.json`.

---

## Testing

```bash
python -m pytest tests/ -m "not slow"            # full suite
python -m pytest tests/ -m unit                  # fast, hermetic
python -m pytest tests/ --cov=drososense --cov-report=term-missing
```

Coverage is reported in the run log; the suite is required to stay above 80 %.

---

## Known limitations

1. **No specimen-level split is possible for D1/D2 as published.** The central limitation.
2. **Class labels are thresholds of TVC** in all three datasets, so classification is a
   discretisation of the regression target.
3. **D3 is unverified and unfetched**; its declared schema is an expectation pending
   confirmation against the archive, and its config carries `schema_verified: false`.
4. **D2 row 2827** has `Temperature = 0` with a simultaneous ~85-unit humidity jump — a
   logging discontinuity. Retained and left visible; repairing it needs a protocol
   amendment, not an ad-hoc edit.
5. **`class` is time-determined in D2**, so any temporal split is optimistic.
6. **Smoke hyperparameters are untuned.** Reported comparisons require the full tuning
   and seeding procedure in the protocol.
7. **Probability values are not calibrated.** Where a classifier exposes only a decision
   function, AUROC consumes a softmax of it — monotone, so ranking is unaffected, but the
   values are not confidences.

---

## Roadmap

| Stage | Content | Depends on |
| --- | --- | --- |
| **M1** ✅ | Protocol frozen; leakage-audited benchmark; manifests | — |
| M2 | FlyWire/Codex olfactory subcircuit → `olfactory_v1.npz` + network report | M1 |
| M3 | Frozen connectome reservoir + topology controls R0–R6 | M2 |
| M4 | E1–E6 experiments, paired statistics, Gates A/B/C | M3 |
| M5 | Edge replay, efficiency, real e-nose proof-of-system | M4 |

`scripts/build_connectome.py`, `run_reservoir.py` and `run_ablation.py` currently print
their plans and exit non-zero rather than half-implementing M2/M3.

### Wording constraint carried forward

Synapse counts are not synaptic strengths. The connectome weight matrix must be described
as a **synapse-count-informed structural weight**, never as true biological synaptic
strength.

---

## License and attribution

Code in this repository is the project's own. Datasets retain their published licenses
(all CC BY 4.0); see each manifest for the required citation. Raw data is not
redistributed here.
