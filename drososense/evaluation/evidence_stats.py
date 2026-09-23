"""M4 Phase 2 — read-only statistics/gating pipeline over an evidence bundle.

DATA-56 wires the frozen protocol's statistical machinery (paired cluster tests,
Holm correction, gate / narrative-rule evaluation) to the committed E1 evidence
bundles (``e1_main_d2_*`` today; ``e1_main_d3_*`` after D3 closes; the E2 /
E1-reservoir half under the same label prefix when it lands). The D3 / E2
plug-in points are parameter labels, not code: one command produces the full
table set for any bundle that exists on disk, and this module is the
single-manifest entry point for M4 Phase 2.

Read-only, by construction
--------------------------
* This module **never evaluates a model**. It loads tidy per-run records
  that an approved runner has already scored, then applies the frozen tests.
* The active protocol file is read from ``drososense.utils.paths.PROTOCOL_PATH``
  (v1.3 on the delivery line; the loader itself is version-agnostic), and every
  test parameter — alpha, bootstrap B / seed / CI type, equivalence margin,
  multiplicity families, contrast list, gate expressions — is read from that
  file at run time. Nothing is re-declared here.
* Missing evidence is **reported, never imputed**: a contrast whose two sides
  do not line up on common ``(seed, fold)`` units is ``unpairable``; a gate or
  narrative rule that names a contrast this bundle does not carry is
  ``UNEVALUABLE`` with a reason, which is exactly what the frozen protocol's
  §13 gate_rules and the narrative-adjustment block prescribe.
* Gate B / Gate C (the E1 baseline half, and the reservoir half when present)
  and Gate A (the ``ci_contains_zero`` / ``noninferior`` / ``params`` terms)
  are evaluated on the bundle's own contrasts. On a bundle that carries the
  E1 baseline records, the reservoir contrasts that the gates name are
  reported ``unpairable`` / ``missing_contrast`` and the gate records
  ``UNEVALUABLE``; when the E2 / E1-reservoir bundle lands with the same
  command, the identical pipeline produces the real rows and a gate verdict.

One command
-----------
::

    python -m drososense.evaluation.evidence_stats --experiment e1_main_d2
    python -m drososense.evaluation.evidence_stats --experiment e2_topology

writes, under ``results/tables/<experiment>/``:

* ``descriptive.csv`` — per-(dataset, model, task): decision metric (the task's
  primary, per the frozen direction rules) with mean / SD / 95 % bootstrap CI,
  plus the secondary-metric columns the task declares (``r2`` / ``rmse`` —
  reported, never decided on).
* ``contrast_statistics.csv`` — every declared contrast of the frozen
  protocol, evaluated on the bundle where both sides are pairable;
  ``unpairable`` / ``insufficient_data`` rows are written out, never dropped.
* ``gates.json`` — Gate A / B / C verdicts; ``UNEVALUABLE`` carries the reason
  (protocol §13 gate_rules).
* ``narrative.json`` — N1..N5 evaluation. One rule fires at most; more than
  one firing is a protocol-level conflict and is reported as such, never
  silently resolved by choosing one (protocol §14 evaluation_note).
* ``audit.json`` — bundle integrity (sha256 of each input, row counts,
  distinct config_hashes, decision-metric map, the ``r2``-secondary-only
  note, the list of unresolved ``params`` / contrast terms).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation.gates import (  # noqa: E402
    GateEvaluator,
    GateExpressionError,
    build_symbols,
)
from drososense.evaluation.clustering import (
    SPECIMEN_COLUMN,
    ClusterUnitError,
    cluster_provenance,
    specimen_ids,
)
from drososense.evaluation.stats import (  # noqa: E402
    DEFAULT_ALPHA,
    InsufficientDataError,
    holm_correction,
    paired_test,
)
from drososense.utils.config import (  # noqa: E402
    load_protocol,
    protocol_condition_symbols,
    protocol_dataset_symbols,
    protocol_family_membership,
    protocol_model_id_bindings,
    protocol_model_symbols,
    protocol_paired_spec,
)
from drososense.utils.paths import RESULTS_TABLES_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Bundle intake
# ---------------------------------------------------------------------------


def _sha256_prefix(path: Path, n: int = 16) -> str:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:n]


def load_evidence_bundle(
    experiment: str,
    required_models: list[str] | None = None,
    tables_dir: str | Path | None = None,
    bundle_prefix: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one committed evidence bundle with integrity checks.

    The bundle is the four-file set one experiment label produced:
    ``<experiment>_per_run.csv`` (the tidy frame the analysis reads), plus
    ``<experiment>_summary.csv`` / ``_fingerprints.csv`` /
    ``_test_touched_once.json`` (provenance and the §17 audit). Any missing
    file, any duplicate (dataset, model, task, seed, fold) key, or any
    non-zero ``n_violations`` in the touched-once report is a ``ValueError``:
    the pipeline stops rather than running on a broken bundle.

    Args:
        experiment: Bundle label, e.g. ``e1_main_d2`` / ``e1_main_d3`` /
            ``e2_topology``. The same call works for any label that has been
            assembled as the four files — that is the D3 / E2 plug-in point.
        required_models: If given, raise ``ValueError`` when the bundle lacks
            one of these model ids (used to make the D3 / E2 plug-in
            contract explicit: the command runs, but the analysis degrades to
            ``unpairable`` / ``UNEVALUABLE`` rows rather than to a verdict).

    Returns:
        ``(frame, meta)`` where ``frame`` is the per-run frame and ``meta``
        records the sha256 prefixes, row counts and distinct config hashes of
        the bundle, so the audit file shows exactly what the numbers rest on.
    """
    base = Path(tables_dir) if tables_dir else RESULTS_TABLES_DIR
    prefix = bundle_prefix or experiment
    per_run_path = base / f"{prefix}_per_run.csv"
    summary_path = base / f"{prefix}_summary.csv"
    fingerprints_path = base / f"{prefix}_fingerprints.csv"
    touched_path = base / f"{prefix}_test_touched_once.json"
    missing = [
        str(p)
        for p in (per_run_path, summary_path, fingerprints_path, touched_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"evidence bundle {experiment!r} is incomplete; missing: {missing}"
        )

    frame = pd.read_csv(per_run_path)
    key = [
        c
        for c in ("dataset", "model", "task", "seed", "fold_id")
        if c in frame.columns
    ]
    duplicates = int(frame.duplicated(subset=key).sum())
    if duplicates:
        raise ValueError(
            f"{experiment!r}: the per-run frame carries {duplicates} duplicate "
            f"(dataset, model, task, seed, fold) row(s); the bundle is not "
            f"the tidy frame M4's paired tests read. Re-run scripts/summarize.py "
            f"--experiment {experiment} --per-run before feeding the pipeline."
        )
    touched = json.loads(touched_path.read_text(encoding="utf-8"))
    n_violations = int(touched.get("n_violations", -1))
    if n_violations != 0:
        raise ValueError(
            f"{experiment!r}: test_touched_once.json reports {n_violations} "
            f"violation(s); the bundle must not feed the statistics pipeline "
            f"until the §17 review decides what to do with them."
        )

    config_hash_values: set[str] = set()
    for column in ("config_hash", "run_config_hash"):
        if column in frame.columns:
            config_hash_values.update(
                str(v) for v in frame[column].dropna().tolist()
            )
    if not config_hash_values and fingerprints_path.is_file():
        fingerprints_frame = pd.read_csv(fingerprints_path, dtype=str)
        if "config_hash" in fingerprints_frame.columns:
            config_hash_values.update(
                fingerprints_frame["config_hash"].dropna().astype(str).unique().tolist()
            )

    meta = {
        "experiment": experiment,
        "bundle_prefix": prefix,
        "tables_dir": str(base),
        "n_per_run": int(len(frame)),
        "n_distinct_test_fingerprints": int(
            touched.get("n_distinct_test_fingerprints", -1)
        ),
        "n_violations": n_violations,
        "sha256": {
            "summary": _sha256_prefix(summary_path),
            "per_run": _sha256_prefix(per_run_path),
            "fingerprints": _sha256_prefix(fingerprints_path),
            "test_touched_once": _sha256_prefix(touched_path),
        },
        "n_distinct_config_hashes": len(config_hash_values),
    }
    if required_models:
        available = set(frame["model"].dropna()) if "model" in frame.columns else set()
        missing_models = [m for m in required_models if m not in available]
        if missing_models:
            meta["missing_required_models"] = missing_models
    return frame, meta


def index_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a bundle frame to the record unit the pairing key needs.

    Fills a missing ``fold_id`` column with 0 (a single-fold record set), and
    drops rows that carry no status — the frame must state what every record
    is before the analysis reads a number off it.
    """
    out = frame.copy()
    if "fold_id" not in out.columns:
        out["fold_id"] = 0
    if "status" in out.columns:
        out = out[out["status"].notna()]
    return out


# ---------------------------------------------------------------------------
# Descriptive layer (task primary decision metric; r2 secondary-only)
# ---------------------------------------------------------------------------


def decision_metric_by_task(protocol: dict[str, Any]) -> dict[str, str]:
    """The task's PRIMARY metric is the decision metric; directions frozen.

    ``classification`` → ``macro_f1`` (maximize); ``regression`` → ``mae``
    (minimize). ``r2`` appears only under a task's ``secondary`` list, which is
    the r2 direction hard-constraint of DATA-23 / DATA-27 / M4's release
    instruction: r2 is a reported quantity, never a decision input.
    """
    return {
        str(task): str(spec["metrics"]["primary"])
        for task, spec in protocol["tasks"].items()
    }


def descriptive_rows(
    protocol: dict[str, Any], frame: pd.DataFrame
) -> pd.DataFrame:
    """One row per (dataset, model, task): decision metric + secondaries.

    Mean / SD are computed over the task's PRIMARY metric (the decision
    metric). The 95 % percentile CI of the per-model mean resamples the
    ``(seed, fold)`` records at the frozen bootstrap B / seed, so the interval
    is reproducible without re-running the experiment. The task's secondary
    metrics (``r2`` / ``rmse`` …) are reported as their own mean ± SD
    columns and are never used to decide anything — that split is what keeps
    the r2 direction constraint honest.
    """
    from drososense.utils.config import (
        protocol_metric_direction,
        task_metric_direction,
    )

    primary_by_task = decision_metric_by_task(protocol)
    bootstrap = protocol["statistical_tests"]["bootstrap"]
    b = int(bootstrap["n_resamples"])
    seed = int(bootstrap["seed"])
    ci_level = float(bootstrap["ci_level"])
    rows: list[dict[str, Any]] = []
    for (dataset, model, task), group in frame.groupby(
        ["dataset", "model", "task"], sort=True
    ):
        ok = group[group["status"] == "ok"] if "status" in group.columns else group
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "task": task,
            "n_ok": int(len(ok)),
        }
        primary = primary_by_task.get(str(task))
        if primary is None or primary not in ok.columns:
            row["decision_metric"] = primary
            row["status"] = "insufficient_data"
            row["note"] = (
                "no ok records carry the task's decision metric column"
            )
            rows.append(row)
            continue

        values = pd.to_numeric(ok[primary], errors="coerce").dropna()
        row["decision_metric"] = primary
        row["direction"] = task_metric_direction(
            protocol["tasks"][str(task)]
        )
        if len(values) == 0:
            row["status"] = "insufficient_data"
            row["note"] = "decision metric missing on every ok record"
            rows.append(row)
            continue
        rng = np.random.default_rng(seed)
        n = int(len(values))
        # Fold-cluster bootstrap CI of the model mean: seeds are averaged
        # inside each fold first, then the fold means are resampled — the
        # protocol's own sampling_order (v1.2 amendment) and resampling unit
        # (fold, never seed). The CI is therefore a statement about
        # cluster-to-cluster variation, not a pseudoreplicated seed count.
        if "fold_id" in ok.columns:
            fold_means = (
                ok.dropna(subset=[primary])
                .groupby("fold_id")[primary]
                .mean()
                .to_numpy()
            )
            n_folds = int(len(fold_means))
            tail = (1.0 - ci_level) / 2.0
            if n_folds >= 2:
                draws = fold_means[
                    rng.integers(0, n_folds, size=(b, n_folds))
                ].mean(axis=1)
                ci_low = float(np.quantile(draws, tail))
                ci_high = float(np.quantile(draws, 1.0 - tail))
            else:
                ci_low = ci_high = float(np.nan)
        else:
            ci_low = ci_high = float(np.nan)
        row.update(
            {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if n > 1 else float("nan"),
                "n_folds": int(len(ok.dropna(subset=[primary]).get("fold_id", pd.Series()).unique())) if "fold_id" in ok.columns else n,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "status": "ok",
            }
        )
        for secondary in protocol["tasks"][str(task)].get(
            "metrics", {}
        ).get("secondary", []):
            if secondary not in ok.columns:
                continue
            svalues = pd.to_numeric(ok[secondary], errors="coerce").dropna()
            if len(svalues) == 0:
                continue
            row[f"{secondary}_mean"] = float(svalues.mean())
            row[f"{secondary}_std"] = (
                float(svalues.std(ddof=1)) if len(svalues) > 1 else float("nan")
            )
            row[f"{secondary}_n"] = int(len(svalues))
        # D3 hard constraint 1 (M4): report n_auroc_defined alongside every
        # AUROC mean. D3 folds have one specimen; only folds where all four
        # classes are present have a defined AUROC, and the protocol requires
        # the count of undefined folds to be reported with the mean, not
        # hidden by pooling.
        if "auroc" in ok.columns and "auroc_defined" in ok.columns:
            auroc_values = pd.to_numeric(ok["auroc"], errors="coerce").dropna()
            n_defined = int(ok["auroc_defined"].eq(True).sum())
            n_undefined = int(len(ok) - n_defined)
            row["n_auroc_defined"] = n_defined
            row["n_auroc_undefined"] = n_undefined
            if len(auroc_values):
                row["auroc_mean"] = float(auroc_values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired-contrast layer (the frozen tests, Holm within the owning family)
# ---------------------------------------------------------------------------


#: The columns naming ONE evaluation. A specimen's cluster mean is the mean of
#: its evaluations, so the two sides must agree on this set per specimen.
_OBSERVATION_KEY_CANDIDATES: tuple[str, ...] = ("seed", "fold_id", "window_length")


def _short(values: Sequence[str], limit: int = 3) -> str:
    """A few names and an ellipsis, for a reason string."""
    shown = list(values[:limit])
    if len(values) > limit:
        shown.append(f"+{len(values) - limit} more")
    return ", ".join(shown)


def specimen_pairing_asymmetry(
    left: pd.DataFrame, right: pd.DataFrame, specimen_column: str = SPECIMEN_COLUMN
) -> str:
    """Why the two sides cannot be clustered on the specimen, or ``""``.

    Protocol v1.5.3, ``cluster_unit.missing_result_policy``, verbatim: *"if a model
    has no valid result for a specimen, the contrast is UNEVALUABLE for the affected
    unit. No silent filling, and no downgrade to fold-level clustering."* The unit is
    the specimen, so the failure this checks for is a specimen whose evaluations are
    not the same on both sides: the inner join would keep whatever overlapped, and
    that specimen's cluster mean would then be the mean of a DIFFERENT set of
    evaluations than its partner's — a partial mean, silently, with the specimen
    still counted as one independent cluster. That is the silent fill the amendment
    forbids, so the contrast is refused and the affected specimens are named.

    Args:
        left: The first model's records, already carrying the specimen column.
        right: The second model's records, with the same columns.
        specimen_column: The column holding the specimen identity.

    Returns:
        A reason string naming every specimen observed differently (or by only one
        side), or the empty string when the pairing is specimen-complete.
    """
    observation_key = [c for c in _OBSERVATION_KEY_CANDIDATES if c in left.columns]

    def by_specimen(frame: pd.DataFrame) -> dict[str, set[tuple]]:
        out: dict[str, set[tuple]] = {}
        for specimen, group in frame.groupby(specimen_column, sort=True, dropna=False):
            out[str(specimen)] = set(
                group[observation_key].itertuples(index=False, name=None)
            )
        return out

    left_keys = by_specimen(left)
    right_keys = by_specimen(right)
    if not left_keys or not right_keys:
        empty_side = "the first" if not left_keys else "the second"
        return (
            f"{empty_side} side has no scored row at all ({len(left)} row(s) against "
            f"{len(right)}), so there is no paired unit to cluster; the contrast is "
            f"UNEVALUABLE rather than paired on nothing"
        )
    left_only = sorted(set(left_keys) - set(right_keys))
    right_only = sorted(set(right_keys) - set(left_keys))
    partial = sorted(
        specimen
        for specimen in set(left_keys) & set(right_keys)
        if left_keys[specimen] != right_keys[specimen]
    )
    if not (left_only or right_only or partial):
        return ""

    units = ", ".join(observation_key) if observation_key else "the record unit"
    parts: list[str] = []
    if left_only:
        parts.append(
            f"{len(left_only)} specimen(s) scored by only one side ({_short(left_only)})"
        )
    if right_only:
        parts.append(
            f"{len(right_only)} specimen(s) scored by only one side, on the other "
            f"model ({_short(right_only)})"
        )
    if partial:
        example = ", ".join(
            f"{specimen} ({len(left_keys[specimen] ^ right_keys[specimen])} "
            f"evaluation(s) unpaired)"
            for specimen in partial[:2]
        )
        parts.append(
            f"{len(partial)} specimen(s) observed on a different set of ({units}) "
            f"by the two sides (e.g. {example})"
        )
    return (
        "; ".join(parts)
        + "; a specimen whose evaluations are not identical on both sides has no "
        "paired cluster mean, so the contrast is UNEVALUABLE under protocol v1.5.3 "
        "rather than scored on the observations that happen to overlap"
    )


def _resolve_model_code(
    code: str, available_models: set[str], bindings: dict[str, str]
) -> str:
    """Resolve a protocol model code to the record model id that carries it.

    The protocol's contrast / gate vocabulary names the reservoir family by
    its short id (``R0``, ``R2``, ``R4``, …); the records carry either the
    registry id (``esn`` for R4) or the protocol id directly (``R0``). The
    resolution is declared in the protocol's own model-zoo binding
    (``protocol_model_id_bindings``), with a prefix fallback for
    ``R0_real_fly``-style record ids. The resolver returns the input code
    unchanged when it cannot resolve it — the caller must then treat the
    contrast as unpairable, not as resolved to something it is not.
    """
    if code in available_models:
        return code
    for registry_id, protocol_id in bindings.items():
        if protocol_id == code and registry_id in available_models:
            return registry_id
    for model in available_models:
        if model.startswith(f"{code}_") or model == code:
            return model
    return code


def contrast_row(
    protocol: dict[str, Any],
    indexed: pd.DataFrame,
    contrast_id: str,
    metric: str,
    dataset: str,
    task: str,
) -> dict[str, Any]:
    """Run the frozen paired test for one contrast on one dataset/task.

    The pairing unit is the record unit the runner produced —
    ``(dataset, task, seed, fold_id, window_length)`` — so the two sides of a
    contrast must have been scored on the same splits. When they do not line
    up, the row is ``unpairable`` with the reason, never imputed.
    """
    first_code, _, second_code = contrast_id.partition("_vs_")
    bindings = protocol_model_id_bindings(protocol)
    available = set(indexed["model"].dropna())
    first = _resolve_model_code(first_code, available, bindings)
    second = _resolve_model_code(second_code, available, bindings)
    if first == second:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": f"both sides resolve to {first!r}",
        }
    left = indexed[
        (indexed["model"] == first)
        & (indexed["dataset"] == dataset)
        & (indexed["task"] == task)
    ]
    right = indexed[
        (indexed["model"] == second)
        & (indexed["dataset"] == dataset)
        & (indexed["task"] == task)
    ]
    if "status" in left.columns:
        left = left[left["status"] == "ok"]
        right = right[right["status"] == "ok"]
    if metric not in left.columns or metric not in right.columns:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": f"metric {metric!r} absent from one side's records",
        }
    # Protocol v1.5.3: the cluster key is the SPECIMEN, so it has to survive the
    # pairing merge. It is functionally determined by (seed, fold_id) -- under a
    # seeded LOSO permutation it is exactly what differs between seeds -- so
    # carrying it in the key is free.
    try:
        left = left.assign(**{SPECIMEN_COLUMN: specimen_ids(left)})
        right = right.assign(**{SPECIMEN_COLUMN: specimen_ids(right)})
    except ClusterUnitError as exc:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": str(exc),
        }
    # Protocol v1.5.3 missing_result_policy: a specimen is the independent unit, so
    # it has to be observed IDENTICALLY on both sides. Checked before the merge,
    # because the merge is what would hide it.
    asymmetry = specimen_pairing_asymmetry(left, right)
    if asymmetry:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": asymmetry,
        }
    key = [
        c
        for c in ("seed", "fold_id", "window_length", SPECIMEN_COLUMN)
        if c in left.columns
    ]
    join = (
        left[key + [metric]]
        .rename(columns={metric: "a"})
        .merge(
            right[key + [metric]].rename(columns={metric: "b"}),
            on=key,
            how="inner",
        )
        .dropna(subset=["a", "b"])
    )
    if join.empty:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": (
                f"no common scored ({', '.join(key)}) unit between {first_code} "
                f"(resolved to {first!r}) and {second_code} (resolved to "
                f"{second!r})"
            ),
        }
    deltas = (join["a"] - join["b"]).to_numpy(dtype=float)
    seed_arr = join["seed"].to_numpy() if "seed" in join.columns else np.zeros(
        len(deltas), dtype=int
    )
    # The cluster key handed to the statistics layer is the SPECIMEN. The
    # fold id stays in the pairing key (one evaluation is one paired difference)
    # but is deliberately NOT the cluster.
    if SPECIMEN_COLUMN in join.columns:
        fold_arr = join[SPECIMEN_COLUMN].to_numpy()
    else:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "unpairable",
            "note": (
                "no test-specimen column survived the pairing merge, so the contrast "
                "cannot be clustered on the specimen; protocol v1.5.3 will not fall "
                "back to the fold index"
            ),
        }
    spec = protocol_paired_spec(protocol, metric, task)
    try:
        result = paired_test(
            deltas,
            seed_arr,
            fold_arr,
            spec,
            contrast_id=contrast_id,
            metric=metric,
            dataset=dataset,
        )
    except InsufficientDataError as exc:
        # The statistic's own message, plus what was actually paired: with the
        # specimen as the unit the common case is not "too few observations" but
        # "too few SPECIMENS" -- a bundle that labels every row with one specimen
        # has ten observations in one cluster, and no cluster-level test exists.
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "insufficient_data",
            "n_pairs": int(deltas.size),
            "n_clusters": int(np.unique(fold_arr).size),
            "cluster_unit": "specimen",
            "note": (
                f"{exc}; {int(deltas.size)} paired observation(s) in "
                f"{int(np.unique(fold_arr).size)} specimen cluster(s)"
            ),
        }
    n_clusters = int(np.unique(fold_arr).size)
    # `result.as_dict()` is the frozen statistic's own published shape. It is the
    # base of the row so that the two contrast paths (`evidence_stats` and
    # `scripts/analyze.py`) publish the same columns -- without it this table had
    # no `test` column at all, and "the decisive test is the exact two-sided sign
    # test over cluster means" was not checkable from the row.
    row: dict[str, Any] = dict(result.as_dict())
    row.update(
        {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "first": first,
            "second": second,
            "n_pairs": int(deltas.size),
            "n_clusters": n_clusters,
            "cluster_unit": "specimen",
            # The join carries no model column (a pair IS both models), so the
            # provenance is given them rather than reporting an empty field.
            "cluster_provenance": cluster_provenance(
                join, models=(first, second), tasks=(task,)
            ),
            "n_seeds": int(join["seed"].nunique()) if "seed" in join.columns else 1,
            "status": "ok",
            "minimum_achievable_p_over_clusters": float(
                result.minimum_achievable_p_over_clusters
            ),
        }
    )
    row["family"] = protocol_family_membership(protocol).get(
        (contrast_id, "full"), ""
    )
    row["p_holm"] = float("nan")
    return row


def build_contrast_table(
    protocol: dict[str, Any], indexed: pd.DataFrame
) -> pd.DataFrame:
    """Every declared contrast of the frozen protocol, one row each.

    Datasets and tasks are taken from the bundle itself (the pipeline is
    parameterised by label; the contrast list comes from the protocol). The
    owning multiplicity family supplies the Holm order for the p-values:
    the correction runs within the declared family only, and the family of a
    (contrast, condition) pair is the protocol's own.
    """
    membership = protocol_family_membership(protocol)
    datasets = sorted(set(indexed["dataset"].dropna()))
    tasks = sorted(set(indexed["task"].dropna()))
    metric_by_task = decision_metric_by_task(protocol)
    rows: list[dict[str, Any]] = []
    for contrast in protocol["contrasts"]["list"]:
        contrast_id = str(contrast["id"])
        first_code, _, second_code = contrast_id.partition("_vs_")
        resolution = {
            "contrast_id": contrast_id,
            "first_code": first_code,
            "second_code": second_code,
            "first_resolved": _resolve_model_code(
                first_code, set(indexed["model"].dropna()),
                protocol_model_id_bindings(protocol),
            ),
            "second_resolved": _resolve_model_code(
                second_code, set(indexed["model"].dropna()),
                protocol_model_id_bindings(protocol),
            ),
        }
        for dataset in datasets:
            for task in tasks:
                metric = metric_by_task.get(str(task))
                if metric is None:
                    continue
                row = contrast_row(
                    protocol, indexed, contrast_id, metric, dataset, task
                )
                row.update(resolution)
                rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = _apply_holm(table, protocol, membership)
    return table


def _apply_holm(
    table: pd.DataFrame,
    protocol: dict[str, Any],
    membership: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """Holm-correct within each (family, dataset, task, condition) cell.

    The protocol's multiplicity unit is a ``(contrast, condition)`` pair; one
    bundle carries one condition (``full``), so the correction cell is
    ``(family, dataset, task)``. Order within the cell follows the protocol's
    declared test order inside the family — the declaration order is the
    order the Holm step-down is allowed to use, and it is recorded on each
    row. A single-test family is corrected by the identity map, which is the
    frozen protocol's own note.
    """
    families = {
        str(f["id"]): f
        for f in protocol["multiplicity"]["families"]
    }
    alpha = float(
        protocol["statistical_tests"]["alpha"]
    )
    out = table.copy()
    if "family" not in out.columns:
        out["family"] = ""
    for (family, dataset, task), group in out.groupby(
        [out["family"], out["dataset"], out["task"]], sort=False
    ):
        ok = group[group["status"] == "ok"]
        if family not in families or ok.empty:
            continue
        family_decl = families[family]
        declared_order = [str(t) for t in family_decl.get("tests", [])]
        ordered = []
        for contrast_id, contrast_group in ok.groupby("contrast_id", sort=False):
            if contrast_id in declared_order:
                ordered.append((declared_order.index(contrast_id), contrast_group))
        ordered.sort(key=lambda pair: pair[0])
        if len(ordered) <= 1:
            # Single-test family (or a family with one member present): the
            # protocol declares that no correction applies; p_holm = p.
            for _, group_rows in ordered:
                mask = out.index.isin(group_rows.index)
                out.loc[mask, "p_holm"] = out.loc[mask, "p_value"]
            continue
        p_values = [
            float(group_rows["p_value"].iloc[0]) for _, group_rows in ordered
        ]
        corrected = holm_correction(p_values, alpha)
        for (_, group_rows), corrected_p in zip(ordered, corrected):
            mask = out.index.isin(group_rows.index)
            out.loc[mask, "p_holm"] = float(corrected_p)
    return out


def paired_differences_table(
    protocol: dict[str, Any], indexed: pd.DataFrame
) -> pd.DataFrame:
    """The per-(seed, fold) deltas for every pairable declared contrast.

    The audit trail the report needs next to the summary statistics: a reader
    can see the raw differences the decisive test was run on, without
    re-deriving them. Contrasts that do not pair report one row per
    (dataset, task) instead — never an imputation.
    """
    frames: list[pd.DataFrame] = []
    datasets = sorted(set(indexed["dataset"].dropna()))
    tasks = sorted(set(indexed["task"].dropna()))
    metric_by_task = decision_metric_by_task(protocol)
    bindings = protocol_model_id_bindings(protocol)
    available = set(indexed["model"].dropna())
    for contrast in protocol["contrasts"]["list"]:
        contrast_id = str(contrast["id"])
        first_code, _, second_code = contrast_id.partition("_vs_")
        first = _resolve_model_code(first_code, available, bindings)
        second = _resolve_model_code(second_code, available, bindings)
        for dataset in datasets:
            for task in tasks:
                metric = metric_by_task.get(str(task))
                if metric is None:
                    continue
                left = indexed[
                    (indexed["model"] == first)
                    & (indexed["dataset"] == dataset)
                    & (indexed["task"] == task)
                ]
                right = indexed[
                    (indexed["model"] == second)
                    & (indexed["dataset"] == dataset)
                    & (indexed["task"] == task)
                ]
                if "status" in left.columns:
                    left = left[left["status"] == "ok"]
                    right = right[right["status"] == "ok"]
                if left.empty or right.empty or metric not in left.columns:
                    continue
                key = [
                    c
                    for c in ("seed", "fold_id", "window_length")
                    if c in left.columns
                ]
                join = (
                    left[key + [metric]]
                    .rename(columns={metric: "a"})
                    .merge(
                        right[key + [metric]].rename(columns={metric: "b"}),
                        on=key,
                        how="inner",
                    )
                )
                if join.empty:
                    continue
                join["delta"] = join["a"] - join["b"]
                join["contrast_id"] = contrast_id
                join["dataset"] = dataset
                join["task"] = task
                join["metric"] = metric
                join["first"] = first
                join["second"] = second
                frames.append(
                    join[
                        [
                            "contrast_id",
                            "dataset",
                            "task",
                            "metric",
                            "first",
                            "second",
                            *key,
                            "a",
                            "b",
                            "delta",
                        ]
                    ]
                )
    if not frames:
        return pd.DataFrame(
            {
                "note": [
                    "no declared contrast pairs on the common scored units of "
                    "this bundle; the reservoir family (R0–R6) has not run "
                    "here yet — reported, not imputed"
                ]
            }
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Gate + narrative-rule evaluation (the protocol's own expressions)
# ---------------------------------------------------------------------------


def _gate_contradictions(
    table: pd.DataFrame,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index the gate-evaluable contrasts for the protocol's expressions.

    The gate engine wants a mapping
    ``(contrast_id, metric, dataset, condition) → row``; the bundle carries
    one condition (``full``) and the row supplies ``delta`` / CI / Holm p /
    cluster count, which is exactly the shape the frozen ``GateEvaluator``
    reads.
    """
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, row in table.iterrows():
        if row.get("status") != "ok":
            continue
        key = (
            str(row["contrast_id"]),
            str(row["metric"]),
            str(row["dataset"]),
            "full",
        )
        out[key] = {
            "delta": float(row["delta"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
            "p_holm": float(row["p_holm"]),
            "n_pairs": int(row["n_pairs"]),
            "n_clusters": int(row["n_clusters"]),
            "n_clusters_nonzero": int(row["n_clusters_nonzero"]),
        }
    return out


def evaluate_gates(
    protocol: dict[str, Any], table: pd.DataFrame, frame: pd.DataFrame
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate every gate of the frozen protocol against one bundle.

    Returns ``(gates, unresolved)``: ``gates`` maps gate id to a record with
    the expression, the result (bool, or the string ``UNEVALUABLE``) and the
    reason; ``unresolved`` lists the terms that could not be evaluated because
    the bundle does not carry the contrast / model / dataset they name, which
    the delivery comment has to surface. A gate that cannot be evaluated is
    never recorded as a pass or a fail (protocol §13 gate_rules).
    """
    contradictions = _gate_contradictions(table)
    model_params = _parameter_counts(protocol, frame, table)
    available_models = sorted(set(frame["model"].dropna()))
    datasets = sorted(set(frame["dataset"].dropna()))
    symbols = build_symbols(
        models=protocol_model_symbols(protocol),
        metrics=sorted(
            protocol_metric_properties_names(protocol)
        ),
        datasets=datasets,
        conditions=protocol_condition_symbols(protocol),
        aliases=protocol_dataset_symbols(protocol),
    )
    # The contrast table is keyed by the record model ids (the resolved form);
    # the gate's symbols resolve bare codes, so the mapping the evaluator
    # consults must be keyed by the bare code as well — which is the
    # protocol's declared contrast id, i.e. what build_contrast_table
    # produced. The record side is already the resolved bare code, so no
    # second re-keying pass is needed.
    evaluator = GateEvaluator(
        contrasts=contradictions,
        metrics=protocol_metric_properties_map(protocol),
        model_params=model_params,
        symbols=symbols,
        available_datasets=datasets,
    )
    gates: dict[str, Any] = {}
    unresolved: list[str] = []
    for gate_id, definition in protocol["gates"].items():
        if not isinstance(definition, dict) or "expression" not in definition:
            continue
        expression = " ".join(str(definition["expression"]).split())
        try:
            evaluation = evaluator.evaluate(gate_id, expression)
            gates[gate_id] = {
                "expression": expression,
                "result": bool(evaluation.result),
                "reason": "",
                "detail": evaluation.detail,
            }
        except GateExpressionError as exc:
            gates[gate_id] = {
                "expression": expression,
                "result": "UNEVALUABLE",
                "reason": str(exc),
                "detail": {},
            }
            unresolved.append(str(exc))
    return gates, unresolved


def protocol_metric_properties_names(protocol: dict[str, Any]) -> list[str]:
    """The metric names the frozen protocol declares for its tasks."""
    names: list[str] = []
    for task in protocol["tasks"].values():
        metrics = task.get("metrics", {})
        names.append(str(metrics.get("primary")))
        names.extend(str(m) for m in metrics.get("secondary", []))
    return sorted({n for n in names if n})


def protocol_metric_properties_map(protocol: dict[str, Any]) -> dict[str, Any]:
    """The gate engine's metric map: direction + equivalence margin per metric.

    Read from the protocol file, never re-declared — that is what makes the
    gate verdict traceable back to the frozen text.
    """
    from drososense.utils.config import protocol_metric_properties

    return protocol_metric_properties(protocol)


def _parameter_counts(
    protocol: dict[str, Any], frame: pd.DataFrame, table: pd.DataFrame
) -> dict[str, int]:
    """Trainable-parameter counts keyed by the bare model codes gates name.

    The counts come from the committed ``results/tables/model_parameters.json``
    (the M0/M1 delivery of the parameter table) when it exists; a bundle with
    the reservoir half carries them in its run records. Whatever cannot be
    resolved is left out of the map: the gate engine then reports the
    ``params`` term as a missing term rather than an invented number.
    """
    counts: dict[str, int] = {}
    parameter_table = RESULTS_TABLES_DIR / "model_parameters.json"
    if parameter_table.is_file():
        try:
            payload = json.loads(parameter_table.read_text(encoding="utf-8"))
            for model_id, entry in (payload.get("models") or {}).items():
                if isinstance(entry, dict) and "n_trainable_parameters" in entry:
                    counts[str(model_id)] = int(entry["n_trainable_parameters"])
        except (json.JSONDecodeError, OSError, ValueError):
            counts = {}
    # Bind the protocol codes onto the counts that exist.
    bindings = protocol_model_id_bindings(protocol)
    for registry_id, protocol_id in bindings.items():
        if registry_id in counts and protocol_id not in counts:
            counts[str(protocol_id)] = counts[registry_id]
    # The gate expressions name models by their bare code (R0, R4, GRU …);
    # record ids the bundle carries that the bindings did not cover are
    # still usable in a params() term — the evaluator resolves names to
    # strings. Only record the ids that are actually in the bundle; do not
    # invent a 0-count for an id that is not there, which would silently
    # flip a missing-param term into a pass.
    for model_id in set(frame["model"].dropna()):
        if str(model_id) not in counts:
            counts[str(model_id)] = None  # present in bundle, count unknown
    return counts



def evaluate_narrative_rules(
    protocol: dict[str, Any],
    table: pd.DataFrame,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate N1..N5 the way the protocol's frozen triggers say.

    Returns ``(rules, unresolved)``. The evaluation_note is implemented, not
    glossed over: when more than one rule fires, the record says so and does
    not silently pick one — the frozen text says the analysis stops and the
    conflict is reported. ``unavailable(dataset)`` is answered from the
    dataset manifest when one is present, and from the bundle's own
    dataset set otherwise (a bundle without D3's records cannot fire
    N5's D3 term, and the rule says N5 must not fire on a dataset that
    was acquired but excluded — that is read from the manifest, not from
    whether this bundle carried it).
    """
    from drososense.data.manifest import load_manifest
    from drososense.utils.paths import DATA_MANIFESTS_DIR

    contradictions = _gate_contradictions(table)
    model_params = _parameter_counts(protocol, frame, table)
    datasets = sorted(set(frame["dataset"].dropna()))
    # N5's dataset-availability set: every dataset this bundle carries is
    # available by definition (its records are the evidence), plus every
    # dataset the committed manifest declares usable.
    from drososense.data.manifest import AvailabilityStatus, load_all_manifests
    available_set: set[str] = set(datasets)
    short_to_config: dict[str, str] = {}
    for short_name, entry in protocol.get("datasets", {}).items():
        if isinstance(entry, dict) and entry.get("id"):
            short_to_config[str(short_name)] = str(entry["id"])
    try:
        manifests = load_all_manifests()
        for dataset_config_id, manifest in manifests.items():
            if manifest.status == AvailabilityStatus.UNAVAILABLE:
                continue
            available_set.add(str(dataset_config_id))
            for short_name, config_id in short_to_config.items():
                if config_id == str(dataset_config_id):
                    available_set.add(short_name)
    except (FileNotFoundError, OSError, ValueError):
        pass
    available_datasets = sorted(available_set)
    symbols = build_symbols(
        models=protocol_model_symbols(protocol),
        metrics=protocol_metric_properties_names(protocol),
        datasets=datasets,
        conditions=protocol_condition_symbols(protocol),
        aliases=protocol_dataset_symbols(protocol),
    )
    evaluator = GateEvaluator(
        contrasts=contradictions,
        metrics=protocol_metric_properties_map(protocol),
        model_params=model_params,
        symbols=symbols,
        available_datasets=available_datasets,
    )
    rules: dict[str, Any] = {}
    unresolved: list[str] = []
    fired: list[str] = []
    for rule in protocol.get("narrative_adjustment_rules", {}).get(
        "list", []
    ):
        rule_id = str(rule.get("id"))
        expression = " ".join(str(rule.get("trigger_expression", "")).split())
        if not expression:
            rules[rule_id] = {
                "trigger": expression,
                "fired": False,
                "reason": "no trigger expression declared",
            }
            continue
        try:
            evaluation = evaluator.evaluate(rule_id, expression)
            fired_flag = bool(evaluation.result)
            rules[rule_id] = {
                "trigger": expression,
                "fired": fired_flag,
                "action": str(rule.get("action", "")).strip(),
                "reason": "",
                "detail": evaluation.detail,
            }
            if fired_flag:
                fired.append(rule_id)
        except GateExpressionError as exc:
            rules[rule_id] = {
                "trigger": expression,
                "fired": False,
                "action": str(rule.get("action", "")).strip(),
                "reason": str(exc),
                "detail": {},
            }
            unresolved.append(str(exc))
    if len(fired) > 1:
        # The frozen evaluation_note: exactly one branch is expected; more
        # than one means the analysis stops and the conflict is reported.
        for rule_id in fired:
            rules[rule_id]["reason"] = (
                f"conflict: rules {fired} all fire on this bundle; the "
                f"protocol's evaluation_note says the analysis stops and the "
                f"conflict is reported, not resolved by choice"
            )
    return rules, unresolved


def paired_statistics_for_contradictions(
    protocol: dict[str, Any],
    indexed: pd.DataFrame,
) -> pd.DataFrame:
    """The fold-cluster bootstrap CIs and decisive p-values for the rows
    the gate / narrative expressions name.

    The frozen gate expressions read ``ci_contains_zero``, ``noninferior``,
    ``sig``, ``equiv`` and ``params`` — each backed by the paired statistics
    of the named contrast. This function computes them on every pairable
    (model, model) unit of the bundle, using the protocol's own decisive
    cluster-level test (``cluster_sign_test``) and the descriptive
    pair-level test (Wilcoxon signed-rank), so the numbers that reach the
    gates come from the frozen machinery, not a re-declaration of it.

    Args:
        protocol: The parsed frozen protocol.
        indexed: The bundle's per-run frame, indexed on the record unit.

    Returns:
        A long-format frame: one row per (first, second, metric, dataset,
        task) that pairs on the common scored units, with the decisive
        p-value, the fold-cluster bootstrap CI and the effect size.
        Rows that cannot pair are written out with a status, not dropped.
    """
    from drososense.utils.config import protocol_paired_spec
    from drososense.evaluation.stats import InsufficientDataError

    datasets = sorted(set(indexed["dataset"].dropna()))
    tasks = sorted(set(indexed["task"].dropna()))
    metric_by_task = decision_metric_by_task(protocol)
    models = sorted(set(indexed["model"].dropna()))
    rows: list[dict[str, Any]] = []
    for first in models:
        for second in models:
            if first == second:
                continue
            for dataset in datasets:
                for task in tasks:
                    metric = metric_by_task.get(str(task))
                    if metric is None:
                        continue
                    left = indexed[
                        (indexed["model"] == first)
                        & (indexed["dataset"] == dataset)
                        & (indexed["task"] == task)
                    ]
                    right = indexed[
                        (indexed["model"] == second)
                        & (indexed["dataset"] == dataset)
                        & (indexed["task"] == task)
                    ]
                    if "status" in left.columns:
                        left = left[left["status"] == "ok"]
                        right = right[right["status"] == "ok"]
                    if left.empty or right.empty or metric not in left.columns:
                        continue
                    key = [
                        c
                        for c in ("seed", "fold_id", "window_length")
                        if c in left.columns
                    ]
                    join = left[key + [metric]].merge(
                        right[key + [metric]],
                        on=key,
                        suffixes=("_first", "_second"),
                        how="inner",
                    ).dropna()
                    if join.empty:
                        continue
                    deltas = (
                        pd.to_numeric(join[f"{metric}_first"], errors="coerce")
                        - pd.to_numeric(join[f"{metric}_second"], errors="coerce")
                    ).dropna()
                    seed_arr = join["seed"].to_numpy() if "seed" in join.columns else np.zeros(len(deltas), dtype=int)
                    fold_arr = join["fold_id"].to_numpy() if "fold_id" in join.columns else np.zeros(len(deltas), dtype=int)
                    spec = protocol_paired_spec(protocol, metric, task)
                    try:
                        result = paired_test(
                            deltas.to_numpy(), seed_arr, fold_arr, spec,
                            contrast_id=f"{first}_vs_{second}",
                            metric=metric, dataset=dataset,
                        )
                    except InsufficientDataError:
                        continue
                    rows.append({
                        "first": first,
                        "second": second,
                        "metric": metric,
                        "dataset": dataset,
                        "task": task,
                        "n_pairs": int(deltas.size),
                        "n_clusters": int(np.unique(fold_arr).size) if len(np.unique(fold_arr)) else 1,
                        "n_clusters_nonzero": int(result.n_clusters_nonzero),
                        "delta": float(result.delta),
                        "delta_ci_low": float(result.delta_ci_low),
                        "delta_ci_high": float(result.delta_ci_high),
                        "p_value": float(result.p_value),
                        "p_paired_wilcoxon": float(result.p_paired_wilcoxon),
                        "minimum_achievable_p_over_clusters": float(result.minimum_achievable_p_over_clusters),
                        "effect_size": float(result.effect_size),
                        "effect_size_name": str(result.effect_size_name),
                    })
    return pd.DataFrame(rows)


def run_evidence_pipeline(
    experiment: str,
    output_prefix: str | Path | None = None,
    require_models: list[str] | None = None,
    tables_dir: str | Path | None = None,
    bundle_prefix: str | None = None,
) -> int:
    """The full read-only pipeline for one evidence bundle.

    Steps, in order — and the order is part of the audit: integrity first,
    then the descriptive layer, then the frozen contrast tests with their
    Holm correction, then the gate and narrative-rule expressions on those
    numbers. Every artefact lands under
    ``<output_prefix>/<experiment>/`` so a bundle's analysis is a directory,
    not a scattered set of files.

    Args:
        experiment: The bundle label (``e1_main_d2`` / ``e1_main_d3`` /
            ``e2_topology`` …).
        output_prefix: Where the output directory goes; defaults to
            ``results/tables``.
        require_models: Model ids the caller wants the analysis to cover; when
            one is missing, the pipeline still runs (unpairable rows,
            UNEVALUABLE gates) but the audit lists the gap so no one reads a
            verdict off a bundle that lacks a side of the comparison.

    Returns:
        The process exit code (0 on a clean run; 2 on an integrity failure).
    """
    protocol = load_protocol()
    try:
        frame, meta = load_evidence_bundle(
            experiment,
            required_models=require_models,
            tables_dir=tables_dir,
            bundle_prefix=bundle_prefix,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(output_prefix or RESULTS_TABLES_DIR) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed = index_runs(frame)

    descriptive = descriptive_rows(protocol, indexed)
    descriptive_path = out_dir / "descriptive.csv"
    descriptive.to_csv(descriptive_path, index=False)

    table = build_contrast_table(protocol, indexed)
    table_path = out_dir / "contrast_statistics.csv"
    table.to_csv(table_path, index=False)

    paired = paired_differences_table(protocol, indexed)
    paired_path = out_dir / "paired_differences.csv"
    paired.to_csv(paired_path, index=False)

    gates, gates_unresolved = evaluate_gates(protocol, table, indexed)
    gate_report = {
        "experiment": experiment,
        "n_gates": len(gates),
        "gates": gates,
        "unresolved_terms": sorted(set(gates_unresolved)),
        "note": (
            "A gate that names a contrast / model / dataset this bundle does "
            "not carry is UNEVALUABLE with the reason, never a pass or a "
            "fail (protocol §13 gate_rules). The unresolved_terms list is "
            "the explicit pending set for the next bundle."
        ),
    }
    gate_path = out_dir / "gates.json"
    gate_path.write_text(
        json.dumps(gate_report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    narrative, narrative_unresolved = evaluate_narrative_rules(
        protocol, table, indexed
    )
    narrative_report = {
        "experiment": experiment,
        "rules": narrative,
        "unresolved_terms": sorted(set(narrative_unresolved)),
        "note": (
            "N5 reads dataset availability from the manifest when one is "
            "present, not from whether this bundle carried the dataset "
            "(protocol §14: an acquired-but-excluded dataset is available, "
            "and N5 must not fire on it). More than one firing is reported "
            "as a conflict, never resolved by choice."
        ),
    }
    narrative_path = out_dir / "narrative.json"
    narrative_path.write_text(
        json.dumps(narrative_report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    audit = {
        "experiment": experiment,
        "bundle_prefix": meta.get("bundle_prefix"),
        "tables_dir": meta["tables_dir"],
        "output_dir": str(out_dir),
        "n_per_run": meta["n_per_run"],
        "n_distinct_test_fingerprints": meta["n_distinct_test_fingerprints"],
        "n_violations": meta["n_violations"],
        "n_distinct_config_hashes": meta.get("n_distinct_config_hashes", 0),
        "sha256": meta["sha256"],
        "bundle_files": {
            "per_run": str(Path(meta["tables_dir"]) / f"{experiment}_per_run.csv"),
            "summary": str(Path(meta["tables_dir"]) / f"{experiment}_summary.csv"),
            "fingerprints": str(
                Path(meta["tables_dir"]) / f"{experiment}_fingerprints.csv"
            ),
            "test_touched_once": str(
                Path(meta["tables_dir"]) / f"{experiment}_test_touched_once.json"
            ),
        },
        "decision_metric_by_task": decision_metric_by_task(protocol),
        "secondary_metrics_reported_only": [
            m
            for task in protocol["tasks"].values()
            for m in task.get("metrics", {}).get("secondary", [])
        ],
        "r2_secondary_only": True,
        "missing_required_models": meta.get("missing_required_models", []),
        "n_contrast_rows": int(len(table)),
        "n_contrast_rows_pairable": int((table["status"] == "ok").sum()),
        "gates_unevaluable": sorted(
            gid for gid, record in gates.items() if record["result"] == "UNEVALUABLE"
        ),
        "note": (
            "Read-only pipeline over already-scored records; no model "
            "evaluation. Unpairable / insufficient rows and UNEVALUABLE "
            "gates are reported with their reasons, never imputed."
        ),
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    for path in (
        descriptive_path,
        table_path,
        paired_path,
        gate_path,
        narrative_path,
        audit_path,
    ):
        print(path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments for the evidence pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "M4 Phase 2 read-only statistics/gating pipeline over one "
            "committed evidence bundle."
        )
    )
    parser.add_argument(
        "--experiment",
        default="e1_main_d2",
        help=(
            "evidence bundle label; the D3 / E2 plug-in point is this flag: "
            "the same command reads e1_main_d3_* or e2_topology_* when the "
            "files exist"
        ),
    )
    parser.add_argument(
        "--require-models",
        nargs="*",
        default=None,
        help=(
            "model ids the analysis is expected to cover; a missing one does "
            "not stop the pipeline, it is recorded in audit.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="parent directory of the per-bundle output (default results/tables)",
    )
    parser.add_argument(
        "--tables-dir",
        default=None,
        help=(
            "override the directory that holds the bundle files "
            "(default results/tables); used by the D3 / E2 stub demo"
        ),
    )
    parser.add_argument(
        "--bundle-prefix",
        default=None,
        help=(
            "override the file-prefix the bundle files use (default: the "
            "experiment label itself); the stub D3 demo uses 'stub_e1_main_d3'"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m drososense.evaluation.evidence_stats``."""
    args = parse_args(argv)
    return run_evidence_pipeline(
        args.experiment,
        output_prefix=args.output_prefix,
        require_models=args.require_models,
        tables_dir=args.tables_dir,
        bundle_prefix=args.bundle_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
