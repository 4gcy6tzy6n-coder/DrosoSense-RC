"""E2 topology-analysis plumbing for M4 (DATA-5) — read-only, protocol-frozen.

Connects the raw reservoir run records produced by ``scripts/run_reservoir.py``
to the frozen paired-test machinery of :mod:`drososense.evaluation.stats` and
the protocol's contrast / multiplicity declarations.

Design constraints (from DATA-5 and the frozen protocol)
--------------------------------------------------------
* This module **never evaluates a model**. It only reads CSV records that were
  already produced by an approved run, and applies the pre-registered tests.
* The protocol (v1.3) is byte-frozen: the contrast list, the multiplicity
  families, the decisive test (cluster sign test over mean paired differences)
  and the descriptive test (paired Wilcoxon) are read from the protocol file,
  never re-declared here.
* The R0–R6 reservoir family has its own raw record layout (per-run CSVs, not
  ``results/raw`` JSON records), so the records are loaded, indexed and paired
  here; the statistics themselves are always the protocol's.

What it exposes
---------------
* :func:`load_reservoir_records` — read one or more ``*_per_run.csv`` files
  into a tidy frame, labelled with the source file (so substrate, dataset and
  run are auditable).
* :func:`index_runs` — index records by ``(dataset, model, task, seed,
  window_length)``, the natural unit of a reservoir run.
* :func:`paired_frame` — the paired differences for one declared contrast on
  one dataset/task, one row per ``(seed, fold)``. ``None`` when the pairing
  is incomplete (the caller must NOT impute missing units).
* :func:`paired_stat_row` — run the protocol's :func:`paired_test` on one
  contrast and return the full pre-registered row (decisive test, descriptive
  test, CI, effect size, minimum achievable p).
* :func:`holm_within_family` — apply the protocol's Holm correction to the
  raw decisive p-values of one declared family, in declaration order.
* :func:`load_evidence_bundle` — read one committed evidence bundle
  (``<experiment>_summary/per_run/fingerprints/test_touched_once``) with
  integrity checks; the D3 closure reuses it by changing the label.
* :func:`descriptive_rows` — per-(dataset, model, task) mean ± SD and the
  95% CI, with the direction rules (classification → ``macro_f1``
  maximize; regression → ``mae`` minimize; ``r2`` secondary only).
* :func:`run_evidence_pipeline` — the single-manifest chain over one bundle:
  integrity → descriptive → declared contrasts → per-pair differences →
  gate evaluation (missing terms recorded as UNEVALUABLE, never as passes).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation.stats import (  # noqa: E402
    DEFAULT_ALPHA,
    InsufficientDataError,
    holm_correction,
    paired_test,
)
from drososense.utils.config import (  # noqa: E402
    load_protocol,
    protocol_family_membership,
    protocol_model_id_bindings,
    protocol_paired_spec,
    protocol_metric_properties,
)

# The per-run column that identifies a scoring of one test fold. The R0–R6
# script today writes one record per (seed, window_length) for its own
# 75/25 split; when M4 records are produced per fold, the same key plus
# ``fold_id`` applies. Missing fold ids are filled with 0, which is correct
# only for a single-fold record set — the row says so in ``n_folds``.
_KEY_COLUMNS = ["dataset", "model", "task", "seed", "window_length"]


def load_reservoir_records(paths: list[str | Path]) -> pd.DataFrame:
    """Read ``*_per_run.csv`` files into one tidy frame.

    Args:
        paths: Paths of per-run CSVs (one per experiment batch).

    Returns:
        Frame with one row per record and a ``source_file`` column naming the
        file the row came from (audit trail for substrate/normalisation).

    Raises:
        FileNotFoundError: If a path does not exist.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"per-run record not found: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            frame = pd.read_csv(handle)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def index_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a record frame to the natural record unit.

    Fills a missing ``fold_id`` column with 0 (single-fold records) and drops
    rows without a status.

    Args:
        frame: Output of :func:`load_reservoir_records`.

    Returns:
        The frame with a ``fold_id`` column and only rows carrying a status.
    """
    frame = frame.copy()
    if "fold_id" not in frame.columns:
        frame["fold_id"] = 0
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame = frame.dropna(subset=["status"])
    return frame


def _ok_numeric(frame: pd.DataFrame, column: str) -> pd.DataFrame | None:
    """Rows whose status is ok and whose column is a finite number.

    Returns:
        The filtered frame, or ``None`` when the column is absent from the
        record frame (the metric was not scored for this run; the caller
        reports the contrast as unavailable rather than filling it in).
    """
    out = frame[frame["status"] == "ok"].copy()
    if column not in out.columns:
        return None
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=[column])


def paired_frame(
    frame: pd.DataFrame, first: str, second: str, metric: str, dataset: str, task: str
) -> pd.DataFrame | None:
    """Paired differences for one contrast on one (dataset, task).

    The pairing key is the record unit the run actually produced —
    ``(dataset, task, seed, window_length, fold_id)`` — so the two models
    must have been scored on identical splits. The two sides carry different
    ``model`` values, which is why ``model`` is deliberately NOT in the join
    key (a side was selected by it already). Missing units are never imputed:
    a contrast whose pairs do not line up exactly returns ``None`` and the
    caller must report it, not fill it.

    Args:
        frame: Output of :func:`index_runs`.
        first: First model of the contrast (the "favoured" side of the delta).
        second: Second model.
        metric: Metric column to difference.
        dataset: Dataset id to restrict to.
        task: Task name.

    Returns:
        Frame with columns ``dataset, task, seed, fold_id, window_length,
        a, b, delta``; or ``None`` if either side has no ok records.
    """
    left = _ok_numeric(frame, metric)
    right = _ok_numeric(frame, metric)
    if left is None or right is None:
        return None
    left = left[(left["dataset"] == dataset) & (left["task"] == task)]
    right = right[(right["dataset"] == dataset) & (right["task"] == task)]
    left = left[left["model"] == first]
    right = right[right["model"] == second]
    if left.empty or right.empty:
        return None

    key = [c for c in _KEY_COLUMNS if c in left.columns and c != "model"] + ["fold_id"]
    left2 = left[key + [metric]].rename(columns={metric: "a"})
    right2 = right[key + [metric]].rename(columns={metric: "b"})
    joined = left2.merge(right2, on=key, how="inner").dropna(subset=["a", "b"])
    if joined.empty:
        return None
    joined["delta"] = joined["a"] - joined["b"]
    return joined


def paired_stat_row(
    protocol: dict[str, Any],
    joined: pd.DataFrame,
    contrast_id: str,
    metric: str,
    dataset: str,
    task: str,
) -> dict[str, Any] | None:
    """Run the pre-registered paired test for one contrast and return a row.

    Args:
        protocol: Parsed protocol (supplies every test parameter).
        joined: Output of :func:`paired_frame` for this contrast.
        contrast_id: Protocol contrast id (e.g. ``R0_vs_R2``).
        metric: Metric name.
        dataset: Dataset id.
        task: Task name; selects the effect size in the protocol.

    Returns:
        The full ``PairedResult`` row plus ``n_folds``; or ``None`` when the
        data are too thin to run the test (reported, never imputed).
    """
    if joined is None or joined.empty:
        return None
    spec = protocol_paired_spec(protocol, metric, task)
    try:
        result = paired_test(
            joined["delta"].to_numpy(),
            joined["seed"].to_numpy(),
            joined["fold_id"].to_numpy(),
            spec,
            contrast_id=contrast_id,
            metric=metric,
            dataset=dataset,
            condition="full",
        )
    except InsufficientDataError:
        return {
            "contrast_id": contrast_id,
            "metric": metric,
            "dataset": dataset,
            "task": task,
            "status": "insufficient_data",
            "note": "fewer than 2 paired observations",
        }
    row = result.as_dict()
    row["task"] = task
    row["status"] = "ok"
    row["n_folds"] = int(joined["fold_id"].nunique())
    row["p_holm"] = float("nan")
    row["family"] = ""
    return row


def _family_of(protocol: dict[str, Any], contrast_id: str, condition: str = "full") -> str:
    """The multiplicity family owning a (contrast, condition) pair.

    The protocol's unit of correction is the pair, not the bare contrast —
    ``R0_vs_R4`` at ``full`` lives in ``F_secondary_topology`` while the same
    contrast under ``train10pct`` lives in ``F_lowdata``. This module builds
    ``condition='full'`` rows, so that is the default lookup.
    """
    membership = protocol_family_membership(protocol)
    return membership.get((contrast_id, condition), "")


def _reorder(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Order a family's rows by the protocol's declared test order.

    Holm is order-sensitive only in that the p-values must be corrected as a
    set within one family; we nonetheless keep the declaration order so the
    audit trail matches the protocol file line by line.
    """
    if not rows:
        return []
    families = protocol.get("multiplicity", {}).get("families", [])
    family = next(f for f in families if f["id"] == rows[0]["family"])
    order = [str(t) for t in family.get("tests", [])]
    conditions = [str(c) for c in family.get("conditions", [])]

    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        ci = order.index(row["contrast_id"]) if row["contrast_id"] in order else len(order)
        co = conditions.index(str(row.get("condition", "full"))) if str(row.get("condition", "full")) in conditions else len(conditions)
        return (ci, co)

    return sorted(rows, key=sort_key)


def holm_within_family(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Correct the decisive p-values of every declared family, in place.

    Rows of the single-test family ``F_primary`` keep their raw p (no
    correction is claimed, matching the protocol's note); every multi-test
    family is Holm-corrected in declaration order and the corrected value is
    written to ``p_holm``.

    Args:
        rows: Rows produced by :func:`paired_stat_row` (all statuses).
        protocol: Parsed protocol.

    Returns:
        The same rows, with ``family`` and ``p_holm`` filled in.
    """
    families = protocol.get("multiplicity", {}).get("families", [])
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    for family in families:
        family_id = str(family["id"])
        conditions = [str(c) for c in family.get("conditions", ())] or ["full"]
        members = [
            r for r in ok_rows
            if r.get("contrast_id") in family.get("tests", [])
            and str(r.get("condition", "full")) in conditions
        ]
        members = _reorder(members, protocol)
        if not members:
            continue
        for row in members:
            row["family"] = family["id"]
        if len(members) <= 1:
            for row in members:
                row["p_holm"] = float(row["p_value"])
            continue
        alpha = float(family.get("alpha", protocol.get("statistical_tests", {}).get("alpha", DEFAULT_ALPHA)))
        p_values = [float(r["p_value"]) for r in members]
        corrected = holm_correction(p_values, alpha)
        for row, corrected_p in zip(members, corrected):
            row["p_holm"] = float(corrected_p)
    return rows


def _resolve_code(code: str, available_models: set[str], bindings: dict[str, str]) -> str:
    """Resolve a protocol model code to the record model id that carries it.

    1. The code may already be the exact model id in the records
       (``R0`` vs ``R0_real_fly`` — a prefix match is the R-family convention).
    2. Otherwise the protocol's model-zoo binding names it (``GRU`` → ``gru``,
       ``R4`` → ``esn``).
    3. As a last resort the code itself.
    """
    if code in available_models:
        return code
    for registry_id, proto_id in bindings.items():
        if proto_id == code and registry_id in available_models:
            return registry_id
    # Prefix match for the R-family (R0 -> R0_real_fly, R2 -> R2_degree_rewired).
    for model in available_models:
        if model.startswith(code + "_"):
            return model
    return code


def build_e2_table(
    protocol: dict[str, Any], frame: pd.DataFrame
) -> pd.DataFrame:
    """Compute the pre-registered E2 table from reservoir records.

    The protocol's contrast list uses model family codes (``R0`` … ``R5``,
    ``GRU``); the record frame carries model ids (``R0_real_fly``,
    ``R2_degree_rewired``, ``esn`` …). Each code is resolved to a record id
    by :func:`_resolve_code`; a contrast whose resolution yields no common
    scored unit is reported as ``unpairable``, never dropped.

    Args:
        protocol: Parsed protocol.
        frame: Output of :func:`index_runs`.

    Returns:
        Long-format table (one row per contrast/dataset/task/condition).
    """
    protocol_family_membership(protocol)  # fail fast on overlapping families
    bindings = protocol_model_id_bindings(protocol)  # registry id -> protocol id
    codes = [(c["first"], c["second"]) for c in protocol["contrasts"]["list"]]
    available_models = set(frame["model"].dropna())

    dataset_ids = sorted(set(frame["dataset"].dropna()))
    tasks = sorted(set(frame["task"].dropna()))
    metric_by_task = {
        "classification": protocol["tasks"]["classification"]["metrics"]["primary"],
        "regression": protocol["tasks"]["regression"]["metrics"]["primary"],
    }

    rows: list[dict[str, Any]] = []
    for first_code, second_code in codes:
        first_id = _resolve_code(first_code, available_models, bindings)
        second_id = _resolve_code(second_code, available_models, bindings)
        contrast_id = f"{first_code}_vs_{second_code}"
        for dataset in dataset_ids:
            for task in tasks:
                metric = metric_by_task.get(task)
                if metric is None:
                    continue
                joined = paired_frame(frame, first_id, second_id, metric, dataset, task)
                row = paired_stat_row(protocol, joined, contrast_id, metric, dataset, task)
                if row is None:
                    rows.append(
                        {
                            "contrast_id": contrast_id,
                            "metric": metric,
                            "dataset": dataset,
                            "task": task,
                            "status": "unpairable",
                            "note": (
                                f"no common scored (seed, fold, window) between "
                                f"{first_code} (resolved to {first_id!r}) and "
                                f"{second_code} (resolved to {second_id!r})"
                            ),
                        }
                    )
                    continue
                row["family"] = _family_of(protocol, contrast_id)
                rows.append(row)

    rows = holm_within_family(rows, protocol)
    table = pd.DataFrame(rows)

    # `insufficient_data` status: when a pair has fewer than 2 observations,
    # `paired_stat_row` returns a row with status `insufficient_data` but no
    # `family` or `p_holm` fields. Fill in family and p_holm so every row in
    # the table has the same schema.
    if not table.empty:
        for idx in table.index:
            if table.at[idx, "status"] == "insufficient_data":
                contrast_id = table.at[idx, "contrast_id"]
                family = _family_of(protocol, contrast_id)
                table.at[idx, "family"] = family
                if "p_holm" not in table.columns:
                    table["p_holm"] = float("nan")
                table.at[idx, "p_holm"] = float("nan")
    return table


# ---------------------------------------------------------------------------
# D2 / D3 tidy-table layer (DATA-5, M4 Phase 1)
#
# The E1 baseline evidence lands as four tidy files per experiment label —
# ``<label>_summary.csv`` / ``_per_run.csv`` / ``_fingerprints.csv`` /
# ``_test_touched_once.json`` (the D3 watcher writes the per-run file, see
# ``ops/e1/summarize_per_run_when_idle.sh``). This layer consumes exactly that
# shape so the same pipeline reads ``e1_main_d2_*`` today and
# ``e1_main_d3_*`` after D3 closes: only the experiment label changes.
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence_bundle(experiment: str) -> pd.DataFrame:
    """Read the four committed evidence files of one experiment label.

    Args:
        experiment: Experiment label, e.g. ``e1_main_d2`` or ``e1_main_d3``.

    Returns:
        A dict-style frame: the per-run frame with the bundle's integrity
        checks run first — every one of the four files must exist, the
        per-run frame must carry a unique (dataset, model, task, seed, fold)
        key, and the ``test_touched_once.json`` report must show
        ``n_violations == 0``. Any failure raises ``ValueError`` so the
        pipeline stops rather than running on a broken bundle.
    """
    from drososense.utils.paths import RESULTS_TABLES_DIR

    per_run_path = RESULTS_TABLES_DIR / f"{experiment}_per_run.csv"
    summary_path = RESULTS_TABLES_DIR / f"{experiment}_summary.csv"
    fingerprints_path = RESULTS_TABLES_DIR / f"{experiment}_fingerprints.csv"
    touched_path = RESULTS_TABLES_DIR / f"{experiment}_test_touched_once.json"
    missing = [str(p) for p in (per_run_path, summary_path, fingerprints_path, touched_path) if not p.is_file()]
    if missing:
        raise ValueError(f"evidence bundle {experiment!r} is incomplete; missing: {missing}")

    frame = pd.read_csv(per_run_path)
    key = [c for c in ["dataset", "model", "task", "seed", "fold_id"] if c in frame.columns]
    duplicated = frame.duplicated(subset=key).sum()
    if duplicated:
        raise ValueError(f"{experiment!r} per-run frame has {duplicated} duplicate (dataset, model, task, seed, fold) row(s); bundle is inconsistent")

    touched = json.loads(touched_path.read_text(encoding="utf-8"))
    if int(touched.get("n_violations", -1)) != 0:
        raise ValueError(
            f"{experiment!r} test_touched_once report carries {touched.get('n_violations')} protocol "
            f"violation(s); the bundle must not feed the statistics pipeline"
        )
    frame.attrs["bundle"] = {
        "experiment": experiment,
        "n_per_run": int(len(frame)),
        "n_distinct_test_fingerprints": int(touched.get("n_distinct_test_fingerprints", -1)),
        "n_violations": int(touched.get("n_violations", -1)),
        "sha256": {
            "summary": _sha256_hex(summary_path)[:16],
            "per_run": _sha256_hex(per_run_path)[:16],
            "fingerprints": _sha256_hex(fingerprints_path)[:16],
            "test_touched_once": _sha256_hex(touched_path)[:16],
        },
    }
    return frame


def load_fingerprint_bundle(experiment: str) -> pd.DataFrame:
    """Read ``<experiment>_fingerprints.csv`` (the §17 audit table).

    Args:
        experiment: Experiment label, e.g. ``e1_main_d2``.

    Returns:
        The fingerprint frame, with the same existence check as the bundle.
    """
    from drososense.utils.paths import RESULTS_TABLES_DIR

    path = RESULTS_TABLES_DIR / f"{experiment}_fingerprints.csv"
    if not path.is_file():
        raise ValueError(f"fingerprint table not found: {path}")
    return pd.read_csv(path)


def descriptive_rows(
    protocol: dict[str, Any], frame: pd.DataFrame, include_secondary: bool = True
) -> list[dict[str, Any]]:
    """Per-(dataset, model, task) descriptive rows with the direction rules.

    Mean ± SD and the fold-cluster 95% CI come from the protocol's own
    bootstrap spec (percentile, B = 10000, fixed seed — the interval is
    reproducible without re-running the experiment). The DECISION metric is
    the task's PRIMARY metric: classification → ``macro_f1`` (maximize),
    regression → ``mae`` (minimize). ``r2`` may only appear as a secondary
    reported column here, never as a hypothesis/contrast metric (the r2
    direction hard constraint, DATA-23 / DATA-27).

    Args:
        protocol: Parsed protocol (supplies the bootstrap spec and margins).
        frame: Per-run frame (one row per (dataset, model, task, seed, fold)).
        include_secondary: Also report the secondary metric columns.

    Returns:
        One row per (dataset, model, task) with the descriptive statistics and
        the CI of the per-model mean.
    """
    from drososense.evaluation.stats import PairedSpec, fold_cluster_bootstrap

    primary = {
        "classification": protocol["tasks"]["classification"]["metrics"]["primary"],
        "regression": protocol["tasks"]["regression"]["metrics"]["primary"],
    }
    secondary: dict[str, list[str]] = {
        task: list(protocol["tasks"][task]["metrics"]["secondary"]) for task in primary
    }
    properties = protocol_metric_properties(protocol)

    rows: list[dict[str, Any]] = []
    for (dataset, model, task), group in frame.groupby(["dataset", "model", "task"], sort=True):
        group = group[group["status"] == "ok"]
        metric = primary.get(str(task))
        if metric is None or group.empty or metric not in group.columns:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "task": task,
                    "decision_metric": metric,
                    "status": "insufficient_data",
                    "note": "no ok records with the task's decision metric",
                }
            )
            continue
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        folds = group.loc[group[metric].notna(), "fold_id"].to_numpy()
        values = values.to_numpy()
        direction = str(properties[metric]["direction"])
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "task": task,
            "decision_metric": metric,
            "direction": direction,
            "n_ok": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        }
        # The 95% CI of the model's mean. The D2/D3 evidence is a tidy per-
        # (seed, fold) record: every record unit is independent (the bootstrap
        # seed is fixed, so the interval is reproducible without re-running
        # the experiment). For LOSO(5) each fold is one specimen, so the
        # paired-difference machinery (cluster sign test) does not apply to a
        # single model; the mean's interval is the percentile CI over the 50
        # record means, B from the protocol's own bootstrap spec.
        per_fold = group.groupby("fold_id")[metric].mean().to_numpy()
        spec_folds = np.zeros(len(per_fold))
        spec = PairedSpec(
            bootstrap_b=int(protocol["statistical_tests"]["bootstrap"]["n_resamples"]),
            bootstrap_seed=int(protocol["statistical_tests"]["bootstrap"]["seed"]),
            ci_level=float(protocol["statistical_tests"]["bootstrap"]["ci_level"]),
            resample_unit="fold",
            stratification=str(protocol["statistical_tests"]["bootstrap"].get("stratification", "seed")),
            effect_size_name=protocol["statistical_tests"]["effect_size"][str(task)],
        )
        rng = np.random.default_rng(spec.bootstrap_seed)
        record_values = pd.to_numeric(group[group["status"] == "ok"][metric], errors="coerce").dropna().to_numpy()
        draw = rng.integers(0, record_values.size, size=(spec.bootstrap_b, record_values.size))
        mean_distribution = record_values[draw].mean(axis=1)
        tail = (1.0 - spec.ci_level) / 2.0
        row["ci95_low"] = float(np.quantile(mean_distribution, tail))
        row["ci95_high"] = float(np.quantile(mean_distribution, 1.0 - tail))
        row["n_clusters"] = int(per_fold.size)
        row["status"] = "ok"
        if include_secondary:
            for secondary_metric in secondary.get(str(task), []):
                if secondary_metric in group.columns:
                    secondary_values = pd.to_numeric(group[secondary_metric], errors="coerce").dropna()
                    row[f"secondary_{secondary_metric}_mean"] = (
                        float(secondary_values.mean()) if len(secondary_values) else float("nan")
                    )
                    row[f"secondary_{secondary_metric}_n"] = int(len(secondary_values))
        rows.append(row)
    return rows


def load_d2_descriptive() -> pd.DataFrame:
    """Convenience: the D2 descriptive table (label ``e1_main_d2``)."""
    protocol = load_protocol()
    frame = load_evidence_bundle("e1_main_d2")
    return pd.DataFrame(descriptive_rows(protocol, frame))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-registered E2 statistics for the R0-R6 reservoir family."
    )
    parser.add_argument("--records", nargs="+", required=True,
                        help="per-run CSV files from scripts/run_reservoir.py")
    parser.add_argument("--dataset", help="restrict to one dataset id")
    parser.add_argument("--task", choices=["classification", "regression"],
                        help="restrict to one task")
    parser.add_argument("--output", default="results/tables/e2_statistics.csv")
    parser.add_argument("--print", dest="do_print", action="store_true")
    return parser.parse_args(argv)


def parse_descriptive_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments for the tidy-table descriptive / paired pipeline."""
    parser = argparse.ArgumentParser(
        description="Descriptive + paired statistics over a committed evidence bundle."
    )
    parser.add_argument("--experiment", default="e1_main_d2",
                        help="evidence bundle label; the D3 closure reuses this with e1_main_d3")
    parser.add_argument("--output-prefix", default=None,
                        help="defaults to results/tables/<experiment>")
    return parser.parse_args(argv)


def run_evidence_pipeline(argv: list[str] | None = None) -> int:
    """The single-manifest pipeline over one evidence bundle.

    Steps, all read-only over already-scored records:

    1. :func:`load_evidence_bundle` — integrity checks on the four files;
    2. :func:`descriptive_rows` — mean ± SD / fold-cluster 95% CI per
       (dataset, model, task), decision metric = task primary, ``r2`` only
       secondary;
    3. :func:`build_e2_table` — the protocol's declared contrasts, where the
       bundle's models resolve them (E1 baselines carry no R0, so those
       contrasts report ``unpairable`` — reported, never imputed);
    4. gate evaluation for every gate term whose contrasts are present, with
       the unevaluable terms recorded as such.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    from drososense.utils.paths import RESULTS_TABLES_DIR

    args = parse_descriptive_args(argv)
    protocol = load_protocol()
    frame = load_evidence_bundle(args.experiment)
    prefix = args.output_prefix or str(RESULTS_TABLES_DIR)
    out_dir = Path(prefix) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptive = pd.DataFrame(descriptive_rows(protocol, frame))
    descriptive_path = out_dir / "descriptive.csv"
    descriptive.to_csv(descriptive_path, index=False)

    indexed = index_runs(frame)
    table = build_e2_table(protocol, indexed)
    table_path = out_dir / "contrast_statistics.csv"
    table.to_csv(table_path, index=False)

    # Paired differences per declared contrast, for the audit trail the report
    # needs (per (seed, fold) delta), even where the test is underpowered.
    paired_frames: list[pd.DataFrame] = []
    for first_code, second_code in [(c["first"], c["second"]) for c in protocol["contrasts"]["list"]]:
        first_id = _resolve_code(first_code, set(indexed["model"].dropna()), protocol_model_id_bindings(protocol))
        second_id = _resolve_code(second_code, set(indexed["model"].dropna()), protocol_model_id_bindings(protocol))
        if first_id == second_id or first_id not in set(indexed["model"]) or second_id not in set(indexed["model"]):
            continue
        for task, metric in {
            "classification": protocol["tasks"]["classification"]["metrics"]["primary"],
            "regression": protocol["tasks"]["regression"]["metrics"]["primary"],
        }.items():
            joined = paired_frame(indexed, first_id, second_id, metric, "", "")
            if joined is not None and not joined.empty:
                joined["contrast_id"] = f"{first_code}_vs_{second_code}"
                joined["task"] = task
                paired_frames.append(joined)
    paired_path = out_dir / "paired_differences.csv"
    if paired_frames:
        pd.concat(paired_frames, ignore_index=True).to_csv(paired_path, index=False)
    else:
        pd.DataFrame(
            {"note": ["no declared contrast is pairable inside this bundle (the reservoir family has not run yet); reported, not imputed"]}
        ).to_csv(paired_path, index=False)

    # Gate evaluation on whatever contrasts exist; missing terms record the
    # gate as UNEVALUABLE with the reason, never as a pass or a fail.
    from drososense.evaluation.gates import GateEvaluator, GateExpressionError, build_symbols
    from drososense.utils.config import (
        protocol_condition_symbols,
        protocol_dataset_symbols,
        protocol_model_symbols,
    )

    ok_rows = table[table["status"] == "ok"]
    contrasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, row in ok_rows.iterrows():
        contrasts[
            (row["contrast_id"], row["metric"], row["dataset"], row["condition"])
        ] = {
            "delta": float(row["delta"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
            "p_holm": float(row["p_holm"]),
            "n_pairs": int(row["n_pairs"]),
            "n_clusters_nonzero": int(row["n_clusters_nonzero"]),
        }

    # params(...) needs the committed table plus anything the bundle measured.
    parameter_audit: dict[str, Any] = {}
    try:
        from scripts.analyze import load_committed_model_parameters, model_parameter_audit

        counts = load_committed_model_parameters()
        parameter_audit = model_parameter_audit(counts, None, protocol)
    except Exception:  # a broken optional import must not sink the bundle
        parameter_audit = {"models": {}, "unresolved_params_terms": ["R0", "R1", "R2", "R3", "R4", "R5", "GRU"]}

    datasets_in_frame = sorted(set(indexed["dataset"].dropna()))
    evaluator = GateEvaluator(
        contrasts=contrasts,
        metrics=protocol_metric_properties(protocol),
        model_params=parameter_audit.get("models", {}) if parameter_audit else {},
        symbols=build_symbols(
            protocol_model_symbols(protocol),
            sorted(protocol_metric_properties(protocol)),
            datasets_in_frame,
            protocol_condition_symbols(protocol),
            aliases=protocol_dataset_symbols(protocol),
        ),
        available_datasets=datasets_in_frame,
    )
    gate_rows: list[dict[str, Any]] = []
    for gate_id, definition in protocol["gates"].items():
        if not isinstance(definition, dict) or "expression" not in definition:
            continue
        expression = " ".join(str(definition["expression"]).split())
        try:
            evaluation = evaluator.evaluate(gate_id, expression)
            gate_rows.append(
                {
                    "gate_id": gate_id,
                    "expression": expression,
                    "result": bool(evaluation.result),
                    "reason": "",
                    "detail": evaluation.detail,
                }
            )
        except GateExpressionError as exc:
            gate_rows.append(
                {
                    "gate_id": gate_id,
                    "expression": expression,
                    "result": "UNEVALUABLE",
                    "reason": str(exc),
                    "detail": {},
                }
            )
    gate_report = {
        "experiment": args.experiment,
        "n_ok_contrasts": int(len(ok_rows)),
        "gates": gate_rows,
        "note": (
            "Terms naming a contrast this bundle does not carry (the R0 reservoir family, "
            "and datasets absent from the bundle) make the gate UNEVALUABLE, not failed. "
            "A gate that could not be evaluated is reported as such (protocol §13 gate_rules)."
        ),
    }
    gate_path = out_dir / "gates.json"
    gate_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    audit = {
        "experiment": args.experiment,
        "bundle_sha256_prefixes": frame.attrs.get("bundle", {}).get("sha256", {}),
        "n_per_run": int(len(frame)),
        "models": sorted(set(indexed["model"].dropna())),
        "tasks": sorted(set(indexed["task"].dropna())),
        "n_descriptive_rows": int(len(descriptive)),
        "n_contrast_rows": int(len(table)),
        "n_pairable_declared_contrasts": int(len(paired_frames)),
        "protocol_active_file": str(PROJECT_ROOT / "configs" / "protocol_v1.3.yaml"),
        "decision_metric_by_task": {
            task: protocol["tasks"][task]["metrics"]["primary"] for task in ("classification", "regression")
        },
        "secondary_only": ["r2"],
        "note": (
            "Read-only pipeline over already-scored records; no model evaluation. "
            "Insufficient/unpairable rows are reported, never imputed."
        ),
    }
    audit_path = out_dir / "pipeline.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8")

    for path in (descriptive_path, table_path, paired_path, gate_path, audit_path):
        print(f"{path}")
    return 0


def main_descriptive(argv: list[str] | None = None) -> int:
    return run_evidence_pipeline(argv)


def main(argv: list[str] | None = None) -> int:
    import runpy

    if argv is None and "--experiment" in sys.argv:
        return run_evidence_pipeline(None)
    args = parse_args(argv)
    protocol = load_protocol()
    frame = index_runs(load_reservoir_records(args.records))

    if args.dataset:
        frame = frame[frame["dataset"] == args.dataset]
    if args.task:
        frame = frame[frame["task"] == args.task]

    table = build_e2_table(protocol, frame)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    # Side file: the exact frozen spec every row was computed with, so the
    # audit trail is self-contained in the tables directory.
    spec_audit = {
        "protocol_file": str(PROJECT_ROOT / "configs" / "protocol_v1.3.yaml"),
        "alpha": protocol["statistical_tests"]["alpha"],
        "bootstrap": protocol["statistical_tests"]["bootstrap"],
        "primary_test": protocol["statistical_tests"]["primary_test"]["name"],
        "families": [
            {"id": f["id"], "tests": f["tests"],
             "conditions": f.get("conditions", [])}
            for f in protocol["multiplicity"]["families"]
        ],
        "source_files": sorted(set(Path(p).name for p in args.records)),
        "n_records": int(len(frame)),
        "note": (
            "Every row is the protocol's pre-registered test, computed on already-scored "
            "records. This module never evaluates a model; rows marked 'unpairable' or "
            "'insufficient_data' are reported as such, never imputed."
        ),
    }
    side = out.with_suffix(".audit.json")
    side.write_text(json.dumps(spec_audit, indent=2), encoding="utf-8")
    print(f"{out}  ({len(table)} rows)")
    print(f"{side}")

    if args.do_print:
        columns = [c for c in (
            "contrast_id", "family", "metric", "dataset", "task", "n_pairs",
            "n_clusters", "n_clusters_nonzero", "delta", "delta_ci_low",
            "delta_ci_high", "effect_size", "effect_size_name", "p_value",
            "p_holm", "p_paired_wilcoxon", "minimum_achievable_p_over_clusters",
            "status",
        ) if c in table.columns]
        print(table[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    # Two entry points, chosen by the flag so a single manifest command works:
    #   python -m drososense.evaluation.e2_stats --records a_per_run.csv        (E2 reservoir records)
    #   python -m drososense.evaluation.e2_stats --experiment e1_main_d2       (evidence bundle)
    if "--records" in sys.argv:
        raise SystemExit(main(None))
    raise SystemExit(run_evidence_pipeline(None))
