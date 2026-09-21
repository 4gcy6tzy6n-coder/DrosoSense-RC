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
* :func:`build_e2_table` — the main entry point: for every declared contrast
  of the protocol, every dataset and every task present, compute the row and
  correct within the owning multiplicity family.
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
# CLI
# ---------------------------------------------------------------------------


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


def main(argv: list[str] | None = None) -> int:
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
    raise SystemExit(main())
