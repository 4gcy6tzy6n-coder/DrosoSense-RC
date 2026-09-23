"""The independent unit of the cluster-level statistics is the SPECIMEN (v1.5.3).

M4-Audit item A9 measured the defect this file guards: on the delivered evidence
every D3 `fold_id` holds a different specimen under each of the ten seeds, so the
frozen statement *"a fold is a single specimen, so the fold bootstrap IS a specimen
bootstrap"* was true of the text and false of the implementation. A cluster mean
taken over a fold index therefore averaged ten DIFFERENT specimens, and the 62
"clusters" were 62 re-partitions of the same 62 specimens.

Protocol v1.5.3 declares the cluster unit as the specimen. The regressions below
are the ones the signed decision (docs/v2_preregistration.md §5, D7) requires:

* one specimen appearing in several folds and windows contributes exactly ONE
  cluster, and `n_clusters == the number of distinct specimens tested`;
* the cluster mean averages that specimen's seeds, not one observation per fold
  index;
* renumbering fold ids and reordering rows change no reported statistic;
* adding seeds to specimens already tested does not change the cluster count;
* a fold holding several specimens, a specimen one side did not score, and a frame
  with no specimen column are all STATED refusals, never a crash and never a silent
  fall back to the fold index;
* every cluster-level line reports `n_clusters`, `n_clusters_nonzero` and
  `minimum_achievable_p_over_clusters`, with the D2 floor (2/2**5 = 0.0625 > alpha)
  stated so a D2 cluster-level claim is never read as reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.evaluation.clustering import (  # noqa: E402
    SPECIMEN_COLUMN,
    ClusterUnitError,
    check_expected_cluster_counts,
    cluster_provenance,
    expected_cluster_counts,
    load_cluster_unit_declaration,
    n_clusters,
    specimen_ids,
)
from drososense.evaluation.evidence_stats import (  # noqa: E402
    contrast_row,
    specimen_pairing_asymmetry,
)
from drososense.evaluation.stats import DEFAULT_ALPHA, cluster_means  # noqa: E402
from drososense.utils.paths import RESULTS_TABLES_DIR  # noqa: E402

DATASET = "d2_beef_uncontrolled"
TASK = "classification"
METRIC = "macro_f1"
CONTRAST = "esn_vs_gru"
MODELS = ("esn", "gru")
SPECIMENS = ("TS1", "TS2", "TS3")


# ---------------------------------------------------------------------------
# Fixtures: synthetic, specimen-labelled per-run rows
# ---------------------------------------------------------------------------
def _observations(
    specimens: tuple[str, ...] = SPECIMENS,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds_per_seed: int = 2,
    windows: tuple[int, ...] = (16, 32),
) -> list[tuple[str, int, int, int, float, float]]:
    """``(specimen, seed, fold_id, window_length, first, second)`` observations.

    Each seed gets its OWN fold indices (``fold = seed * folds_per_seed + j``) and
    each specimen keeps a distinct, positive paired difference, which is the shape
    the delivered evidence has: one specimen, ten seeds, ten different fold ids.
    """
    rows: list[tuple[str, int, int, int, float, float]] = []
    for index, specimen in enumerate(specimens):
        for seed in seeds:
            for offset in range(folds_per_seed):
                fold = seed * folds_per_seed + offset
                for window in windows:
                    first = 0.40 + 0.10 * index + 0.010 * seed + 0.001 * offset + 0.0001 * window
                    rows.append((specimen, seed, fold, window, first, first - 0.05 * (index + 1)))
    return rows


def _indexed(
    observations: list[tuple[str, int, int, int, float, float]],
    *,
    dataset: str = DATASET,
    task: str = TASK,
    metric: str = METRIC,
) -> pd.DataFrame:
    """The same observations as two models' per-run rows."""
    rows = []
    for specimen, seed, fold, window, first, second in observations:
        for model, value in zip(MODELS, (first, second)):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "fold_id": fold,
                    "window_length": window,
                    "test_specimens_joined": specimen,
                    "status": "ok",
                    metric: value,
                }
            )
    return pd.DataFrame(rows)


def _contrast(
    frame: pd.DataFrame,
    protocol: dict,
    metric: str = METRIC,
    dataset: str = DATASET,
) -> dict:
    return contrast_row(protocol, frame, CONTRAST, metric, dataset, TASK)


# ---------------------------------------------------------------------------
# 1. one specimen = one cluster, whatever it was evaluated under
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_one_specimen_is_one_cluster_however_many_folds_and_windows(protocol):
    """The D7 requirement, verbatim: several folds/windows -> exactly ONE cluster.

    The frame is built so that clustering on `fold_id` and clustering on the
    specimen give different answers -- 6 fold indices against 3 specimens -- which
    is what makes this a test of the unit rather than of the arithmetic.
    """
    frame = _indexed(_observations())
    assert frame["fold_id"].nunique() == 6, "the fixture must separate the two units"

    row = _contrast(frame, protocol)
    assert row["status"] == "ok", row.get("note")
    assert row["n_pairs"] == 36, "3 specimens x 3 seeds x 2 folds x 2 windows"
    assert row["n_clusters"] == 3, "one cluster per specimen, not per fold index"
    assert row["n_clusters"] == len(SPECIMENS)
    assert row["cluster_unit"] == "specimen"

    provenance = row["cluster_provenance"]
    assert [c.specimen_id for c in provenance] == list(SPECIMENS)
    for cluster in provenance:
        assert cluster.n_rows == 12, "every evaluation of that specimen is in it"
        assert cluster.source_seeds == (0, 1, 2)
        assert len(cluster.source_folds) == 6, "6 fold indices, still one specimen"
        assert cluster.models_present == ("esn", "gru")
        assert cluster.tasks_present == (TASK,)


@pytest.mark.unit
def test_the_cluster_mean_averages_the_specimens_observations_not_one_per_fold():
    """D7: *the cluster mean averages that specimen's seeds, not one per fold index*.

    Hand-computed: the mean of one specimen's six observations, each of which sits
    under a different fold index under a different seed.
    """
    deltas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fold_index = np.array([0, 0, 1, 1, 2, 2])  # what v1.4 clustered on
    specimen = np.array(["TS1"] * 6)

    by_specimen = cluster_means(deltas, specimen)
    by_fold_index = cluster_means(deltas, fold_index)

    assert by_specimen.tolist() == [pytest.approx(3.5)]
    assert by_fold_index.tolist() == [pytest.approx(1.5), pytest.approx(3.5), pytest.approx(5.5)]
    assert by_specimen.size == 1, "one specimen is one cluster"
    assert by_fold_index.size == 3, "the superseded fold-index unit gives three"


@pytest.mark.unit
def test_renumbering_fold_ids_and_reordering_rows_change_no_statistic(protocol):
    """A declared v1.5.3 invariant: fold ids are labels, not the unit."""
    observations = _observations()
    baseline = _contrast(_indexed(observations), protocol)

    renumbered = [
        (specimen, seed, fold * 13 + 7, window, first, second)
        for specimen, seed, fold, window, first, second in observations
    ]
    shuffled = list(reversed(renumbered))
    for frame, label in (
        (_indexed(renumbered), "renumbered fold ids"),
        (_indexed(shuffled), "renumbered and reversed row order"),
    ):
        row = _contrast(frame, protocol)
        assert row["status"] == "ok", row.get("note")
        for field in (
            "n_pairs",
            "n_clusters",
            "n_clusters_nonzero",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "p_value",
            "effect_size",
            "minimum_achievable_p_over_clusters",
        ):
            assert row[field] == pytest.approx(baseline[field]), (
                f"{label} moved {field}: {row[field]!r} != {baseline[field]!r}"
            )
        assert [c.specimen_id for c in row["cluster_provenance"]] == [
            c.specimen_id for c in baseline["cluster_provenance"]
        ]


@pytest.mark.unit
def test_adding_seeds_for_specimens_already_tested_does_not_add_clusters(protocol):
    """A declared v1.5.3 invariant, and the pseudoreplication A9 identified."""
    fewer = _indexed(_observations(seeds=(0, 1)))
    more = _indexed(_observations(seeds=(0, 1, 2, 3, 4)))

    row_fewer = _contrast(fewer, protocol)
    row_more = _contrast(more, protocol)
    assert row_fewer["status"] == row_more["status"] == "ok"
    assert row_fewer["n_clusters"] == row_more["n_clusters"] == len(SPECIMENS)
    assert row_more["n_pairs"] > row_fewer["n_pairs"], "the seeds are still measured"
    assert row_more["n_seeds"] == 5 and row_fewer["n_seeds"] == 2
    for cluster in row_more["cluster_provenance"]:
        assert cluster.source_seeds == (0, 1, 2, 3, 4)
        assert cluster.n_rows == 20


# ---------------------------------------------------------------------------
# 2. the refusals are stated outcomes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_multi_specimen_fold_is_refused_with_a_reason_not_a_crash(protocol):
    """A grouped k-fold cannot be attributed to one specimen; v1.5.3 says so.

    Measured on the committed `m1_benchmark_per_run.csv`, whose D3 folds hold 12-13
    fillet tokens each. The refusal has to be a reported status, because a crash
    would take the rest of the contrast table down with it.
    """
    observations = [
        (
            "TS1|TS2" if (seed, fold) == (0, 0) else specimen,
            seed,
            fold,
            window,
            first,
            second,
        )
        for specimen, seed, fold, window, first, second in _observations()
    ]
    frame = _indexed(observations)

    with pytest.raises(ClusterUnitError, match="more than one test specimen"):
        specimen_ids(frame)

    row = _contrast(frame, protocol)
    assert row["status"] == "unpairable"
    assert "cannot be attributed to one specimen" in row["note"]
    assert "v1.5.3" in row["note"]
    assert "n_clusters" not in row, "a refused contrast reports no cluster count"


@pytest.mark.unit
def test_a_specimen_scored_by_only_one_side_is_refused(protocol):
    """v1.5.3 missing_result_policy: no silent fill, no partial cluster mean."""
    observations = [
        observation for observation in _observations() if observation[0] != "TS3"
    ]
    indexed = _indexed(observations)
    partial = _indexed(_observations())
    # the first model keeps TS3; the second never scored it
    frame = pd.concat(
        [
            indexed[indexed["model"] == "esn"],
            partial[partial["model"] == "gru"],
        ],
        ignore_index=True,
    )

    row = _contrast(frame, protocol)
    assert row["status"] == "unpairable"
    assert "TS3" in row["note"]
    assert "only one side" in row["note"]
    assert "v1.5.3" in row["note"]


@pytest.mark.unit
def test_a_specimen_observed_differently_by_the_two_sides_is_refused(protocol):
    """The other half of the policy: an overlapping subset is not a pair.

    Under the old inner join this frame produced a silent 'ok': TS2's cluster mean
    would have been the mean of the 8 evaluations that happened to overlap, while
    the specimen still counted as one of 3 independent clusters.
    """
    dropped = (1, 32)  # seed 1, window 32 -- one evaluation of TS2, one side only
    full = _indexed(_observations())
    without = _indexed(
        [
            observation
            for observation in _observations()
            if not (observation[0] == "TS2" and (observation[1], observation[3]) == dropped)
        ]
    )
    esn_rows = full[full["model"] == "esn"]
    gru_rows = without[without["model"] == "gru"]
    frame = pd.concat([esn_rows, gru_rows], ignore_index=True)

    def as_sides(left: pd.DataFrame, right: pd.DataFrame):
        return (
            left.assign(**{SPECIMEN_COLUMN: specimen_ids(left)}),
            right.assign(**{SPECIMEN_COLUMN: specimen_ids(right)}),
        )

    side_left, side_right = as_sides(esn_rows, gru_rows)
    reason = specimen_pairing_asymmetry(side_left, side_right)
    assert "TS2" in reason and "different set of" in reason
    assert specimen_pairing_asymmetry(side_left, side_left.copy()) == "", (
        "the same evaluations on both sides are symmetric"
    )

    row = _contrast(frame, protocol)
    assert row["status"] == "unpairable"
    assert "TS2" in row["note"]
    assert "different set of" in row["note"]


@pytest.mark.unit
def test_a_frame_without_a_specimen_column_is_refused_not_clustered_on_the_fold():
    """The unit is never defaulted: no specimen column is a refusal, not a fallback."""
    frame = _indexed(_observations()).drop(columns=["test_specimens_joined"])
    with pytest.raises(ClusterUnitError, match="no specimen column"):
        specimen_ids(frame)


@pytest.mark.unit
def test_the_specimen_column_is_found_under_its_declared_names():
    frame = _indexed(_observations()).rename(
        columns={"test_specimens_joined": "specimen_id"}
    )
    assert set(specimen_ids(frame)) == set(SPECIMENS)
    assert n_clusters(frame) == len(SPECIMENS)


# ---------------------------------------------------------------------------
# 3. what every cluster-level line must report
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_every_cluster_level_line_reports_its_cluster_count_and_p_floor(protocol):
    """D7: `n_clusters`, `n_clusters_nonzero` and the minimum achievable p."""
    row = _contrast(_indexed(_observations()), protocol)
    assert row["n_clusters"] == 3
    assert row["n_clusters_nonzero"] == 3
    assert row["minimum_achievable_p_over_clusters"] == pytest.approx(2 / 2**3)
    assert row["test"] == "cluster_sign_test"
    assert row["p_value"] >= row["minimum_achievable_p_over_clusters"]


@pytest.mark.integration
def test_the_delivered_d2_evidence_reports_five_clusters_and_an_unreachable_alpha(protocol):
    """The D2 floor is a decision input: 2/2**5 = 0.0625 > alpha, so no D2
    cluster-level claim is reachable however favourable the numbers look."""
    frame = pd.read_csv(RESULTS_TABLES_DIR / "e1_main_d2_per_run.csv")
    frame = frame[(frame["status"] == "ok") & (frame["task"] == TASK)]

    row = _contrast(frame, protocol)
    assert row["status"] == "ok", row.get("note")
    assert row["n_pairs"] == 50, "5 specimens x 10 seeds"
    assert row["n_clusters"] == 5
    assert row["cluster_unit"] == "specimen"
    assert row["n_clusters_nonzero"] == 5
    assert row["minimum_achievable_p_over_clusters"] == pytest.approx(0.0625)
    assert row["minimum_achievable_p_over_clusters"] > DEFAULT_ALPHA, (
        "the D2 floor must never be reported as reachable"
    )
    assert row["p_value"] >= row["minimum_achievable_p_over_clusters"]
    for cluster in row["cluster_provenance"]:
        assert len(cluster.source_seeds) == 10, "ten seeds inside one cluster"
        assert cluster.n_rows == 10, "averaged inside the cluster, not counted"


@pytest.mark.integration
def test_the_delivered_d3_evidence_clusters_sixty_two_specimens_not_five_folds(protocol):
    """The A9 measurement, restated as the fix: 62 specimens, 62 clusters.

    On the same committed evidence the v1.x line reported `n_clusters = 5`, because
    it counted fold indices; each of those cluster means averaged ten different
    specimens. Each specimen here is still seen under ten different fold ids -- that
    is exactly why the fold index cannot be the unit.
    """
    frame = pd.read_csv(RESULTS_TABLES_DIR / "e1_main_d3_per_run.csv")
    frame = frame[(frame["status"] == "ok") & (frame["task"] == TASK)]
    assert frame["fold_id"].nunique() == 62, "the delivered D3 split is LOSO-62"

    row = _contrast(frame, protocol, dataset="d3_rainbow_trout")
    assert row["status"] == "ok", row.get("note")
    assert row["n_pairs"] == 620, "62 specimens x 10 seeds"
    assert row["n_clusters"] == 62
    assert len(row["cluster_provenance"]) == 62
    assert [c.specimen_id for c in row["cluster_provenance"]][0] == "F1F1"
    for cluster in row["cluster_provenance"]:
        assert cluster.n_rows == 10
        assert 8 <= len(cluster.source_folds) <= 10, (
            "a specimen's ten seed evaluations are scattered across 8-10 different "
            "fold ids (measured), so a fold index is not an identity"
        )
    assert check_expected_cluster_counts({DATASET: 5, "d3_rainbow_trout": 62}) == []


# ---------------------------------------------------------------------------
# 4. the declaration itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_cluster_unit_is_declared_in_the_frozen_amendment():
    declaration = load_cluster_unit_declaration()
    block = declaration["cluster_unit"]
    assert declaration["protocol_version"] == "1.5.3"
    assert declaration["frozen"] is True
    assert block["unit"] == "specimen"
    assert block["cluster_id"] == "specimen_id"
    assert block["never_inferred_from"] == "fold_id"
    assert block["resample_unit"] == "specimen"
    assert block["one_specimen_one_cluster"] is True
    assert block["windows_are_not_extra_samples"] is True
    assert expected_cluster_counts(declaration) == {
        "d2_beef_uncontrolled": 5,
        "d3_rainbow_trout": 62,
    }


@pytest.mark.unit
def test_a_cluster_count_that_disagrees_with_the_declaration_is_a_defect():
    """The counts are a property of the datasets, so a mismatch is a defect."""
    assert check_expected_cluster_counts({"d2_beef_uncontrolled": 5}) == []
    problems = check_expected_cluster_counts(
        {"d2_beef_uncontrolled": 10, "d3_rainbow_trout": 62}
    )
    assert len(problems) == 1
    assert "10 clusters against the 5 protocol v1.5.3 declares" in problems[0]
    # a dataset the declaration does not name is not silently accepted as agreeing
    assert check_expected_cluster_counts({"d9_unknown": 1}) == []


@pytest.mark.unit
def test_a_missing_or_incomplete_declaration_is_an_error_not_a_default(tmp_path):
    with pytest.raises(FileNotFoundError, match="cluster unit is never defaulted"):
        load_cluster_unit_declaration(tmp_path / "protocol_v1.5.3.yaml")

    incomplete = tmp_path / "protocol_v1.5.3.yaml"
    incomplete.write_text('protocol_version: "1.5.3"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no cluster_unit block"):
        load_cluster_unit_declaration(incomplete)


@pytest.mark.unit
def test_the_declaration_matches_the_committed_sidecar():
    from drososense.utils.config import protocol_sha256, recorded_protocol_sha256
    from drososense.utils.paths import (
        PROTOCOL_V1_5_3_PATH,
        PROTOCOL_V1_5_3_SHA256_PATH,
    )

    assert PROTOCOL_V1_5_3_PATH.is_file() and PROTOCOL_V1_5_3_SHA256_PATH.is_file()
    assert protocol_sha256(PROTOCOL_V1_5_3_PATH) == recorded_protocol_sha256(
        PROTOCOL_V1_5_3_SHA256_PATH
    )


@pytest.mark.unit
def test_the_provenance_names_the_specimen_and_everything_behind_it():
    frame = _indexed(_observations())
    provenance = cluster_provenance(frame)
    assert len(provenance) == len(SPECIMENS)
    for cluster in provenance:
        payload = cluster.as_dict()
        assert payload["specimen_id"].startswith("TS")
        # `n_rows` counts the ROWS handed to it, and this frame carries both
        # models' rows; on the paired join it is the paired observations.
        assert payload["n_rows"] == 24
        assert sorted(payload["source_seeds"]) == [0, 1, 2]
        assert len(payload["source_folds"]) == 6
        assert payload["models_present"] == ["esn", "gru"]
        assert payload["tasks_present"] == [TASK]
        assert len(cluster) == 24

    one_model = cluster_provenance(frame[frame["model"] == "esn"])
    assert [cluster.n_rows for cluster in one_model] == [12, 12, 12]
    assert one_model[0].models_present == ("esn",)
