"""The frozen protocol and the dataset manifests are deliverables, so they are tested.

A protocol that is merely present is not frozen; it has to be machine-readable,
internally consistent, complete enough that a reader can tell what was decided
before the results were seen, and — since protocol v1.1 — checkable against a
recorded digest so that a post-freeze edit is detectable rather than merely
discouraged.
"""

from __future__ import annotations

import hashlib
import re

import pytest
import yaml

from drososense.data.manifest import (
    AvailabilityStatus,
    load_all_manifests,
    load_manifest,
    load_member_manifest,
    sha256_of,
    verify_manifest,
)
from drososense.data.schema import SpecimenSource, load_dataset_config, resolve_specimen_source
from drososense.evaluation.gates import GateEvaluator, build_symbols
from drososense.utils.config import (
    load_protocol,
    protocol_metric_direction,
    protocol_metric_properties,
    task_metric_direction,
    verify_protocol_freeze,
)
from drososense.utils.paths import (
    CONFIGS_DIR,
    DATA_MANIFESTS_DIR,
    PROTOCOL_PATH,
    PROTOCOL_SHA256_PATH,
)

EXPECTED_DATASETS = {
    "d1_beef_controlled",
    "d2_beef_uncontrolled",
    "d3_rainbow_trout",
    "synthetic_enose",
}

# The dataset binding the R0 audit corrected (item X1), checked against the
# providers' own titles rather than against a preference.
EXPECTED_DOI = {
    "d1_beef_controlled": "10.17632/n8mc3nspfn.1",
    "d2_beef_uncontrolled": "10.17632/mwmhh766fc.3",
    "d3_rainbow_trout": "10.5281/zenodo.20649184",
}


# ---------------------------------------------------------------------------
# Protocol — freeze and identity
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_protocol_declares_it_is_frozen_with_a_timestamp(protocol):
    """The protocol states its own freeze status, version and RFC3339 timestamp."""
    assert protocol["frozen"] is True
    assert protocol["protocol_version"].startswith("1.3")
    frozen_at = protocol["frozen_at"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", frozen_at), frozen_at


@pytest.mark.unit
def test_every_superseded_protocol_is_kept_unchanged_on_disk():
    """The amendment rule is that a new version file is added, never an edit.

    Each superseded version is still present with its own version number, so the
    numbers produced under it stay attributable to the text that produced them.
    v1.2 additionally records the superseded digest inside itself, because a
    file that is never edited cannot be the place that remembers it moved on.
    v1.3 follows the same rule and adds the LOSO(62) amendment on D3 only —
    the sidecar mechanism keeps every earlier file's identity checkable.
    """
    expected = {
        "protocol_v1.yaml": "1.0.0",
        "protocol_v1.1.yaml": "1.1.0",
        "protocol_v1.2.yaml": "1.2.0",
        "protocol_v1.3.yaml": "1.3.0",
    }
    for name, version in expected.items():
        path = CONFIGS_DIR / name
        assert path.is_file(), f"{name} must stay on disk"
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["protocol_version"] == version


@pytest.mark.unit
def test_the_active_protocol_is_v1_3_and_v1_1_and_v1_2_still_verify():
    """Switching the active protocol must not disturb the frozen ones.

    DATA-31 switched the active protocol to v1.3. v1.1 and v1.2 must still
    verify against their sidecars, byte for byte, because the amendment rule
    forbids touching a frozen file: if they have drifted, a test fails before
    any production code runs.
    """
    report = verify_protocol_freeze(PROTOCOL_PATH, PROTOCOL_SHA256_PATH)
    assert report["path"].endswith("protocol_v1.3.yaml")
    assert report["matches"], report

    previous = verify_protocol_freeze(
        CONFIGS_DIR / "protocol_v1.1.yaml", CONFIGS_DIR / "protocol_v1.1.sha256"
    )
    assert previous["matches"], "v1.1 must be untouched: its sidecar still matches"

    v1_2 = verify_protocol_freeze(
        CONFIGS_DIR / "protocol_v1.2.yaml", CONFIGS_DIR / "protocol_v1.2.sha256"
    )
    assert v1_2["matches"], "v1.2 must verify against its own sidecar"


@pytest.mark.unit
def test_v1_2_records_the_digest_of_the_version_it_supersedes():
    """The superseded file's identity is recorded in the file that moves forward."""
    v1_2 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.2.yaml").read_text(encoding="utf-8"))
    previous = v1_2["freeze_evidence"]["previous_version"]
    assert previous["protocol_version"] == "1.1.0"
    recorded = previous["sha256"]
    actual = hashlib.sha256(
        (CONFIGS_DIR / "protocol_v1.1.yaml").read_bytes()
    ).hexdigest()
    assert recorded == actual


@pytest.mark.unit
def test_v1_3_records_the_digest_of_the_version_it_supersedes():
    """The amendment rule applies to v1.3 too: it must cite v1.2's identity."""
    v1_3 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.3.yaml").read_text(encoding="utf-8"))
    previous = v1_3["freeze_evidence"]["previous_version"]
    assert previous["protocol_version"] == "1.2.0"
    assert previous["sha256"] == hashlib.sha256(
        (CONFIGS_DIR / "protocol_v1.2.yaml").read_bytes()
    ).hexdigest()
    # And v1.2 itself still cites v1.1: nothing in v1.3 was allowed to edit v1.2.
    v1_2 = yaml.safe_load((CONFIGS_DIR / "protocol_v1.2.yaml").read_text(encoding="utf-8"))
    assert v1_2["freeze_evidence"]["previous_version"]["protocol_version"] == "1.1.0"


@pytest.mark.unit
def test_v1_2_states_the_amendment_trigger_change_and_affected_runs(protocol):
    """Amending a frozen protocol requires the amendment to say what it amends."""
    amendment = protocol["amendment_v1_2"]
    assert amendment["trigger"].strip()
    assert len(amendment["changed"]) >= 5
    assert amendment["unchanged"]
    assert "NONE ARE RE-RUN" in amendment["runs_affected"]
    assert amendment["open_decisions"][0]["id"] == "OD1"


@pytest.mark.unit
def test_protocol_matches_its_recorded_digest():
    """A post-freeze edit changes the digest and fails this test.

    This is what makes the freeze rule enforceable: the digest is recorded in a
    sidecar at freeze time, and the protocol cannot contain its own hash.
    """
    report = verify_protocol_freeze()
    assert report["recorded"], f"no digest recorded in {PROTOCOL_SHA256_PATH}"
    assert report["matches"], (
        f"{report['path']} has changed since it was frozen.\n"
        f"  recorded {report['recorded']}\n  actual   {report['actual']}\n"
        f"A change to a frozen protocol requires a NEW version file, not an edit."
    )


@pytest.mark.unit
def test_protocol_declares_the_data_contact_rule(protocol):
    """Freeze evidence names the timestamp, the sidecar and the contact log."""
    evidence = protocol["freeze_evidence"]
    assert evidence["protocol_frozen_at"] == protocol["frozen_at"]
    assert evidence["protocol_sha256_sidecar"].endswith("protocol_v1.3.sha256")
    assert PROTOCOL_SHA256_PATH.name == evidence["protocol_sha256_sidecar"].split("/")[-1]
    assert "data_contact_log" in evidence
    # H4: the frozen field is a placeholder and the check must say where the live
    # value is, or it prints "not started" beside a log that has entries.
    assert "live_state_rule" in evidence
    assert "data_contact_log.json" in evidence["live_state_rule"]


@pytest.mark.unit
def test_protocol_forbids_changing_the_primary_metric_after_results(protocol):
    """The anti-tuning rule is explicit and machine-readable."""
    forbidden = " ".join(protocol["freeze_evidence"]["freeze_policy"]["forbidden"])
    assert "primary metric" in forbidden
    assert "test results" in forbidden
    assert "excluded_specimen_criterion" in forbidden


# ---------------------------------------------------------------------------
# Protocol — the R0 audit's CRITICAL and HIGH items
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_protocol_fixes_the_split_unit_to_specimen(protocol):
    """``split_unit: specimen`` is the red line and must be declared."""
    assert protocol["split_unit"] == "specimen"
    assert protocol["split_protocol"]["grouping_variable"] == "specimen_id"


@pytest.mark.unit
def test_protocol_fixes_seeds_zero_through_nine(protocol):
    """The seed set is exactly 0..9, and seeds are not the resampling unit."""
    assert protocol["seeds"]["root_seeds"] == list(range(10))
    assert protocol["seeds"]["primary_seed_count"] == 10
    assert set(protocol["seeds"]["rng_hierarchy"]) == {
        "split",
        "graph",
        "input_mapping",
        "readout",
    }


@pytest.mark.unit
def test_protocol_resamples_folds_and_never_seeds(protocol):
    """C1: the bootstrap's resampling unit must not be the seed."""
    pairing = protocol["pairing"]
    bootstrap = protocol["statistical_tests"]["bootstrap"]
    assert pairing["resample_unit"] == "fold"
    assert bootstrap["resample_unit"] == "fold"
    assert bootstrap["stratification"] == "seed"
    assert pairing["resample_unit"] != "seed"
    assert bootstrap["seed"] is not None
    assert bootstrap["ci_type"] == "percentile"


@pytest.mark.unit
def test_protocol_declares_a_numeric_equivalence_margin_per_metric(protocol):
    """C2: ``~=`` needs a margin in the metric's own units, not a figure of speech."""
    equivalence = protocol["equivalence"]
    assert equivalence["method"] == "tost"
    margins = equivalence["margins"]
    for metric in ("macro_f1", "balanced_accuracy", "mae", "rmse"):
        assert isinstance(margins[metric], (int, float)) and margins[metric] > 0


@pytest.mark.unit
def test_protocol_primary_metrics_match_the_parent_issue(protocol):
    """Primary metrics are macro-F1 (classification) and MAE (regression)."""
    assert protocol["tasks"]["classification"]["metrics"]["primary"] == "macro_f1"
    assert protocol["tasks"]["regression"]["metrics"]["primary"] == "mae"


@pytest.mark.unit
def test_protocol_declares_an_auroc_empty_class_policy(protocol):
    """X5: v1 named four classes while the implementation averaged over two or three."""
    auroc = protocol["tasks"]["classification"]["metrics"]["auroc"]
    assert auroc["scheme"] == "macro_ovr"
    assert auroc["empty_class_policy"] == "require_all_classes"
    assert auroc["empty_class_policy_detail"].strip()


@pytest.mark.unit
def test_protocol_declares_a_co_primary_policy(protocol):
    """M2: macro-F1 and MAE are co-primary and must say how alpha is handled."""
    policy = protocol["co_primary_policy"]
    assert policy["policy"] in ("both_required_no_alpha_split", "shared_family_alpha_split")
    assert policy["detail"].strip()


@pytest.mark.unit
def test_protocol_declares_hypotheses_bound_to_contrasts(protocol):
    """Every hypothesis references a contrast id that exists in the registry."""
    contrast_ids = {c["id"] for c in protocol["contrasts"]["list"]}
    primary = protocol["primary_hypothesis"]
    assert primary["id"] == "H1"
    assert primary["contrast"] in contrast_ids
    assert protocol["contrasts"]["primary"] == primary["contrast"]
    assert len(protocol["secondary_hypotheses"]) >= 3
    for hypothesis in protocol["secondary_hypotheses"]:
        assert hypothesis["id"] and hypothesis["statement"]
        assert hypothesis["contrast"] in contrast_ids


@pytest.mark.unit
def test_protocol_fixes_every_wilcoxon_parameter(protocol):
    """H2: dropping zero pairs changes the effective n; the parameters are frozen.

    In v1.2 these parameters belong to the DESCRIPTIVE pair-level test. The
    decisive test is a cluster-level sign test and has no continuity correction,
    no zero method and no asymptotic mode to fix.
    """
    tests = protocol["statistical_tests"]
    assert tests["primary_test"]["name"] == "cluster_sign_test"
    wilcoxon = tests["paired_descriptive_test"]
    assert wilcoxon["name"] == "wilcoxon_signed_rank"
    assert wilcoxon["decisive"] is False
    assert wilcoxon["alternative"] in ("two-sided", "less", "greater")
    assert wilcoxon["zero_method"] in ("wilcox", "pratt", "zsplit")
    assert isinstance(wilcoxon["correction"], bool)
    assert wilcoxon["mode"] in ("auto", "exact", "approx")
    assert tests["effect_size"]["classification"] in ("rank_biserial", "cliff_delta")


@pytest.mark.unit
def test_protocol_makes_the_decisive_test_the_cluster_level_one(protocol):
    """R0.1 item C2: the decision may not rest on n_folds * n_seeds pairs."""
    tests = protocol["statistical_tests"]
    primary = tests["primary_test"]
    assert primary["applied_to"].startswith("one mean paired difference per held-out cluster")
    assert primary["exact"] is True
    assert primary["minimum_p_formula"] == "2 / 2**n_clusters_nonzero"
    assert "six non-zero clusters" in primary["reachability_rule"]
    # The pair-level test must be marked as unfit to decide, explicitly.
    assert tests["paired_descriptive_test"]["decisive"] is False
    assert "NO gate" in tests["paired_descriptive_test"]["note"]


@pytest.mark.unit
def test_protocol_keys_the_effect_size_on_the_task_not_the_metric_name(protocol):
    """R0.1 item M3: r2 is a regression metric and takes Hodges-Lehmann."""
    effect = protocol["statistical_tests"]["effect_size"]
    assert effect["selection_key"] == "task"
    assert effect["classification"] == "rank_biserial"
    assert effect["regression"] == "hodges_lehmann"


@pytest.mark.unit
def test_protocol_publishes_the_equivalence_number_as_a_proxy(protocol):
    """R0.1 item M4: the interval-inclusion number is not a TOST p-value."""
    equivalence = protocol["statistical_tests"]["equivalence"]
    reported = equivalence["reported_p_value"]
    assert reported["column"] == "tost_proxy_p"
    assert reported["is_a_tost_p_value"] is False
    assert equivalence["decision_basis"] == "interval_inclusion"


@pytest.mark.unit
def test_protocol_enumerates_multiplicity_families(protocol):
    """H1: Holm needs an enumerated family; a test id in two families is an error."""
    multiplicity = protocol["multiplicity"]
    assert multiplicity["method"] == "holm"
    families = multiplicity["families"]
    assert families

    # The unit of correction is a (contrast, condition) pair. R0_vs_R4 at full
    # data and R0_vs_R4 under dropout are different tests and belong to different
    # families; the same PAIR in two families would make the correction
    # unreproducible.
    families_of: dict[tuple[str, str], list[str]] = {}
    for family in families:
        assert family["id"] and family["tests"]
        assert 0 < float(family["alpha"]) <= 0.05
        conditions = family.get("conditions") or ["full"]
        for test_id in family["tests"]:
            for condition in conditions:
                families_of.setdefault((test_id, condition), []).append(family["id"])

    declared = {c["id"] for c in protocol["contrasts"]["list"]}
    undeclared = {t for t, _ in families_of} - declared
    assert not undeclared, f"a family references undeclared contrasts: {undeclared}"
    duplicated = {k: v for k, v in families_of.items() if len(v) > 1}
    assert not duplicated, f"these (contrast, condition) pairs sit in two families: {duplicated}"

    corrected_conditions = {c for _, c in families_of if c != "full"}
    assert {"dropout_p0.3", "noise_s0.1", "train10pct", "train25pct"} <= corrected_conditions, (
        "the robustness and low-data families the gates reference must be enumerated"
    )


@pytest.mark.unit
def test_protocol_declares_hyperparameter_selection_and_test_isolation(protocol):
    """H3: spectral scaling is a known confound and must be chosen the same way."""
    selection = protocol["hyperparameter_selection"]
    assert selection["selection_split"] == "validation"
    assert selection["scope"] in ("per_fold", "global")
    assert selection["test_touched_once"]["enabled"] is True
    assert "spectral scaling" in selection["spectral_scaling_rule"]
    assert set(selection["grid"]["spectral_scaling"])


@pytest.mark.unit
def test_protocol_declares_the_robustness_injection_stage(protocol):
    """H6: eval-only and train+eval support different claims."""
    robustness = protocol["robustness_protocol"]
    assert robustness["injection_stage"] in ("eval_only", "train_and_eval")
    assert isinstance(robustness["retrain_readout"], bool)
    assert isinstance(robustness["drift_applied_after_standardization"], bool)
    assert set(robustness["channel_dropout"]["levels"]) == {0.1, 0.2, 0.3, 0.4, 0.5}


@pytest.mark.unit
def test_protocol_declares_stopping_failure_and_degradation_rules(protocol):
    """H5: without these, optional stopping and selective dropping are unbeatable."""
    assert protocol["stopping_rules"]["sequential_testing"] == "forbidden"
    assert protocol["stopping_rules"]["optional_stopping"] == "forbidden"
    for key in ("nan_policy", "nonconvergence_policy", "missing_class_policy", "record_schema"):
        assert protocol["failure_handling"][key].strip()
    assert protocol["excluded_specimen_criteria"]["frozen"] is True
    degradation = protocol["compute_degradation"]
    assert degradation["seed_reduction_allowed"] is False
    assert degradation["minimum_seeds"] == 10
    assert degradation["order"][-1] == "E1"


@pytest.mark.unit
def test_protocol_declares_the_size_study_sampling(protocol):
    """M5: a Performance(N) curve needs nested sampling or it carries resample noise."""
    size_study = protocol["size_study"]
    assert size_study["sampling"] in ("nested", "independent")
    assert size_study["sizes"] == sorted(size_study["sizes"])
    assert size_study["matching_rules"]


@pytest.mark.unit
def test_protocol_records_the_d1_power_limitation_before_any_result(protocol):
    """D2 has five specimens; that fact is written down in advance, not discovered later."""
    note = protocol["split_protocol"]["power_note"]
    assert "0.0625" in note
    assert "underpowered" in note


# ---------------------------------------------------------------------------
# Protocol — the declared metric directions
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_declared_metric_directions_are_read_from_the_protocol(protocol):
    """The direction a metric is read in has to come from where it is declared.

    `tasks.classification` declares it under `metrics`; `tasks.regression`
    declares it at the task level instead. Only the first was read, so MAE and
    RMSE — which are minimised — were reported as metrics to maximise, and
    GateEvaluator's `favourable` test and the selector's argmax/argmin both read
    that value.
    """
    assert protocol_metric_direction(protocol, "macro_f1") == "maximize"
    assert protocol_metric_direction(protocol, "mae") == "minimize"
    assert protocol_metric_direction(protocol, "rmse") == "minimize"

    # The protocol declares ONE direction per task, so every regression metric
    # carries `minimize` — including r2, whose natural direction is the other
    # way. That is what the protocol says, and reading it faithfully is what
    # this test pins. It is flagged rather than corrected: making r2 maximised
    # needs a per-metric declaration in the protocol, which is frozen. Until
    # then a gate or a selector that reads a direction for r2 reads `minimize`.
    assert protocol_metric_direction(protocol, "r2") == "minimize"

    properties = protocol_metric_properties(protocol)
    for metric in ("mae", "rmse", "r2"):
        assert properties[metric]["direction"] == protocol_metric_direction(protocol, metric)


@pytest.mark.unit
def test_a_task_that_states_two_different_directions_is_refused():
    """A protocol that contradicts itself stops the analysis rather than guessing."""
    with pytest.raises(ValueError, match="conflicting metric directions"):
        task_metric_direction(
            {
                "metrics": {"primary": "mae", "direction": "maximize"},
                "direction": "minimize",
            }
        )
    with pytest.raises(ValueError, match="no metric direction"):
        task_metric_direction({"metrics": {"primary": "mae"}})
    # One statement is enough, wherever it is made.
    assert task_metric_direction({"direction": "minimize"}) == "minimize"


# Pre-registered guard (DATA-23). A hypothesis that names a decision metric
# is making a claim about that metric's direction. The protocol today declares
# ONE direction per task — so a regression task's `r2` line reads as
# `minimize`, which is the OPPOSITE of r2's natural meaning. The guard makes
# that trap explicit: a hypothesis that names `r2` fails the suite rather than
# silently being read the wrong way round. The required follow-up is recorded
# in the failure message itself, so the next person to trip it does not have
# to re-derive the remedy.
_GUARD_FAILURE_NOTE = (
    "Decision metric {!r} violates the pre-registered direction rule. "
    "需要 protocol v1.3 引入 per-metric direction，并另行复核 "
    "(need protocol_v1.3 to introduce per-metric direction and re-review). "
    "Until then no hypothesis may bind a decision metric to {!r}."
)

# Direction the protocol's task layer declares for each task-declared metric.
# A regression hypothesis that names `mae` reads it as `minimize`; a
# classification hypothesis that names `macro_f1` reads it as `maximize`.
# These are the per-task directions the gate engine and the selector read;
# the test pins them so a drift in either place breaks CI before it breaks
# a result.
_TASK_LAYER_DECISION_DIRECTION: dict[str, str] = {
    "macro_f1": "maximize",
    "balanced_accuracy": "maximize",
    "auroc": "maximize",
    "accuracy": "maximize",
    "mae": "minimize",
    "rmse": "minimize",
}


def _decision_metrics(protocol: dict) -> list[str]:
    """Collect every metric the protocol uses to decide a hypothesis.

    Each hypothesis names its decision metric directly. The contrast list
    does not carry one today — each contrast inherits its metric from the
    hypothesis that references it — so the source of truth for "what metric
    decides this comparison" is the hypothesis list. A future protocol that
    introduces a contrast-level metric must update this helper too.
    """
    metrics: list[str] = []
    for hypothesis in (protocol["primary_hypothesis"], *protocol["secondary_hypotheses"]):
        if hypothesis.get("metric"):
            metrics.append(hypothesis["metric"])
    return metrics


@pytest.mark.unit
def test_no_decision_metric_is_r2(protocol):
    """Pre-registered guard: no hypothesis may bind its metric to r2.

    r2's natural direction is `maximize`, but the protocol declares one
    direction per task, so a regression task's r2 line currently reads as
    `minimize` — which is the opposite of what a gate or a selector would
    expect. The current freeze (protocol_v1.1 and protocol_v1.2) does not
    participate in any decision through r2, so the trap has not bitten yet.
    A future protocol that does must do the per-metric direction work first.
    """
    for metric in _decision_metrics(protocol):
        if metric == "r2":
            pytest.fail(_GUARD_FAILURE_NOTE.format(metric, "r2"))


@pytest.mark.unit
def test_decision_metrics_match_the_task_layer_direction(protocol):
    """The direction a hypothesis's metric is read in must agree with the task.

    A hypothesis that names `mae` as its decision metric is committed to
    `mae` being read as `minimize` — the regression task's declared
    direction. If the protocol's per-task direction ever drifts from what
    the hypothesis expects, the gate engine reads the wrong sign and the
    result is a defect, not a finding. Pin it.

    A decision metric that is NOT declared by any task — the protocol's
    current shape uses ``retention_ratio_10pct`` in H5 without declaring
    it as a task-level metric — has no task-layer direction reading to
    compare against, so the assertion does not pin a direction for it.
    Promoting such a metric to a task declaration with an explicit
    direction is part of the protocol v1.3 work this guard exists to gate.
    """
    for metric in _decision_metrics(protocol):
        expected = _TASK_LAYER_DECISION_DIRECTION.get(metric)
        if expected is None:
            # The metric has no task-layer declaration; the direction
            # reading cannot be verified. Do not guess — and do not fail
            # the suite for an already-declared hypothesis. The first
            # check above already ruled r2 out; this branch is the
            # permissive path for derived metrics the task layer does
            # not enumerate.
            continue
        actual = protocol_metric_direction(protocol, metric)
        assert actual == expected, _GUARD_FAILURE_NOTE.format(metric, metric) + (
            f" Expected direction {expected!r} (task-layer declaration), "
            f"got {actual!r}."
        )


@pytest.mark.unit
def test_no_contrast_carries_a_metric_outside_the_decision_set(protocol):
    """A contrast that names its own metric must also stay outside r2.

    The protocol's contrast list has no `metric` field today — each contrast
    inherits its metric from the hypothesis that references it. If a future
    protocol introduces a contrast-level metric, it inherits the same rule:
    it cannot be `r2`, for the same reason. A metric not declared by any
    task also has no declared direction and falls under the same guard.
    """
    contrasts = protocol["contrasts"]["list"]
    for contrast in contrasts:
        metric = contrast.get("metric")
        if metric is None:
            continue
        if metric == "r2":
            pytest.fail(_GUARD_FAILURE_NOTE.format(metric, "r2"))


# ---------------------------------------------------------------------------
# Protocol — gates and narrative rules are evaluable
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_every_gate_declares_an_evaluable_expression(protocol):
    """C2: each gate must be a boolean expression, not a phrase like 'competitive'."""
    for name in ("Gate_A", "Gate_B", "Gate_C"):
        gate = protocol["gates"][name]
        assert gate["expression"].strip(), f"{name} has no expression"
        assert gate["if_failed"].strip() if name in ("Gate_A", "Gate_B") else True

    evaluator = GateEvaluator(
        contrasts={},
        metrics=protocol_metric_properties(protocol),
        model_params={},
        symbols=build_symbols(
            models=["R0", "R1", "R2", "R3", "R4", "R5", "GRU"],
            metrics=["macro_f1", "mae", "balanced_accuracy"],
            datasets=["D1", "D2", "D3"],
            conditions=["dropout_p0.3", "noise_s0.1", "train10pct", "train25pct"],
        ),
    )
    for name in ("Gate_A", "Gate_B", "Gate_C"):
        # With no contrasts the expression must report itself unevaluable rather
        # than silently evaluating to False.
        from drososense.evaluation.gates import GateExpressionError

        with pytest.raises(GateExpressionError):
            evaluator.evaluate(name, protocol["gates"][name]["expression"])


@pytest.mark.unit
def test_narrative_rules_have_evaluable_triggers(protocol):
    """C2: a narrative branch chosen after seeing the results is post-hoc storytelling."""
    rules = protocol["narrative_adjustment_rules"]["list"]
    assert len(rules) >= 5
    for rule in rules:
        assert rule["id"] and rule["action"]
        assert rule.get("trigger_expression", "").strip()
    ids = {rule["id"] for rule in rules}
    assert {"N1", "N2", "N3", "N4", "N5"} <= ids


@pytest.mark.unit
def test_protocol_forbids_row_level_splitting(protocol):
    """The row-level split prohibition is stated in the protocol itself."""
    forbidden = " ".join(protocol["split_protocol"]["forbidden"])
    assert "train_test_split" in forbidden
    preprocessing_forbidden = " ".join(protocol["preprocessing"]["normalization"]["forbidden"])
    assert "test" in preprocessing_forbidden


@pytest.mark.unit
def test_protocol_requires_windows_to_stay_inside_specimen_and_session(protocol):
    """Both boundaries are frozen: specimen AND acquisition session."""
    windowing = protocol["preprocessing"]["windowing"]
    assert set(windowing["length_candidates"]) == {8, 16, 32, 64}
    forbidden = " ".join(windowing["forbidden"])
    assert "two different specimens" in forbidden
    assert "two different acquisition sessions" in forbidden
    assert "test set" in forbidden


@pytest.mark.unit
def test_protocol_scope_boundaries_forbid_connectome_claims_before_gate_b(protocol):
    """M1 must not claim a connectome advantage."""
    forbidden = " ".join(protocol["scope_boundaries"]["current_claims_forbidden"])
    assert "connectome" in forbidden.lower()
    assert "synthetic fixture" in forbidden.lower()


@pytest.mark.unit
def test_protocol_registers_all_deliverable_datasets(protocol):
    """D1, D2 and D3 are declared with a config path and a DOI each."""
    assert set(protocol["datasets"]) >= {"D1", "D2", "D3"}
    for key, doi in EXPECTED_DOI.items():
        entry = next(e for e in protocol["datasets"].values() if e["id"] == key)
        assert entry["doi"] == doi
        assert (CONFIGS_DIR.parent / entry["config"]).is_file()


@pytest.mark.unit
def test_protocol_binds_d1_and_d2_to_the_providers_own_titles(protocol):
    """X1: the two beef DOIs were swapped. The provider titles are the evidence."""
    by_id = {e["id"]: e for e in protocol["datasets"].values() if "id" in e}
    d1 = by_id["d1_beef_controlled"]
    d2 = by_id["d2_beef_uncontrolled"]
    assert d1["doi"] == "10.17632/n8mc3nspfn.1"
    assert d2["doi"] == "10.17632/mwmhh766fc.3"
    assert "uncontrolled environment" in d2["provider_title"]
    assert d2["split_strategy"] == "loso"
    assert d2["protocol_compliant"] is True
    assert d1["protocol_compliant"] is False


@pytest.mark.unit
def test_protocol_model_zoo_covers_the_required_baselines(protocol):
    """All nine baselines are declared, matching the parent issue's list."""
    zoo = protocol["model_zoo"]
    declared = {
        entry["id"]
        for family in ("classical", "sequence", "reservoir")
        for entry in zoo[family]
    }
    assert declared == {
        "svm_rbf",
        "random_forest",
        "xgboost",
        "pca_svm",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "esn",
    }


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("dataset_id", sorted(EXPECTED_DATASETS))
def test_every_dataset_config_loads_and_is_internally_consistent(dataset_id):
    """Each config parses and declares a usable schema."""
    schema, config = load_dataset_config(CONFIGS_DIR / "datasets" / f"{dataset_id}.yaml")
    assert schema.dataset_id == dataset_id
    assert schema.n_features == len(schema.feature_columns)
    assert schema.n_classes == 4
    assert config["tasks"]


@pytest.mark.unit
def test_only_d1_lacks_a_published_specimen_identifier():
    """X2: D2 v3 publishes five cuts, so the previous blocker no longer applies to it."""
    d1, _ = load_dataset_config(CONFIGS_DIR / "datasets" / "d1_beef_controlled.yaml")
    d2, _ = load_dataset_config(CONFIGS_DIR / "datasets" / "d2_beef_uncontrolled.yaml")
    assert d1.specimen_source is SpecimenSource.ASSUMED_TIME_BLOCK
    assert not d1.protocol_compliant
    assert "ASSUMED" in d1.specimen_note

    assert d2.specimen_source is SpecimenSource.SERIES_FILE
    assert d2.protocol_compliant


@pytest.mark.unit
def test_the_fixture_declares_published_specimens():
    """The smoke fixture has genuine specimen ids, so its split is compliant."""
    schema, _ = load_dataset_config(CONFIGS_DIR / "datasets" / "synthetic_enose.yaml")
    assert schema.specimen_source is SpecimenSource.PUBLISHED
    assert schema.protocol_compliant


@pytest.mark.unit
def test_d3_config_carries_a_measured_schema_not_an_expected_one():
    """X4: the schema used to be an unverified expectation; it is now measured."""
    _, config = load_dataset_config(CONFIGS_DIR / "datasets" / "d3_rainbow_trout.yaml")
    assert config["schema_verified"] is True
    assert "UNVERIFIED" not in yaml.safe_dump(config)
    assert config["raw"]["column_map"]["time_index"] == "@row_index"
    assert config["raw"]["feature_map"]["temperature"] == "Temperature(°C)"


@pytest.mark.unit
def test_d3_derives_its_four_level_label_from_frozen_tvc_thresholds():
    """The provider ships only Fresh/Spoiled; the four levels are ours and say so."""
    _, config = load_dataset_config(CONFIGS_DIR / "datasets" / "d3_rainbow_trout.yaml")
    derivation = config["raw"]["class_thresholds"]
    assert derivation["thresholds"] == [4.3, 5.0, 6.5]
    assert derivation["use_provider_labels"] is False
    assert "Fresh" in derivation["rationale"]
    labels = config["labels"]
    assert "NOT PROVIDER-STATED NAMES" in labels["naming_caveat"]
    assert labels["native_labels_present"] == ["Fresh", "Spoiled"]


@pytest.mark.unit
def test_d3_filename_aliases_state_their_evidence():
    """A specimen id may be corrected, but only with the reasoning attached."""
    _, config = load_dataset_config(CONFIGS_DIR / "datasets" / "d3_rainbow_trout.yaml")
    aliases = config["raw"]["specimen_aliases"]
    assert set(aliases) == {"202600515F1F1.csv", "20260513F4F1M.csv"}
    for name, alias in aliases.items():
        assert alias["specimen"].startswith("F")
        assert len(alias["reason"]) > 100, f"{name} alias has no real justification"


@pytest.mark.unit
def test_specimen_source_aliases_resolve():
    """Config spellings map onto the canonical enum."""
    assert resolve_specimen_source("column") is SpecimenSource.PUBLISHED
    assert resolve_specimen_source("series_file") is SpecimenSource.SERIES_FILE
    assert resolve_specimen_source("time_block") is SpecimenSource.ASSUMED_TIME_BLOCK
    assert resolve_specimen_source("none") is SpecimenSource.UNAVAILABLE
    assert SpecimenSource.SERIES_FILE.protocol_compliant
    with pytest.raises(ValueError, match="specimen.source must be"):
        resolve_specimen_source("guesswork")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_all_manifests_load_and_cover_the_declared_datasets():
    """Every dataset has a manifest, and every manifest is well-formed."""
    manifests = load_all_manifests()
    assert EXPECTED_DATASETS <= set(manifests)
    for manifest in manifests.values():
        assert manifest.display_name
        assert manifest.files
    assert "d3_sensor_files" not in manifests, "a member listing is not a dataset manifest"


@pytest.mark.unit
def test_unavailable_datasets_must_explain_themselves():
    """A dataset that is not auto-available must state steps or a blocker."""
    for manifest in load_all_manifests().values():
        if manifest.status is AvailabilityStatus.AUTO:
            continue
        assert manifest.blockers or manifest.manual_steps, (
            f"{manifest.dataset_id} is {manifest.status.value} without an explanation"
        )


@pytest.mark.unit
def test_d3_is_acquired_by_range_request_not_treated_as_a_blocker():
    """X3: the 21.1 GB archive was never a reason to skip a 1.04 MB extraction."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d3_rainbow_trout.yaml")
    assert manifest.status is AvailabilityStatus.AUTO
    assert manifest.archive is not None
    assert manifest.archive.kind == "remote_zip"
    assert manifest.archive.size_bytes == 21134576519
    assert not manifest.blockers
    assert manifest.archive.member_glob == "dataset/day*/sensors/*.csv"


@pytest.mark.unit
def test_d3_manifest_states_what_it_cannot_verify():
    """The archive md5 cannot be checked without a 21.1 GB download; say so."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d3_rainbow_trout.yaml")
    assert manifest.archive.sha256 is None
    assert not manifest.archive.sha256_verifiable
    note = manifest.archive.integrity_note
    assert "NOT verified" in note
    assert "CRC-32" in note


@pytest.mark.unit
def test_d3_member_manifest_pins_every_extracted_file():
    """R5: one SHA-256 per CSV, not one checksum for a 21 GB blob."""
    members = load_member_manifest(DATA_MANIFESTS_DIR / "d3_sensor_files.yaml")
    assert len(members) == 210
    assert len({m.name for m in members}) == 210
    for member in members:
        assert member.origin == "remote_archive_member"
        assert member.sha256 and len(member.sha256) == 64
        assert member.size_bytes and member.size_bytes > 0
        assert member.resolved_local_name.endswith(".csv")
    assert sum(m.size_bytes for m in members) == 1092874


@pytest.mark.unit
def test_beef_manifests_pin_a_sha256_for_every_required_file():
    """The auto-downloadable datasets are pinned by content hash."""
    for dataset_id in ("d1_beef_controlled", "d2_beef_uncontrolled"):
        manifest = load_manifest(DATA_MANIFESTS_DIR / f"{dataset_id}.yaml")
        assert manifest.status is AvailabilityStatus.AUTO
        for entry in manifest.files:
            if entry.required:
                assert entry.sha256 and len(entry.sha256) == 64
                assert entry.size_bytes


@pytest.mark.unit
def test_d2_manifest_supersedes_v1_and_keeps_its_digest():
    """The v1 file is not used, but its identity is recorded so it cannot sneak back."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d2_beef_uncontrolled.yaml")
    superseded = manifest.extra.get("superseded_versions") or []
    assert superseded, "the v3 manifest must record what it replaced"
    assert superseded[0]["doi"] == "10.17632/mwmhh766fc.1"
    assert superseded[0]["sha256"] == (
        "6b426956600de3bff1ae72fc716dabbc92fce86b1b066a233c56c3fd8a58ba03"
    )
    # v2 is named but carries no digest, because it was never fetched. Recording
    # an invented checksum there would be worse than recording none.
    assert superseded[1]["version"] == "2"
    assert superseded[1]["sha256"] is None


@pytest.mark.unit
def test_d2_manifest_documents_the_non_uniform_column_order():
    """The five files do not share one column order; the manifest records that."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d2_beef_uncontrolled.yaml")
    assert "COLUMN ORDER IS NOT UNIFORM" in manifest.schema_notes
    assert "read BY NAME" in manifest.schema_notes


@pytest.mark.unit
def test_only_d1_has_a_specimen_blocker_now():
    """The blocker must be recorded where it exists and absent where it does not."""
    d1 = load_manifest(DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml")
    assert "NO SPECIMEN IDENTIFIER IS PUBLISHED" in d1.specimen_note
    assert any("specimen" in blocker for blocker in d1.blockers)

    d2 = load_manifest(DATA_MANIFESTS_DIR / "d2_beef_uncontrolled.yaml")
    assert "SPECIMEN IDENTIFIER IS PUBLISHED" in d2.specimen_note
    assert not d2.blockers


@pytest.mark.unit
def test_d1_manifest_states_the_label_names_are_not_the_providers():
    """Mika's ruling: a derived four-level name must never be attributed to the source."""
    raw = yaml.safe_load((DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml").read_text())
    provenance = raw["label_provenance"]
    assert "NO names" in provenance
    assert "must not be cited as the dataset's own class names" in provenance


@pytest.mark.unit
def test_d2_manifest_states_the_label_names_are_the_providers():
    """D2 is the one dataset whose four-level names come from the provider."""
    raw = yaml.safe_load((DATA_MANIFESTS_DIR / "d2_beef_uncontrolled.yaml").read_text())
    assert "provider-stated words" in raw["schema_notes"]
    config = yaml.safe_load(
        (CONFIGS_DIR / "datasets" / "d2_beef_uncontrolled.yaml").read_text()
    )
    assert "ARE provider-stated names" in config["labels"]["naming_caveat"]


@pytest.mark.unit
def test_fixture_manifest_declares_the_data_is_synthetic():
    """The fixture cannot be mistaken for observed data."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "synthetic_enose.yaml")
    assert manifest.synthetic is True
    assert "SMOKE TEST" in manifest.display_name


@pytest.mark.unit
def test_sha256_of_matches_a_known_digest(tmp_path):
    """The hasher computes the digest the provider publishes for the same bytes."""
    import hashlib

    path = tmp_path / "payload.bin"
    payload = b"drososense"
    path.write_bytes(payload)
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()
    assert len(sha256_of(path)) == 64


@pytest.mark.unit
def test_verify_manifest_reports_missing_files_without_raising(tmp_path):
    """A missing payload is reported, so the caller decides whether it is fatal."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml")
    report = verify_manifest(manifest, base_dir=tmp_path)
    assert report["ok"] is False
    assert report["missing"] == ["d1_beef_controlled.csv"]
    assert report["corrupt"] == []


@pytest.mark.unit
def test_verify_manifest_detects_a_checksum_mismatch(tmp_path):
    """A file that is not the pinned revision is reported as corrupt."""
    payload = tmp_path / "d1_beef_controlled.csv"
    payload.write_text("not the real dataset\n")
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d1_beef_controlled.yaml")
    report = verify_manifest(manifest, base_dir=tmp_path)
    assert report["ok"] is False
    assert report["corrupt"] and report["corrupt"][0]["name"] == "d1_beef_controlled.csv"

    # Sanity: the report is JSON-serialisable, so it can be attached to a record.
    yaml.safe_dump(report)


@pytest.mark.unit
def test_d3_verification_reports_the_archive_gap_explicitly(tmp_path):
    """A verification report must not imply the archive digest was checked."""
    manifest = load_manifest(DATA_MANIFESTS_DIR / "d3_rainbow_trout.yaml")
    report = verify_manifest(manifest, base_dir=tmp_path)
    assert report["ok"] is False
    assert report["archive"]["sha256_verified"] is False
    assert "md5" in report["archive"]["reason"] or "21.1 GB" in report["archive"]["reason"]
