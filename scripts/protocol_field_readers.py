#!/usr/bin/env python
"""Inventory of the frozen protocol: every declared field, and who reads it.

Protocol v1.2 declared three things that no production code read — the dataset
symbol namespace, the model parameter table behind ``params(...)``, and
``multiplicity.families``. Each was found by a reviewer rather than by the code,
because nothing checked. This script is the check: it walks the protocol, and for
every declarative field asks whether the production path mentions it.

A "reader" is a string literal in ``drososense/**`` or ``scripts/**`` that is
equal to the field's name, parsed out of the module rather than grepped, so a
literal inside a docstring or a comment does not count. Two earlier, looser
rules were retired because roughly one row in seven of the "read" bucket was a
coincidence rather than a reader:

* matching a field's *declared value* made ``tasks.*.label_column`` read by
  ``schema.py``'s hardcoded constant, and ``gates.Gate_A.evaluated_on`` read by
  the ``D2`` in a docstring;
* matching any quoted text on a line made a docstring or a comment count.

Neither survives here. What remains is still a mechanical test — a field read
through a computed key stays a false negative, and a field whose name is a
common word in the code stays a false positive — so it is a triage list, not a
proof. Rows the tightened test can no longer certify are not silently promoted
to readers or demoted to holes: they go to *necessary human determination*, with
the match the old rule used printed beside them, because that match is what has
to be adjudicated. What the test is still good at is the failure that actually
happened three times: a field whose name appears nowhere at all.

    python scripts/protocol_field_readers.py                 # summary
    python scripts/protocol_field_readers.py --write         # refresh the doc
    python scripts/protocol_field_readers.py --orphans       # only the holes
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from drososense.utils.paths import PROTOCOL_PATH, PROJECT_ROOT as ROOT  # noqa: E402

DOC_PATH = ROOT / "docs" / "protocol_field_readers.md"
# v1.3 lives alongside the active protocol in configs/, so the inventory can
# be regenerated for any frozen version. The CLI accepts --protocol; tests
# use inventory(path=...) for the same.
PRODUCTION_DIRS = ("drososense", "scripts")
SKIP_FROM_SEARCH = {"protocol_field_readers.py"}

# A literal longer than this is a message, not a field name or a declared value.
MAX_LITERAL = 80
MAX_HITS = 3

# Fields whose *name* is generic enough that a literal hit proves nothing, or
# whose reader is known to go through a different expression. Each is listed with
# the reason so the exception is on the record rather than in someone's head.
KNOWN_READERS: dict[str, str] = {
    "protocol_version": "drososense/utils/config.py:load_protocol",
    "frozen": "drososense/utils/config.py:load_protocol",
    "frozen_at": "drososense/utils/protocol.py:freeze_status",
    "supersedes": "drososense/utils/protocol.py:freeze_status (reported, not decided)",
    "status": "PROSE marker on dataset configs; the run-record `status` is a different field",
    "gate_rules": "read by tests/test_protocol_and_manifests.py, not by production code",
    "auroc.scheme": "read via drososense/evaluation/metrics.py:EMPTY_CLASS_POLICY",
    "auroc.averaging": "read via drososense/evaluation/metrics.py",
    "multiplicity.method": "read as drososense/evaluation/stats.py:holm_correction",
    "equivalence.method": "read as drososense/evaluation/stats.py:tost_from_interval",
    "test_touched_once.enabled": "read as drososense/evaluation/runner.py:prior_touches",
    "test_touched_once.enforcement": "read as drososense/evaluation/runner.py:prior_touches",
    "effect_size.classification": "read via drososense/utils/config.py:protocol_effect_size_name, keyed by the task",
    "effect_size.regression": "read via drososense/utils/config.py:protocol_effect_size_name, keyed by the task",
    "selection_metric.classification": "read via drososense/evaluation/selection.py:SelectionSpec.metric_for, keyed by the task",
    "selection_metric.regression": "read via drososense/evaluation/selection.py:SelectionSpec.metric_for, keyed by the task",
}

# Ordered prefix rules for the orphans. Three kinds, and the kind matters:
#
#   statement  — written for a human; asking "who reads it" is the wrong question
#   mirrored   — the operative copy lives elsewhere and the protocol copy is the
#                frozen record of it (a divergence would be a defect, and is not
#                detected by anything — noted where it matters)
#   HOLE       — nothing reads it and something should; either it is wired before
#                the experiment that needs it, or it is a declaration nothing can
#                currently rely on
DISPOSITION_PREFIXES: tuple[tuple[str, str], ...] = (
    # --- written for a human reader -----------------------------------------
    ("amendment_v1_2", "statement — the amendment record, for a human reader"),
    ("amendment_v1_3", "statement — the v1.3 amendment record (decision metadata, change list, integrity disclosures, open decisions), for a human reader"),
    ("amendment_v1_4", "statement — the v1.4 §17 implementation-clarification record, for a human reader"),
    ("rework_log", "statement — history of what changed and why"),
    ("open_decisions", "statement — an open decision for the project owner"),
    ("freeze_evidence.freeze_policy", "statement — a human rule; nothing enforces it in code"),
    ("freeze_evidence.self_hash_rule", "statement — describes the sidecar mechanism, which IS read"),
    ("freeze_evidence.live_state_rule", "statement — describes the H4 behaviour; --check implements it"),
    ("freeze_evidence.previous_version", "statement — records the superseded digest; the sidecar is what is checked"),
    ("freeze_evidence.data_contact_log.note", "statement — the rule the contact log implements"),
    ("scope_boundaries", "statement — claim guards for a human author"),
    ("co_primary_policy", "statement — a policy for the write-up, applied by a human"),
    ("reporting", "statement — a reporting convention"),
    ("labels", "statement — a wording rule for reports"),
    ("split_protocol.strategies", "statement — describes what drososense/data/splits.py implements"),
    ("split_protocol.forbidden", "statement — a red line; leakage.py audits the produced windows instead"),
    ("split_protocol.session_rule", "statement — describes the session boundary splits.py enforces"),
    ("split_protocol.cluster_count_rule", "statement — the arithmetic behind gate_reachability"),
    ("split_protocol.power_note", "statement — the pre-registered power limitation"),
    ("split_protocol.reachability_note", "statement — the arithmetic behind gate_reachability"),
    ("split_protocol.reachability_examples", "statement — worked examples of the floor arithmetic"),
    ("seeds.extension_rule", "statement — the rule for extending the seed block; enforced at the aggregation boundary by results.assert_seed_blocks_are_not_pooled"),
    ("seeds.seed_role", "statement — why a seed is not a replicate; enforced by pairing.resample_unit"),
    ("seeds.deterministic_models_note", "statement — an evaluation convention"),
    ("tasks.classification.label_provenance", "statement — provenance wording repeated in every table"),
    ("tasks.classification.metrics.macro_f1_empty_class_policy", "statement — describes metrics.py's zero_division=0 choice"),
    ("tasks.classification.metrics.auroc.empty_class_policy_detail", "statement — the reasoning behind EMPTY_CLASS_POLICY"),
    ("tasks.regression.definition", "statement — the unit of the regression target"),
    ("pairing.unit", "statement — describes what paired_test builds"),
    ("pairing.n_pairs_formula", "statement — describes what paired_test reports"),
    ("pairing.n_clusters_formula", "statement — describes what paired_test reports"),
    ("pairing.matching_requirements", "statement — a checklist for the M4 runner"),
    ("primary_hypothesis", "statement — the pre-registered hypothesis text"),
    ("secondary_hypotheses", "statement — the pre-registered hypothesis text"),
    ("hyperparameter_selection.spectral_scaling_rule", "statement — the prose behind selection.SHARED_KNOBS; the knob set it implies is read by the selector and pinned by tests/test_hyperparameter_selection.py"),
    ("hyperparameter_selection.detail", "statement — describes what drososense/evaluation/selection.py implements"),
    ("robustness_protocol.detail", "statement — the reasoning behind the injection-stage choice"),
    ("robustness_protocol.drift.note", "statement — a matching rule for the M4 runner"),
    ("size_study.detail", "statement — the reasoning behind nested sampling"),
    ("size_study.seed_policy", "statement — a matching rule for the M4 runner"),
    ("excluded_specimen_criteria", "statement — the exclusion criteria, applied by a human"),
    ("stopping_rules", "statement — no sequential testing is performed; declared to forbid it"),
    ("experiments.environment_reporting", "statement — env_report records more than the required minimum"),
    ("data_verified", "statement — measured at freeze time; the manifests are the machine-readable copy"),
    ("model_zoo.interface_contract", "statement — describes the interface drososense/baselines implements"),
    ("robustness_protocol.single_channel_ablation", "statement — an exploratory method, declared not to be in a family"),
    # --- the operative copy lives elsewhere ---------------------------------
    ("split_protocol.per_dataset", "mirrored — the operative copy is the dataset config and each run record; the protocol copy is the frozen record of it"),
    ("datasets.D1.config", "mirrored — the runner resolves configs/<id>.yaml by id; this pointer is never read, so a rename here would go unnoticed"),
    ("datasets.D2.config", "mirrored — see datasets.D1.config"),
    ("datasets.D3.config", "mirrored — see datasets.D1.config"),
    ("datasets.D1.role", "statement — a label for the write-up"),
    ("datasets.D2.role", "statement — a label for the write-up"),
    ("datasets.D3.role", "statement — a label for the write-up"),
    ("datasets.D1.provider_title", "statement — provenance, repeated in the manifest"),
    ("datasets.D2.campaign_confound", "mirrored — the operative copy is configs/datasets/d2_beef_uncontrolled.yaml:acquisition_campaigns"),
    ("datasets.D3.endpoint_degrees_of_freedom", "mirrored — the operative copy is configs/datasets/d3_rainbow_trout.yaml:label_stratum"),
    ("datasets.acquisition_policy", "statement — implemented by drososense/data/manifest.py"),
    ("statistical_tests.primary_test.null_hypothesis", "statement — describes what cluster_sign_test tests"),
    ("statistical_tests.primary_test.exact", "statement — describes cluster_sign_test, which is exact by construction"),
    ("statistical_tests.primary_test.minimum_p_formula", "statement — implemented as stats.py:minimum_achievable_p"),
    ("statistical_tests.paired_descriptive_test.applied_to", "statement — describes what paired_test computes"),
    ("statistical_tests.paired_descriptive_test.decisive", "mirrored — enforced by the column name and by gate tests, not by a reader of this flag"),
    ("statistical_tests.bootstrap.sampling_order", "mirrored — implemented as stats.py:fold_cluster_bootstrap; this states it"),
    ("statistical_tests.bootstrap.seed_invariance_required", "mirrored — read by tests/test_stats.py, not by production code"),
    ("statistical_tests.effect_size.report_ci", "statement — every line always carries an interval"),
    ("statistical_tests.equivalence.decision_basis", "mirrored — implemented as stats.py:tost_from_interval"),
    ("statistical_tests.equivalence.reported_p_value", "mirrored — implemented as PairedResult.tost_proxy_p"),
    ("equivalence.enabled", "statement — the margins are read; the flag is not"),
    ("pairing.minimum_reported_clusters", "mirrored — a REPORTING minimum; each line carries n_clusters, and no code enforces this number"),
    ("multiplicity.family_rule", "mirrored — implemented as drososense/utils/config.py:protocol_family_membership"),
    ("gates.Gate_A.if_failed", "statement — guidance for the write-up"),
    ("gates.Gate_B.if_failed", "statement — guidance for the write-up"),
    ("gates.Gate_C.if_failed", "statement — guidance for the write-up"),
    # One disposition for the three, so the family is classified alike. The
    # claim is the author's and is only approximately checkable: Gate_A's and
    # Gate_B's expressions name exactly the datasets declared here, while
    # Gate_C's expression also names D2 (inside sig_any) even though the field
    # does not. Whether that earns a real reader — an expression-vs-declaration
    # check in the gate engine — is a judgement this inventory does not make.
    ("gates.Gate_A.evaluated_on", "statement — the expression itself names the datasets it uses, and the field is descriptive"),
    ("gates.Gate_B.evaluated_on", "statement — the expression itself names the datasets it uses, and the field is descriptive"),
    ("gates.Gate_C.evaluated_on", "statement — the expression itself names the datasets it uses"),
    ("gates.symbol_namespace", "mirrored — implemented as config.py:protocol_dataset_symbols + gates.py:build_symbols"),
    ("gates.significance_reachability", "mirrored — implemented as gates.py:_contrast_predicate 'sig'"),
    ("gates.gate_reachability", "statement — the reachability arithmetic, for the reader and the owner"),
    # --- declared, nothing reads it, and something should --------------------
    ("robustness_protocol.injection_stage", "HOLE — E4-E6 dependency. No injection code exists yet, so nothing can violate it, but nothing enforces it either"),
    ("robustness_protocol.retrain_readout", "HOLE — E4-E6 dependency, same as injection_stage"),
    ("robustness_protocol.drift_applied_after_standardization", "HOLE — E6 dependency, same as injection_stage"),
    ("robustness_protocol.channel_dropout", "HOLE — E4 dependency: the levels and the mechanism are declared and unimplemented"),
    ("robustness_protocol.noise", "HOLE — E5 dependency: the levels and the units are declared and unimplemented"),
    ("robustness_protocol.drift.a_t", "HOLE — E6 dependency, same as channel_dropout"),
    ("robustness_protocol.drift.b_t", "HOLE — E6 dependency, same as channel_dropout"),
    ("size_study.sizes", "HOLE — E9 dependency: declared and unimplemented"),
    ("size_study.sampling", "HOLE — E9 dependency, same as size_study.sizes"),
    ("size_study.matching_rules", "HOLE — E9 dependency, same as size_study.sizes"),
    ("compute_degradation", "HOLE — the declared order to shed work under compute pressure; nothing reads it, and nothing would stop a run shedding work in another order"),
    ("experiments", "HOLE — the experiment register. Nothing reads needs_connectome or the declared fractions, so an experiment cannot be checked against its own declaration"),
    ("results.raw_dir", "HOLE — the code resolves results paths through drososense/utils/paths.py, not through here. A change to this field would silently not move anything"),
    ("results.tables_dir", "HOLE — see results.raw_dir"),
    ("results.figures_dir", "HOLE — see results.raw_dir"),
    ("results.separation_rule", "mirrored — implemented by drososense/evaluation/results.py writing two files"),
    ("freeze_evidence.protocol_frozen_at", "mirrored — drososense/utils/protocol.py:freeze_status reports frozen_at; the equality of the two is asserted by tests/test_protocol_and_manifests.py"),
    ("freeze_evidence.protocol_sha256_sidecar", "mirrored — the active sidecar is drososense/utils/paths.py:PROTOCOL_SHA256_PATH; this field records which one, and only a test reads it"),
    ("freeze_evidence.data_contact_log.live_log", "mirrored — the live path is drososense/evaluation/contact_log.py:CONTACT_LOG_NAME; this field records it"),
    ("datasets.D1.group_semantics_verified", "mirrored — the operative copy is configs/datasets/d1_beef_controlled.yaml:specimen.group_semantics_verified"),
    ("datasets.D2.group_semantics_verified", "mirrored — the operative copy is configs/datasets/d2_beef_uncontrolled.yaml:specimen.group_semantics_verified"),
    ("datasets.D3.group_semantics_verified", "mirrored — the operative copy is configs/datasets/d3_rainbow_trout.yaml:specimen.group_semantics_verified"),
    ("statistical_tests.bootstrap.enabled", "statement — the bootstrap is always run; the flag records that it is not optional"),
    ("datasets.channel_intersection", "HOLE — the matched channel subset for a cross-food comparison. Nothing computes or enforces it, so an M4 'channel-matched' claim would be unchecked"),
    ("tasks.classification.n_classes", "mirrored — the label set is fixed in drososense/evaluation/metrics.py"),
    ("tasks.classification.label_names", "mirrored — the display names live in the dataset configs"),
    ("seeds.rng_hierarchy", "mirrored — implemented as drososense/utils/seeding.py; this states the order"),
    ("preprocessing.normalization.formula", "mirrored — implemented as drososense/data/scaling.py"),
    ("preprocessing.normalization.fit_scope", "mirrored — implemented as scaling.py + audited by leakage.py"),
    ("preprocessing.normalization.per", "mirrored — implemented as scaling.py"),
    ("preprocessing.normalization.save_artifact", "mirrored — implemented by the runner writing scaler.pkl"),
    ("preprocessing.normalization.forbidden", "mirrored — leak tests assert the scaler is train-only"),
    ("preprocessing.windowing.enabled", "mirrored — implemented as drososense/data/windowing.py"),
    ("preprocessing.windowing.forbidden", "mirrored — asserted by tests/test_leakage.py"),
    ("preprocessing.windowing.selection", "mirrored — implemented as drososense/evaluation/selection.py; the same split and metric are declared in hyperparameter_selection"),
    ("preprocessing.missing_values.policy", "mirrored — implemented at load time in loaders.py"),
)

# The bucket a row lands in when the tightened test cannot certify a reader but
# the retired looser test found something. It is not a hole: a hole is a
# determination someone has already made. It is not a reader either.
AWAITING_DETERMINATION = "awaiting determination"


def _prefix_disposition(path: str) -> str | None:
    """Return the disposition recorded for a path, by exact match then prefix.

    Args:
        path: Dotted path of the field.

    Returns:
        The disposition, or ``None`` when nothing covers the path.
    """
    # Index markers are stripped from the WHOLE path, not just the tail, or a
    # rule written for `rework_log` would miss `rework_log[3].change`.
    stripped = re.sub(r"\[\d+\]", "", path)
    if path in DISPOSITIONS:
        return DISPOSITIONS[path]
    if stripped in DISPOSITIONS:
        return DISPOSITIONS[stripped]
    for prefix, text in DISPOSITION_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + "."):
            return text
    return None


# Kept for fields whose *key* alone carries the disposition, whatever its parent.
DISPOSITIONS: dict[str, str] = {
    "gate_rules": "read by tests/test_protocol_and_manifests.py, not by production code",
}


def iter_leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield every leaf of a nested mapping with its dotted path.

    Args:
        node: The node to walk.
        prefix: Path accumulated so far.

    Yields:
        ``(path, value)`` for every scalar, empty list and empty mapping, plus
        every element of a list of scalars.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaves(value, path)
    elif isinstance(node, list):
        scalars = [item for item in node if not isinstance(item, (dict, list))]
        if scalars and len(scalars) == len(node):
            yield prefix, node
        else:
            for index, item in enumerate(node):
                yield from iter_leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, node


def production_sources() -> list[Path]:
    """Return every production Python file.

    Returns:
        Paths under ``drososense/`` and ``scripts/``, excluding this script.
    """
    paths: list[Path] = []
    for directory in PRODUCTION_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if path.name in SKIP_FROM_SEARCH or "__pycache__" in path.parts:
                continue
            paths.append(path)
    return paths


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the ``id()`` of every docstring constant in a parsed module.

    Args:
        tree: The parsed module.

    Returns:
        Ids of the ``ast.Constant`` nodes that are docstrings.
    """
    found: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", ())
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def literals_by_kind() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Index the string literals of the production path, split by kind.

    The split is the whole point: a literal that is *code* is a reader, a literal
    that is a *docstring* is prose about the code, and the retired test treated
    them alike. ``ast`` is used rather than a regular expression over lines
    because it distinguishes the two structurally — a comment, for instance, is
    not a node at all and cannot be mistaken for code.

    F-strings are included through their constant parts, so an interpolated
    ``f"...{name}..."`` still registers.

    Returns:
        ``(code, docstring)``, each mapping literal text to ``file:line`` hits.
    """
    code: dict[str, list[str]] = {}
    docstrings: dict[str, list[str]] = {}
    for path in production_sources():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        doc_nodes = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            literal = node.value
            if not literal or len(literal) > MAX_LITERAL:
                continue
            target = docstrings if id(node) in doc_nodes else code
            hits = target.setdefault(literal, [])
            if len(hits) < MAX_HITS:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return code, docstrings


PROSE_KEYS = frozenset(
    {
        "note", "notes", "detail", "description", "rule", "why", "rationale",
        "use_when", "caveats", "session_rule", "cluster_count_rule", "power_note",
        "reachability_note", "extension_rule", "derive_fn", "seed_role",
        "deterministic_models_note", "independence_caveat", "naming_caveat",
        "label_provenance", "upgrade_condition", "safe_claim_wording", "amendment_rule",
        "self_hash_rule", "live_state_rule", "obligation", "consequence", "separation",
        "current_state", "not_taken_here", "effect_on_the_project", "arithmetic",
        "owner_note", "trigger", "runs_affected", "id_binding", "group_semantics_detail",
        "label_stratum_detail", "note_text",
    }
)


def is_prose(key: str, value: Any) -> bool:
    """Whether a leaf is free text rather than a declarative field.

    "Declarative" means a value a program could act on: a boolean, a number, a
    short enum-like string, or a list of them. Everything else is a sentence
    written for a human reader, and asking "who reads it" of a sentence is the
    wrong question — which is why the first run of this inventory reported 403
    orphans and buried the three that mattered.

    Args:
        key: The leaf's key name.
        value: The leaf's value.

    Returns:
        True when the leaf is free text.
    """
    if key in PROSE_KEYS:
        return True
    if isinstance(value, str):
        return len(value) > 80 or "\n" in value
    return False


def leaf_key(path: str) -> str:
    """Return the bare name a field is looked up by.

    Args:
        path: Dotted path of the field.

    Returns:
        The final path segment, with any list index removed.
    """
    return re.sub(r"\[\d+\]$", "", path).split(".")[-1]


def reader_evidence(value: Any, index: dict[str, list[str]]) -> list[str]:
    """Return the locations where a field's declared value appears.

    Value matches are no longer readers — ``schema.py``'s hardcoded
    ``freshness_class`` is not a reader of ``tasks.*.label_column``, it is the
    same string written twice. They are kept only so a demoted row can name the
    match that has to be adjudicated.

    Args:
        value: The field's declared value.
        index: Literal index from :func:`literals_by_kind`.

    Returns:
        Sorted distinct ``file:line`` locations, empty when nothing matches.
    """
    candidates: list[str] = []
    if isinstance(value, str) and 0 < len(value) <= 40 and " " not in value:
        candidates.append(value)
    if isinstance(value, list):
        candidates.extend(
            str(item) for item in value if isinstance(item, str) and 0 < len(item) <= 40
        )
    found: set[str] = set()
    for candidate in candidates:
        found.update(index.get(candidate, ()))
    return sorted(found)


def inventory(protocol_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Walk the protocol and classify every field.

    Args:
        protocol_path: Override for the protocol to inspect; defaults to the
            active one (``PROTOCOL_PATH``).

    Returns:
        One record per leaf: path, value, readers, the evidence the retired test
        would have used, and whether it is orphaned.
    """
    target = Path(protocol_path) if protocol_path is not None else PROTOCOL_PATH
    protocol = yaml.safe_load(target.read_text(encoding="utf-8"))
    code_index, docstring_index = literals_by_kind()
    all_index: dict[str, list[str]] = {
        literal: sorted(set(hits) | set(docstring_index.get(literal, ())))
        for literal, hits in code_index.items()
    }
    for literal, hits in docstring_index.items():
        all_index.setdefault(literal, list(hits))

    records: list[dict[str, Any]] = []
    for path, value in iter_leaves(protocol):
        key = leaf_key(path)
        prose = is_prose(key, value)
        readers = [] if prose else sorted(code_index.get(key, ()))
        prose_hits = [] if prose else sorted(docstring_index.get(key, ()))
        value_hits = [] if prose else reader_evidence(value, all_index)
        known = KNOWN_READERS.get(key)
        if known and "PROSE" in known:
            prose = True
            readers = []
            prose_hits = []
            value_hits = []
        records.append(
            {
                "path": path,
                "key": key,
                "value": value,
                "readers": readers,
                "prose_hits": prose_hits,
                "value_hits": value_hits,
                "known_reader": known,
                "orphan": not prose and not readers and not known,
                "prose": prose,
            }
        )
    return records


def awaiting_determination(record: dict[str, Any]) -> bool:
    """Whether a row needs a human rather than a disposition.

    Args:
        record: An inventory record.

    Returns:
        True when the tightened test finds no reader but the retired test found
        something that still has to be adjudicated.
    """
    if not record["orphan"]:
        return False
    if _prefix_disposition(record["path"]) or record["key"] in DISPOSITIONS:
        return False
    return bool(record["prose_hits"] or record["value_hits"])


def determination_reason(record: dict[str, Any]) -> str:
    """Name the match the retired reader test used, for a human to adjudicate.

    Args:
        record: An inventory record.

    Returns:
        A one-line statement of what matched and where.
    """
    parts: list[str] = []
    if record["prose_hits"]:
        parts.append(
            "the field name appears only in a docstring or comment "
            + ", ".join(f"`{hit}`" for hit in record["prose_hits"])
        )
    if record["value_hits"]:
        parts.append(
            "only the declared value appears, as a literal "
            + ", ".join(f"`{hit}`" for hit in record["value_hits"])
        )
    detail = "; ".join(parts) if parts else "the retired test found a match that is not recorded"
    return (
        f"{AWAITING_DETERMINATION} — the tightened test finds no reader. The retired test matched "
        f"{detail}. Decide whether that is a reader; if it is, write it into KNOWN_READERS here, "
        f"and if it is not, give the field a HOLE disposition."
    )


def disposition(record: dict[str, Any]) -> str:
    """Return the recorded disposition for an orphaned field.

    Args:
        record: An inventory record.

    Returns:
        The disposition text, or a marker that flags it as un-triaged.
    """
    recorded = _prefix_disposition(record["path"])
    if recorded:
        return recorded
    if record["key"] in DISPOSITIONS:
        return DISPOSITIONS[record["key"]]
    stripped = re.sub(r"\[\d+\]", "", record["path"])
    for key, reader in KNOWN_READERS.items():
        if stripped == key or stripped.endswith("." + key):
            return f"mirrored — read as {reader}"
    if awaiting_determination(record):
        return determination_reason(record)
    return "NOT TRIAGED — needs a disposition"


def is_hole(record: dict[str, Any]) -> bool:
    """Whether an orphan is a real hole rather than a statement.

    Args:
        record: An inventory record.

    Returns:
        True when the disposition is marked ``HOLE``.
    """
    return disposition(record).startswith("HOLE")


def render(records: list[dict[str, Any]], source_path: Path | None = None) -> str:
    """Render the inventory as Markdown.

    Args:
        records: Inventory records.
        source_path: Protocol the inventory was walked over. Defaults to the
            active protocol (``PROTOCOL_PATH``).

    Returns:
        The document text.
    """
    orphans = [r for r in records if r["orphan"]]
    holes = [r for r in orphans if is_hole(r)]
    awaiting = [r for r in orphans if awaiting_determination(r)]
    prose = [r for r in records if r["prose"]]
    read = [r for r in records if r["readers"]]
    # Not prose, no literal hit, but KNOWN_READERS names a reader that goes
    # through a different expression. Counted so the table sums to the total.
    named = [
        r for r in records if not r["prose"] and not r["readers"] and r["known_reader"]
    ]
    source = source_path or PROTOCOL_PATH
    try:
        source_label = source.relative_to(ROOT)
    except ValueError:
        source_label = source
    lines = [
        "# Protocol field readers",
        "",
        "Generated by `python scripts/protocol_field_readers.py --write`. Do not edit by hand.",
        "",
        f"Source: `{source_label}`. A reader is a string literal in",
        "`drososense/**` or `scripts/**` equal to the field's NAME, parsed from the module so",
        "that a literal inside a docstring or a comment does not count, and never a match on the",
        "field's declared VALUE. Still a mechanical test: a field read through a computed key is",
        "a false negative, and a field whose name is a common word is a false positive. What it",
        "is good at is the failure that happened three times — a field whose name appears",
        "nowhere in the production path at all.",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
        f"| Declared leaves | {len(records)} |",
        f"| With a literal reader in the production path | {len(read)} |",
        f"| Read elsewhere, by a named exception | {len(named)} |",
        f"| Prose (no reader expected) | {len(prose)} |",
        f"| Orphaned (a recorded disposition) | {len(orphans)} |",
        f"| — of which statements or mirrored declarations | {len(orphans) - len(holes) - len(awaiting)} |",
        f"| — of which awaiting human determination | {len(awaiting)} |",
        f"| **— of which real holes (nothing reads them and something should)** | **{len(holes)}** |",
        "",
        "## Real holes",
        "",
        "A hole is a field nothing reads whose reader would have to exist before the thing it",
        "declares could be relied on: an M4/E-series dependency, or a declaration that has already",
        "diverged from the code. The rest of the orphans are statements, declarations whose",
        "operative copy lives elsewhere, or rows still awaiting determination — all three are",
        "listed under *Every declared field* below.",
        "",
    ]
    if not holes:
        lines.append("None.")
    else:
        lines += ["| Field | Why it is a hole |", "| --- | --- |"]
        for record in holes:
            lines.append(f"| `{record['path']}` | {disposition(record)[len('HOLE — '):]} |")
    lines += [
        "",
        "## Awaiting human determination",
        "",
        "Rows the tightened reader test can no longer certify. The retired test matched something",
        "— a docstring, a comment, or the field's declared value written again — and that match",
        "is printed beside each row because it is what has to be adjudicated. Until then these",
        "rows count as neither readers nor holes, so the hole count above is a lower bound.",
        "",
    ]
    if not awaiting:
        lines.append("None.")
    else:
        lines += ["| Field | What the retired test matched |", "| --- | --- |"]
        for record in awaiting:
            reason = determination_reason(record)
            lines.append(f"| `{record['path']}` | {reason[len(AWAITING_DETERMINATION) + 3:]} |")
    lines += ["", "## Orphaned fields (statements and mirrored declarations)", ""]
    statement_orphans = [r for r in orphans if r not in holes and r not in awaiting]
    if not statement_orphans:
        lines.append("None.")
    else:
        lines += ["| Field | Disposition |", "| --- | --- |"]
        for record in statement_orphans:
            lines.append(f"| `{record['path']}` | {disposition(record)} |")
    lines += ["", "## Every declared field", "", "| Field | Value | Production reader |", "| --- | --- | --- |"]
    for record in records:
        value = record["value"]
        rendered = str(value)
        if len(rendered) > 60:
            rendered = rendered[:57] + "…"
        rendered = rendered.replace("|", "\\|").replace("\n", " ")
        if record["prose"]:
            reader = "_prose — no reader expected_"
        elif record["readers"]:
            reader = ", ".join(f"`{hit}`" for hit in record["readers"])
        elif record["known_reader"]:
            reader = f"`{record['known_reader']}`"
        elif awaiting_determination(record):
            reader = f"_awaiting determination — see above_"
        else:
            reader = "**NO READER**"
        lines.append(f"| `{record['path']}` | {rendered} | {reader} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list for testing.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="refresh docs/protocol_field_readers.md")
    parser.add_argument("--orphans", action="store_true", help="list only the orphaned fields")
    parser.add_argument(
        "--protocol",
        default=None,
        help="protocol file to inspect; defaults to the active one",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="output markdown path; defaults to docs/protocol_field_readers.md",
    )
    args = parser.parse_args(argv)

    source = Path(args.protocol) if args.protocol is not None else PROTOCOL_PATH
    records = inventory(source)
    orphans = [r for r in records if r["orphan"]]

    if args.orphans:
        for record in orphans:
            print(f"{record['path']}: {disposition(record)}")
        print(f"\n{len(orphans)} orphaned of {len(records)} declared leaves")
        return 0

    if args.write:
        out_path = Path(args.doc) if args.doc is not None else DOC_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render(records, source_path=source), encoding="utf-8")
        print(f"{out_path}  ({len(records)} leaves, {len(orphans)} orphaned)")
        return 0

    awaiting = [r for r in orphans if awaiting_determination(r)]
    holes = [r for r in orphans if is_hole(r)]
    print(
        f"{len(records)} declared leaves; "
        f"{len([r for r in records if r['readers']])} with a literal production reader; "
        f"{len(orphans)} orphaned; {len(awaiting)} awaiting determination; {len(holes)} holes"
    )
    for record in orphans:
        print(f"  {record['path']}: {disposition(record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
