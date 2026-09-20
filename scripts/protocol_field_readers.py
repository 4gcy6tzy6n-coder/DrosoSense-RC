#!/usr/bin/env python
"""Inventory of the frozen protocol: every declared field, and who reads it.

Protocol v1.2 declared three things that no production code read — the dataset
symbol namespace, the model parameter table behind ``params(...)``, and
``multiplicity.families``. Each was found by a reviewer rather than by the code,
because nothing checked. This script is the check: it walks the protocol, and for
every declarative field asks whether the production path mentions it.

A "reader" here is a literal occurrence of the field name (or of a declared
scalar value) in ``drososense/**`` or ``scripts/**``. That is a mechanical test,
so it has both false negatives (a field read through a computed key) and false
positives (a name that coincides with something unrelated). It is a triage list,
not a proof; each row still has to be read. What it is good at is the failure
that actually happened three times: a field whose name appears nowhere at all.

    python scripts/protocol_field_readers.py                 # summary
    python scripts/protocol_field_readers.py --write         # refresh the doc
    python scripts/protocol_field_readers.py --orphans       # only the holes
"""

from __future__ import annotations

import argparse
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
PRODUCTION_DIRS = ("drososense", "scripts")
SKIP_FROM_SEARCH = {"protocol_field_readers.py"}

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
    ("seeds.extension_rule", "statement — the rule for extending the seed block"),
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
    ("hyperparameter_selection.spectral_scaling_rule", "statement — a matching rule for the M4 runner"),
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
    ("gates.Gate_C.evaluated_on", "statement — the expression itself names the datasets it uses"),
    ("gates.symbol_namespace", "mirrored — implemented as config.py:protocol_dataset_symbols + gates.py:build_symbols"),
    ("gates.significance_reachability", "mirrored — implemented as gates.py:_contrast_predicate 'sig'"),
    ("gates.gate_reachability", "statement — the reachability arithmetic, for the reader and the owner"),
    # --- declared, nothing reads it, and something should --------------------
    ("hyperparameter_selection.grid", "HOLE — the declared tuning space. No selector reads it: the M1 runner takes per-model defaults, so nothing today could notice a model tuned outside this grid. Wire it into the M4 selector."),
    ("hyperparameter_selection.selection_split", "HOLE — same selector; nothing reads it today"),
    ("hyperparameter_selection.scope", "HOLE — same selector; nothing reads it today"),
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
    ("seeds.root_seeds", "HOLE — the declared seed set. The runner takes seeds from its arguments, so a run outside 0..9 would not be refused"),
    ("seeds.primary_seed_count", "HOLE — same as seeds.root_seeds"),
    ("seeds.extension_to", "HOLE — same as seeds.root_seeds"),
    ("seeds.rng_hierarchy", "mirrored — implemented as drososense/utils/seeding.py; this states the order"),
    ("preprocessing.normalization.formula", "mirrored — implemented as drososense/data/scaling.py"),
    ("preprocessing.normalization.fit_scope", "mirrored — implemented as scaling.py + audited by leakage.py"),
    ("preprocessing.normalization.per", "mirrored — implemented as scaling.py"),
    ("preprocessing.normalization.save_artifact", "mirrored — implemented by the runner writing scaler.pkl"),
    ("preprocessing.normalization.forbidden", "mirrored — leak tests assert the scaler is train-only"),
    ("preprocessing.windowing.enabled", "mirrored — implemented as drososense/data/windowing.py"),
    ("preprocessing.windowing.length_candidates", "HOLE — declared candidate lengths. The candidate list is read from the run arguments, so a length outside this set would not be refused"),
    ("preprocessing.windowing.forbidden", "mirrored — asserted by tests/test_leakage.py"),
    ("preprocessing.missing_values.policy", "mirrored — implemented at load time in loaders.py"),
)


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


def build_index() -> dict[str, list[str]]:
    """Index every quoted string literal in the production sources.

    Returns:
        Mapping of literal text to ``file:line`` locations, capped at three per
        literal so one common word cannot flood the report.
    """
    index: dict[str, list[str]] = {}
    pattern = re.compile(r"""["']([^"'\n]{1,80})["']""")
    for path in production_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in pattern.finditer(line):
                literal = match.group(1)
                hits = index.setdefault(literal, [])
                if len(hits) < 3:
                    hits.append(f"{path.relative_to(ROOT)}:{number}")
    return index


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


def readers_for(path: str, value: Any, index: dict[str, list[str]]) -> list[str]:
    """Return the production locations that mention a field or its value.

    Args:
        path: Dotted path of the field.
        value: Its declared value.
        index: Literal index from :func:`build_index`.

    Returns:
        Sorted distinct ``file:line`` locations, empty when nothing mentions it.
    """
    key = re.sub(r"\[\d+\]$", "", path).split(".")[-1]
    candidates = [key]
    if isinstance(value, str) and len(value) <= 40 and " " not in value:
        candidates.append(value)
    if isinstance(value, list):
        candidates.extend(str(item) for item in value if isinstance(item, str) and len(item) <= 40)
    found: set[str] = set()
    for candidate in candidates:
        found.update(index.get(candidate, ()))
    return sorted(found)


def inventory() -> list[dict[str, Any]]:
    """Walk the protocol and classify every field.

    Returns:
        One record per leaf: path, value, readers and whether it is orphaned.
    """
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    index = build_index()
    records: list[dict[str, Any]] = []
    for path, value in iter_leaves(protocol):
        key = re.sub(r"\[\d+\]$", "", path).split(".")[-1]
        prose = is_prose(key, value)
        readers = [] if prose else readers_for(path, value, index)
        known = KNOWN_READERS.get(key)
        if known and "PROSE" in known:
            prose = True
            readers = []
        records.append(
            {
                "path": path,
                "key": key,
                "value": value,
                "readers": readers,
                "known_reader": known,
                "orphan": not prose and not readers and not known,
                "prose": prose,
            }
        )
    return records


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
    return "NOT TRIAGED — needs a disposition"


def is_hole(record: dict[str, Any]) -> bool:
    """Whether an orphan is a real hole rather than a statement.

    Args:
        record: An inventory record.

    Returns:
        True when the disposition is marked ``HOLE``.
    """
    return disposition(record).startswith("HOLE")


def render(records: list[dict[str, Any]]) -> str:
    """Render the inventory as Markdown.

    Args:
        records: Inventory records.

    Returns:
        The document text.
    """
    orphans = [r for r in records if r["orphan"]]
    prose = [r for r in records if r["prose"]]
    read = [r for r in records if r["readers"]]
    lines = [
        "# Protocol field readers",
        "",
        "Generated by `python scripts/protocol_field_readers.py --write`. Do not edit by hand.",
        "",
        f"Source: `{PROTOCOL_PATH.relative_to(ROOT)}`. Reader = a literal occurrence of the field",
        "name (or of a declared scalar value) in `drososense/**` or `scripts/**`. A mechanical",
        "test: false positives and false negatives both exist, so every row still has to be read.",
        "What it is good at is the failure that happened three times — a field whose name appears",
        "nowhere in the production path at all.",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
        f"| Declared leaves | {len(records)} |",
        f"| With a literal reader in the production path | {len(read)} |",
        f"| Prose (no reader expected) | {len(prose)} |",
        f"| **Orphaned (needs a disposition)** | **{len(orphans)}** |",
        "",
        "## Orphaned fields",
        "",
    ]
    if not orphans:
        lines.append("None.")
    else:
        lines += ["| Field | Disposition |", "| --- | --- |"]
        for record in orphans:
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
    args = parser.parse_args(argv)

    records = inventory()
    orphans = [r for r in records if r["orphan"]]

    if args.orphans:
        for record in orphans:
            print(f"{record['path']}: {disposition(record)}")
        print(f"\n{len(orphans)} orphaned of {len(records)} declared leaves")
        return 0

    if args.write:
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(render(records), encoding="utf-8")
        print(f"{DOC_PATH}  ({len(records)} leaves, {len(orphans)} orphaned)")
        return 0

    print(
        f"{len(records)} declared leaves; "
        f"{len([r for r in records if r['readers']])} with a literal production reader; "
        f"{len(orphans)} orphaned"
    )
    for record in orphans:
        print(f"  {record['path']}: {disposition(record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
