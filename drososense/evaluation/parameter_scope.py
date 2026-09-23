"""Parameter evidence scope (protocol v1.5.1).

WHAT WAS WRONG. `params(model)` in a gate expression was resolved by scanning
every record under ``results/raw/**`` and taking the largest
``n_trainable_parameters`` it found. ``params(R0)`` was therefore not a defined
experimental quantity but "whichever R0 count currently sits in the tree". With
the E2 and E9 evidence on disk that is the E9 size study's N=4000 readout
(16004), against GRU's 4452 — so Gate_A's ``params(R0) < params(GRU)`` evaluated
**false** for a reason that has nothing to do with the model comparison, and the
same code reported a different answer on a tree without ``results/raw``. A gate
outcome that depends on the presence of unrelated evidence is not a gate outcome.

WHAT THIS MODULE DOES. It resolves a gate's parameter terms from **matched result
rows** inside a declared scope, and fails closed:

    scope = (experiment labels, condition, task, datasets)

* every matched row for a model must carry the SAME
  ``n_trainable_parameters``; differing values are a conflict, not a choice;
* a model with no rows in the scope is unresolved;
* a model with a reservoir must have one unique ``reservoir_size`` in the scope;
* a condition that the amendment does not bind to experiment labels is unbound;
* any of those failures makes the term UNEVALUABLE. There is no fallback to the
  committed table, no cross-experiment search, and no max/min/first/last.

Provenance is returned with every resolved count, so a published parameter number
can be traced to the rows that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from drososense.utils.paths import PROTOCOL_V1_5_1_PATH

#: Outcomes a parameter term can have. `resolved` is the only one that yields a
#: number; every other one leaves the term unevaluable.
RESOLVED = "resolved"
NO_RECORDS = "no_records"
CONFLICTING_VALUES = "conflicting_values"
NO_RESERVOIR_SIZE = "no_reservoir_size"
UNBOUND_CONDITION = "unbound_condition"
MISSING_SCOPE = "missing_scope"


class ParameterScopeError(ValueError):
    """Raised when a gate's parameter term cannot be resolved from its scope."""


def load_parameter_scope_declaration(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the v1.5.1 ``parameter_evidence_scope`` block.

    Args:
        path: Override for ``configs/protocol_v1.5.1.yaml``.

    Returns:
        The declaration block.

    Raises:
        FileNotFoundError: If the amendment is absent — a scope that cannot be
            read is not a scope that can be defaulted.
        ValueError: If the block is missing.
    """
    import yaml

    target = Path(path) if path is not None else PROTOCOL_V1_5_1_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"parameter-scope declaration not found at {target}; protocol v1.5.1 "
            f"must be present — the scope is never defaulted"
        )
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    block = payload.get("parameter_evidence_scope")
    if not block:
        raise ValueError(f"{target} carries no parameter_evidence_scope block")
    return block


@dataclass(frozen=True)
class ModelParameterEvidence:
    """One model's parameter count, with the provenance that justifies it.

    Attributes:
        model: The protocol id the gate names.
        status: One of the module-level outcome constants.
        parameter_count: The count, or ``None`` when unresolved.
        experiment_labels: The scope's experiment labels for this model.
        condition: The declared condition.
        task: The task the count is read under.
        datasets: The datasets the gate is evaluated on.
        reservoir_size: The unique reservoir size in the scope, when applicable.
        n_source_records: How many ok rows the count was read from.
        source_run_ids: Their run ids.
        config_hashes: The distinct config hashes among them.
        conflicting_values: The distinct counts found, when they disagree.
        detail: A human-readable reason, always populated.
    """

    model: str
    status: str
    parameter_count: int | None
    experiment_labels: tuple[str, ...] = ()
    condition: str = ""
    task: str = ""
    datasets: tuple[str, ...] = ()
    reservoir_size: int | None = None
    n_source_records: int = 0
    source_run_ids: tuple[str, ...] = ()
    config_hashes: tuple[str, ...] = ()
    conflicting_values: tuple[int, ...] = ()
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "parameter_count": self.parameter_count,
            "experiment": list(self.experiment_labels),
            "condition": self.condition,
            "task": self.task,
            "datasets": list(self.datasets),
            "reservoir_size": self.reservoir_size,
            "n_source_records": self.n_source_records,
            "source_run_ids": list(self.source_run_ids),
            "config_hashes": list(self.config_hashes),
            "conflicting_values": list(self.conflicting_values),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateParameterScope:
    """The resolved parameter scope of one gate, DATASET BY DATASET.

    Protocol v1.5.2 conditions a parameter predicate on the dataset: a count is
    resolved inside each registered dataset's own matched configuration, and the
    enclosing comparison must hold for every dataset in ``evaluated_on``. The
    scope therefore carries one evidence set per dataset, and there is no
    dataset-agnostic count to read.

    Attributes:
        gate_id: The gate.
        task: The task the parameter terms are read under.
        datasets: The datasets the gate is evaluated on, in protocol order.
        per_dataset: ``{dataset: {model: ModelParameterEvidence}}``.
    """

    gate_id: str
    task: str
    datasets: tuple[str, ...]
    per_dataset: dict[str, dict[str, ModelParameterEvidence]] = field(default_factory=dict)

    @property
    def evaluable(self) -> bool:
        """True only when EVERY dataset resolved EVERY declared term.

        The conjunction is the point: a PASS requires all of ``evaluated_on``.
        """
        if not self.per_dataset:
            return False
        return all(
            evidence and all(e.resolved for e in evidence.values())
            for evidence in self.per_dataset.values()
        )

    @property
    def counts_by_dataset(self) -> dict[str, dict[str, int]]:
        """The resolved counts per dataset; raises when the scope is not evaluable."""
        if not self.evaluable:
            raise ParameterScopeError(self.reason)
        return {
            dataset: {m: int(e.parameter_count) for m, e in evidence.items()}  # type: ignore[arg-type]
            for dataset, evidence in self.per_dataset.items()
        }

    @property
    def reason(self) -> str:
        if not self.per_dataset:
            return f"{self.gate_id}: the parameter scope declares no terms"
        problems = [
            f"{dataset}/{m}: {e.detail}"
            for dataset in self.datasets
            for m, e in sorted((self.per_dataset.get(dataset) or {}).items())
            if not e.resolved
        ]
        if not problems:
            return ""
        return f"{self.gate_id}: " + "; ".join(problems)

    def provenance(self) -> dict[str, Any]:
        return {
            "gate": self.gate_id,
            "task": self.task,
            "datasets": list(self.datasets),
            "conditioned_on_dataset": True,
            "evaluable": self.evaluable,
            "reason": self.reason,
            "per_dataset": {
                dataset: {m: e.as_dict() for m, e in sorted(evidence.items())}
                for dataset, evidence in self.per_dataset.items()
            },
            # A flat, honest summary of the question the gate actually asks.
            "per_dataset_counts": {
                dataset: {
                    m: (e.parameter_count if e.resolved else None)
                    for m, e in sorted(evidence.items())
                }
                for dataset, evidence in self.per_dataset.items()
            },
        }


def _protocol_aliases(protocol: Mapping[str, Any] | None) -> dict[str, str]:
    """Map registry ids to protocol ids and back (``gru`` <-> ``GRU``)."""
    out: dict[str, str] = {}
    if not protocol:
        return out
    from drososense.utils.config import protocol_model_id_bindings

    for record_id, protocol_id in protocol_model_id_bindings(protocol).items():
        out[str(record_id)] = str(protocol_id)
        out[str(protocol_id)] = str(record_id)
    return out


def _reservoir_size(record: Any) -> int | None:
    description = getattr(record, "model_description", None) or {}
    topology = description.get("topology") or {}
    value = topology.get("n_nodes")
    return None if value is None else int(value)


def _trainable(record: Any) -> int | None:
    description = getattr(record, "model_description", None) or {}
    value = description.get("n_trainable_parameters")
    return None if value is None else int(value)


def resolve_gate_parameter_scope(
    gate_id: str,
    records: Iterable[Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    declaration: Mapping[str, Any] | None = None,
    condition: str = "full",
) -> GateParameterScope:
    """Resolve one gate's parameter terms from matched result rows, PER DATASET.

    Protocol v1.5.2: a count is resolved inside each dataset's own matched
    registered configuration, and the gate's comparison must hold for every
    dataset in ``evaluated_on``. There is deliberately no dataset-agnostic count.

    Args:
        gate_id: The gate whose scope to resolve.
        records: Run records (any experiment); only rows matching the declared
            scope are read.
        protocol: The parsed protocol, for the registry/protocol id bindings.
        declaration: The ``parameter_evidence_scope`` block; loaded from v1.5.1
            when omitted.
        condition: The condition to resolve under.

    Returns:
        A :class:`GateParameterScope`. Its ``evaluable`` property says whether the
        conjunction over datasets resolved; an unevaluable scope carries the
        reason and any conflicting values rather than raising, so the gate can
        report UNEVALUABLE with its provenance.
    """
    block = dict(declaration or load_parameter_scope_declaration())
    gates = block.get("gates") or {}
    spec = gates.get(gate_id)
    if not spec:
        blank = ModelParameterEvidence(
            model="__scope__",
            status=MISSING_SCOPE,
            parameter_count=None,
            detail=(
                f"protocol v1.5.1 declares no parameter scope for {gate_id}; a gate "
                f"whose parameter source is undeclared cannot be evaluated"
            ),
        )
        return GateParameterScope(
            gate_id=gate_id, task="", datasets=(),
            per_dataset={"__scope__": {"__scope__": blank}},
        )

    task = str(spec.get("task", ""))
    datasets = tuple(str(d) for d in spec.get("datasets", ()))
    aliases = _protocol_aliases(protocol)
    rows = [r for r in records if getattr(r, "status", None) == "ok"]

    per_dataset: dict[str, dict[str, ModelParameterEvidence]] = {}
    for short in datasets:
        long_name = _long_name(short, protocol)
        evidence: dict[str, ModelParameterEvidence] = {}
        for term in spec.get("terms", []):
            model = str(term["model"])
            bindings = term.get("condition_bindings") or {}
            labels = tuple(str(x) for x in bindings.get(condition, ()))
            common = {
                "model": model,
                "experiment_labels": labels,
                "condition": condition,
                "task": task,
                "datasets": (short,),
            }
            if not labels:
                evidence[model] = ModelParameterEvidence(
                    status=UNBOUND_CONDITION,
                    parameter_count=None,
                    detail=(
                        f"condition {condition!r} is not bound to any experiment label for "
                        f"{model}; the declaration binds {sorted(bindings)}"
                    ),
                    **common,
                )
                continue

            wanted = {model, aliases.get(model, model)}
            matched = [
                r
                for r in rows
                if str(getattr(r, "model", "")) in wanted
                and str(getattr(r, "task", "")) == task
                and str(getattr(r, "experiment", "")) in labels
                and str(getattr(r, "dataset", "")) == long_name
            ]
            if not matched:
                evidence[model] = ModelParameterEvidence(
                    status=NO_RECORDS,
                    parameter_count=None,
                    detail=(
                        f"no ok {task} row for {model} in {list(labels)} on {short}; the "
                        f"term is unevaluable for this dataset rather than substituted "
                        f"from another dataset or another experiment"
                    ),
                    **common,
                )
                continue

            values = sorted({_trainable(r) for r in matched if _trainable(r) is not None})
            if not values:
                evidence[model] = ModelParameterEvidence(
                    status=NO_RECORDS,
                    parameter_count=None,
                    detail=(
                        f"{len(matched)} row(s) matched on {short} but none carries "
                        f"model_description.n_trainable_parameters"
                    ),
                    n_source_records=len(matched),
                    **common,
                )
                continue
            if len(values) > 1:
                by_value: dict[int, list[str]] = {}
                for r in matched:
                    value = _trainable(r)
                    if value is not None:
                        by_value.setdefault(value, []).append(str(getattr(r, "run_id", "")))
                detail = "; ".join(
                    f"{v} from {len(ids)} row(s) e.g. {ids[0]}"
                    for v, ids in sorted(by_value.items())
                )
                evidence[model] = ModelParameterEvidence(
                    status=CONFLICTING_VALUES,
                    parameter_count=None,
                    conflicting_values=tuple(values),
                    n_source_records=len(matched),
                    source_run_ids=tuple(sorted(str(getattr(r, "run_id", "")) for r in matched)),
                    config_hashes=tuple(sorted({str(getattr(r, "config_hash", "")) for r in matched})),
                    detail=(
                        f"{len(values)} different trainable-parameter counts for {model} on "
                        f"{short} inside the declared scope ({detail}); taking "
                        f"max/min/first/last would be a silent choice of one configuration "
                        f"over another"
                    ),
                    **common,
                )
                continue

            sizes = sorted({_reservoir_size(r) for r in matched if _reservoir_size(r) is not None})
            if len(sizes) > 1:
                evidence[model] = ModelParameterEvidence(
                    status=NO_RESERVOIR_SIZE,
                    parameter_count=None,
                    conflicting_values=tuple(values),
                    n_source_records=len(matched),
                    detail=(
                        f"the scope mixes reservoir sizes {sizes} for {model} on {short}; a "
                        f"count stated for one substrate cannot be reported for another"
                    ),
                    **common,
                )
                continue

            evidence[model] = ModelParameterEvidence(
                status=RESOLVED,
                parameter_count=values[0],
                reservoir_size=sizes[0] if sizes else None,
                n_source_records=len(matched),
                source_run_ids=tuple(sorted(str(getattr(r, "run_id", "")) for r in matched)),
                config_hashes=tuple(sorted({str(getattr(r, "config_hash", "")) for r in matched})),
                detail=(
                    f"{values[0]} trainable parameters on {short}, read from "
                    f"{len(matched)} matched row(s)"
                ),
                **common,
            )
        per_dataset[short] = evidence

    return GateParameterScope(
        gate_id=gate_id, task=task, datasets=datasets, per_dataset=per_dataset
    )


def _long_name(short: str, protocol: Mapping[str, Any] | None) -> str:
    """Map a protocol dataset short name (``D2``) to its config id."""
    if not protocol:
        return short
    from drososense.utils.config import protocol_dataset_symbols

    return str(protocol_dataset_symbols(protocol).get(short, short))


def scope_from_provenance(payload: Mapping[str, Any]) -> GateParameterScope:
    """Rebuild a scope from a written provenance block (for audits)."""

    def one(t: Mapping[str, Any]) -> ModelParameterEvidence:
        return ModelParameterEvidence(
            model=str(t["model"]),
            status=str(t["status"]),
            parameter_count=t["parameter_count"],
            experiment_labels=tuple(t.get("experiment", ())),
            condition=str(t.get("condition", "")),
            task=str(t.get("task", "")),
            datasets=tuple(t.get("datasets", ())),
            reservoir_size=t.get("reservoir_size"),
            n_source_records=int(t.get("n_source_records", 0) or 0),
            source_run_ids=tuple(t.get("source_run_ids", ())),
            config_hashes=tuple(t.get("config_hashes", ())),
            conflicting_values=tuple(t.get("conflicting_values", ())),
            detail=str(t.get("detail", "")),
        )

    per_dataset = {
        dataset: {m: one(entry) for m, entry in (terms or {}).items()}
        for dataset, terms in (payload.get("per_dataset") or {}).items()
    }
    return GateParameterScope(
        gate_id=str(payload.get("gate", "")),
        task=str(payload.get("task", "")),
        datasets=tuple(payload.get("datasets", ())),
        per_dataset=per_dataset,
    )


def dumps(scope: GateParameterScope) -> str:
    """JSON rendering of a scope's provenance."""
    return json.dumps(scope.provenance(), indent=2, sort_keys=True)
