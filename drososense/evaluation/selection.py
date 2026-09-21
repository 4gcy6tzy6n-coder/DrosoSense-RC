"""Hyperparameter selection against the grid the protocol declares.

Protocol §17 (``hyperparameter_selection``) declares the tuning space, the split
the choice is made on and the metric it is made by. Nothing read that block: the
runner took per-model defaults, so a model tuned outside the declared grid was
indistinguishable from one tuned inside it, and the whole section was
decoration. This module is the reader.

Two jobs, kept apart on purpose:

* :class:`HyperparameterGrid` refuses a value the protocol did not declare. It is
  the gate the runner calls at entry, so a run whose knobs fall outside the grid
  stops before any data is touched.
* :func:`select_hyperparameters` makes the choice itself — per fold, on that
  fold's validation specimens, by the metric the protocol names for the task.
  The test split is never read, and the choice is returned in a form the run
  record can carry so it can be checked after the fact.

The knob names are the protocol's; the models use shorter ones. The binding
lives in :data:`KNOB_BINDINGS` rather than being left implicit, so a protocol
knob that no model parameter answers to raises at load time instead of matching
nothing forever.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from drososense.baselines.registry import build_model
from drososense.data.pipeline import FoldTensors
from drososense.evaluation.metrics import classification_metrics, regression_metrics
from drososense.utils.config import load_protocol, protocol_metric_direction

GRID_SECTION = "hyperparameter_selection"
WINDOW_LENGTH_KNOB = "window_length"

# Protocol knob -> the parameter name the model layer answers to. ``None`` marks
# a knob the runner owns: the window length is a property of the tensors that
# were built, not something passed to ``build_model``, so it is validated
# against ``BenchmarkConfig.window_lengths`` instead.
KNOB_BINDINGS: Mapping[str, str | None] = {
    WINDOW_LENGTH_KNOB: None,
    "readout_regularisation": "ridge_lambda",
    "reservoir_leak_alpha": "leak",
    "reservoir_gain_g": "gain",
    "input_scale_gamma": "input_scale",
    "spectral_scaling": "spectral_radius",
}

# ``hyperparameter_selection.spectral_scaling_rule`` is prose, so it cannot be
# read mechanically: "chosen once per (dataset, seed, fold) and applied
# identically to R0..R6". The set of knobs the rule covers is therefore declared
# here, and ``tests/test_hyperparameter_selection.py`` asserts the protocol still
# says "once per (dataset, seed, fold)" and still names the reservoir family. If
# that wording changes, the test fails and a human revisits the sharing rather
# than the code silently keeping it.
SHARED_KNOBS: frozenset[str] = frozenset({"spectral_scaling"})

# The protocol names the selection split ``validation``; the tensors call that
# split ``val``. ``test`` is listed so the binding is total, and then refused:
# selecting on the test split is what §17 exists to forbid.
SPLIT_BINDINGS: Mapping[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

SUPPORTED_SCOPE = "per_fold"


class OutOfGridError(ValueError):
    """A hyperparameter value the protocol's grid does not declare."""


def _matches(value: Any, candidates: Sequence[Any]) -> bool:
    """Whether a value is one of the declared candidates.

    Numeric comparison is by value, so ``1.0e-4`` and ``0.0001`` are the same
    candidate and a YAML round-trip cannot make a legal value illegal. Booleans
    are compared by identity so ``True`` never satisfies a candidate of ``1``.

    Args:
        value: The value to check.
        candidates: The declared candidates.

    Returns:
        True when the value is declared.
    """
    for candidate in candidates:
        if isinstance(value, bool) or isinstance(candidate, bool):
            if value is candidate:
                return True
            continue
        if isinstance(value, (int, float)) and isinstance(candidate, (int, float)):
            if float(value) == float(candidate):
                return True
            continue
        if value == candidate:
            return True
    return False


@dataclass(frozen=True)
class HyperparameterGrid:
    """The tuning space protocol §17 declares, and the only values it allows.

    Attributes:
        candidates: Knob name -> declared values, in declaration order.
        window_length_candidates: The window lengths, which are declared twice
            (here and in ``preprocessing.windowing.length_candidates``) and are
            required to agree.
    """

    candidates: Mapping[str, tuple[Any, ...]]
    window_length_candidates: tuple[int, ...] = ()

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any] | None = None) -> "HyperparameterGrid":
        """Build the grid from the active protocol.

        Args:
            protocol: Parsed protocol; the active one when omitted.

        Returns:
            The declared grid.

        Raises:
            ValueError: If the section or its grid is missing or malformed, if a
                declared knob answers to no model parameter, or if the two
                declarations of the window-length candidates disagree.
        """
        document = protocol if protocol is not None else load_protocol()
        declared = document.get(GRID_SECTION, {}).get("grid")
        if not isinstance(declared, Mapping) or not declared:
            raise ValueError(f"protocol declares no {GRID_SECTION}.grid")

        candidates: dict[str, tuple[Any, ...]] = {}
        for name, values in declared.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{GRID_SECTION}.grid.{name} is not a non-empty list")
            candidates[str(name)] = tuple(values)

        unbound = sorted(set(candidates) - set(KNOB_BINDINGS))
        if unbound:
            raise ValueError(
                f"{GRID_SECTION}.grid declares knobs no model parameter answers to: {unbound}. "
                f"Add the binding to selection.KNOB_BINDINGS, or a selector that draws from this "
                f"grid would tune a knob nothing reads."
            )

        window_lengths = tuple(int(value) for value in candidates.get(WINDOW_LENGTH_KNOB, ()))

        # The window-length candidates are declared in two places. Rather than
        # pick one and hope, require them to agree: a divergence is a protocol
        # defect that would otherwise make one of the two declarations dead.
        preprocessing = document.get("preprocessing", {}).get("windowing", {})
        length_candidates = tuple(int(v) for v in preprocessing.get("length_candidates", ()))
        if window_lengths and length_candidates and window_lengths != length_candidates:
            raise ValueError(
                f"{GRID_SECTION}.grid.{WINDOW_LENGTH_KNOB} is {list(window_lengths)} but "
                f"preprocessing.windowing.length_candidates is {list(length_candidates)}; the "
                f"window length is declared in both places and they must agree."
            )

        return cls(candidates=candidates, window_length_candidates=window_lengths)

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared knob, in declaration order."""
        return tuple(self.candidates)

    def model_knob_names(self) -> tuple[str, ...]:
        """The knobs a model receives, i.e. every knob but the window length."""
        return tuple(name for name in self.candidates if name != WINDOW_LENGTH_KNOB)

    def is_declared(self, name: str) -> bool:
        """Whether a knob is part of the declared grid.

        Args:
            name: Knob name.

        Returns:
            True when the protocol declares it.
        """
        return name in self.candidates

    def candidates_for(self, name: str) -> tuple[Any, ...]:
        """Return a knob's declared candidates.

        Args:
            name: Knob name.

        Returns:
            The candidate values.

        Raises:
            KeyError: If the knob is not declared.
        """
        if name not in self.candidates:
            raise KeyError(f"{name!r} is not a declared knob; declared: {list(self.candidates)}")
        return self.candidates[name]

    def check(self, name: str, value: Any) -> None:
        """Refuse a value the protocol did not declare for a knob.

        Args:
            name: Protocol knob name.
            value: The value to check.

        Raises:
            OutOfGridError: If the knob is declared and the value is not one of
                its candidates.
        """
        if not self.is_declared(name):
            return
        candidates = self.candidates_for(name)
        if not _matches(value, candidates):
            raise OutOfGridError(
                f"{GRID_SECTION}.grid.{name} = {value!r} is outside the pre-registered grid "
                f"{list(candidates)}. A model tuned outside the declared grid cannot be "
                f"reported as a protocol result; extend the protocol with a new version file "
                f"instead of tuning past it."
            )

    def check_params(self, params: Mapping[str, Any], model_id: str = "") -> None:
        """Refuse out-of-grid values among a model's hyperparameters.

        Only knobs the grid declares are checked. A model's architecture
        parameters (``n_estimators``, ``hidden_size``, ``epochs``) are not part
        of the tuning space §17 froze, and passing them is not a violation.

        Args:
            params: The parameters a caller supplied for one model.
            model_id: Model id, for the error message.

        Raises:
            OutOfGridError: If a declared knob carries an undeclared value.
        """
        for name in self.model_knob_names():
            alias = KNOB_BINDINGS[name]
            if alias is None or alias not in params:
                continue
            try:
                self.check(name, params[alias])
            except OutOfGridError as exc:
                raise OutOfGridError(f"{model_id or 'model'}: {exc}") from None

    def check_window_lengths(self, window_lengths: Sequence[int]) -> None:
        """Refuse window lengths the protocol did not declare.

        Args:
            window_lengths: Window lengths a run asked for.

        Raises:
            OutOfGridError: If one of them is outside the declared candidates.
        """
        for value in window_lengths:
            self.check(WINDOW_LENGTH_KNOB, value)

    def combinations(
        self, names: Sequence[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Enumerate the grid points over the named knobs.

        Args:
            names: Knobs to vary; every model knob when omitted.

        Yields:
            One mapping per grid point, in declaration order.
        """
        selected = tuple(self.model_knob_names() if names is None else names)
        value_lists = [self.candidates_for(name) for name in selected]
        for point in itertools.product(*value_lists):
            yield dict(zip(selected, point))

    def size(self, names: Sequence[str] | None = None) -> int:
        """Return the number of grid points over the named knobs.

        Args:
            names: Knobs to count over; every model knob when omitted.

        Returns:
            The product of the candidate counts.
        """
        selected = tuple(self.model_knob_names() if names is None else names)
        total = 1
        for name in selected:
            total *= len(self.candidates_for(name))
        return total

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, for the run config fingerprint.

        Returns:
            Mapping of knob name to its declared candidates.
        """
        return {name: list(values) for name, values in self.candidates.items()}


@dataclass(frozen=True)
class SelectionSpec:
    """How §17 says the choice is made.

    Attributes:
        scope: How often selection runs. Only ``per_fold`` is implemented; any
            other declared value raises rather than being ignored.
        selection_split: The split the choice is made on.
        selection_metric: Task -> metric name, e.g. ``validation_macro_f1``.
        spectral_scaling_rule: The prose rule, carried so the run record can
            state the rule the sharing implements.
    """

    scope: str
    selection_split: str
    selection_metric: Mapping[str, str]
    spectral_scaling_rule: str = ""

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any] | None = None) -> "SelectionSpec":
        """Build the spec from the active protocol.

        Args:
            protocol: Parsed protocol; the active one when omitted.

        Returns:
            The declared selection spec.

        Raises:
            ValueError: If the section is missing, the scope is one this module
                does not implement, or the split is not one it can address.
        """
        document = protocol if protocol is not None else load_protocol()
        section = document.get(GRID_SECTION)
        if not isinstance(section, Mapping):
            raise ValueError(f"protocol declares no {GRID_SECTION} block")

        scope = str(section.get("scope", ""))
        if scope != SUPPORTED_SCOPE:
            raise ValueError(
                f"{GRID_SECTION}.scope is {scope!r}; only {SUPPORTED_SCOPE!r} is implemented. "
                f"Ignoring an unimplemented scope would silently select on the wrong split."
            )

        selection_split = str(section.get("selection_split", ""))
        if selection_split not in SPLIT_BINDINGS:
            raise ValueError(
                f"{GRID_SECTION}.selection_split is {selection_split!r}; expected one of "
                f"{sorted(SPLIT_BINDINGS)}"
            )

        metrics = section.get("selection_metric")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"{GRID_SECTION}.selection_metric declares no task")

        return cls(
            scope=scope,
            selection_split=selection_split,
            selection_metric={str(k): str(v) for k, v in metrics.items()},
            spectral_scaling_rule=str(section.get("spectral_scaling_rule", "")),
        )

    def split_name(self) -> str:
        """Return the tensor split this spec selects on.

        Returns:
            ``train``, ``val`` or ``test``.

        Raises:
            ValueError: If the declared split is the test split.
        """
        resolved = SPLIT_BINDINGS[self.selection_split]
        if resolved == "test":
            raise ValueError(
                f"{GRID_SECTION}.selection_split is 'test'. §17 forbids selecting on the test "
                f"split; a run that did would report a tuned-on-test number."
            )
        return resolved

    def metric_for(self, task: str) -> str:
        """Return the declared selection metric for a task.

        Args:
            task: ``classification`` or ``regression``.

        Returns:
            The declared metric name, e.g. ``validation_macro_f1``.

        Raises:
            KeyError: If the task declares no selection metric.
        """
        if task not in self.selection_metric:
            raise KeyError(
                f"task {task!r} declares no {GRID_SECTION}.selection_metric; declared: "
                f"{sorted(self.selection_metric)}"
            )
        return self.selection_metric[task]

    def metric_name_for(self, task: str) -> str:
        """Return the metric key, with the selection split's prefix stripped.

        The declared names are ``<split>_<metric>``, so the split and the metric
        are both read from this one field rather than the split being assumed.

        Args:
            task: ``classification`` or ``regression``.

        Returns:
            The metric key as :mod:`drososense.evaluation.metrics` names it.

        Raises:
            ValueError: If the declared name does not carry the split prefix.
        """
        declared = self.metric_for(task)
        prefix = f"{self.selection_split}_"
        if not declared.startswith(prefix):
            raise ValueError(
                f"{GRID_SECTION}.selection_metric.{task} is {declared!r}, which does not name "
                f"the declared selection split {self.selection_split!r} as a prefix. The metric "
                f"and the split it is measured on have to be readable from the one name."
            )
        return declared[len(prefix):]

    def shared_knobs(self) -> frozenset[str]:
        """The knobs the spectral-scaling rule holds fixed across models.

        Returns:
            The declared shared knob names.
        """
        return SHARED_KNOBS


@dataclass(frozen=True)
class SelectionResult:
    """One fold's choice, and enough context to check it after the fact.

    Attributes:
        model_id: Model the choice was made for.
        task: Task the choice was made for.
        seed: Split seed.
        fold_id: Fold the choice was made for.
        params: Model-layer parameters drawn from the grid.
        shared: Choices for knobs that are shared across models in the group.
        grid_point: The same choice in the protocol's own knob names.
        metric: The metric the choice maximised or minimised.
        direction: ``maximize`` or ``minimize``.
        score: The chosen candidate's score on the validation split.
        n_candidates: How many grid points were scored.
        scope: The scope the choice was made under.
        selection_split: The split the choice was made on.
    """

    model_id: str
    task: str
    seed: int
    fold_id: int
    params: Mapping[str, Any]
    shared: Mapping[str, Any]
    grid_point: Mapping[str, Any]
    metric: str
    direction: str
    score: float
    n_candidates: int
    scope: str
    selection_split: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the run record.

        Returns:
            Mapping of the choice and the rule it was made under.
        """
        return {
            "model_id": self.model_id,
            "task": self.task,
            "seed": self.seed,
            "fold_id": self.fold_id,
            "params": dict(self.params),
            "shared": dict(self.shared),
            "grid_point": dict(self.grid_point),
            "metric": self.metric,
            "direction": self.direction,
            "score": self.score,
            "n_candidates": self.n_candidates,
            "scope": self.scope,
            "selection_split": self.selection_split,
            "spectral_scaling_shared": sorted(SHARED_KNOBS),
        }


def _score_on_validation(
    model: Any,
    tensors: FoldTensors,
    split_name: str,
    task: str,
    n_classes: int | None,
) -> dict[str, Any]:
    """Fit on the training split and score the selection split.

    Args:
        model: The constructed model.
        tensors: The fold's tensors.
        split_name: ``train``, ``val`` or ``test``.
        task: ``classification`` or ``regression``.
        n_classes: Class count for the AUROC denominator.

    Returns:
        The metric mapping for the selection split.
    """
    train = tensors.train
    scored = tensors.split(split_name)
    if task == "classification":
        model.fit(train.X, train.y_class)
        return classification_metrics(
            scored.y_class,
            model.predict(scored.X),
            model.predict_proba(scored.X),
            n_classes=n_classes,
        )
    model.fit(train.X, train.y_reg)
    return regression_metrics(scored.y_reg, model.predict(scored.X))


def _is_better(candidate: float, incumbent: float, direction: str) -> bool:
    """Whether a candidate score beats the incumbent.

    A NaN never beats a real number, and a real number always beats a NaN, so a
    degenerate candidate cannot win a selection it scored nothing on.

    Args:
        candidate: The new score.
        incumbent: The best score so far.
        direction: ``maximize`` or ``minimize``.

    Returns:
        True when the candidate should replace the incumbent.
    """
    candidate_nan = candidate != candidate
    incumbent_nan = incumbent != incumbent
    if candidate_nan:
        return False
    if incumbent_nan:
        return True
    return candidate > incumbent if direction == "maximize" else candidate < incumbent


def select_hyperparameters(
    *,
    model_id: str,
    task: str,
    seed: int,
    fold_tensors: FoldTensors,
    n_channels: int,
    n_classes: int | None = None,
    grid: HyperparameterGrid | None = None,
    spec: SelectionSpec | None = None,
    protocol: Mapping[str, Any] | None = None,
    shared: Mapping[str, Any] | None = None,
    builder: Callable[..., Any] = build_model,
) -> SelectionResult:
    """Choose one model's hyperparameters on one fold's validation split.

    Every candidate comes from the declared grid, so the selector cannot propose
    a value the protocol did not pre-register. The test split is not read: the
    caller scores it afterwards, once, with the returned parameters.

    Args:
        model_id: Registered model id.
        task: ``classification`` or ``regression``.
        seed: Model and split seed.
        fold_tensors: The fold, carrying the training and validation windows.
        n_channels: Input channel count.
        n_classes: Class count for the AUROC denominator.
        grid: The declared grid; loaded from the protocol when omitted.
        spec: The declared selection spec; loaded from the protocol when omitted.
        protocol: Parsed protocol; the active one when omitted.
        shared: Values for shared knobs already chosen for this
            ``(dataset, seed, fold)``. When omitted they are searched here and
            returned in the result, so the next model in the group can be given
            the identical value rather than choosing its own.
        builder: Model constructor; injectable so a test can avoid a backend.

    Returns:
        The :class:`SelectionResult`.

    Raises:
        KeyError: If the task declares no selection metric.
        ValueError: If the fold has no windows in the selection split, or a
            pinned shared knob is not one the rule covers.
    """
    resolved_grid = grid if grid is not None else HyperparameterGrid.from_protocol(protocol)
    resolved_spec = spec if spec is not None else SelectionSpec.from_protocol(protocol)

    metric_name = resolved_spec.metric_name_for(task)
    document = protocol if protocol is not None else load_protocol()
    direction = protocol_metric_direction(document, metric_name)
    split_name = resolved_spec.split_name()

    if len(fold_tensors.split(split_name)) == 0:
        raise ValueError(
            f"fold {fold_tensors.fold.fold_id}: the {resolved_spec.selection_split} split is "
            f"empty, so there is nothing to select on"
        )

    pinned = dict(shared or {})
    unpinned = [name for name in resolved_grid.model_knob_names() if name not in pinned]
    for name, value in pinned.items():
        if name not in SHARED_KNOBS:
            raise ValueError(
                f"{name!r} was passed as shared, but §17 shares only {sorted(SHARED_KNOBS)} "
                f"across models; sharing anything else would pin a knob per model family."
            )
        if not resolved_grid.is_declared(name):
            # check() ignores an undeclared knob on purpose — a model's own
            # parameters are not part of the frozen grid — so pinning one has to
            # be refused here, or sharing would be a way to tune outside it.
            raise OutOfGridError(
                f"{name!r} is shared across models but the declared grid does not carry it, "
                f"so pinning it would tune a knob §17 never froze."
            )
        resolved_grid.check(name, value)

    best_point: dict[str, Any] | None = None
    best_score = float("nan")
    considered = 0
    for varied in resolved_grid.combinations(unpinned):
        point = {**pinned, **varied}
        params = {
            alias: point[name]
            for name in point
            if (alias := KNOB_BINDINGS.get(name)) is not None
        }
        model = builder(model_id, task, seed, params, n_channels=n_channels)
        metrics = _score_on_validation(model, fold_tensors, split_name, task, n_classes)
        score = float(metrics[metric_name])
        considered += 1
        if best_point is None or _is_better(score, best_score, direction):
            best_point, best_score = point, score

    if best_point is None:
        raise ValueError(
            f"{model_id}/{task}: the declared grid yields no candidate over {unpinned}"
        )

    folded = fold_tensors.fold
    return SelectionResult(
        model_id=model_id,
        task=task,
        seed=seed,
        fold_id=folded.fold_id,
        params={
            alias: best_point[name]
            for name in best_point
            if (alias := KNOB_BINDINGS.get(name)) is not None
        },
        shared={name: best_point[name] for name in best_point if name in SHARED_KNOBS},
        grid_point=dict(best_point),
        metric=metric_name,
        direction=direction,
        score=best_score,
        n_candidates=considered,
        scope=resolved_spec.scope,
        selection_split=resolved_spec.selection_split,
    )


def validate_run_hyperparameters(
    *,
    window_lengths: Sequence[int],
    model_params: Mapping[str, Mapping[str, Any]],
    grid: HyperparameterGrid | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> HyperparameterGrid:
    """Refuse a run whose window lengths or model parameters are out of grid.

    This is the runner's entry gate. It is deliberately separate from
    :func:`select_hyperparameters`: a run that supplies its own parameters —
    every M1 run does — is not selecting, but it still may not tune past the
    grid the protocol froze.

    Args:
        window_lengths: Window lengths the run asked for.
        model_params: Per-model parameter overrides.
        grid: The declared grid; loaded from the protocol when omitted.
        protocol: Parsed protocol; the active one when omitted.

    Returns:
        The grid the run was checked against.

    Raises:
        OutOfGridError: If a window length or a declared knob is out of grid.
    """
    resolved = grid if grid is not None else HyperparameterGrid.from_protocol(protocol)
    resolved.check_window_lengths(window_lengths)
    for model_id, params in model_params.items():
        resolved.check_params(params, model_id)
    return resolved


__all__ = [
    "GRID_SECTION",
    "KNOB_BINDINGS",
    "SHARED_KNOBS",
    "HyperparameterGrid",
    "OutOfGridError",
    "SelectionResult",
    "SelectionSpec",
    "select_hyperparameters",
    "validate_run_hyperparameters",
]
