"""The declared tuning grid is enforced, and the selector only draws from it.

Protocol §17 declared a grid, a split and a metric, and nothing read any of it:
the runner took per-model defaults, so a model tuned outside the pre-registered
space was indistinguishable from one tuned inside it. These tests hold the two
halves of the fix apart — the gate that refuses an out-of-grid value, and the
selector that can only propose an in-grid one — because a gate alone would still
let a caller tune by hand past the grid, and a selector alone would leave the
supplied-parameter path unguarded.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from drososense.baselines.registry import MODEL_IDS, build_model, model_availability
from drososense.evaluation.runner import BenchmarkConfig, run_benchmark
from drososense.evaluation.selection import (
    KNOB_BINDINGS,
    SHARED_KNOBS,
    HyperparameterGrid,
    OutOfGridError,
    SelectionSpec,
    select_hyperparameters,
)
from drososense.utils.config import load_protocol

WINDOW_LENGTH = 8
N_CHANNELS = 2


@pytest.fixture(scope="module")
def grid() -> HyperparameterGrid:
    return HyperparameterGrid.from_protocol()


@pytest.fixture(scope="module")
def spec() -> SelectionSpec:
    return SelectionSpec.from_protocol()


# ---------------------------------------------------------------------------
# The declared grid
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_every_declared_knob_answers_to_a_model_parameter(grid):
    """A knob no model parameter answers to would be tuneable by nothing."""
    protocol = load_protocol()
    declared = set(protocol["hyperparameter_selection"]["grid"])
    assert declared == set(grid.candidates)
    assert declared <= set(KNOB_BINDINGS), (
        "from_protocol raises on an unbound knob; reaching here with a gap means the check moved"
    )
    assert grid.model_knob_names() == tuple(
        name for name in grid.candidates if name != "window_length"
    )


@pytest.mark.unit
def test_the_window_length_candidates_are_declared_twice_and_agree(grid):
    """Two declarations of one thing is a divergence waiting to happen."""
    protocol = load_protocol()
    assert grid.window_length_candidates == tuple(
        protocol["preprocessing"]["windowing"]["length_candidates"]
    )


@pytest.mark.unit
def test_the_grid_comes_from_the_protocol_not_from_a_constant(grid):
    """The values are the frozen ones, read from the file rather than restated."""
    protocol = load_protocol()
    for name, values in protocol["hyperparameter_selection"]["grid"].items():
        assert list(grid.candidates_for(name)) == list(values)


@pytest.mark.unit
def test_a_value_outside_the_grid_is_refused(grid):
    """The grid is a gate, not a suggestion."""
    with pytest.raises(OutOfGridError, match="outside the pre-registered grid"):
        grid.check("reservoir_leak_alpha", 0.42)
    with pytest.raises(OutOfGridError):
        grid.check("window_length", 7)
    # ...and the declared values are accepted, by value rather than by spelling.
    grid.check("readout_regularisation", 0.0001)
    grid.check("readout_regularisation", 1.0e-4)
    grid.check("window_length", 16)


@pytest.mark.unit
def test_a_model_parameter_is_checked_under_its_protocol_name(grid):
    """The binding is what makes the grid enforceable against real models."""
    grid.check_params({"spectral_radius": 0.9, "reservoir_size": 200}, "esn")
    with pytest.raises(OutOfGridError, match="spectral_scaling"):
        grid.check_params({"spectral_radius": 0.7}, "esn")
    with pytest.raises(OutOfGridError, match="reservoir_leak_alpha"):
        grid.check_params({"leak": 0.25}, "esn")


@pytest.mark.unit
def test_architecture_parameters_are_not_part_of_the_grid(grid):
    """§17 froze a tuning space, not the whole parameter dict."""
    grid.check_params(
        {"n_estimators": 501, "hidden_size": 77, "epochs": 3, "reservoir_size": 150}, "gru"
    )


@pytest.mark.unit
def test_every_model_default_is_inside_the_declared_grid(grid):
    """The gap that made the grid decorative: a default outside it went unnoticed.

    The runner merges a model's own defaults under whatever the caller supplied,
    so a default outside the grid is a tuned value the protocol never declared —
    and nothing would have said so.
    """
    availability = model_availability()
    checked = []
    for model_id in MODEL_IDS:
        if not availability[model_id]["available"]:
            continue
        model = build_model(model_id, "classification", 0, {}, n_channels=5)
        params = getattr(model, "params", None)
        if not isinstance(params, dict):
            continue
        grid.check_params(params, f"{model_id} default")
        checked.append(model_id)
    assert checked, "no model could be constructed, so nothing was checked"


# ---------------------------------------------------------------------------
# The declared selection spec
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_selection_spec_reads_the_protocol(spec):
    protocol = load_protocol()["hyperparameter_selection"]
    assert spec.scope == protocol["scope"] == "per_fold"
    assert spec.selection_split == protocol["selection_split"] == "validation"
    assert spec.metric_for("classification") == "validation_macro_f1"
    assert spec.metric_for("regression") == "validation_mae"


@pytest.mark.unit
def test_the_metric_and_the_split_are_read_from_one_name(spec):
    """The split is not assumed; it is the prefix of the declared metric."""
    assert spec.metric_name_for("classification") == "macro_f1"
    assert spec.metric_name_for("regression") == "mae"
    assert spec.split_name() == "val"


@pytest.mark.unit
def test_selecting_on_the_test_split_is_refused():
    """§17's whole point: the test split takes no part in the choice."""
    spec = SelectionSpec(
        scope="per_fold",
        selection_split="test",
        selection_metric={"classification": "test_macro_f1"},
    )
    with pytest.raises(ValueError, match="forbids selecting on the test split"):
        spec.split_name()


@pytest.mark.unit
def test_an_unimplemented_scope_is_refused_rather_than_ignored():
    protocol = load_protocol()
    protocol["hyperparameter_selection"] = {
        **protocol["hyperparameter_selection"],
        "scope": "per_dataset",
    }
    with pytest.raises(ValueError, match="only 'per_fold' is implemented"):
        SelectionSpec.from_protocol(protocol)


@pytest.mark.unit
def test_the_spectral_scaling_rule_is_pinned_to_the_protocol_wording(spec):
    """The sharing rule is prose, so it is pinned by an assertion, not parsed.

    If the protocol stops saying the scaling is chosen once per (dataset, seed,
    fold) and applied to the whole reservoir family, the set of knobs shared
    across models has to be revisited by a human.
    """
    assert "once per" in spec.spectral_scaling_rule
    assert "(dataset, seed, fold)" in spec.spectral_scaling_rule
    assert "R0" in spec.spectral_scaling_rule and "R6" in spec.spectral_scaling_rule
    assert SHARED_KNOBS == {"spectral_scaling"}


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------
class _FakeWindows:
    """Just enough of a window set for the selector.

    The first channel carries the label, so a stub model can score exactly
    against the split it was handed without ever seeing the other split's
    targets.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.X = x
        self.y_class = y
        self.y_reg = y.astype(float)

    def __len__(self) -> int:
        return int(self.X.shape[0])


class _FakeFold:
    """A fold with a stable id."""

    fold_id = 0


class _FakeTensors:
    """A fold whose channel 0 is the label, so scoring is deterministic.

    ``split`` refuses ``test`` outright: §17 forbids selecting on it, so a
    selector that ever asked for it would fail here rather than quietly tune.
    """

    def __init__(self, n_train: int = 24, n_val: int = 12) -> None:
        rng = np.random.default_rng(0)
        self.fold = _FakeFold()
        self.requested: list[str] = []
        self._splits = {
            "train": self._make(n_train, rng),
            "val": self._make(n_val, rng),
        }

    @staticmethod
    def _make(n: int, rng: np.random.Generator) -> _FakeWindows:
        labels = np.tile(np.arange(4), n // 4 + 1)[:n]
        x = rng.normal(size=(n, N_CHANNELS))
        x[:, 0] = labels
        return _FakeWindows(x, labels)

    @property
    def train(self) -> _FakeWindows:
        return self._splits["train"]

    def split(self, name: str) -> _FakeWindows:
        self.requested.append(name)
        if name == "test":
            raise AssertionError("the selector read the test split")
        return self._splits[name]


class _StubModel:
    """A model whose accuracy is fixed by construction.

    A ``quality`` of 1.0 predicts every validation label correctly and scores a
    perfect macro-F1 (and a zero MAE); 0.0 gets none of them right. In between
    is monotone, which is all a selection needs to be deterministic.
    """

    def __init__(self, task: str, params: dict, quality: float) -> None:
        self.task = task
        self.params = dict(params)
        self.quality = float(quality)

    def fit(self, x, y) -> None:  # noqa: D102 - the stub learns nothing
        return None

    def predict(self, x) -> np.ndarray:
        labels = np.asarray(x)[:, 0]
        if self.task == "regression":
            return labels + (1.0 - self.quality)
        correct = int(round(self.quality * len(labels)))
        out = labels.astype(int).copy()
        out[correct:] = (out[correct:] + 1) % 4
        return out

    def predict_proba(self, x) -> np.ndarray:
        predictions = np.asarray(self.predict(x)).astype(int) % 4
        proba = np.zeros((len(predictions), 4))
        proba[np.arange(len(predictions)), predictions] = 1.0
        return proba


class _StubBuilder:
    """Builds :class:`_StubModel`, scoring each grid point by a lookup table."""

    def __init__(self, table: dict) -> None:
        self.table = table
        self.built: list[dict] = []

    def __call__(self, model_id, task, seed, params, n_channels=None):
        self.built.append(dict(params))
        key = tuple(sorted(params.items()))
        return _StubModel(task, params, self.table[key])


def _tiny_grid(*leaks: float) -> HyperparameterGrid:
    return HyperparameterGrid(
        candidates={
            "readout_regularisation": (1.0e-3, 1.0e-2),
            "reservoir_leak_alpha": tuple(leaks),
        },
        window_length_candidates=(WINDOW_LENGTH,),
    )


def _quality_table(names: tuple, values: tuple, quality) -> dict:
    """Build a grid-point -> quality table.

    The key is the model-layer parameter dict the selector will build, derived
    through :data:`KNOB_BINDINGS` rather than restated — so a binding that
    changed would break these tests rather than let them pass on a stale name.

    Args:
        names: Protocol knob names, in the order ``values`` supplies them.
        values: One tuple of candidates per name.
        quality: Callable over one candidate per name.

    Returns:
        Mapping of a sorted parameter tuple to a quality in ``[0, 1]``.
    """
    table: dict = {}
    for point in itertools.product(*values):
        params = tuple(
            sorted(
                (KNOB_BINDINGS[name], value)
                for name, value in zip(names, point)
                if KNOB_BINDINGS[name] is not None
            )
        )
        table[params] = quality(*point)
    return table


def _select(tensors, grid, spec, builder, **kwargs):
    return select_hyperparameters(
        model_id="esn",
        task=kwargs.pop("task", "classification"),
        seed=0,
        fold_tensors=tensors,
        n_channels=N_CHANNELS,
        n_classes=4,
        grid=grid,
        spec=spec,
        builder=builder,
        **kwargs,
    )


@pytest.mark.unit
def test_the_selector_draws_only_from_the_declared_grid(spec):
    """No candidate outside the grid can be proposed, by construction."""
    grid = _tiny_grid(0.1, 0.3, 0.5)
    builder = _StubBuilder(
        _quality_table(
            ("readout_regularisation", "reservoir_leak_alpha"),
            (grid.candidates_for("readout_regularisation"),
             grid.candidates_for("reservoir_leak_alpha")),
            lambda _reg, leak: leak / 0.5,
        )
    )
    result = _select(_FakeTensors(), grid, spec, builder)

    assert result.n_candidates == grid.size() == 6
    assert result.metric == "macro_f1"
    assert result.direction == "maximize"
    # The best candidate is the largest leak, reached by maximising macro_f1.
    assert result.params == {"ridge_lambda": 1.0e-3, "leak": 0.5}
    assert result.score == pytest.approx(1.0)
    for candidate in builder.built:
        assert candidate["leak"] in grid.candidates_for("reservoir_leak_alpha")
        assert candidate["ridge_lambda"] in grid.candidates_for("readout_regularisation")


@pytest.mark.unit
def test_a_minimised_metric_selects_the_smallest_value(spec):
    """Regression selects on validation MAE, so a smaller value wins.

    The quality here is the mirror of the classification test's: the candidate
    that would win on macro-F1 loses on MAE. That is what makes this a test of
    the direction rather than of the search.
    """
    grid = _tiny_grid(0.1, 0.5)
    builder = _StubBuilder(
        _quality_table(
            ("readout_regularisation", "reservoir_leak_alpha"),
            (grid.candidates_for("readout_regularisation"),
             grid.candidates_for("reservoir_leak_alpha")),
            lambda _reg, leak: 1.0 - leak / 0.5,
        )
    )
    result = _select(_FakeTensors(), grid, spec, builder, task="regression")

    assert result.metric == "mae"
    assert result.direction == "minimize"
    assert result.params["leak"] == 0.1
    assert result.score == pytest.approx(0.2)


@pytest.mark.unit
def test_the_test_split_is_never_read(spec):
    """The property §17 exists for, asserted on the access rather than the intent."""
    tensors = _FakeTensors()
    grid = _tiny_grid(0.3)
    builder = _StubBuilder(
        _quality_table(
            ("readout_regularisation", "reservoir_leak_alpha"),
            (grid.candidates_for("readout_regularisation"),
             grid.candidates_for("reservoir_leak_alpha")),
            lambda _reg, _leak: 1.0,
        )
    )
    _select(tensors, grid, spec, builder)

    assert tensors.requested, "the selection split was never read at all"
    assert "test" not in tensors.requested, f"selection read {tensors.requested}"


@pytest.mark.unit
def test_a_shared_knob_is_taken_once_and_reused(spec):
    """§17: one spectral scaling per (dataset, seed, fold), for every reservoir.

    The first model in the group selects it; the second is handed the same
    value, so R0 and its control cannot end up on differently scaled graphs —
    which would make the primary contrast uninterpretable.
    """
    tensors = _FakeTensors()
    grid = HyperparameterGrid(
        candidates={"spectral_scaling": (0.9, 1.1), "reservoir_leak_alpha": (0.3,)},
        window_length_candidates=(WINDOW_LENGTH,),
    )
    builder = _StubBuilder(
        _quality_table(
            ("spectral_scaling", "reservoir_leak_alpha"),
            (grid.candidates_for("spectral_scaling"),
             grid.candidates_for("reservoir_leak_alpha")),
            lambda scaling, _leak: scaling,
        )
    )

    first = _select(tensors, grid, spec, builder)
    assert first.shared == {"spectral_scaling": 1.1}
    assert first.grid_point["spectral_scaling"] == 1.1

    builder.built.clear()
    second = _select(tensors, grid, spec, builder, shared=first.shared)
    assert second.shared == first.shared
    assert all(point["spectral_radius"] == 1.1 for point in builder.built), (
        "the shared scaling was re-searched for the second model instead of being reused"
    )
    assert second.n_candidates == 1, "a pinned shared knob must not be searched again"


@pytest.mark.unit
def test_sharing_a_knob_the_rule_does_not_cover_is_refused(spec):
    grid = _tiny_grid(0.3)
    with pytest.raises(ValueError, match="shares only"):
        _select(_FakeTensors(), grid, spec, _StubBuilder({}),
                shared={"reservoir_leak_alpha": 0.3})


@pytest.mark.unit
def test_a_pinned_shared_value_is_still_checked_against_the_grid(spec):
    """Sharing cannot be used to smuggle in a value the grid does not declare."""
    grid = _tiny_grid(0.3)
    with pytest.raises(OutOfGridError, match="spectral_scaling"):
        _select(_FakeTensors(), grid, spec, _StubBuilder({}),
                shared={"spectral_scaling": 0.7})


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_runner_refuses_a_window_length_outside_the_grid(temporary_dataset, tmp_path):
    dataset_id, _ = temporary_dataset
    with pytest.raises(OutOfGridError, match="window_length"):
        run_benchmark(
            BenchmarkConfig(
                dataset_id=dataset_id, models=("svm_rbf",), tasks=("classification",),
                seeds=(0,), window_lengths=(7,), n_splits=3, max_folds=1,
            ),
            raw_dir=tmp_path / "raw", tables_dir=tmp_path / "tables",
        )


@pytest.mark.unit
def test_the_runner_refuses_a_model_parameter_outside_the_grid(temporary_dataset, tmp_path):
    dataset_id, _ = temporary_dataset
    with pytest.raises(OutOfGridError, match="reservoir_leak_alpha"):
        run_benchmark(
            BenchmarkConfig(
                dataset_id=dataset_id, models=("esn",), tasks=("classification",),
                seeds=(0,), window_lengths=(WINDOW_LENGTH,), n_splits=3, max_folds=1,
                model_params={"esn": {"leak": 0.25}},
            ),
            raw_dir=tmp_path / "raw", tables_dir=tmp_path / "tables",
        )


@pytest.mark.unit
def test_the_run_configuration_fingerprints_the_grid():
    """A run that selected has to hash differently from one that did not."""
    plain = BenchmarkConfig(dataset_id="d2_beef_uncontrolled", seeds=(0,))
    selecting = replace(plain, selection_grid=_tiny_grid(0.3))
    assert plain.as_dict()["selection_grid"] is None
    assert selecting.as_dict()["selection_grid"] == {
        "readout_regularisation": [1.0e-3, 1.0e-2],
        "reservoir_leak_alpha": [0.3],
    }


@pytest.mark.integration
def test_the_selector_picks_a_real_model_from_the_declared_grid(
    fixture_dataset, tmp_path, spec
):
    """With the real ESN and the real knobs, not a stub."""
    from drososense.data.pipeline import build_fold_tensors, usable_specimens
    from drososense.data.splits import make_folds

    specimens = usable_specimens(fixture_dataset, WINDOW_LENGTH)
    fold = make_folds(specimens, "group_kfold", seed=0, n_splits=3)[0]
    tensors = build_fold_tensors(fixture_dataset, fold, WINDOW_LENGTH, tmp_path / "artifacts")

    grid = HyperparameterGrid(
        candidates={
            "readout_regularisation": (1.0e-3, 1.0e-1),
            "reservoir_leak_alpha": (0.3, 0.9),
            "spectral_scaling": (0.9,),
        },
        window_length_candidates=(WINDOW_LENGTH,),
    )
    result = select_hyperparameters(
        model_id="esn",
        task="classification",
        seed=0,
        fold_tensors=tensors,
        n_channels=int(tensors.train.X.shape[-1]),
        n_classes=fixture_dataset.schema.n_classes,
        grid=grid,
        spec=spec,
    )

    assert result.n_candidates == 4
    for name, value in result.grid_point.items():
        assert value in grid.candidates_for(name)
    assert result.params["leak"] == result.grid_point["reservoir_leak_alpha"]
    assert result.params["ridge_lambda"] == result.grid_point["readout_regularisation"]
    assert result.shared == {"spectral_scaling": 0.9}


@pytest.mark.integration
def test_a_selecting_run_records_the_choice_on_every_record(temporary_dataset, tmp_path):
    """§17 asks for the chosen value to be recorded, so that it is checkable."""
    from drososense.evaluation.results import load_records

    dataset_id, _ = temporary_dataset
    raw_dir = tmp_path / "raw"
    summary = run_benchmark(
        BenchmarkConfig(
            dataset_id=dataset_id,
            experiment="e2e_selection",
            models=("esn",),
            tasks=("classification", "regression"),
            seeds=(0,),
            window_lengths=(WINDOW_LENGTH,),
            n_splits=3,
            max_folds=1,
            model_params={"esn": {"reservoir_size": 40, "density": 0.1, "washout": 1}},
            selection_grid=_tiny_grid(0.3, 0.9),
        ),
        raw_dir=raw_dir,
        tables_dir=tmp_path / "tables",
    )
    assert not summary.empty

    records = load_records(raw_dir)
    assert records and all(record.selection for record in records)
    for record in records:
        assert record.selection["selection_split"] == "validation"
        assert record.selection["scope"] == "per_fold"
        assert record.selection["grid_point"]["reservoir_leak_alpha"] in (0.3, 0.9)
        assert record.selection["metric"] in ("macro_f1", "mae")
        assert record.status == "ok"
