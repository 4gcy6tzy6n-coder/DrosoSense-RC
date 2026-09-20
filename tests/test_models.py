"""The model zoo conforms to one interface.

The point of these tests is not that each baseline is accurate — it is that each
baseline is *interchangeable*. If two models disagree about input shape, output
shape, or how probabilities are produced, the comparison in E1 would be
measuring plumbing rather than method.
"""

from __future__ import annotations

import numpy as np
import pytest

from drososense.baselines.base import ModelUnavailableError
from drososense.baselines.registry import MODEL_IDS, build_model, model_availability, model_spec

# Kept small: these tests check contract conformance, not accuracy. The neural
# models get a handful of epochs because the contract does not depend on
# convergence.
FAST_PARAMS: dict[str, dict] = {
    "svm_rbf": {"C": 1.0},
    "random_forest": {"n_estimators": 20},
    "xgboost": {"n_estimators": 20, "max_depth": 3},
    "pca_svm": {"n_components": 4},
    "gru": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "lstm": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "cnn1d": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "tcn": {"hidden_size": 8, "epochs": 2, "batch_size": 32},
    "esn": {"reservoir_size": 40, "density": 0.1, "washout": 1},
}

N_CHANNELS = 3
WINDOW_LENGTH = 8


def _toy_windows(n_samples: int = 60, seed: int = 0):
    """Build a small, learnable window tensor.

    Args:
        n_samples: Number of windows.
        seed: RNG seed.

    Returns:
        ``(X, y_class, y_reg)``.
    """
    rng = np.random.default_rng(seed)
    y_class = rng.integers(0, 4, size=n_samples)
    x = rng.normal(size=(n_samples, WINDOW_LENGTH, N_CHANNELS))
    # Make the label recoverable so a model has something to learn.
    x[:, -1, 0] += y_class * 2.0
    y_reg = y_class.astype(float) + rng.normal(scale=0.1, size=n_samples)
    return x, y_class, y_reg


def _build_or_skip(model_id: str, task: str, seed: int = 0):
    """Build a model, skipping the test when its backend is unavailable.

    Args:
        model_id: Registered model id.
        task: Task name.
        seed: Seed.

    Returns:
        The constructed model.
    """
    availability = model_availability()
    if not availability[model_id]["available"]:
        pytest.skip(f"{model_id} unavailable: {availability[model_id]['reason']}")
    return build_model(model_id, task, seed, FAST_PARAMS[model_id], n_channels=N_CHANNELS)


@pytest.mark.unit
def test_registry_covers_every_declared_model():
    """The registry exposes exactly the nine protocol baselines."""
    assert set(MODEL_IDS) == {
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


@pytest.mark.unit
def test_registry_rejects_an_unknown_model():
    """An unknown id fails loudly."""
    with pytest.raises(KeyError, match="unknown model"):
        model_spec("transformer")


@pytest.mark.unit
def test_availability_report_covers_every_model():
    """Availability is reported for all models, with a reason when unavailable."""
    report = model_availability()
    assert set(report) == set(MODEL_IDS)
    for model_id, entry in report.items():
        if not entry["available"]:
            assert entry["reason"], f"{model_id} is unavailable without a stated reason"


@pytest.mark.unit
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_classification_contract(model_id):
    """Every classifier fits, predicts in-range labels and yields valid probabilities."""
    model = _build_or_skip(model_id, "classification")
    x, y, _ = _toy_windows()

    model.fit(x, y)
    predictions = model.predict(x)
    assert predictions.shape == (len(x),)
    assert set(np.unique(predictions)).issubset({0, 1, 2, 3})

    scores = model.predict_proba(x)
    assert scores is not None, f"{model_id} must expose scores for AUROC"
    assert scores.shape == (len(x), 4)
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_regression_contract(model_id):
    """Every regressor fits and returns finite predictions of the right shape."""
    model = _build_or_skip(model_id, "regression")
    x, _, y = _toy_windows()

    model.fit(x, y)
    predictions = model.predict(x)
    assert predictions.shape == (len(x),)
    assert np.all(np.isfinite(predictions))
    assert model.predict_proba(x) is None


@pytest.mark.unit
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_describe_is_json_serialisable(model_id):
    """A model can describe itself well enough to be written into a run record."""
    import json

    model = _build_or_skip(model_id, "classification")
    x, y, _ = _toy_windows()
    model.fit(x, y)
    payload = json.dumps(model.describe(), default=str)
    assert model_id in payload


@pytest.mark.unit
def test_predict_before_fit_is_an_error():
    """Calling predict on an unfitted model fails explicitly."""
    model = build_model("svm_rbf", "classification", 0, {}, n_channels=N_CHANNELS)
    x, _, _ = _toy_windows(4)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(x)


@pytest.mark.unit
def test_fit_rejects_a_shape_mismatch():
    """X and y must agree on the number of samples."""
    model = build_model("svm_rbf", "classification", 0, {}, n_channels=N_CHANNELS)
    x, y, _ = _toy_windows(10)
    with pytest.raises(ValueError, match="but y has"):
        model.fit(x, y[:5])


@pytest.mark.unit
def test_fit_rejects_a_non_tensor_input():
    """A 2-D input is rejected so a caller cannot silently skip windowing."""
    model = build_model("svm_rbf", "classification", 0, {}, n_channels=N_CHANNELS)
    with pytest.raises(ValueError, match="3-D"):
        model.fit(np.zeros((10, 5)), np.zeros(10, dtype=int))


@pytest.mark.unit
def test_task_is_validated_against_model_capability():
    """An unsupported task is refused at construction."""
    model = build_model("esn", "classification", 0, FAST_PARAMS["esn"], n_channels=N_CHANNELS)
    assert model.task == "classification"
    with pytest.raises(ValueError, match="task must be"):
        build_model("esn", "clustering", 0, {}, n_channels=N_CHANNELS)


@pytest.mark.unit
def test_sequence_models_require_n_channels():
    """A sequence model cannot silently guess its channel count."""
    model = build_model("gru", "classification", 0, FAST_PARAMS["gru"], n_channels=None)
    x, y, _ = _toy_windows(8)
    with pytest.raises(ValueError, match="n_channels"):
        model.fit(x, y)


@pytest.mark.unit
def test_esn_reports_a_frozen_trainable_split():
    """The ESN distinguishes frozen reservoir parameters from trained readout ones."""
    model = _build_or_skip("esn", "classification")
    x, y, _ = _toy_windows()
    model.fit(x, y)

    trainable = model.n_trainable_parameters()
    frozen = model.n_frozen_parameters()
    assert trainable is not None and frozen > 0
    assert trainable < frozen, "a frozen reservoir must train far fewer parameters than it freezes"
    assert 0.0 < model.reservoir_sparsity() < 1.0


@pytest.mark.unit
def test_esn_is_deterministic_for_a_seed():
    """Two ESNs with the same seed produce identical predictions."""
    x, y, _ = _toy_windows()
    first = build_model("esn", "classification", 3, FAST_PARAMS["esn"], n_channels=N_CHANNELS)
    second = build_model("esn", "classification", 3, FAST_PARAMS["esn"], n_channels=N_CHANNELS)
    first.fit(x, y)
    second.fit(x, y)
    np.testing.assert_allclose(first.predict(x), second.predict(x))


@pytest.mark.unit
def test_esn_rejects_non_contiguous_labels():
    """A fold whose training split omits a class fails loudly rather than remapping."""
    model = build_model("esn", "classification", 0, FAST_PARAMS["esn"], n_channels=N_CHANNELS)
    x, _, _ = _toy_windows(21)
    labels = np.array([0, 1, 3] * 7)  # class 2 is absent, so 3 is not arange(3)
    with pytest.raises(ValueError, match="contiguous from 0"):
        model.fit(x, labels)


@pytest.mark.unit
def test_unavailable_backend_is_reported_not_silently_skipped(monkeypatch):
    """A missing backend surfaces as a clear error at build time."""
    import drososense.baselines.registry as registry

    monkeypatch.setitem(
        registry.MODEL_SPECS,
        "fake_model",
        registry.ModelSpec("fake_model", "sequence", "definitely_not_a_real_module"),
    )
    monkeypatch.setattr(registry, "MODEL_IDS", (*registry.MODEL_IDS, "fake_model"))
    report = registry.model_availability()
    assert report["fake_model"]["available"] is False
    assert "not installed" in report["fake_model"]["reason"]
    del registry.MODEL_SPECS["fake_model"]
