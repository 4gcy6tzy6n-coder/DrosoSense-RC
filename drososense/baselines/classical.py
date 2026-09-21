"""Classical baselines: SVM-RBF, Random Forest, XGBoost, PCA+SVM.

These operate on the flattened window tensor. Because the frozen standardizer
has already centred and scaled every channel, none of these pipelines applies a
second scaler — a redundant per-model scaler fitted inside a fold would be a
quiet way for models to diverge in preprocessing, which the protocol forbids.

Probability estimates
---------------------
``SVC(probability=True)`` fits an internal Platt calibration with its own
cross-validation split, which is slow and introduces a nested split inside the
test fold. Instead, where a classifier exposes only ``decision_function``, the
one-vs-rest decision values are mapped through a softmax. That transform is
strictly monotone per class, so AUROC — a ranking metric — is unaffected, while
the values are NOT calibrated probabilities. This is stated rather than implied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drososense.baselines.base import BaseModel, ModelUnavailableError, TaskType


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax with an overflow guard.

    Args:
        scores: Array of shape ``(n_samples, n_classes)``.

    Returns:
        Array of the same shape whose rows sum to one.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


class ClassicalModel(BaseModel):
    """Base class for estimators that consume a flat design matrix."""

    def __init__(self, task: TaskType, seed: int, params: dict[str, Any] | None = None) -> None:
        """Initialise and build the underlying estimator.

        Args:
            task: ``classification`` or ``regression``.
            seed: Seed for every stochastic component.
            params: Hyperparameters forwarded to the estimator.
        """
        super().__init__(task, seed, params)
        self._estimator = self._build_estimator()
        self._proba_from_decision = False

    def _build_estimator(self) -> Any:
        """Construct the underlying estimator.

        Returns:
            An unfitted scikit-learn (or XGBoost) estimator.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying estimator.

        Args:
            x: Design matrix.
            y: Targets.
        """
        self._estimator.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the underlying estimator.

        Args:
            x: Design matrix.

        Returns:
            Predictions.
        """
        return self._estimator.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Return class scores for AUROC.

        Args:
            x: Window tensor of shape ``(n_samples, L, C)``.

        Returns:
            ``(n_samples, n_classes)`` scores, or ``None`` for regressors.
        """
        if self.task != "classification":
            return None
        flat = self.flatten(x)
        if hasattr(self._estimator, "predict_proba"):
            proba = self._estimator.predict_proba(flat)
            self._proba_from_decision = False
            return np.asarray(proba)
        if hasattr(self._estimator, "decision_function"):
            scores = np.asarray(self._estimator.decision_function(flat))
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            self._proba_from_decision = True
            return _softmax(scores)
        return None

    def describe(self) -> dict[str, Any]:
        """Extend the base description with fitted-estimator facts.

        Returns:
            Mapping including whether AUROC inputs are calibrated.
        """
        info = super().describe()
        info["proba_source"] = (
            "softmax(decision_function)" if self._proba_from_decision else "predict_proba"
        )
        if hasattr(self, "_estimator"):
            params = self._estimator.get_params() if hasattr(self._estimator, "get_params") else {}
            info["resolved_params"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in params.items()
            }
        return info


class SVMRBF(ClassicalModel):
    """Support Vector Machine with an RBF kernel."""

    model_id = "svm_rbf"

    def _build_estimator(self) -> Any:
        """Build ``SVC`` or ``SVR`` with an RBF kernel.

        Returns:
            The estimator.
        """
        from sklearn.svm import SVC, SVR

        c = float(self.params.get("C", 1.0))
        gamma = self.params.get("gamma", "scale")
        if self.task == "classification":
            return SVC(C=c, gamma=gamma, kernel="rbf", random_state=self.seed)
        return SVR(C=c, gamma=gamma, kernel="rbf")


class RandomForest(ClassicalModel):
    """Random Forest classifier or regressor."""

    model_id = "random_forest"

    def _build_estimator(self) -> Any:
        """Build ``RandomForestClassifier`` or ``RandomForestRegressor``.

        Returns:
            The estimator.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        common = {
            "n_estimators": int(self.params.get("n_estimators", 300)),
            "max_depth": self.params.get("max_depth"),
            "min_samples_leaf": int(self.params.get("min_samples_leaf", 1)),
            "n_jobs": int(self.params.get("n_jobs", -1)),
            "random_state": self.seed,
        }
        if self.task == "classification":
            return RandomForestClassifier(**common)
        return RandomForestRegressor(**common)

    def n_trainable_parameters(self) -> int | None:
        """Count fitted leaf values across all trees.

        Returns:
            Number of leaf outputs, or ``None`` before fitting.
        """
        if not self._fitted:
            return None
        # A leaf is a node with no left child; counting that way is stable
        # across scikit-learn versions, unlike the internal node-pointer arrays.
        return int(
            sum(
                int((estimator.tree_.children_left == -1).sum())
                for estimator in self._estimator.estimators_
            )
        )


class XGBoost(ClassicalModel):
    """Gradient-boosted trees via XGBoost."""

    model_id = "xgboost"

    def _build_estimator(self) -> Any:
        """Build ``XGBClassifier`` or ``XGBRegressor``.

        Returns:
            The estimator.

        Raises:
            ModelUnavailableError: If xgboost cannot be imported or loaded.
        """
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except Exception as exc:  # pragma: no cover - depends on host libomp
            raise ModelUnavailableError(
                f"xgboost is not usable in this environment: {exc}. On macOS this usually "
                f"means the OpenMP runtime is missing — install it with `brew install libomp`."
            ) from exc

        common = {
            "n_estimators": int(self.params.get("n_estimators", 300)),
            "max_depth": int(self.params.get("max_depth", 6)),
            "learning_rate": float(self.params.get("learning_rate", 0.1)),
            "subsample": float(self.params.get("subsample", 0.9)),
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.9)),
            "random_state": self.seed,
            "n_jobs": int(self.params.get("n_jobs", -1)),
            "tree_method": self.params.get("tree_method", "hist"),
        }
        if self.task == "classification":
            return XGBClassifier(eval_metric="mlogloss", **common)
        return XGBRegressor(**common)

    def n_trainable_parameters(self) -> int | None:
        """Count fitted leaf values across all boosting rounds.

        Each leaf carries one learned output value, so the leaf count is the
        number of parameters the booster actually fitted.

        Returns:
            Number of leaf outputs, or ``None`` before fitting.
        """
        if not self._fitted:
            return None
        import json

        try:
            n_leaves = 0
            for tree_json in self._estimator.get_booster().get_dump(dump_format="json"):
                stack = [json.loads(tree_json)]
                while stack:
                    node = stack.pop()
                    children = node.get("children", [])
                    if children:
                        stack.extend(children)
                    else:
                        n_leaves += 1
            return int(n_leaves)
        except Exception:
            # A backend change that breaks the dump format must not fail the run;
            # an unreported parameter count is better than a lost result.
            return None


class PCASVM(ClassicalModel):
    """PCA dimensionality reduction followed by an RBF SVM."""

    model_id = "pca_svm"

    def _build_estimator(self) -> Any:
        """Build a PCA + SVM pipeline.

        Returns:
            The pipeline.
        """
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC, SVR

        n_components = self.params.get("n_components", 0.95)
        steps: list[tuple[str, Any]] = [
            ("pca", PCA(n_components=n_components, random_state=self.seed))
        ]
        c = float(self.params.get("C", 1.0))
        if self.task == "classification":
            steps.append(("svm", SVC(C=c, kernel="rbf", gamma=self.params.get("gamma", "scale"))))
        else:
            steps.append(("svm", SVR(C=c, kernel="rbf", gamma=self.params.get("gamma", "scale"))))
        return Pipeline(steps)


CLASSICAL_MODELS: dict[str, type[ClassicalModel]] = {
    SVMRBF.model_id: SVMRBF,
    RandomForest.model_id: RandomForest,
    XGBoost.model_id: XGBoost,
    PCASVM.model_id: PCASVM,
}
