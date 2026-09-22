"""Standard leaky echo-state network — the R4 control.

Dynamics follow the frozen M3 formulation::

    h_t = (1 - alpha) h_{t-1} + alpha * tanh(g * A_hat h_{t-1} + W_in x_t + b)

Only the linear readout is trained; ``A_hat``, ``W_in`` and ``b`` are drawn once
from the run seed and then frozen. That frozen-reservoir / trained-readout split
is the whole point of the comparison later: the same readout machinery will be
attached to the biological connectome in M3, so any performance difference in
M4 is attributable to the reservoir matrix rather than to the readout.

Here ``A_hat`` is a *random sparse* matrix with a prescribed density and
spectral radius. It is not a connectome, and nothing in this module supports a
claim about biological topology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from drososense.baselines.base import BaseModel, TaskType
from drososense.utils.seeding import make_rng

DEFAULT_PARAMS: dict[str, Any] = {
    "reservoir_size": 200,
    "density": 0.05,
    "spectral_radius": 0.9,
    "leak": 0.3,
    "gain": 1.0,
    "input_scale": 0.5,
    "ridge_lambda": 1.0e-3,
    "washout": 10,
    "state_pooling": "last",  # last | mean
    "readout": "ridge",  # ridge | logistic
}


class EchoStateNetwork(BaseModel):
    """A frozen random sparse reservoir with a trained linear readout."""

    model_id = "esn"

    def __init__(
        self,
        task: TaskType,
        seed: int,
        params: dict[str, Any] | None = None,
        n_channels: int | None = None,
    ) -> None:
        """Initialise the ESN.

        Args:
            task: ``classification`` or ``regression``.
            seed: Seed for the reservoir and input-matrix draws.
            params: Hyperparameters, merged over :data:`DEFAULT_PARAMS`.
            n_channels: Input channel count; required to draw ``W_in``.
        """
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(task, seed, merged)
        self.n_channels = n_channels
        self._A: sparse.csr_matrix | None = None
        self._w_in: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._n_outputs: int | None = None
        self._readout_kind: str = str(merged["readout"])

    # -- reservoir construction -------------------------------------------
    def _build_reservoir(self) -> None:
        """Draw the frozen reservoir, input matrix and bias.

        The spectral radius is imposed by rescaling the largest-magnitude
        eigenvalue of the sparse draw, which is the standard ESN construction
        and the same procedure every control reservoir in M4 must use so that
        the comparison is like-for-like.
        """
        size = int(self.params["reservoir_size"])
        density = float(self.params["density"])
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density}")
        if self.n_channels is None:
            raise ValueError("n_channels is required to build the ESN input matrix")

        rng = make_rng(self.seed)

        # Sparse random reservoir with zero diagonal, as is standard.
        n_edges = max(1, int(density * size * size))
        rows = rng.integers(0, size, size=n_edges)
        cols = rng.integers(0, size, size=n_edges)
        values = rng.uniform(-1.0, 1.0, size=n_edges)
        matrix = sparse.csr_matrix((values, (rows, cols)), shape=(size, size))
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()

        target_radius = float(self.params["spectral_radius"])
        if matrix.nnz > 0:
            # eigsh needs k < n; fall back to a dense estimate for small reservoirs.
            if size > 3:
                # ARPACK starts from a random vector by default, which would make
                # the spectral-radius estimate — and therefore the whole reservoir —
                # differ between two runs of the same seed. Pin the start vector.
                start = make_rng(self.seed + 1).standard_normal(size)
                largest = float(
                    np.max(
                        np.abs(
                            sparse_linalg.eigsh(
                                matrix, k=1, return_eigenvectors=False, v0=start
                            )
                        )
                    )
                )
            else:
                largest = float(np.max(np.abs(np.linalg.eigvals(matrix.toarray()))))
            if largest > 0:
                matrix = matrix * (target_radius / largest)

        self._A = matrix.tocsr()
        scale = float(self.params["input_scale"])
        self._w_in = rng.uniform(-scale, scale, size=(size, self.n_channels))
        self._bias = rng.uniform(-scale, scale, size=size)

    def _run_reservoir(self, x: np.ndarray) -> np.ndarray:
        """Drive the reservoir and summarise each window's trajectory.

        The first ``washout`` timesteps are transient and excluded from the
        summary. The remaining states are reduced by ``state_pooling``: ``last``
        keeps the final state (the standard readout for windowed classification)
        and ``mean`` averages the post-washout trajectory, which is less
        sensitive to where in the window the spoilage signature happens to peak.

        Args:
            x: Window tensor of shape ``(n_samples, length, n_channels)``.

        Returns:
            State matrix of shape ``(n_samples, reservoir_size)``.
        """
        if self._A is None:
            raise RuntimeError("reservoir not built; call fit() first")

        leak = float(self.params["leak"])
        gain = float(self.params["gain"])
        pooling = str(self.params["state_pooling"])
        size = int(self.params["reservoir_size"])
        washout = min(int(self.params["washout"]), max(x.shape[1] - 1, 0))

        n_samples, length, _ = x.shape
        states = np.zeros((n_samples, size), dtype=np.float64)
        # Pre-transposed input so the inner loop does dot products only.
        input_projection = x @ self._w_in.T  # (n_samples, length, size)

        for i in range(n_samples):
            state = np.zeros(size, dtype=np.float64)
            accumulated = np.zeros(size, dtype=np.float64)
            collected = 0
            for t in range(length):
                drive = gain * (self._A @ state) + input_projection[i, t] + self._bias
                state = (1.0 - leak) * state + leak * np.tanh(drive)
                if t >= washout:
                    accumulated += state
                    collected += 1
            if collected and pooling == "mean":
                states[i] = accumulated / collected
            else:
                states[i] = state
        return states

    # -- readout -----------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Freeze the reservoir, then train the readout.

        Args:
            x: Flattened window matrix, reshaped back to ``(n, L, C)`` here.
            y: Targets.

        Raises:
            ValueError: If ``n_channels`` is unset or the shape is inconsistent.
        """
        if self.n_channels is None:
            raise ValueError("n_channels is required to build the ESN")
        total = x.shape[1]
        if total % self.n_channels != 0:
            raise ValueError(
                f"flattened width {total} is not a multiple of n_channels {self.n_channels}"
            )
        length = total // self.n_channels
        tensor = x.reshape(x.shape[0], length, self.n_channels)

        if self._A is None:
            self._build_reservoir()

        states = self._run_reservoir(tensor)
        augmented = np.column_stack([states, np.ones(states.shape[0])])

        if self.task == "classification":
            classes = np.unique(y)
            if not np.array_equal(classes, np.arange(len(classes))):
                raise ValueError(
                    f"esn: classification labels must be contiguous from 0, got {classes.tolist()}"
                )
            self._n_outputs = int(len(classes))
            targets = np.zeros((y.shape[0], self._n_outputs), dtype=np.float64)
            targets[np.arange(y.shape[0]), y.astype(int)] = 1.0
        else:
            self._n_outputs = 1
            targets = y.reshape(-1, 1).astype(np.float64)

        if self._readout_kind == "logistic":
            self._w_out = self._fit_logistic_readout(augmented, y)
        else:
            self._w_out = self._ridge_solve(augmented, targets)

    def _ridge_solve(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Solve the ridge readout in its numerically stable form.

        Solves ``(H^T H + lambda I) W = H^T Y`` rather than forming the pseudo-
        inverse explicitly, which keeps the conditioning well-behaved for the
        wide state matrices produced by a large reservoir.

        Args:
            features: Design matrix ``(n_samples, n_features)``.
            targets: Target matrix ``(n_samples, n_outputs)``.

        Returns:
            Readout weights of shape ``(n_features, n_outputs)``.
        """
        ridge_lambda = float(self.params["ridge_lambda"])
        gram = features.T @ features
        regulariser = ridge_lambda * np.eye(gram.shape[0])
        return np.linalg.solve(gram + regulariser, features.T @ targets)

    def _fit_logistic_readout(self, features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit a multinomial logistic readout.

        Args:
            features: Design matrix.
            y: Integer class labels.

        Returns:
            Weight matrix of shape ``(n_features, n_classes)``.

        Raises:
            ValueError: If used for regression.
        """
        if self.task != "classification":
            raise ValueError("the logistic readout is only available for classification")
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=1.0 / max(float(self.params["ridge_lambda"]), 1e-12),
            max_iter=2000,
            random_state=self.seed,
        )
        model.fit(features, y.astype(int))
        coefficients = model.coef_
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        return np.vstack([coefficients.T, model.intercept_.reshape(1, -1)])

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the trained readout to new windows.

        Args:
            x: Flattened window matrix.

        Returns:
            Predictions of shape ``(n_samples,)``.
        """
        if self._w_out is None:
            raise RuntimeError("readout not fitted")
        length = x.shape[1] // self.n_channels
        tensor = x.reshape(x.shape[0], length, self.n_channels)
        states = self._run_reservoir(tensor)
        augmented = np.column_stack([states, np.ones(states.shape[0])])
        scores = augmented @ self._w_out
        if self.task == "classification":
            return scores.argmax(axis=1)
        return scores[:, 0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Return softmax scores over the readout outputs.

        Note:
            For the ridge readout these are softmax-transformed regression
            outputs, not calibrated probabilities. The transform is monotone, so
            AUROC — a ranking metric — is unaffected; the values must not be read
            as confidence.

        Args:
            x: Window tensor of shape ``(n_samples, L, C)``.

        Returns:
            ``(n_samples, n_classes)`` scores, or ``None`` for regressors.
        """
        if self.task != "classification":
            return None
        flat = self.flatten(x)
        length = flat.shape[1] // self.n_channels
        states = self._run_reservoir(flat.reshape(flat.shape[0], length, self.n_channels))
        augmented = np.column_stack([states, np.ones(states.shape[0])])
        scores = augmented @ self._w_out
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    # -- reporting ---------------------------------------------------------
    def n_trainable_parameters(self) -> int | None:
        """Count only the trained readout weights.

        This is the number TAFE cares about: the reservoir is frozen, so its
        nodes cost nothing to train.

        Returns:
            ``n_outputs * (reservoir_size + 1)``, or ``None`` before fitting.
        """
        if self._w_out is None or self._n_outputs is None:
            return None
        return int(self._w_out.size)

    def n_frozen_parameters(self) -> int:
        """Count the frozen reservoir, input and bias entries.

        Returns:
            Total count of values drawn once and never trained.
        """
        if self._A is None or self._w_in is None or self._bias is None:
            return 0
        return int(self._A.nnz + self._w_in.size + self._bias.size)

    def reservoir_sparsity(self) -> float:
        """Fraction of absent entries in the reservoir matrix.

        Returns:
            ``1 - nnz / N^2``.
        """
        if self._A is None:
            return 0.0
        size = int(self.params["reservoir_size"])
        return float(1.0 - self._A.nnz / (size * size))

    def describe(self) -> dict[str, Any]:
        """Extend the base description with the frozen/trainable split.

        Returns:
            Mapping including frozen parameters and reservoir sparsity.
        """
        info = super().describe()
        info["frozen_parameters"] = self.n_frozen_parameters()
        info["reservoir_sparsity"] = self.reservoir_sparsity()
        info["readout"] = self._readout_kind
        return info
