"""Sequence baselines: GRU, LSTM, 1D-CNN and TCN.

These consume the window tensor directly as ``(batch, length, channels)``, which
is exactly how the connectome reservoir will consume it in M3 — so the input
plumbing is shared from the start rather than retrofitted.

Reproducibility
---------------
Torch CPU results depend on the number of threads, because reduction order
changes. The runner therefore pins ``torch.set_num_threads`` to a configurable
value (default 1) and seeds torch from the run seed. This trades wall-clock for
the ability to re-run a seed and get the same number back, which the frozen
protocol requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drososense.baselines.base import BaseModel, ModelUnavailableError, TaskType
from drososense.utils.seeding import seed_everything

try:  # pragma: no cover - exercised by presence/absence of torch
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

DEFAULT_PARAMS: dict[str, Any] = {
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "torch_num_threads": 1,
}


def _require_torch() -> None:
    """Raise if torch is not importable.

    Raises:
        ModelUnavailableError: If torch is missing, with the install hint.
    """
    if torch is None:
        raise ModelUnavailableError(
            "torch is required for the sequence baselines but is not importable. "
            "Install it with `python -m pip install torch` (CPU wheel is sufficient)."
        )


if nn is not None:

    class _RecurrentNet(nn.Module):
        """Shared GRU/LSTM head-and-classifier module."""

        def __init__(
            self,
            cell: str,
            n_channels: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            n_outputs: int,
        ) -> None:
            """Build the recurrent network.

            Args:
                cell: ``gru`` or ``lstm``.
                n_channels: Number of input channels.
                hidden_size: Recurrent hidden width.
                num_layers: Number of stacked recurrent layers.
                dropout: Dropout probability between layers.
                n_outputs: 1 for regression, ``n_classes`` for classification.
            """
            super().__init__()
            factory = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = factory(
                input_size=n_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, n_outputs)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the network.

            Args:
                x: Tensor of shape ``(batch, length, channels)``.

            Returns:
                Output of shape ``(batch, n_outputs)``.
            """
            output, _ = self.rnn(x)
            return self.head(output[:, -1, :])

    class _Conv1DNet(nn.Module):
        """Stacked causal 1-D convolutions with global average pooling."""

        def __init__(
            self,
            n_channels: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            n_outputs: int,
        ) -> None:
            """Build the convolutional network.

            Args:
                n_channels: Number of input channels.
                hidden_size: Channel width of the convolution stack.
                num_layers: Number of convolution blocks.
                dropout: Dropout probability.
                n_outputs: 1 for regression, ``n_classes`` for classification.
            """
            super().__init__()
            blocks: list[nn.Module] = []
            in_channels = n_channels
            for _ in range(max(1, num_layers)):
                blocks.extend(
                    [
                        nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_channels = hidden_size
            self.features = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden_size, n_outputs)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the network.

            Args:
                x: Tensor of shape ``(batch, length, channels)``.

            Returns:
                Output of shape ``(batch, n_outputs)``.
            """
            x = x.transpose(1, 2)  # (batch, channels, length)
            features = self.features(x)
            pooled = features.mean(dim=2)
            return self.head(pooled)

    class _TemporalBlock(nn.Module):
        """Dilated causal convolution block with a residual connection."""

        def __init__(
            self,
            n_channels: int,
            hidden_size: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            """Build the block.

            Args:
                n_channels: Input channel count.
                hidden_size: Output channel count.
                kernel_size: Convolution kernel width.
                dilation: Dilation factor controlling the receptive field.
                dropout: Dropout probability.
            """
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            # Pad on both sides during the convolution, then crop the trailing
            # padding so the output length equals the input length and no
            # timestep can see the future.
            self.conv = nn.Conv1d(
                n_channels,
                hidden_size,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.downsample = (
                nn.Conv1d(n_channels, hidden_size, 1) if n_channels != hidden_size else None
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the block with a causal crop.

            Args:
                x: Tensor of shape ``(batch, channels, length)``.

            Returns:
                Tensor of shape ``(batch, hidden_size, length)``.
            """
            residual = x if self.downsample is None else self.downsample(x)
            out = self.conv(x)
            if self.padding > 0:
                out = out[:, :, : -self.padding]  # crop the future
            return self.relu(out + residual)

    class _TCNNet(nn.Module):
        """A small dilated causal temporal convolutional network."""

        def __init__(
            self,
            n_channels: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            n_outputs: int,
            kernel_size: int = 3,
        ) -> None:
            """Build the TCN.

            Args:
                n_channels: Number of input channels.
                hidden_size: Channel width of the dilated stack.
                num_layers: Number of temporal blocks.
                dropout: Dropout probability.
                n_outputs: 1 for regression, ``n_classes`` for classification.
                kernel_size: Convolution kernel width.
            """
            super().__init__()
            blocks: list[nn.Module] = []
            in_channels = n_channels
            for layer in range(max(1, num_layers)):
                blocks.append(
                    _TemporalBlock(
                        in_channels,
                        hidden_size,
                        kernel_size=kernel_size,
                        dilation=2**layer,
                        dropout=dropout,
                    )
                )
                in_channels = hidden_size
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden_size, n_outputs)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the network.

            Args:
                x: Tensor of shape ``(batch, length, channels)``.

            Returns:
                Output of shape ``(batch, n_outputs)``.
            """
            features = self.tcn(x.transpose(1, 2))
            return self.head(features[:, :, -1])

else:  # pragma: no cover - torch absent
    _RecurrentNet = _Conv1DNet = _TemporalBlock = _TCNNet = None  # type: ignore[assignment]


class TorchSequenceModel(BaseModel):
    """Base class for the torch sequence baselines."""

    def __init__(
        self,
        task: TaskType,
        seed: int,
        params: dict[str, Any] | None = None,
        n_channels: int | None = None,
    ) -> None:
        """Initialise the model.

        Args:
            task: ``classification`` or ``regression``.
            seed: Seed for every stochastic component.
            params: Hyperparameters, merged over :data:`DEFAULT_PARAMS`.
            n_channels: Input channel count, needed to build the network.
        """
        _require_torch()
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(task, seed, merged)
        self.n_channels = n_channels
        self._network: Any = None
        self._n_outputs: int | None = None

    def _build_network(self, n_channels: int, n_outputs: int) -> Any:
        """Construct the network for this model.

        Args:
            n_channels: Number of input channels.
            n_outputs: Output width.

        Returns:
            An ``nn.Module``.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    def _ensure_network(self, n_channels: int, y: np.ndarray) -> None:
        """Create the network and output width on first fit.

        Args:
            n_channels: Number of input channels.
            y: Training targets, used to size the classification head.
        """
        if self._network is not None:
            return
        self.n_channels = n_channels
        if self.task == "classification":
            classes = np.unique(np.asarray(y))
            if not np.array_equal(classes, np.arange(len(classes))):
                raise ValueError(
                    f"{self.model_id}: classification labels must be contiguous from 0, "
                    f"got {classes.tolist()}. A fold whose training split omits a class cannot "
                    f"train a cross-entropy head — report it, do not silently remap labels."
                )
            self._n_outputs = int(len(classes))
        else:
            self._n_outputs = 1
        self._network = self._build_network(n_channels, self._n_outputs)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the network with Adam and a task-appropriate loss.

        Args:
            x: Window tensor reshaped to ``(n, L, C)`` by the caller's flatten
                contract is undone here, so this receives the flat matrix.
            y: Targets.

        Raises:
            ValueError: If the flattened input cannot be reshaped.
        """
        # BaseModel.fit passes the flattened matrix; recover the tensor shape.
        window_length, n_channels = self._infer_windows(x)
        tensor = x.reshape(x.shape[0], window_length, n_channels)

        self._ensure_network(n_channels, y)

        torch.set_num_threads(int(self.params["torch_num_threads"]))
        seed_everything(self.seed)
        if torch.cuda.is_available():
            # cuDNN's algorithm choice is non-deterministic by default for
            # convolutions; the deterministic flag keeps the per-seed number
            # reproducible across re-runs, at the cost of slightly slower
            # convolution kernels. Sequence-model training is the rate
            # bottleneck on E1's 62-fold D3 split; the GPU path pays off
            # there even with the deterministic flag.
            torch.backends.cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = self._network.to(device)
        network.train()

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        if self.task == "classification":
            criterion: Any = nn.CrossEntropyLoss()
            targets = torch.as_tensor(y, dtype=torch.long, device=device)
        else:
            criterion = nn.MSELoss()
            targets = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        inputs = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        batch_size = int(self.params["batch_size"])
        generator = torch.Generator(device=device).manual_seed(self.seed)

        for _ in range(int(self.params["epochs"])):
            order = torch.randperm(inputs.shape[0], generator=generator)
            for start in range(0, inputs.shape[0], batch_size):
                index = order[start : start + batch_size]
                optimizer.zero_grad()
                output = network(inputs[index])
                loss = criterion(output, targets[index])
                loss.backward()
                optimizer.step()

    def _infer_windows(self, x: np.ndarray) -> tuple[int, int]:
        """Recover ``(window_length, n_channels)`` from a flattened matrix.

        Args:
            x: Flattened design matrix.

        Returns:
            Tuple of window length and channel count.

        Raises:
            ValueError: If ``n_channels`` was never supplied and cannot be
                inferred unambiguously.
        """
        if self.n_channels is None:
            raise ValueError(
                f"{self.model_id}: n_channels must be supplied at construction so that the "
                f"flattened window matrix can be reshaped back to (n, L, C)"
            )
        total = x.shape[1]
        if total % self.n_channels != 0:
            raise ValueError(
                f"{self.model_id}: flattened width {total} is not a multiple of n_channels "
                f"{self.n_channels}"
            )
        return total // self.n_channels, self.n_channels

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference in eval mode.

        Args:
            x: Flattened design matrix.

        Returns:
            Predictions of shape ``(n_samples,)``.
        """
        window_length, n_channels = self._infer_windows(x)
        device = next(self._network.parameters()).device
        tensor = torch.as_tensor(
            x.reshape(x.shape[0], window_length, n_channels), dtype=torch.float32, device=device
        )
        self._network.eval()
        with torch.no_grad():
            output = self._network(tensor)
        if self.task == "classification":
            return output.argmax(dim=1).cpu().numpy()
        return output.squeeze(1).cpu().numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Return softmax class probabilities.

        Args:
            x: Window tensor of shape ``(n_samples, L, C)``.

        Returns:
            ``(n_samples, n_classes)`` probabilities, or ``None`` for regressors.
        """
        if self.task != "classification":
            return None
        flat = self.flatten(x)
        window_length, n_channels = self._infer_windows(flat)
        device = next(self._network.parameters()).device
        tensor = torch.as_tensor(
            flat.reshape(flat.shape[0], window_length, n_channels), dtype=torch.float32, device=device
        )
        self._network.eval()
        with torch.no_grad():
            logits = self._network(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def n_trainable_parameters(self) -> int | None:
        """Count parameters with ``requires_grad``.

        Returns:
            Parameter count, or ``None`` before the network is built.
        """
        if self._network is None:
            return None
        return int(sum(p.numel() for p in self._network.parameters() if p.requires_grad))


class GRU(TorchSequenceModel):
    """Gated recurrent unit sequence model."""

    model_id = "gru"

    def _build_network(self, n_channels: int, n_outputs: int) -> Any:
        """Build the GRU.

        Args:
            n_channels: Number of input channels.
            n_outputs: Output width.

        Returns:
            The network.
        """
        return _RecurrentNet(
            "gru",
            n_channels,
            int(self.params["hidden_size"]),
            int(self.params["num_layers"]),
            float(self.params["dropout"]),
            n_outputs,
        )


class LSTM(TorchSequenceModel):
    """Long short-term memory sequence model."""

    model_id = "lstm"

    def _build_network(self, n_channels: int, n_outputs: int) -> Any:
        """Build the LSTM.

        Args:
            n_channels: Number of input channels.
            n_outputs: Output width.

        Returns:
            The network.
        """
        return _RecurrentNet(
            "lstm",
            n_channels,
            int(self.params["hidden_size"]),
            int(self.params["num_layers"]),
            float(self.params["dropout"]),
            n_outputs,
        )


class CNN1D(TorchSequenceModel):
    """One-dimensional convolutional sequence model."""

    model_id = "cnn1d"

    def _build_network(self, n_channels: int, n_outputs: int) -> Any:
        """Build the convolutional network.

        Args:
            n_channels: Number of input channels.
            n_outputs: Output width.

        Returns:
            The network.
        """
        return _Conv1DNet(
            n_channels,
            int(self.params["hidden_size"]),
            int(self.params["num_layers"]),
            float(self.params["dropout"]),
            n_outputs,
        )


class TCN(TorchSequenceModel):
    """Dilated causal temporal convolutional network."""

    model_id = "tcn"

    def _build_network(self, n_channels: int, n_outputs: int) -> Any:
        """Build the TCN.

        Args:
            n_channels: Number of input channels.
            n_outputs: Output width.

        Returns:
            The network.
        """
        return _TCNNet(
            n_channels,
            int(self.params["hidden_size"]),
            int(self.params["num_layers"]),
            float(self.params["dropout"]),
            n_outputs,
        )


DEEP_MODELS: dict[str, type[TorchSequenceModel]] = {
    GRU.model_id: GRU,
    LSTM.model_id: LSTM,
    CNN1D.model_id: CNN1D,
    TCN.model_id: TCN,
}
