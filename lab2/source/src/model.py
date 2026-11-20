"""Multi-Layer Perceptron (MLP) for binary image classification on flattened 1D vectors."""

from __future__ import annotations

import torch
from torch import nn

from src.activation import get_activation
from src.config import ExperimentConfig
from src.initializer import get_initializer


class ImageMLP(nn.Module):
    """Flexible MLP accepting flattened image vectors for binary classification.

    Note: BatchNorm is intentionally omitted so weight initialization experiments
    (Random Normal vs. Xavier vs. He) reflect genuine gradient/activation scaling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation_name: str = "relu",
        init_method: str = "he",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}.")
        if len(hidden_dims) == 0 or not all(d > 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims must contain positive ints, got {hidden_dims}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation_name = activation_name
        self.init_method = init_method
        self.dropout = dropout

        # Dynamically construct hidden blocks: Linear -> CustomActivation -> [Dropout]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(get_activation(activation_name))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Binary classification head outputs 1 raw logit per sample (for BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # Apply the selected weight initialization strategy across all Linear layers
        init_kwargs = {}
        if init_method.strip().lower() in {"he", "he_normal"} and activation_name.strip().lower() == "leaky_relu":
            init_kwargs["negative_slope"] = 0.01
        self.initializer = get_initializer(init_method, **init_kwargs)
        self.initializer.apply(self)

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> ImageMLP:
        """Instantiate an ImageMLP directly from an ExperimentConfig object."""
        return cls(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            activation_name=cfg.activation_name,
            init_method=cfg.init_method,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw binary logits of shape (batch_size,) from flattened image vectors."""
        # Safety fallback: flatten multi-dimensional batch (B, C, H, W) -> (B, C*H*W)
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        # Use squeeze(-1) instead of squeeze() so batch_size == 1 still produces shape (1,)
        return self.network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted positive-class probabilities in (0, 1)."""
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary class predictions (0 or 1) using the given probability threshold."""
        return (self.predict_proba(x) >= threshold).to(dtype=torch.long)

    def count_parameters(self) -> int:
        """Return total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_specs(self) -> list[dict[str, object]]:
        """Return structured layer dimensions and parameter counts for visualization."""
        specs: list[dict[str, object]] = [
            {"name": "Input (Flattened Image)", "dim": self.input_dim, "params": 0}
        ]
        for idx, h_dim in enumerate(self.hidden_dims, start=1):
            in_d = self.input_dim if idx == 1 else self.hidden_dims[idx - 2]
            params = in_d * h_dim + h_dim  # weights + bias
            detail = f"Linear + {self.activation_name}"
            if self.dropout > 0.0:
                detail += f" + Dropout({self.dropout})"
            specs.append({"name": f"Hidden {idx}\n({detail})", "dim": h_dim, "params": params})

        out_params = self.hidden_dims[-1] * 1 + 1
        specs.append({"name": "Output\n(Linear + Sigmoid)", "dim": 1, "params": out_params})
        return specs
