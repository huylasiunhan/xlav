"""Custom weight initialization strategies (Random Normal baseline, Xavier/Glorot, He/Kaiming)."""

from abc import ABC, abstractmethod
import math

import torch
from torch import nn


class BaseInitializer(ABC):
    """Abstract base class for applying weight initialization to Linear layers."""

    @abstractmethod
    def init_linear(self, layer: nn.Linear) -> None:
        """Initialize weights and biases of a single nn.Linear layer in-place."""

    def apply(self, model: nn.Module) -> None:
        """Traverse all submodules in model and initialize every nn.Linear layer."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.init_linear(module)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomNormalInit(BaseInitializer):
    """Baseline initialization: W ~ N(mean, std^2), ignoring layer dimensions."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        if std <= 0.0:
            raise ValueError(f"std must be > 0, got {std}.")
        self.mean = mean
        self.std = std

    def init_linear(self, layer: nn.Linear) -> None:
        # Modify parameter tensors inside no_grad() so autograd does not track initialization
        with torch.no_grad():
            layer.weight.normal_(mean=self.mean, std=self.std)
            if layer.bias is not None:
                layer.bias.zero_()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class XavierNormalInit(BaseInitializer):
    """Xavier/Glorot Normal (2010): W ~ N(0, 2 / (fan_in + fan_out))."""

    def init_linear(self, layer: nn.Linear) -> None:
        # In PyTorch nn.Linear, weight has shape (fan_out, fan_in)
        fan_out, fan_in = layer.weight.shape
        std = math.sqrt(2.0 / (fan_in + fan_out))

        with torch.no_grad():
            layer.weight.normal_(mean=0.0, std=std)
            if layer.bias is not None:
                layer.bias.zero_()


class XavierUniformInit(BaseInitializer):
    """Xavier/Glorot Uniform (2010): W ~ U(-limit, +limit), limit = sqrt(6 / (fan_in + fan_out))."""

    def init_linear(self, layer: nn.Linear) -> None:
        fan_out, fan_in = layer.weight.shape
        limit = math.sqrt(6.0 / (fan_in + fan_out))

        with torch.no_grad():
            layer.weight.uniform_(-limit, limit)
            if layer.bias is not None:
                layer.bias.zero_()


class HeNormalInit(BaseInitializer):
    """He/Kaiming Normal (2015): W ~ N(0, 2 / ((1 + a^2) * fan_in)) for ReLU / LeakyReLU."""

    def __init__(self, negative_slope: float = 0.0) -> None:
        # negative_slope = 0.0 for ReLU, or alpha (e.g., 0.01) for LeakyReLU
        if negative_slope < 0.0:
            raise ValueError(f"negative_slope must be >= 0, got {negative_slope}.")
        self.negative_slope = negative_slope

    def init_linear(self, layer: nn.Linear) -> None:
        _, fan_in = layer.weight.shape
        # Factor 2 / (1 + a^2) preserves activation variance after zeroing/scaling negative half
        std = math.sqrt(2.0 / ((1.0 + self.negative_slope ** 2) * fan_in))

        with torch.no_grad():
            layer.weight.normal_(mean=0.0, std=std)
            if layer.bias is not None:
                layer.bias.zero_()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(negative_slope={self.negative_slope})"


# Registry mapping string keys (from ExperimentConfig.init_method) to initializer classes
INITIALIZER_REGISTRY: dict[str, type[BaseInitializer]] = {
    "normal": RandomNormalInit,
    "xavier": XavierNormalInit,
    "xavier_normal": XavierNormalInit,
    "xavier_uniform": XavierUniformInit,
    "he": HeNormalInit,
    "he_normal": HeNormalInit,
}


def get_initializer(name: str, **kwargs) -> BaseInitializer:
    """Factory function to create a BaseInitializer instance by registry name."""
    key = name.strip().lower()
    if key not in INITIALIZER_REGISTRY:
        valid = ", ".join(sorted(INITIALIZER_REGISTRY.keys()))
        raise ValueError(f"Unknown initializer '{name}'. Supported: {valid}.")
    return INITIALIZER_REGISTRY[key](**kwargs)
