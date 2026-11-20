"""Custom activation functions, their mathematical derivatives, and PyTorch autograd bridge."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseActivation(ABC):
    """Abstract base class defining forward pass f(x) and derivative f'(x)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute activation output f(x)."""

    @abstractmethod
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute exact mathematical derivative f'(x) with respect to input x."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ReLUActivation(BaseActivation):
    """Rectified Linear Unit: f(x) = max(0, x), f'(x) = 1 if x > 0 else 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp negative values to 0 while preserving positive values
        return torch.clamp(x, min=0.0)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # Derivative is 1 for x > 0 and 0 for x <= 0 (preserve input dtype and device)
        return (x > 0).to(dtype=x.dtype)


class LeakyReLUActivation(BaseActivation):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x (mitigates dying ReLU)."""

    def __init__(self, alpha: float = 0.01) -> None:
        # Guard clause: negative slope alpha must lie strictly inside (0, 1)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep x where x > 0; scale negative inputs by alpha
        return torch.where(x > 0, x, self.alpha * x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # Slope is 1.0 for x > 0 and alpha for x <= 0 (matching x's dtype & device)
        ones = torch.ones_like(x)
        alphas = torch.full_like(x, self.alpha)
        return torch.where(x > 0, ones, alphas)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class TanhActivation(BaseActivation):
    """Hyperbolic Tangent: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map real inputs smoothly into (-1, 1)
        return torch.tanh(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # Closed-form derivative: d/dx tanh(x) = 1 - tanh^2(x)
        t = torch.tanh(x)
        return 1.0 - t * t


class SigmoidActivation(BaseActivation):
    """Logistic Sigmoid: f(x) = 1 / (1 + exp(-x)), f'(x) = f(x) * (1 - f(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map real inputs into probability range (0, 1)
        return torch.sigmoid(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # Closed-form derivative: d/dx sigmoid(x) = s * (1 - s)
        s = torch.sigmoid(x)
        return s * (1.0 - s)


class _ActivationAutogradFn(torch.autograd.Function):
    """Generic PyTorch Autograd bridge that injects BaseActivation.derivative into backprop."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        activation: BaseActivation,
    ) -> torch.Tensor:
        # Save input tensor x and activation object for the backward pass
        ctx.save_for_backward(x)
        ctx.activation = activation
        return activation.forward(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # Retrieve saved input x and apply Chain Rule: dL/dx = (dL/dy) * f'(x)
        (x,) = ctx.saved_tensors
        grad_input = grad_output * ctx.activation.derivative(x)
        # Return None for the second argument (activation is not a Tensor)
        return grad_input, None


class CustomActivation(nn.Module):
    """PyTorch nn.Module wrapper around any BaseActivation using custom backward derivatives."""

    def __init__(self, activation: BaseActivation) -> None:
        super().__init__()
        if not isinstance(activation, BaseActivation):
            raise TypeError(
                f"activation must be a BaseActivation instance, got {type(activation)}."
            )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate through _ActivationAutogradFn so backpropagation calls activation.derivative(x)
        return _ActivationAutogradFn.apply(x, self.activation)

    def extra_repr(self) -> str:
        # Display activation details cleanly when printing the parent model
        return repr(self.activation)


# Registry mapping string keys (from ExperimentConfig) to BaseActivation subclasses
ACTIVATION_REGISTRY: dict[str, type[BaseActivation]] = {
    "relu": ReLUActivation,
    "leaky_relu": LeakyReLUActivation,
    "tanh": TanhActivation,
    "sigmoid": SigmoidActivation,
}


def get_activation(name: str, **kwargs) -> CustomActivation:
    """Factory function to create a CustomActivation module by registry name."""
    key = name.strip().lower()
    if key not in ACTIVATION_REGISTRY:
        valid = ", ".join(sorted(ACTIVATION_REGISTRY.keys()))
        raise ValueError(f"Unknown activation '{name}'. Supported: {valid}.")
    activation_cls = ACTIVATION_REGISTRY[key]
    return CustomActivation(activation_cls(**kwargs))


def verify_activation_derivatives(eps: float = 1e-6) -> dict[str, float]:
    """Compare analytical derivatives against central finite-difference approximations."""
    # Sample test points away from x = 0 (where ReLU/LeakyReLU are non-differentiable)
    x = torch.tensor([-2.5, -1.0, -0.3, 0.3, 1.0, 2.5], dtype=torch.float64)
    max_errors: dict[str, float] = {}

    for name, act_cls in ACTIVATION_REGISTRY.items():
        act = act_cls()
        analytical = act.derivative(x)
        # Central difference formula: f'(x) ~= (f(x + eps) - f(x - eps)) / (2 * eps)
        numerical = (act.forward(x + eps) - act.forward(x - eps)) / (2.0 * eps)
        max_errors[name] = (analytical - numerical).abs().max().item()

    return max_errors


