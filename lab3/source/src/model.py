"""Multi-class neural network architectures for Practice Lab #03 (MLP upgrade + CurrencyCNN)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.config import Lab3Config


class BaseClassifier(nn.Module):
    """Shared base class providing He weight init, probability inference, and parameter counting."""

    def _init_weights(self) -> None:
        """Initialize Conv2d/Linear weights with PyTorch built-in Kaiming Normal and BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return multi-class Softmax probability distribution of shape (B, C) summing to 1.0."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return Top-1 predicted class indices of shape (B,) with dtype torch.long."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiClassImageMLP(BaseClassifier):
    """Multi-Class upgrade of Lab #02 ImageMLP outputting C raw logits for nn.CrossEntropyLoss."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        # Guard clauses for architecture dimensions and regularization bounds
        if input_dim <= 0 or num_classes < 2:
            raise ValueError(f"Invalid input_dim={input_dim} or num_classes={num_classes}.")
        if not hidden_dims or any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must contain positive ints, got {hidden_dims}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim

        # Multi-class classification head emits C raw logits (Softmax is applied inside CrossEntropyLoss)
        layers.append(nn.Linear(last_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits of shape (B, C) from flattened or 4D image batches."""
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)


class CurrencyCNN(BaseClassifier):
    """Lightweight 2D CNN with Concat (Avg + Max) Pooling for robust real-world banknote recognition."""

    @staticmethod
    def conv_block(
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ) -> nn.Sequential:
        """Build one Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d feature extraction stage."""
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 3) or num_classes < 2 or not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"Invalid CurrencyCNN args: in_channels={in_channels}, num_classes={num_classes}, dropout={dropout}."
            )

        # 4 convolutional downsampling stages: 3 -> 32 -> 64 -> 128 -> 128
        self.features = nn.Sequential(
            self.conv_block(in_channels, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 128),
        )
        # Dual global pooling: AvgPool captures overall hue; MaxPool locks onto sharp banknote patterns
        # even when room background occupies part of the webcam frame
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classification head maps concatenated 256-D feature vector -> C raw logits
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits of shape (B, C) from 4D RGB tensor batches (B, C, H, W)."""
        if x.ndim != 4:
            raise ValueError(f"CurrencyCNN expects 4D tensor (B, C, H, W), got shape {tuple(x.shape)}.")
        feat = self.features(x)
        pooled_avg = self.avg_pool(feat).flatten(start_dim=1)
        pooled_max = self.max_pool(feat).flatten(start_dim=1)
        combined = torch.cat([pooled_avg, pooled_max], dim=1)
        return self.classifier(combined)


class CurrencyMobileNetV2(BaseClassifier):
    """Transfer-learning MobileNetV2 classifier with built-in ImageNet normalization.

    Preserves rich pretrained edge/texture/digit features so real-world webcams recognize
    Vietnamese banknotes robustly across varied camera sensors and room lighting.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.25,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2 or not (0.0 <= dropout < 1.0):
            raise ValueError(f"Invalid CurrencyMobileNetV2 args: num_classes={num_classes}, dropout={dropout}.")

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # Replace final ImageNet 1000-class head with our C-class VND banknote head
        in_features = self.backbone.last_channel
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        # Initialize ONLY the newly attached linear head (never overwrite pretrained conv filters!)
        nn.init.kaiming_normal_(self.backbone.classifier[1].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.backbone.classifier[1].bias)

        # Register ImageNet channel statistics as buffers so [0, 1] inputs auto-normalize on any device
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize [0, 1] RGB batch (B, 3, H, W) using ImageNet stats and emit raw logits (B, C)."""
        if x.ndim != 4:
            raise ValueError(f"CurrencyMobileNetV2 expects 4D tensor (B, 3, H, W), got shape {tuple(x.shape)}.")
        x_norm = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x_norm)


def build_model(cfg: Lab3Config, pretrained: bool = True) -> BaseClassifier:
    """Factory instantiating MultiClassImageMLP, CurrencyCNN, or CurrencyMobileNetV2 from Lab3Config."""
    if cfg.model_type == "mlp":
        return MultiClassImageMLP(cfg.input_dim, cfg.num_classes, cfg.hidden_dims, cfg.dropout)
    if cfg.model_type == "cnn":
        return CurrencyCNN(cfg.num_channels, cfg.num_classes, cfg.dropout)
    if cfg.model_type == "mobilenet":
        return CurrencyMobileNetV2(cfg.num_classes, cfg.dropout, pretrained=pretrained)
    raise ValueError(f"Unsupported model_type '{cfg.model_type}'. Expected 'mlp', 'cnn', or 'mobilenet'.")
