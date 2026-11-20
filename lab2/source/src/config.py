from dataclasses import dataclass, field
import math
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    # Guard clause: reject bool, non-int, or negative seed values
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)  # Fix Python hash randomization
    random.seed(seed)  # Fix Python built-in random module
    np.random.seed(seed)  # Fix NumPy RNG
    torch.manual_seed(seed)  # Fix PyTorch CPU RNG

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Fix PyTorch CUDA RNG for all GPUs
        torch.backends.cudnn.deterministic = True  # Enforce deterministic cuDNN algorithms
        torch.backends.cudnn.benchmark = False  # Disable auto-tuner to avoid nondeterminism

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # Fix PyTorch Apple Silicon MPS RNG


def get_device() -> str:
    """Automatically select the fastest available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ExperimentConfig:
    """Centralized configuration for dataset, MLP architecture, and training."""

    # --- Data settings ---
    data_dir: str = "/content/data"  # Root folder containing Positive/ and Negative/
    img_size: int = 64  # Target height and width after resizing (64x64)
    num_channels: int = 3  # 3 for RGB images, 1 for grayscale
    train_ratio: float = 0.70  # Fraction of data used for training
    val_ratio: float = 0.15  # Fraction of data used for validation
    test_ratio: float = 0.15  # Fraction of data used for final testing
    batch_size: int = 64  # Mini-batch size for DataLoader
    num_workers: int = 2  # Parallel data-loading worker processes

    # --- Model settings ---
    hidden_dims: tuple[int, ...] = (512, 256, 128)  # Neurons per hidden MLP layer
    activation_name: str = "relu"  # Activation function key (e.g., relu, leaky_relu, tanh)
    init_method: str = "he"  # Weight initializer key (normal, xavier, he)
    dropout: float = 0.2  # Dropout probability for regularization

    # --- Training settings ---
    lr: float = 1e-4  # Learning rate tuned for high-dimensional (12,288-D) MLP inputs
    epochs: int = 30  # Number of training epochs
    seed: int = 42  # Global random seed for reproducible splits and training
    device: str = field(default_factory=get_device)  # Evaluated dynamically per instance

    def __post_init__(self) -> None:
        """Validate configuration values immediately after initialization."""
        if self.img_size <= 0:
            raise ValueError(f"img_size must be > 0, got {self.img_size}.")
        if self.num_channels not in (1, 3):
            raise ValueError(f"num_channels must be 1 or 3, got {self.num_channels}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}.")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if len(self.hidden_dims) == 0 or not all(dim > 0 for dim in self.hidden_dims):
            raise ValueError(f"hidden_dims must be non-empty positive ints, got {self.hidden_dims}.")

        # Validate dataset split ratios are strictly positive and sum to 1.0
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"train/val/test ratios must be > 0 and sum to 1.0, got {ratios}."
            )

    @property
    def input_dim(self) -> int:
        """Flattened 1D input vector dimension for the MLP (C * H * W)."""
        return self.num_channels * self.img_size * self.img_size

