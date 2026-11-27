"""Centralized configuration and deterministic seeding for Lab #03 multi-class system."""

from dataclasses import dataclass, field
import math
import os
import random

import numpy as np
import torch


# Fixed ordering of the 6 active Vietnamese polymer banknote classes (C = 6 >= 3)
DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "10000_VND",
    "20000_VND",
    "50000_VND",
    "100000_VND",
    "200000_VND",
    "500000_VND",
)

# Human-readable labels for charts, overlays, and the interactive UI
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "10000_VND": "10,000 VND",
    "20000_VND": "20,000 VND",
    "50000_VND": "50,000 VND",
    "100000_VND": "100,000 VND",
    "200000_VND": "200,000 VND",
    "500000_VND": "500,000 VND",
}


def seed_everything(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducible experiments."""
    # Guard clause: reject booleans, non-integers, or negative seed values
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)  # Fix hash randomization in Python
    random.seed(seed)  # Fix Python built-in random module
    np.random.seed(seed)  # Fix NumPy RNG
    torch.manual_seed(seed)  # Fix PyTorch CPU RNG

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Fix PyTorch CUDA RNG across all GPUs
        torch.backends.cudnn.deterministic = True  # Enforce deterministic cuDNN ops
        torch.backends.cudnn.benchmark = False  # Disable auto-tuner to avoid variance

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # Fix PyTorch Apple Silicon MPS RNG


def get_device() -> str:
    """Automatically select the fastest available compute backend (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Lab3Config:
    """Centralized configuration for dataset, model architecture, and training."""

    # --- Data settings ---
    data_dir: str = "../data/vietnamese_currency_6classes"  # Root folder containing class subfolders
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES  # Ordered class folder names
    img_size: int = 64  # Target spatial resolution (e.g., 64 for MLP, 128 for CNN)
    num_channels: int = 3  # 3 for RGB images (color cues are critical for VND banknotes)
    train_ratio: float = 0.70  # Fraction of samples assigned to Train
    val_ratio: float = 0.15  # Fraction of samples assigned to Validation
    test_ratio: float = 0.15  # Fraction of samples assigned to Test
    batch_size: int = 32  # Mini-batch size suited to ~915 training images
    num_workers: int = 0  # Worker processes (0 avoids macOS/Streamlit fork overhead)

    # --- Model settings ---
    model_type: str = "mlp"  # "mlp" (Lab #02 upgrade), "cnn", or "mobilenet" (Transfer Learning)
    hidden_dims: tuple[int, ...] = (512, 256, 128)  # Hidden layer widths for MultiClassImageMLP
    activation_name: str = "relu"  # Custom activation key (relu, leaky_relu, tanh, sigmoid)
    init_method: str = "he"  # Weight initializer key (normal, xavier, he)
    dropout: float = 0.25  # Dropout rate for regularization

    # --- Training settings ---
    lr: float = 3e-4  # Adam optimizer learning rate
    weight_decay: float = 1e-4  # L2 weight decay to reduce overfitting
    epochs: int = 35  # Number of training epochs
    seed: int = 42  # Global seed for reproducible splits and weight initialization
    device: str = field(default_factory=get_device)  # Evaluated dynamically per instance

    def __post_init__(self) -> None:
        """Validate configuration invariants immediately after initialization."""
        # Validate class list has at least 3 distinct classes for multi-class Lab #03
        if len(self.class_names) < 3:
            raise ValueError(f"Multi-class Lab #03 requires >= 3 classes, got {len(self.class_names)}.")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"class_names contains duplicate entries: {self.class_names}.")

        # Validate spatial and channel settings
        if self.img_size <= 0:
            raise ValueError(f"img_size must be > 0, got {self.img_size}.")
        if self.num_channels not in (1, 3):
            raise ValueError(f"num_channels must be 1 or 3, got {self.num_channels}.")

        # Validate DataLoader and optimization hyperparameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}.")

        # Validate model settings
        if self.model_type not in {"mlp", "cnn", "mobilenet"}:
            raise ValueError(
                f"model_type must be 'mlp', 'cnn', or 'mobilenet', got '{self.model_type}'."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if len(self.hidden_dims) == 0 or not all(dim > 0 for dim in self.hidden_dims):
            raise ValueError(f"hidden_dims must contain positive ints, got {self.hidden_dims}.")

        # Validate split ratios are strictly positive and sum to 1.0
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"train/val/test ratios must be > 0 and sum to 1.0, got {ratios}."
            )

    @property
    def num_classes(self) -> int:
        """Return total number of target classes C (Single Source of Truth)."""
        return len(self.class_names)

    @property
    def input_dim(self) -> int:
        """Return flattened 1D input vector dimension (C * H * W) for MultiClassImageMLP."""
        return self.num_channels * self.img_size * self.img_size
