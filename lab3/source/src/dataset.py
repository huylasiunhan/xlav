"""Multi-class dataset scanning, stratified Train/Val/Test splitting, and PyTorch DataLoaders."""

from collections.abc import Callable
import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DEFAULT_CLASS_NAMES, Lab3Config


# Allowed raster image file extensions when scanning class directories
VALID_IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png"}


def collect_image_paths_and_labels(
    data_dir: str = Lab3Config.data_dir,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> tuple[list[str], list[int], dict[str, int]]:
    """Scan class folders and return deterministic (image_paths, int_labels, class_to_idx)."""
    root = Path(data_dir)
    # Guard clause: ensure the dataset root directory exists before scanning
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    # Build deterministic class-name -> integer-index mapping (0 .. C-1)
    class_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    image_paths: list[str] = []
    labels: list[int] = []

    for class_name, label_idx in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        # Sort filenames so sample ordering is identical across macOS and Linux/Colab
        class_files = [
            f
            for f in sorted(class_dir.iterdir())
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_SUFFIXES
        ]
        if not class_files:
            raise FileNotFoundError(f"No valid image files found in class folder: {class_dir}")

        for img_file in class_files:
            image_paths.append(str(img_file))
            labels.append(label_idx)

    return image_paths, labels, class_to_idx


def split_dataset(
    paths: list[str],
    labels: list[int],
    train_ratio: float = Lab3Config.train_ratio,
    val_ratio: float = Lab3Config.val_ratio,
    test_ratio: float = Lab3Config.test_ratio,
    seed: int = Lab3Config.seed,
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
]:
    """Split paths and labels into stratified Train / Validation / Test partitions."""
    if (
        train_ratio <= 0
        or val_ratio <= 0
        or test_ratio <= 0
        or not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6)
    ):
        raise ValueError("train_ratio, val_ratio, test_ratio must be > 0 and sum to 1.0.")

    # Step 1: separate Train from (Val + Test) while preserving per-class proportions
    x_train, x_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: separate Validation and Test using relative proportion inside temp
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y_temp,
        random_state=seed,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class CurrencyDataset(Dataset):
    """PyTorch Dataset performing lazy RGB image loading for multi-class classification."""

    def __init__(
        self,
        img_paths: list[str],
        labels: list[int],
        transform: Callable | None = None,
    ) -> None:
        if len(img_paths) != len(labels):
            raise ValueError("Length of img_paths and labels must be equal.")
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | np.ndarray, torch.Tensor]:
        # Context manager closes file descriptor immediately after reading
        with Image.open(self.img_paths[index]) as img:
            rgb_array = np.array(img.convert("RGB"))

        # Apply image transform pipeline when provided
        x = self.transform(rgb_array) if self.transform is not None else rgb_array
        # torch.long (int64) class index [0..C-1] is required by nn.CrossEntropyLoss
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


def create_dataloaders(
    config: Lab3Config,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Create stratified Train, Validation, and Test DataLoaders plus class_to_idx map."""
    img_paths, labels, class_to_idx = collect_image_paths_and_labels(
        data_dir=config.data_dir,
        class_names=config.class_names,
    )
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        paths=img_paths,
        labels=labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    # Lazily load default transform pipelines from src.transform when not passed explicitly
    if train_transform is None or eval_transform is None:
        from src.transform import (
            Compose,
            build_augmentation_pipeline,
            build_preprocess_pipeline,
        )

        # MLP expects flattened 1D vectors; CNN expects 3D (C, H, W) image tensors
        flatten_input = config.model_type == "mlp"
        prep_pipeline = build_preprocess_pipeline(
            img_size=config.img_size,
            flatten=flatten_input,
        )
        if train_transform is None:
            aug_pipeline = build_augmentation_pipeline()
            train_transform = Compose([aug_pipeline, prep_pipeline])
        if eval_transform is None:
            eval_transform = prep_pipeline

    # Wrap each split into a CurrencyDataset (augmentation applied ONLY to train_ds)
    train_ds = CurrencyDataset(img_paths=x_train, labels=y_train, transform=train_transform)
    val_ds = CurrencyDataset(img_paths=x_val, labels=y_val, transform=eval_transform)
    test_ds = CurrencyDataset(img_paths=x_test, labels=y_test, transform=eval_transform)

    # Enable CUDA pinned memory and persistent workers only when beneficial
    use_pin_memory = config.device == "cuda"
    use_persistent = config.num_workers > 0

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )
    return train_loader, val_loader, test_loader, class_to_idx
