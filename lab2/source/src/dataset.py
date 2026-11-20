
"""Dataset scanning, stratified Train/Val/Test splitting, and PyTorch DataLoader creation."""

import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.transform import (
    BaseTransform,
    Compose,
    build_augmentation_pipeline,
    build_preprocess_pipeline,
)


def collect_image_paths_and_label(
    data_dir: str = ExperimentConfig.data_dir,
) -> tuple[list[str], list[int]]:
    """Collect sorted image file paths and binary labels (Negative=0, Positive=1)."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    mapping = {"Negative": 0, "Positive": 1}
    accept_suffixes = {".jpg", ".jpeg", ".png"}
    image_paths: list[str] = []
    labels: list[int] = []

    for class_name, label in mapping.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        # Sort filenames so ordering is deterministic across macOS and Linux/Colab
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in accept_suffixes:
                image_paths.append(str(f))
                labels.append(label)

    if not image_paths:
        raise FileNotFoundError(f"No valid image files found in {root}.")
    return image_paths, labels


def split_dataset(
    paths: list[str],
    labels: list[int],
    train_ratio: float = ExperimentConfig.train_ratio,
    val_ratio: float = ExperimentConfig.val_ratio,
    test_ratio: float = ExperimentConfig.test_ratio,
    seed: int = ExperimentConfig.seed,
):
    """Split paths and labels into stratified Train / Validation / Test partitions."""
    if (
        train_ratio <= 0
        or val_ratio <= 0
        or test_ratio <= 0
        or not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6)
    ):
        raise ValueError("train_ratio, val_ratio, test_ratio must be > 0 and sum to 1.0.")

    # Step 1: separate Train from (Val + Test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=test_ratio + val_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Step 2: separate Val and Test using relative proportion inside temp
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_ratio / (test_ratio + val_ratio),
        stratify=y_temp,
        random_state=seed,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class SurfaceCrackDataset(Dataset):
    """PyTorch Dataset performing lazy RGB image loading and custom transforms."""

    def __init__(
        self,
        img_paths: list[str],
        labels: list[int],
        transform: BaseTransform | None = None,
    ) -> None:
        if len(img_paths) != len(labels):
            raise ValueError("Length of img_paths and labels must be equal.")
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Context manager closes file descriptor immediately after reading
        with Image.open(self.img_paths[index]) as img:
            rgb_array = np.array(img.convert("RGB"))

        x = self.transform(rgb_array) if self.transform is not None else rgb_array
        # float32 label matches BCEWithLogitsLoss target dtype requirement
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


def create_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train (augmented + shuffled), Validation, and Test DataLoaders."""
    img_paths, labels = collect_image_paths_and_label(config.data_dir)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        img_paths,
        labels,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )

    aug_pipeline = build_augmentation_pipeline()
    prep_pipeline = build_preprocess_pipeline(config.img_size)

    # Apply augmentation only to Train; keep Val & Test strictly deterministic
    train_transform = Compose([aug_pipeline, prep_pipeline])
    eval_transform = prep_pipeline

    train_ds = SurfaceCrackDataset(img_paths=x_train, labels=y_train, transform=train_transform)
    val_ds = SurfaceCrackDataset(img_paths=x_val, labels=y_val, transform=eval_transform)
    test_ds = SurfaceCrackDataset(img_paths=x_test, labels=y_test, transform=eval_transform)

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
    return train_loader, val_loader, test_loader
