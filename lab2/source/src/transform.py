"""Custom OOP image augmentation and preprocessing transforms for MLP binary classification."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np
import torch


class BaseTransform(ABC):
    """Abstract base class defining the callable interface for all image transforms."""

    @abstractmethod
    def __call__(self, x):
        """Apply transformation to input x and return transformed output."""

    def __repr__(self) -> str:
        """Default string representation showing the transform class name."""
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    """Composite transform that chains multiple BaseTransform steps sequentially."""

    def __init__(self, transforms: Sequence[BaseTransform]) -> None:
        # Validate every step implements BaseTransform before building the pipeline
        for t in transforms:
            if not isinstance(t, BaseTransform):
                raise TypeError(f"Transform must be an instance of BaseTransform, got {type(t)}")
        # Store an independent list copy to avoid external mutation side-effects
        self.transforms = list(transforms)

    def __call__(self, x):
        # Pass data sequentially through each transform in order
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        return "Compose(" + " -> ".join(repr(t) for t in self.transforms) + ")"


# ---------------------------------------------------------------------------
# Data Augmentation Transforms (applied to Train RGB NumPy images [H, W, C])
# ---------------------------------------------------------------------------

class RandomHorizontalFlip(BaseTransform):
    """Randomly flip an RGB image horizontally (left-right) with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # Flip along width axis (axis=1) and copy to avoid negative-stride PyTorch error
            return np.flip(x, axis=1).copy()
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(BaseTransform):
    """Randomly rotate an image within [-max_angle, +max_angle] using reflective borders."""

    def __init__(self, max_angle: float = 15.0, p: float = 0.5) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        if max_angle < 0.0:
            raise ValueError("max_angle must be non-negative")
        self.p = p
        self.max_angle = max_angle

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            return self.rotate_image(x, angle)
        return x

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image while expanding bounding box and reflecting concrete texture borders."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        # Compute 2x3 affine rotation matrix around the image center
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute new bounding dimensions so image corners are not clipped
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Shift translation terms to re-center rotated image inside the expanded frame
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # Use BORDER_REFLECT so rotated corners mimic concrete surface instead of black triangles
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_angle={self.max_angle}, p={self.p})"


class RandomBrightnessContrast(BaseTransform):
    """Randomly perturb per-channel brightness and contrast to simulate outdoor lighting."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        if brightness_range[0] < 0.0 or brightness_range[1] > 2.0 or brightness_range[0] > brightness_range[1]:
            raise ValueError("brightness_range must be in [0, 2] with min <= max")
        if contrast_range[0] < 0.0 or contrast_range[1] > 2.0 or contrast_range[0] > contrast_range[1]:
            raise ValueError("contrast_range must be in [0, 2] with min <= max")
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self._adjust_brightness_contrast(x)
        return x

    def _adjust_brightness_contrast(self, x: np.ndarray) -> np.ndarray:
        """Scale deviations around per-channel mean (contrast) and shift mean (brightness)."""
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        # Compute per-channel spatial mean with shape (1, 1, C) in float precision
        mean = np.mean(x, axis=(0, 1), keepdims=True)
        x = (x - mean) * contrast_factor + mean * brightness_factor
        # Clamp pixel intensities back to valid uint8 range [0, 255]
        return np.clip(x, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_range={self.brightness_range}, "
            f"contrast_range={self.contrast_range}, p={self.p})"
        )


# ---------------------------------------------------------------------------
# Preprocessing Transforms (applied consistently to Train, Val, and Test)
# ---------------------------------------------------------------------------

class Resize(BaseTransform):
    """Resize RGB image to fixed (size, size) using area interpolation for clean downscaling."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # INTER_AREA minimizes aliasing artifacts when downsampling 227x227 -> 64x64
        return cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ToTensorNormalize(BaseTransform):
    """Convert uint8 [0, 255] image (H, W, C) to float32 [0.0, 1.0] PyTorch tensor (C, H, W)."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        # Cast to float32 and normalize pixel range to [0.0, 1.0]
        x = x.astype(np.float32) / 255.0
        if x.ndim == 2:
            # Promote grayscale (H, W) images to 3-channel (3, H, W)
            x = np.stack([x, x, x], axis=0)
        else:
            # Convert channel-last (H, W, C) to PyTorch channel-first (C, H, W)
            x = np.transpose(x, (2, 0, 1))
        # Ensure contiguous memory layout before zero-copy tensor conversion
        return torch.from_numpy(np.ascontiguousarray(x))


class Flatten1D(BaseTransform):
    """Flatten a 3D image tensor (C, H, W) into a 1D vector (C * H * W,) for MLP input."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()


def build_augmentation_pipeline() -> Compose:
    """Build the default 3-step Data Augmentation pipeline for training images."""
    return Compose([
        RandomRotation(),
        RandomBrightnessContrast(),
        RandomHorizontalFlip(),
    ])


def build_preprocess_pipeline(img_size: int = 64) -> Compose:
    """Build the standardized Resize -> Normalize [0, 1] -> Flatten 1D preprocessing pipeline."""
    return Compose([
        Resize(img_size),
        ToTensorNormalize(),
        Flatten1D(),
    ])
