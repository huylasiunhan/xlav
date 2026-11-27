"""Custom OOP image augmentation and preprocessing transforms for Lab #03 multi-class classification."""

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
                raise TypeError(f"Transform must be an instance of BaseTransform, got {type(t)}.")
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

class RandomResizedCrop(BaseTransform):
    """Randomly crop a sub-region and resize back so models focus on banknote features, not room background."""

    def __init__(self, scale: tuple[float, float] = (0.65, 1.0), p: float = 0.7) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        if not (0.0 < scale[0] <= scale[1] <= 1.0):
            raise ValueError(f"scale must satisfy 0 < min <= max <= 1.0, got {scale}.")
        self.scale = scale
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.p:
            return x
        h, w = x.shape[:2]
        # Sample independent height/width crop fractions within the configured scale range
        frac_h = np.random.uniform(self.scale[0], self.scale[1])
        frac_w = np.random.uniform(self.scale[0], self.scale[1])
        crop_h = max(16, int(round(h * frac_h)))
        crop_w = max(16, int(round(w * frac_w)))

        # Pick a valid random top-left anchor inside the image bounds
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        cropped = x[top : top + crop_h, left : left + crop_w]

        # Resize the cropped region back to the original (w, h) dimensions
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale}, p={self.p})"


class RandomHorizontalFlip(BaseTransform):
    """Randomly flip an RGB image horizontally (left-right) with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
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

    def __init__(self, max_angle: float = 20.0, p: float = 0.6) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        if max_angle < 0.0:
            raise ValueError(f"max_angle must be non-negative, got {max_angle}.")
        self.p = p
        self.max_angle = max_angle

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            return self.rotate_image(x, angle)
        return x

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate around the image center while preserving frame size with reflective borders."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        # Compute 2x3 affine rotation matrix around image center
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use BORDER_REFLECT to avoid artificial black corner triangles
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_angle={self.max_angle}, p={self.p})"


class RandomBrightnessContrast(BaseTransform):
    """Randomly perturb brightness and contrast to simulate varied indoor/webcam lighting."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.75, 1.25),
        contrast_range: tuple[float, float] = (0.80, 1.25),
        p: float = 0.6,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        if brightness_range[0] < 0.0 or brightness_range[1] > 2.0 or brightness_range[0] > brightness_range[1]:
            raise ValueError("brightness_range must be in [0, 2] with min <= max.")
        if contrast_range[0] < 0.0 or contrast_range[1] > 2.0 or contrast_range[0] > contrast_range[1]:
            raise ValueError("contrast_range must be in [0, 2] with min <= max.")
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
        img_f = x.astype(np.float32)
        # Compute per-channel spatial mean with shape (1, 1, C)
        mean = np.mean(img_f, axis=(0, 1), keepdims=True)
        out = (img_f - mean) * contrast_factor + mean * brightness_factor
        # Clamp pixel intensities back to valid uint8 range [0, 255]
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_range={self.brightness_range}, "
            f"contrast_range={self.contrast_range}, p={self.p})"
        )


# ---------------------------------------------------------------------------
# Preprocessing Transforms (applied consistently to Train, Val, Test, and Inference)
# ---------------------------------------------------------------------------

class Resize(BaseTransform):
    """Resize RGB image to fixed (size, size) choosing area/linear interpolation automatically."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}.")
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        # INTER_AREA prevents aliasing when downscaling; INTER_LINEAR smooths upscaling
        interp = cv2.INTER_AREA if (self.size < h or self.size < w) else cv2.INTER_LINEAR
        return cv2.resize(x, (self.size, self.size), interpolation=interp)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ToTensorNormalize(BaseTransform):
    """Convert uint8 [0, 255] image (H, W, C) to float32 [0.0, 1.0] PyTorch tensor (C, H, W)."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        # Cast to float32 and scale intensity range into [0.0, 1.0]
        arr = x.astype(np.float32) / 255.0
        if arr.ndim == 2:
            # Promote grayscale (H, W) images to 3-channel (3, H, W)
            arr = np.stack([arr, arr, arr], axis=0)
        else:
            # Convert channel-last (H, W, C) to PyTorch channel-first (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        # Ensure contiguous memory layout before zero-copy tensor conversion
        return torch.from_numpy(np.ascontiguousarray(arr))


class Flatten1D(BaseTransform):
    """Flatten a 3D image tensor (C, H, W) into a 1D vector (C * H * W,) for MultiClassImageMLP."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()


def build_augmentation_pipeline() -> Compose:
    """Build the 4-step training Data Augmentation pipeline for handheld VND banknotes."""
    return Compose([
        RandomResizedCrop(scale=(0.65, 1.0), p=0.7),
        RandomRotation(max_angle=20.0, p=0.6),
        RandomBrightnessContrast(brightness_range=(0.75, 1.25), contrast_range=(0.80, 1.25), p=0.6),
        RandomHorizontalFlip(p=0.5),
    ])


def build_preprocess_pipeline(img_size: int = 64, flatten: bool = True) -> Compose:
    """Build deterministic preprocessing pipeline for MLP (flatten=True) or CNN (flatten=False)."""
    steps: list[BaseTransform] = [
        Resize(img_size),
        ToTensorNormalize(),
    ]
    if flatten:
        steps.append(Flatten1D())
    return Compose(steps)
