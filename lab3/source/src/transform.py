"""Custom OOP image augmentation and preprocessing transforms for Lab #03 multi-class classification.

Includes domain-randomization augmentations (synthetic room backgrounds, wide lighting/white-balance
jitter, and mild blur) so models trained on indoor Kaggle imagery generalize robustly to real-world
webcams under bright or varied room lighting.
"""

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
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        return "Compose(" + " -> ".join(repr(t) for t in self.transforms) + ")"


# ---------------------------------------------------------------------------
# Data Augmentation Transforms (applied to Train RGB NumPy images [H, W, C])
# ---------------------------------------------------------------------------

class RandomSyntheticBackground(BaseTransform):
    """Paste the central banknote region onto a randomized bright/dark/gradient room background.

    Breaks shortcut learning on the Kaggle author's dim room & teal shirt so live webcams with
    bright white walls or varied backgrounds do not skew predictions toward 20,000 VND / 50,000 VND.
    """

    def __init__(
        self,
        bill_scale: tuple[float, float] = (0.55, 0.92),
        inner_crop: tuple[float, float] = (0.55, 0.88),
        p: float = 0.65,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        self.bill_scale = bill_scale
        self.inner_crop = inner_crop
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.p:
            return x
        h, w = x.shape[:2]

        # Extract inner banknote region (stripping away original room border)
        ch_frac = np.random.uniform(self.inner_crop[0], self.inner_crop[1])
        cw_frac = np.random.uniform(self.inner_crop[0], self.inner_crop[1])
        crop_h = max(16, int(round(h * ch_frac)))
        crop_w = max(16, int(round(w * cw_frac)))
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        patch = x[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # Synthesize diverse room backgrounds: bright white/cool walls, warm walls, dark rooms, or gradients
        mode = np.random.choice(["bright_wall", "neutral", "dark", "gradient"])
        if mode == "bright_wall":
            base = np.random.randint(195, 252, size=(1, 1, 3), dtype=np.int32)
            bg = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        elif mode == "dark":
            base = np.random.randint(20, 85, size=(1, 1, 3), dtype=np.int32)
            bg = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        elif mode == "gradient":
            c1 = np.random.uniform(40.0, 245.0, size=(3,)).astype(np.float32)
            c2 = np.random.uniform(40.0, 245.0, size=(3,)).astype(np.float32)
            ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
            bg = np.broadcast_to(c1 * (1.0 - ramp) + c2 * ramp, (h, w, 3)).copy()
        else:
            base = np.random.randint(90, 210, size=(1, 1, 3), dtype=np.int32)
            bg = np.broadcast_to(base, (h, w, 3)).astype(np.float32)

        # Add subtle camera sensor grain to synthetic background
        noise = np.random.normal(0.0, 6.0, size=(h, w, 3)).astype(np.float32)
        canvas = np.clip(bg + noise, 0.0, 255.0).astype(np.uint8)

        # Paste resized banknote patch at a slightly jittered position filling 55%..92% of the frame
        scale_h = np.random.uniform(self.bill_scale[0], self.bill_scale[1])
        scale_w = np.random.uniform(self.bill_scale[0], self.bill_scale[1])
        new_h = max(16, min(h, int(round(h * scale_h))))
        new_w = max(16, min(w, int(round(w * scale_w))))
        patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = np.random.randint(0, max(1, h - new_h + 1))
        left = np.random.randint(0, max(1, w - new_w + 1))
        canvas[top : top + new_h, left : left + new_w] = patch_resized
        return canvas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bill_scale={self.bill_scale}, p={self.p})"


class RandomResizedCrop(BaseTransform):
    """Randomly crop a sub-region and resize back so models handle both close-ups and normal views."""

    def __init__(self, scale: tuple[float, float] = (0.50, 1.0), p: float = 0.65) -> None:
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
        frac_h = np.random.uniform(self.scale[0], self.scale[1])
        frac_w = np.random.uniform(self.scale[0], self.scale[1])
        crop_h = max(16, int(round(h * frac_h)))
        crop_w = max(16, int(round(w * frac_w)))

        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        cropped = x[top : top + crop_h, left : left + crop_w]
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
            return np.flip(x, axis=1).copy()
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(BaseTransform):
    """Randomly rotate an image within [-max_angle, +max_angle] using reflective borders."""

    def __init__(self, max_angle: float = 25.0, p: float = 0.6) -> None:
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
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
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
    """Wide exposure, contrast, and mild white-balance jitter bridging dim datasets to bright webcams."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.65, 1.65),
        contrast_range: tuple[float, float] = (0.75, 1.35),
        wb_jitter: float = 0.08,
        p: float = 0.8,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        if brightness_range[0] <= 0.0 or brightness_range[0] > brightness_range[1]:
            raise ValueError("brightness_range must be positive with min <= max.")
        if contrast_range[0] <= 0.0 or contrast_range[0] > contrast_range[1]:
            raise ValueError("contrast_range must be positive with min <= max.")
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.wb_jitter = wb_jitter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self._adjust_lighting(x)
        return x

    def _adjust_lighting(self, x: np.ndarray) -> np.ndarray:
        """Simulate brighter/dimmer rooms and warm/cool webcam auto-white-balance shifts."""
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        # Mild per-channel gain preserves banknote hue while tolerating LED/daylight tint shifts
        wb_gains = np.random.uniform(1.0 - self.wb_jitter, 1.0 + self.wb_jitter, size=(1, 1, 3))

        img_f = x.astype(np.float32)
        mean = np.mean(img_f, axis=(0, 1), keepdims=True)
        out = ((img_f - mean) * contrast_factor + mean * brightness_factor) * wb_gains
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_range={self.brightness_range}, "
            f"contrast_range={self.contrast_range}, wb_jitter={self.wb_jitter}, p={self.p})"
        )


class RandomMildBlur(BaseTransform):
    """Occasionally apply light Gaussian blur to mimic handheld webcam motion or autofocus lag."""

    def __init__(self, p: float = 0.25) -> None:
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            k = int(np.random.choice([3, 5]))
            return cv2.GaussianBlur(x, (k, k), sigmaX=0.0)
        return x


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
        interp = cv2.INTER_AREA if (self.size < h or self.size < w) else cv2.INTER_LINEAR
        return cv2.resize(x, (self.size, self.size), interpolation=interp)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ToTensorNormalize(BaseTransform):
    """Convert uint8 [0, 255] image (H, W, C) to float32 [0.0, 1.0] PyTorch tensor (C, H, W)."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        arr = x.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr))


class Flatten1D(BaseTransform):
    """Flatten a 3D image tensor (C, H, W) into a 1D vector (C * H * W,) for MultiClassImageMLP."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()


def build_augmentation_pipeline() -> Compose:
    """Build domain-randomized augmentation pipeline tailored for cropped Vietnamese banknotes."""
    return Compose([
        RandomSyntheticBackground(bill_scale=(0.72, 0.96), inner_crop=(0.88, 1.0), p=0.25),
        RandomResizedCrop(scale=(0.68, 1.0), p=0.60),
        RandomRotation(max_angle=20.0, p=0.60),
        RandomBrightnessContrast(brightness_range=(0.60, 1.75), contrast_range=(0.75, 1.35), wb_jitter=0.10, p=0.85),
        RandomHorizontalFlip(p=0.50),
        RandomMildBlur(p=0.25),
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
