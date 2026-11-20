from collections.abc import Sequence
import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Compose(BaseTransform):
    def __init__(self, transforms : Sequence[BaseTransform]):
        for t in transforms:
            if not isinstance(t, BaseTransform):
                raise TypeError(f"Transform must be an instance of BaseTransform, got {type(t)}")
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self) -> str:
        return "Compose(" + " -> ".join([repr(t) for t in self.transforms]) + ")"


# Data Augmentation Transforms

class RandomHorizontalFlip(BaseTransform):
    def __init__(self, p : float = 0.5):
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def __call__(self, x : np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return np.flip(x, axis=1).copy()   # Flip horizontally
        return x

class RandomRotation(BaseTransform):
    def __init__(self, max_angle: float = 15.0 , p: float = 0.5):
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        if max_angle < 0.0:
            raise ValueError("max_angle must be non-negative")
        self.p = p
        self.max_angle = max_angle

    def __call__(self, x : np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            return self.rotate_image(x, angle)
        return x
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

class RandomBrightnessContrast(BaseTransform):
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p: float = 0.5):
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1")
        if brightness_range[0] < 0.0 or brightness_range[1] > 2.0 or brightness_range[0] > brightness_range[1]:
            raise ValueError("brightness_range must be between 0 and 2 and the first element must be less than the second element")
        if contrast_range[0] < 0.0 or contrast_range[1] > 2.0 or contrast_range[0] > contrast_range[1]:
            raise ValueError("contrast_range must be between 0 and 2 and the first element must be less than the second element")
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, x : np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self._adjust_brightness_contrast(x)
        return x
    
    def _adjust_brightness_contrast(self, x : np.ndarray) -> np.ndarray:
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = np.mean(x, axis=(0,1), keepdims=True)
        x = (x - mean) * contrast_factor + mean * brightness_factor

        x = np.clip(x, 0, 255)
        return x.astype(np.uint8)

class Resize(BaseTransform):
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
    
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class ToTensorNormalize(BaseTransform):
    def __init__(self):
        pass
    
    def __call__(self, x : np.ndarray) -> torch.Tensor:
        x = x.astype(np.float32) / 255.0
        if x.ndim == 2:
            x = np.stack([x,x,x], axis=0)
        else:
            x = np.transpose(x, (2,0,1))
        return torch.from_numpy(np.ascontiguousarray(x))

class Flatten1D(BaseTransform):
    def __init__(self):
        pass
    
    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        return x.flatten()

def build_augmentation_pipeline() -> Compose:
    compose = Compose([
        RandomRotation(),
        RandomBrightnessContrast(),
        RandomHorizontalFlip(),
    ])
    return compose

def build_preprocess_pipeline(img_size : int = 64) -> Compose:
    compose = Compose([
        Resize(img_size),
        ToTensorNormalize(),
        Flatten1D()
    ])
    return compose