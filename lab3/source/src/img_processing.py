"""Module 1 (Lab #01): Traditional OpenCV image & video-frame processing operations.

Every transformation in this module follows a strict RGB contract:
    Input  : np.ndarray, RGB uint8, shape (H, W, 3)
    Output : np.ndarray, RGB uint8, shape (H, W, 3)
Keeping channel count and frame dimensions invariant ensures seamless compatibility
with both VideoWriter (.mp4 export) and the downstream Multi-Class AI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


# Valid option tuples shared with the Streamlit UI (Single Source of Truth)
COLOR_MODES: tuple[str, ...] = ("original", "grayscale", "hsv", "invert", "hist_eq")
FLIP_MODES: tuple[str, ...] = ("none", "horizontal", "vertical", "both")
INTERPOLATION_MODES: tuple[str, ...] = ("bilinear", "nearest")
SMOOTH_METHODS: tuple[str, ...] = ("none", "mean", "gaussian", "median", "bilateral")


@dataclass
class ImageProcessingParams:
    """Interactive parameter bundle controlling Module 1 OpenCV transformations."""

    # --- 1. Color transformation controls ---
    color_mode: str = "original"  # Color space / tonal mode (see COLOR_MODES)
    contrast: float = 1.0  # Linear gain alpha in [0.2, 3.0]
    brightness: int = 0  # Linear offset beta in [-100, 100]
    gamma: float = 1.0  # Non-linear gamma correction in [0.2, 3.0]

    # --- 2. Geometric transformation controls ---
    angle: float = 0.0  # Rotation angle in degrees [-180.0, 180.0]
    scale: float = 1.0  # Isotropic zoom factor in [0.5, 2.0]
    tx: int = 0  # Horizontal translation in pixels
    ty: int = 0  # Vertical translation in pixels
    flip_mode: str = "none"  # Mirroring mode (see FLIP_MODES)
    interpolation: str = "bilinear"  # Resampling filter (see INTERPOLATION_MODES)

    # --- 3. Image smoothing controls ---
    smooth_method: str = "none"  # Smoothing algorithm (see SMOOTH_METHODS)
    kernel_size: int = 3  # Filter window size (odd integer >= 3)
    sigma: float = 1.0  # Standard deviation for Gaussian / Bilateral filters

    def __post_init__(self) -> None:
        """Validate parameter ranges and normalize odd kernel_size immediately."""
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"Invalid color_mode '{self.color_mode}'. Choose from {COLOR_MODES}.")
        if self.flip_mode not in FLIP_MODES:
            raise ValueError(f"Invalid flip_mode '{self.flip_mode}'. Choose from {FLIP_MODES}.")
        if self.interpolation not in INTERPOLATION_MODES:
            raise ValueError(
                f"Invalid interpolation '{self.interpolation}'. Choose from {INTERPOLATION_MODES}."
            )
        if self.smooth_method not in SMOOTH_METHODS:
            raise ValueError(
                f"Invalid smooth_method '{self.smooth_method}'. Choose from {SMOOTH_METHODS}."
            )
        if self.contrast <= 0.0:
            raise ValueError(f"contrast must be > 0, got {self.contrast}.")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}.")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}.")

        # Guarantee kernel_size is always an odd integer >= 3 for Gaussian/Median filters
        k = max(3, int(self.kernel_size))
        self.kernel_size = k if (k % 2 == 1) else (k + 1)


@lru_cache(maxsize=64)
def _build_gamma_lut(gamma_rounded: float) -> np.ndarray:
    """Build and cache a 256-entry uint8 Look-Up Table for fast per-frame gamma correction."""
    inv_gamma = 1.0 / gamma_rounded
    # Map each 8-bit intensity [0..255] via (i / 255)^(1 / gamma) * 255
    indices = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((indices ** inv_gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
    return lut


def apply_gamma(img_rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Apply non-linear gamma correction using a cached OpenCV LUT for video speed."""
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}.")
    lut = _build_gamma_lut(round(float(gamma), 3))
    return cv2.LUT(img_rgb, lut)


def apply_color_transform(img_rgb: np.ndarray, params: ImageProcessingParams) -> np.ndarray:
    """Apply contrast/brightness, gamma correction, and color space / tonal conversion."""
    out = img_rgb.copy()

    # Step 1: Linear contrast (alpha) and brightness (beta) adjustment
    if params.contrast != 1.0 or params.brightness != 0:
        out = cv2.convertScaleAbs(out, alpha=params.contrast, beta=params.brightness)

    # Step 2: Non-linear gamma adjustment (fast LUT lookup)
    if params.gamma != 1.0:
        out = apply_gamma(out, params.gamma)

    # Step 3: Color-space / tonal transformation (always return 3-channel RGB uint8)
    if params.color_mode == "grayscale":
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        # Promote back to 3 channels so VideoWriter and AI inputs remain consistent
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif params.color_mode == "hsv":
        # Visualize HSV channels directly as a 3-channel tensor
        out = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    elif params.color_mode == "invert":
        # Bitwise photographic negative (255 - pixel)
        out = cv2.bitwise_not(out)
    elif params.color_mode == "hist_eq":
        # Equalize luminance channel (Y) in YCrCb to boost contrast while preserving banknote color
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return out


def apply_geometric_transform(img_rgb: np.ndarray, params: ImageProcessingParams) -> np.ndarray:
    """Apply combined Affine (rotate + scale + translate) and flip operations preserving (H, W, 3)."""
    out = img_rgb.copy()
    h, w = out.shape[:2]

    # Select resampling interpolation method (matches Lab #01 Nearest vs. Bilinear)
    interp = cv2.INTER_NEAREST if params.interpolation == "nearest" else cv2.INTER_LINEAR

    # Step 1: Fuse rotation, isotropic scaling, and translation into a single 2x3 Affine warp
    if params.angle != 0.0 or params.scale != 1.0 or params.tx != 0 or params.ty != 0:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, params.angle, params.scale)
        # Add translation offsets to the affine matrix's third column
        M[0, 2] += float(params.tx)
        M[1, 2] += float(params.ty)
        # Keep output frame size fixed at (w, h) so VideoWriter frames stay uniform
        out = cv2.warpAffine(
            out,
            M,
            (w, h),
            flags=interp,
            borderMode=cv2.BORDER_REFLECT,
        )

    # Step 2: Optional spatial mirroring (horizontal, vertical, or both)
    if params.flip_mode == "horizontal":
        out = cv2.flip(out, 1)
    elif params.flip_mode == "vertical":
        out = cv2.flip(out, 0)
    elif params.flip_mode == "both":
        out = cv2.flip(out, -1)

    return out


def apply_smoothing(img_rgb: np.ndarray, params: ImageProcessingParams) -> np.ndarray:
    """Apply spatial image smoothing / denoising filter (Mean, Gaussian, Median, or Bilateral)."""
    if params.smooth_method == "none":
        return img_rgb.copy()

    # Ensure kernel size is odd >= 3 even if params was mutated after __post_init__
    k = max(3, int(params.kernel_size))
    if k % 2 == 0:
        k += 1

    if params.smooth_method == "mean":
        return cv2.blur(img_rgb, (k, k))
    if params.smooth_method == "gaussian":
        return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=params.sigma)
    if params.smooth_method == "median":
        return cv2.medianBlur(img_rgb, k)
    if params.smooth_method == "bilateral":
        # Edge-preserving denoising: scales color & spatial sigma smoothly with user slider
        sigma_val = float(params.sigma * 25.0)
        return cv2.bilateralFilter(img_rgb, d=k, sigmaColor=sigma_val, sigmaSpace=sigma_val)

    return img_rgb.copy()


def apply_image_pipeline(img_rgb: np.ndarray, params: ImageProcessingParams) -> np.ndarray:
    """Run full Module 1 pipeline (Color -> Geometric -> Smoothing) on one RGB uint8 image/frame."""
    # Guard clauses: enforce strict (H, W, 3) uint8 RGB contract
    if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image array of shape (H, W, 3), got {getattr(img_rgb, 'shape', type(img_rgb))}."
        )
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Execute the 3 traditional OpenCV stages sequentially
    out = apply_color_transform(img_rgb, params)
    out = apply_geometric_transform(out, params)
    out = apply_smoothing(out, params)
    return out
