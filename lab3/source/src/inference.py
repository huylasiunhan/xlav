"""End-to-end inference & video processing engine bridging Module 1 (OpenCV) and Module 2 (AI)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import torch

from src.config import CLASS_DISPLAY_NAMES, get_device
from src.img_processing import ImageProcessingParams, apply_image_pipeline
from src.trainer import load_checkpoint
from src.transform import build_preprocess_pipeline


@dataclass
class PredictionResult:
    """Structured output returned for a single image or video/webcam frame."""

    processed_rgb: np.ndarray  # Frame after Module 1 OpenCV filters (for Original vs. Processed UI)
    annotated_rgb: np.ndarray  # Processed frame with Top-1/confidence/latency overlay (for Snapshot/Video)
    top1_label: str  # Raw class folder key, e.g., "500000_VND"
    top1_display: str  # Human-readable label, e.g., "500,000 VND"
    confidence_pct: float  # Top-1 confidence score in [0, 100]
    probabilities: dict[str, float]  # All-class probability distribution (%) for bar chart
    preprocess_ms: float  # Module 1 OpenCV processing time (milliseconds)
    inference_ms: float  # Module 2 AI forward pass time (milliseconds)
    total_ms: float  # Combined end-to-end latency (milliseconds)
    roi_box: tuple[int, int, int, int] | None = None  # Optional (x1, y1, x2, y2) center ROI box


def compute_center_roi_box(
    height: int,
    width: int,
    roi_ratio: float,
    aspect_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    """Compute centered rectangular ROI coordinates (x1, y1, x2, y2) for guided webcam capture.

    When aspect_ratio (width / height, e.g. 1.9 for VND polymer banknotes) is provided,
    the box matches a horizontal banknote rectangle so room background borders are excluded.
    """
    if not (0.1 <= roi_ratio <= 1.0):
        raise ValueError(f"roi_ratio must be in [0.1, 1.0], got {roi_ratio}.")
    box_w = max(16, int(round(width * roi_ratio)))
    if aspect_ratio is not None and aspect_ratio > 0.0:
        box_h = max(16, int(round(box_w / aspect_ratio)))
        if box_h > height:
            box_h = max(16, int(round(height * roi_ratio)))
            box_w = min(width, max(16, int(round(box_h * aspect_ratio))))
    else:
        box_h = max(16, int(round(height * roi_ratio)))
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = min(width, x1 + box_w)
    y2 = min(height, y1 + box_h)
    return x1, y1, x2, y2


def smooth_probabilities_ema(
    prev_probs: dict[str, float] | None,
    curr_probs: dict[str, float],
    alpha: float = 0.40,
) -> dict[str, float]:
    """Apply Exponential Moving Average (EMA) across live frames to stabilize webcam predictions."""
    if prev_probs is None or set(prev_probs.keys()) != set(curr_probs.keys()):
        return dict(curr_probs)
    return {
        k: float(alpha * curr_probs[k] + (1.0 - alpha) * prev_probs[k])
        for k in curr_probs
    }


def draw_prediction_overlay(
    img_rgb: np.ndarray,
    top1_display: str,
    confidence_pct: float,
    total_ms: float,
    roi_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Render a translucent Top-1 prediction banner and optional ROI guide box onto an RGB image."""
    canvas = img_rgb.copy()
    h, w = canvas.shape[:2]

    # Draw optional centered ROI box so users know where to hold the banknote on webcam
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 120), 2)

    # Scale banner and font smoothly across small (192x144) and HD (1280x720) frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, w / 720.0))
    thickness = max(1, int(round(font_scale * 2.2)))

    line1 = f"{top1_display} ({confidence_pct:.1f}%)"
    line2 = f"Time: {total_ms:.1f} ms"

    # Measure exact text sizes so the translucent banner wraps tightly around both lines
    (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale * 0.8, max(1, thickness - 1))
    pad = max(6, int(round(w * 0.015)))
    box_w = min(w - 2 * pad, max(tw1, tw2) + 2 * pad)
    box_h = min(h - 2 * pad, th1 + th2 + 3 * pad)

    # Blend dark rectangle onto top-left corner using alpha transparency
    overlay = canvas.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 24, 32), -1)
    canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)

    # Color-code confidence: green >= 75%, amber 50-75%, red < 50%
    text_color = (80, 240, 140) if confidence_pct >= 75.0 else (
        (255, 200, 60) if confidence_pct >= 50.0 else (255, 95, 95)
    )
    cv2.putText(
        canvas,
        line1,
        (pad * 2, pad + th1 + pad // 2),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        line2,
        (pad * 2, pad + th1 + th2 + int(pad * 1.4)),
        font,
        font_scale * 0.8,
        (230, 235, 245),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    return canvas


def encode_image_png_bytes(img_rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 image into PNG bytes ready for Streamlit download ('Snapshot & export')."""
    # OpenCV expects BGR when encoding PNG/JPEG buffers
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image snapshot to PNG.")
    return buffer.tobytes()


class CurrencyInferenceEngine:
    """Unified single-frame inference wrapper combining Module 1 OpenCV and Module 2 AI model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.device_str = device or get_device()
        self.device = torch.device(self.device_str)

        # Load trained model + metadata (.pt) in eval() mode
        self.model, self.meta = load_checkpoint(checkpoint_path, device=self.device_str)
        self.model_type = str(self.meta["model_type"])
        self.img_size = int(self.meta["img_size"])
        self.class_names = list(self.meta["class_names"])

        # Build matching deterministic preprocessing pipeline (flatten=True for MLP, False for CNN)
        self.preprocess = build_preprocess_pipeline(
            img_size=self.img_size,
            flatten=(self.model_type == "mlp"),
        )

    @torch.no_grad()
    def predict_frame(
        self,
        img_rgb: np.ndarray,
        params: ImageProcessingParams | None = None,
        roi_ratio: float | None = None,
        roi_aspect_ratio: float | None = None,
    ) -> PredictionResult:
        """Run Module 1 -> Preprocess -> Module 2 on one RGB uint8 image/frame and measure latency."""
        active_params = params if params is not None else ImageProcessingParams()

        # --- Stage 1: Module 1 traditional OpenCV processing ---
        t0 = time.perf_counter()
        processed = apply_image_pipeline(img_rgb, active_params)
        t1 = time.perf_counter()

        # Optional center ROI crop (helps webcam mode focus strictly on the held banknote)
        roi_box: tuple[int, int, int, int] | None = None
        if roi_ratio is not None and 0.1 <= roi_ratio < 1.0:
            h, w = processed.shape[:2]
            roi_box = compute_center_roi_box(h, w, roi_ratio, aspect_ratio=roi_aspect_ratio)
            x1, y1, x2, y2 = roi_box
            crop_rgb = processed[y1:y2, x1:x2]
        else:
            crop_rgb = processed

        # --- Stage 2: Preprocess & Module 2 Multi-Class AI inference ---
        # unsqueeze(0) adds batch dimension B=1: (C,H,W)->(1,C,H,W) or (D,)->(1,D)
        input_tensor = self.preprocess(crop_rgb).unsqueeze(0).to(self.device)
        probs_1d = self.model.predict_proba(input_tensor).squeeze(0).cpu().numpy()
        t2 = time.perf_counter()

        # Extract Top-1 prediction and full probability distribution (%)
        top1_idx = int(np.argmax(probs_1d))
        top1_label = self.class_names[top1_idx]
        top1_display = CLASS_DISPLAY_NAMES.get(top1_label, top1_label)
        confidence_pct = float(probs_1d[top1_idx] * 100.0)
        probabilities = {
            CLASS_DISPLAY_NAMES.get(name, name): float(prob * 100.0)
            for name, prob in zip(self.class_names, probs_1d)
        }

        preprocess_ms = (t1 - t0) * 1000.0
        inference_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        # Render visual overlay for display and Snapshot/Video export
        annotated = draw_prediction_overlay(
            processed,
            top1_display=top1_display,
            confidence_pct=confidence_pct,
            total_ms=total_ms,
            roi_box=roi_box,
        )

        return PredictionResult(
            processed_rgb=processed,
            annotated_rgb=annotated,
            top1_label=top1_label,
            top1_display=top1_display,
            confidence_pct=confidence_pct,
            probabilities=probabilities,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
            roi_box=roi_box,
        )


def _open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    """Open a VideoWriter trying browser-friendly H.264 ('avc1') first, then falling back to 'mp4v'."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Note: OpenCV VideoWriter requires (width, height), opposite of NumPy (H, W)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not initialize VideoWriter for {output_path}.")


def process_video_file(
    input_path: str | Path,
    output_path: str | Path,
    engine: CurrencyInferenceEngine,
    params: ImageProcessingParams | None = None,
    frame_stride: int = 1,
    roi_ratio: float | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Process an uploaded video frame-by-frame through Module 1 + Module 2 and export an annotated .mp4."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}.")

    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {in_path}")

    # Read source video metadata with safe fallbacks
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    active_params = params if params is not None else ImageProcessingParams()
    writer, used_codec = _open_video_writer(out_path, fps=fps, frame_size=(width, height))

    processed_frames = 0
    total_latency_ms = 0.0
    label_counts: Counter[str] = Counter()
    last_result: PredictionResult | None = None
    last_probabilities: dict[str, float] = {}

    # try/finally guarantees both VideoCapture and VideoWriter close cleanly so the .mp4 is never corrupted
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert OpenCV BGR -> RGB immediately to avoid swapping red/blue banknote colors
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if last_result is None or (processed_frames % frame_stride == 0):
                # Run full Module 1 + Module 2 inference on sampled frames
                last_result = engine.predict_frame(
                    frame_rgb,
                    params=active_params,
                    roi_ratio=roi_ratio,
                )
                out_rgb = last_result.annotated_rgb
                total_latency_ms += last_result.total_ms
                last_probabilities = last_result.probabilities
            else:
                # On skipped frames (when stride > 1), apply Module 1 and reuse latest prediction overlay
                t0 = time.perf_counter()
                processed_rgb = apply_image_pipeline(frame_rgb, active_params)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                out_rgb = draw_prediction_overlay(
                    processed_rgb,
                    top1_display=last_result.top1_display,
                    confidence_pct=last_result.confidence_pct,
                    total_ms=elapsed_ms,
                    roi_box=last_result.roi_box,
                )
                total_latency_ms += elapsed_ms

            # Convert back RGB -> BGR right before writing to the output video stream
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            label_counts[last_result.top1_display] += 1
            processed_frames += 1

            if progress_cb is not None:
                progress_cb(processed_frames, max(total_frames, processed_frames))
    finally:
        cap.release()
        writer.release()

    avg_ms = total_latency_ms / max(processed_frames, 1)
    dominant_label = label_counts.most_common(1)[0][0] if label_counts else "N/A"

    return {
        "output_path": str(out_path),
        "codec": used_codec,
        "processed_frames": processed_frames,
        "fps": fps,
        "resolution": (width, height),
        "avg_ms_per_frame": avg_ms,
        "dominant_label": dominant_label,
        "label_counts": dict(label_counts),
        "last_probabilities": last_probabilities,
    }
