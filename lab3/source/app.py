"""Practice Lab #03 — Interactive Multi-Class Vietnamese Currency Image & Video Processing System.

End-to-end Streamlit dashboard integrating:
  - Input Sources : File Mode (Static Image + Video .mp4) & Real-time Mode (Webcam / CCTV Stream)
  - Module 1      : Traditional OpenCV operations from Lab #01 (Color, Geometric, Image Smoothing)
  - Module 2      : Multi-Class AI Prediction (MultiClassImageMLP / CurrencyCNN, C = 6 VND classes)
  - Export        : "Snapshot & Export" downloads for annotated images (.png) and videos (.mp4)

Run locally from `lab3/source/`:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from src.config import CLASS_DISPLAY_NAMES, get_device
from src.img_processing import (
    COLOR_MODES,
    FLIP_MODES,
    INTERPOLATION_MODES,
    SMOOTH_METHODS,
    ImageProcessingParams,
    apply_image_pipeline,
)
from src.inference import (
    CurrencyInferenceEngine,
    PredictionResult,
    _open_video_writer,
    draw_prediction_overlay,
    encode_image_png_bytes,
    process_video_file,
    smooth_probabilities_ema,
)


# ---------------------------------------------------------------------------
# Paths, Page Configuration & Default Filter State
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(__file__).resolve().parent
SEARCH_CKPT_DIRS = (SOURCE_DIR / "outputs", SOURCE_DIR / "checkpoints", SOURCE_DIR)
SAMPLE_DATA_DIR = SOURCE_DIR.parent / "data" / "vietnamese_currency_6classes"
EXPORT_DIR = SOURCE_DIR / "exports"

# Default slider values used both at startup and by the "Reset All Filters" button
DEFAULT_FILTER_STATE: dict[str, object] = {
    "color_mode": "original",
    "contrast": 1.0,
    "brightness": 0,
    "gamma": 1.0,
    "angle": 0.0,
    "scale": 1.0,
    "tx": 0,
    "ty": 0,
    "flip_mode": "none",
    "interpolation": "bilinear",
    "smooth_method": "none",
    "kernel_size": 3,
    "sigma": 1.0,
}


def init_session_state() -> None:
    """Ensure all Module 1 control keys and export buffers exist in Streamlit session_state."""
    for key, default_val in DEFAULT_FILTER_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    # Buffers holding latest video export & real-time live snapshot across Streamlit reruns
    st.session_state.setdefault("video_export_bytes", None)
    st.session_state.setdefault("video_export_stats", None)
    st.session_state.setdefault("video_export_name", "processed_currency.mp4")
    st.session_state.setdefault("live_last_result", None)
    st.session_state.setdefault("live_last_orig", None)
    st.session_state.setdefault("live_video_bytes", None)


def reset_module1_filters() -> None:
    """Callback that resets every Module 1 OpenCV slider/selector to its identity default."""
    for key, default_val in DEFAULT_FILTER_STATE.items():
        st.session_state[key] = default_val


def discover_checkpoints() -> list[Path]:
    """Scan local outputs/ and checkpoints/ folders for trained .pt model files."""
    found: list[Path] = []
    seen: set[Path] = set()
    for folder in SEARCH_CKPT_DIRS:
        if folder.is_dir():
            for ckpt in sorted(folder.glob("*.pt")):
                resolved = ckpt.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(ckpt)
    return found


@st.cache_resource(show_spinner="Loading Multi-Class AI checkpoint...")
def get_inference_engine(checkpoint_path: str, device_str: str) -> CurrencyInferenceEngine:
    """Cache the loaded PyTorch model in memory so slider updates stay instantaneous."""
    return CurrencyInferenceEngine(checkpoint_path, device=device_str)


def build_probability_figure(probabilities: dict[str, float], top1_display: str) -> plt.Figure:
    """Create a horizontal Matplotlib bar chart visualizing class probability distribution (%)."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ["#2ECC71" if lbl == top1_display else "#5DADE2" for lbl in labels]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.62, edgecolor="#2C3E50", linewidth=0.6)

    ax.set_yticks(y_pos, labels=labels, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()  # Display 10k -> 500k cleanly from top to bottom
    ax.set_xlim(0, 112)
    ax.set_xlabel("Probability (%)", fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate exact percentage score next to every bar
    for bar, val in zip(bars, values):
        ax.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1B2631",
        )

    fig.tight_layout()
    return fig


def build_frame_counts_figure(label_counts: dict[str, int]) -> plt.Figure:
    """Create a bar chart showing predicted frame counts per denomination across a video."""
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, counts, color="#3498DB", width=0.55, edgecolor="#2C3E50", linewidth=0.6)
    ax.set_xticks(x_pos, labels=labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Frames", fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            cnt + max(counts) * 0.02,
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def render_prediction_panel(result: PredictionResult, model_label: str) -> None:
    """Display Module 2 KPIs (Top-1 label, confidence %, timing breakdown) and probability chart."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-1 Prediction", result.top1_display)
    c2.metric("Confidence Score", f"{result.confidence_pct:.2f}%")
    c3.metric("Total Time", f"{result.total_ms:.2f} ms")
    c4.metric(
        "Breakdown (M1 / AI)",
        f"{result.preprocess_ms:.1f} / {result.inference_ms:.1f} ms",
    )

    # Confidence advisory for live/real-world frames
    if result.confidence_pct < 50.0:
        st.warning(
            "Low confidence (< 50%). Try bringing the banknote closer, centering it in the ROI, "
            "or adjusting lighting/filters."
        )

    st.caption(f"Active Model: `{model_label}` — Probability Distribution across all 6 VND classes:")
    fig = build_probability_figure(result.probabilities, result.top1_display)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def save_local_export(file_bytes: bytes, filename: str) -> Path:
    """Persist an exported snapshot or video inside lab3/source/exports/ with timestamp."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / filename
    out_path.write_bytes(file_bytes)
    return out_path


# ---------------------------------------------------------------------------
# Sidebar: Model Selection + Module 1 Interactive OpenCV Controls
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[CurrencyInferenceEngine | None, str, ImageProcessingParams, float | None]:
    """Render sidebar controls for Module 2 model loading and Module 1 OpenCV parameters."""
    st.sidebar.title("System Controls")

    # === Section A: Module 2 Model Checkpoint Selection ===
    st.sidebar.header("Module 2 — AI Classifier")
    device_options = [get_device()]
    if "cpu" not in device_options:
        device_options.append("cpu")
    selected_device = st.sidebar.selectbox("Inference Device", options=device_options, index=0)

    ckpts = discover_checkpoints()
    engine: CurrencyInferenceEngine | None = None
    active_model_label = "No checkpoint loaded"

    ckpt_source = st.sidebar.radio(
        "Checkpoint Source",
        options=("Local outputs/ (.pt)", "Upload .pt file"),
        horizontal=True,
    )

    if ckpt_source == "Local outputs/ (.pt)":
        if ckpts:
            # Prefer best_cnn.pt by default when both CNN and MLP checkpoints exist
            labels = [f"{p.parent.name}/{p.name}" for p in ckpts]
            default_idx = next((i for i, p in enumerate(ckpts) if "cnn" in p.name.lower()), 0)
            chosen_idx = st.sidebar.selectbox(
                "Trained Model (.pt)",
                options=range(len(ckpts)),
                format_func=lambda i: labels[i],
                index=default_idx,
            )
            chosen_path = ckpts[chosen_idx]
            engine = get_inference_engine(str(chosen_path), selected_device)
            active_model_label = f"{chosen_path.name} ({engine.model_type.upper()}, {engine.img_size}x{engine.img_size})"
        else:
            st.sidebar.info(
                "No `.pt` checkpoint found in `outputs/` or `checkpoints/`. "
                "Run `train.ipynb` (Colab/Local) and copy `best_cnn.pt` / `best_mlp.pt` into `outputs/`, "
                "or switch to **Upload .pt file**."
            )
    else:
        uploaded_ckpt = st.sidebar.file_uploader("Upload trained `.pt` checkpoint", type=["pt"])
        if uploaded_ckpt is not None:
            # Persist uploaded checkpoint to a deterministic temp file for caching
            tmp_ckpt = Path(tempfile.gettempdir()) / f"lab3_uploaded_{uploaded_ckpt.name}"
            if not tmp_ckpt.exists() or tmp_ckpt.stat().st_size != uploaded_ckpt.size:
                tmp_ckpt.write_bytes(uploaded_ckpt.getvalue())
            engine = get_inference_engine(str(tmp_ckpt), selected_device)
            active_model_label = f"{uploaded_ckpt.name} ({engine.model_type.upper()}, {engine.img_size}x{engine.img_size})"

    if engine is not None:
        acc_meta = engine.meta.get("test_accuracy_pct")
        acc_txt = f" | Test Acc: **{acc_meta:.2f}%**" if isinstance(acc_meta, (int, float)) else ""
        st.sidebar.success(
            f"Loaded **{engine.model_type.upper()}** (`{engine.img_size}×{engine.img_size}`){acc_txt}"
        )

    # Optional center ROI guide (especially helpful for live Webcam currency placement)
    use_roi = st.sidebar.checkbox("Enable Center ROI Guide", value=False)
    roi_ratio: float | None = None
    if use_roi:
        roi_ratio = float(
            st.sidebar.slider(
                "Center ROI Size Ratio",
                min_value=0.35,
                max_value=0.95,
                value=0.70,
                step=0.05,
            )
        )

    st.sidebar.divider()

    # === Section B: Module 1 Traditional OpenCV Controls (Lab #01) ===
    st.sidebar.header("Module 1 — OpenCV (Lab #01)")
    st.sidebar.button("Reset All Filters", on_click=reset_module1_filters, use_container_width=True)

    with st.sidebar.expander("1. Color Transformation", expanded=True):
        color_mode = st.selectbox("Color Mode", options=COLOR_MODES, key="color_mode")
        contrast = st.slider("Contrast (alpha)", 0.2, 3.0, step=0.05, key="contrast")
        brightness = st.slider("Brightness (beta)", -100, 100, step=2, key="brightness")
        gamma = st.slider("Gamma (non-linear)", 0.2, 3.0, step=0.05, key="gamma")

    with st.sidebar.expander("2. Geometric Transformation", expanded=False):
        angle = st.slider("Rotation Angle (deg)", -180.0, 180.0, step=1.0, key="angle")
        scale = st.slider("Scale / Zoom", 0.5, 2.0, step=0.05, key="scale")
        tx = st.slider("Translate X (px)", -150, 150, step=2, key="tx")
        ty = st.slider("Translate Y (px)", -150, 150, step=2, key="ty")
        flip_mode = st.selectbox("Flip", options=FLIP_MODES, key="flip_mode")
        interpolation = st.selectbox("Interpolation", options=INTERPOLATION_MODES, key="interpolation")

    with st.sidebar.expander("3. Image Smoothing", expanded=False):
        smooth_method = st.selectbox("Smoothing Algorithm", options=SMOOTH_METHODS, key="smooth_method")
        kernel_size = st.slider("Kernel Size (odd)", 3, 21, step=2, key="kernel_size")
        sigma = st.slider("Sigma (Gaussian / Bilateral)", 0.1, 5.0, step=0.1, key="sigma")

    params = ImageProcessingParams(
        color_mode=str(color_mode),
        contrast=float(contrast),
        brightness=int(brightness),
        gamma=float(gamma),
        angle=float(angle),
        scale=float(scale),
        tx=int(tx),
        ty=int(ty),
        flip_mode=str(flip_mode),
        interpolation=str(interpolation),
        smooth_method=str(smooth_method),
        kernel_size=int(kernel_size),
        sigma=float(sigma),
    )
    return engine, active_model_label, params, roi_ratio


# ---------------------------------------------------------------------------
# Tab 1: File Mode — Static Image Processing & Multi-Class Prediction
# ---------------------------------------------------------------------------

def render_image_tab(
    engine: CurrencyInferenceEngine | None,
    model_label: str,
    params: ImageProcessingParams,
    roi_ratio: float | None,
) -> None:
    """Interactive Static Image mode: Original vs. Processed + AI Prediction + Snapshot & Export."""
    st.subheader("File Mode — Static Image (.jpg / .png)")

    input_choice = st.radio(
        "Image Input",
        options=("Upload Image", "Pick Sample from Local Dataset"),
        horizontal=True,
    )

    orig_rgb: np.ndarray | None = None
    source_stem = "snapshot"

    if input_choice == "Upload Image":
        uploaded_img = st.file_uploader(
            "Upload a Vietnamese banknote image",
            type=["jpg", "jpeg", "png"],
            key="static_image_uploader",
        )
        if uploaded_img is not None:
            with Image.open(uploaded_img) as pil_im:
                orig_rgb = np.array(pil_im.convert("RGB"))
            source_stem = Path(uploaded_img.name).stem
    else:
        if SAMPLE_DATA_DIR.is_dir():
            class_dirs = sorted([d for d in SAMPLE_DATA_DIR.iterdir() if d.is_dir()])
            chosen_cls = st.selectbox(
                "Sample Denomination",
                options=class_dirs,
                format_func=lambda d: CLASS_DISPLAY_NAMES.get(d.name, d.name),
            )
            sample_imgs = sorted(chosen_cls.glob("*.png"))[:25]
            if sample_imgs:
                chosen_file = st.selectbox(
                    "Sample Image File",
                    options=sample_imgs,
                    format_func=lambda p: p.name,
                )
                with Image.open(chosen_file) as pil_im:
                    orig_rgb = np.array(pil_im.convert("RGB"))
                source_stem = chosen_file.stem
        else:
            st.info(f"Local sample folder not found at `{SAMPLE_DATA_DIR}`. Please use **Upload Image**.")

    if orig_rgb is None:
        st.info("Upload an image (or select a sample) to start Module 1 + Module 2 processing.")
        return

    # Run either full Module 1 + Module 2 pipeline or Module 1 only when no checkpoint is loaded
    if engine is not None:
        result = engine.predict_frame(orig_rgb, params=params, roi_ratio=roi_ratio)
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(orig_rgb, use_container_width=True)
        with col_proc:
            show_overlay = st.checkbox("Show AI Overlay on Processed Image", value=True, key="img_overlay_chk")
            display_img = result.annotated_rgb if show_overlay else result.processed_rgb
            st.markdown("**Processed Image (Module 1 → Module 2)**")
            st.image(display_img, use_container_width=True)

        st.divider()
        st.markdown("### Module 2 — Multi-Class AI Prediction")
        render_prediction_panel(result, model_label)

        # Snapshot & Export controls (PNG download + local disk export)
        st.divider()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{source_stem}_{result.top1_label}_{ts}.png"
        png_bytes = encode_image_png_bytes(result.annotated_rgb)

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.download_button(
                label="Snapshot & Export (Download PNG)",
                data=png_bytes,
                file_name=export_name,
                mime="image/png",
                type="primary",
                use_container_width=True,
            )
        with exp_c2:
            if st.button("Save Snapshot to local `exports/`", use_container_width=True):
                saved = save_local_export(png_bytes, export_name)
                st.success(f"Saved snapshot to `{saved}`")
    else:
        # Module 1 preview remains fully functional even before loading a .pt checkpoint
        t0 = time.perf_counter()
        processed = apply_image_pipeline(orig_rgb, params)
        m1_ms = (time.perf_counter() - t0) * 1000.0

        col_orig, col_proc = st.columns(2)
        col_orig.markdown("**Original Image**")
        col_orig.image(orig_rgb, use_container_width=True)
        col_proc.markdown(f"**Processed Image (Module 1 — {m1_ms:.2f} ms)**")
        col_proc.image(processed, use_container_width=True)

        png_bytes = encode_image_png_bytes(processed)
        st.download_button(
            label="Snapshot & Export Processed Image (PNG)",
            data=png_bytes,
            file_name=f"{source_stem}_processed.png",
            mime="image/png",
        )


# ---------------------------------------------------------------------------
# Tab 2: File Mode — Video Processing (.mp4) & Annotated Video Export
# ---------------------------------------------------------------------------

def render_video_tab(
    engine: CurrencyInferenceEngine | None,
    model_label: str,
    params: ImageProcessingParams,
    roi_ratio: float | None,
) -> None:
    """Video File Mode: upload .mp4, process frame-by-frame with Module 1 + Module 2, and export .mp4."""
    st.subheader("File Mode — Video Processing & Export (.mp4)")

    if engine is None:
        st.warning("Please load a trained `.pt` checkpoint in the sidebar to run Video AI prediction.")
        return

    uploaded_vid = st.file_uploader(
        "Upload a banknote video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    c_opt1, c_opt2 = st.columns([2, 1])
    with c_opt1:
        frame_stride = st.slider(
            "AI Inference Frame Stride (1 = every frame, 2–5 = faster on long videos)",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )
    with c_opt2:
        run_btn = st.button(
            "Process Video",
            type="primary",
            disabled=(uploaded_vid is None),
            use_container_width=True,
        )

    if uploaded_vid is not None:
        with st.expander("Preview Original Uploaded Video", expanded=False):
            st.video(uploaded_vid.getvalue())

    if run_btn and uploaded_vid is not None:
        progress_bar = st.progress(0, text="Processing video frames through Module 1 → Module 2...")

        def on_progress(done: int, total: int) -> None:
            pct = min(100, int(round(100.0 * done / max(total, 1))))
            progress_bar.progress(pct, text=f"Processed frame {done}/{total} ({pct}%)")

        # Save uploaded bytes to a temporary file so OpenCV VideoCapture can read it
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            in_path = tmp_dir / uploaded_vid.name
            out_name = f"{Path(uploaded_vid.name).stem}_processed_{datetime.now().strftime('%H%M%S')}.mp4"
            out_path = tmp_dir / out_name
            in_path.write_bytes(uploaded_vid.getvalue())

            stats = process_video_file(
                input_path=in_path,
                output_path=out_path,
                engine=engine,
                params=params,
                frame_stride=frame_stride,
                roi_ratio=roi_ratio,
                progress_cb=on_progress,
            )
            st.session_state["video_export_bytes"] = out_path.read_bytes()
            st.session_state["video_export_stats"] = stats
            st.session_state["video_export_name"] = out_name

        progress_bar.progress(100, text="Video processing complete!")

    # Persist processed video player, summary stats, and download button across reruns
    if st.session_state["video_export_bytes"] is not None and st.session_state["video_export_stats"] is not None:
        stats = st.session_state["video_export_stats"]
        vid_bytes = st.session_state["video_export_bytes"]
        vid_name = st.session_state["video_export_name"]

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dominant Prediction", stats["dominant_label"])
        m2.metric("Processed Frames", f"{stats['processed_frames']}")
        m3.metric("Avg Latency", f"{stats['avg_ms_per_frame']:.2f} ms/frame")
        m4.metric("Codec / FPS", f"{stats['codec']} / {stats['fps']:.1f}")

        st.markdown(f"**Processed & Annotated Video** (`{model_label}`)")
        st.video(vid_bytes)

        # Show frame distribution by predicted denomination + final frame probability chart
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Predicted Frame Count per Denomination**")
            if stats["label_counts"]:
                cnt_fig = build_frame_counts_figure(stats["label_counts"])
                st.pyplot(cnt_fig, use_container_width=True)
                plt.close(cnt_fig)
        with ch2:
            if stats["last_probabilities"]:
                st.markdown("**Final Frame Probability Distribution (%)**")
                prob_fig = build_probability_figure(stats["last_probabilities"], stats["dominant_label"])
                st.pyplot(prob_fig, use_container_width=True)
                plt.close(prob_fig)

        # Snapshot & Export controls for the processed video
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="Snapshot & Export Video (Download .mp4)",
                data=vid_bytes,
                file_name=vid_name,
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )
        with dl2:
            if st.button("Save Processed Video to local `exports/`", use_container_width=True):
                saved = save_local_export(vid_bytes, vid_name)
                st.success(f"Saved processed video to `{saved}`")


# ---------------------------------------------------------------------------
# Tab 3: Real-time Mode (Bonus Point) — Live Webcam / CCTV Stream
# ---------------------------------------------------------------------------

def render_realtime_tab(
    engine: CurrencyInferenceEngine | None,
    model_label: str,
    params: ImageProcessingParams,
    roi_ratio: float | None,
) -> None:
    """Real-time Mode (Bonus): stream live Webcam or RTSP/HTTP stream with Module 1 + Module 2."""
    st.subheader("Real-time Mode (Bonus) — Live Webcam / CCTV Stream")

    if engine is None:
        st.warning("Please load a trained `.pt` checkpoint in the sidebar to run Real-time AI prediction.")
        return

    src_c1, src_c2, src_c3, src_c4 = st.columns([2, 1, 1, 1])
    with src_c1:
        stream_mode = st.radio(
            "Stream Source",
            options=("Local Webcam", "CCTV / Stream URL (RTSP/HTTP)"),
            horizontal=True,
        )
    with src_c2:
        if stream_mode == "Local Webcam":
            cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
            capture_source: int | str = int(cam_index)
        else:
            capture_source = st.text_input("Stream URL", value="rtsp://127.0.0.1:8554/stream")
    with src_c3:
        use_banknote_roi = st.checkbox("Banknote ROI (1.9:1)", value=True)
    with src_c4:
        record_clip = st.checkbox("Record Live (.mp4)", value=False)

    # Effective ROI settings for live stream (defaults to 1.9:1 horizontal banknote box)
    live_roi_ratio = roi_ratio if roi_ratio is not None else (0.58 if use_banknote_roi else None)
    live_roi_aspect = 1.9 if use_banknote_roi else None

    st.caption(
        "Tip: Hold the Vietnamese banknote so it fills the centered green rectangle — "
        "Temporal EMA smoothing stabilizes live predictions across frames!"
    )

    run_live = st.toggle("Start Live Stream", value=False, key="live_stream_toggle")

    # Reserve stable UI slots updated in-place during the live streaming loop
    col_orig, col_proc = st.columns(2)
    orig_placeholder = col_orig.empty()
    proc_placeholder = col_proc.empty()
    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()

    if run_live:
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            st.error(
                f"Cannot open stream source `{capture_source}`. On macOS, verify Camera permissions for "
                "Terminal / VS Code in System Settings → Privacy & Security → Camera."
            )
            return

        writer = None
        tmp_rec_dir = None
        rec_path: Path | None = None
        prev_time = time.perf_counter()
        ema_probs: dict[str, float] | None = None

        try:
            if record_clip:
                tmp_rec_dir = tempfile.TemporaryDirectory()
                rec_path = Path(tmp_rec_dir.name) / f"webcam_recording_{datetime.now().strftime('%H%M%S')}.mp4"

            while st.session_state.get("live_stream_toggle", False):
                ok, frame_bgr = cap.read()
                if not ok:
                    st.warning("Stream ended or frame could not be read.")
                    break

                # Convert BGR -> RGB immediately before running Module 1 + Module 2
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = engine.predict_frame(
                    frame_rgb,
                    params=params,
                    roi_ratio=live_roi_ratio,
                    roi_aspect_ratio=live_roi_aspect,
                )

                # Smooth probabilities via EMA across consecutive frames to prevent jitter
                ema_probs = smooth_probabilities_ema(ema_probs, res.probabilities, alpha=0.40)
                smooth_top1 = max(ema_probs, key=ema_probs.get)
                smooth_conf = ema_probs[smooth_top1]
                smooth_annotated = draw_prediction_overlay(
                    res.processed_rgb,
                    top1_display=smooth_top1,
                    confidence_pct=smooth_conf,
                    total_ms=res.total_ms,
                    roi_box=res.roi_box,
                )
                res = PredictionResult(
                    processed_rgb=res.processed_rgb,
                    annotated_rgb=smooth_annotated,
                    top1_label=res.top1_label,
                    top1_display=smooth_top1,
                    confidence_pct=smooth_conf,
                    probabilities=ema_probs,
                    preprocess_ms=res.preprocess_ms,
                    inference_ms=res.inference_ms,
                    total_ms=res.total_ms,
                    roi_box=res.roi_box,
                )

                # Compute live end-to-end display FPS
                now = time.perf_counter()
                live_fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # Lazily open VideoWriter using exact live frame (width, height) when recording
                if record_clip and rec_path is not None and writer is None:
                    fh, fw = res.annotated_rgb.shape[:2]
                    writer, _ = _open_video_writer(rec_path, fps=15.0, frame_size=(fw, fh))
                if writer is not None:
                    writer.write(cv2.cvtColor(res.annotated_rgb, cv2.COLOR_RGB2BGR))

                # Save latest frame into session_state so Snapshot & Export works immediately on Stop
                st.session_state["live_last_orig"] = frame_rgb
                st.session_state["live_last_result"] = res

                # Refresh live UI slots without flickering the entire page
                orig_placeholder.image(frame_rgb, caption="Live Original Feed", use_container_width=True)
                proc_placeholder.image(
                    res.annotated_rgb,
                    caption=f"Processed + AI Overlay ({model_label})",
                    use_container_width=True,
                )

                with kpi_placeholder.container():
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Top-1 Prediction", res.top1_display)
                    k2.metric("Confidence", f"{res.confidence_pct:.1f}%")
                    k3.metric("Pipeline Time", f"{res.total_ms:.1f} ms")
                    k4.metric("Live FPS", f"{live_fps:.1f}")

                live_fig = build_probability_figure(res.probabilities, res.top1_display)
                chart_placeholder.pyplot(live_fig, use_container_width=True)
                plt.close(live_fig)
        finally:
            # Always release hardware camera and close video container cleanly
            cap.release()
            if writer is not None:
                writer.release()
            if rec_path is not None and rec_path.is_file() and rec_path.stat().st_size > 0:
                st.session_state["live_video_bytes"] = rec_path.read_bytes()
            if tmp_rec_dir is not None:
                tmp_rec_dir.cleanup()

    # When paused/stopped, render last captured live frame + Snapshot & Export download buttons
    last_res: PredictionResult | None = st.session_state.get("live_last_result")
    last_orig: np.ndarray | None = st.session_state.get("live_last_orig")
    if not run_live and last_res is not None and last_orig is not None:
        orig_placeholder.image(last_orig, caption="Last Captured Original Frame", use_container_width=True)
        proc_placeholder.image(
            last_res.annotated_rgb,
            caption="Last Captured Processed + AI Frame",
            use_container_width=True,
        )
        with kpi_placeholder.container():
            render_prediction_panel(last_res, model_label)

        st.divider()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"webcam_snapshot_{last_res.top1_label}_{ts}.png"
        snap_bytes = encode_image_png_bytes(last_res.annotated_rgb)

        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                label="Snapshot & Export Live Frame (PNG)",
                data=snap_bytes,
                file_name=snap_name,
                mime="image/png",
                type="primary",
                use_container_width=True,
            )
        with ex2:
            if st.session_state.get("live_video_bytes") is not None:
                st.download_button(
                    label="Export Recorded Live Video (.mp4)",
                    data=st.session_state["live_video_bytes"],
                    file_name=f"webcam_recording_{ts}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            elif st.button("Save Snapshot to local `exports/`", use_container_width=True):
                saved = save_local_export(snap_bytes, snap_name)
                st.success(f"Saved live snapshot to `{saved}`")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Configure Streamlit layout and dispatch across Image, Video, and Real-time tabs."""
    st.set_page_config(
        page_title="Lab #03 — Multi-Class VND Currency Vision System",
        page_icon="💵",
        layout="wide",
    )
    init_session_state()

    st.title("Interactive Multi-Class Vietnamese Currency Image & Video Processing System")
    st.caption(
        "Practice Lab #03 — End-to-End Pipeline: Input (Image / Video / Webcam) → "
        "Module 1 (OpenCV Lab #01) → Module 2 (Multi-Class AI, 6 VND Denominations) → Snapshot & Export"
    )

    engine, model_label, params, roi_ratio = render_sidebar()

    tab_img, tab_vid, tab_live = st.tabs(
        [
            "File Mode: Static Image",
            "File Mode: Video (.mp4)",
            "Real-time Mode: Webcam / Stream (Bonus)",
        ]
    )

    with tab_img:
        render_image_tab(engine, model_label, params, roi_ratio)
    with tab_vid:
        render_video_tab(engine, model_label, params, roi_ratio)
    with tab_live:
        render_realtime_tab(engine, model_label, params, roi_ratio)


if __name__ == "__main__":
    main()
