"""Visualization utilities covering all required Lab 2 charts and diagrams."""

from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.activation import ACTIVATION_REGISTRY
from src.model import ImageMLP
from src.trainer import ExperimentResult, TrainingHistory
from src.transform import BaseTransform


def _finalize_figure(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Optionally save figure at high DPI for the Lab 2 report."""
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_augmentation_samples(
    image_paths: list[str],
    labels: list[int],
    aug_pipeline: BaseTransform,
    num_per_class: int = 2,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a grid comparing Original vs. two Augmented views for both classes."""
    class_names = {0: "Negative (No Crack)", 1: "Positive (Crack)"}
    chosen: list[tuple[str, int]] = []
    for cls_id in (0, 1):
        cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls_id][:num_per_class]
        chosen.extend((image_paths[i], cls_id) for i in cls_indices)

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original Image", "Augmented Sample #1", "Augmented Sample #2"]
    for row_idx, (path, cls_id) in enumerate(chosen):
        with Image.open(path) as img:
            raw_rgb = np.array(img.convert("RGB"))
        aug_1 = aug_pipeline(raw_rgb.copy())
        aug_2 = aug_pipeline(raw_rgb.copy())

        for col_idx, arr in enumerate((raw_rgb, aug_1, aug_2)):
            ax = axes[row_idx, col_idx]
            ax.imshow(arr)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(class_names[cls_id], fontsize=10, fontweight="bold")

    fig.suptitle("Input Samples: Before vs. After Data Augmentation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize_figure(fig, save_path)


def plot_model_architecture(
    model: ImageMLP,
    save_path: str | None = None,
) -> plt.Figure:
    """Draw a clean flowchart diagram of ImageMLP dimensions and parameter counts."""
    specs = model.layer_specs()
    n_blocks = len(specs)
    fig, ax = plt.subplots(figsize=(3.0 * n_blocks, 4.2))
    ax.axis("off")

    colors = ["#DCEEFB", "#D5F5E3", "#D5F5E3", "#D5F5E3", "#FADBD8"]
    box_w, box_h, gap = 2.2, 1.5, 0.8

    for idx, spec in enumerate(specs):
        x0 = idx * (box_w + gap)
        y0 = 0.5
        face_color = colors[0] if idx == 0 else (colors[-1] if idx == n_blocks - 1 else "#D5F5E3")

        rect = patches.FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            linewidth=1.5,
            edgecolor="#2C3E50",
            facecolor=face_color,
        )
        ax.add_patch(rect)

        params_txt = f"Params: {spec['params']:,}" if spec["params"] > 0 else "Flattened 1D"
        label = f"{spec['name']}\nDim: {spec['dim']:,}\n{params_txt}"
        ax.text(
            x0 + box_w / 2.0,
            y0 + box_h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1B2631",
        )

        # Draw connecting arrow to the next block
        if idx < n_blocks - 1:
            ax.annotate(
                "",
                xy=(x0 + box_w + gap - 0.14, y0 + box_h / 2.0),
                xytext=(x0 + box_w + 0.14, y0 + box_h / 2.0),
                arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#2C3E50"},
            )

    ax.set_xlim(-0.4, n_blocks * (box_w + gap) - gap + 0.4)
    ax.set_ylim(0.0, 2.6)
    ax.set_title(
        f"ImageMLP Architecture Flowchart | Init: {model.init_method} | "
        f"Total Trainable Params: {model.count_parameters():,}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    return _finalize_figure(fig, save_path)


def plot_loss_curves(
    history: TrainingHistory,
    title: str = "Training & Validation Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Training vs. Validation Loss and Accuracy curves over epochs."""
    epochs = np.arange(1, len(history.train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: Loss curves
    ax_loss.plot(epochs, history.train_losses, label="Train Loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history.val_losses, label="Val Loss", marker="s", markersize=3)
    if history.best_epoch > 0:
        ax_loss.axvline(
            history.best_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best Epoch ({history.best_epoch})",
        )
    ax_loss.set_title("Loss over Epochs", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # Right panel: Accuracy (%) curves
    ax_acc.plot(epochs, history.train_accs, label="Train Acc (%)", marker="o", markersize=3)
    ax_acc.plot(epochs, history.val_accs, label="Val Acc (%)", marker="s", markersize=3)
    ax_acc.set_title("Accuracy (%) over Epochs", fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize_figure(fig, save_path)


def plot_initialization_comparison(
    results: dict[str, ExperimentResult],
    save_path: str | None = None,
) -> plt.Figure:
    """Compare training loss convergence (log scale) and validation accuracy across initializers."""
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for label, res in results.items():
        epochs = np.arange(1, len(res.history.train_losses) + 1)
        ax_loss.plot(epochs, res.history.train_losses, label=label, linewidth=2.0, marker="o", markersize=3)
        ax_val.plot(epochs, res.history.val_accs, label=label, linewidth=2.0, marker="s", markersize=3)

    # Log scale prevents large Random Normal losses from squashing Xavier/He curves
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training Loss Convergence (Log Scale)", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training BCE Loss (log)")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend()

    ax_val.set_title("Validation Accuracy (%) Comparison", fontweight="bold")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Accuracy (%)")
    ax_val.set_ylim(0, 105)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend()

    fig.suptitle("Weight Initialization Convergence Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize_figure(fig, save_path)


def plot_confusion_matrix_and_roc(
    metrics: dict[str, Any],
    title: str = "Test Set Evaluation",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot annotated 2x2 Confusion Matrix alongside the ROC Curve."""
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left panel: Confusion Matrix
    cm_arr = metrics["cm_array"]
    im = ax_cm.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    class_labels = ["Negative (0)", "Positive (1)"]
    ax_cm.set_xticks([0, 1], labels=class_labels)
    ax_cm.set_yticks([0, 1], labels=class_labels)
    ax_cm.set_xlabel("Predicted Label", fontweight="bold")
    ax_cm.set_ylabel("True Label", fontweight="bold")
    ax_cm.set_title("Confusion Matrix", fontweight="bold")

    total = max(int(cm_arr.sum()), 1)
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm_arr[i, j]
            pct = 100.0 * count / total
            color = "white" if count > thresh else "black"
            ax_cm.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", color=color, fontsize=11, fontweight="bold",
            )

    # Right panel: ROC Curve
    fpr, tpr, auc = metrics["fpr"], metrics["tpr"], metrics["roc_auc"]
    ax_roc.plot(fpr, tpr, linewidth=2.2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax_roc.fill_between(fpr, tpr, alpha=0.12)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontweight="bold")
    ax_roc.set_title("ROC Curve", fontweight="bold")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right")

    fig.suptitle(
        f"{title} | Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['f1_score']:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _finalize_figure(fig, save_path)


def plot_activation_functions(save_path: str | None = None) -> plt.Figure:
    """Plot custom activation functions f(x) and their exact derivatives f'(x)."""
    x = torch.linspace(-4.0, 4.0, 400)
    x_np = x.numpy()
    fig, (ax_f, ax_df) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, act_cls in ACTIVATION_REGISTRY.items():
        act = act_cls()
        ax_f.plot(x_np, act.forward(x).numpy(), linewidth=2.0, label=name)
        ax_df.plot(x_np, act.derivative(x).numpy(), linewidth=2.0, label=f"{name}'")

    ax_f.set_title("Activation Functions f(x)", fontweight="bold")
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("f(x)")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()

    ax_df.set_title("Analytical Derivatives f'(x)", fontweight="bold")
    ax_df.set_xlabel("x")
    ax_df.set_ylabel("f'(x)")
    ax_df.grid(True, alpha=0.3)
    ax_df.legend()

    fig.suptitle("Custom Activation Functions & Mathematical Derivatives", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finalize_figure(fig, save_path)
