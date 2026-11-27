"""Multi-class evaluation metrics (Accuracy, Per-Class/Macro/Weighted P/R/F1, Confusion Matrix)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_multiclass_metrics(
    y_true: np.ndarray | Sequence[int],
    y_probs: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute comprehensive multi-class evaluation metrics from ground-truth labels and Softmax probabilities.

    Args:
        y_true: 1D sequence/array of integer ground-truth class indices in [0, C - 1].
        y_probs: 2D array of predicted Softmax probabilities with shape (N, C).
        class_names: Ordered sequence of C class labels matching columns of y_probs.

    Returns:
        Dictionary containing overall Accuracy (%), Mean Top-1 Confidence (%),
        Macro/Weighted Precision/Recall/F1, per-class metrics, and the C x C confusion matrix.
    """
    # Convert inputs into NumPy arrays with explicit dtypes (asarray also accepts Python lists)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_probs_arr = np.asarray(y_probs, dtype=np.float64)

    # Guard clauses: validate tensor ranks, sample counts, and number of classes
    if y_true_arr.ndim != 1 or y_probs_arr.ndim != 2:
        raise ValueError(
            f"y_true must be 1D and y_probs must be 2D; got {y_true_arr.shape} and {y_probs_arr.shape}."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("Cannot evaluate metrics on an empty dataset (0 samples).")
    if y_true_arr.shape[0] != y_probs_arr.shape[0]:
        raise ValueError(
            f"Sample count mismatch: len(y_true)={y_true_arr.shape[0]} vs y_probs rows={y_probs_arr.shape[0]}."
        )
    if y_probs_arr.shape[1] != len(class_names):
        raise ValueError(
            f"Class count mismatch: y_probs cols={y_probs_arr.shape[1]} vs len(class_names)={len(class_names)}."
        )

    # Derive Top-1 predicted class index and Top-1 confidence score per sample
    y_pred = np.argmax(y_probs_arr, axis=1)
    top1_confidence = np.max(y_probs_arr, axis=1)
    mean_confidence_pct = float(np.mean(top1_confidence) * 100.0)

    # Lock class index ordering [0, 1, ..., C-1] so confusion matrix is always C x C
    num_classes = len(class_names)
    label_ids = list(range(num_classes))

    # Confusion matrix: rows = True label, columns = Predicted label
    cm_array = confusion_matrix(y_true_arr, y_pred, labels=label_ids)

    # Overall classification accuracy in percentage [0, 100]
    accuracy_pct = float(accuracy_score(y_true_arr, y_pred) * 100.0)

    # Per-class Precision, Recall, F1-score, and sample count (support); zero_division=0 avoids NaN/warnings
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    # Macro averages treat every class equally regardless of class frequency
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    # Weighted averages weight each class metric by its true sample count (support)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

    # Package per-class statistics into JSON-serializable Python floats/ints
    per_class: dict[str, dict[str, float | int]] = {}
    for name, p_val, r_val, f1_val, sup_val in zip(class_names, p_cls, r_cls, f1_cls, sup_cls):
        per_class[str(name)] = {
            "precision": float(p_val),
            "recall": float(r_val),
            "f1": float(f1_val),
            "support": int(sup_val),
        }

    return {
        "accuracy_pct": accuracy_pct,
        "mean_confidence_pct": mean_confidence_pct,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "cm_array": cm_array,
        "class_names": list(class_names),
    }


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format multi-class evaluation metrics into a clean, aligned multi-line report table."""
    lines = [
        f"Accuracy            : {metrics['accuracy_pct']:.2f}%",
        f"Mean Top-1 Conf     : {metrics['mean_confidence_pct']:.2f}%",
        f"Macro P / R / F1    : {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}",
        f"Weighted P / R / F1 : {metrics['weighted_precision']:.4f} / {metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}",
        "",
        f"{'Class':<14} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7}",
        "-" * 60,
    ]

    # Append one aligned row per class
    for name, stats in metrics["per_class"].items():
        lines.append(
            f"{name:<14} | "
            f"{stats['precision']:>9.4f} | "
            f"{stats['recall']:>9.4f} | "
            f"{stats['f1']:>9.4f} | "
            f"{stats['support']:>7d}"
        )
    return "\n".join(lines)
