"""From-scratch binary classification evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)."""

from typing import Any

import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Compute TP, TN, FP, FN counts from 1D binary label arrays."""
    y_true_bin = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_bin = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true_bin.shape != y_pred_bin.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_bin.shape}, y_pred={y_pred_bin.shape}.")

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_roc_curve_and_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve (FPR, TPR) and exact Area Under Curve (AUC) from scratch."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true={y_t.shape}, y_prob={y_p.shape}.")

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        # ROC-AUC is undefined when only a single class exists in y_true
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # Sort samples by predicted probability in descending order
    order = np.argsort(-y_p, kind="mergesort")
    y_sorted = y_t[order]
    p_sorted = y_p[order]

    # Accumulate true positives and false positives at every unique score threshold
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    distinct_idx = np.where(np.diff(p_sorted) != 0)[0]
    threshold_idx = np.r_[distinct_idx, len(y_sorted) - 1]

    # Prepend origin (0, 0) so ROC curve starts cleanly at threshold = +inf
    tpr = np.r_[0.0, tps[threshold_idx] / n_pos]
    fpr = np.r_[0.0, fps[threshold_idx] / n_neg]

    # Compute area via explicit trapezoidal rule (compatible across NumPy 1.x and 2.x)
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    return fpr, tpr, auc


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate Accuracy (%), Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_p >= threshold).astype(np.int64)

    cm = compute_confusion_matrix(y_t, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    accuracy_pct = (100.0 * (tp + tn) / total) if total > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr, tpr, roc_auc = compute_roc_curve_and_auc(y_t, y_p)

    return {
        "accuracy_pct": accuracy_pct,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "cm_array": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
        "fpr": fpr,
        "tpr": tpr,
    }


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format binary evaluation metrics as a readable multi-line report string."""
    cm = metrics["confusion_matrix"]
    return (
        f"Accuracy  : {metrics['accuracy_pct']:.2f}%\n"
        f"Precision : {metrics['precision']:.4f}\n"
        f"Recall    : {metrics['recall']:.4f}\n"
        f"F1-Score  : {metrics['f1_score']:.4f}\n"
        f"ROC-AUC   : {metrics['roc_auc']:.4f}\n"
        f"Confusion : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}"
    )
