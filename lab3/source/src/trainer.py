"""Multi-class training loop, validation best-checkpoint tracking, testing, and .pt serialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Lab3Config
from src.metrics import evaluate_multiclass_metrics, format_metrics_summary
from src.model import BaseClassifier, build_model


@dataclass
class TrainingHistory:
    """Per-epoch training/validation loss & accuracy logs plus best checkpoint info."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class MultiClassTrainer:
    """Handles PyTorch multi-class training, validation, testing, and cross-device checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Lab3Config,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.cfg = cfg
        # Convert string device ("cuda", "mps", "cpu") into a torch.device object
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        # Multi-class Cross-Entropy combines LogSoftmax + NLLLoss on raw logits (B, C)
        self.criterion = nn.CrossEntropyLoss()
        # Default to Adam with L2 weight decay to reduce overfitting on 1,308 banknote images
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.history = TrainingHistory()

    def train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Run one training epoch and return (avg_loss, accuracy_pct in [0, 100])."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Standard forward -> CrossEntropy loss -> backward -> parameter update step
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            # Accumulate sample-weighted loss and Top-1 correct predictions
            batch_size = y_batch.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == y_batch).sum().item())

        avg_loss = total_loss / max(total_samples, 1)
        acc_pct = 100.0 * total_correct / max(total_samples, 1)
        return avg_loss, acc_pct

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a loader and return (avg_loss, y_true_1d, y_probs_2d)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_targets: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            # Convert raw logits (B, C) into class probability distribution summing to 1.0
            probs = torch.softmax(logits, dim=-1)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        if all_targets:
            y_true = np.concatenate(all_targets, axis=0)
            y_probs = np.concatenate(all_probs, axis=0)
        else:
            # Preserve exact 1D and 2D (0, C) shapes even on an empty loader
            y_true = np.empty((0,), dtype=np.int64)
            y_probs = np.empty((0, self.cfg.num_classes), dtype=np.float32)

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, y_true, y_probs

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train for cfg.epochs, retain lowest validation-loss weights, and restore best state."""
        history = TrainingHistory()
        # Deep-copy initial weights so later parameter updates never mutate the snapshot
        best_state = copy.deepcopy(self.model.state_dict())
        start_time = time.perf_counter()

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, y_val, p_val = self.evaluate_loader(val_loader)

            # Compute Top-1 validation accuracy (%) from 2D probability matrix (N, C)
            if len(y_val) > 0:
                val_acc = float(100.0 * np.mean(np.argmax(p_val, axis=1) == y_val))
            else:
                val_acc = 0.0

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_acc)

            # Snapshot weights whenever validation loss reaches a new minimum
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.cfg.epochs):
                print(
                    f"Epoch {epoch:02d}/{self.cfg.epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} ({train_acc:.2f}%) | "
                    f"Val Loss: {val_loss:.4f} ({val_acc:.2f}%)"
                )

        # Restore best generalization weights before final testing or exporting
        self.model.load_state_dict(best_state)
        self.history = history
        if verbose:
            elapsed = time.perf_counter() - start_time
            print(
                f"Training finished in {elapsed:.1f}s | "
                f"Restored best model from epoch {history.best_epoch} "
                f"(val_loss={history.best_val_loss:.4f})."
            )
        return history

    def test(self, test_loader: DataLoader) -> dict[str, Any]:
        """Run full multi-class evaluation on the Test set and attach formatted summary."""
        test_loss, y_true, y_probs = self.evaluate_loader(test_loader)
        metrics = evaluate_multiclass_metrics(y_true, y_probs, self.cfg.class_names)
        metrics["test_loss"] = float(test_loss)
        metrics["summary"] = format_metrics_summary(metrics)
        return metrics

    def save_checkpoint(
        self,
        save_path: str | Path,
        test_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Save CPU-portable model weights together with architecture & class metadata."""
        out_path = Path(save_path)
        # Ensure parent output directory exists before writing
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Move every weight tensor to CPU so Colab-trained .pt files load cleanly on Mac/CPU
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        payload: dict[str, Any] = {
            "state_dict": cpu_state,
            "model_type": self.cfg.model_type,
            "img_size": self.cfg.img_size,
            "num_channels": self.cfg.num_channels,
            "hidden_dims": tuple(self.cfg.hidden_dims),
            "dropout": self.cfg.dropout,
            "class_names": list(self.cfg.class_names),
        }
        if test_metrics is not None:
            payload["test_accuracy_pct"] = float(test_metrics["accuracy_pct"])
            payload["macro_f1"] = float(test_metrics["macro_f1"])

        torch.save(payload, out_path)
        return out_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[BaseClassifier, dict[str, Any]]:
    """Rebuild model architecture from checkpoint metadata, load weights, and set eval() mode."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # map_location guarantees CUDA checkpoints load smoothly on Apple Silicon MPS or CPU
    payload: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct matching Lab3Config from saved metadata (tuples match dataclass types)
    cfg = Lab3Config(
        model_type=str(payload["model_type"]),
        img_size=int(payload["img_size"]),
        num_channels=int(payload["num_channels"]),
        hidden_dims=tuple(payload["hidden_dims"]),
        dropout=float(payload["dropout"]),
        class_names=tuple(payload["class_names"]),
        device=device,
    )

    # Instantiate exact architecture (pretrained=False avoids network download during local inference)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.device(device))
    model.eval()
    return model, payload
