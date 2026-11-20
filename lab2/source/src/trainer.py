"""Training, validation, best-checkpoint selection, and experiment orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, seed_everything
from src.metrics import evaluate_binary_metrics
from src.model import ImageMLP


@dataclass
class TrainingHistory:
    """Per-epoch training/validation loss & accuracy logs plus best checkpoint info."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


@dataclass
class ExperimentResult:
    """Container bundling config, training history, test metrics, and trained model."""

    name: str
    config: ExperimentConfig
    history: TrainingHistory
    test_metrics: dict[str, Any]
    num_params: int
    elapsed_seconds: float
    model: ImageMLP


class Trainer:
    """Handles PyTorch training, validation, best-model restoration, and test evaluation."""

    def __init__(
        self,
        model: ImageMLP,
        cfg: ExperimentConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        # Numerically stable binary cross-entropy operating directly on raw logits
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Run one training epoch and return (avg_loss, accuracy_pct)."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Standard PyTorch optimization step
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits) >= 0.5).to(dtype=y_batch.dtype)
            total_correct += int((preds == y_batch).sum().item())
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        acc_pct = 100.0 * total_correct / max(total_samples, 1)
        return avg_loss, acc_pct

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a loader and return (avg_loss, y_true_array, y_prob_array)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_targets: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            probs = torch.sigmoid(logits)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        y_true = np.concatenate(all_targets) if all_targets else np.array([])
        y_prob = np.concatenate(all_probs) if all_probs else np.array([])
        return avg_loss, y_true, y_prob

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train for cfg.epochs, keep the lowest validation-loss weights, and return history."""
        history = TrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, y_val, p_val = self.evaluate_loader(val_loader)
            val_acc = float(100.0 * np.mean((p_val >= 0.5) == y_val)) if len(y_val) > 0 else 0.0

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_acc)

            # Save model snapshot whenever validation loss improves
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

        # Restore best generalization weights before final testing
        self.model.load_state_dict(best_state)
        return history

    def test(self, test_loader: DataLoader, threshold: float = 0.5) -> dict[str, Any]:
        """Run full binary classification evaluation on the Test set."""
        test_loss, y_true, y_prob = self.evaluate_loader(test_loader)
        metrics = evaluate_binary_metrics(y_true, y_prob, threshold=threshold)
        metrics["test_loss"] = test_loss
        return metrics


def run_experiment(
    name: str,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    verbose: bool = True,
) -> ExperimentResult:
    """Run one complete, seed-controlled training + testing experiment."""
    # Re-seed right before building the model so every comparison run is reproducible
    seed_everything(cfg.seed)
    model = ImageMLP.from_config(cfg)
    trainer = Trainer(model, cfg)

    if verbose:
        print(
            f"\n=== [{name}] activation={cfg.activation_name}, "
            f"init={cfg.init_method}, device={cfg.device} ==="
        )

    start_time = time.perf_counter()
    history = trainer.fit(train_loader, val_loader, verbose=verbose)
    test_metrics = trainer.test(test_loader)
    elapsed = time.perf_counter() - start_time

    if verbose:
        print(
            f"--> Best Epoch: {history.best_epoch} | "
            f"Test Acc: {test_metrics['accuracy_pct']:.2f}% | "
            f"F1: {test_metrics['f1_score']:.4f} | "
            f"AUC: {test_metrics['roc_auc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    return ExperimentResult(
        name=name,
        config=cfg,
        history=history,
        test_metrics=test_metrics,
        num_params=model.count_parameters(),
        elapsed_seconds=elapsed,
        model=model,
    )
