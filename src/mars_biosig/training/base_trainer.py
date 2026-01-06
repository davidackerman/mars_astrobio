"""
Base trainer class with common training infrastructure.

Provides mixed precision training, checkpointing, logging, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base trainer class with common training infrastructure.

    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Checkpoint management
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        checkpoint_dir: str = "models/checkpoints",
        tensorboard_dir: str = "outputs/tensorboard",
        log_dir: str = "outputs/logs",
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        log_interval: int = 10,
        save_frequency: int = 10,
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize base trainer.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        optimizer : Optimizer, optional
            PyTorch optimizer
        scheduler : optional
            Learning rate scheduler
        criterion : nn.Module, optional
            Loss function
        device : str
            Device to train on ('cuda' or 'cpu')
        checkpoint_dir : str
            Directory to save checkpoints
        tensorboard_dir : str
            Directory for TensorBoard logs
        log_dir : str
            Directory for text logs
        mixed_precision : bool
            Use mixed precision training (FP16)
        gradient_clip : float
            Max gradient norm for clipping
        accumulation_steps : int
            Gradient accumulation steps
        log_interval : int
            Log every N batches
        save_frequency : int
            Save checkpoint every N epochs
        save_top_k : int
            Keep best K checkpoints
        monitor : str
            Metric to monitor for checkpointing
        mode : str
            'min' or 'max' for monitor metric
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # Training config
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.log_interval = log_interval

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        # Mixed precision
        self.scaler = GradScaler() if mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_checkpoints = []  # List of (metric_value, checkpoint_path) tuples
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Initialized BaseTrainer on {device}")
        logger.info(f"Mixed precision: {mixed_precision}")
        logger.info(f"Gradient accumulation steps: {accumulation_steps}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns
        -------
        dict
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            unit="batch",
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss = self.train_step(batch)

            # Backward pass with gradient accumulation
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.mixed_precision:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )

                    # Optimizer step
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Log to TensorBoard
                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar(
                        "train/loss", loss.item(), self.global_step
                    )
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            "train/lr",
                            self.get_lr(),
                            self.global_step,
                        )

            # Update metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average loss
        avg_loss = total_loss / n_batches

        return {"train_loss": avg_loss}

    def train_step(self, batch: Any) -> torch.Tensor:
        """
        Single training step (to be overridden by subclasses).

        Parameters
        ----------
        batch : Any
            Batch of data from DataLoader

        Returns
        -------
        loss : Tensor
            Loss value
        """
        raise NotImplementedError("Subclasses must implement train_step()")

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns
        -------
        dict
            Validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", unit="batch"):
            loss = self.val_step(batch)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        return {"val_loss": avg_loss}

    def val_step(self, batch: Any) -> torch.Tensor:
        """
        Single validation step (to be overridden by subclasses).

        Parameters
        ----------
        batch : Any
            Batch of data from DataLoader

        Returns
        -------
        loss : Tensor
            Loss value
        """
        raise NotImplementedError("Subclasses must implement val_step()")

    def train(self, epochs: int) -> Dict[str, list]:
        """
        Train the model for multiple epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs to train

        Returns
        -------
        dict
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric
                    metric = val_metrics.get("val_loss", train_metrics["train_loss"])
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Log metrics
            metrics = {**train_metrics, **val_metrics, "learning_rate": self.get_lr()}
            self.log_metrics(metrics)

            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self.save_checkpoint(metrics)

        logger.info("Training complete")
        return self.history

    def save_checkpoint(self, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Parameters
        ----------
        metrics : dict
            Current metrics
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint: {last_path}")

        # Save best checkpoints
        if self.monitor in metrics:
            metric_value = metrics[self.monitor]

            # Check if this is a top-k checkpoint
            if len(self.best_checkpoints) < self.save_top_k:
                should_save = True
            else:
                # Check if better than worst checkpoint
                if self.mode == "min":
                    should_save = metric_value < max(c[0] for c in self.best_checkpoints)
                else:
                    should_save = metric_value > min(c[0] for c in self.best_checkpoints)

            if should_save:
                # Save checkpoint
                ckpt_name = f"epoch_{self.current_epoch + 1}_{self.monitor}_{metric_value:.4f}.pth"
                ckpt_path = self.checkpoint_dir / ckpt_name
                torch.save(checkpoint, ckpt_path)

                # Update best checkpoints list
                self.best_checkpoints.append((metric_value, ckpt_path))

                # Sort and keep only top-k
                if self.mode == "min":
                    self.best_checkpoints.sort(key=lambda x: x[0])
                else:
                    self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

                # Remove worst checkpoint if exceeding top-k
                if len(self.best_checkpoints) > self.save_top_k:
                    _, worst_path = self.best_checkpoints.pop()
                    if worst_path.exists():
                        worst_path.unlink()
                        logger.info(f"Removed checkpoint: {worst_path}")

                logger.info(f"Saved best checkpoint: {ckpt_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint.get("history", self.history)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch + 1}")

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to TensorBoard and history.

        Parameters
        ----------
        metrics : dict
            Metrics to log
        """
        # Update history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        # Log to TensorBoard
        for key, value in metrics.items():
            if key != "learning_rate":  # Already logged during training
                self.writer.add_scalar(f"epoch/{key}", value, self.current_epoch)

        # Log to console
        log_str = f"Epoch {self.current_epoch + 1}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
