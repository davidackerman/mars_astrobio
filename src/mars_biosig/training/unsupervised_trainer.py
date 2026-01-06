"""
Unsupervised trainer for SimCLR contrastive learning.

Implements training loop for self-supervised representation learning on Mars images.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging

from .base_trainer import BaseTrainer
from mars_biosig.models.unsupervised import SimCLR, NTXentLoss, SimCLRWithLinearProbe
from mars_biosig.data.dataset import WATSONTextureDataset

logger = logging.getLogger(__name__)


class UnsupervisedTrainer(BaseTrainer):
    """
    Trainer for unsupervised SimCLR model.

    Extends BaseTrainer with SimCLR-specific training logic including:
    - Contrastive loss on augmented pairs
    - Linear probe evaluation
    """

    def __init__(
        self,
        model: SimCLR,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[NTXentLoss] = None,
        linear_probe_loader: Optional[DataLoader] = None,
        linear_probe_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize unsupervised trainer.

        Parameters
        ----------
        model : SimCLR
            SimCLR model
        train_loader : DataLoader
            Training data loader (returns two views per image)
        val_loader : DataLoader, optional
            Validation data loader
        optimizer : Optimizer, optional
            PyTorch optimizer
        scheduler : optional
            Learning rate scheduler
        criterion : NTXentLoss, optional
            NT-Xent contrastive loss
        linear_probe_loader : DataLoader, optional
            Data loader for linear probe evaluation (with labels)
        linear_probe_config : dict, optional
            Configuration for linear probe evaluation
        **kwargs
            Additional arguments passed to BaseTrainer
        """
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            **kwargs,
        )

        # Linear probe for evaluation
        self.linear_probe_loader = linear_probe_loader
        self.linear_probe_config = linear_probe_config or {}
        self.linear_probe_model = None
        self.linear_probe_enabled = (
            linear_probe_loader is not None and
            self.linear_probe_config.get("enabled", True)
        )

        logger.info(f"Linear probe enabled: {self.linear_probe_enabled}")

    def train_step(self, batch: Any) -> torch.Tensor:
        """
        Single training step for SimCLR.

        Parameters
        ----------
        batch : tuple
            Batch containing (view1, view2) - two augmented views of same images

        Returns
        -------
        loss : Tensor
            Contrastive loss value
        """
        # Unpack two views
        view1, view2 = batch
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)

        # Forward pass with mixed precision
        if self.mixed_precision:
            from torch.cuda.amp import autocast
            with autocast():
                # Get projections for both views
                z1 = self.model(view1)
                z2 = self.model(view2)

                # Compute contrastive loss
                loss = self.criterion(z1, z2)
        else:
            # Get projections
            z1 = self.model(view1)
            z2 = self.model(view2)

            # Compute contrastive loss
            loss = self.criterion(z1, z2)

        return loss

    @torch.no_grad()
    def val_step(self, batch: Any) -> torch.Tensor:
        """
        Single validation step for SimCLR.

        Parameters
        ----------
        batch : tuple
            Batch containing (view1, view2)

        Returns
        -------
        loss : Tensor
            Contrastive loss value
        """
        # Unpack two views
        view1, view2 = batch
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)

        # Forward pass
        z1 = self.model(view1)
        z2 = self.model(view2)

        # Compute contrastive loss
        loss = self.criterion(z1, z2)

        return loss

    def evaluate_linear_probe(self) -> Dict[str, float]:
        """
        Evaluate representation quality using a linear probe.

        Trains a linear classifier on frozen features to measure how well
        the representations separate texture classes.

        Returns
        -------
        dict
            Linear probe metrics (accuracy, loss)
        """
        if not self.linear_probe_enabled:
            return {}

        logger.info("Evaluating linear probe...")

        # Create linear probe model
        num_classes = self.linear_probe_config.get("num_classes", 5)
        freeze_encoder = self.linear_probe_config.get("freeze_encoder", True)

        probe = SimCLRWithLinearProbe(
            simclr_model=self.model,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        # Train linear probe
        probe_epochs = self.linear_probe_config.get("epochs", 100)
        probe_lr = self.linear_probe_config.get("learning_rate", 0.01)
        probe_batch_size = self.linear_probe_config.get("batch_size", 32)

        probe_optimizer = torch.optim.SGD(
            probe.parameters(),
            lr=probe_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        probe_criterion = nn.CrossEntropyLoss()

        # Training loop
        best_acc = 0.0
        for epoch in range(probe_epochs):
            probe.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.linear_probe_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                logits = probe(images)
                loss = probe_criterion(logits, labels)

                # Backward
                probe_optimizer.zero_grad()
                loss.backward()
                probe_optimizer.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Epoch metrics
            epoch_loss = total_loss / len(self.linear_probe_loader)
            epoch_acc = 100.0 * correct / total

            if epoch_acc > best_acc:
                best_acc = epoch_acc

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Linear probe epoch {epoch + 1}/{probe_epochs}: "
                    f"loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%"
                )

        logger.info(f"Linear probe best accuracy: {best_acc:.2f}%")

        return {
            "linear_probe_accuracy": best_acc,
            "linear_probe_loss": epoch_loss,
        }

    def train(self, epochs: int) -> Dict[str, list]:
        """
        Train the model for multiple epochs with linear probe evaluation.

        Parameters
        ----------
        epochs : int
            Number of epochs to train

        Returns
        -------
        dict
            Training history
        """
        logger.info(f"Starting SimCLR training for {epochs} epochs")

        eval_frequency = self.linear_probe_config.get("eval_frequency", 10)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Evaluate linear probe
            if self.linear_probe_enabled and (epoch + 1) % eval_frequency == 0:
                probe_metrics = self.evaluate_linear_probe()
                val_metrics.update(probe_metrics)

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

        logger.info("SimCLR training complete")
        return self.history


def create_simclr_trainer(
    model: SimCLR,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    linear_probe_loader: Optional[DataLoader] = None,
    config: Optional[Dict] = None,
    device: str = "cuda",
) -> UnsupervisedTrainer:
    """
    Factory function to create SimCLR trainer from configuration.

    Parameters
    ----------
    model : SimCLR
        SimCLR model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    linear_probe_loader : DataLoader, optional
        Linear probe data loader (with labels)
    config : dict, optional
        Training configuration
    device : str
        Device to train on

    Returns
    -------
    trainer : UnsupervisedTrainer
        Configured trainer
    """
    config = config or {}

    # Loss function
    loss_config = config.get("loss", {})
    criterion = NTXentLoss(
        temperature=loss_config.get("temperature", 0.5),
    )

    # Optimizer
    optim_config = config.get("optimizer", {})
    base_lr = optim_config.get("base_lr", 0.3)
    batch_size = config.get("batch_size", 256)
    lr = base_lr * (batch_size / 256)  # Scale learning rate

    if optim_config.get("type", "SGD") == "LARS":
        # Try to use LARS if available
        try:
            from torchlars import LARS
            base_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=optim_config.get("momentum", 0.9),
                weight_decay=optim_config.get("weight_decay", 1e-6),
            )
            optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
            logger.info("Using LARS optimizer")
        except ImportError:
            logger.warning("LARS not available, falling back to SGD")
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=optim_config.get("momentum", 0.9),
                weight_decay=optim_config.get("weight_decay", 1e-6),
            )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optim_config.get("momentum", 0.9),
            weight_decay=optim_config.get("weight_decay", 1e-6),
        )

    # Scheduler
    sched_config = config.get("scheduler", {})
    T_max = config.get("epochs", 200)
    warmup_epochs = sched_config.get("warmup_epochs", 10)

    # Simple cosine annealing (warmup can be added later)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=sched_config.get("eta_min", 0.0),
    )

    # Training config
    training_config = config.get("training", {})

    # Create trainer
    trainer = UnsupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        linear_probe_loader=linear_probe_loader,
        linear_probe_config=config.get("linear_probe", {}),
        device=device,
        mixed_precision=training_config.get("mixed_precision", True),
        gradient_clip=training_config.get("gradient_clip", 1.0),
        accumulation_steps=training_config.get("accumulation_steps", 1),
        checkpoint_dir=config.get("checkpoint_dir", "models/checkpoints/simclr"),
        tensorboard_dir=config.get("tensorboard_dir", "outputs/tensorboard/simclr"),
        log_interval=config.get("log_interval", 10),
        save_frequency=config.get("save_frequency", 10),
        save_top_k=3,
        monitor="linear_probe_accuracy",
        mode="max",
    )

    return trainer
