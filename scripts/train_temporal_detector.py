#!/usr/bin/env python3
"""
Train temporal object detector using leave-one-out cross-validation.

This script implements the complete training pipeline for detecting movers
and dipoles in Backyard Worlds temporal sequences.
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scientific_pipelines.core.training.models import (
    TemporalObjectDetector,
    FrameStackObjectDetector,
    DiffStreamObjectDetector,
)
from scientific_pipelines.core.training.losses import DetectionLoss
from scientific_pipelines.core.training.datasets import (
    BackyardWorldsTemporalDataset,
    TemporalSequenceAugmentation,
)
from scientific_pipelines.core.training.metrics import DetectionMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Custom collate function for batching variable-length annotations.

    Handles 'keypoints' and 'labels' which are lists of variable length.
    """
    # Stack/concatenate fixed-size tensors
    frames = torch.stack([item['frames'] for item in batch])
    class_labels = torch.stack([item['class_labels'] for item in batch])
    heatmap_targets = torch.stack([item['heatmap_targets'] for item in batch])

    # Keep variable-length items as lists
    sequence_ids = [item['sequence_id'] for item in batch]
    keypoints = [item['keypoints'] for item in batch]
    labels = [item['labels'] for item in batch]

    return {
        'frames': frames,
        'class_labels': class_labels,
        'heatmap_targets': heatmap_targets,
        'sequence_id': sequence_ids,
        'keypoints': keypoints,
        'labels': labels,
    }


class LOOCVTrainer:
    """
    Leave-One-Out Cross-Validation trainer for tiny datasets.

    Trains and validates model using LOOCV, where each of the 5 samples
    serves as validation set once.

    Args:
        model: TemporalObjectDetector model
        loss_fn: DetectionLoss function
        metrics_fn: DetectionMetrics function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs per fold
        augmentation_multiplier: How many augmented copies per training sample
        checkpoint_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics_fn: DetectionMetrics,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str,
        num_epochs: int = 100,
        augmentation_multiplier: int = 50,
        checkpoint_dir: Path = Path("checkpoints"),
        log_dir: Path = Path("logs"),
        score_threshold: float = 0.1,
        log_heatmaps: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.augmentation_multiplier = augmentation_multiplier
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.score_threshold = score_threshold
        self.log_heatmaps = log_heatmaps

        # Create TensorBoard writer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None  # Will be created per fold

        # Store optimizer config for recreating between folds
        self.optimizer_config = {
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0].get('weight_decay', 0),
        }
        self.scheduler_config = {
            'mode': scheduler.mode,
            'factor': scheduler.factor,
            'patience': scheduler.patience,
        }

    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int,
    ) -> Dict:
        """Train one LOOCV fold."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}")
        logger.info(f"{'='*60}")

        # Create TensorBoard writer for this fold
        self.writer = SummaryWriter(log_dir=self.log_dir / f"fold_{fold_idx}")

        best_val_loss = float('inf')
        best_val_metrics = {}
        best_model_state = None
        patience_counter = 0
        patience = 100

        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_loss_dict = self._train_epoch(train_loader)

            # Validation
            val_loss, val_loss_dict, val_metrics = self._validate_epoch(val_loader, epoch)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # TensorBoard logging
            # Each fold is in a separate run (separate writer), no need for fold prefix
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for key, value in train_loss_dict.items():
                self.writer.add_scalar(f'Loss_Components/train_{key}', value, epoch)
            for key, value in val_loss_dict.items():
                self.writer.add_scalar(f'Loss_Components/val_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

            # Logging
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:03d}/{self.num_epochs}: "
                    f"Train Loss={train_loss:.4f} | "
                    f"Val Loss={val_loss:.4f} | "
                    f"Val F1={val_metrics['det_f1']:.3f} | "
                    f"Val KeypointErr={val_metrics['keypoint_error']:.2f}px"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics.copy()
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / f"fold_{fold_idx}_best.pth"
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_loss': best_val_loss,
                    'val_metrics': best_val_metrics,
                }, checkpoint_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        self.model.load_state_dict(best_model_state)

        logger.info(f"\nBest validation loss: {best_val_loss:.4f}")
        logger.info(f"Best validation metrics: {best_val_metrics}")

        # Close TensorBoard writer for this fold
        self.writer.close()

        return best_val_metrics

    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_loss_dict = {'loss_cls': 0.0, 'loss_heatmap': 0.0}
        num_batches = 0

        for batch in train_loader:
            frames = batch['frames'].to(self.device)
            class_labels = batch['class_labels'].to(self.device)
            heatmap_targets = batch['heatmap_targets'].to(self.device)

            # Forward pass
            class_logits, heatmaps = self.model(frames)

            # Compute loss
            predictions = (class_logits, heatmaps)
            targets = (class_labels, heatmap_targets)
            loss, loss_dict = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            for key in total_loss_dict:
                total_loss_dict[key] += loss_dict[key]
            num_batches += 1

        # Average
        avg_loss = total_loss / num_batches
        avg_loss_dict = {k: v / num_batches for k, v in total_loss_dict.items()}

        return avg_loss, avg_loss_dict

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_loss_dict = {'loss_cls': 0.0, 'loss_heatmap': 0.0}
        num_batches = 0

        all_predictions = []
        all_targets = []

        logged_heatmaps = False

        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(self.device)
                class_labels = batch['class_labels'].to(self.device)
                heatmap_targets = batch['heatmap_targets'].to(self.device)

                # Forward pass
                class_logits, heatmaps = self.model(frames)

                # Compute loss
                predictions = (class_logits, heatmaps)
                targets = (class_labels, heatmap_targets)
                loss, loss_dict = self.loss_fn(predictions, targets)

                # Accumulate loss
                total_loss += loss.item()
                for key in total_loss_dict:
                    total_loss_dict[key] += loss_dict[key]
                num_batches += 1

                # Decode predictions for metrics
                decoded_preds = self.model.decode_predictions(
                    class_logits,
                    heatmaps,
                    score_threshold=self.score_threshold,
                )

                if (
                    self.writer is not None
                    and self.log_heatmaps
                    and not logged_heatmaps
                ):
                    self._log_heatmaps(frames, heatmap_targets, heatmaps, epoch)
                    logged_heatmaps = True

                # Prepare targets for metrics
                for i in range(len(batch['sequence_id'])):
                    all_predictions.append(decoded_preds[i])
                    all_targets.append({
                        'class_labels': class_labels[i],
                        'keypoints': batch['keypoints'][i],
                        'labels': batch['labels'][i],
                    })

        # Average loss
        avg_loss = total_loss / num_batches
        avg_loss_dict = {k: v / num_batches for k, v in total_loss_dict.items()}

        # Compute metrics
        metrics = self.metrics_fn.compute_metrics(all_predictions, all_targets)

        return avg_loss, avg_loss_dict, metrics

    def _log_heatmaps(
        self,
        frames: torch.Tensor,
        heatmap_targets: torch.Tensor,
        heatmaps: torch.Tensor,
        epoch: int,
    ) -> None:
        """Log a single validation sample's frames and heatmaps to TensorBoard."""
        frames_cpu = frames[0].detach().cpu()  # (T, C, H, W)
        target_maps = heatmap_targets[0].detach().cpu()  # (C, H', W')
        pred_maps = torch.sigmoid(heatmaps[0]).detach().cpu()

        frame_grid = torchvision.utils.make_grid(frames_cpu, nrow=frames_cpu.shape[0])
        self.writer.add_image("Samples/frames", frame_grid, epoch)

        target_grid = torchvision.utils.make_grid(target_maps.unsqueeze(1), nrow=target_maps.shape[0])
        pred_grid = torchvision.utils.make_grid(pred_maps.unsqueeze(1), nrow=pred_maps.shape[0])
        self.writer.add_image("Samples/heatmap_targets", target_grid, epoch)
        self.writer.add_image("Samples/heatmap_preds", pred_grid, epoch)

    def cross_validate(self, full_dataset: BackyardWorldsTemporalDataset) -> Dict:
        """Run full leave-one-out cross-validation."""
        logger.info(f"\n{'='*60}")
        logger.info("Starting Leave-One-Out Cross-Validation")
        logger.info(f"Dataset size: {len(full_dataset)} samples")
        logger.info(f"Augmentation multiplier: {self.augmentation_multiplier}x")
        logger.info(f"{'='*60}\n")

        fold_results = []

        for fold_idx in range(len(full_dataset)):
            # Create train/val split
            val_indices = [fold_idx]
            train_indices = [i for i in range(len(full_dataset)) if i != fold_idx]

            # Create datasets
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

            # Apply augmentation multiplier to training set
            # Repeat the dataset N times to create more training samples
            train_dataset_aug = ConcatDataset([train_dataset] * self.augmentation_multiplier)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset_aug,
                batch_size=4,
                shuffle=True,
                num_workers=2,
                pin_memory=(self.device == "cuda"),
                collate_fn=collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

            logger.info(f"Fold {fold_idx + 1}: Training on {len(train_dataset_aug)} samples, "
                       f"Validating on {len(val_dataset)} sample")

            # Reset model for each fold
            self.model.apply(self._reset_weights)

            # Recreate optimizer and scheduler for new parameters
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_config['mode'],
                factor=self.scheduler_config['factor'],
                patience=self.scheduler_config['patience'],
            )

            # Train fold
            val_metrics = self.train_fold(train_loader, val_loader, fold_idx)
            fold_results.append(val_metrics)

        # Aggregate results
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*60}")

        avg_metrics = {}
        for key in fold_results[0].keys():
            values = [fold[key] for fold in fold_results]
            avg_metrics[key] = np.mean(values)
            std_metrics = np.std(values)
            logger.info(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")

        logger.info(f"\n{'='*60}\n")

        return avg_metrics

    @staticmethod
    def _reset_weights(m):
        """Reset model weights."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description="Train temporal object detector")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth"),
        help="Data directory containing subjects_groundtruth/"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/temporal_detector"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs per fold"
    )
    parser.add_argument(
        "--aug-multiplier",
        type=int,
        default=50,
        help="Augmentation multiplier (50x = 200 samples from 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/temporal_detector"),
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["temporal", "framestack", "diff"],
        default="temporal",
        help="Model variant to train"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Heatmap peak score threshold for decoding detections"
    )
    parser.add_argument(
        "--alpha-heatmap",
        type=float,
        default=5.0,
        help="Heatmap loss weight"
    )
    parser.add_argument(
        "--log-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log heatmap targets/predictions to TensorBoard"
    )

    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    logger.info(f"Data directory: {args.data_dir}")
    run_id = f"{args.model}_run_{datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S')}"
    args.checkpoint_dir = args.checkpoint_dir / args.model / run_id
    args.log_dir = args.log_dir / args.model / run_id

    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"TensorBoard log directory: {args.log_dir}")

    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create datasets
    annotations_path = args.data_dir / "annotations.json"

    # Training dataset with augmentation
    train_transform = TemporalSequenceAugmentation(
        input_size=(256, 256),
        training=True
    )

    train_dataset = BackyardWorldsTemporalDataset(
        data_dir=args.data_dir,
        annotations_path=annotations_path,
        transform=train_transform,
        heatmap_size=(8, 8),
        input_size=(256, 256),
    )

    logger.info(f"Loaded {len(train_dataset)} annotated sequences")

    # Create model
    if args.model == "framestack":
        model = FrameStackObjectDetector(
            num_classes=2,
            pretrained=True,
            freeze_backbone=True,
            heatmap_size=(8, 8),
        ).to(args.device)
    elif args.model == "diff":
        model = DiffStreamObjectDetector(
            num_classes=2,
            pretrained=True,
            freeze_backbone=True,
            heatmap_size=(8, 8),
        ).to(args.device)
    else:
        model = TemporalObjectDetector(
            num_classes=2,
            pretrained=True,
            freeze_backbone=True,
            heatmap_size=(8, 8),
        ).to(args.device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Create loss function
    loss_fn = DetectionLoss(
        alpha_cls=1.0,
        alpha_heatmap=args.alpha_heatmap,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )

    # Create metrics
    metrics_fn = DetectionMetrics(distance_threshold=15.0)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    # Create trainer
    trainer = LOOCVTrainer(
        model=model,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        num_epochs=args.epochs,
        augmentation_multiplier=args.aug_multiplier,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        score_threshold=args.score_threshold,
        log_heatmaps=args.log_heatmaps,
    )

    # Run cross-validation
    avg_metrics = trainer.cross_validate(train_dataset)

    # Save final results
    results_path = args.checkpoint_dir / "cv_results.txt"
    with open(results_path, 'w') as f:
        f.write("Leave-One-Out Cross-Validation Results\n")
        f.write("="*60 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    logger.info(f"Results saved to {results_path}")
    logger.info(f"\nTo view training logs in TensorBoard, run:")
    logger.info(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
