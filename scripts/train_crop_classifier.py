#!/usr/bin/env python3
"""
Train a crop-based temporal classifier with leave-one-subject-out CV.

This trains a lightweight 3D CNN on random crops and positive-centered crops
to predict mover/dipole presence per crop.
"""

import argparse
import copy
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scientific_pipelines.core.training.datasets import (
    BackyardWorldsTemporalCropDataset,
    TemporalSequenceAugmentation,
)
from scientific_pipelines.core.training.models import TemporalCropClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def collate_fn(batch: List[Dict], any_object: bool = False) -> Dict:
    frames = torch.stack([item['frames'] for item in batch])
    class_labels = torch.stack([item['class_labels'] for item in batch])
    if any_object:
        class_labels = (class_labels.sum(dim=1, keepdim=True) > 0).float()
    sequence_ids = [item['sequence_id'] for item in batch]
    crop_centers = [item['crop_center'] for item in batch]
    crop_origins = [item['crop_origin'] for item in batch]
    keypoints = [item['keypoints'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'frames': frames,
        'class_labels': class_labels,
        'sequence_id': sequence_ids,
        'crop_center': crop_centers,
        'crop_origin': crop_origins,
        'keypoints': keypoints,
        'labels': labels,
    }


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    eps = 1e-8
    per_class = {}

    for class_idx in range(targets.shape[1]):
        pred = preds[:, class_idx]
        true = targets[:, class_idx]
        tp = (pred * true).sum().item()
        fp = (pred * (1 - true)).sum().item()
        fn = ((1 - pred) * true).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        per_class[f'class_{class_idx}_precision'] = precision
        per_class[f'class_{class_idx}_recall'] = recall
        per_class[f'class_{class_idx}_f1'] = f1

    micro_tp = (preds * targets).sum().item()
    micro_fp = (preds * (1 - targets)).sum().item()
    micro_fn = ((1 - preds) * targets).sum().item()
    micro_precision = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    per_class['micro_precision'] = micro_precision
    per_class['micro_recall'] = micro_recall
    per_class['micro_f1'] = micro_f1
    return per_class


def summarize_class_balance(loader: DataLoader) -> Dict[str, float]:
    totals = None
    count = 0
    for batch in loader:
        labels = batch['class_labels']
        totals = labels.sum(dim=0) if totals is None else totals + labels.sum(dim=0)
        count += labels.size(0)
    if totals is None:
        return {'num_samples': 0}
    balance = {'num_samples': count}
    for idx in range(totals.numel()):
        balance[f'class_{idx}_pos'] = totals[idx].item()
        balance[f'class_{idx}_pos_rate'] = totals[idx].item() / max(1, count)
    return balance


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    totals = None
    count = 0
    for batch in loader:
        labels = batch['class_labels']
        totals = labels.sum(dim=0) if totals is None else totals + labels.sum(dim=0)
        count += labels.size(0)
    if totals is None:
        return torch.ones(1)
    neg = count - totals
    pos_weight = neg / (totals + 1e-8)
    return pos_weight


class SoftF1Loss(nn.Module):
    """Differentiable F1 loss for multi-label classification."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=0)
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)
        f1 = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        return 1 - f1.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of two losses."""

    def __init__(self, loss_a: nn.Module, loss_b: nn.Module, weight_b: float):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_b = weight_b

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_a(logits, targets) + self.weight_b * self.loss_b(logits, targets)


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.to(probs.dtype)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * torch.log(pt.clamp_min(1e-8))
        return loss.mean()

def build_subject_index_map(samples: List[Dict]) -> Dict[str, List[int]]:
    subject_map: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        subject_id = sample['subject_id']
        subject_map.setdefault(subject_id, []).append(idx)
    return subject_map


def build_label_map(labels: List[List[float]]) -> Dict[str, List[int]]:
    class_map: Dict[str, List[int]] = {}
    for idx, label_vec in enumerate(labels):
        for class_idx, value in enumerate(label_vec):
            if value >= 0.5:
                class_map.setdefault(f"class_{class_idx}", []).append(idx)
    class_map['negative'] = [idx for idx, vec in enumerate(labels) if sum(vec) == 0]
    return class_map


def balanced_sample_indices(
    train_indices: List[int],
    labels: List[List[float]],
    seed: int,
    negative_ratio: float,
) -> List[int]:
    rng = random.Random(seed)
    label_map = build_label_map(labels)
    class_keys = [k for k in label_map if k.startswith("class_")]
    class_lists = [
        [idx for idx in label_map[key] if idx in train_indices]
        for key in sorted(class_keys)
    ]
    class_lists = [lst for lst in class_lists if lst]
    negatives = [idx for idx in label_map['negative'] if idx in train_indices]
    if not class_lists:
        return train_indices

    target = min(len(lst) for lst in class_lists)
    if len(class_lists) == 1:
        if not negatives:
            return train_indices
        target = min(target, len(negatives))
    selected = set()
    for lst in class_lists:
        selected.update(rng.sample(lst, target))

    num_negatives = int(target * negative_ratio)
    if negatives and num_negatives > 0:
        selected.update(rng.sample(negatives, min(num_negatives, len(negatives))))

    selected_list = list(selected)
    rng.shuffle(selected_list)
    return selected_list


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    num_epochs: int,
    threshold: float,
    threshold_sweep: bool,
    early_stopping_patience: int,
    early_stopping_delta: float,
    early_stopping_metric: str,
    early_stopping_ema_alpha: float,
    checkpoint_dir: Optional[Path],
    checkpoint_every: int,
) -> Tuple[Dict[str, float], dict]:
    best_monitor = None
    best_metrics: Dict[str, float] = {}
    best_threshold = threshold
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    ema_value = None

    minimize = early_stopping_metric in ("val_loss", "val_loss_ema")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            frames = batch['frames'].to(device)
            labels = batch['class_labels'].to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.size(0)

        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                labels = batch['class_labels'].to(device)
                logits = model(frames)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * frames.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        val_loss /= max(1, len(val_loader.dataset))
        logits_all = torch.cat(all_logits)
        labels_all = torch.cat(all_labels)
        metrics = compute_metrics(logits_all, labels_all, threshold=threshold)
        epoch_best_threshold = threshold
        log_metrics = metrics
        if threshold_sweep:
            best_micro = -1.0
            for t in [i / 20 for i in range(1, 20)]:
                sweep_metrics = compute_metrics(logits_all, labels_all, threshold=t)
                if sweep_metrics['micro_f1'] > best_micro:
                    best_micro = sweep_metrics['micro_f1']
                    epoch_best_threshold = t
                    log_metrics = sweep_metrics

        if early_stopping_metric in ("micro_f1", "micro_f1_ema"):
            raw_monitor = log_metrics['micro_f1']
        else:
            raw_monitor = val_loss

        if early_stopping_metric.endswith("_ema"):
            if ema_value is None:
                ema_value = raw_monitor
            else:
                ema_value = early_stopping_ema_alpha * raw_monitor + (1 - early_stopping_ema_alpha) * ema_value
            monitor_value = ema_value
        else:
            monitor_value = raw_monitor

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_value if minimize else -monitor_value)
            else:
                scheduler.step()

        if best_monitor is None:
            improved = True
        elif minimize:
            improved = monitor_value < best_monitor - early_stopping_delta
        else:
            improved = monitor_value > best_monitor + early_stopping_delta

        if improved:
            best_monitor = monitor_value
            if threshold_sweep:
                best_metrics = log_metrics
                best_threshold = epoch_best_threshold
            else:
                best_metrics = metrics
                best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(
                "Epoch %03d/%d | Train Loss=%.4f | Val Loss=%.4f | Val micro-F1=%.3f | Thr=%.2f",
                epoch, num_epochs, train_loss, val_loss, log_metrics['micro_f1'], epoch_best_threshold
            )
        if checkpoint_dir is not None and checkpoint_every > 0:
            if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": log_metrics,
                    },
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                )

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    best_metrics['best_threshold'] = best_threshold
    logger.info("Best threshold for fold: %.2f", best_threshold)
    for key, value in best_metrics.items():
        if key.endswith('_f1') and key.startswith('class_'):
            logger.info("Fold metric %s=%.3f", key, value)

    return best_metrics, best_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop-based temporal classifier")
    parser.add_argument("--data-dir", type=Path, required=True, help="Ground truth data directory")
    parser.add_argument("--annotations-path", type=Path, required=True, help="annotations.json path")
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128), help="Crop size H,W")
    parser.add_argument("--samples-per-subject", type=int, default=50)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", action="store_true", help="Find best threshold per epoch")
    parser.add_argument("--use-pos-weight", action="store_true", help="Use BCE pos_weight from train data")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training augmentation")
    parser.add_argument("--no-noise", action="store_true", help="Disable Gaussian noise augmentation")
    parser.add_argument("--no-blur", action="store_true", help="Disable blur augmentation")
    parser.add_argument("--noise-sigma-min", type=float, default=2.0, help="Min Gaussian noise sigma")
    parser.add_argument("--noise-sigma-max", type=float, default=8.0, help="Max Gaussian noise sigma")
    parser.add_argument("--noise-prob", type=float, default=0.2, help="Probability of noise augmentation")
    parser.add_argument("--blur-prob", type=float, default=0.3, help="Probability of blur augmentation")
    parser.add_argument("--denoise", action="store_true", help="Enable denoising augmentation")
    parser.add_argument("--denoise-prob", type=float, default=0.1, help="Probability of denoising augmentation")
    parser.add_argument("--denoise-strength", type=float, default=7.0, help="Denoising strength (NLMeans h)")
    parser.add_argument("--adaptive-noise", action="store_true", help="Adapt noise to reach target sigma range")
    parser.add_argument(
        "--threshold-low",
        action="store_true",
        help="Zero out pixels below threshold before noise augmentation",
    )
    parser.add_argument("--threshold-value", type=int, default=125, help="Grayscale threshold value")
    parser.add_argument("--balanced-sampling", action="store_true", help="Balance class sampling in training")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negatives per class in balanced sampling")
    parser.add_argument("--any-object", action="store_true", help="Collapse mover/dipole into one label")
    parser.add_argument("--no-crop", action="store_true", help="Disable crop config for already-cropped frames")
    parser.add_argument(
        "--positive-jitter",
        type=parse_size,
        default=None,
        help="Max jitter (dy,dx) for positive crop centers",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Use a single train/val split with this fraction of subjects for validation",
    )
    parser.add_argument(
        "--val-subjects",
        type=int,
        default=None,
        help="Use a single train/val split with this many subjects for validation",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=30, help="Epochs without val improvement")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-3, help="Min metric delta to reset patience")
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="val_loss_ema",
        choices=["val_loss", "val_loss_ema", "micro_f1", "micro_f1_ema"],
        help="Metric to monitor for early stopping",
    )
    parser.add_argument(
        "--early-stopping-ema-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor when using *_ema metrics",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "softf1", "bce+softf1", "focal"],
        help="Loss function",
    )
    parser.add_argument("--softf1-weight", type=float, default=0.5, help="Weight for SoftF1 in combined loss")
    parser.add_argument("--save-checkpoints", type=Path, default=None, help="Directory to save checkpoints")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "none"],
        help="Learning rate scheduler type",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor for plateau")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs to wait before LR reduction")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for plateau scheduler")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save periodic checkpoints every N epochs (0 disables)",
    )
    parser.add_argument(
        "--negative-bright-fraction",
        type=float,
        default=0.5,
        help="Fraction of negative crops biased toward bright background regions",
    )
    parser.add_argument(
        "--negative-bright-percentile",
        type=float,
        default=85.0,
        help="Brightness percentile used to define bright negative crops",
    )
    parser.add_argument(
        "--negative-bright-samples",
        type=int,
        default=200,
        help="Number of samples to estimate brightness threshold per subject",
    )
    parser.add_argument(
        "--positive-mover-min-frames",
        type=int,
        default=0,
        help="Require mover crops to include at least this many frames (0 disables)",
    )
    args = parser.parse_args()

    logger.info("Command: %s", " ".join(sys.argv))

    if args.save_checkpoints is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_checkpoints = Path("outputs") / "checkpoints" / f"crop_{timestamp}"

    train_transform = None
    if not args.no_augmentation:
        train_transform = TemporalSequenceAugmentation(
            input_size=args.crop_size,
            training=True,
            enable_noise=not args.no_noise,
            noise_sigma_range=(args.noise_sigma_min, args.noise_sigma_max),
            enable_blur=not args.no_blur,
            noise_prob=args.noise_prob,
            blur_prob=args.blur_prob,
            enable_denoise=args.denoise,
            denoise_prob=args.denoise_prob,
            denoise_strength=args.denoise_strength,
            adaptive_noise=args.adaptive_noise,
            enable_threshold=args.threshold_low,
            threshold_value=args.threshold_value,
        )
    val_transform = TemporalSequenceAugmentation(
        input_size=args.crop_size,
        training=False,
        enable_threshold=args.threshold_low,
        threshold_value=args.threshold_value,
    )

    base_dataset = BackyardWorldsTemporalCropDataset(
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        transform=None,
        crop_size=args.crop_size,
        samples_per_subject=args.samples_per_subject,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
        apply_crop=not args.no_crop,
        positive_jitter=args.positive_jitter,
        negative_bright_fraction=args.negative_bright_fraction,
        negative_bright_percentile=args.negative_bright_percentile,
        negative_bright_samples=args.negative_bright_samples,
        positive_mover_min_frames=args.positive_mover_min_frames,
    )

    subject_map = build_subject_index_map(base_dataset.samples)
    sample_class_labels = base_dataset.get_sample_class_labels()
    balance_labels: Optional[List[List[float]]] = None
    if args.balanced_sampling:
        if args.any_object:
            balance_labels = [[1.0] if sum(vec) > 0 else [0.0] for vec in sample_class_labels]
        else:
            balance_labels = sample_class_labels
    subject_ids = list(subject_map.keys())

    use_single_split = args.val_fraction is not None or args.val_subjects is not None
    fold_metrics = []
    split_subject_ids = subject_ids
    if use_single_split:
        rng = random.Random(args.seed)
        split_subject_ids = subject_ids[:]
        rng.shuffle(split_subject_ids)

        if args.val_subjects is not None:
            num_val = max(1, min(args.val_subjects, len(split_subject_ids) - 1))
        else:
            num_val = max(1, int(round(len(split_subject_ids) * args.val_fraction)))

        val_subjects = set(split_subject_ids[:num_val])
        train_subjects = [sid for sid in split_subject_ids if sid not in val_subjects]
        logger.info(
            "Using single split: %d train subjects, %d val subjects",
            len(train_subjects),
            len(val_subjects),
        )

        folds = [(0, None, train_subjects, list(val_subjects))]
    else:
        logger.info("Starting LOOCV across %d subjects", len(subject_ids))
        folds = [(idx, sid, None, [sid]) for idx, sid in enumerate(split_subject_ids)]

    for fold_idx, val_subject, train_subjects, val_subjects in folds:
        if val_subject is None:
            logger.info("Split run | Val subjects=%d", len(val_subjects))
        else:
            logger.info(
                "Fold %d/%d | Val subject=%s",
                fold_idx + 1,
                len(subject_ids),
                val_subject,
            )

        if train_subjects is None:
            train_subjects = [sid for sid in subject_ids if sid not in val_subjects]

        train_indices = [idx for sid in train_subjects for idx in subject_map[sid]]
        val_indices = [idx for sid in val_subjects for idx in subject_map[sid]]

        train_dataset = BackyardWorldsTemporalCropDataset(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path,
            transform=train_transform,
            crop_size=args.crop_size,
            samples_per_subject=args.samples_per_subject,
            positive_fraction=args.positive_fraction,
            seed=args.seed,
            samples=base_dataset.samples,
            apply_crop=not args.no_crop,
            positive_jitter=args.positive_jitter,
            negative_bright_fraction=args.negative_bright_fraction,
            negative_bright_percentile=args.negative_bright_percentile,
            negative_bright_samples=args.negative_bright_samples,
            positive_mover_min_frames=args.positive_mover_min_frames,
        )
        val_dataset = BackyardWorldsTemporalCropDataset(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path,
            transform=val_transform,
            crop_size=args.crop_size,
            samples_per_subject=args.samples_per_subject,
            positive_fraction=args.positive_fraction,
            seed=args.seed,
            samples=base_dataset.samples,
            apply_crop=not args.no_crop,
            positive_jitter=args.positive_jitter,
            negative_bright_fraction=args.negative_bright_fraction,
            negative_bright_percentile=args.negative_bright_percentile,
            negative_bright_samples=args.negative_bright_samples,
            positive_mover_min_frames=args.positive_mover_min_frames,
        )

        if args.balanced_sampling:
            balanced_indices = balanced_sample_indices(
                train_indices,
                balance_labels or sample_class_labels,
                seed=args.seed + fold_idx,
                negative_ratio=args.negative_ratio,
            )
            logger.info(
                "Balanced sampling: %d -> %d samples",
                len(train_indices),
                len(balanced_indices),
            )
            train_subset = Subset(train_dataset, balanced_indices)
        else:
            train_subset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, any_object=args.any_object),
        )
        val_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, any_object=args.any_object),
        )

        model = TemporalCropClassifier(
            num_classes=1 if args.any_object else 2,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(args.device)

        bce_pos_weight = None
        if args.use_pos_weight:
            bce_pos_weight = compute_pos_weight(train_loader).to(args.device)
            logger.info("Using pos_weight=%s", bce_pos_weight.detach().cpu().numpy().round(3))

        if args.loss == "softf1":
            loss_fn = SoftF1Loss()
        elif args.loss == "bce+softf1":
            bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
            loss_fn = CombinedLoss(bce, SoftF1Loss(), args.softf1_weight)
        elif args.loss == "focal":
            loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        if args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min" if args.early_stopping_metric in ("val_loss", "val_loss_ema") else "max",
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.lr_min,
            )

        train_balance = summarize_class_balance(train_loader)
        val_balance = summarize_class_balance(val_loader)
        logger.info("Train balance: %s", train_balance)
        logger.info("Val balance: %s", val_balance)

        metrics, best_state = train_fold(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            args.device,
            args.num_epochs,
            args.threshold,
            args.threshold_sweep,
            args.early_stopping_patience,
            args.early_stopping_delta,
            args.early_stopping_metric,
            args.early_stopping_ema_alpha,
            args.save_checkpoints if args.checkpoint_every > 0 else None,
            args.checkpoint_every,
        )
        fold_metrics.append(metrics)
        if args.save_checkpoints is not None:
            args.save_checkpoints.mkdir(parents=True, exist_ok=True)
            ckpt_name = "split.pt" if use_single_split else f"fold_{fold_idx}.pt"
            ckpt_path = args.save_checkpoints / ckpt_name
            torch.save(
                {
                    "fold": fold_idx,
                    "model_state_dict": best_state,
                    "metrics": metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )

    logger.info("Completed %d runs", len(fold_metrics))
    if fold_metrics:
        avg_micro_f1 = sum(m['micro_f1'] for m in fold_metrics) / len(fold_metrics)
        logger.info("Average micro-F1 across folds: %.3f", avg_micro_f1)
        for class_idx in range(1 if args.any_object else 2):
            key = f'class_{class_idx}_f1'
            avg_class_f1 = sum(m.get(key, 0.0) for m in fold_metrics) / len(fold_metrics)
            logger.info("Average %s across folds: %.3f", key, avg_class_f1)


if __name__ == "__main__":
    main()
