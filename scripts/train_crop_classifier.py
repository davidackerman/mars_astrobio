#!/usr/bin/env python3
"""
Train a crop-based temporal classifier with leave-one-subject-out CV.

This trains a lightweight 3D CNN on random crops and positive-centered crops
to predict mover/dipole presence per crop.
"""

import argparse
import logging
import random
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
    device: str,
    num_epochs: int,
    threshold: float,
    threshold_sweep: bool,
) -> Dict[str, float]:
    best_val_loss = float('inf')
    best_metrics: Dict[str, float] = {}
    best_threshold = threshold

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
        if threshold_sweep:
            best_micro = -1.0
            for t in [i / 20 for i in range(1, 20)]:
                sweep_metrics = compute_metrics(logits_all, labels_all, threshold=t)
                if sweep_metrics['micro_f1'] > best_micro:
                    best_micro = sweep_metrics['micro_f1']
                    epoch_best_threshold = t

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if threshold_sweep:
                best_metrics = compute_metrics(logits_all, labels_all, threshold=epoch_best_threshold)
                best_threshold = epoch_best_threshold
            else:
                best_metrics = metrics
                best_threshold = threshold

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(
                "Epoch %03d/%d | Train Loss=%.4f | Val Loss=%.4f | Val micro-F1=%.3f | Thr=%.2f",
                epoch, num_epochs, train_loss, val_loss, metrics['micro_f1'], epoch_best_threshold
            )

    best_metrics['best_threshold'] = best_threshold
    logger.info("Best threshold for fold: %.2f", best_threshold)
    for key, value in best_metrics.items():
        if key.endswith('_f1') and key.startswith('class_'):
            logger.info("Fold metric %s=%.3f", key, value)

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop-based temporal classifier")
    parser.add_argument("--data-dir", type=Path, required=True, help="Ground truth data directory")
    parser.add_argument("--annotations-path", type=Path, required=True, help="annotations.json path")
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128), help="Crop size H,W")
    parser.add_argument("--samples-per-subject", type=int, default=50)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", action="store_true", help="Find best threshold per epoch")
    parser.add_argument("--use-pos-weight", action="store_true", help="Use BCE pos_weight from train data")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training augmentation")
    parser.add_argument("--balanced-sampling", action="store_true", help="Balance class sampling in training")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negatives per class in balanced sampling")
    parser.add_argument("--any-object", action="store_true", help="Collapse mover/dipole into one label")
    args = parser.parse_args()

    train_transform = None
    if not args.no_augmentation:
        train_transform = TemporalSequenceAugmentation(
            input_size=args.crop_size,
            training=True,
        )
    val_transform = TemporalSequenceAugmentation(
        input_size=args.crop_size,
        training=False,
    )

    base_dataset = BackyardWorldsTemporalCropDataset(
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        transform=None,
        crop_size=args.crop_size,
        samples_per_subject=args.samples_per_subject,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
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

    logger.info("Starting LOOCV across %d subjects", len(subject_ids))

    fold_metrics = []
    for fold_idx, val_subject in enumerate(subject_ids):
        logger.info("Fold %d/%d | Val subject=%s", fold_idx + 1, len(subject_ids), val_subject)

        train_indices = [
            idx for subject_id in subject_ids if subject_id != val_subject
            for idx in subject_map[subject_id]
        ]
        val_indices = subject_map[val_subject]

        train_dataset = BackyardWorldsTemporalCropDataset(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path,
            transform=train_transform,
            crop_size=args.crop_size,
            samples_per_subject=args.samples_per_subject,
            positive_fraction=args.positive_fraction,
            seed=args.seed,
            samples=base_dataset.samples,
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
        ).to(args.device)

        if args.use_pos_weight:
            pos_weight = compute_pos_weight(train_loader).to(args.device)
            logger.info("Using pos_weight=%s", pos_weight.detach().cpu().numpy().round(3))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        train_balance = summarize_class_balance(train_loader)
        val_balance = summarize_class_balance(val_loader)
        logger.info("Train balance: %s", train_balance)
        logger.info("Val balance: %s", val_balance)

        metrics = train_fold(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            args.device,
            args.num_epochs,
            args.threshold,
            args.threshold_sweep,
        )
        fold_metrics.append(metrics)

    logger.info("Completed %d folds", len(fold_metrics))
    if fold_metrics:
        avg_micro_f1 = sum(m['micro_f1'] for m in fold_metrics) / len(fold_metrics)
        logger.info("Average micro-F1 across folds: %.3f", avg_micro_f1)
        for class_idx in range(1 if args.any_object else 2):
            key = f'class_{class_idx}_f1'
            avg_class_f1 = sum(m.get(key, 0.0) for m in fold_metrics) / len(fold_metrics)
            logger.info("Average %s across folds: %.3f", key, avg_class_f1)


if __name__ == "__main__":
    main()
