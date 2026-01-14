#!/usr/bin/env python3
"""
Run inference with a trained crop classifier on one subject.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_pipelines.core.training.datasets import (
    BackyardWorldsTemporalCropDataset,
    TemporalSequenceAugmentation,
)
from scientific_pipelines.core.training.models import TemporalCropClassifier


def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch])
    class_labels = torch.stack([item['class_labels'] for item in batch])
    crop_centers = [item['crop_center'] for item in batch]
    crop_origins = [item['crop_origin'] for item in batch]
    keypoints = [item['keypoints'] for item in batch]
    labels = [item['labels'] for item in batch]
    sequence_ids = [item['sequence_id'] for item in batch]
    return {
        'frames': frames,
        'class_labels': class_labels,
        'crop_center': crop_centers,
        'crop_origin': crop_origins,
        'keypoints': keypoints,
        'labels': labels,
        'sequence_id': sequence_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for crop classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument("--crop-size", type=str, default="128,128")
    parser.add_argument("--samples-per-subject", type=int, default=50)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--any-object", action="store_true")
    args = parser.parse_args()

    h, w = [int(x) for x in args.crop_size.split(",")]
    transform = TemporalSequenceAugmentation(input_size=(h, w), training=False)

    dataset = BackyardWorldsTemporalCropDataset(
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        transform=transform,
        crop_size=(h, w),
        samples_per_subject=args.samples_per_subject,
        positive_fraction=args.positive_fraction,
        seed=0,
    )

    subject_indices: List[int] = [
        idx for idx, sample in enumerate(dataset.samples)
        if sample['subject_id'] == args.subject_id
    ]
    if not subject_indices:
        raise SystemExit(f"No samples found for subject {args.subject_id}")

    loader = DataLoader(
        Subset(dataset, subject_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    num_classes = 1 if args.any_object else 2
    model = TemporalCropClassifier(
        num_classes=num_classes,
        base_channels=16,
        dropout=0.4,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(args.device)
            logits = model(frames)
            probs = sigmoid(logits).cpu()
            for i in range(probs.size(0)):
                seq_id = batch['sequence_id'][i]
                center = batch['crop_center'][i]
                true_labels = batch['class_labels'][i].tolist()
                p = probs[i].tolist()
                pred = [1 if v >= args.threshold else 0 for v in p]
                print(f"{seq_id} center={center} probs={p} pred={pred} true={true_labels}")


if __name__ == "__main__":
    main()
