#!/usr/bin/env python3
"""
Sample training crops and save them as GIFs, grouped by category.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_pipelines.core.training.datasets import TemporalSequenceAugmentation
from scientific_pipelines.core.training.datasets.temporal_crop_dataset import (
    BackyardWorldsTemporalCropDataset,
)


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def tensor_to_uint8(frames_tensor: torch.Tensor, normalized: bool) -> np.ndarray:
    frames = frames_tensor.detach().cpu()
    if normalized:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        frames = frames * std + mean
    frames = frames.clamp(0.0, 1.0)
    frames = (frames * 255.0).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).numpy()


def classify_sample(
    dataset: BackyardWorldsTemporalCropDataset,
    subject_id: str,
    center: Tuple[float, float],
    labels: List[int],
) -> Tuple[str, float, float]:
    if labels:
        has_mover = 0 in labels
        has_dipole = 1 in labels
        if has_mover and has_dipole:
            return "object_mixed", 0.0, 0.0
        if has_mover:
            return "object_mover", 0.0, 0.0
        if has_dipole:
            return "object_dipole", 0.0, 0.0
        return "object_other", 0.0, 0.0

    integral = dataset.subject_brightness_integrals.get(subject_id)
    threshold = dataset.subject_bright_thresholds.get(subject_id, 0.0)
    image_size = dataset.subject_sizes[subject_id]
    if integral is None:
        return "background", 0.0, threshold
    mean_brightness = dataset._crop_mean_brightness(integral, center, image_size)
    if mean_brightness >= threshold:
        return "background_bright", mean_brightness, threshold
    return "background", mean_brightness, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample training crops and save GIFs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/training_crop_samples"))
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128))
    parser.add_argument("--samples-per-subject", type=int, default=10)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--positive-jitter", type=parse_size, default=None)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--negative-bright-fraction", type=float, default=0.5)
    parser.add_argument("--negative-bright-percentile", type=float, default=85.0)
    parser.add_argument("--negative-bright-samples", type=int, default=200)
    parser.add_argument("--num-per-class", type=int, default=50)
    parser.add_argument("--gif-duration", type=float, default=0.8)
    args = parser.parse_args()

    transform = None
    normalized = False
    if args.use_augmentation:
        transform = TemporalSequenceAugmentation(
            input_size=args.crop_size,
            training=True,
            enable_noise=not args.no_noise,
            enable_blur=not args.no_blur,
        )
        normalized = True

    dataset = BackyardWorldsTemporalCropDataset(
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        transform=transform,
        crop_size=args.crop_size,
        samples_per_subject=args.samples_per_subject,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
        apply_crop=not args.no_crop,
        positive_jitter=args.positive_jitter,
        negative_bright_fraction=args.negative_bright_fraction,
        negative_bright_percentile=args.negative_bright_percentile,
        negative_bright_samples=args.negative_bright_samples,
    )

    categories = [
        "object_mover",
        "object_dipole",
        "object_mixed",
        "background_bright",
        "background",
    ]
    counts: Dict[str, int] = {key: 0 for key in categories}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        (args.out_dir / category).mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        labels = sample["labels"]
        center = sample["crop_center"]
        subject_id = sample["sequence_id"]
        category, brightness, threshold = classify_sample(dataset, subject_id, center, labels)
        if category not in counts:
            continue
        if counts[category] >= args.num_per_class:
            if all(counts[c] >= args.num_per_class for c in categories):
                break
            continue

        frames_u8 = tensor_to_uint8(sample["frames"], normalized)
        out_path = args.out_dir / category / f"{subject_id}_idx_{idx:05d}.gif"
        with imageio.get_writer(out_path, mode="I", duration=args.gif_duration, loop=0) as writer:
            for frame in frames_u8:
                writer.append_data(frame)

        rows.append(
            {
                "subject_id": subject_id,
                "sample_idx": idx,
                "category": category,
                "center_x": f"{center[0]:.2f}",
                "center_y": f"{center[1]:.2f}",
                "labels": ",".join(str(x) for x in labels),
                "mean_brightness": f"{brightness:.3f}",
                "bright_threshold": f"{threshold:.3f}",
                "gif_path": str(out_path),
            }
        )
        counts[category] += 1

    csv_path = args.out_dir / "crop_sample_index.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "sample_idx",
                "category",
                "center_x",
                "center_y",
                "labels",
                "mean_brightness",
                "bright_threshold",
                "gif_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Saved samples to {args.out_dir} ({summary})")


if __name__ == "__main__":
    main()
