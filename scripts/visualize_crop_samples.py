#!/usr/bin/env python3
"""
Export crop samples as side-by-side frame strips for quick inspection.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scientific_pipelines.core.training.datasets import (
    BackyardWorldsTemporalCropDataset,
    TemporalSequenceAugmentation,
)


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def frames_tensor_to_strip(frames: torch.Tensor) -> np.ndarray:
    # frames shape: (T, C, H, W), RGB, [0,1]
    frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    strip = np.concatenate(list(frames_np), axis=1)
    return strip


def add_label_text(image: np.ndarray, lines: List[str]) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    thickness = 1
    y = 14
    for line in lines:
        cv2.putText(output, line, (6, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += 14
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize crop samples")
    parser.add_argument("--data-dir", type=Path, required=True, help="Ground truth data directory")
    parser.add_argument("--annotations-path", type=Path, required=True, help="annotations.json path")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/crop_samples"))
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128), help="Crop size H,W")
    parser.add_argument("--samples-per-subject", type=int, default=50)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Randomly sample crops")
    parser.add_argument("--augmented", action="store_true", help="Apply training augmentations")
    args = parser.parse_args()

    if args.augmented:
        transform = TemporalSequenceAugmentation(input_size=args.crop_size, training=True)
    else:
        transform = None

    dataset = BackyardWorldsTemporalCropDataset(
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        transform=transform,
        crop_size=args.crop_size,
        samples_per_subject=args.samples_per_subject,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )

    indices = list(range(len(dataset)))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices[:args.num_samples]):
        sample = dataset[idx]
        strip = frames_tensor_to_strip(sample['frames'])
        class_labels = sample['class_labels'].tolist()
        lines = [
            f"id={sample['sequence_id']}",
            f"labels={sample['labels']}",
            f"class_labels={class_labels}",
        ]
        strip = add_label_text(strip, lines)
        out_path = args.out_dir / f"crop_{i:03d}.png"
        bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)


if __name__ == "__main__":
    main()
