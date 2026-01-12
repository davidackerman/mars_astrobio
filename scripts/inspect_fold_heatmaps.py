#!/usr/bin/env python3
"""
Visualize predicted vs. target heatmaps for a given fold checkpoint.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scientific_pipelines.core.training.models import TemporalObjectDetector
from scientific_pipelines.core.training.datasets import BackyardWorldsTemporalDataset


def save_heatmap_image(heatmap: torch.Tensor, path: Path) -> None:
    hm = heatmap.numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    img = (hm * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fold heatmaps for a single sample.")
    parser.add_argument("--run", required=True, help="Run directory name under checkpoints/temporal_detector/")
    parser.add_argument("--fold", type=int, default=3, help="Fold index (0-based)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/backyard_worlds/ground_truth"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for images")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    args = parser.parse_args()

    run_dir = Path("checkpoints/temporal_detector") / args.run
    ckpt_path = run_dir / f"fold_{args.fold}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    annotations_path = args.data_dir / "annotations.json"

    out_dir = args.out_dir or Path("logs/temporal_detector") / args.run / "diagnostics" / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BackyardWorldsTemporalDataset(
        data_dir=args.data_dir,
        annotations_path=annotations_path,
        transform=None,
        heatmap_size=(8, 8),
        input_size=(256, 256),
    )

    if args.fold >= len(dataset):
        raise ValueError(f"Fold index {args.fold} out of range for dataset size {len(dataset)}")

    sample = dataset[args.fold]
    sequence_id = sample["sequence_id"]

    model = TemporalObjectDetector(
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
        heatmap_size=(8, 8),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frames = sample["frames"].unsqueeze(0)
    with torch.no_grad():
        class_logits, heatmaps = model(frames)
        pred_maps = torch.sigmoid(heatmaps)[0]
        target_maps = sample["heatmap_targets"]

    pred_up = F.interpolate(pred_maps.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
    target_up = F.interpolate(target_maps.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]

    frame0 = sample["frames"][0].permute(1, 2, 0).numpy() * 255.0
    frame0 = frame0.clip(0, 255).astype(np.uint8)

    class_names = ["mover", "dipole"]
    for class_id, name in enumerate(class_names):
        save_heatmap_image(pred_up[class_id], out_dir / f"{sequence_id}_pred_{name}.png")
        save_heatmap_image(target_up[class_id], out_dir / f"{sequence_id}_target_{name}.png")

    pred_sum = pred_up.sum(dim=0).numpy()
    pred_sum = (pred_sum - pred_sum.min()) / (pred_sum.max() - pred_sum.min() + 1e-6)
    heat_rgb = np.stack([pred_sum, np.zeros_like(pred_sum), np.zeros_like(pred_sum)], axis=-1)
    overlay = (0.7 * frame0 / 255.0 + 0.3 * heat_rgb)
    overlay_img = (overlay * 255).astype(np.uint8)
    Image.fromarray(overlay_img).save(out_dir / f"{sequence_id}_pred_overlay.png")

    frame_gt = Image.fromarray(frame0.copy())
    draw = ImageDraw.Draw(frame_gt)
    for (x, y), label in zip(sample["keypoints"], sample["labels"]):
        r = 6
        color = (0, 255, 0) if label == 0 else (0, 0, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
    frame_gt.save(out_dir / f"{sequence_id}_gt_keypoints.png")

    print(f"Sequence: {sequence_id}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
