#!/usr/bin/env python3
"""
Export scored crop windows for a subject as image strips.

Each strip concatenates the 4 temporal frames for a window.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_pipelines.core.training.datasets.augmentation import TemporalSequenceAugmentation
from scientific_pipelines.core.training.datasets.crop_config import crop_bounds, crop_frames
from scientific_pipelines.core.training.models import TemporalCropClassifier


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def load_frames(frames_dir: Path, apply_crop: bool) -> List[np.ndarray]:
    frames = []
    for i in range(4):
        frame_path = frames_dir / f"frame_{i:02d}.jpg"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    frames = crop_frames(frames) if apply_crop else frames
    return frames


def iter_windows(
    image_size: Tuple[int, int],
    crop_size: Tuple[int, int],
    stride: int,
):
    h, w = image_size
    crop_h, crop_w = crop_size
    for top in range(0, max(1, h - crop_h + 1), stride):
        for left in range(0, max(1, w - crop_w + 1), stride):
            bottom = min(h, top + crop_h)
            right = min(w, left + crop_w)
            top = bottom - crop_h
            left = right - crop_w
            yield left, top, right, bottom


def to_tensor(frames: List[np.ndarray], normalize: bool, crop_size: Tuple[int, int]) -> torch.Tensor:
    if normalize:
        transform = TemporalSequenceAugmentation(input_size=crop_size, training=False)
        frames_tensor, _, _ = transform(frames, [], [])
        return frames_tensor

    frames_tensor = []
    for frame in frames:
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames_tensor.append(frame_tensor)
    return torch.stack(frames_tensor, dim=0)


def build_scores(
    frames: List[np.ndarray],
    model: TemporalCropClassifier,
    crop_size: Tuple[int, int],
    stride: int,
    device: str,
    any_object: bool,
    normalize: bool,
    batch_size: int,
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    h, w = frames[0].shape[:2]
    num_classes = 1 if any_object else 2
    if num_classes != 1:
        raise ValueError("This exporter supports --any-object only.")

    model.eval()
    results: List[Tuple[float, Tuple[int, int, int, int]]] = []
    with torch.no_grad():
        if batch_size < 1:
            batch_size = 1
        batch_tensors = []
        batch_windows = []

        def flush_batch():
            if not batch_tensors:
                return
            batch = torch.cat(batch_tensors, dim=0).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 1:
                probs = probs[:, None]
            for i, window in enumerate(batch_windows):
                results.append((float(probs[i, 0]), window))
            batch_tensors.clear()
            batch_windows.clear()

        for left, top, right, bottom in iter_windows((h, w), crop_size, stride):
            crop_frames = [f[top:bottom, left:right].copy() for f in frames]
            crop_tensor = to_tensor(crop_frames, normalize, crop_size).unsqueeze(0)
            batch_tensors.append(crop_tensor)
            batch_windows.append((left, top, right, bottom))
            if len(batch_tensors) >= batch_size:
                flush_batch()
        flush_batch()
    return results


def make_strip(
    frames: List[np.ndarray],
    window: Tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = window
    crops = [f[top:bottom, left:right] for f in frames]
    return np.concatenate(crops, axis=1)


def save_strip(path: Path, strip: np.ndarray) -> None:
    bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def tensor_to_view(frames_tensor: torch.Tensor) -> List[np.ndarray]:
    # Map normalized tensors to 0-255 for visualization.
    frames = []
    for i in range(frames_tensor.shape[0]):
        frame = frames_tensor[i].permute(1, 2, 0).cpu().numpy()
        vmin = float(frame.min())
        vmax = float(frame.max())
        if vmax - vmin < 1e-6:
            scaled = np.zeros_like(frame)
        else:
            scaled = (frame - vmin) / (vmax - vmin)
        frame_u8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        frames.append(frame_u8)
    return frames


def make_strip_from_tensor(frames_tensor: torch.Tensor) -> np.ndarray:
    frames = tensor_to_view(frames_tensor)
    return np.concatenate(frames, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scored crop windows for a subject")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128))
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--no-crop", action="store_true", help="Disable crop config for already-cropped frames")
    parser.add_argument("--any-object", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", help="Apply ImageNet normalization")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--bottom-k", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/crop_windows"))
    parser.add_argument(
        "--save-normalized-view",
        action="store_true",
        help="Save normalized-input visualization strips alongside raw crops",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    subjects_dir = args.data_dir / "subjects_groundtruth"
    if not subjects_dir.exists():
        subjects_alt = args.data_dir / "subjects"
        if subjects_alt.exists():
            subjects_dir = subjects_alt
        else:
            subjects_dir = args.data_dir
    frames_dir = subjects_dir / args.subject_id

    frames = load_frames(frames_dir, apply_crop=not args.no_crop)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model = TemporalCropClassifier(num_classes=1, base_channels=16)
    model.load_state_dict(state)
    model.to(args.device)

    scores = build_scores(
        frames=frames,
        model=model,
        crop_size=args.crop_size,
        stride=args.stride,
        device=args.device,
        any_object=args.any_object,
        normalize=args.normalize,
        batch_size=args.batch_size,
    )

    scores_sorted = sorted(scores, key=lambda s: s[0])
    top = list(reversed(scores_sorted[-args.top_k:])) if args.top_k > 0 else []
    bottom = scores_sorted[:args.bottom_k] if args.bottom_k > 0 else []

    out_dir = args.out_dir / args.subject_id
    top_dir = out_dir / "top"
    bottom_dir = out_dir / "bottom"
    top_dir.mkdir(parents=True, exist_ok=True)
    bottom_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "left", "top", "right", "bottom"])
        for score, (left, top_, right, bottom_) in scores_sorted:
            writer.writerow([f"{score:.6f}", left, top_, right, bottom_])

    for idx, (score, window) in enumerate(top):
        strip = make_strip(frames, window)
        left, top_, right, bottom_ = window
        out_path = top_dir / f"{idx:02d}_score_{score:.4f}_x{left}_y{top_}.png"
        save_strip(out_path, strip)
        if args.save_normalized_view:
            crop_frames = [f[top_:bottom_, left:right].copy() for f in frames]
            crop_tensor = to_tensor(crop_frames, True, args.crop_size)
            norm_strip = make_strip_from_tensor(crop_tensor)
            norm_path = top_dir / f"{idx:02d}_score_{score:.4f}_x{left}_y{top_}_norm.png"
            save_strip(norm_path, norm_strip)

    for idx, (score, window) in enumerate(bottom):
        strip = make_strip(frames, window)
        left, top_, right, bottom_ = window
        out_path = bottom_dir / f"{idx:02d}_score_{score:.4f}_x{left}_y{top_}.png"
        save_strip(out_path, strip)
        if args.save_normalized_view:
            crop_frames = [f[top_:bottom_, left:right].copy() for f in frames]
            crop_tensor = to_tensor(crop_frames, True, args.crop_size)
            norm_strip = make_strip_from_tensor(crop_tensor)
            norm_path = bottom_dir / f"{idx:02d}_score_{score:.4f}_x{left}_y{top_}_norm.png"
            save_strip(norm_path, norm_strip)


if __name__ == "__main__":
    main()
