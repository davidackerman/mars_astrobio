#!/usr/bin/env python3
"""
Run crop classifier over a full sequence and export heatmap overlays.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scientific_pipelines.core.training.models import TemporalCropClassifier
from scientific_pipelines.core.training.datasets.augmentation import TemporalSequenceAugmentation
from scientific_pipelines.core.training.datasets.crop_config import crop_frames


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def load_frames(frames_dir: Path) -> List[np.ndarray]:
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
    return crop_frames(frames)


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


def build_score_map(
    frames: List[np.ndarray],
    model: TemporalCropClassifier,
    crop_size: Tuple[int, int],
    stride: int,
    device: str,
    any_object: bool,
    normalize: bool,
) -> np.ndarray:
    h, w = frames[0].shape[:2]
    num_classes = 1 if any_object else 2
    score_map = np.zeros((num_classes, h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for left, top, right, bottom in iter_windows((h, w), crop_size, stride):
            crop_frames = [f[top:bottom, left:right].copy() for f in frames]
            crop_tensor = to_tensor(crop_frames, normalize, crop_size).unsqueeze(0).to(device)
            logits = model(crop_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            if any_object:
                probs = np.array([float(probs[0])], dtype=np.float32)
            for class_idx in range(num_classes):
                score_map[class_idx, top:bottom, left:right] = np.maximum(
                    score_map[class_idx, top:bottom, left:right],
                    probs[class_idx],
                )

    return score_map


def extract_peaks(score_map: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    binary = (score_map >= threshold).astype(np.uint8)
    if binary.max() == 0:
        return []
    kernel = np.ones((7, 7), np.uint8)
    max_map = cv2.dilate(score_map, kernel)
    peaks = (score_map == max_map) & (binary > 0)
    ys, xs = np.where(peaks)
    return list(zip(xs.tolist(), ys.tolist()))


def build_overlay(
    frame: np.ndarray,
    score_map: np.ndarray,
    peaks: List[Tuple[int, int]],
):
    heat = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.7, heat_color, 0.3, 0)
    for x, y in peaks:
        cv2.circle(overlay, (x, y), 6, (0, 255, 255), 1)
    return overlay


def save_overlay(
    frame: np.ndarray,
    score_map: np.ndarray,
    peaks: List[Tuple[int, int]],
    out_path: Path,
):
    overlay = build_overlay(frame, score_map, peaks)
    cv2.imwrite(str(out_path), overlay)


def save_side_by_side(
    frame: np.ndarray,
    overlay: np.ndarray,
    out_path: Path,
):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    side = np.concatenate([frame_bgr, overlay], axis=1)
    cv2.imwrite(str(out_path), side)


def save_raw_heatmap(score_map: np.ndarray, out_path: Path) -> None:
    heat = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), heat)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer crop classifier over full images")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/crop_inference"))
    parser.add_argument("--subject-id", type=str, default=None, help="Optional subject ID to process")
    parser.add_argument("--crop-size", type=parse_size, default=(128, 128))
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--any-object", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Apply ImageNet normalization")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-raw-heatmap", action="store_true", help="Save raw heatmap as grayscale PNG")
    parser.add_argument(
        "--side-by-side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save original and overlay side-by-side",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    model = TemporalCropClassifier(
        num_classes=1 if args.any_object else 2,
        base_channels=args.base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    subjects_dir = args.data_dir / "subjects_groundtruth"
    subject_ids = [args.subject_id] if args.subject_id else sorted(p.name for p in subjects_dir.iterdir() if p.is_dir())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in subject_ids:
        frames_dir = subjects_dir / subject_id
        frames = load_frames(frames_dir)
        score_map = build_score_map(
            frames,
            model,
            args.crop_size,
            args.stride,
            args.device,
            args.any_object,
            args.normalize,
        )
        for class_idx in range(score_map.shape[0]):
            peaks = extract_peaks(score_map[class_idx], args.threshold)
            out_path = args.out_dir / f"{subject_id}_class_{class_idx}_overlay.png"
            overlay = build_overlay(frames[0], score_map[class_idx], peaks)
            cv2.imwrite(str(out_path), overlay)
            if args.side_by_side:
                side_path = args.out_dir / f"{subject_id}_class_{class_idx}_side_by_side.png"
                save_side_by_side(frames[0], overlay, side_path)
            if args.save_raw_heatmap:
                raw_path = args.out_dir / f"{subject_id}_class_{class_idx}_raw.png"
                save_raw_heatmap(score_map[class_idx], raw_path)


if __name__ == "__main__":
    main()
