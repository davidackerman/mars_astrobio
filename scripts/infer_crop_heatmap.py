#!/usr/bin/env python3
"""
Run crop classifier over a full sequence and export heatmap overlays.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scientific_pipelines.core.training.models import TemporalCropClassifier
from scientific_pipelines.core.training.datasets.augmentation import TemporalSequenceAugmentation
from scientific_pipelines.core.training.datasets.crop_config import crop_bounds, crop_frames


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be H,W")
    return int(parts[0]), int(parts[1])


def load_frames(frames_dir: Path, apply_crop: bool) -> Tuple[List[np.ndarray], Tuple[int, int]]:
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
    original_size = frames[0].shape[:2]
    frames = crop_frames(frames) if apply_crop else frames
    return frames, original_size


def has_required_frames(frames_dir: Path) -> bool:
    for i in range(4):
        if not (frames_dir / f"frame_{i:02d}.jpg").exists():
            return False
    return True


def load_annotations(annotations_path: Path) -> dict:
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    with annotations_path.open() as f:
        return json.load(f)


def _parse_boxes(value) -> List[List[float]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value or []


def get_annotation_boxes(annotations: dict, subject_id: str) -> dict:
    info = annotations.get(subject_id, {}) if annotations else {}
    mover = _parse_boxes(info.get("mover_circles"))
    dipole = _parse_boxes(info.get("dipole_circles"))
    artifact = _parse_boxes(info.get("artifact_circles"))

    if not mover and not dipole and not artifact:
        mover = _parse_boxes(info.get("mover_boxes"))
        dipole = _parse_boxes(info.get("dipole_boxes"))
        artifact = _parse_boxes(info.get("artifact_boxes"))

    if not mover and not dipole and not artifact:
        circles = _parse_boxes(info.get("circles")) or _parse_boxes(info.get("boxes"))
        obj_type = info.get("type", "")
        if obj_type == "mover":
            mover = circles
        elif obj_type == "dipole":
            dipole = circles
        elif obj_type == "artifact":
            artifact = circles

    return {"mover": mover or [], "dipole": dipole or [], "artifact": artifact or []}


def adjust_boxes_for_crop(
    boxes: List[List[float]],
    image_size: Tuple[int, int],
    apply_crop: bool,
) -> List[List[float]]:
    if not apply_crop or not boxes:
        return boxes
    left, top, right, bottom = crop_bounds(image_size)
    adjusted = []
    for x, y, w, h in boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        if left <= cx < right and top <= cy < bottom:
            adjusted.append([x - left, y - top, w, h])
    return adjusted


def draw_annotation_circles(
    frame_bgr: np.ndarray,
    boxes: List[List[float]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for x, y, w, h in boxes:
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        r = int(round(max(w, h) / 2.0))
        cv2.circle(frame_bgr, (cx, cy), r, color, thickness)


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


def to_tensor(
    frames: List[np.ndarray],
    normalize: bool,
    crop_size: Tuple[int, int],
    threshold_low: bool,
    threshold_value: int,
) -> torch.Tensor:
    if normalize:
        transform = TemporalSequenceAugmentation(
            input_size=crop_size,
            training=False,
            enable_threshold=threshold_low,
            threshold_value=threshold_value,
        )
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
    threshold_low: bool,
    threshold_value: int,
    batch_size: int,
) -> np.ndarray:
    h, w = frames[0].shape[:2]
    num_classes = 1 if any_object else 2
    score_map = np.zeros((num_classes, h, w), dtype=np.float32)

    model.eval()
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
            for i, (left, top, right, bottom) in enumerate(batch_windows):
                for class_idx in range(num_classes):
                    score_map[class_idx, top:bottom, left:right] = np.maximum(
                        score_map[class_idx, top:bottom, left:right],
                        probs[i, class_idx],
                    )
            batch_tensors.clear()
            batch_windows.clear()

        for left, top, right, bottom in iter_windows((h, w), crop_size, stride):
            crop_frames = [f[top:bottom, left:right].copy() for f in frames]
            crop_tensor = to_tensor(
                crop_frames,
                normalize,
                crop_size,
                threshold_low,
                threshold_value,
            ).unsqueeze(0)
            batch_tensors.append(crop_tensor)
            batch_windows.append((left, top, right, bottom))
            if len(batch_tensors) >= batch_size:
                flush_batch()
        flush_batch()

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
    alpha: float,
    contour_level: float | None,
    contour_color: Tuple[int, int, int],
    contour_thickness: int,
    annotation_boxes: dict | None,
    annotation_thickness: int,
):
    heat = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET).astype(np.float32)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha_map = np.clip(score_map, 0.0, 1.0) * alpha
    alpha_map = alpha_map[:, :, None]
    overlay = frame_bgr * (1.0 - alpha_map) + heat_color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    for x, y in peaks:
        cv2.circle(overlay, (x, y), 6, (0, 255, 255), 1)
    if contour_level is not None:
        mask = (score_map >= contour_level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness)
    if annotation_boxes:
        draw_annotation_circles(overlay, annotation_boxes.get("mover", []), (255, 255, 255), annotation_thickness)
        draw_annotation_circles(overlay, annotation_boxes.get("dipole", []), (60, 170, 255), annotation_thickness)
        draw_annotation_circles(overlay, annotation_boxes.get("artifact", []), (255, 0, 255), annotation_thickness)
    return overlay


def save_overlay(
    frame: np.ndarray,
    score_map: np.ndarray,
    peaks: List[Tuple[int, int]],
    alpha: float,
    contour_level: float | None,
    contour_color: Tuple[int, int, int],
    contour_thickness: int,
    annotation_boxes: dict | None,
    annotation_thickness: int,
    out_path: Path,
):
    overlay = build_overlay(
        frame,
        score_map,
        peaks,
        alpha,
        contour_level,
        contour_color,
        contour_thickness,
        annotation_boxes,
        annotation_thickness,
    )
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
    parser.add_argument("--no-crop", action="store_true", help="Disable crop config for already-cropped frames")
    parser.add_argument("--any-object", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Apply ImageNet normalization")
    parser.add_argument("--threshold-low", action="store_true", help="Apply grayscale thresholding")
    parser.add_argument("--threshold-value", type=int, default=125, help="Grayscale threshold value")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for window inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-raw-heatmap", action="store_true", help="Save raw heatmap as grayscale PNG")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Max overlay alpha for heatmap (0-1)")
    parser.add_argument(
        "--contour-level",
        type=float,
        default=None,
        help="Optional score threshold to draw boundary contours",
    )
    parser.add_argument(
        "--contour-color",
        type=str,
        default="0,255,255",
        help="Contour color as B,G,R",
    )
    parser.add_argument("--contour-thickness", type=int, default=1)
    parser.add_argument("--annotations-path", type=Path, default=None)
    parser.add_argument(
        "--draw-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw ground-truth annotation circles if available",
    )
    parser.add_argument("--annotation-thickness", type=int, default=2)
    parser.add_argument(
        "--side-by-side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save original and overlay side-by-side",
    )
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save overlay animation as GIF (requires imageio)",
    )
    args = parser.parse_args()
    contour_color = tuple(int(x) for x in args.contour_color.split(","))
    annotations = None
    if args.draw_annotations:
        annotations_path = args.annotations_path or (args.data_dir / "annotations.json")
        try:
            annotations = load_annotations(annotations_path)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Disabling annotation overlays.")
            args.draw_annotations = False

    checkpoint = load_checkpoint(args.checkpoint)
    model = TemporalCropClassifier(
        num_classes=1 if args.any_object else 2,
        base_channels=args.base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    subjects_dir = args.data_dir / "subjects_groundtruth"
    if not subjects_dir.exists():
        subjects_alt = args.data_dir / "subjects"
        if subjects_alt.exists():
            subjects_dir = subjects_alt
        else:
            subjects_dir = args.data_dir
    subject_ids = [args.subject_id] if args.subject_id else sorted(p.name for p in subjects_dir.iterdir() if p.is_dir())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for subject_id in subject_ids:
        frames_dir = subjects_dir / subject_id
        if not has_required_frames(frames_dir):
            print(f"Skipping {subject_id}: missing one or more frame_00..03.jpg files")
            continue
        frames, original_size = load_frames(frames_dir, apply_crop=not args.no_crop)
        score_map = build_score_map(
            frames,
            model,
            args.crop_size,
            args.stride,
            args.device,
            args.any_object,
            args.normalize,
            args.threshold_low,
            args.threshold_value,
            args.batch_size,
        )
        annotation_boxes = None
        if args.draw_annotations:
            annotation_boxes = get_annotation_boxes(annotations, subject_id)
            annotation_boxes = {
                key: adjust_boxes_for_crop(boxes, original_size, apply_crop=not args.no_crop)
                for key, boxes in annotation_boxes.items()
            }
        for class_idx in range(score_map.shape[0]):
            peak_score = float(score_map[class_idx].max())
            filename_prefix = f"{peak_score:.4f}_"
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "class_idx": class_idx,
                    "peak_score": peak_score,
                }
            )
            peaks = extract_peaks(score_map[class_idx], args.threshold)
            out_path = args.out_dir / f"{filename_prefix}{subject_id}_class_{class_idx}_overlay.png"
            overlay = build_overlay(
                frames[0],
                score_map[class_idx],
                peaks,
                args.overlay_alpha,
                args.contour_level,
                contour_color,
                args.contour_thickness,
                annotation_boxes,
                args.annotation_thickness,
            )
            cv2.imwrite(str(out_path), overlay)
            if args.side_by_side:
                side_path = args.out_dir / f"{filename_prefix}{subject_id}_class_{class_idx}_side_by_side.png"
                save_side_by_side(frames[0], overlay, side_path)
            if args.save_raw_heatmap:
                raw_path = args.out_dir / f"{filename_prefix}{subject_id}_class_{class_idx}_raw.png"
                save_raw_heatmap(score_map[class_idx], raw_path)
            if args.save_gif:
                try:
                    import imageio.v2 as imageio
                except ImportError as exc:
                    raise SystemExit("imageio is required for --save-gif") from exc
                gif_path = args.out_dir / f"{filename_prefix}{subject_id}_class_{class_idx}_overlay.gif"
                gif_frames = []
                for frame in frames:
                    gif_overlay = build_overlay(
                        frame,
                        score_map[class_idx],
                        [],
                        args.overlay_alpha,
                        args.contour_level,
                        contour_color,
                        args.contour_thickness,
                        annotation_boxes,
                        args.annotation_thickness,
                    )
                    gif_frames.append(cv2.cvtColor(gif_overlay, cv2.COLOR_BGR2RGB))
                # Use writer to embed frame durations reliably for GIF viewers.
                with imageio.get_writer(gif_path, mode="I", duration=2.5, loop=0) as writer:
                    for frame in gif_frames:
                        writer.append_data(frame)

    if summary_rows:
        summary_path = args.out_dir / "heatmap_peaks.csv"
        summary_rows.sort(key=lambda row: row["peak_score"], reverse=True)
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject_id", "class_idx", "peak_score"])
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
