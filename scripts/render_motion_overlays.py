#!/usr/bin/env python3
"""Render looping videos with a circle over likely movers."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _read_frames_from_paths(frame_paths: List[Path]) -> List[np.ndarray]:
    frames = []
    target_size = None
    for path in frame_paths:
        if not path.exists():
            return []
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return []
        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return frames


def _compute_motion_circles(
    frames: List[np.ndarray],
    max_circles: int,
    min_blob_area: int,
    threshold_quantile: float,
    suppress_bright_percentile: float,
    bright_component_percentile: float,
    min_radius: int,
    max_radius: int,
    min_peak_value: int,
) -> List[Tuple[Tuple[int, int], int]]:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    heatmap = np.zeros_like(gray_frames[0], dtype=np.float32)
    for i in range(len(gray_frames) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1]).astype(np.float32)
        heatmap += diff

    nonzero = heatmap[heatmap > 0]
    if nonzero.size == 0:
        return []

    if suppress_bright_percentile < 100.0:
        clip_value = float(np.percentile(nonzero, suppress_bright_percentile))
        heatmap = np.minimum(heatmap, clip_value)

    heatmap_u8 = np.clip(heatmap / (heatmap.max() + 1e-6) * 255.0, 0, 255).astype(
        np.uint8
    )
    if heatmap_u8.max() < min_peak_value:
        return []

    nonzero_u8 = heatmap_u8[heatmap_u8 > 0]
    if nonzero_u8.size == 0:
        return []

    thresh_val = float(np.quantile(nonzero_u8, threshold_quantile))
    binary = (heatmap_u8 >= thresh_val).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    bright_clip = float(
        np.percentile(nonzero_u8, bright_component_percentile)
    )

    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob_area:
            continue
        mask = labels == label
        mean_intensity = float(heatmap_u8[mask].mean())
        if mean_intensity > bright_clip:
            continue
        cx, cy = centroids[label]
        radius = int(np.clip(np.sqrt(area), min_radius, max_radius))
        candidates.append((mean_intensity, (int(cx), int(cy)), radius))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    circles = [
        (center, radius) for _, center, radius in candidates[:max_circles]
    ]
    return circles


def _draw_circles(
    frames: List[np.ndarray],
    circles: List[Tuple[Tuple[int, int], int]],
) -> None:
    for frame in frames:
        for center, radius in circles:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)


def _write_video(frames: List[np.ndarray], out_path: Path, fps: int, loop: int) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    for _ in range(loop):
        for frame in frames:
            writer.write(frame)
    writer.release()


def _write_gif(frames: List[np.ndarray], out_path: Path, fps: int, loop: int) -> None:
    duration_ms = int(1000 / max(fps, 1))
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    gif_frames = [Image.fromarray(f) for f in rgb_frames]
    if loop > 1:
        gif_frames = gif_frames * loop
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _load_top_subject_ids(csv_path: Path, top_n: int) -> List[str]:
    subject_ids: List[str] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if len(subject_ids) >= top_n:
                break
            subject_ids.append(row["subject_id"].strip())
    return subject_ids


def _load_subject_frame_paths(subjects_csv: Path) -> dict:
    subject_frames = {}
    with subjects_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = row.get("subject_id")
            frame_paths_raw = row.get("frame_paths")
            if not subject_id or not frame_paths_raw:
                continue
            try:
                paths = ast.literal_eval(frame_paths_raw)
            except (ValueError, SyntaxError):
                continue
            subject_frames[str(subject_id)] = [Path(p) for p in paths]
    return subject_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render looping videos with circles over likely movers."
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/backyard_worlds_motion/moving_object_ranking.csv",
        type=Path,
    )
    parser.add_argument(
        "--subjects-dir",
        default="outputs/backyard_worlds_motion/subjects",
        type=Path,
    )
    parser.add_argument(
        "--subjects-csv",
        default="outputs/backyard_worlds_motion/subjects.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/backyard_worlds_motion/overlays",
        type=Path,
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--loop", type=int, default=4)
    parser.add_argument("--max-circles", type=int, default=3)
    parser.add_argument("--min-blob-area", type=int, default=12)
    parser.add_argument("--threshold-quantile", type=float, default=0.88)
    parser.add_argument("--suppress-bright-percentile", type=float, default=99.7)
    parser.add_argument("--bright-component-percentile", type=float, default=99.5)
    parser.add_argument("--min-radius", type=int, default=6)
    parser.add_argument("--max-radius", type=int, default=24)
    parser.add_argument("--min-peak-value", type=int, default=6)
    parser.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="gif",
        help="Output format.",
    )
    args = parser.parse_args()

    subject_ids = _load_top_subject_ids(args.input_csv, args.top_n)
    subject_frames = _load_subject_frame_paths(args.subjects_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subject_ids:
        frame_paths = subject_frames.get(subject_id)
        if frame_paths is None:
            subject_dir = args.subjects_dir / f"subject_{subject_id}"
            fallback_paths = [
                subject_dir / f"frame_{i:02d}.jpg" for i in range(4)
            ]
            frame_paths = fallback_paths
        frames = _read_frames_from_paths(frame_paths)
        if not frames:
            continue

        circles = _compute_motion_circles(
            frames,
            max_circles=args.max_circles,
            min_blob_area=args.min_blob_area,
            threshold_quantile=args.threshold_quantile,
            suppress_bright_percentile=args.suppress_bright_percentile,
            bright_component_percentile=args.bright_component_percentile,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_peak_value=args.min_peak_value,
        )
        if circles:
            _draw_circles(frames, circles)

        if args.format == "gif":
            out_path = args.out_dir / f"subject_{subject_id}.gif"
            _write_gif(frames, out_path, args.fps, args.loop)
        else:
            out_path = args.out_dir / f"subject_{subject_id}.mp4"
            _write_video(frames, out_path, args.fps, args.loop)


if __name__ == "__main__":
    main()
