#!/usr/bin/env python3
"""Render GIFs with baseline diff/blob/tracklet detections overlaid."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _read_frames(frame_paths: List[Path]) -> List:
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


def _write_gif(frames: List, out_path: Path, fps: int, loop: int) -> None:
    duration_ms = int(1000 / max(fps, 1))
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    from PIL import Image

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


def _resolve_subjects_csv(subjects_csv: Path) -> Path:
    if subjects_csv.exists():
        return subjects_csv
    fallback = Path("data/subjects_metadata.csv")
    if fallback.exists():
        return fallback
    alt = Path("outputs/backyard_worlds_motion/subjects.csv")
    if alt.exists():
        return alt
    return subjects_csv


def _load_subject_frames(subjects_csv: Path) -> Dict[str, List[Path]]:
    subject_frames: Dict[str, List[Path]] = {}
    resolved_csv = _resolve_subjects_csv(subjects_csv)
    with resolved_csv.open(newline="") as handle:
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


def _load_subject_ids_from_csv(subjects_csv: Path) -> List[str]:
    resolved_csv = _resolve_subjects_csv(subjects_csv)
    subject_ids: List[str] = []
    with resolved_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = row.get("subject_id")
            if subject_id:
                subject_ids.append(str(subject_id))
    return subject_ids


def _load_detections(path: Path) -> Dict[str, List[Tuple[int, float, float, str]]]:
    dets: Dict[str, List[Tuple[int, float, float, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = row["subject_id"]
            diff_index = int(row["diff_index"])
            x = float(row["x"])
            y = float(row["y"])
            sign = row["sign"]
            dets[subject_id].append((diff_index, x, y, sign))
    return dets


def _load_tracklets(path: Path) -> Dict[str, Dict[int, List[Tuple[int, float, float]]]]:
    tracklets: Dict[str, Dict[int, List[Tuple[int, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = row["subject_id"]
            tracklet_id = int(row["tracklet_id"])
            diff_index = int(row["diff_index"])
            x = float(row["x"])
            y = float(row["y"])
            tracklets[subject_id][tracklet_id].append((diff_index, x, y))
    return tracklets


def _load_dipoles(path: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    dipoles: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    if not path.exists():
        return dipoles
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = row["subject_id"]
            diff_index = int(row["diff_index"])
            x = float(row["x"])
            y = float(row["y"])
            dipoles[subject_id].append((diff_index, x, y))
    return dipoles


def _color_for_id(item_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (255, 180, 0),
        (0, 200, 0),
        (255, 0, 120),
        (0, 160, 255),
        (200, 0, 255),
        (0, 255, 120),
    ]
    return palette[item_id % len(palette)]


def _draw_detections(
    frames: List,
    dets: List[Tuple[int, float, float, str]],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    for diff_index, x, y, _ in dets:
        frame_idx = min(diff_index + 1, len(frames) - 1)
        cv2.circle(frames[frame_idx], (int(x), int(y)), radius, color, 2)


def _draw_tracklets(
    frames: List,
    tracklets: Dict[int, List[Tuple[int, float, float]]],
    radius: int,
    line_width: int,
    draw_points: bool,
) -> None:
    for tracklet_id, points in tracklets.items():
        color = _color_for_id(tracklet_id)
        points_sorted = sorted(points, key=lambda p: p[0])
        if len(points_sorted) < 2:
            continue
        _, x_start, y_start = points_sorted[0]
        _, x_end, y_end = points_sorted[-1]
        dx = x_end - x_start
        dy = y_end - y_start
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            continue
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        half = length / 2.0
        angle = float(np.degrees(np.arctan2(dy, dx)))
        center_x = (x_start + x_end) / 2.0
        center_y = (y_start + y_end) / 2.0
        for frame in frames:
            cv2.ellipse(
                frame,
                (int(center_x), int(center_y)),
                (int(half), int(radius)),
                angle,
                0,
                360,
                color,
                line_width,
            )
            cap_x1 = x_start + ux * radius
            cap_y1 = y_start + uy * radius
            cap_x2 = x_end - ux * radius
            cap_y2 = y_end - uy * radius
            p1x = cap_x1 + px * radius
            p1y = cap_y1 + py * radius
            p2x = cap_x1 - px * radius
            p2y = cap_y1 - py * radius
            p3x = cap_x2 + px * radius
            p3y = cap_y2 + py * radius
            p4x = cap_x2 - px * radius
            p4y = cap_y2 - py * radius
            cv2.line(frame, (int(p1x), int(p1y)), (int(p3x), int(p3y)), color, line_width)
            cv2.line(frame, (int(p2x), int(p2y)), (int(p4x), int(p4y)), color, line_width)
        if draw_points:
            for diff_index, x, y in points_sorted:
                frame_idx = min(diff_index + 1, len(frames) - 1)
                cv2.circle(frames[frame_idx], (int(x), int(y)), radius, color, 2)


def _draw_dipoles(
    frames: List,
    dipoles: List[Tuple[int, float, float]],
    size: int,
    color: Tuple[int, int, int],
) -> None:
    half = max(2, size // 2)
    for diff_index, x, y in dipoles:
        for frame_idx in (diff_index, diff_index + 1):
            frame_idx = min(frame_idx, len(frames) - 1)
            cx, cy = int(x), int(y)
            cv2.rectangle(
                frames[frame_idx],
                (cx - half, cy - half),
                (cx + half, cy + half),
                color,
                2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render GIF overlays for diff/blob/tracklet baseline outputs."
    )
    parser.add_argument(
        "--subjects-csv",
        default="outputs/backyard_worlds_motion/subjects.csv",
        type=Path,
    )
    parser.add_argument(
        "--subjects-dir",
        default="outputs/backyard_worlds_motion/subjects",
        type=Path,
    )
    parser.add_argument(
        "--detections-csv",
        default="outputs/backyard_worlds_motion/diff_blob_baseline/diff_blob_detections.csv",
        type=Path,
    )
    parser.add_argument(
        "--tracklets-csv",
        default="outputs/backyard_worlds_motion/diff_blob_baseline/diff_blob_tracklets.csv",
        type=Path,
    )
    parser.add_argument(
        "--dipoles-csv",
        default="outputs/backyard_worlds_motion/diff_blob_baseline/diff_blob_dipoles.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/backyard_worlds_motion/diff_blob_baseline/overlays",
        type=Path,
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--loop", type=int, default=4)
    parser.add_argument("--det-radius", type=int, default=8)
    parser.add_argument("--track-radius", type=int, default=8)
    parser.add_argument("--track-line-width", type=int, default=2)
    parser.add_argument("--tracklet-points", action="store_true")
    parser.add_argument("--dipole-size", type=int, default=14)
    parser.add_argument("--no-tracklets", action="store_true")
    args = parser.parse_args()

    subject_frames = _load_subject_frames(args.subjects_csv)
    subject_id_list = _load_subject_ids_from_csv(args.subjects_csv)
    detections = _load_detections(args.detections_csv)
    tracklets = _load_tracklets(args.tracklets_csv)
    dipoles = _load_dipoles(args.dipoles_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eligible_subjects = {
        sid
        for sid in subject_id_list
        if tracklets.get(sid) or dipoles.get(sid)
    }
    subject_ids = [sid for sid in subject_id_list if sid in eligible_subjects][: args.top_n]
    for subject_id in subject_ids:
        frame_paths = subject_frames.get(subject_id)
        if frame_paths is None:
            subject_dir = args.subjects_dir / f"subject_{subject_id}"
            frame_paths = [subject_dir / f"frame_{i:02d}.jpg" for i in range(4)]
        frames = _read_frames(frame_paths)
        if not frames:
            continue

        if not args.no_tracklets:
            _draw_tracklets(
                frames,
                tracklets.get(subject_id, {}),
                radius=args.track_radius,
                line_width=args.track_line_width,
                draw_points=args.tracklet_points,
            )
        else:
            _draw_detections(
                frames,
                detections.get(subject_id, []),
                args.det_radius,
                (0, 255, 255),
            )
        _draw_dipoles(
            frames,
            dipoles.get(subject_id, []),
            size=args.dipole_size,
            color=(0, 0, 255),
        )

        out_path = args.out_dir / f"subject_{subject_id}.gif"
        _write_gif(frames, out_path, args.fps, args.loop)


if __name__ == "__main__":
    main()
