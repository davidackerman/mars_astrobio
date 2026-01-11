#!/usr/bin/env python3
"""Baseline diff->blob->dipole->tracklet detection for WISE-style flipbooks."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class Detection:
    det_id: int
    subject_id: str
    diff_index: int
    sign: str
    x: float
    y: float
    area: int
    peak: float
    mean: float
    brightness: float


@dataclass
class Dipole:
    dipole_id: int
    subject_id: str
    diff_index: int
    x: float
    y: float
    separation: float
    pos_id: int
    neg_id: int


@dataclass
class TrackletPoint:
    tracklet_id: int
    subject_id: str
    diff_index: int
    x: float
    y: float
    source: str


@dataclass
class SubjectResult:
    subject_id: str
    detections: List[Detection]
    dipoles: List[Dipole]
    tracklet_points: List[TrackletPoint]


def _read_frames_from_paths(frame_paths: List[Path]) -> List[np.ndarray]:
    frames = []
    target_size = None
    for path in frame_paths:
        if not path.exists():
            return []
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return []
        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32))
    return frames


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


def _load_subject_frame_paths(subjects_csv: Path) -> Dict[str, List[Path]]:
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


def _load_subject_ids(
    input_csv: Optional[Path], top_n: int
) -> Optional[List[str]]:
    if input_csv is None:
        return None
    if not input_csv.exists():
        return None
    subject_ids: List[str] = []
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if len(subject_ids) >= top_n:
                break
            subject_ids.append(row["subject_id"].strip())
    return subject_ids


def _robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    if sigma <= 1e-6:
        sigma = float(np.std(values))
    return sigma if sigma > 1e-6 else 1.0


def _detect_blobs(
    diff: np.ndarray,
    frame: np.ndarray,
    diff_index: int,
    subject_id: str,
    sign: str,
    thresh_sigma: float,
    min_area: int,
    max_brightness: float,
    max_detections: int,
    det_id_start: int,
) -> Tuple[List[Detection], int]:
    median = float(np.median(diff))
    sigma = _robust_sigma(diff)
    if sign == "positive":
        mask = diff >= (median + thresh_sigma * sigma)
    else:
        mask = diff <= (median - thresh_sigma * sigma)

    mask_u8 = (mask.astype(np.uint8) * 255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    detections: List[Detection] = []
    next_id = det_id_start
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blob_mask = labels == label
        if sign == "positive":
            peak = float(diff[blob_mask].max())
            mean = float(diff[blob_mask].mean())
        else:
            peak = float(diff[blob_mask].min())
            mean = float(diff[blob_mask].mean())
        brightness = float(frame[blob_mask].mean())
        if brightness > max_brightness:
            continue
        cx, cy = centroids[label]
        detections.append(
            Detection(
                det_id=next_id,
                subject_id=subject_id,
                diff_index=diff_index,
                sign=sign,
                x=float(cx),
                y=float(cy),
                area=area,
                peak=peak,
                mean=mean,
                brightness=brightness,
            )
        )
        next_id += 1

    detections.sort(key=lambda d: abs(d.peak), reverse=True)
    return detections[:max_detections], next_id


def _pair_dipoles(
    positives: List[Detection],
    negatives: List[Detection],
    min_sep: float,
    max_sep: float,
    dipole_id_start: int,
) -> Tuple[List[Dipole], int]:
    dipoles: List[Dipole] = []
    next_id = dipole_id_start
    used_neg = set()

    positives_sorted = sorted(positives, key=lambda d: abs(d.peak), reverse=True)
    for pos in positives_sorted:
        best = None
        best_dist = None
        for neg in negatives:
            if neg.det_id in used_neg:
                continue
            dx = pos.x - neg.x
            dy = pos.y - neg.y
            dist = float(np.hypot(dx, dy))
            if dist < min_sep or dist > max_sep:
                continue
            if best is None or dist < best_dist:
                best = neg
                best_dist = dist
        if best is None:
            continue
        used_neg.add(best.det_id)
        dipoles.append(
            Dipole(
                dipole_id=next_id,
                subject_id=pos.subject_id,
                diff_index=pos.diff_index,
                x=(pos.x + best.x) / 2.0,
                y=(pos.y + best.y) / 2.0,
                separation=best_dist or 0.0,
                pos_id=pos.det_id,
                neg_id=best.det_id,
            )
        )
        next_id += 1

    return dipoles, next_id


def _build_tracklets(
    subject_id: str,
    points_by_diff: Dict[int, List[Tuple[float, float, str]]],
    max_step: float,
    max_angle: float,
    max_residual: float,
    min_length: int,
) -> List[TrackletPoint]:
    tracklets: List[List[Tuple[int, float, float, str]]] = []
    used_by_diff: Dict[int, set] = {k: set() for k in points_by_diff}

    diffs = sorted(points_by_diff.keys())
    for diff_idx in diffs:
        points = points_by_diff.get(diff_idx, [])
        for p_idx, (x, y, source) in enumerate(points):
            if p_idx in used_by_diff[diff_idx]:
                continue
            tracklets.append([(diff_idx, x, y, source)])
            used_by_diff[diff_idx].add(p_idx)

        for tracklet in tracklets:
            last_diff, last_x, last_y, _ = tracklet[-1]
            next_diff = last_diff + 1
            candidates = points_by_diff.get(next_diff)
            if not candidates:
                continue
            best_idx = None
            best_dist = None
            for cand_idx, (cx, cy, source) in enumerate(candidates):
                if cand_idx in used_by_diff[next_diff]:
                    continue
                dist = float(np.hypot(cx - last_x, cy - last_y))
                if dist > max_step:
                    continue
                if len(tracklet) >= 2:
                    prev_diff, prev_x, prev_y, _ = tracklet[-2]
                    v1 = np.array([last_x - prev_x, last_y - prev_y], dtype=float)
                    v2 = np.array([cx - last_x, cy - last_y], dtype=float)
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        cosang = np.clip(cosang, -1.0, 1.0)
                        angle = float(np.degrees(np.arccos(cosang)))
                        if angle > max_angle:
                            continue
                if best_idx is None or dist < best_dist:
                    best_idx = cand_idx
                    best_dist = dist
            if best_idx is not None:
                cx, cy, source = candidates[best_idx]
                tracklet.append((next_diff, cx, cy, source))
                used_by_diff[next_diff].add(best_idx)

    filtered: List[List[Tuple[int, float, float, str]]] = []
    for tracklet in tracklets:
        if len(tracklet) < min_length:
            continue
        if len(tracklet) >= 3:
            t = np.array([p[0] for p in tracklet], dtype=float)
            xs = np.array([p[1] for p in tracklet], dtype=float)
            ys = np.array([p[2] for p in tracklet], dtype=float)
            x_fit = np.polyfit(t, xs, 1)
            y_fit = np.polyfit(t, ys, 1)
            xs_pred = np.polyval(x_fit, t)
            ys_pred = np.polyval(y_fit, t)
            residuals = np.hypot(xs - xs_pred, ys - ys_pred)
            if float(residuals.max()) > max_residual:
                continue
        filtered.append(tracklet)

    points: List[TrackletPoint] = []
    for tracklet_id, tracklet in enumerate(filtered, start=1):
        for diff_idx, x, y, source in tracklet:
            points.append(
                TrackletPoint(
                    tracklet_id=tracklet_id,
                    subject_id=subject_id,
                    diff_index=diff_idx,
                    x=x,
                    y=y,
                    source=source,
                )
            )
    return points


def _process_subject(
    subject_id: str,
    frame_paths: List[Path],
    args: argparse.Namespace,
) -> Optional[SubjectResult]:
    frames = _read_frames_from_paths(frame_paths)
    if not frames:
        return None

    if args.blur_sigma > 0:
        frames = [
            cv2.GaussianBlur(f, (0, 0), args.blur_sigma)
            for f in frames
        ]

    brightness_thresholds = []
    for frame in frames:
        if args.bright_percentile >= 100.0:
            brightness_thresholds.append(float("inf"))
        else:
            brightness_thresholds.append(
                float(np.percentile(frame, args.bright_percentile))
            )

    detections: List[Detection] = []
    dipoles: List[Dipole] = []
    points_by_diff: Dict[int, List[Tuple[float, float, str]]] = {}
    dipole_points_by_diff: Dict[int, List[Tuple[float, float, str]]] = {}

    det_id = 1
    dipole_id = 1
    for i in range(len(frames) - 1):
        diff = frames[i + 1] - frames[i]
        pos_dets, det_id = _detect_blobs(
            diff=diff,
            frame=frames[i + 1],
            diff_index=i,
            subject_id=subject_id,
            sign="positive",
            thresh_sigma=args.thresh_sigma,
            min_area=args.min_area,
            max_brightness=brightness_thresholds[i + 1],
            max_detections=args.max_detections,
            det_id_start=det_id,
        )
        neg_dets, det_id = _detect_blobs(
            diff=diff,
            frame=frames[i],
            diff_index=i,
            subject_id=subject_id,
            sign="negative",
            thresh_sigma=args.thresh_sigma,
            min_area=args.min_area,
            max_brightness=brightness_thresholds[i],
            max_detections=args.max_detections,
            det_id_start=det_id,
        )
        detections.extend(pos_dets)
        detections.extend(neg_dets)

        dipole_list, dipole_id = _pair_dipoles(
            positives=pos_dets,
            negatives=neg_dets,
            min_sep=args.dipole_min_sep,
            max_sep=args.dipole_max_sep,
            dipole_id_start=dipole_id,
        )
        dipoles.extend(dipole_list)
        dipole_points_by_diff[i] = [
            (dip.x, dip.y, "dipole") for dip in dipole_list
        ]

        points = []
        if args.use_dipole_centers and dipole_list:
            for dip in dipole_list:
                points.append((dip.x, dip.y, "dipole"))
        else:
            for det in pos_dets:
                points.append((det.x, det.y, "positive"))
            for det in neg_dets:
                points.append((det.x, det.y, "negative"))
        points_by_diff[i] = points

    if args.enforce_dipole_min_frames and dipoles:
        dipole_tracklet_points = _build_tracklets(
            subject_id=subject_id,
            points_by_diff=dipole_points_by_diff,
            max_step=args.dipole_tracklet_max_step,
            max_angle=args.dipole_tracklet_max_angle,
            max_residual=args.dipole_tracklet_max_residual,
            min_length=args.dipole_min_frames,
        )
        keep_keys = {
            (p.diff_index, round(p.x, 2), round(p.y, 2))
            for p in dipole_tracklet_points
        }
        dipoles = [
            d
            for d in dipoles
            if (d.diff_index, round(d.x, 2), round(d.y, 2)) in keep_keys
        ]

    tracklet_min_length = args.tracklet_min_length
    if args.enforce_tracklet_min_frames:
        tracklet_min_length = max(tracklet_min_length, args.tracklet_min_frames)

    tracklet_points = _build_tracklets(
        subject_id=subject_id,
        points_by_diff=points_by_diff,
        max_step=args.tracklet_max_step,
        max_angle=args.tracklet_max_angle,
        max_residual=args.tracklet_max_residual,
        min_length=tracklet_min_length,
    )

    return SubjectResult(
        subject_id=subject_id,
        detections=detections,
        dipoles=dipoles,
        tracklet_points=tracklet_points,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline diff->blob->dipole->tracklet detector."
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
        "--input-csv",
        default=None,
        type=Path,
        help="Optional ranking CSV to limit processing to top-N subjects.",
    )
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        default="outputs/backyard_worlds_motion/diff_blob_baseline",
        type=Path,
    )
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--thresh-sigma", type=float, default=2.5)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-detections", type=int, default=40)
    parser.add_argument("--bright-percentile", type=float, default=98.0)
    parser.add_argument("--dipole-min-sep", type=float, default=2.0)
    parser.add_argument("--dipole-max-sep", type=float, default=25.0)
    parser.add_argument("--tracklet-max-step", type=float, default=20.0)
    parser.add_argument("--tracklet-max-angle", type=float, default=25.0)
    parser.add_argument("--tracklet-max-residual", type=float, default=6.0)
    parser.add_argument("--tracklet-min-length", type=int, default=2)
    parser.add_argument("--enforce-tracklet-min-frames", action="store_true")
    parser.add_argument("--tracklet-min-frames", type=int, default=3)
    parser.add_argument("--enforce-dipole-min-frames", action="store_true")
    parser.add_argument("--dipole-min-frames", type=int, default=3)
    parser.add_argument("--dipole-tracklet-max-step", type=float, default=20.0)
    parser.add_argument("--dipole-tracklet-max-angle", type=float, default=25.0)
    parser.add_argument("--dipole-tracklet-max-residual", type=float, default=6.0)
    parser.add_argument("--dipole-match-radius", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--use-dipole-centers",
        action="store_true",
        help="Use dipole centers for tracklets (fallback to raw detections if none).",
    )
    args = parser.parse_args()

    subject_frames = _load_subject_frame_paths(args.subjects_csv)
    subject_ids = _load_subject_ids(args.input_csv, args.top_n)
    if subject_ids is None:
        subject_ids = list(subject_frames.keys())

    if args.workers <= 0:
        args.workers = os.cpu_count() or 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detections_out = args.out_dir / "diff_blob_detections.csv"
    dipoles_out = args.out_dir / "diff_blob_dipoles.csv"
    tracklets_out = args.out_dir / "diff_blob_tracklets.csv"

    with detections_out.open("w", newline="") as det_handle, \
        dipoles_out.open("w", newline="") as dip_handle, \
        tracklets_out.open("w", newline="") as trk_handle:
        det_writer = csv.DictWriter(
            det_handle,
            fieldnames=[
                "det_id",
                "subject_id",
                "diff_index",
                "sign",
                "x",
                "y",
                "area",
                "peak",
                "mean",
                "brightness",
            ],
        )
        dip_writer = csv.DictWriter(
            dip_handle,
            fieldnames=[
                "dipole_id",
                "subject_id",
                "diff_index",
                "x",
                "y",
                "separation",
                "pos_id",
                "neg_id",
            ],
        )
        trk_writer = csv.DictWriter(
            trk_handle,
            fieldnames=[
                "tracklet_id",
                "subject_id",
                "diff_index",
                "x",
                "y",
                "source",
            ],
        )
        det_writer.writeheader()
        dip_writer.writeheader()
        trk_writer.writeheader()

        det_id = 1
        dipole_id = 1
        results: Iterable[Optional[SubjectResult]]

        def _frame_paths_for_subject(sid: str) -> List[Path]:
            frame_paths = subject_frames.get(sid)
            if frame_paths is None:
                subject_dir = args.subjects_dir / f"subject_{sid}"
                frame_paths = [subject_dir / f"frame_{i:02d}.jpg" for i in range(4)]
            return frame_paths

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        _process_subject,
                        subject_id,
                        _frame_paths_for_subject(subject_id),
                        args,
                    ): subject_id
                    for subject_id in subject_ids
                }
                iterator = as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(
                        iterator, total=len(futures), desc="Subjects", unit="subject"
                    )
                for future in iterator:
                    result = future.result()
                    if result is None:
                        continue
                    det_id, dipole_id = _write_subject_results(
                        result,
                        det_writer,
                        dip_writer,
                        trk_writer,
                        det_id,
                        dipole_id,
                    )
        else:
            progress = subject_ids
            if tqdm is not None:
                progress = tqdm(subject_ids, desc="Subjects", unit="subject")
            for subject_id in progress:
                result = _process_subject(
                    subject_id,
                    _frame_paths_for_subject(subject_id),
                    args,
                )
                if result is None:
                    continue
                det_id, dipole_id = _write_subject_results(
                    result,
                    det_writer,
                    dip_writer,
                    trk_writer,
                    det_id,
                    dipole_id,
                )


def _write_subject_results(
    result: SubjectResult,
    det_writer: csv.DictWriter,
    dip_writer: csv.DictWriter,
    trk_writer: csv.DictWriter,
    det_id: int,
    dipole_id: int,
) -> Tuple[int, int]:
    det_id_map: Dict[int, int] = {}
    for det in result.detections:
        det_id_map[det.det_id] = det_id
        det_writer.writerow(
            {
                "det_id": det_id,
                "subject_id": det.subject_id,
                "diff_index": det.diff_index,
                "sign": det.sign,
                "x": f"{det.x:.2f}",
                "y": f"{det.y:.2f}",
                "area": det.area,
                "peak": f"{det.peak:.2f}",
                "mean": f"{det.mean:.2f}",
                "brightness": f"{det.brightness:.2f}",
            }
        )
        det_id += 1

    for dip in result.dipoles:
        pos_id = det_id_map.get(dip.pos_id)
        neg_id = det_id_map.get(dip.neg_id)
        if pos_id is None or neg_id is None:
            continue
        dip_writer.writerow(
            {
                "dipole_id": dipole_id,
                "subject_id": dip.subject_id,
                "diff_index": dip.diff_index,
                "x": f"{dip.x:.2f}",
                "y": f"{dip.y:.2f}",
                "separation": f"{dip.separation:.2f}",
                "pos_id": pos_id,
                "neg_id": neg_id,
            }
        )
        dipole_id += 1

    for point in result.tracklet_points:
        trk_writer.writerow(
            {
                "tracklet_id": point.tracklet_id,
                "subject_id": point.subject_id,
                "diff_index": point.diff_index,
                "x": f"{point.x:.2f}",
                "y": f"{point.y:.2f}",
                "source": point.source,
            }
        )

    return det_id, dipole_id


if __name__ == "__main__":
    main()
