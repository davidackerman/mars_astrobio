#!/usr/bin/env python3
"""Calibrate diff-blob baseline thresholds using ground-truth annotations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from diff_blob_tracklet_baseline import _process_subject, _load_subject_frame_paths


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _centers_from_circles(circles: List[List[float]]) -> List[Tuple[float, float]]:
    return [(x + w / 2.0, y + h / 2.0) for x, y, w, h in circles]


def _match_points(
    preds: List[Tuple[float, float]],
    gts: List[Tuple[float, float]],
    radius: float,
) -> Metrics:
    if not preds and not gts:
        return Metrics(precision=1.0, recall=1.0, f1=1.0, tp=0, fp=0, fn=0)
    if not preds:
        return Metrics(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gts))
    if not gts:
        return Metrics(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=len(preds), fn=0)

    preds_used = [False] * len(preds)
    tp = 0
    for gx, gy in gts:
        best_idx = None
        best_dist = None
        for i, (px, py) in enumerate(preds):
            if preds_used[i]:
                continue
            dist = float(np.hypot(px - gx, py - gy))
            if dist <= radius and (best_dist is None or dist < best_dist):
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            preds_used[best_idx] = True
            tp += 1

    fp = sum(1 for used in preds_used if not used)
    fn = len(gts) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def _tracklet_centers(tracklet_points) -> Dict[int, Tuple[float, float]]:
    grouped: Dict[int, List[Tuple[float, float]]] = {}
    for point in tracklet_points:
        grouped.setdefault(point.tracklet_id, []).append((point.x, point.y))
    centers: Dict[int, Tuple[float, float]] = {}
    for tracklet_id, points in grouped.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        centers[tracklet_id] = (float(np.mean(xs)), float(np.mean(ys)))
    return centers


def _make_args(
    args: argparse.Namespace,
    thresh_sigma: float,
    min_area: int,
    bright_percentile: float,
    tracklet_min_frames: int,
    dipole_min_frames: int,
    use_dipole_centers: bool,
):
    return argparse.Namespace(
        blur_sigma=args.blur_sigma,
        thresh_sigma=thresh_sigma,
        min_area=min_area,
        max_detections=args.max_detections,
        bright_percentile=bright_percentile,
        dipole_min_sep=args.dipole_min_sep,
        dipole_max_sep=args.dipole_max_sep,
        tracklet_max_step=args.tracklet_max_step,
        tracklet_max_angle=args.tracklet_max_angle,
        tracklet_max_residual=args.tracklet_max_residual,
        tracklet_min_length=args.tracklet_min_length,
        enforce_tracklet_min_frames=True,
        tracklet_min_frames=tracklet_min_frames,
        enforce_dipole_min_frames=True,
        dipole_min_frames=dipole_min_frames,
        dipole_tracklet_max_step=args.dipole_tracklet_max_step,
        dipole_tracklet_max_angle=args.dipole_tracklet_max_angle,
        dipole_tracklet_max_residual=args.dipole_tracklet_max_residual,
        dipole_match_radius=args.dipole_match_radius,
        use_dipole_centers=use_dipole_centers,
    )


def _ensure_ground_truth_paths(
    subject_frames: Dict[str, List[Path]],
    data_dir: Path,
) -> Dict[str, List[Path]]:
    updated: Dict[str, List[Path]] = {}
    for subject_id, paths in subject_frames.items():
        fixed_paths = []
        for path in paths:
            if path.exists():
                fixed_paths.append(path)
                continue
            if "data/backyard_worlds/subjects_groundtruth" in str(path):
                alt = Path(str(path).replace(
                    "data/backyard_worlds/subjects_groundtruth",
                    "data/backyard_worlds/ground_truth/subjects_groundtruth",
                ))
                if alt.exists():
                    fixed_paths.append(alt)
                    continue
            alt = data_dir / "subjects_groundtruth" / Path(path).parent.name / path.name
            if alt.exists():
                fixed_paths.append(alt)
                continue
            fixed_paths.append(path)
        updated[subject_id] = fixed_paths
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate diff-blob baseline thresholds.")
    parser.add_argument(
        "--subjects-csv",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth/subjects_groundtruth_metadata.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth/annotations.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/diff_blob_calibration"))
    parser.add_argument("--match-radius", type=float, default=12.0)

    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--max-detections", type=int, default=40)
    parser.add_argument("--dipole-min-sep", type=float, default=2.0)
    parser.add_argument("--dipole-max-sep", type=float, default=25.0)
    parser.add_argument("--tracklet-max-step", type=float, default=20.0)
    parser.add_argument("--tracklet-max-angle", type=float, default=25.0)
    parser.add_argument("--tracklet-max-residual", type=float, default=6.0)
    parser.add_argument("--tracklet-min-length", type=int, default=2)
    parser.add_argument("--tracklet-min-frames", type=int, nargs="+", default=[3])
    parser.add_argument("--dipole-min-frames", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--dipole-tracklet-max-step", type=float, default=20.0)
    parser.add_argument("--dipole-tracklet-max-angle", type=float, default=25.0)
    parser.add_argument("--dipole-tracklet-max-residual", type=float, default=6.0)
    parser.add_argument("--dipole-match-radius", type=float, default=6.0)
    parser.add_argument("--use-dipole-centers", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--thresh-sigma", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    parser.add_argument("--min-area", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--bright-percentile", type=float, nargs="+", default=[97.0, 98.0, 99.0])

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "calibration_results.csv"
    best_path = args.out_dir / "best_params.json"

    with args.annotations.open() as f:
        annotations = json.load(f)

    subject_frames = _load_subject_frame_paths(args.subjects_csv)
    if not subject_frames:
        alt_csv = args.annotations.parent / "subjects_groundtruth_metadata.csv"
        if alt_csv.exists():
            subject_frames = _load_subject_frame_paths(alt_csv)
    subject_frames = _ensure_ground_truth_paths(subject_frames, args.annotations.parent)
    subject_ids = [
        sid for sid in subject_frames.keys()
        if sid in annotations or f"subject_{sid}" in annotations
    ]
    if not subject_ids:
        raise RuntimeError("No subjects with annotations found for calibration.")

    rows = []
    best = None

    for thresh_sigma in args.thresh_sigma:
        for min_area in args.min_area:
            for bright_percentile in args.bright_percentile:
                for tracklet_min_frames in args.tracklet_min_frames:
                    for dipole_min_frames in args.dipole_min_frames:
                        for use_dipole_centers in [args.use_dipole_centers]:
                            cfg = _make_args(
                                args,
                                thresh_sigma,
                                min_area,
                                bright_percentile,
                                tracklet_min_frames,
                                dipole_min_frames,
                                use_dipole_centers,
                            )

                            mover_preds = []
                            mover_gts = []
                            dipole_preds = []
                            dipole_gts = []
                            subjects_used = 0

                            for subject_id in subject_ids:
                                annotation_key = subject_id
                                if annotation_key not in annotations:
                                    annotation_key = f"subject_{subject_id}"
                                result = _process_subject(subject_id, subject_frames[subject_id], cfg)
                                if result is None:
                                    continue
                                subjects_used += 1

                                annotation = annotations[annotation_key]
                                mover_gts.extend(_centers_from_circles(annotation.get("mover_circles", [])))
                                dipole_gts.extend(_centers_from_circles(annotation.get("dipole_circles", [])))

                                dipole_preds.extend([(dip.x, dip.y) for dip in result.dipoles])

                                tracklet_centers = _tracklet_centers(result.tracklet_points)
                                mover_preds.extend(list(tracklet_centers.values()))

                            mover_metrics = _match_points(mover_preds, mover_gts, args.match_radius)
                            dipole_metrics = _match_points(dipole_preds, dipole_gts, args.match_radius)
                            macro_f1 = (mover_metrics.f1 + dipole_metrics.f1) / 2.0

                            row = {
                                "thresh_sigma": thresh_sigma,
                                "min_area": min_area,
                                "bright_percentile": bright_percentile,
                                "tracklet_min_frames": tracklet_min_frames,
                                "dipole_min_frames": dipole_min_frames,
                                "use_dipole_centers": use_dipole_centers,
                                "mover_precision": mover_metrics.precision,
                                "mover_recall": mover_metrics.recall,
                                "mover_f1": mover_metrics.f1,
                                "dipole_precision": dipole_metrics.precision,
                                "dipole_recall": dipole_metrics.recall,
                                "dipole_f1": dipole_metrics.f1,
                                "macro_f1": macro_f1,
                                "mover_tp": mover_metrics.tp,
                                "mover_fp": mover_metrics.fp,
                                "mover_fn": mover_metrics.fn,
                                "dipole_tp": dipole_metrics.tp,
                                "dipole_fp": dipole_metrics.fp,
                                "dipole_fn": dipole_metrics.fn,
                                "num_subjects": subjects_used,
                            }
                            rows.append(row)

                            if best is None or macro_f1 > best["macro_f1"]:
                                best = row
                            if (
                                thresh_sigma == args.thresh_sigma[0]
                                and min_area == args.min_area[0]
                                and bright_percentile == args.bright_percentile[0]
                                and tracklet_min_frames == args.tracklet_min_frames[0]
                                and dipole_min_frames == args.dipole_min_frames[0]
                            ):
                                print(f"Sanity: processed {subjects_used} subjects out of {len(subject_ids)}")

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best is not None:
        with best_path.open("w") as handle:
            json.dump(best, handle, indent=2)

    print(f"Wrote {len(rows)} rows to {results_path}")
    if best is not None:
        print(f"Best params written to {best_path}")


if __name__ == "__main__":
    main()
