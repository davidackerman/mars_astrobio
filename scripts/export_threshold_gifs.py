#!/usr/bin/env python3
"""
Export thresholded GIFs for quick visual inspection.

Zeros out pixels below a grayscale threshold for each frame.
"""

import argparse
import random
from pathlib import Path
from typing import List

from PIL import Image


def load_frames(subject_dir: Path) -> List[Image.Image]:
    frames = []
    for path in sorted(subject_dir.glob("frame_*.jpg")):
        frames.append(Image.open(path).convert("RGB"))
    return frames


def apply_threshold(frame: Image.Image, threshold_value: int) -> Image.Image:
    gray = frame.convert("L")
    mask = gray.point(lambda p: 255 if p >= threshold_value else 0)
    black = Image.new("RGB", frame.size, (0, 0, 0))
    return Image.composite(frame, black, mask)


def write_gif(frames: List[Image.Image], out_path: Path, duration: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thresholded subject GIFs.")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth/subjects_groundtruth"),
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path("data/backyard_worlds/subjects"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/threshold_gifs"))
    parser.add_argument("--threshold-value", type=int, default=125)
    parser.add_argument("--num-subjects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif-duration", type=float, default=0.8)
    args = parser.parse_args()

    gt_subjects = sorted(p for p in args.gt_dir.iterdir() if p.is_dir())
    subjects = sorted(p for p in args.subjects_dir.iterdir() if p.is_dir())

    random.seed(args.seed)
    sample_subjects = random.sample(subjects, min(args.num_subjects, len(subjects)))

    for subject_dir in gt_subjects:
        frames = load_frames(subject_dir)
        frames = [apply_threshold(f, args.threshold_value) for f in frames]
        out_path = args.out_dir / "ground_truth" / f"{subject_dir.name}.gif"
        write_gif(frames, out_path, args.gif_duration)

    for subject_dir in sample_subjects:
        frames = load_frames(subject_dir)
        frames = [apply_threshold(f, args.threshold_value) for f in frames]
        out_path = args.out_dir / "random_subjects" / f"{subject_dir.name}.gif"
        write_gif(frames, out_path, args.gif_duration)


if __name__ == "__main__":
    main()
