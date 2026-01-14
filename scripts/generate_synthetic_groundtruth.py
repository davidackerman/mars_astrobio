#!/usr/bin/env python3
"""
Generate synthetic Backyard Worlds ground truth sequences.

Uses real subject frames as background (noise + star field) and injects
synthetic dipoles and movers. By default, outputs cropped frames using
the shared crop config.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys
import colorsys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_pipelines.core.training.datasets.crop_config import crop_bounds


def list_subject_dirs(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "frame_00.jpg").exists())


def load_frames(subject_dir: Path) -> List[np.ndarray]:
    frames = []
    for i in range(4):
        path = subject_dir / f"frame_{i:02d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        frames.append(img)
    return frames


def apply_crop(frames: List[np.ndarray]) -> List[np.ndarray]:
    h, w = frames[0].shape[:2]
    left, top, right, bottom = crop_bounds((h, w))
    return [frame[top:bottom, left:right].copy() for frame in frames]


def estimate_noise_std(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    resid = gray - blur
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    return max(1.0, 1.4826 * mad)


def add_gaussian(
    frame: np.ndarray,
    x: float,
    y: float,
    amp: float,
    sigma: float,
    color: Tuple[float, float, float],
) -> None:
    h, w = frame.shape[:2]
    radius = max(1, int(math.ceil(3 * sigma)))
    x0 = int(round(x))
    y0 = int(round(y))

    x_min = max(0, x0 - radius)
    x_max = min(w - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(h - 1, y0 + radius)

    if x_min >= x_max or y_min >= y_max:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))

    for c in range(3):
        frame[y_min : y_max + 1, x_min : x_max + 1, c] += amp * color[c] * g


def add_dipole_half(
    frame: np.ndarray,
    x: float,
    y: float,
    amp: float,
    sigma: float,
    color: Tuple[float, float, float],
    angle: float,
    flipped: bool,
) -> None:
    h, w = frame.shape[:2]
    radius = max(1, int(math.ceil(3 * sigma)))
    x0 = int(round(x))
    y0 = int(round(y))

    x_min = max(0, x0 - radius)
    x_max = min(w - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(h - 1, y0 + radius)

    if x_min >= x_max or y_min >= y_max:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))

    nx = math.cos(angle)
    ny = math.sin(angle)
    half_plane = np.sign((xx - x) * nx + (yy - y) * ny)
    if flipped:
        half_plane *= -1.0
    half_plane[half_plane == 0] = 1.0

    for c in range(3):
        frame[y_min : y_max + 1, x_min : x_max + 1, c] += amp * color[c] * g * half_plane


def random_color(rng: random.Random) -> Tuple[float, float, float]:
    # Backyard Worlds colors: several red/orange/blue/white bases with
    # small HSV jitter so colors better match image-dominant hues.
    palette = [
        (1.0, 0.6, 0.45),
        (1.0, 0.7, 0.55),
        (1.0, 0.5, 0.35),
        (1.0, 0.8, 0.6),
        (0.65, 0.75, 1.0),
        (0.5, 0.7, 1.0),
        (0.4, 0.6, 1.0),
        (1.0, 1.0, 1.0),
    ]
    base = palette[rng.randrange(len(palette))]
    # convert to HSV, apply small jitter to hue/sat/value, convert back
    h, s, v = colorsys.rgb_to_hsv(*base)
    h_jitter = rng.uniform(-0.06, 0.06)
    s_jitter = rng.uniform(-0.12, 0.12)
    v_jitter = rng.uniform(-0.08, 0.08)
    h = (h + h_jitter) % 1.0
    s = max(0.0, min(1.0, s + s_jitter))
    v = max(0.0, min(1.0, v + v_jitter))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def place_dipole(
    frames: List[np.ndarray],
    rng: random.Random,
    noise_std: float,
    bounds: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    radius = rng.uniform(6.0, 20.0)
    sigma = max(0.9, radius / 4.0)
    amp = rng.uniform(6.0, 14.0) * noise_std
    color = random_color(rng)
    core_sigma = max(0.6, sigma * 0.45)
    core_amp = amp * 1.4

    # Use a separation tied to the visible radius (~3*sigma) so overlaps
    # match what you see in the GIFs.
    visible_radius = sigma / 2
    sep = rng.uniform(visible_radius * 1.0, visible_radius * 1.75)
    margin = int(math.ceil(3 * sigma + sep / 2 + 2))
    x = rng.uniform(left + margin, right - margin)
    y = rng.uniform(top + margin, bottom - margin)

    amp_a = amp * rng.uniform(0.7, 1.1)
    amp_b = amp * rng.uniform(0.7, 1.1)
    core_amp_a = core_amp * rng.uniform(0.8, 1.1)
    core_amp_b = core_amp * rng.uniform(0.8, 1.1)

    switch_frame = rng.choice([1, 2, 3])
    angle = rng.uniform(0, 2 * math.pi)
    dx = math.cos(angle) * (sep / 2.0)
    dy = math.sin(angle) * (sep / 2.0)
    for i, frame in enumerate(frames):
        jitter_x = rng.uniform(-0.2, 0.2)
        jitter_y = rng.uniform(-0.2, 0.2)
        flipped = i >= switch_frame
        sign_a = -1.0 if flipped else 1.0
        sign_b = 1.0 if flipped else -1.0
        ax = x + dx + jitter_x
        ay = y + dy + jitter_y
        bx = x - dx + jitter_x
        by = y - dy + jitter_y

        add_gaussian(frame, ax, ay, sign_a * amp_a, sigma, color)
        add_gaussian(frame, bx, by, sign_b * amp_b, sigma, color)
        add_gaussian(frame, ax, ay, sign_a * core_amp_a, core_sigma, color)
        add_gaussian(frame, bx, by, sign_b * core_amp_b, core_sigma, color)

    box_size = max(10, int(math.ceil(sep + radius * 2)))
    box_x = int(round(x - box_size / 2))
    box_y = int(round(y - box_size / 2))
    return box_x, box_y, box_size, box_size


def place_mover(
    frames: List[np.ndarray],
    rng: random.Random,
    noise_std: float,
    bounds: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    radius = rng.uniform(5.0, 20.0)
    sigma = max(0.9, radius / 4.0)
    amp = rng.uniform(5.0, 12.0) * noise_std
    color = random_color(rng)
    core_sigma = max(0.6, sigma * 0.45)
    core_amp = amp * 1.4

    angle = rng.uniform(0, 2 * math.pi)
    step1 = rng.uniform(4.0, 10.0)
    step2 = rng.uniform(2.0, 6.0)
    step3 = rng.uniform(2.0, 6.0)
    dx = math.cos(angle)
    dy = math.sin(angle)

    total_step = step1 + step2 + step3
    margin = int(math.ceil(3 * sigma + total_step + 2))

    x0 = rng.uniform(left + margin, right - margin)
    y0 = rng.uniform(top + margin, bottom - margin)

    positions = [
        (x0, y0),
        (x0 + dx * step1, y0 + dy * step1),
        (x0 + dx * (step1 + step2), y0 + dy * (step1 + step2)),
        (x0 + dx * total_step, y0 + dy * total_step),
    ]

    for i, frame in enumerate(frames):
        x, y = positions[i]
        add_gaussian(frame, x, y, amp, sigma, color)
        add_gaussian(frame, x, y, core_amp, core_sigma, color)
        if i > 0:
            px, py = positions[i - 1]
            add_gaussian(frame, px, py, -0.5 * amp, sigma, color)

    box_size = max(10, int(math.ceil(radius * 2)))
    box_x = int(round(x0 - box_size / 2))
    box_y = int(round(y0 - box_size / 2))
    return box_x, box_y, box_size, box_size


def write_metadata_csv(path: Path, subjects: List[str], output_dir: Path) -> None:
    header = "subject_id,frame_paths,n_frames,metadata,ra,dec\n"
    lines = [header]
    for sid in subjects:
        frames = [
            str(output_dir / "subjects_groundtruth" / sid / f"frame_{i:02d}.jpg") for i in range(4)
        ]
        line = f'{sid},"{frames}",4,{{}},,\n'
        lines.append(line)
    path.write_text("".join(lines))


def draw_annotation_circles(
    image: Image.Image,
    mover_boxes: List[List[int]],
    dipole_boxes: List[List[int]],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x, y, w, h in mover_boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        r = max(w, h) / 2.0
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=(255, 255, 255), width=2)
    for x, y, w, h in dipole_boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        r = max(w, h) / 2.0
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=(255, 170, 60), width=2)
    return output


def write_side_by_side_gif(
    background_frames: List[np.ndarray],
    synthetic_frames: List[np.ndarray],
    mover_boxes: List[List[int]],
    dipole_boxes: List[List[int]],
    out_path: Path,
) -> None:
    gif_frames: List[Image.Image] = []
    for bg, syn in zip(background_frames, synthetic_frames):
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        syn_rgb = cv2.cvtColor(syn, cv2.COLOR_BGR2RGB)
        syn_img = Image.fromarray(syn_rgb)
        syn_img = draw_annotation_circles(syn_img, mover_boxes, dipole_boxes)
        bg_img = Image.fromarray(bg_rgb)
        bg_img = draw_annotation_circles(bg_img, mover_boxes, dipole_boxes)
        side = Image.new("RGB", (bg_img.width + syn_img.width, bg_img.height))
        side.paste(bg_img, (0, 0))
        side.paste(syn_img, (bg_img.width, 0))
        border = ImageDraw.Draw(side)
        border.rectangle(
            [0, 0, bg_img.width - 1, bg_img.height - 1], outline=(255, 255, 255), width=1
        )
        border.rectangle(
            [bg_img.width, 0, bg_img.width + syn_img.width - 1, syn_img.height - 1],
            outline=(255, 255, 255),
            width=1,
        )
        gif_frames.append(side)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ground truth sequences")
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=Path("data/backyard_worlds/subjects"),
        help="Directory containing subject frame folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backyard_worlds/synthetic_ground_truth"),
        help="Output directory",
    )
    parser.add_argument("--num-sequences", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-cropped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save cropped frames and cropped-coordinate annotations",
    )
    parser.add_argument("--min-movers", type=int, default=0)
    parser.add_argument("--max-movers", type=int, default=2)
    parser.add_argument("--min-dipoles", type=int, default=0)
    parser.add_argument("--max-dipoles", type=int, default=2)
    parser.add_argument(
        "--write-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write side-by-side GIFs with annotation circles",
    )
    parser.add_argument(
        "--ensure-object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure at least one mover or dipole per sequence",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    subject_dirs = list_subject_dirs(args.background_dir)
    if not subject_dirs:
        raise SystemExit(f"No subject folders found in {args.background_dir}")

    out_subjects_dir = args.output_dir / "subjects_groundtruth"
    out_subjects_dir.mkdir(parents=True, exist_ok=True)

    annotations: Dict[str, Dict] = {}
    subject_ids: List[str] = []

    for idx in range(args.num_sequences):
        src_dir = rng.choice(subject_dirs)
        frames = load_frames(src_dir)

        original_h, original_w = frames[0].shape[:2]
        crop = crop_bounds((original_h, original_w))

        if args.output_cropped:
            frames = apply_crop(frames)
            bounds = (0, 0, frames[0].shape[1], frames[0].shape[0])
        else:
            bounds = crop

        background_frames = [f.copy() for f in frames]

        frames = [f.astype(np.float32) for f in frames]
        noise_std = float(np.mean([estimate_noise_std(f.astype(np.uint8)) for f in frames]))

        movers = rng.randint(args.min_movers, args.max_movers)
        dipoles = rng.randint(args.min_dipoles, args.max_dipoles)
        if args.ensure_object and movers + dipoles == 0:
            if rng.random() < 0.5:
                movers = 1
            else:
                dipoles = 1

        mover_boxes: List[List[int]] = []
        dipole_boxes: List[List[int]] = []

        for _ in range(movers):
            box = place_mover(frames, rng, noise_std, bounds)
            mover_boxes.append([int(v) for v in box])

        for _ in range(dipoles):
            box = place_dipole(frames, rng, noise_std, bounds)
            dipole_boxes.append([int(v) for v in box])

        frames_out = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]

        subject_id = f"synthetic_{idx:06d}"
        subject_ids.append(subject_id)
        out_dir = out_subjects_dir / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames_out):
            out_path = out_dir / f"frame_{i:02d}.jpg"
            cv2.imwrite(str(out_path), frame)

        annotations[subject_id] = {
            "notes": "synthetic",
            "mover_circles": mover_boxes,
            "dipole_circles": dipole_boxes,
            "artifact_circles": [],
        }

        if args.write_gifs:
            gif_dir = args.output_dir / "gifs"
            gif_path = gif_dir / f"{subject_id}.gif"
            write_side_by_side_gif(
                background_frames, frames_out, mover_boxes, dipole_boxes, gif_path
            )

    annotations_path = args.output_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations, indent=2))
    write_metadata_csv(
        args.output_dir / "subjects_groundtruth_metadata.csv", subject_ids, args.output_dir
    )


if __name__ == "__main__":
    main()
