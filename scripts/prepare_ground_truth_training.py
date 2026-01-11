#!/usr/bin/env python3
"""Prepare ground truth data for training temporal detector.

This script:
1. Extracts frames from GIF sequences
2. Loads manual annotations from annotations.json
3. Creates inpainted versions (removes white circles and text annotations)
4. Outputs ground_truth_labels.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_gif(gif_path: Path, output_dir: Path, target_size: Tuple[int, int] = (256, 256)) -> List[Path]:
    """Extract frames from GIF and save as individual images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gif = Image.open(gif_path)
    frame_paths = []

    try:
        frame_idx = 0
        while True:
            frame = gif.copy()
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.LANCZOS)
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            frame_path = output_dir / f"frame_{frame_idx:02d}.jpg"
            frame.save(frame_path, 'JPEG', quality=95)
            frame_paths.append(frame_path)

            frame_idx += 1
            gif.seek(frame_idx)

    except EOFError:
        pass

    logger.info(f"Extracted {len(frame_paths)} frames from {gif_path.name}")
    return frame_paths


def create_annotation_mask(img: np.ndarray) -> np.ndarray:
    """Create a mask of white/orange annotations (circles, ovals, text) to inpaint.

    Returns binary mask where 255 = annotation pixels to remove.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask bright white pixels (circles and text)
    # Very bright threshold to get only annotations, not stars
    white_lower = np.array([245, 245, 245])
    white_upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(img_rgb, white_lower, white_upper)

    # Mask orange pixels (for planet 9 ovals)
    orange_lower = np.array([200, 100, 0])
    orange_upper = np.array([255, 200, 100])
    orange_mask = cv2.inRange(img_rgb, orange_lower, orange_upper)

    # Combine masks
    mask = cv2.bitwise_or(white_mask, orange_mask)

    # Clean up - remove tiny isolated pixels (individual stars)
    # Keep connected components (circles, text)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilate slightly to ensure we get the full annotation stroke
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def inpaint_frame(frame_path: Path, output_path: Path) -> None:
    """Remove annotation circles and text using inpainting."""
    img = cv2.imread(str(frame_path))
    if img is None:
        logger.warning(f"Could not read {frame_path}")
        return

    # Create mask of annotations
    mask = create_annotation_mask(img)

    # Inpaint using Telea algorithm
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), inpainted)


def process_sequence(
    gif_path: Path,
    annotations: Dict,
    output_base: Path,
    sequence_id: str
) -> Dict:
    """Process a single GIF sequence."""
    logger.info(f"Processing {sequence_id}...")

    # Create output directories
    frames_dir = output_base / "frames" / sequence_id
    inpainted_dir = output_base / "frames_inpainted" / sequence_id

    # Extract frames
    frame_paths = extract_frames_from_gif(gif_path, frames_dir)

    if not frame_paths:
        logger.warning(f"No frames extracted from {gif_path}")
        return None

    # Create inpainted versions (remove annotations)
    for frame_path in frame_paths:
        inpainted_path = inpainted_dir / frame_path.name
        inpaint_frame(frame_path, inpainted_path)

    # Get annotation info
    annotation_info = annotations.get(sequence_id, {})
    notes = annotation_info.get("notes", "")

    mover_circles = annotation_info.get("mover_circles")
    dipole_circles = annotation_info.get("dipole_circles")
    artifact_circles = annotation_info.get("artifact_circles")

    if mover_circles is None and dipole_circles is None and artifact_circles is None:
        mover_circles = annotation_info.get("mover_boxes")
        dipole_circles = annotation_info.get("dipole_boxes")
        artifact_circles = annotation_info.get("artifact_boxes")

    if mover_circles is None and dipole_circles is None and artifact_circles is None:
        obj_type = annotation_info.get("type", "unknown")
        circles = annotation_info.get("circles", annotation_info.get("boxes", []))
        mover_circles = circles if obj_type == "mover" else []
        dipole_circles = circles if obj_type == "dipole" else []
        artifact_circles = circles if obj_type == "artifact" else []

    mover_circles = mover_circles or []
    dipole_circles = dipole_circles or []
    artifact_circles = artifact_circles or []

    # Create label entry
    label_entry = {
        "sequence_id": sequence_id,
        "has_mover": len(mover_circles) > 0,
        "has_dipole": len(dipole_circles) > 0,
        "has_artifact": len(artifact_circles) > 0,
        "num_frames": len(frame_paths),
        "notes": notes,
        "mover_circles": json.dumps(mover_circles),
        "dipole_circles": json.dumps(dipole_circles),
        "artifact_circles": json.dumps(artifact_circles),
    }

    return label_entry


def main():
    parser = argparse.ArgumentParser(description="Prepare ground truth training data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth"),
        help="Input directory containing GIF files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth/processed"),
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/backyard_worlds/ground_truth/annotations.json"),
        help="Path to annotations JSON file"
    )

    args = parser.parse_args()

    # Load annotations
    if not args.annotations.exists():
        logger.error(f"Annotations file not found: {args.annotations}")
        return

    with args.annotations.open() as f:
        annotations = json.load(f)

    # Find all GIF files (skip annotated overlays)
    gif_files = [
        path for path in args.input.glob("*.gif")
        if not path.name.endswith("_annotated.gif")
    ]

    if not gif_files:
        logger.error(f"No GIF files found in {args.input}")
        return

    logger.info(f"Found {len(gif_files)} GIF files")
    logger.info(f"Loaded annotations for {len(annotations)} sequences")

    # Process each sequence
    labels = []
    for gif_path in sorted(gif_files):
        sequence_id = gif_path.stem
        label_entry = process_sequence(gif_path, annotations, args.output, sequence_id)
        if label_entry:
            labels.append(label_entry)

    # Save labels to CSV
    labels_path = args.output / "ground_truth_labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if labels:
        fieldnames = ["sequence_id", "has_mover", "has_dipole", "has_artifact",
                     "num_frames", "mover_circles", "dipole_circles", "artifact_circles", "notes"]

        with labels_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(labels)

        logger.info(f"Saved labels to {labels_path}")
        logger.info(f"Processed {len(labels)} sequences")

    logger.info("Data preparation complete!")
    logger.info(f"Original frames: {args.output / 'frames'}")
    logger.info(f"Inpainted frames (annotations removed): {args.output / 'frames_inpainted'}")


if __name__ == "__main__":
    main()
