"""
Shared crop configuration for Backyard Worlds training/inference.
"""

from typing import Iterable, List, Tuple

import numpy as np

# Crop out left border and bottom legend from ground truth frames.
CROP_LEFT = 21
CROP_TOP = 0
CROP_BOTTOM = 508  # exclusive
CROP_RIGHT = None  # use full width


def crop_bounds(image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) crop bounds for an image."""
    height, width = image_size
    left = max(0, CROP_LEFT)
    top = max(0, CROP_TOP)
    right = width if CROP_RIGHT is None else min(width, CROP_RIGHT)
    bottom = min(height, CROP_BOTTOM)
    return left, top, right, bottom


def crop_frames(frames: Iterable[np.ndarray]) -> List[np.ndarray]:
    """Crop each frame using the configured bounds."""
    frames_list = list(frames)
    if not frames_list:
        return []
    h, w = frames_list[0].shape[:2]
    left, top, right, bottom = crop_bounds((h, w))
    return [frame[top:bottom, left:right].copy() for frame in frames_list]


def adjust_keypoints(
    keypoints: Iterable[Tuple[float, float]],
    labels: Iterable[int],
    image_size: Tuple[int, int],
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Adjust keypoints to cropped coordinates and drop out-of-bounds points."""
    left, top, right, bottom = crop_bounds(image_size)
    adj_keypoints = []
    adj_labels = []
    for (x, y), label in zip(keypoints, labels):
        if left <= x < right and top <= y < bottom:
            adj_keypoints.append((x - left, y - top))
            adj_labels.append(label)
    return adj_keypoints, adj_labels
