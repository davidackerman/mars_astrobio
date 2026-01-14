"""
Temporal crop dataset for Backyard Worlds classification.

Builds fixed-size crops from 4-frame sequences and assigns multi-label
targets indicating whether a mover or dipole appears inside each crop.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .augmentation import TemporalSequenceAugmentation
    from .crop_config import crop_frames, adjust_keypoints, crop_bounds
except ImportError:
    from augmentation import TemporalSequenceAugmentation
    from crop_config import crop_frames, adjust_keypoints, crop_bounds

logger = logging.getLogger(__name__)


class BackyardWorldsTemporalCropDataset(Dataset):
    """
    PyTorch Dataset that returns cropped temporal sequences for classification.

    Args:
        data_dir: Root directory containing subjects_groundtruth/
        annotations_path: Path to annotations.json
        transform: Augmentation pipeline (TemporalSequenceAugmentation)
        crop_size: Crop size (height, width)
        samples_per_subject: Number of crops to generate per subject
        positive_fraction: Fraction of crops centered on a labeled object
        min_negative_distance: Minimum distance (pixels) between negative crop
            centers and any labeled object
        class_names: List of class names (default: ['mover', 'dipole'])
        seed: Random seed for crop sampling
    """

    def __init__(
        self,
        data_dir: Path,
        annotations_path: Path,
        transform: Optional[TemporalSequenceAugmentation] = None,
        crop_size: Tuple[int, int] = (128, 128),
        samples_per_subject: int = 50,
        positive_fraction: float = 0.5,
        min_negative_distance: int = 20,
        class_names: List[str] = ['mover', 'dipole'],
        seed: int = 0,
        samples: Optional[List[Dict]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        self.crop_size = crop_size
        self.samples_per_subject = samples_per_subject
        self.positive_fraction = positive_fraction
        self.min_negative_distance = min_negative_distance
        self.class_names = class_names
        self.num_classes = len(class_names)
        self._rng = random.Random(seed)

        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.subject_ids = list(self.annotations.keys())
        self.subject_keypoints = {
            subject_id: self._parse_annotations(self.annotations[subject_id])
            for subject_id in self.subject_ids
        }
        self.subject_sizes, self.subject_original_sizes = self._load_subject_sizes()
        self.samples = samples if samples is not None else self._build_samples()

        logger.info("Loaded %d annotated sequences", len(self.subject_ids))
        logger.info("Generated %d crop samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        center = sample['center']

        frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
        frames, original_size = self._load_frames(frames_dir)

        keypoints, labels = self.subject_keypoints[subject_id]
        keypoints, labels = adjust_keypoints(keypoints, labels, original_size)

        crop_frames, keypoints_crop, labels_crop, crop_origin = self._crop_sequence(
            frames, keypoints, labels, center
        )

        if self.transform is not None:
            frames_tensor, keypoints_aug, labels_crop = self.transform(
                crop_frames, keypoints_crop, labels_crop
            )
        else:
            frames_tensor, keypoints_aug = self._to_tensor(crop_frames, keypoints_crop)

        class_labels = self._get_class_labels(labels_crop)

        return {
            'frames': frames_tensor,
            'class_labels': class_labels,
            'sequence_id': subject_id,
            'crop_center': center,
            'crop_origin': crop_origin,
            'keypoints': keypoints_aug,
            'labels': labels_crop,
        }

    def _build_samples(self) -> List[Dict]:
        samples = []
        for subject_id in self.subject_ids:
            keypoints, labels = self.subject_keypoints[subject_id]
            original_size = self.subject_original_sizes[subject_id]
            keypoints, labels = adjust_keypoints(keypoints, labels, original_size)
            image_size = self.subject_sizes[subject_id]
            positive_keypoints = list(zip(keypoints, labels))

            num_positive = int(self.samples_per_subject * self.positive_fraction)
            num_negative = self.samples_per_subject - num_positive

            if not positive_keypoints:
                num_positive = 0
                num_negative = self.samples_per_subject

            for _ in range(num_positive):
                (x, y), label = self._rng.choice(positive_keypoints)
                samples.append({
                    'subject_id': subject_id,
                    'center': (x, y),
                })

            for _ in range(num_negative):
                center = self._sample_negative_center(keypoints, image_size)
                samples.append({
                    'subject_id': subject_id,
                    'center': center,
                })

        return samples

    def _load_frames(self, frames_dir: Path) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        frames = []
        for i in range(4):
            frame_path = frames_dir / f"frame_{i:02d}.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        original_size = frames[0].shape[:2]
        return crop_frames(frames), original_size

    def _load_subject_sizes(self) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        sizes = {}
        original_sizes = {}
        for subject_id in self.subject_ids:
            frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
            frame_path = frames_dir / "frame_00.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            h, w = frame.shape[:2]
            original_sizes[subject_id] = (h, w)
            left, top, right, bottom = crop_bounds((h, w))
            sizes[subject_id] = (bottom - top, right - left)
        return sizes, original_sizes

    def _parse_annotations(
        self,
        annotation: Dict,
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        keypoints = []
        labels = []

        mover_boxes = annotation.get("mover_circles", [])
        for box in mover_boxes:
            x, y, w, h = box
            keypoints.append((x + w / 2, y + h / 2))
            labels.append(0)

        dipole_boxes = annotation.get("dipole_circles", [])
        for box in dipole_boxes:
            x, y, w, h = box
            keypoints.append((x + w / 2, y + h / 2))
            labels.append(1)

        return keypoints, labels

    def _crop_sequence(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
        labels: List[int],
        center: Optional[Tuple[float, float]],
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[int], Tuple[int, int]]:
        h, w = frames[0].shape[:2]
        if center is None:
            center = self._sample_negative_center(keypoints, (h, w))
        left, top, right, bottom = self._compute_crop_bounds(center, (h, w))

        crop_frames = [f[top:bottom, left:right].copy() for f in frames]

        keypoints_crop = []
        labels_crop = []
        for (x, y), label in zip(keypoints, labels):
            if left <= x < right and top <= y < bottom:
                keypoints_crop.append((x - left, y - top))
                labels_crop.append(label)

        return crop_frames, keypoints_crop, labels_crop, (left, top)

    def _compute_crop_bounds(
        self,
        center: Tuple[float, float],
        image_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        h, w = image_size
        crop_h, crop_w = self.crop_size
        cx, cy = center
        half_h = crop_h // 2
        half_w = crop_w // 2
        left = int(np.clip(cx - half_w, 0, w - crop_w))
        top = int(np.clip(cy - half_h, 0, h - crop_h))
        right = left + crop_w
        bottom = top + crop_h
        return left, top, right, bottom

    def get_sample_class_labels(self) -> List[List[float]]:
        labels_list: List[List[float]] = []
        for sample in self.samples:
            subject_id = sample['subject_id']
            center = sample['center']
            keypoints, labels = self.subject_keypoints[subject_id]
            original_size = self.subject_original_sizes[subject_id]
            keypoints, labels = adjust_keypoints(keypoints, labels, original_size)
            image_size = self.subject_sizes[subject_id]
            left, top, right, bottom = self._compute_crop_bounds(center, image_size)
            labels_in_crop = []
            for (x, y), label in zip(keypoints, labels):
                if left <= x < right and top <= y < bottom:
                    labels_in_crop.append(label)
            class_labels = self._get_class_labels(labels_in_crop).tolist()
            labels_list.append(class_labels)
        return labels_list

    def _sample_negative_center(
        self,
        keypoints: List[Tuple[float, float]],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        h, w = image_size
        crop_h, crop_w = self.crop_size
        half_h = crop_h // 2
        half_w = crop_w // 2

        min_dist_sq = self.min_negative_distance ** 2
        for _ in range(50):
            cx = self._rng.uniform(half_w, w - half_w)
            cy = self._rng.uniform(half_h, h - half_h)
            if all(((cx - x) ** 2 + (cy - y) ** 2) >= min_dist_sq for x, y in keypoints):
                return (cx, cy)

        return (
            self._rng.uniform(half_w, w - half_w),
            self._rng.uniform(half_h, h - half_h),
        )

    def _get_class_labels(self, labels: List[int]) -> torch.Tensor:
        class_labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if 0 <= label < self.num_classes:
                class_labels[label] = 1.0
        return class_labels

    def _to_tensor(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        frames_tensor = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames_tensor.append(frame_tensor)
        frames_tensor = torch.stack(frames_tensor, dim=0)
        return frames_tensor, keypoints
