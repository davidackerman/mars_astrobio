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
        sampling_mode: "random" for random positive/negative sampling,
            "grid" for sliding-window style sampling over each subject
        grid_stride: Stride (pixels) for grid sampling mode
        grid_max_samples_per_subject: Cap grid samples per subject (None or 0 = no cap)
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
        apply_crop: bool = True,
        positive_jitter: Optional[Tuple[int, int]] = None,
        negative_bright_fraction: float = 0.0,
        negative_bright_percentile: float = 85.0,
        negative_bright_samples: int = 200,
        positive_mover_min_frames: int = 0,
        lazy_brightness: bool = False,
        ignore_dipoles: bool = False,
        sampling_mode: str = "random",
        grid_stride: Optional[int] = None,
        grid_max_samples_per_subject: Optional[int] = None,
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
        self.apply_crop = apply_crop
        self.positive_jitter = positive_jitter
        self.negative_bright_fraction = negative_bright_fraction
        self.negative_bright_percentile = negative_bright_percentile
        self.negative_bright_samples = negative_bright_samples
        self.positive_mover_min_frames = positive_mover_min_frames
        self.lazy_brightness = lazy_brightness
        self.ignore_dipoles = ignore_dipoles
        self.sampling_mode = sampling_mode
        self.grid_stride = grid_stride
        self.grid_max_samples_per_subject = grid_max_samples_per_subject
        self.seed = seed
        self._rng = random.Random(seed)
        # Cache brightest centers for bright-negative sampling.
        self._brightest_centers: Dict[str, Optional[Tuple[float, float]]] = {}

        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.subject_ids = list(self.annotations.keys())
        self.subject_keypoints = {
            subject_id: self._parse_annotations(self.annotations[subject_id])
            for subject_id in self.subject_ids
        }
        self.subject_tracks = {
            subject_id: self._parse_tracks(self.annotations[subject_id])
            for subject_id in self.subject_ids
        }
        (
            self.subject_sizes,
            self.subject_original_sizes,
            self.subject_brightness_integrals,
            self.subject_bright_thresholds,
        ) = self._load_subject_sizes()
        if self.apply_crop:
            self._adjust_tracks_for_crop()
        if self.sampling_mode not in ("random", "grid"):
            raise ValueError(f"sampling_mode must be 'random' or 'grid', got {self.sampling_mode!r}")
        if self.sampling_mode == "grid" and (self.grid_stride is None or self.grid_stride <= 0):
            raise ValueError("grid_stride must be set to a positive integer when sampling_mode='grid'")

        self.samples = samples if samples is not None else self._build_samples()

        logger.info("Loaded %d annotated sequences", len(self.subject_ids))
        logger.info("Generated %d crop samples", len(self.samples))

    def resample(self, seed: Optional[int] = None) -> None:
        """Regenerate crop samples using a new random seed."""
        if seed is None:
            self.seed += 1
        else:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        center = sample['center']

        frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
        frames, original_size = self._load_frames(frames_dir)

        keypoints, labels = self.subject_keypoints[subject_id]
        if self.apply_crop:
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
        if self.sampling_mode == "grid":
            return self._build_grid_samples()
        samples = []
        for subject_id in self.subject_ids:
            keypoints, labels = self.subject_keypoints[subject_id]
            original_size = self.subject_original_sizes[subject_id]
            if self.apply_crop:
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
                center = self._jitter_positive_center((x, y), image_size)
                if label == 0 and self.positive_mover_min_frames > 0:
                    center = self._ensure_mover_frames_in_crop(
                        subject_id,
                        center,
                        image_size,
                    )
                samples.append({
                    'subject_id': subject_id,
                    'center': center,
                })

            num_bright = int(round(num_negative * self.negative_bright_fraction))
            for _ in range(num_bright):
                center = self._sample_bright_negative_center(subject_id, keypoints, image_size)
                samples.append({
                    'subject_id': subject_id,
                    'center': center,
                })

            for _ in range(num_negative - num_bright):
                center = self._sample_negative_center(keypoints, image_size)
                samples.append({
                    'subject_id': subject_id,
                    'center': center,
                })

        return samples

    def _build_grid_samples(self) -> List[Dict]:
        samples = []
        for subject_id in self.subject_ids:
            image_size = self.subject_sizes[subject_id]
            centers = list(self._iter_grid_centers(image_size))
            if self.grid_max_samples_per_subject and self.grid_max_samples_per_subject > 0:
                if len(centers) > self.grid_max_samples_per_subject:
                    centers = self._rng.sample(centers, self.grid_max_samples_per_subject)
            for center in centers:
                samples.append({
                    'subject_id': subject_id,
                    'center': center,
                })
        return samples

    def _iter_grid_centers(
        self,
        image_size: Tuple[int, int],
    ):
        h, w = image_size
        crop_h, crop_w = self.crop_size
        stride = int(self.grid_stride or 1)
        max_top = max(1, h - crop_h + 1)
        max_left = max(1, w - crop_w + 1)
        for top in range(0, max_top, stride):
            for left in range(0, max_left, stride):
                yield (left + crop_w / 2.0, top + crop_h / 2.0)

    def _jitter_positive_center(
        self,
        center: Tuple[float, float],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        if self.positive_jitter is None:
            return center
        h, w = image_size
        crop_h, crop_w = self.crop_size
        jitter_y, jitter_x = self.positive_jitter
        cx, cy = center
        dx = self._rng.uniform(-jitter_x, jitter_x)
        dy = self._rng.uniform(-jitter_y, jitter_y)
        cx = float(np.clip(cx + dx, crop_w / 2, w - crop_w / 2))
        cy = float(np.clip(cy + dy, crop_h / 2, h - crop_h / 2))
        return (cx, cy)

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
        if self.apply_crop:
            frames = crop_frames(frames)
        return frames, original_size

    def _load_subject_sizes(
        self,
    ) -> Tuple[
        Dict[str, Tuple[int, int]],
        Dict[str, Tuple[int, int]],
        Dict[str, np.ndarray],
        Dict[str, float],
    ]:
        sizes = {}
        original_sizes = {}
        brightness_integrals = {}
        bright_thresholds = {}
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
            if self.apply_crop:
                left, top, right, bottom = crop_bounds((h, w))
                sizes[subject_id] = (bottom - top, right - left)
                frame = frame[top:bottom, left:right]
            else:
                sizes[subject_id] = (h, w)
            if not self.lazy_brightness:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                brightness_integrals[subject_id] = cv2.integral(gray)
                bright_thresholds[subject_id] = self._estimate_bright_threshold(
                    brightness_integrals[subject_id],
                    sizes[subject_id],
                )
        return sizes, original_sizes, brightness_integrals, bright_thresholds

    def _estimate_bright_threshold(
        self,
        integral: np.ndarray,
        image_size: Tuple[int, int],
    ) -> float:
        if self.negative_bright_samples <= 0:
            return 0.0
        samples = []
        for _ in range(self.negative_bright_samples):
            cx = self._rng.uniform(self.crop_size[1] / 2, image_size[1] - self.crop_size[1] / 2)
            cy = self._rng.uniform(self.crop_size[0] / 2, image_size[0] - self.crop_size[0] / 2)
            samples.append(self._crop_mean_brightness(integral, (cx, cy), image_size))
        if not samples:
            return 0.0
        return float(np.percentile(samples, self.negative_bright_percentile))

    def _crop_mean_brightness(
        self,
        integral: np.ndarray,
        center: Tuple[float, float],
        image_size: Tuple[int, int],
    ) -> float:
        left, top, right, bottom = self._compute_crop_bounds(center, image_size)
        x1, y1, x2, y2 = left, top, right, bottom
        crop_sum = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return float(crop_sum / ((y2 - y1) * (x2 - x1)))

    def _sample_bright_negative_center(
        self,
        subject_id: str,
        keypoints: List[Tuple[float, float]],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        integral = self.subject_brightness_integrals.get(subject_id)
        threshold = self.subject_bright_thresholds.get(subject_id, 0.0)
        if integral is None:
            integral, threshold = self._ensure_subject_brightness(subject_id, image_size)
        if integral is None:
            return self._sample_negative_center(keypoints, image_size)

        h, w = image_size
        crop_h, crop_w = self.crop_size
        half_h = crop_h // 2
        half_w = crop_w // 2
        min_dist_sq = self.min_negative_distance ** 2

        if self._rng.random() < 0.5:
            brightest_center = self._find_brightest_center(subject_id, keypoints, image_size)
            if brightest_center is not None:
                return brightest_center

        for _ in range(100):
            cx = self._rng.uniform(half_w, w - half_w)
            cy = self._rng.uniform(half_h, h - half_h)
            if not all(((cx - x) ** 2 + (cy - y) ** 2) >= min_dist_sq for x, y in keypoints):
                continue
            mean_brightness = self._crop_mean_brightness(integral, (cx, cy), image_size)
            if mean_brightness >= threshold:
                return (cx, cy)

        return self._sample_negative_center(keypoints, image_size)

    def _find_brightest_center(
        self,
        subject_id: str,
        keypoints: List[Tuple[float, float]],
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        if subject_id in self._brightest_centers:
            return self._brightest_centers[subject_id]

        gray = self._load_subject_gray(subject_id)
        if gray is None:
            self._brightest_centers[subject_id] = None
            return None

        h, w = gray.shape[:2]
        crop_h, crop_w = self.crop_size
        half_h = crop_h // 2
        half_w = crop_w // 2
        min_dist_sq = self.min_negative_distance ** 2

        flat_idx = int(np.argmax(gray))
        cy, cx = divmod(flat_idx, w)
        cx = float(np.clip(cx, half_w, w - half_w))
        cy = float(np.clip(cy, half_h, h - half_h))

        if keypoints and not all(
            ((cx - x) ** 2 + (cy - y) ** 2) >= min_dist_sq for x, y in keypoints
        ):
            self._brightest_centers[subject_id] = None
            return None

        center = (cx, cy)
        self._brightest_centers[subject_id] = center
        return center

    def _ensure_subject_brightness(
        self,
        subject_id: str,
        image_size: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], float]:
        integral = self.subject_brightness_integrals.get(subject_id)
        threshold = self.subject_bright_thresholds.get(subject_id, 0.0)
        if integral is not None:
            return integral, threshold

        frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
        frame_path = frames_dir / "frame_00.jpg"
        if not frame_path.exists():
            return None, 0.0
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None, 0.0
        if self.apply_crop:
            left, top, right, bottom = crop_bounds(frame.shape[:2])
            frame = frame[top:bottom, left:right]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        integral = cv2.integral(gray)
        threshold = self._estimate_bright_threshold(integral, image_size)
        self.subject_brightness_integrals[subject_id] = integral
        self.subject_bright_thresholds[subject_id] = threshold
        return integral, threshold

    def _load_subject_gray(self, subject_id: str) -> Optional[np.ndarray]:
        frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
        frame_path = frames_dir / "frame_00.jpg"
        if not frame_path.exists():
            return None
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        if self.apply_crop:
            left, top, right, bottom = crop_bounds(frame.shape[:2])
            frame = frame[top:bottom, left:right]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def ensure_subject_brightness(self, subject_id: str) -> Tuple[Optional[np.ndarray], float]:
        image_size = self.subject_sizes[subject_id]
        return self._ensure_subject_brightness(subject_id, image_size)

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
        if not self.ignore_dipoles:
            for box in dipole_boxes:
                x, y, w, h = box
                keypoints.append((x + w / 2, y + h / 2))
                labels.append(1)

        return keypoints, labels

    def _parse_tracks(
        self,
        annotation: Dict,
    ) -> Dict[str, List[List[Tuple[float, float]]]]:
        mover_tracks = []
        for track in annotation.get("mover_tracks", []) or []:
            coords = []
            for item in track:
                if len(item) >= 2:
                    coords.append((float(item[0]), float(item[1])))
            if coords:
                mover_tracks.append(coords)
        return {"mover": mover_tracks}

    def _adjust_tracks_for_crop(self) -> None:
        for subject_id, tracks in self.subject_tracks.items():
            mover_tracks = tracks.get("mover", [])
            if not mover_tracks:
                continue
            original_size = self.subject_original_sizes[subject_id]
            left, top, _, _ = crop_bounds(original_size)
            adjusted = []
            for track in mover_tracks:
                adjusted.append([(x - left, y - top) for x, y in track])
            tracks["mover"] = adjusted

    def _ensure_mover_frames_in_crop(
        self,
        subject_id: str,
        center: Tuple[float, float],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        tracks = self.subject_tracks.get(subject_id, {}).get("mover", [])
        if not tracks:
            return center

        for _ in range(25):
            left, top, right, bottom = self._compute_crop_bounds(center, image_size)
            for track in tracks:
                count = sum(1 for x, y in track if left <= x < right and top <= y < bottom)
                if count >= self.positive_mover_min_frames:
                    return center
            center = self._jitter_positive_center(center, image_size)
        return center

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
            if self.apply_crop:
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
