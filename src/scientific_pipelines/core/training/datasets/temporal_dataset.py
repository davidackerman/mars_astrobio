"""
Temporal dataset for Backyard Worlds object detection.

Loads 4-frame sequences from subjects_groundtruth directories and
generates keypoint heatmap targets for training.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .augmentation import TemporalSequenceAugmentation
    from .crop_config import crop_frames, adjust_keypoints
except ImportError:
    # Allow running as script for testing
    from augmentation import TemporalSequenceAugmentation
    from crop_config import crop_frames, adjust_keypoints

logger = logging.getLogger(__name__)


class BackyardWorldsTemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal object detection.

    Loads 4-frame sequences with multi-label annotations and generates
    keypoint heatmap targets for training.

    Args:
        data_dir: Root directory containing subjects_groundtruth/
        annotations_path: Path to annotations.json
        transform: Augmentation pipeline (TemporalSequenceAugmentation)
        heatmap_size: Output heatmap resolution (H, W)
        input_size: Input image size (H, W)
        class_names: List of class names (default: ['mover', 'dipole'])
    """

    def __init__(
        self,
        data_dir: Path,
        annotations_path: Path,
        transform: Optional[TemporalSequenceAugmentation] = None,
        heatmap_size: Tuple[int, int] = (8, 8),
        input_size: Tuple[int, int] = (256, 256),
        class_names: List[str] = ['mover', 'dipole'],
        apply_crop: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.apply_crop = apply_crop

        # Load annotations
        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Get list of annotated subject IDs
        self.subject_ids = list(self.annotations.keys())

        logger.info(f"Loaded {len(self.subject_ids)} annotated sequences")
        logger.info(f"Class names: {self.class_names}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and process one sequence.

        Returns:
            Dictionary containing:
            - frames: (4, 3, H, W) tensor
            - class_labels: (num_classes,) binary tensor
            - heatmap_targets: (num_classes, H', W') heatmap tensor
            - sequence_id: str
            - keypoints: List of (x, y) tuples (for debugging)
        """
        subject_id = self.subject_ids[idx]
        annotation = self.annotations[subject_id]

        # Load frames from subjects_groundtruth/{subject_id}/
        frames_dir = self.data_dir / "subjects_groundtruth" / subject_id
        frames, original_size = self._load_frames(frames_dir)

        # Parse annotations to get keypoints and labels
        keypoints, labels = self._parse_annotations(annotation, original_size)
        if self.apply_crop:
            keypoints, labels = adjust_keypoints(keypoints, labels, original_size)

        # Apply augmentation
        if self.transform is not None:
            frames_tensor, keypoints_aug, labels = self.transform(frames, keypoints, labels)
        else:
            # No augmentation: just convert to tensor
            frames_tensor, keypoints_aug = self._to_tensor(frames, keypoints)

        # Generate class labels (multi-label binary vector)
        class_labels = self._get_class_labels(labels)

        # Generate heatmap targets
        heatmap_targets = self._generate_heatmap_targets(keypoints_aug, labels)

        return {
            'frames': frames_tensor,
            'class_labels': class_labels,
            'heatmap_targets': heatmap_targets,
            'sequence_id': subject_id,
            'keypoints': keypoints_aug,  # For debugging/visualization
            'labels': labels,
        }

    def _load_frames(self, frames_dir: Path) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        """Load 4 frames from directory."""
        frames = []
        for i in range(4):
            frame_path = frames_dir / f"frame_{i:02d}.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")

            # Load as RGB
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        original_size = frames[0].shape[:2]
        if self.apply_crop:
            frames = crop_frames(frames)
        return frames, original_size

    def _parse_annotations(
        self,
        annotation: Dict,
        original_size: Tuple[int, int],
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Parse annotations to extract keypoints and labels.

        Args:
            annotation: Annotation dict from annotations.json
            original_size: (height, width) of original frames

        Returns:
            keypoints: List of (x, y) center coordinates
            labels: List of class IDs
        """
        keypoints = []
        labels = []

        # Extract movers (class 0)
        mover_boxes = annotation.get("mover_circles", [])
        for box in mover_boxes:
            x, y, w, h = box
            center_x = x + w / 2
            center_y = y + h / 2
            keypoints.append((center_x, center_y))
            labels.append(0)  # mover class

        # Extract dipoles (class 1)
        dipole_boxes = annotation.get("dipole_circles", [])
        for box in dipole_boxes:
            x, y, w, h = box
            center_x = x + w / 2
            center_y = y + h / 2
            keypoints.append((center_x, center_y))
            labels.append(1)  # dipole class

        return keypoints, labels

    def _get_class_labels(self, labels: List[int]) -> torch.Tensor:
        """
        Generate multi-label binary vector.

        Args:
            labels: List of class IDs

        Returns:
            (num_classes,) binary tensor where 1 = class present
        """
        class_labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if 0 <= label < self.num_classes:
                class_labels[label] = 1.0
        return class_labels

    def _generate_heatmap_targets(
        self,
        keypoints: List[Tuple[float, float]],
        labels: List[int],
    ) -> torch.Tensor:
        """
        Generate Gaussian heatmap targets.

        Args:
            keypoints: List of (x, y) center coordinates in input image space
            labels: List of class IDs

        Returns:
            (num_classes, H, W) heatmap tensor with Gaussian blobs at object centers
        """
        H, W = self.heatmap_size
        heatmaps = torch.zeros(self.num_classes, H, W, dtype=torch.float32)

        # Scale factor from input to heatmap
        scale_x = W / self.input_size[1]  # width
        scale_y = H / self.input_size[0]  # height

        for (x, y), label in zip(keypoints, labels):
            if not (0 <= label < self.num_classes):
                continue

            # Map to heatmap coordinates
            hm_x = x * scale_x
            hm_y = y * scale_y

            # Check bounds
            if not (0 <= hm_x < W and 0 <= hm_y < H):
                continue

            # Draw Gaussian blob
            radius = max(1, int(3 * min(scale_x, scale_y)))  # Adaptive radius
            self._draw_gaussian(heatmaps[label], (hm_x, hm_y), radius)

        return heatmaps

    def _draw_gaussian(
        self,
        heatmap: torch.Tensor,
        center: Tuple[float, float],
        radius: int,
    ):
        """
        Draw 2D Gaussian blob on heatmap (in-place).

        Args:
            heatmap: (H, W) tensor
            center: (x, y) center coordinates
            radius: Gaussian radius in pixels
        """
        H, W = heatmap.shape
        cx, cy = center
        cx_int, cy_int = int(cx), int(cy)

        # Generate Gaussian kernel
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)

        # Determine region to draw
        left = max(0, cx_int - radius)
        right = min(W, cx_int + radius + 1)
        top = max(0, cy_int - radius)
        bottom = min(H, cy_int + radius + 1)

        # Compute kernel region
        k_left = radius - (cx_int - left)
        k_right = radius + (right - cx_int)
        k_top = radius - (cy_int - top)
        k_bottom = radius + (bottom - cy_int)

        # Apply Gaussian (take maximum to handle overlaps)
        masked_heatmap = heatmap[top:bottom, left:right]
        masked_gaussian = gaussian[k_top:k_bottom, k_left:k_right]

        torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    @staticmethod
    def _gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> torch.Tensor:
        """Generate 2D Gaussian kernel."""
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return torch.from_numpy(h).float()

    def _to_tensor(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Convert frames to tensor without augmentation."""
        # Resize and normalize
        frames_resized = []
        for frame in frames:
            frame_resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frames_resized.append(frame_tensor)

        frames_tensor = torch.stack(frames_resized, dim=0)

        # Resize keypoints
        orig_h, orig_w = frames[0].shape[:2]
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        keypoints_resized = [(x * scale_x, y * scale_y) for x, y in keypoints]

        return frames_tensor, keypoints_resized


def test_dataset():
    """Test the BackyardWorldsTemporalDataset."""
    print("Testing BackyardWorldsTemporalDataset...")

    data_dir = Path("/mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/ground_truth")
    annotations_path = data_dir / "annotations.json"

    # Create dataset (no augmentation)
    dataset = BackyardWorldsTemporalDataset(
        data_dir=data_dir,
        annotations_path=annotations_path,
        transform=None,
        heatmap_size=(8, 8),
        input_size=(256, 256),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Subject IDs: {dataset.subject_ids}")

    # Load first sample
    sample = dataset[0]

    print(f"\nSample 0 ({sample['sequence_id']}):")
    print(f"  Frames shape: {sample['frames'].shape}")
    print(f"  Class labels: {sample['class_labels']}")
    print(f"  Heatmap targets shape: {sample['heatmap_targets'].shape}")
    print(f"  Number of keypoints: {len(sample['keypoints'])}")
    print(f"  Keypoints: {sample['keypoints']}")
    print(f"  Labels: {sample['labels']}")

    # Check heatmap has peaks at keypoints
    for class_id in range(dataset.num_classes):
        heatmap = sample['heatmap_targets'][class_id]
        max_val = heatmap.max().item()
        print(f"  Class {class_id} heatmap max: {max_val:.3f}")

    # Test with augmentation
    print("\nTesting with augmentation...")
    try:
        from augmentation import TemporalSequenceAugmentation
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from augmentation import TemporalSequenceAugmentation

    aug = TemporalSequenceAugmentation(input_size=(256, 256), training=True)
    dataset_aug = BackyardWorldsTemporalDataset(
        data_dir=data_dir,
        annotations_path=annotations_path,
        transform=aug,
    )

    sample_aug = dataset_aug[0]
    print(f"Augmented frames shape: {sample_aug['frames'].shape}")
    print(f"Augmented keypoints: {sample_aug['keypoints']}")

    print("\nTest passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset()
