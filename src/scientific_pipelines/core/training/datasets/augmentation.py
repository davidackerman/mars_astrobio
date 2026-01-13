"""
Data augmentation for temporal sequences.

Implements frame-consistent spatial augmentations and copy-paste augmentation
to maximize training diversity from a tiny dataset.
"""

import random
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms as T


class TemporalSequenceAugmentation:
    """
    Heavy augmentation pipeline for temporal sequences.

    All spatial augmentations applied consistently across all 4 frames
    to preserve motion patterns. Temporal order is PRESERVED.

    Args:
        input_size: Target image size (height, width)
        training: If True, apply augmentations. If False, only resize + normalize
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        training: bool = True,
    ):
        self.input_size = input_size
        self.training = training

        # ImageNet normalization (for ResNet pretrained weights)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
        labels: List[int],
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]], List[int]]:
        """
        Apply augmentation to temporal sequence.

        Args:
            frames: List of 4 RGB frames (H, W, 3) as numpy arrays
            keypoints: List of (x, y) center coordinates
            labels: List of class labels (0=mover, 1=dipole)

        Returns:
            frames_tensor: (4, 3, H, W) tensor
            keypoints_aug: List of augmented (x, y) coordinates
            labels: Unchanged labels
        """
        if not self.training:
            # Validation: only resize + normalize
            frames_aug = [self._resize(f) for f in frames]
            keypoints_aug = self._resize_keypoints(keypoints, frames[0].shape[:2], self.input_size)
        else:
            # Training: apply random augmentations
            frames_aug, keypoints_aug = self._apply_augmentations(frames, keypoints)

        # Convert to tensors and normalize
        frames_tensor = []
        for frame in frames_aug:
            # Convert to torch tensor (H, W, C) -> (C, H, W)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = self.normalize(frame_tensor)
            frames_tensor.append(frame_tensor)

        frames_tensor = torch.stack(frames_tensor, dim=0)  # (4, 3, H, W)

        return frames_tensor, keypoints_aug, labels

    def _apply_augmentations(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """
        Apply random augmentations consistently across all frames.

        Args:
            frames: List of frames
            keypoints: List of (x, y) keypoints

        Returns:
            Augmented frames and keypoints
        """
        # Get original size
        orig_h, orig_w = frames[0].shape[:2]

        # 1. Random rotation (90 degree increments)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            frames = [np.rot90(f, k=k) for f in frames]
            keypoints = self._rotate_keypoints(keypoints, (orig_h, orig_w), k)

        # Update size after rotation
        orig_h, orig_w = frames[0].shape[:2]

        # 2. Random horizontal flip
        if random.random() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
            keypoints = [(orig_w - 1 - x, y) for x, y in keypoints]

        # 3. Random vertical flip
        if random.random() < 0.5:
            frames = [cv2.flip(f, 0) for f in frames]
            keypoints = [(x, orig_h - 1 - y) for x, y in keypoints]

        # 4. Random affine transform (shift, scale, rotate)
        if random.random() < 0.7:
            frames, keypoints = self._random_affine(frames, keypoints)

        # 5. Photometric augmentations
        frames = self._photometric_augmentations(frames)

        # 6. Resize to target size
        frames = [self._resize(f) for f in frames]
        keypoints = self._resize_keypoints(keypoints, (orig_h, orig_w), self.input_size)

        return frames, keypoints

    def _random_affine(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """Apply random affine transformation (shift, scale, rotate)."""
        h, w = frames[0].shape[:2]
        center = (w / 2, h / 2)

        # Random parameters
        angle = random.uniform(-15, 15)  # degrees
        scale = random.uniform(0.8, 1.2)
        tx = random.uniform(-0.1, 0.1) * w  # 10% shift
        ty = random.uniform(-0.1, 0.1) * h

        # Compute affine matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply to frames
        frames_aug = [
            cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            for f in frames
        ]

        # Apply to keypoints
        keypoints_aug = []
        for x, y in keypoints:
            pt = np.array([x, y, 1.0])
            pt_new = M @ pt
            keypoints_aug.append((pt_new[0], pt_new[1]))

        return frames_aug, keypoints_aug

    def _photometric_augmentations(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply photometric augmentations (brightness, contrast, noise, blur)."""
        frames_aug = []

        # Choose random augmentation parameters (same for all frames)
        # Brightness and contrast
        if random.random() < 0.7:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)  # brightness
        else:
            alpha, beta = 1.0, 0.0

        # Gaussian noise (reduced to avoid overwhelming signal)
        add_noise = random.random() < 0.2
        noise_sigma = random.uniform(2, 8) if add_noise else 0

        # Blur
        add_blur = random.random() < 0.3
        blur_kernel = random.choice([3, 5]) if add_blur else None

        # Apply to all frames
        for frame in frames:
            # Brightness and contrast
            frame_aug = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # Gaussian noise
            if add_noise:
                noise = np.random.normal(0, noise_sigma, frame_aug.shape).astype(np.int16)
                frame_aug = np.clip(frame_aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Blur
            if add_blur:
                frame_aug = cv2.GaussianBlur(frame_aug, (blur_kernel, blur_kernel), 0)

            frames_aug.append(frame_aug)

        return frames_aug

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size."""
        return cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)

    def _resize_keypoints(
        self,
        keypoints: List[Tuple[float, float]],
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """Resize keypoint coordinates."""
        orig_h, orig_w = orig_size
        target_h, target_w = target_size

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        return [(x * scale_x, y * scale_y) for x, y in keypoints]

    def _rotate_keypoints(
        self,
        keypoints: List[Tuple[float, float]],
        image_size: Tuple[int, int],
        k: int,
    ) -> List[Tuple[float, float]]:
        """Rotate keypoints by k*90 degrees."""
        h, w = image_size
        keypoints_rot = []

        for x, y in keypoints:
            if k == 1:  # 90 degrees CCW
                x_new, y_new = y, w - 1 - x
            elif k == 2:  # 180 degrees
                x_new, y_new = w - 1 - x, h - 1 - y
            elif k == 3:  # 270 degrees CCW
                x_new, y_new = h - 1 - y, x
            else:
                x_new, y_new = x, y

            keypoints_rot.append((x_new, y_new))

        return keypoints_rot


class CopyPasteAugmentation:
    """
    Copy-paste augmentation for object detection.

    Extracts object patches from annotated images and pastes them
    into other images at random locations to create synthetic training samples.

    Args:
        patch_size: Size of extracted patches (default: 80x80)
        max_objects_per_image: Maximum objects to paste per image
        blend_mode: How to blend patches ('replace' or 'alpha')
    """

    def __init__(
        self,
        patch_size: int = 80,
        max_objects_per_image: int = 3,
        blend_mode: str = 'replace',
    ):
        self.patch_size = patch_size
        self.max_objects_per_image = max_objects_per_image
        self.blend_mode = blend_mode

        # Storage for extracted object patches
        self.object_patches: List[dict] = []

    def add_object_patches(
        self,
        frames: List[np.ndarray],
        keypoints: List[Tuple[float, float]],
        labels: List[int],
    ):
        """
        Extract and store object patches for later pasting.

        Args:
            frames: List of 4 frames
            keypoints: List of (x, y) center coordinates
            labels: List of class labels
        """
        for (x, y), label in zip(keypoints, labels):
            # Extract patch around center from each frame
            frame_patches = []
            for frame in frames:
                patch = self._extract_patch(frame, x, y)
                if patch is not None:
                    frame_patches.append(patch)

            if len(frame_patches) == len(frames):
                self.object_patches.append({
                    'frames': frame_patches,
                    'label': label,
                })

    def generate_synthetic_sample(
        self,
        background_frames: List[np.ndarray],
        num_objects: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[int]]:
        """
        Generate synthetic sample by pasting objects onto background.

        Args:
            background_frames: List of 4 background frames
            num_objects: Number of objects to paste (random if None)

        Returns:
            Augmented frames, keypoints, labels
        """
        if not self.object_patches:
            return background_frames, [], []

        if num_objects is None:
            num_objects = random.randint(1, self.max_objects_per_image)

        h, w = background_frames[0].shape[:2]
        frames_aug = [f.copy() for f in background_frames]
        keypoints = []
        labels = []

        # Paste random objects
        for _ in range(min(num_objects, len(self.object_patches))):
            obj = random.choice(self.object_patches)

            # Random paste location (avoid edges)
            margin = self.patch_size // 2 + 10
            x = random.randint(margin, w - margin)
            y = random.randint(margin, h - margin)

            # Paste patch into each frame
            for i, patch in enumerate(obj['frames']):
                frames_aug[i] = self._paste_patch(frames_aug[i], patch, x, y)

            keypoints.append((x, y))
            labels.append(obj['label'])

        return frames_aug, keypoints, labels

    def _extract_patch(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
    ) -> Optional[np.ndarray]:
        """Extract patch around center point."""
        h, w = frame.shape[:2]
        half_size = self.patch_size // 2

        x_int, y_int = int(x), int(y)

        # Check bounds
        if (x_int - half_size < 0 or x_int + half_size >= w or
            y_int - half_size < 0 or y_int + half_size >= h):
            return None

        # Extract patch
        patch = frame[
            y_int - half_size:y_int + half_size,
            x_int - half_size:x_int + half_size
        ].copy()

        return patch

    def _paste_patch(
        self,
        frame: np.ndarray,
        patch: np.ndarray,
        x: float,
        y: float,
    ) -> np.ndarray:
        """Paste patch at location."""
        h, w = frame.shape[:2]
        half_size = self.patch_size // 2

        x_int, y_int = int(x), int(y)

        # Check bounds
        if (x_int - half_size < 0 or x_int + half_size >= w or
            y_int - half_size < 0 or y_int + half_size >= h):
            return frame

        # Paste (simple replace, could add alpha blending)
        frame[
            y_int - half_size:y_int + half_size,
            x_int - half_size:x_int + half_size
        ] = patch

        return frame


def test_augmentation():
    """Test augmentation pipeline."""
    print("Testing TemporalSequenceAugmentation...")

    # Create dummy data
    frames = [np.random.randint(0, 255, (532, 528, 3), dtype=np.uint8) for _ in range(4)]
    keypoints = [(200.0, 150.0), (300.0, 400.0)]
    labels = [0, 1]

    # Test training augmentation
    aug = TemporalSequenceAugmentation(input_size=(256, 256), training=True)
    frames_tensor, keypoints_aug, labels_out = aug(frames, keypoints, labels)

    print(f"Augmented frames shape: {frames_tensor.shape}")  # (4, 3, 256, 256)
    print(f"Augmented keypoints: {keypoints_aug}")
    print(f"Labels: {labels_out}")

    # Test copy-paste
    print("\nTesting CopyPasteAugmentation...")
    cp_aug = CopyPasteAugmentation(patch_size=80)

    # Add object patches
    cp_aug.add_object_patches(frames, keypoints, labels)
    print(f"Stored {len(cp_aug.object_patches)} object patches")

    # Generate synthetic sample
    synthetic_frames, synthetic_keypoints, synthetic_labels = cp_aug.generate_synthetic_sample(
        frames, num_objects=2
    )
    print(f"Synthetic sample: {len(synthetic_keypoints)} objects")

    print("\nTest passed!")


if __name__ == "__main__":
    test_augmentation()
