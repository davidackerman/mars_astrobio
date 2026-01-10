"""Motion-based feature extraction for moving object detection.

Combines classical computer vision (frame differencing, optical flow) with
AI-based video understanding for robust motion detection.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MotionFeatureExtractor:
    """
    Extract motion features from video sequences using hybrid CV + AI approach.

    Features extracted:
    1. Classical CV features:
       - Frame differencing statistics
       - Optical flow magnitude and direction
       - Moving blob detection and tracking
       - Motion trajectory analysis

    2. AI features (optional):
       - VideoMAE embeddings for learned motion patterns
       - Temporal attention weights

    Args:
        use_optical_flow: Enable optical flow computation
        use_ai_features: Enable AI-based video understanding
        device: Device for AI model ('cuda' or 'cpu')
    """

    def __init__(
        self,
        use_optical_flow: bool = True,
        use_ai_features: bool = False,
        device: str = "cpu",
    ):
        self.use_optical_flow = use_optical_flow
        self.use_ai_features = use_ai_features
        self.device = device

        # Initialize VideoMAE if requested
        self.video_model = None
        if use_ai_features:
            try:
                from transformers import VideoMAEModel, VideoMAEImageProcessor
                logger.info("Loading VideoMAE model for motion understanding...")
                self.video_processor = VideoMAEImageProcessor.from_pretrained(
                    "MCG-NJU/videomae-base"
                )
                self.video_model = VideoMAEModel.from_pretrained(
                    "MCG-NJU/videomae-base"
                ).to(device)
                self.video_model.eval()
                logger.info("VideoMAE loaded successfully")
            except ImportError:
                logger.warning(
                    "transformers not available, AI features disabled. "
                    "Install with: pip install transformers"
                )
                self.use_ai_features = False

    def extract_features(self, frame_paths: List[Path]) -> np.ndarray:
        """
        Extract motion features from a sequence of frames.

        Args:
            frame_paths: List of paths to frame images (typically 4 frames)

        Returns:
            Feature vector combining classical CV and optional AI features
        """
        # Load frames as grayscale for CV operations
        # Resize all frames to a common size (256x256) since WISE frames vary slightly
        target_size = (256, 256)
        frames_gray = []
        frames_rgb = []

        for path in frame_paths:
            img = Image.open(path)
            # Resize to common size
            img = img.resize(target_size, Image.LANCZOS)
            img_rgb = np.array(img.convert('RGB'))
            img_gray = np.array(img.convert('L'))  # Load as grayscale directly
            frames_gray.append(img_gray)
            frames_rgb.append(img_rgb)

        # Extract classical CV features
        cv_features = self._extract_cv_features(frames_gray)

        # Extract AI features if enabled
        if self.use_ai_features and self.video_model is not None:
            ai_features = self._extract_ai_features(frames_rgb)
            features = np.concatenate([cv_features, ai_features])
        else:
            features = cv_features

        return features

    def _extract_cv_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract classical computer vision motion features."""
        features = []

        # 1. Frame differencing features
        diff_features = self._compute_frame_diff_features(frames)
        features.append(diff_features)

        # 2. Optical flow features (if enabled)
        if self.use_optical_flow:
            flow_features = self._compute_optical_flow_features(frames)
            features.append(flow_features)

        # 3. Blob detection and tracking
        blob_features = self._compute_blob_features(frames)
        features.append(blob_features)

        # 4. Temporal statistics
        temporal_features = self._compute_temporal_features(frames)
        features.append(temporal_features)

        return np.concatenate(features)

    def _compute_frame_diff_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute features from frame-to-frame differences.

        Returns:
            Feature vector (16-dim):
            - Mean/std/max of absolute differences (3 values × 3 frame pairs = 9)
            - Difference image entropy (3 values)
            - Spatial variance of differences (3 values)
            - Total motion score (1 value)
        """
        features = []

        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i+1])

            # Basic statistics
            features.extend([
                diff.mean(),
                diff.std(),
                diff.max(),
            ])

            # Entropy (measure of information/complexity)
            hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)

            # Spatial variance (how spread out motion is)
            features.append(diff.var())

        # Total motion score (sum of all differences)
        total_motion = sum([cv2.absdiff(frames[i], frames[i+1]).mean()
                           for i in range(len(frames) - 1)])
        features.append(total_motion)

        return np.array(features, dtype=np.float32)

    def _compute_optical_flow_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute dense optical flow between consecutive frames.

        Returns:
            Feature vector (15-dim):
            - Flow magnitude statistics (mean, std, max) × 3 pairs = 9
            - Flow direction histogram (4 bins: up, down, left, right) = 4
            - Flow coherence score (1 value)
            - Dominant motion direction (1 value)
        """
        features = []
        all_flow_mags = []
        all_flow_dirs = []

        for i in range(len(frames) - 1):
            # Compute dense optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Flow magnitude and direction
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Statistics
            features.extend([
                mag.mean(),
                mag.std(),
                mag.max(),
            ])

            all_flow_mags.append(mag)
            all_flow_dirs.append(ang)

        # Direction histogram (4 cardinal directions)
        all_ang = np.concatenate([a.flatten() for a in all_flow_dirs])
        all_mag = np.concatenate([m.flatten() for m in all_flow_mags])

        # Weighted by magnitude
        hist_bins = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        direction_hist, _ = np.histogram(all_ang, bins=hist_bins, weights=all_mag)
        direction_hist = direction_hist / (direction_hist.sum() + 1e-10)
        features.extend(direction_hist.tolist())

        # Flow coherence (how consistent is the motion direction)
        coherence = direction_hist.max()  # Peakedness of direction histogram
        features.append(coherence)

        # Dominant direction
        dominant_dir = direction_hist.argmax()
        features.append(dominant_dir)

        return np.array(features, dtype=np.float32)

    def _compute_blob_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Detect and track moving blobs (connected components).

        Returns:
            Feature vector (8-dim):
            - Number of blobs detected (1 value)
            - Mean/std blob size (2 values)
            - Blob displacement (mean, std, max) = 3 values
            - Blob trajectory linearity (1 value)
            - Blob persistence (fraction of frames with detected blob) = 1 value
        """
        # Compute difference images
        diff_imgs = []
        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i+1])
            # Threshold to get binary mask
            _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            diff_imgs.append(binary)

        blob_centroids = []
        blob_sizes = []

        for diff_img in diff_imgs:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                diff_img, connectivity=8
            )

            # Filter out background (label 0) and very small blobs
            valid_labels = []
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > 5:  # Minimum blob size
                    valid_labels.append(label)
                    blob_sizes.append(area)
                    blob_centroids.append(centroids[label])

        features = []

        # Number of blobs
        features.append(len(blob_sizes) if blob_sizes else 0)

        # Blob size statistics
        if blob_sizes:
            features.extend([np.mean(blob_sizes), np.std(blob_sizes)])
        else:
            features.extend([0.0, 0.0])

        # Blob displacement (track centroids across frames)
        if len(blob_centroids) >= 2:
            displacements = []
            for i in range(len(blob_centroids) - 1):
                dist = np.linalg.norm(
                    np.array(blob_centroids[i]) - np.array(blob_centroids[i+1])
                )
                displacements.append(dist)

            features.extend([
                np.mean(displacements),
                np.std(displacements),
                np.max(displacements),
            ])

            # Trajectory linearity (ratio of total displacement to path length)
            total_disp = np.linalg.norm(
                np.array(blob_centroids[-1]) - np.array(blob_centroids[0])
            )
            path_length = sum(displacements)
            linearity = total_disp / (path_length + 1e-10)
            features.append(linearity)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Blob persistence (fraction of frames with blobs)
        persistence = len([s for s in blob_sizes if s > 0]) / len(diff_imgs)
        features.append(persistence)

        return np.array(features, dtype=np.float32)

    def _compute_temporal_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute temporal statistics across the sequence.

        Returns:
            Feature vector (8-dim):
            - Mean/std intensity per frame (2 × 4 frames = 8 values)
        """
        features = []

        for frame in frames:
            features.append(frame.mean())
            features.append(frame.std())

        return np.array(features, dtype=np.float32)

    def _extract_ai_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract learned features using VideoMAE.

        Returns:
            Feature vector (768-dim VideoMAE embeddings)
        """
        if self.video_model is None:
            return np.zeros(768, dtype=np.float32)

        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(f) for f in frames]

        # Process through VideoMAE
        inputs = self.video_processor(pil_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.video_model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings.flatten().astype(np.float32)

    def get_feature_dim(self) -> int:
        """Get the dimensionality of the extracted feature vector."""
        # CV features: 16 (diff) + 15 (flow) + 8 (blob) + 8 (temporal) = 47
        cv_dim = 47
        if not self.use_optical_flow:
            cv_dim -= 15

        ai_dim = 768 if self.use_ai_features else 0

        return cv_dim + ai_dim
