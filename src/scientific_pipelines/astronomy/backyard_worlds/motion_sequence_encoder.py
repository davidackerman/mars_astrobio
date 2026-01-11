"""Motion-based sequence encoder for video analysis.

Alternative to DINOv3-based encoding that focuses on motion detection
using classical CV and optional AI features.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from .motion_features import MotionFeatureExtractor

logger = logging.getLogger(__name__)


class MotionSequenceEncoder:
    """
    Encode video sequences using motion-specific features.

    This is an alternative to FlipbookSequenceEncoder that uses motion-focused
    features (frame differencing, optical flow, blob tracking) instead of
    semantic embeddings from DINOv3.

    Args:
        use_optical_flow: Enable optical flow computation
        use_ai_features: Enable AI-based video understanding (VideoMAE)
        device: Device for AI model ('cuda' or 'cpu')
    """

    def __init__(
        self,
        use_optical_flow: bool = True,
        use_ai_features: bool = False,
        device: str = "cpu",
        ai_input: str = "rgb",
    ):
        self.feature_extractor = MotionFeatureExtractor(
            use_optical_flow=use_optical_flow,
            use_ai_features=use_ai_features,
            device=device,
            ai_input=ai_input,
        )

        self.sequence_embedding_dim = self.feature_extractor.get_feature_dim()

        logger.info(
            f"MotionSequenceEncoder initialized: "
            f"optical_flow={use_optical_flow}, "
            f"ai_features={use_ai_features}, "
            f"feature_dim={self.sequence_embedding_dim}"
        )

    def encode_sequence(self, frame_paths: List[Path]) -> np.ndarray:
        """
        Encode a sequence of frames into a motion feature vector.

        Args:
            frame_paths: List of paths to frame images (typically 4 frames)

        Returns:
            Motion feature vector
        """
        features = self.feature_extractor.extract_features(frame_paths)
        return features
