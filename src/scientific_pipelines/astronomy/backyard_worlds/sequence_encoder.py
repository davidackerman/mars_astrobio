"""Encode flipbook sequences into embeddings capturing motion and appearance."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from scientific_pipelines.core.embeddings import DINOv3Extractor

logger = logging.getLogger(__name__)


class FlipbookSequenceEncoder:
    """
    Encode flipbook sequences into embeddings capturing motion + appearance.

    Strategy:
    1. Extract DINOv3 embeddings for each frame independently
    2. Compute temporal differences (frame-to-frame embedding deltas)
    3. Aggregate into single sequence embedding:
       [mean_frame_embed, std_frame_embed, mean_temporal_delta]

    This produces a (D_embed * 3)-dimensional vector that captures both
    the appearance of objects and their motion across frames.

    Args:
        embedder: DINOv3Extractor instance for frame embedding extraction
        resize_size: Size to resize frames to (default: 224x224 for DINOv3)
    """

    def __init__(
        self,
        embedder: DINOv3Extractor,
        resize_size: int = 224,
    ):
        self.embedder = embedder
        self.resize_size = resize_size
        self.embedding_dim = embedder.get_embedding_dim()

        # Sequence embedding will be 3x the base embedding dimension
        self.sequence_embedding_dim = self.embedding_dim * 3

        logger.info(
            f"FlipbookSequenceEncoder initialized: "
            f"base_dim={self.embedding_dim}, "
            f"sequence_dim={self.sequence_embedding_dim}"
        )

    def encode_sequence(self, frame_paths: List[Path]) -> np.ndarray:
        """
        Encode flipbook sequence into single embedding vector.

        Args:
            frame_paths: List of paths to flipbook frames (in temporal order)

        Returns:
            Sequence embedding of shape (D_embed * 3,) containing:
            - [0:D]: mean frame embedding (appearance)
            - [D:2D]: std frame embedding (appearance variability)
            - [2D:3D]: mean temporal delta (motion)
        """
        # Load and preprocess frames
        frames = []
        for frame_path in frame_paths:
            frame = self._load_frame(frame_path)
            frames.append(frame)

        if len(frames) == 0:
            raise ValueError("No frames provided")

        # Stack into batch tensor
        frames_tensor = torch.stack(frames)

        # Extract frame embeddings
        frame_embeddings = self.embedder.extract(frames_tensor)  # (N_frames, D)

        # Compute statistics
        mean_embed = frame_embeddings.mean(axis=0)  # (D,)
        std_embed = frame_embeddings.std(axis=0)  # (D,)

        # Compute temporal deltas (frame-to-frame changes)
        if len(frame_embeddings) > 1:
            deltas = np.diff(frame_embeddings, axis=0)  # (N_frames-1, D)
            mean_delta = deltas.mean(axis=0)  # (D,)
        else:
            # Single frame - no motion
            mean_delta = np.zeros_like(mean_embed)

        # Concatenate into sequence embedding
        sequence_embedding = np.concatenate([mean_embed, std_embed, mean_delta])

        return sequence_embedding

    def encode_batch(
        self,
        batch_frame_paths: List[List[Path]],
    ) -> np.ndarray:
        """
        Encode multiple flipbook sequences.

        Args:
            batch_frame_paths: List of frame path lists (one per subject)

        Returns:
            Array of shape (N_subjects, D_embed * 3) containing sequence embeddings
        """
        logger.info(f"Encoding {len(batch_frame_paths)} flipbook sequences")

        sequence_embeddings = []
        for frame_paths in batch_frame_paths:
            seq_embed = self.encode_sequence(frame_paths)
            sequence_embeddings.append(seq_embed)

        return np.array(sequence_embeddings)

    def _load_frame(self, path: Path) -> torch.Tensor:
        """
        Load WISE frame and preprocess for DINOv3.

        Handles both FITS and image formats (JPG, PNG).

        Args:
            path: Path to frame file

        Returns:
            Preprocessed frame tensor of shape (3, H, W)
        """
        path = Path(path)

        # Load based on file extension
        if path.suffix.lower() in ['.fits', '.fit']:
            # FITS file - astronomical image format
            try:
                from astropy.io import fits

                with fits.open(path) as hdul:
                    data = hdul[0].data

                # Handle single-band WISE data
                if data.ndim == 2:
                    # Convert to RGB by replicating channel
                    image_array = np.stack([data, data, data], axis=-1)
                else:
                    image_array = data

                # Normalize to 0-255 range
                image_array = image_array - image_array.min()
                if image_array.max() > 0:
                    image_array = 255 * image_array / image_array.max()

                image_array = image_array.astype(np.uint8)
                image = Image.fromarray(image_array)

            except ImportError:
                logger.warning(
                    "astropy not installed. Install with: pip install astropy"
                )
                # Fallback to black image
                image = Image.new('RGB', (self.resize_size, self.resize_size), 0)

        else:
            # Standard image format (JPG, PNG, etc.)
            image = Image.open(path).convert('RGB')

        # Apply DINOv3 preprocessing
        transform = DINOv3Extractor.get_default_transforms()
        frame_tensor = transform(image)

        return frame_tensor

    def get_embedding_components(
        self,
        sequence_embedding: np.ndarray,
    ) -> dict:
        """
        Split sequence embedding into its components.

        Args:
            sequence_embedding: Sequence embedding of shape (D_embed * 3,)

        Returns:
            Dictionary with:
            - 'mean_frame': Mean frame embedding (appearance)
            - 'std_frame': Std frame embedding (variability)
            - 'mean_delta': Mean temporal delta (motion)
        """
        D = self.embedding_dim

        return {
            'mean_frame': sequence_embedding[:D],
            'std_frame': sequence_embedding[D : 2 * D],
            'mean_delta': sequence_embedding[2 * D :],
        }

    def compute_motion_magnitude(
        self,
        sequence_embedding: np.ndarray,
    ) -> float:
        """
        Compute motion magnitude from sequence embedding.

        Args:
            sequence_embedding: Sequence embedding of shape (D_embed * 3,)

        Returns:
            Scalar motion magnitude (L2 norm of mean_delta)
        """
        components = self.get_embedding_components(sequence_embedding)
        motion_magnitude = np.linalg.norm(components['mean_delta'])
        return float(motion_magnitude)
