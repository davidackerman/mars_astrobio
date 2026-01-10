"""Score moving object candidates using multi-factor heuristics."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MovingObjectScorer:
    """
    Score moving object candidates using multi-factor heuristics.

    Combines multiple signals to rank subjects by moving object likelihood:
    1. Motion magnitude (30%): Non-zero displacement detection
    2. Motion consistency (25%): Linear vs erratic motion
    3. Temporal coherence (20%): Object appears in all frames
    4. Embedding novelty (15%): Outlier in embedding space
    5. Dipole signature (10%): WISE-specific bright/dark pairs
    6. Artifact penalties (-10%): Edge effects, known artifacts

    This scorer is designed for general moving objects (asteroids, comets,
    brown dwarfs, planets) rather than specialized brown dwarf detection.

    Args:
        embeddings: Array of sequence embeddings (N, D) where D = 3 * base_dim
        metadata: DataFrame with subject metadata
        weights: Optional custom weights dictionary
        novelty_detector: Optional pre-computed NoveltyDetector instance
        sequence_encoder: Optional FlipbookSequenceEncoder for helper methods
    """

    DEFAULT_WEIGHTS = {
        'motion_magnitude': 0.30,
        'motion_consistency': 0.25,
        'temporal_coherence': 0.20,
        'novelty': 0.15,
        'dipole_signature': 0.10,
        'artifact_penalty': -0.10,
    }

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        novelty_detector=None,
        sequence_encoder=None,
        optimal_motion: float = 0.15,
    ):
        self.embeddings = embeddings
        self.metadata = metadata
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.novelty_detector = novelty_detector
        self.sequence_encoder = sequence_encoder
        self.optimal_motion = optimal_motion

        # Pre-compute novelty scores if detector provided
        self.novelty_scores = None
        if self.novelty_detector is not None:
            logger.info("Pre-computing novelty scores for all subjects")
            self.novelty_scores = self.novelty_detector.score(self.embeddings)

        logger.info(
            f"MovingObjectScorer initialized with {len(embeddings)} subjects"
        )
        logger.info(f"Scoring weights: {self.weights}")
        logger.info(f"Optimal motion magnitude: {self.optimal_motion}")

    def score_subjects(self) -> pd.DataFrame:
        """
        Score all subjects.

        Returns:
            DataFrame sorted by total_score (descending) with columns:
            - subject_id
            - total_score
            - motion_magnitude_score, motion_consistency_score,
              temporal_coherence_score, novelty_score,
              dipole_signature_score, artifact_penalty
        """
        logger.info("Scoring moving object candidates...")

        scores = []

        for idx in range(len(self.embeddings)):
            embedding = self.embeddings[idx]
            subject_meta = self.metadata.iloc[idx]

            # Compute component scores
            motion_magnitude_score = self._score_motion_magnitude(embedding)
            motion_consistency_score = self._score_motion_consistency(embedding)
            temporal_coherence_score = self._score_temporal_coherence(embedding)
            novelty_score = self._score_novelty(idx)
            dipole_signature_score = self._score_dipole_signature(embedding)
            artifact_penalty = self._score_artifacts(subject_meta)

            # Weighted total
            total_score = (
                self.weights['motion_magnitude'] * motion_magnitude_score
                + self.weights['motion_consistency'] * motion_consistency_score
                + self.weights['temporal_coherence'] * temporal_coherence_score
                + self.weights['novelty'] * novelty_score
                + self.weights['dipole_signature'] * dipole_signature_score
                + self.weights['artifact_penalty'] * artifact_penalty
            )

            scores.append(
                {
                    'subject_id': subject_meta['subject_id'],
                    'total_score': total_score,
                    'motion_magnitude_score': motion_magnitude_score,
                    'motion_consistency_score': motion_consistency_score,
                    'temporal_coherence_score': temporal_coherence_score,
                    'novelty_score': novelty_score,
                    'dipole_signature_score': dipole_signature_score,
                    'artifact_penalty': artifact_penalty,
                }
            )

        # Create dataframe and sort by score
        scores_df = pd.DataFrame(scores).sort_values('total_score', ascending=False)

        logger.info(
            f"Scoring complete. Top score: {scores_df.iloc[0]['total_score']:.3f}"
        )
        logger.info(
            f"Score statistics - Mean: {scores_df['total_score'].mean():.3f}, "
            f"Std: {scores_df['total_score'].std():.3f}"
        )

        return scores_df

    def _score_motion_magnitude(self, sequence_embedding: np.ndarray) -> float:
        """
        Score based on motion magnitude.

        Moving objects should have non-zero displacement. We prefer
        moderate motion (0.10-0.25 range) using a sigmoid function.

        Args:
            sequence_embedding: Sequence embedding (3*D,)

        Returns:
            Motion magnitude score in [0, 1]
        """
        # Extract mean_delta component (motion)
        embed_dim_base = len(sequence_embedding) // 3
        mean_delta = sequence_embedding[2 * embed_dim_base :]

        # Compute motion magnitude (L2 norm)
        delta_magnitude = np.linalg.norm(mean_delta)

        # Score: sigmoid centered on optimal motion
        # Higher scores for motion near optimal_motion
        score = 1.0 / (1.0 + np.exp(-10 * (delta_magnitude - self.optimal_motion)))

        return float(np.clip(score, 0, 1))

    def _score_motion_consistency(self, sequence_embedding: np.ndarray) -> float:
        """
        Score based on motion consistency.

        Consistent movers have stable appearance across frames (low std).
        Erratic artifacts or blinking objects have high variability.

        We use std_frame as a proxy: low std suggests the object appears
        consistently in all frames with similar appearance.

        Args:
            sequence_embedding: Sequence embedding (3*D,)

        Returns:
            Consistency score in [0, 1]
        """
        # Extract std_frame component (appearance variability)
        embed_dim_base = len(sequence_embedding) // 3
        std_frame = sequence_embedding[embed_dim_base : 2 * embed_dim_base]

        # Compute variability magnitude
        std_magnitude = np.linalg.norm(std_frame)

        # Score: low std -> high consistency
        # Use tanh to map [0, inf) -> [0, 1]
        consistency_score = 1.0 - np.tanh(std_magnitude / 10.0)

        return float(np.clip(consistency_score, 0, 1))

    def _score_temporal_coherence(self, sequence_embedding: np.ndarray) -> float:
        """
        Score based on temporal coherence.

        Objects that appear in all 4 frames should have low std_frame.
        Intermittent detections or ghosting artifacts have high std_frame.

        This is similar to motion_consistency but emphasized separately
        to account for the physics requirement that real movers appear
        in all frames.

        Args:
            sequence_embedding: Sequence embedding (3*D,)

        Returns:
            Coherence score in [0, 1]
        """
        # Extract std_frame component
        embed_dim_base = len(sequence_embedding) // 3
        std_frame = sequence_embedding[embed_dim_base : 2 * embed_dim_base]

        # Compute variability magnitude
        std_magnitude = np.linalg.norm(std_frame)

        # Score: low std -> high coherence
        # Slightly different scaling than consistency
        coherence_score = np.exp(-(std_magnitude ** 2) / (2 * 5.0 ** 2))

        return float(np.clip(coherence_score, 0, 1))

    def _score_novelty(self, idx: int) -> float:
        """
        Score based on embedding novelty.

        Rare/unusual objects (interesting movers) should be outliers
        in embedding space. Uses pre-computed novelty scores if available,
        otherwise returns neutral score.

        Args:
            idx: Index of subject in embeddings array

        Returns:
            Novelty score in [0, 1]
        """
        if self.novelty_scores is not None:
            # Use pre-computed scores
            score = self.novelty_scores[idx]
            return float(np.clip(score, 0, 1))

        # No novelty detector - return neutral score
        logger.debug(
            "No novelty detector provided, returning neutral novelty score"
        )
        return 0.5

    def _score_dipole_signature(self, sequence_embedding: np.ndarray) -> float:
        """
        Score based on dipole signature.

        WISE moving objects appear as dipoles (bright/dark pairs) due to
        the imaging process. We use embedding statistics as a proxy for
        dipole presence.

        A dipole should have:
        - Moderate mean_frame magnitude (object is visible)
        - Some spatial structure (not uniform)

        This is a heuristic-based score using embedding statistics.

        Args:
            sequence_embedding: Sequence embedding (3*D,)

        Returns:
            Dipole score in [0, 1]
        """
        # Extract mean_frame component (appearance)
        embed_dim_base = len(sequence_embedding) // 3
        mean_frame = sequence_embedding[:embed_dim_base]

        # Compute statistics
        mean_magnitude = np.linalg.norm(mean_frame)
        mean_variance = np.var(mean_frame)

        # Heuristic: moderate magnitude + some variance suggests structure
        # Score based on magnitude being in reasonable range
        magnitude_score = np.exp(-((mean_magnitude - 1.0) ** 2) / (2 * 0.5 ** 2))

        # Score based on variance (structure present)
        # Higher variance = more spatial structure
        variance_score = np.tanh(mean_variance / 0.1)

        # Combine: need both magnitude and structure
        dipole_score = 0.6 * magnitude_score + 0.4 * variance_score

        return float(np.clip(dipole_score, 0, 1))

    def _score_artifacts(self, metadata: pd.Series) -> float:
        """
        Penalty for known artifacts.

        Edge artifacts, saturation, and known flags reduce score.

        Args:
            metadata: Subject metadata

        Returns:
            Artifact penalty (negative value or zero)
        """
        penalty = 0.0

        # Edge artifact: objects near image edge are suspicious
        # WISE images are typically 1024x1024 or smaller
        if 'x_position' in metadata and 'y_position' in metadata:
            x = metadata.get('x_position', 500)
            y = metadata.get('y_position', 500)

            if pd.notna(x) and pd.notna(y):
                # Assume 1024x1024 image (adjust based on actual dimensions)
                edge_threshold = 50
                image_size = 1024

                if (
                    x < edge_threshold
                    or x > (image_size - edge_threshold)
                    or y < edge_threshold
                    or y > (image_size - edge_threshold)
                ):
                    penalty -= 0.5

        # Known artifact flags
        if 'artifact_flag' in metadata:
            artifact_flag = metadata.get('artifact_flag', False)
            if pd.notna(artifact_flag) and artifact_flag:
                penalty -= 0.8

        # Saturated pixels
        if 'saturated' in metadata:
            saturated = metadata.get('saturated', False)
            if pd.notna(saturated) and saturated:
                penalty -= 0.6

        # Diffraction spike flag (if available)
        if 'diffraction_spike' in metadata:
            spike = metadata.get('diffraction_spike', False)
            if pd.notna(spike) and spike:
                penalty -= 0.7

        return penalty

    def get_top_candidates(self, n: int = 100) -> pd.DataFrame:
        """
        Get top N moving object candidates.

        Args:
            n: Number of top candidates to return

        Returns:
            DataFrame with top N candidates sorted by score
        """
        scores_df = self.score_subjects()
        return scores_df.head(n)

    def get_motion_magnitudes(self) -> np.ndarray:
        """
        Extract motion magnitudes for all subjects.

        Useful for validation and distribution analysis.

        Returns:
            Array of motion magnitudes (L2 norms of mean_delta)
        """
        embed_dim_base = self.embeddings.shape[1] // 3
        mean_deltas = self.embeddings[:, 2 * embed_dim_base :]

        magnitudes = np.linalg.norm(mean_deltas, axis=1)
        return magnitudes

    def get_consistency_scores(self) -> np.ndarray:
        """
        Extract consistency scores for all subjects.

        Useful for validation and distribution analysis.

        Returns:
            Array of consistency scores based on std_frame magnitudes
        """
        embed_dim_base = self.embeddings.shape[1] // 3
        std_frames = self.embeddings[:, embed_dim_base : 2 * embed_dim_base]

        std_magnitudes = np.linalg.norm(std_frames, axis=1)
        consistency_scores = 1.0 - np.tanh(std_magnitudes / 10.0)

        return consistency_scores
