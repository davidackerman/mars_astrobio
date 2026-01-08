"""Score brown dwarf candidates using multi-factor heuristics."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BrownDwarfScorer:
    """
    Score brown dwarf candidates using multi-factor heuristics.

    Combines multiple signals to rank subjects by brown dwarf likelihood:
    1. Motion consistency (40%): Slow, consistent motion (not asteroid-like)
    2. Point-source morphology (20%): PSF-like appearance
    3. Red IR color (20%): W1-W2 color index (if available)
    4. Embedding novelty (15%): Outlier in embedding space
    5. Artifact penalties (-5%): Edge effects, known artifacts

    Args:
        embeddings: Array of sequence embeddings (N, D)
        metadata: DataFrame with subject metadata
        weights: Optional custom weights dictionary
    """

    DEFAULT_WEIGHTS = {
        'motion_consistency': 0.40,
        'point_source': 0.20,
        'red_color': 0.20,
        'novelty': 0.15,
        'artifact_penalty': -0.05,
    }

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        weights: Dict[str, float] = None,
    ):
        self.embeddings = embeddings
        self.metadata = metadata
        self.weights = weights or self.DEFAULT_WEIGHTS

        logger.info(
            f"BrownDwarfScorer initialized with {len(embeddings)} subjects"
        )
        logger.info(f"Scoring weights: {self.weights}")

    def score_subjects(self) -> pd.DataFrame:
        """
        Score all subjects.

        Returns:
            DataFrame sorted by total_score (descending) with columns:
            - subject_id
            - total_score
            - motion_score, point_source_score, color_score, novelty_score, artifact_penalty
        """
        logger.info("Scoring brown dwarf candidates...")

        scores = []

        for idx in range(len(self.embeddings)):
            embedding = self.embeddings[idx]
            subject_meta = self.metadata.iloc[idx]

            # Compute component scores
            motion_score = self._score_motion(embedding)
            point_source_score = self._score_point_source(subject_meta)
            color_score = self._score_color(subject_meta)
            novelty_score = self._score_novelty(idx)
            artifact_penalty = self._score_artifacts(subject_meta)

            # Weighted total
            total_score = (
                self.weights['motion_consistency'] * motion_score
                + self.weights['point_source'] * point_source_score
                + self.weights['red_color'] * color_score
                + self.weights['novelty'] * novelty_score
                + self.weights['artifact_penalty'] * artifact_penalty
            )

            scores.append(
                {
                    'subject_id': subject_meta['subject_id'],
                    'total_score': total_score,
                    'motion_score': motion_score,
                    'point_source_score': point_source_score,
                    'color_score': color_score,
                    'novelty_score': novelty_score,
                    'artifact_penalty': artifact_penalty,
                }
            )

        # Create dataframe and sort by score
        scores_df = pd.DataFrame(scores).sort_values('total_score', ascending=False)

        logger.info(
            f"Scoring complete. Top score: {scores_df.iloc[0]['total_score']:.3f}"
        )

        return scores_df

    def _score_motion(self, sequence_embedding: np.ndarray) -> float:
        """
        Score based on slow, consistent motion.

        Brown dwarfs have proper motion but move slowly compared to
        asteroids or image artifacts.

        Args:
            sequence_embedding: Sequence embedding (3*D,)

        Returns:
            Motion consistency score in [0, 1]
        """
        # Extract mean_delta component (motion)
        embed_dim_base = len(sequence_embedding) // 3
        mean_delta = sequence_embedding[2 * embed_dim_base :]

        # Compute motion magnitude
        delta_magnitude = np.linalg.norm(mean_delta)

        # Score: prefer slow but non-zero motion
        # Too fast -> likely asteroid or artifact
        # Too slow -> likely stationary star or noise
        optimal_motion = 0.1  # Tunable parameter
        motion_score = np.exp(-((delta_magnitude - optimal_motion) ** 2) / (2 * 0.05 ** 2))

        return float(np.clip(motion_score, 0, 1))

    def _score_point_source(self, metadata: pd.Series) -> float:
        """
        Score based on point-source morphology.

        Brown dwarfs should appear as point sources (PSF-like) rather
        than extended objects.

        Args:
            metadata: Subject metadata

        Returns:
            Point source score in [0, 1]
        """
        # Check metadata for morphology indicators
        # This is highly dataset-dependent

        # If FWHM available (Full Width Half Maximum)
        if 'fwhm' in metadata and pd.notna(metadata['fwhm']):
            fwhm = metadata['fwhm']
            # Typical PSF FWHM for WISE ~6 arcsec
            # Score higher for values near this
            score = np.exp(-((fwhm - 6.0) ** 2) / (2 * 2.0 ** 2))
            return float(np.clip(score, 0, 1))

        # If ellipticity available
        if 'ellipticity' in metadata and pd.notna(metadata['ellipticity']):
            ellipticity = metadata['ellipticity']
            # Point sources have low ellipticity
            score = 1.0 - np.clip(ellipticity, 0, 1)
            return float(score)

        # Default: neutral score if no information
        return 0.5

    def _score_color(self, metadata: pd.Series) -> float:
        """
        Score based on red IR color (W1-W2).

        Brown dwarfs are characteristically red in the IR, with
        W1-W2 color typically > 1.5 magnitudes.

        Args:
            metadata: Subject metadata

        Returns:
            Color score in [0, 1]
        """
        # Check for WISE magnitude data
        w1_key = 'w1_magnitude' if 'w1_magnitude' in metadata else 'W1'
        w2_key = 'w2_magnitude' if 'w2_magnitude' in metadata else 'W2'

        if w1_key in metadata and w2_key in metadata:
            w1 = metadata[w1_key]
            w2 = metadata[w2_key]

            if pd.notna(w1) and pd.notna(w2):
                w1_w2_color = w1 - w2

                # Brown dwarfs: W1-W2 > 1.5
                # Score linearly from 0 (at W1-W2=0) to 1 (at W1-W2=2.5)
                score = (w1_w2_color - 0.0) / 2.5
                return float(np.clip(score, 0, 1))

        # Default: neutral score if no color information
        return 0.5

    def _score_novelty(self, idx: int) -> float:
        """
        Score based on embedding novelty (kNN distance).

        Rare objects (brown dwarfs) should be outliers in embedding space.

        Args:
            idx: Index of subject in embeddings array

        Returns:
            Novelty score in [0, 1]
        """
        # Compute kNN distance
        k = min(11, len(self.embeddings))  # k=10 neighbors + self
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.embeddings)

        embedding = self.embeddings[idx : idx + 1]
        distances, _ = nbrs.kneighbors(embedding)

        # Average distance to neighbors (excluding self)
        avg_distance = distances[0, 1:].mean()

        # Normalize by 95th percentile of all distances
        all_distances = []
        for i in range(min(1000, len(self.embeddings))):  # Sample for efficiency
            emb_i = self.embeddings[i : i + 1]
            dists, _ = nbrs.kneighbors(emb_i)
            all_distances.append(dists[0, 1:].mean())

        max_distance = np.percentile(all_distances, 95)

        if max_distance > 0:
            novelty_score = avg_distance / max_distance
        else:
            novelty_score = 0.0

        return float(np.clip(novelty_score, 0, 1))

    def _score_artifacts(self, metadata: pd.Series) -> float:
        """
        Penalty for known artifacts.

        Args:
            metadata: Subject metadata

        Returns:
            Artifact penalty (negative value)
        """
        penalty = 0.0

        # Edge artifact: objects near image edge are suspicious
        if 'x_position' in metadata and 'y_position' in metadata:
            x = metadata.get('x_position', 500)
            y = metadata.get('y_position', 500)

            # Assume 1024x1024 image (adjust based on actual WISE dimensions)
            edge_threshold = 50
            if (
                x < edge_threshold
                or x > (1024 - edge_threshold)
                or y < edge_threshold
                or y > (1024 - edge_threshold)
            ):
                penalty -= 0.5

        # Known artifact flags
        if 'artifact_flag' in metadata and metadata['artifact_flag']:
            penalty -= 0.8

        # Saturated pixels
        if 'saturated' in metadata and metadata['saturated']:
            penalty -= 0.6

        return penalty

    def get_top_candidates(self, n: int = 100) -> pd.DataFrame:
        """
        Get top N brown dwarf candidates.

        Args:
            n: Number of top candidates to return

        Returns:
            DataFrame with top N candidates sorted by score
        """
        scores_df = self.score_subjects()
        return scores_df.head(n)
