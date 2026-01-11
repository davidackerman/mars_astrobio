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
        'blinker_penalty': -0.15,
        'blinker_classifier_penalty': -0.20,
    }

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        novelty_detector=None,
        sequence_encoder=None,
        optimal_motion: float = 0.15,
        encoder_type: str = "dino",
        motion_feature_config: Optional[Dict[str, bool]] = None,
        blinker_classifier_config: Optional[Dict[str, float]] = None,
    ):
        self.embeddings = embeddings
        self.metadata = metadata
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.novelty_detector = novelty_detector
        self.sequence_encoder = sequence_encoder
        self.optimal_motion = optimal_motion
        self.encoder_type = encoder_type
        self.motion_feature_config = motion_feature_config or {}
        self.blinker_classifier_config = blinker_classifier_config or {}

        # Pre-compute embedding statistics for normalized scoring
        embed_dim_base = self.embeddings.shape[1] // 3
        mean_frames = self.embeddings[:, :embed_dim_base]
        std_frames = self.embeddings[:, embed_dim_base : 2 * embed_dim_base]
        mean_deltas = self.embeddings[:, 2 * embed_dim_base :]

        self._delta_magnitudes = np.linalg.norm(mean_deltas, axis=1)
        self._std_magnitudes = np.linalg.norm(std_frames, axis=1)
        self._mean_magnitudes = np.linalg.norm(mean_frames, axis=1)
        self._mean_variances = np.var(mean_frames, axis=1)

        self._delta_norm, self._delta_low, self._delta_high = self._robust_minmax(
            self._delta_magnitudes
        )
        self._std_norm, _, _ = self._robust_minmax(self._std_magnitudes)
        self._mean_norm, _, _ = self._robust_minmax(self._mean_magnitudes)
        self._var_norm, _, _ = self._robust_minmax(self._mean_variances)

        self._optimal_motion_norm = self._normalize_value(
            self.optimal_motion, self._delta_low, self._delta_high
        )

        # Pre-compute blinker classifier scores if configured
        self._blinker_classifier_scores = None
        self._fit_blinker_classifier()

        # Pre-compute novelty scores if detector provided
        self.novelty_scores = None
        self._novelty_norm = None
        if self.novelty_detector is not None:
            logger.info("Pre-computing novelty scores for all subjects")
            self.novelty_scores = self.novelty_detector.score(self.embeddings)
            self._novelty_norm, _, _ = self._robust_minmax(self.novelty_scores)

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
            motion_magnitude_score = self._score_motion_magnitude(idx)
            motion_consistency_score = self._score_motion_consistency(idx)
            temporal_coherence_score = self._score_temporal_coherence(idx)
            novelty_score = self._score_novelty(idx)
            dipole_signature_score = self._score_dipole_signature(idx)
            artifact_penalty = self._score_artifacts(subject_meta)
            blinker_penalty = self._score_blinker_penalty(idx)
            blinker_classifier_score = self._score_blinker_classifier_penalty(idx)
            blinker_artifact_label = self._label_blinker_artifact(
                artifact_penalty, blinker_penalty, blinker_classifier_score
            )

            # Weighted total
            total_score = (
                self.weights['motion_magnitude'] * motion_magnitude_score
                + self.weights['motion_consistency'] * motion_consistency_score
                + self.weights['temporal_coherence'] * temporal_coherence_score
                + self.weights['novelty'] * novelty_score
                + self.weights['dipole_signature'] * dipole_signature_score
                + self.weights['artifact_penalty'] * artifact_penalty
                + self.weights['blinker_penalty'] * blinker_penalty
                + self.weights['blinker_classifier_penalty'] * blinker_classifier_score
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
                    'blinker_penalty': blinker_penalty,
                    'blinker_classifier_score': blinker_classifier_score,
                    'blinker_artifact_label': blinker_artifact_label,
                }
            )

        # Create dataframe and sort by score
        scores_df = pd.DataFrame(scores).sort_values('total_score', ascending=False)

        self._log_score_stats(scores_df)

        logger.info(
            f"Scoring complete. Top score: {scores_df.iloc[0]['total_score']:.3f}"
        )
        logger.info(
            f"Score statistics - Mean: {scores_df['total_score'].mean():.3f}, "
            f"Std: {scores_df['total_score'].std():.3f}"
        )

        return scores_df

    def _log_score_stats(self, scores_df: pd.DataFrame) -> None:
        """Log basic distribution stats for score components."""
        components = [
            'motion_magnitude_score',
            'motion_consistency_score',
            'temporal_coherence_score',
            'novelty_score',
            'dipole_signature_score',
            'artifact_penalty',
            'blinker_penalty',
            'blinker_classifier_score',
            'blinker_artifact_label',
            'total_score',
        ]
        logger.info("Score component stats (min/median/p95/max):")
        for name in components:
            values = scores_df[name].to_numpy(dtype=float)
            p50 = float(np.median(values))
            p95 = float(np.percentile(values, 95))
            logger.info(
                f"  {name}: min={values.min():.4f}, "
                f"p50={p50:.4f}, p95={p95:.4f}, max={values.max():.4f}"
            )

    def _score_motion_magnitude(self, idx: int) -> float:
        """
        Score based on motion magnitude.

        Moving objects should have non-zero displacement. We prefer
        moderate motion (0.10-0.25 range) using a sigmoid function.

        Args:
            idx: Index of subject in embeddings array

        Returns:
            Motion magnitude score in [0, 1]
        """
        # Score: sigmoid centered on optimal motion
        # Higher scores for motion near optimal_motion
        delta_norm = self._delta_norm[idx]
        score = 1.0 / (
            1.0 + np.exp(-10 * (delta_norm - self._optimal_motion_norm))
        )

        return self._clip01(score)

    def _score_motion_consistency(self, idx: int) -> float:
        """
        Score based on motion consistency.

        Consistent movers have stable appearance across frames (low std).
        Erratic artifacts or blinking objects have high variability.

        We use std_frame as a proxy: low std suggests the object appears
        consistently in all frames with similar appearance.

        Args:
            idx: Index of subject in embeddings array

        Returns:
            Consistency score in [0, 1]
        """
        # Score: low std -> high consistency (normalized, avoids saturation)
        consistency_score = 1.0 - self._std_norm[idx]

        return self._clip01(consistency_score)

    def _score_temporal_coherence(self, idx: int) -> float:
        """
        Score based on temporal coherence.

        Objects that appear in all 4 frames should have low std_frame.
        Intermittent detections or ghosting artifacts have high std_frame.

        This is similar to motion_consistency but emphasized separately
        to account for the physics requirement that real movers appear
        in all frames.

        Args:
            idx: Index of subject in embeddings array

        Returns:
            Coherence score in [0, 1]
        """
        # Score: low std -> high coherence (quadratic penalty)
        coherence_score = 1.0 - (self._std_norm[idx] ** 2)

        return self._clip01(coherence_score)

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
        if self._novelty_norm is not None:
            return float(self._novelty_norm[idx])

        # No novelty detector - return neutral score
        logger.debug(
            "No novelty detector provided, returning neutral novelty score"
        )
        return 0.5

    def _score_dipole_signature(self, idx: int) -> float:
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
        # Heuristic: moderate magnitude + some variance suggests structure.
        mean_norm = self._mean_norm[idx]
        var_norm = self._var_norm[idx]

        # Prefer mid-range mean magnitude (penalize extremes)
        magnitude_score = 1.0 - (abs(mean_norm - 0.5) * 2.0)
        magnitude_score = self._clip01(magnitude_score)

        # Combine: need both magnitude and structure
        dipole_score = 0.6 * magnitude_score + 0.4 * var_norm

        return self._clip01(dipole_score)

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

    def _score_blinker_penalty(self, idx: int) -> float:
        """
        Penalize likely blinkers or transient artifacts for motion encoder outputs.

        Uses blob persistence from the motion feature vector:
        low persistence -> higher penalty.
        """
        if self.encoder_type != "motion":
            return 0.0

        use_optical_flow = bool(self.motion_feature_config.get("use_optical_flow", True))
        use_ai_features = bool(self.motion_feature_config.get("use_ai_features", False))

        # Motion feature layout:
        # diff (16) + flow (15 if enabled) + blob (8) + temporal (8) + optional AI (768)
        if use_optical_flow:
            blob_persistence_idx = 16 + 15 + 7
            min_dim = 47
        else:
            blob_persistence_idx = 16 + 7
            min_dim = 32

        if use_ai_features:
            min_dim += 768

        if self.embeddings.shape[1] < min_dim:
            return 0.0

        persistence = float(self.embeddings[idx][blob_persistence_idx])
        penalty = 1.0 - np.clip(persistence, 0.0, 1.0)
        return float(penalty)

    @staticmethod
    def _label_blinker_artifact(
        artifact_penalty: float,
        blinker_penalty: float,
        blinker_classifier_score: float,
    ) -> bool:
        """Flag likely blinkers/artifacts for downstream filtering."""
        return (
            artifact_penalty < 0.0
            or blinker_penalty >= 0.5
            or blinker_classifier_score >= 0.7
        )

    def _fit_blinker_classifier(self) -> None:
        """Train a lightweight classifier to predict blinkers from features."""
        if self.encoder_type != "motion":
            return
        if not self.blinker_classifier_config.get("enabled", False):
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("scikit-learn not available; skipping blinker classifier")
            return

        use_optical_flow = bool(self.motion_feature_config.get("use_optical_flow", True))
        low_thresh = float(self.blinker_classifier_config.get("low_persistence_threshold", 0.2))
        high_thresh = float(self.blinker_classifier_config.get("high_persistence_threshold", 0.8))
        min_samples = int(self.blinker_classifier_config.get("min_samples", 50))

        # Derive persistence labels from motion features.
        if use_optical_flow:
            persistence_idx = 16 + 15 + 7
        else:
            persistence_idx = 16 + 7

        if self.embeddings.shape[1] <= persistence_idx:
            return

        persistence = self.embeddings[:, persistence_idx].astype(float)
        y = np.full(len(persistence), -1, dtype=int)
        y[persistence <= low_thresh] = 1
        y[persistence >= high_thresh] = 0

        # Use only confident pseudo-labels.
        mask = y >= 0
        if mask.sum() < min_samples:
            return

        # Ensure there are at least two classes present before training
        unique_classes = np.unique(y[mask])
        if unique_classes.size < 2:
            logger.warning(
                "Blinker classifier: insufficient label variety (classes=%s), skipping training",
                unique_classes.tolist(),
            )
            return

        X = self.embeddings[mask]
        y = y[mask]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=200,
                solver="saga",
                n_jobs=1,
            ),
        )
        model.fit(X, y)
        self._blinker_classifier_scores = model.predict_proba(self.embeddings)[:, 1]

    def _score_blinker_classifier_penalty(self, idx: int) -> float:
        if self._blinker_classifier_scores is None:
            return 0.0
        return float(np.clip(self._blinker_classifier_scores[idx], 0.0, 1.0))

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

    @staticmethod
    def _robust_minmax(
        values: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0
    ):
        """Scale values to [0, 1] using robust percentiles to avoid saturation."""
        low = float(np.percentile(values, low_pct))
        high = float(np.percentile(values, high_pct))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            scaled = np.zeros_like(values, dtype=float)
            return scaled, low, high
        scaled = (values - low) / (high - low)
        return np.clip(scaled, 0.0, 1.0), low, high

    @staticmethod
    def _normalize_value(value: float, low: float, high: float) -> float:
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return 0.5
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

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
