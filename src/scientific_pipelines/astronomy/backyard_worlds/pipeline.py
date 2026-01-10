"""End-to-end Backyard Worlds brown dwarf detection pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from scientific_pipelines.core.clustering import HDBSCANClusterer
from scientific_pipelines.core.embeddings import DINOv3Extractor

from .brown_dwarf_scorer import BrownDwarfScorer
from .downloader import BackyardWorldsDownloader
from .sequence_encoder import FlipbookSequenceEncoder

logger = logging.getLogger(__name__)


class BackyardWorldsPipeline:
    """
    End-to-end pipeline for brown dwarf candidate detection.

    Pipeline steps:
    1. Download subjects from Zooniverse (or use existing)
    2. Encode flipbook sequences into embeddings (DINOv3 + temporal aggregation)
    3. Cluster subjects by behavior (HDBSCAN)
    4. Score brown dwarf candidates (multi-factor heuristics)
    5. Generate outputs (CSV files)

    Args:
        config: Configuration dictionary
        output_dir: Directory for all pipeline outputs
    """

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Backyard Worlds brown dwarf detection pipeline")

        # Downloader
        panoptes_auth = config.get('panoptes_auth', {})
        self.downloader = BackyardWorldsDownloader(
            output_dir=self.output_dir / "subjects",
            username=panoptes_auth.get('username'),
            password=panoptes_auth.get('password'),
        )

        # Embedding extractor
        embedding_config = config.get('embedding', {})
        self.embedder = DINOv3Extractor(**embedding_config)

        # Sequence encoder
        self.sequence_encoder = FlipbookSequenceEncoder(embedder=self.embedder)

        # Clustering
        clustering_config = config.get('clustering', {})
        self.clusterer = HDBSCANClusterer(**clustering_config)

        logger.info("Backyard Worlds pipeline initialized")

    def run(
        self,
        skip_download: bool = False,
        skip_encoding: bool = False,
    ):
        """
        Execute the full Backyard Worlds brown dwarf detection pipeline.

        Args:
            skip_download: If True, skip download step (assumes subjects already exist)
            skip_encoding: If True, skip encoding step (assumes embeddings exist)
        """
        logger.info("=" * 80)
        logger.info("Starting Backyard Worlds Brown Dwarf Detection Pipeline")
        logger.info("=" * 80)

        # Step 1: Download subjects
        if not skip_download:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: Downloading Backyard Worlds Subjects")
            logger.info("=" * 80)

            download_config = self.config.get('download', {})
            subjects_metadata = self.downloader.download_subjects(**download_config)

            # Save subjects metadata
            subjects_df = pd.DataFrame(subjects_metadata)
            subjects_csv_path = self.output_dir / "subjects.csv"
            subjects_df.to_csv(subjects_csv_path, index=False)
            logger.info(f"Subject metadata saved to {subjects_csv_path}")
            logger.info(f"Downloaded {len(subjects_metadata)} subjects")

        else:
            # Load existing subjects metadata
            subjects_csv_path = self.output_dir / "subjects.csv"
            logger.info(f"Skipping download, loading metadata from {subjects_csv_path}")
            subjects_df = pd.read_csv(subjects_csv_path)

            # Convert frame_paths string back to list
            import ast
            subjects_df['frame_paths'] = subjects_df['frame_paths'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            subjects_metadata = subjects_df.to_dict('records')

        logger.info(f"Processing {len(subjects_metadata)} subjects")

        # Step 2: Encode flipbook sequences
        if not skip_encoding:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Encoding Flipbook Sequences")
            logger.info("=" * 80)

            embeddings = []
            for subject in tqdm(subjects_metadata, desc="Encoding flipbooks"):
                frame_paths = [Path(p) for p in subject['frame_paths']]

                try:
                    sequence_embedding = self.sequence_encoder.encode_sequence(frame_paths)
                    embeddings.append(sequence_embedding)
                except Exception as e:
                    logger.warning(
                        f"Failed to encode subject {subject['subject_id']}: {e}"
                    )
                    # Add zero embedding as placeholder
                    embeddings.append(
                        self.sequence_encoder.sequence_embedding_dim * [0.0]
                    )

            embeddings = pd.np.array(embeddings)

            # Save embeddings
            embeddings_df = pd.DataFrame(
                {
                    'subject_id': [s['subject_id'] for s in subjects_metadata],
                    'embedding': list(embeddings),
                    'embedding_dim': embeddings.shape[1],
                }
            )

            embeddings_output = self.output_dir / "embeddings.parquet"
            embeddings_df.to_parquet(embeddings_output, index=False)
            logger.info(f"Embeddings saved to {embeddings_output}")
            logger.info(
                f"Encoded embeddings: shape={embeddings.shape}, "
                f"dim={embeddings.shape[1]}"
            )

        else:
            # Load existing embeddings
            embeddings_output = self.output_dir / "embeddings.parquet"
            logger.info(f"Skipping encoding, loading from {embeddings_output}")

            import pyarrow.parquet as pq

            table = pq.read_table(embeddings_output)
            df = table.to_pandas()
            embeddings = pd.np.vstack(df['embedding'].values)

            logger.info(f"Loaded embeddings: shape={embeddings.shape}")

        # Step 3: Cluster subjects by behavior
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Clustering Subjects by Behavior")
        logger.info("=" * 80)

        cluster_results = self.clusterer.fit_predict(embeddings)

        # Save cluster results
        clusters_df = pd.DataFrame(
            {
                'subject_id': [s['subject_id'] for s in subjects_metadata],
                'cluster_id': cluster_results['labels'],
                'cluster_probability': cluster_results['probabilities'],
                'outlier_score': cluster_results['outlier_scores'],
            }
        )

        clusters_csv_path = self.output_dir / "subject_clusters.csv"
        clusters_df.to_csv(clusters_csv_path, index=False)
        logger.info(f"Cluster results saved to {clusters_csv_path}")

        # Print cluster statistics
        logger.info(f"Clustering results:")
        logger.info(f"  Number of clusters: {cluster_results['n_clusters']}")
        logger.info(f"  Noise points: {cluster_results['noise_count']}")
        logger.info(f"  Noise fraction: {cluster_results['noise_fraction']:.1%}")

        # Interpret clusters as behavior categories
        behavior_names = [
            "Fast movers",
            "Slow movers",
            "Stationary",
            "Variable",
            "Artifacts",
        ]

        for cluster_id in sorted(
            [k for k in cluster_results['cluster_sizes'].keys() if k != -1]
        ):
            size = cluster_results['cluster_sizes'][cluster_id]
            behavior = (
                behavior_names[cluster_id]
                if cluster_id < len(behavior_names)
                else f"Cluster {cluster_id}"
            )
            logger.info(f"  {behavior}: {size} subjects")

        # Step 4: Score candidates
        scorer_type = self.config.get('scorer_type', 'brown_dwarf')
        logger.info("\n" + "=" * 80)
        if scorer_type == 'moving_object':
            logger.info("STEP 4: Scoring Moving Object Candidates")
        else:
            logger.info("STEP 4: Scoring Brown Dwarf Candidates")
        logger.info("=" * 80)

        # Create metadata dataframe for scorer
        metadata_df = pd.DataFrame(subjects_metadata)

        # Initialize scorer based on type
        if scorer_type == 'moving_object':
            from scientific_pipelines.core.clustering import NoveltyDetector

            from .moving_object_scorer import MovingObjectScorer

            # Pre-compute novelty scores for efficiency
            novelty_config = self.config.get('novelty', {})
            novelty_detector = NoveltyDetector(**novelty_config)
            novelty_detector.fit(embeddings)

            # Get moving object scoring config
            mo_config = self.config.get('moving_object_scoring', {})
            scoring_weights = mo_config.get('weights')
            optimal_motion = mo_config.get('optimal_motion_magnitude', 0.15)

            scorer = MovingObjectScorer(
                embeddings=embeddings,
                metadata=metadata_df,
                weights=scoring_weights,
                novelty_detector=novelty_detector,
                sequence_encoder=self.sequence_encoder,
                optimal_motion=optimal_motion,
            )
        else:  # 'brown_dwarf'
            scoring_weights = self.config.get('brown_dwarf_scoring', {}).get('weights')
            scorer = BrownDwarfScorer(
                embeddings=embeddings,
                metadata=metadata_df,
                weights=scoring_weights,
            )

        # Score all subjects
        ranking_df = scorer.score_subjects()

        # Save ranking
        ranking_csv_path = self.output_dir / f"{scorer_type}_ranking.csv"
        ranking_df.to_csv(ranking_csv_path, index=False)
        logger.info(f"Rankings saved to {ranking_csv_path}")

        # Print top candidates
        top_n = self.config.get('gallery', {}).get('top_n_candidates', 10)
        candidate_type = "Moving Object" if scorer_type == 'moving_object' else "Brown Dwarf"
        logger.info(f"\nTop {top_n} {candidate_type} Candidates:")

        for idx, row in ranking_df.head(top_n).iterrows():
            if scorer_type == 'moving_object':
                logger.info(
                    f"  #{idx+1}: Subject {row['subject_id']} - "
                    f"Score: {row['total_score']:.3f} "
                    f"(motion_mag: {row['motion_magnitude_score']:.2f}, "
                    f"consistency: {row['motion_consistency_score']:.2f}, "
                    f"novelty: {row['novelty_score']:.2f})"
                )
            else:
                logger.info(
                    f"  #{idx+1}: Subject {row['subject_id']} - "
                    f"Score: {row['total_score']:.3f} "
                    f"(motion: {row['motion_score']:.2f}, "
                    f"novelty: {row['novelty_score']:.2f})"
                )

        # Pipeline complete
        logger.info("\n" + "=" * 80)
        logger.info("Backyard Worlds Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info(f"  - subjects.csv: Subject metadata")
        logger.info(f"  - embeddings.parquet: Sequence embeddings")
        logger.info(f"  - subject_clusters.csv: Behavior clusters")
        logger.info(f"  - {scorer_type}_ranking.csv: Ranked candidates")

        return {
            'subjects_metadata': subjects_metadata,
            'embeddings': embeddings,
            'cluster_results': cluster_results,
            'ranking_df': ranking_df,
        }
