#!/usr/bin/env python3
"""
Download classifications for Backyard Worlds subjects from Zooniverse.

This script downloads volunteer classifications (annotations) for subjects.
You can download classifications for specific subjects or all project classifications.

Examples:
    # Download classifications for subjects in metadata CSV
    python scripts/download_backyard_worlds_classifications.py --metadata data/backyard_worlds/subjects_metadata.csv

    # Download all classifications for the project (can be large!)
    python scripts/download_backyard_worlds_classifications.py --all --limit 10000

    # Download with authentication for higher rate limits
    python scripts/download_backyard_worlds_classifications.py --metadata data/backyard_worlds/subjects_metadata.csv --username myuser --password mypass
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_pipelines.astronomy.backyard_worlds import BackyardWorldsDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Download Backyard Worlds classifications from Zooniverse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/backyard_worlds/classifications.csv'),
        help='Output file for classifications CSV (default: data/backyard_worlds/classifications.csv)',
    )

    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Path to subjects_metadata.csv to get subject IDs from',
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all classifications for the project (not just specific subjects)',
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of classifications to download (default: unlimited)',
    )
    parser.add_argument(
        '--since',
        type=str,
        default=None,
        help='Optional ISO-8601 lower bound for created_at (inclusive)',
    )
    parser.add_argument(
        '--until',
        type=str,
        default=None,
        help='Optional ISO-8601 upper bound for created_at (inclusive)',
    )

    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='Panoptes username for authentication (optional, for higher rate limits)',
    )

    parser.add_argument(
        '--password',
        type=str,
        default=None,
        help='Panoptes password for authentication (optional)',
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.metadata:
        parser.error("Either --metadata or --all must be specified")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Backyard Worlds Classifications Download")
    logger.info("=" * 80)
    logger.info(f"Output file: {args.output}")
    if args.metadata:
        logger.info(f"Loading subject IDs from: {args.metadata}")
    if args.all:
        logger.info("Downloading all project classifications")
    if args.limit:
        logger.info(f"Download limit: {args.limit} classifications")
    if args.username:
        logger.info(f"Panoptes username: {args.username}")
    logger.info("=" * 80)

    # Load subject IDs if metadata provided
    subject_ids = None
    if args.metadata:
        logger.info(f"Loading subject IDs from {args.metadata}")
        subjects_df = pd.read_csv(args.metadata)
        subject_ids = subjects_df['subject_id'].tolist()
        logger.info(f"Found {len(subject_ids)} subjects in metadata")

    # Initialize downloader
    try:
        downloader = BackyardWorldsDownloader(
            output_dir=Path('data/backyard_worlds/temp'),  # Not used for classifications
            username=args.username,
            password=args.password,
        )

        # Download classifications
        logger.info("Starting classification download...")
        classifications = downloader.download_classifications(
            subject_ids=subject_ids,
            limit=args.limit,
            created_after=args.since,
            created_before=args.until,
        )

        # Convert to DataFrame and save
        logger.info("Processing and saving classifications...")

        # Flatten the data for CSV export
        flattened_data = []
        for classification in classifications:
            # Convert annotations to JSON string for CSV
            base_data = {
                'classification_id': classification['classification_id'],
                'workflow_id': classification['workflow_id'],
                'user_id': classification['user_id'],
                'created_at': classification['created_at'],
                'annotations': json.dumps(classification['annotations']),
                'metadata': json.dumps(classification['metadata']),
            }

            # Handle subject_ids (can be list or single value)
            if 'subject_id' in classification:
                base_data['subject_id'] = classification['subject_id']
            elif 'subject_ids' in classification:
                base_data['subject_ids'] = json.dumps(classification['subject_ids'])

            flattened_data.append(base_data)

        classifications_df = pd.DataFrame(flattened_data)
        classifications_df.to_csv(args.output, index=False)

        logger.info("=" * 80)
        logger.info("Download Complete!")
        logger.info("=" * 80)
        logger.info(f"Downloaded {len(classifications)} classifications")
        logger.info(f"Classifications saved to: {args.output}")
        logger.info("=" * 80)

        # Print summary statistics
        logger.info("\nClassifications Summary:")
        logger.info(f"  Total classifications: {len(classifications)}")

        if 'subject_id' in classifications_df.columns:
            unique_subjects = classifications_df['subject_id'].nunique()
            logger.info(f"  Unique subjects classified: {unique_subjects}")
            logger.info(
                f"  Avg classifications per subject: {len(classifications) / unique_subjects:.1f}"
            )

        if 'user_id' in classifications_df.columns:
            unique_users = classifications_df['user_id'].nunique()
            logger.info(f"  Unique volunteers: {unique_users}")

        logger.info("\nTo parse annotations, you can use:")
        logger.info("  import json")
        logger.info("  import pandas as pd")
        logger.info("  df = pd.read_csv('{}')".format(args.output))
        logger.info("  df['annotations'] = df['annotations'].apply(json.loads)")

    except Exception as e:
        logger.exception(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
