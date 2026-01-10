#!/usr/bin/env python3
"""
Download Backyard Worlds flipbook data from Zooniverse.

This script downloads flipbook sequences from the Backyard Worlds: Planet 9
citizen science project for brown dwarf detection analysis.

Examples:
    # Download 1000 subjects (default)
    python scripts/download_backyard_worlds.py

    # Download 5000 subjects to custom directory
    python scripts/download_backyard_worlds.py --limit 5000 --output data/backyard_worlds

    # Download with Panoptes authentication (higher rate limits)
    python scripts/download_backyard_worlds.py --username myuser --password mypass

    # Download from specific subject set
    python scripts/download_backyard_worlds.py --subject-set-id 12345 --limit 1000
"""

import argparse
import logging
import sys
from pathlib import Path

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
        description='Download Backyard Worlds flipbook data from Zooniverse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/backyard_worlds/subjects'),
        help='Output directory for downloaded subjects (default: data/backyard_worlds/subjects)',
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Number of subjects to download (default: 1000)',
    )

    parser.add_argument(
        '--subject-set-id',
        type=int,
        default=None,
        help='Optional: Download from specific subject set ID',
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

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum download retries per file (default: 3)',
    )

    parser.add_argument(
        '--retry-delay',
        type=float,
        default=5.0,
        help='Delay between retries in seconds (default: 5.0)',
    )

    parser.add_argument(
        '--retired',
        type=str,
        choices=['true', 'false', 'all'],
        default='all',
        help='Filter by retirement status: true (examined), false (unexamined), all (default: all)',
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Backyard Worlds Data Download")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Download limit: {args.limit} subjects")
    if args.subject_set_id:
        logger.info(f"Subject set ID: {args.subject_set_id}")
    if args.username:
        logger.info(f"Panoptes username: {args.username}")
    logger.info("=" * 80)

    # Parse retired filter
    retired_filter = None
    if args.retired == 'true':
        retired_filter = True
    elif args.retired == 'false':
        retired_filter = False

    # Initialize downloader
    try:
        downloader = BackyardWorldsDownloader(
            output_dir=args.output,
            username=args.username,
            password=args.password,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        # Download subjects
        subjects_metadata = downloader.download_subjects(
            limit=args.limit,
            subject_set_id=args.subject_set_id,
            retired=retired_filter,
        )

        # Save metadata to CSV
        import pandas as pd

        subjects_df = pd.DataFrame(subjects_metadata)
        metadata_path = args.output.parent / "subjects_metadata.csv"
        subjects_df.to_csv(metadata_path, index=False)

        logger.info("=" * 80)
        logger.info("Download Complete!")
        logger.info("=" * 80)
        logger.info(f"Downloaded {len(subjects_metadata)} subjects")
        logger.info(f"Subjects saved to: {args.output}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info("=" * 80)

        # Print summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"  Total subjects: {len(subjects_metadata)}")
        logger.info(f"  Total frames: {sum(s['n_frames'] for s in subjects_metadata)}")
        logger.info(
            f"  Avg frames per subject: {sum(s['n_frames'] for s in subjects_metadata) / len(subjects_metadata):.1f}"
        )

        # Check for subjects with coordinates
        with_coords = sum(
            1 for s in subjects_metadata if s.get('ra') is not None and s.get('dec') is not None
        )
        logger.info(f"  Subjects with RA/Dec: {with_coords} ({with_coords/len(subjects_metadata)*100:.1f}%)")

    except Exception as e:
        logger.exception(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
