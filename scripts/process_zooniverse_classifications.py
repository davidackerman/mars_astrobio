#!/usr/bin/env python3
"""
Process Zooniverse classification export data.

This script helps you work with classification CSV exports downloaded from Zooniverse.
Since the API has limited access to classification data, you'll need to:

1. Go to: https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9/data-exports
2. Sign in with your Zooniverse account
3. Download the classifications CSV export
4. Run this script to process and filter the classifications

Examples:
    # Process full classification export
    python scripts/process_zooniverse_classifications.py --input classifications_export.csv --output data/backyard_worlds/classifications.csv

    # Filter classifications for specific subjects
    python scripts/process_zooniverse_classifications.py --input classifications_export.csv --subjects data/backyard_worlds/subjects_metadata.csv --output data/backyard_worlds/classifications_filtered.csv

    # Parse and summarize annotations
    python scripts/process_zooniverse_classifications.py --input classifications_export.csv --summarize
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_annotations(annotations_str: str) -> List[Dict]:
    """Parse JSON annotation string."""
    try:
        return json.loads(annotations_str)
    except:
        return []


def extract_classification_labels(annotations: List[Dict]) -> Dict:
    """
    Extract human-readable labels from Backyard Worlds annotations.

    Backyard Worlds uses task-based annotations where volunteers mark:
    - Moving objects (potential brown dwarfs)
    - Dipoles (artifacts)
    - Other features
    """
    labels = {
        'has_moving_object': False,
        'has_dipole': False,
        'num_markings': 0,
        'tasks': [],
    }

    for annotation in annotations:
        task = annotation.get('task', '')
        labels['tasks'].append(task)

        # Check for markings (moving objects or dipoles)
        value = annotation.get('value', [])
        if isinstance(value, list):
            labels['num_markings'] += len(value)

            # Check types of markings
            for marking in value:
                if isinstance(marking, dict):
                    tool = marking.get('tool', '')
                    if 'moving' in tool.lower():
                        labels['has_moving_object'] = True
                    elif 'dipole' in tool.lower():
                        labels['has_dipole'] = True

    return labels


def summarize_classifications(df: pd.DataFrame) -> None:
    """Print summary statistics about classifications."""
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 80)

    # Basic stats
    logger.info(f"\nTotal classifications: {len(df)}")
    logger.info(f"Unique subjects: {df['subject_ids'].nunique()}")
    logger.info(f"Unique users: {df['user_id'].nunique()}")

    # Parse annotations for more detailed stats
    logger.info("\nParsing annotations...")
    df['annotations_parsed'] = df['annotations'].apply(parse_annotations)
    df['classification_labels'] = df['annotations_parsed'].apply(extract_classification_labels)

    # Extract label fields
    df['has_moving_object'] = df['classification_labels'].apply(lambda x: x['has_moving_object'])
    df['has_dipole'] = df['classification_labels'].apply(lambda x: x['has_dipole'])
    df['num_markings'] = df['classification_labels'].apply(lambda x: x['num_markings'])

    logger.info(f"\nClassifications with moving objects: {df['has_moving_object'].sum()} ({df['has_moving_object'].sum()/len(df)*100:.1f}%)")
    logger.info(f"Classifications with dipoles: {df['has_dipole'].sum()} ({df['has_dipole'].sum()/len(df)*100:.1f}%)")
    logger.info(f"Classifications with any markings: {(df['num_markings'] > 0).sum()} ({(df['num_markings'] > 0).sum()/len(df)*100:.1f}%)")
    logger.info(f"Average markings per classification: {df['num_markings'].mean():.2f}")

    # Top classified subjects
    logger.info("\nTop 10 most classified subjects:")
    top_subjects = df['subject_ids'].value_counts().head(10)
    for subject_id, count in top_subjects.items():
        logger.info(f"  Subject {subject_id}: {count} classifications")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Process Zooniverse classification export data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input CSV file (classification export from Zooniverse)',
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV file for processed classifications',
    )

    parser.add_argument(
        '--subjects',
        type=Path,
        default=None,
        help='Optional: subjects_metadata.csv to filter classifications',
    )

    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Print summary statistics',
    )

    args = parser.parse_args()

    # Check input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("\nTo get classification data:")
        logger.info("1. Go to: https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9/data-exports")
        logger.info("2. Sign in with your Zooniverse account")
        logger.info("3. Request/download the classifications export")
        logger.info("4. Run this script with the downloaded CSV file")
        sys.exit(1)

    logger.info(f"Loading classifications from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} classifications")

    # Filter by subjects if provided
    if args.subjects:
        logger.info(f"\nFiltering by subjects in: {args.subjects}")
        subjects_df = pd.read_csv(args.subjects)
        subject_ids = set(subjects_df['subject_id'].astype(str).tolist())

        # Filter classifications
        df['subject_ids'] = df['subject_ids'].astype(str)
        df_filtered = df[df['subject_ids'].isin(subject_ids)]

        logger.info(f"Filtered to {len(df_filtered)} classifications for {len(subject_ids)} subjects")
        df = df_filtered

    # Summarize if requested
    if args.summarize:
        summarize_classifications(df)

    # Save output if specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"\nSaved processed classifications to: {args.output}")


if __name__ == '__main__':
    main()
