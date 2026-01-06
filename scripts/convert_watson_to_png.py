#!/usr/bin/env python3
"""
Convert all WATSON IMG files to PNG format.

This script processes all IMG files in the watson_full directory,
converting them to PNG using the IMGConverter class.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mars_biosig.data.parsers.img_converter import IMGConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "raw" / "watson_full"
    output_dir = base_dir / "data" / "processed" / "watson_png"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Find all IMG files
    img_files = sorted(input_dir.rglob("*.IMG"))
    logger.info(f"Found {len(img_files)} IMG files to convert")

    # Initialize converter
    converter = IMGConverter()

    # Track statistics
    success_count = 0
    fail_count = 0
    failed_files = []

    # Convert each file
    for i, img_file in enumerate(img_files, 1):
        try:
            # Log progress every 100 files
            if i % 100 == 0 or i == 1:
                logger.info(f"Processing {i}/{len(img_files)} ({100*i/len(img_files):.1f}%)")

            # Determine output path, maintaining directory structure
            rel_path = img_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.png')

            # Create output subdirectory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if already converted
            if output_path.exists():
                logger.debug(f"Skipping {img_file.name} (already exists)")
                success_count += 1
                continue

            # Convert
            converter.convert_img_to_png(
                img_file,
                output_path=output_path,
                normalize=True,
                stretch='linear'
            )
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to convert {img_file.name}: {e}")
            fail_count += 1
            failed_files.append(str(img_file))

    # Print summary
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files: {len(img_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Success rate: {100 * success_count / len(img_files):.1f}%")

    if failed_files:
        logger.info("\nFailed files:")
        for f in failed_files[:10]:  # Show first 10
            logger.info(f"  {f}")
        if len(failed_files) > 10:
            logger.info(f"  ... and {len(failed_files) - 10} more")

    logger.info(f"\nPNG files saved to: {output_dir}")


if __name__ == "__main__":
    main()
