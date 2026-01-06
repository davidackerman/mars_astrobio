#!/usr/bin/env python
"""
Download full mission WATSON data with parallel processing and progress tracking.

This script downloads WATSON micro-imager images from the Mars 2020 Perseverance
mission across multiple sols in parallel for efficient data acquisition.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mars_biosig.data.downloaders.watson import WATSONDownloader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_sol_batch(
    sol_range: tuple,
    output_dir: Path,
    data_type: str = "rdr",
    overwrite: bool = False,
) -> Dict[int, List[Path]]:
    """
    Download a batch of sols (for parallel execution).

    Parameters
    ----------
    sol_range : tuple
        (start_sol, end_sol) tuple
    output_dir : Path
        Output directory for downloaded data
    data_type : str
        Data product type ('edr', 'rdr', 'fdr')
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dict
        Mapping of sol -> list of downloaded file paths
    """
    start_sol, end_sol = sol_range

    # Create downloader for this batch
    with WATSONDownloader(output_dir=output_dir) as downloader:
        results = downloader.download_sol_range(
            start_sol=start_sol,
            end_sol=end_sol,
            data_type=data_type,
            overwrite=overwrite,
        )

    return results


def split_sol_range(start_sol: int, end_sol: int, n_batches: int) -> List[tuple]:
    """
    Split sol range into batches for parallel processing.

    Parameters
    ----------
    start_sol : int
        Starting sol
    end_sol : int
        Ending sol
    n_batches : int
        Number of batches to create

    Returns
    -------
    list of tuple
        List of (batch_start, batch_end) tuples
    """
    total_sols = end_sol - start_sol + 1
    sols_per_batch = max(1, total_sols // n_batches)

    batches = []
    for i in range(n_batches):
        batch_start = start_sol + i * sols_per_batch
        batch_end = min(batch_start + sols_per_batch - 1, end_sol)

        if batch_start <= end_sol:
            batches.append((batch_start, batch_end))

    # Handle any remaining sols in the last batch
    if batches and batches[-1][1] < end_sol:
        batches[-1] = (batches[-1][0], end_sol)

    return batches


def download_tier(
    tier_name: str,
    sol_ranges: List[tuple],
    output_dir: Path,
    parallel: int = 10,
    data_type: str = "rdr",
    overwrite: bool = False,
) -> Dict:
    """
    Download a tier of data (multiple sol ranges).

    Parameters
    ----------
    tier_name : str
        Name of the tier (e.g., "Tier 1")
    sol_ranges : list of tuple
        List of (start_sol, end_sol) tuples
    output_dir : Path
        Output directory
    parallel : int
        Number of parallel workers
    data_type : str
        Data product type
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dict
        Download statistics
    """
    logger.info(f"Starting download for {tier_name}")
    logger.info(f"Sol ranges: {sol_ranges}")
    logger.info(f"Parallel workers: {parallel}")

    all_results = {}

    # Download each sol range
    for start_sol, end_sol in sol_ranges:
        logger.info(f"Downloading sols {start_sol}-{end_sol}")

        # Split into batches for parallel processing
        batches = split_sol_range(start_sol, end_sol, parallel)
        logger.info(f"Split into {len(batches)} batches")

        # Download batches in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    download_sol_batch,
                    batch,
                    output_dir,
                    data_type,
                    overwrite,
                ): batch
                for batch in batches
            }

            # Collect results with progress bar
            with tqdm(total=len(batches), desc=f"Batches {start_sol}-{end_sol}", unit="batch") as pbar:
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        results = future.result()
                        all_results.update(results)

                        # Log batch completion
                        n_images = sum(len(files) for files in results.values())
                        logger.debug(
                            f"Batch {batch[0]}-{batch[1]} complete: "
                            f"{n_images} images"
                        )
                    except Exception as e:
                        logger.error(f"Batch {batch} failed: {e}")

                    pbar.update(1)

    # Calculate statistics
    total_images = sum(len(files) for files in all_results.values())
    total_sols = len([s for s in all_results.values() if s])

    stats = {
        "tier": tier_name,
        "sol_ranges": sol_ranges,
        "total_images": total_images,
        "total_sols": total_sols,
        "sols_with_data": sorted(all_results.keys()),
    }

    logger.info(f"{tier_name} complete: {total_images} images across {total_sols} sols")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download full mission WATSON data with parallel processing"
    )
    parser.add_argument(
        "--sol-start",
        type=int,
        default=0,
        help="Starting sol number (default: 0)",
    )
    parser.add_argument(
        "--sol-end",
        type=int,
        default=1300,
        help="Ending sol number (default: 1300)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Download specific tier (1, 2, or 3). If not specified, uses sol-start/sol-end.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel download workers (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/watson_full"),
        help="Output directory (default: data/raw/watson_full)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="rdr",
        choices=["edr", "rdr", "fdr"],
        help="Data product type: edr (raw), rdr (calibrated), fdr (formatted) (default: rdr)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define tier sol ranges (from plan)
    tier_ranges = {
        1: [
            (0, 100),      # Mission start
            (500, 600),    # Wildcat Ridge region
            (1170, 1220),  # Cheyava Falls region
        ],
        2: [
            (100, 500),    # Crater floor exploration
            (600, 1100),   # Delta deposits
        ],
        3: [
            (1100, 1300),  # Recent exploration
        ],
    }

    # Determine what to download
    if args.tier:
        # Download specific tier
        tier_name = f"Tier {args.tier}"
        sol_ranges = tier_ranges[args.tier]
    else:
        # Download custom range
        tier_name = f"Custom (sols {args.sol_start}-{args.sol_end})"
        sol_ranges = [(args.sol_start, args.sol_end)]

    logger.info("=" * 60)
    logger.info("WATSON Full Mission Download")
    logger.info("=" * 60)
    logger.info(f"Download: {tier_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Parallel workers: {args.parallel}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 60)

    # Download the tier
    try:
        stats = download_tier(
            tier_name=tier_name,
            sol_ranges=sol_ranges,
            output_dir=args.output_dir,
            parallel=args.parallel,
            data_type=args.data_type,
            overwrite=args.overwrite,
        )

        # Save download summary
        summary_path = args.output_dir / f"download_summary_{tier_name.replace(' ', '_').lower()}.json"
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Total sols with data: {stats['total_sols']}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
