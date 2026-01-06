#!/usr/bin/env python3
"""
Download Mars rover data from NASA PDS.

This script provides a command-line interface for downloading WATSON,
SHERLOC, and PIXL data from the Mars 2020 Perseverance rover.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mars_biosig.data.downloaders.watson import WATSONDownloader


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_sol_range(sol_range_str: str) -> tuple[int, int]:
    """
    Parse sol range string.

    Parameters
    ----------
    sol_range_str : str
        Sol range in format "start-end" or single sol number

    Returns
    -------
    tuple of (int, int)
        (start_sol, end_sol)

    Examples
    --------
    >>> parse_sol_range("0-100")
    (0, 100)
    >>> parse_sol_range("50")
    (50, 50)
    """
    if "-" in sol_range_str:
        start, end = sol_range_str.split("-")
        return int(start), int(end)
    else:
        sol = int(sol_range_str)
        return sol, sol


def download_watson(args):
    """Download WATSON images."""
    logger = logging.getLogger(__name__)

    start_sol, end_sol = parse_sol_range(args.sols)

    logger.info(f"Starting WATSON download: sols {start_sol}-{end_sol}")
    logger.info(f"Output directory: {args.output}")

    with WATSONDownloader(
        output_dir=args.output,
        api_key=args.api_key,
    ) as downloader:
        # Download sol range
        results = downloader.download_sol_range(
            start_sol=start_sol,
            end_sol=end_sol,
            skip_calibration=not args.include_calibration,
            overwrite=args.overwrite,
        )

        # Print statistics
        stats = downloader.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Download Statistics:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Total sols: {stats['total_sols']}")
        logger.info(f"  Sol range: {stats['sol_range']}")
        logger.info(f"  Output directory: {stats['output_dir']}")
        logger.info("=" * 60)


def download_sherloc(args):
    """Download SHERLOC data."""
    logger = logging.getLogger(__name__)
    logger.error("SHERLOC downloader not yet implemented (Phase 6)")
    sys.exit(1)


def download_pixl(args):
    """Download PIXL data."""
    logger = logging.getLogger(__name__)
    logger.error("PIXL downloader not yet implemented (Phase 6)")
    sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Mars rover data from NASA PDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download WATSON images for first 10 sols
  python download_data.py --instrument watson --sols 0-10

  # Download single sol with calibration targets
  python download_data.py --instrument watson --sols 100 --include-calibration

  # Download with custom output directory
  python download_data.py --instrument watson --sols 0-50 --output /path/to/data

  # Use specific API key
  python download_data.py --instrument watson --sols 0-10 --api-key YOUR_KEY
        """,
    )

    parser.add_argument(
        "--instrument",
        type=str,
        choices=["watson", "sherloc", "pixl"],
        required=True,
        help="Instrument to download data for",
    )

    parser.add_argument(
        "--sols",
        type=str,
        required=True,
        help='Sol range to download (format: "start-end" or single sol number)',
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/raw/{instrument})",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NASA API key (or set NASA_API_KEY environment variable)",
    )

    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Include calibration target images (default: skip)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (DEBUG level logging)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Set default output directory
    if args.output is None:
        args.output = Path("data") / "raw" / args.instrument

    # Route to appropriate downloader
    try:
        if args.instrument == "watson":
            download_watson(args)
        elif args.instrument == "sherloc":
            download_sherloc(args)
        elif args.instrument == "pixl":
            download_pixl(args)
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=args.verbose)
        sys.exit(1)

    logger.info("Download complete!")


if __name__ == "__main__":
    main()
