"""CLI scripts for Backyard Worlds brown dwarf detection pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .pipeline import BackyardWorldsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def run_pipeline():
    """Main entry point for Backyard Worlds brown dwarf detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Backyard Worlds Brown Dwarf Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config
  backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml

  # Run with custom output directory
  backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml --output outputs/bw_custom

  # Skip download (use existing subjects)
  backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml --skip-download

  # Skip encoding (use existing embeddings)
  backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml --skip-encoding

  # Provide Panoptes credentials
  backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml \\
    --username myuser --password mypass
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to pipeline configuration YAML file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing subjects)",
    )

    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip encoding step (use existing embeddings)",
    )

    parser.add_argument(
        "--force-encode",
        action="store_true",
        help="Force re-encoding of embeddings even if embeddings.parquet exists",
    )

    parser.add_argument(
        "--username",
        type=str,
        help="Panoptes username for authentication",
    )

    parser.add_argument(
        "--password",
        type=str,
        help="Panoptes password for authentication",
    )

    parser.add_argument(
        "--subjects",
        type=Path,
        help="Path to existing subjects CSV (used when --skip-download)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override output directory if specified
    output_dir = args.output or Path(config.get("output_dir", "outputs/backyard_worlds"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging in addition to console
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    # Override Panoptes credentials if provided
    if args.username or args.password:
        if "panoptes_auth" not in config:
            config["panoptes_auth"] = {}
        if args.username:
            config["panoptes_auth"]["username"] = args.username
        if args.password:
            config["panoptes_auth"]["password"] = args.password

    # Create and run pipeline
    try:
        pipeline = BackyardWorldsPipeline(config, output_dir)
        # Determine subjects CSV path: CLI overrides config
        subjects_csv_arg = args.subjects or (Path(config.get('subjects_csv')) if config.get('subjects_csv') else None)

        pipeline.run(
            skip_download=args.skip_download,
            skip_encoding=args.skip_encoding,
            subjects_csv=subjects_csv_arg,
            force_encode=args.force_encode,
        )

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
