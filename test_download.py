#!/usr/bin/env python3
"""
Quick test script to download a small sample of WATSON images.

This script automatically loads your NASA API key from the .env file.

Usage:
    1. Add your API key to .env file:
       NASA_API_KEY=your_actual_key_here

    2. Run the script:
       python test_download.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mars_biosig.utils.config import get_config
from mars_biosig.data.downloaders.watson import WATSONDownloader


def main():
    print("🚀 Mars Biosignature Detection - Download Test")
    print("=" * 60)

    # Load configuration from .env file
    try:
        config = get_config()
        api_key = config.nasa_api_key
        print(f"✅ Loaded API key from .env file")
        print(f"   API Key: {api_key[:8]}..." + "*" * (len(api_key) - 8))
    except ValueError as e:
        print(f"❌ Error: {e}")
        print()
        print("Please add your NASA API key to the .env file:")
        print("  1. Open .env in a text editor")
        print("  2. Replace the line 'NASA_API_KEY=' with:")
        print("     NASA_API_KEY=your_actual_key_here")
        print("  3. Save the file and run this script again")
        print()
        print("Get a free API key at: https://api.nasa.gov/")
        sys.exit(1)

    print()
    print("📥 Downloading WATSON images for sols 0-1 (small test)...")
    print()

    output_dir = config.data_raw_dir / "watson"

    with WATSONDownloader(output_dir=output_dir, api_key=api_key) as downloader:
        # Download sols 0-1
        results = downloader.download_sol_range(
            start_sol=0,
            end_sol=1,
            skip_calibration=True,  # Skip calibration targets
            overwrite=False,  # Don't re-download existing files
        )

        # Show results
        print()
        print("=" * 60)
        print("✅ Download Complete!")
        print()

        stats = downloader.get_statistics()
        print(f"📊 Statistics:")
        print(f"   Total images downloaded: {stats['total_images']}")
        print(f"   Total sols: {stats['total_sols']}")
        print(f"   Sol range: {stats['sol_range']}")
        print(f"   Output directory: {stats['output_dir']}")
        print()

        # Show what was downloaded for each sol
        for sol, files in results.items():
            if files:
                print(f"   Sol {sol}: {len(files)} images")
                # Show first few filenames
                for file_path in files[:3]:
                    print(f"      - {file_path.name}")
                if len(files) > 3:
                    print(f"      ... and {len(files) - 3} more")

        print()
        print("🎉 Success! Your data pipeline is working!")
        print()
        print("Next steps:")
        print("  1. Download more sols:")
        print("     python scripts/download_data.py --instrument watson --sols 0-10")
        print()
        print("  2. Explore the data in Jupyter:")
        print("     pixi run notebook")
        print()
        print("  3. Check the downloaded files:")
        print(f"     ls -lh {output_dir}/sol_0001/")


if __name__ == "__main__":
    main()
