#!/usr/bin/env python3
"""
Download Cheyava Falls images automatically using the Mars 2020 API.

This uses the official NASA RSS API via the mars2020 library.
"""

import sys
from pathlib import Path
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mars_biosig.data.mars2020_api import Mars2020


def download_sol_images(sol, cameras=["MCZ_LEFT", "MCZ_RIGHT"], output_dir="data/cheyava_falls_auto"):
    """Download all images for a specific sol."""
    
    mars = Mars2020()
    
    print(f"Fetching images for Sol {sol} with cameras: {cameras}")
    print("=" * 70)
    
    # The API doesn't have a direct sol filter, so we'll get recent images
    # and filter by sol
    images_found = []
    page = 1
    max_pages = 50  # Safety limit
    
    while page <= max_pages:
        print(f"\nQuerying page {page}...")
        images, query = mars.get_data(results=100, page=page, cameras=cameras, sort="newest")
        
        # Filter for our target sol
        sol_images = [img for img in images if img.sol == sol]
        images_found.extend(sol_images)
        
        print(f"  Found {len(sol_images)} images for Sol {sol} on this page")
        print(f"  Total so far: {len(images_found)}")
        
        # If we found images from a sol earlier than target, we can stop
        min_sol = min(img.sol for img in images) if images else sol
        if min_sol < sol:
            print(f"  Reached Sol {min_sol}, stopping search")
            break
        
        # If no more pages
        if len(images) < 100:
            print("  No more pages available")
            break
            
        page += 1
    
    if not images_found:
        print(f"\n⚠️  No images found for Sol {sol}")
        print(f"Note: The API may not have data for this sol yet.")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Found {len(images_found)} total images for Sol {sol}")
    print(f"{'=' * 70}\n")
    
    # Download images
    output_path = Path(output_dir) / f"sol_{sol:04d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, image in enumerate(images_found, 1):
        # Get filename from URL
        *_, basename = image.image_fullres.split("/")
        full_path_file = output_path / basename
        
        print(f"[{i}/{len(images_found)}] Downloading {basename}...")
        print(f"  Camera: {image.instrument}")
        print(f"  Date: {image.date_taken_utc}")
        
        try:
            image_req = requests.get(image.image_fullres, timeout=30)
            if image_req.status_code == 200:
                with open(full_path_file, 'wb') as f:
                    f.write(image_req.content)
                print(f"  ✓ Saved ({len(image_req.content):,} bytes)\n")
            else:
                print(f"  ✗ HTTP {image_req.status_code}\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print(f"{'=' * 70}")
    print(f"Downloaded {len(images_found)} images to: {output_path.absolute()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Mars 2020 images for a specific sol using the official API"
    )
    parser.add_argument("--sol", type=int, default=1217, help="Sol number (default: 1217 for Cheyava Falls)")
    parser.add_argument("--cameras", nargs="+", default=["MCZ_LEFT", "MCZ_RIGHT"], 
                       help="Camera types (default: MCZ_LEFT MCZ_RIGHT for Mastcam-Z)")
    parser.add_argument("--output", type=str, default="data/cheyava_falls_auto",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print(f"\n🔴 Mars 2020 Automatic Image Downloader")
    print(f"Using official NASA RSS API\n")
    
    download_sol_images(args.sol, args.cameras, args.output)
