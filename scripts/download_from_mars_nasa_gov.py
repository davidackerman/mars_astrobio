#!/usr/bin/env python3
"""
Download images from mars.nasa.gov/mars2020/multimedia/raw-images/

This uses the direct URL pattern discovered from the website:
https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{SOL}/ids/edr/browse/{CAMERA}/{FILENAME}
"""

import argparse
import requests
from pathlib import Path
import json
from typing import List, Dict
import re


def generate_filename_patterns(sol: int, camera: str = "zcam") -> List[str]:
    """
    Generate common filename patterns based on observed naming conventions.
    
    Format: {CAMERA_PREFIX}_{SOL}_{TIMESTAMP}_{SEQ}_{PRODUCT}_{SUFFIX}.jpg
    
    Examples:
    - ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.jpg (Left Mastcam-Z)
    - ZR0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.jpg (Right Mastcam-Z)
    """
    patterns = []
    
    # Camera prefixes
    camera_prefixes = {
        "zcam": ["ZL0", "ZR0"],  # Mastcam-Z Left/Right
        "sherloc": ["SI0", "SI1"],  # SHERLOC/WATSON
        "ncam": ["NCL", "NCR"],  # Navigation
        "fcam": ["FCL", "FCR"],  # Front hazard
        "rcam": ["RCL", "RCR"],  # Rear hazard
    }
    
    prefixes = camera_prefixes.get(camera, ["ZL0"])
    
    # Common patterns (this is a heuristic approach)
    for prefix in prefixes:
        # Pattern 1: Standard browse image
        patterns.append(f"{prefix}_{sol}_*_N*_{prefix.replace('0', '')}*_*.jpg")
    
    return patterns


def try_download_image(sol: int, camera: str, filename: str, output_dir: Path) -> bool:
    """Try to download a specific image file."""
    
    sol_padded = f"{sol:05d}"
    url = f"https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{sol_padded}/ids/edr/browse/{camera}/{filename}"
    
    output_path = output_dir / f"sol_{sol:04d}" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ {filename} ({len(response.content):,} bytes)")
            return True
        elif response.status_code == 404:
            return False
        else:
            print(f"  ✗ {filename} - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ {filename} - {e}")
        return False


def download_known_images(sol: int, output_dir: Path):
    """Download images using known filenames for specific sols."""
    
    # Known Cheyava Falls images
    known_images = {
        1217: {
            "zcam": [
                "ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png",
                "ZR0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png",
            ]
        }
    }
    
    if sol in known_images:
        print(f"Downloading known images for Sol {sol}...")
        for camera, filenames in known_images[sol].items():
            for filename in filenames:
                try_download_image(sol, camera, filename, output_dir)
    else:
        print(f"No known image filenames for Sol {sol}")
        print(f"Tip: Visit https://mars.nasa.gov/mars2020/multimedia/raw-images/")
        print(f"     Filter by Sol {sol}, then inspect image URLs to find filenames")


def download_by_url_list(url_file: Path, output_dir: Path):
    """Download images from a list of URLs in a file."""
    
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Downloading {len(urls)} images from URL list...")
    
    for url in urls:
        # Extract filename and sol from URL
        match = re.search(r'sol/(\d+)/.*?/([^/]+)$', url)
        if match:
            sol = int(match.group(1))
            filename = match.group(2)
            
            output_path = output_dir / f"sol_{sol:04d}" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"  ✓ Saved ({len(response.content):,} bytes)")
                else:
                    print(f"  ✗ HTTP {response.status_code}")
            except Exception as e:
                print(f"  ✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download images from mars.nasa.gov raw images gallery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download known Cheyava Falls images (Sol 1217)
  python download_from_mars_nasa_gov.py --sol 1217 --camera zcam
  
  # Download from a URL list file
  python download_from_mars_nasa_gov.py --url-list image_urls.txt
  
  # Download specific filename
  python download_from_mars_nasa_gov.py --sol 1217 --camera zcam --filename ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png

URL List Format (image_urls.txt):
  https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
  https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZR0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
        """,
    )
    
    parser.add_argument("--sol", type=int, help="Sol number to download")
    parser.add_argument("--camera", type=str, default="zcam", 
                       choices=["zcam", "sherloc", "ncam", "fcam", "rcam"],
                       help="Camera type")
    parser.add_argument("--filename", type=str, help="Specific filename to download")
    parser.add_argument("--url-list", type=Path, help="File containing list of URLs to download")
    parser.add_argument("--output", type=Path, default=Path("data/raw/mars_nasa_gov"),
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Mars 2020 Raw Images Downloader")
    print("Source: mars.nasa.gov/mars2020/multimedia/raw-images/")
    print("=" * 70)
    print()
    
    if args.url_list:
        download_by_url_list(args.url_list, args.output)
    elif args.sol and args.filename:
        try_download_image(args.sol, args.camera, args.filename, args.output)
    elif args.sol:
        download_known_images(args.sol, args.output)
    else:
        parser.error("Must provide --sol or --url-list")
    
    print()
    print("=" * 70)
    print(f"Downloaded to: {args.output.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
