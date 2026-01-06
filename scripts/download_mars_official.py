#!/usr/bin/env python3
"""
Download Mars 2020 images using the official NASA RSS API.

Direct API implementation - simpler and more reliable.
"""

import requests
import json
import argparse
from pathlib import Path
from typing import List, Dict


def download_images_by_sol(sol: int, cameras: List[str] = None, output_dir: str = "data/mars_official"):
    """
    Download all images for a specific sol using NASA's official RSS API.
    
    Args:
        sol: Sol number (e.g., 1217 for Cheyava Falls)
        cameras: List of camera instruments (e.g., ['MCZ_LEFT', 'MCZ_RIGHT'])
        output_dir: Where to save images
    """
    
    if cameras is None:
        cameras = ["MCZ_LEFT", "MCZ_RIGHT"]  # Mastcam-Z by default
    
    print(f"🔴 Downloading Mars 2020 images for Sol {sol}")
    print(f"   Cameras: {', '.join(cameras)}")
    print(f"   Output: {output_dir}")
    print("=" * 70)
    
    # Build API URL
    camera_filter = "|".join(cameras)
    api_url = (
        f"https://mars.nasa.gov/rss/api/"
        f"?feed=raw_images"
        f"&category=mars2020"
        f"&feedtype=json"
        f"&num=100"
        f"&page=0"
        f"&sol={sol}"
        f"&search={camera_filter}"
    )
    
    print(f"\nQuerying NASA API...")
    print(f"URL: {api_url}\n")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Error querying API: {e}")
        return
    
    images = data.get("images", [])
    total_results = data.get("total_results", 0)
    
    print(f"Found {len(images)} images for Sol {sol}")
    print(f"Total matching results: {total_results}")
    
    if not images:
        print(f"\n⚠️  No images found for Sol {sol}")
        print(f"Note: The API might not have images for this sol yet,")
        print(f"      or they might be from different cameras.")
        return
    
    print("\n" + "=" * 70)
    
    # Create output directory
    output_path = Path(output_dir) / f"sol_{sol:04d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download each image
    for i, img in enumerate(images, 1):
        image_id = img.get("imageid", "unknown")
        instrument = img["camera"]["instrument"]
        date_taken = img.get("date_taken_utc", "unknown")
        
        # Get full resolution URL
        full_res = img["image_files"]["full_res"]
        
        # Extract filename
        filename = full_res.split("/")[-1]
        output_file = output_path / filename
        
        print(f"\n[{i}/{len(images)}] {filename}")
        print(f"    ID: {image_id}")
        print(f"    Camera: {instrument}")
        print(f"    Date: {date_taken}")
        print(f"    Downloading...", end=" ")
        
        try:
            img_response = requests.get(full_res, timeout=60)
            img_response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(img_response.content)
            
            size_mb = len(img_response.content) / 1024 / 1024
            print(f"✓ ({size_mb:.2f} MB)")
            
            # Save metadata
            metadata_file = output_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(img, f, indent=2)
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ Downloaded {len(images)} images to:")
    print(f"   {output_path.absolute()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download Mars 2020 images using NASA's official RSS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Cheyava Falls images (Sol 1217, Mastcam-Z)
  python download_mars_official.py --sol 1217
  
  # Download with specific cameras
  python download_mars_official.py --sol 1217 --cameras MCZ_LEFT MCZ_RIGHT
  
  # Download SHERLOC/WATSON images
  python download_mars_official.py --sol 1217 --cameras SHERLOC_WATSON
  
  # Download to custom directory
  python download_mars_official.py --sol 1217 --output data/my_images

Available cameras:
  MCZ_LEFT, MCZ_RIGHT     - Mastcam-Z (Left/Right)
  SHERLOC_WATSON          - WATSON micro-imager
  NAVCAM_LEFT, NAVCAM_RIGHT - Navigation cameras
  And many more...
        """
    )
    
    parser.add_argument("--sol", type=int, required=True, 
                       help="Sol number to download")
    parser.add_argument("--cameras", nargs="+", 
                       default=["MCZ_LEFT", "MCZ_RIGHT"],
                       help="Camera instruments (default: MCZ_LEFT MCZ_RIGHT)")
    parser.add_argument("--output", type=str, 
                       default="data/mars_official",
                       help="Output directory (default: data/mars_official)")
    
    args = parser.parse_args()
    
    download_images_by_sol(args.sol, args.cameras, args.output)


if __name__ == "__main__":
    main()
