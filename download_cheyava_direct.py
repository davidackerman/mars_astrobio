#!/usr/bin/env python3
"""
Download Cheyava Falls Mastcam-Z images directly from NASA's raw image server.
Sol 1217 (July 23, 2024) - Cheyava Falls discovery
"""

import requests
from pathlib import Path

def download_image(url, output_path):
    """Download an image from URL."""
    print(f"Downloading {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"  ✓ Saved to {output_path} ({len(response.content)} bytes)")
        return True
    else:
        print(f"  ✗ Failed: HTTP {response.status_code}")
        return False

def main():
    # Create output directory
    output_dir = Path("data/cheyava_falls")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading Cheyava Falls Images - Mars Perseverance Sol 1217")
    print("July 23, 2024 - Potential Biosignature Discovery")
    print("=" * 70)
    print()
    
    # Base URL for Mars 2020 raw images
    base_url = "https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol"
    
    # Images to download from Sol 1217 (Cheyava Falls)
    images = [
        {
            "sol": 1217,
            "camera": "ZL0",  # Left Mastcam-Z
            "filename": "ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png",
            "description": "Cheyava Falls - Left Mastcam-Z"
        },
    ]
    
    # Try to download both browse (PNG) and full resolution (IMG) versions
    for img_info in images:
        sol = img_info["sol"]
        filename = img_info["filename"]
        
        # Try browse version (PNG)
        browse_url = f"{base_url}/{sol:05d}/ids/edr/browse/zcam/{filename}"
        output_file = output_dir / f"sol_{sol}_{filename}"
        
        print(f"Image: {img_info['description']}")
        download_image(browse_url, output_file)
        
        # Try to get full resolution version (replace .png with .IMG)
        if filename.endswith('.png'):
            img_filename = filename.replace('.png', '.IMG')
            img_url = f"{base_url}/{sol:05d}/ids/edr/zcam/{img_filename}"
            img_output = output_dir / f"sol_{sol}_{img_filename}"
            
            print(f"  Attempting full resolution...")
            download_image(img_url, img_output)
        
        print()
    
    # Also try to get images from nearby sols
    print("Attempting to download additional sols (1212, 1215, 1218)...")
    print()
    
    # Search for more images (this is a guess at the naming pattern)
    for sol in [1212, 1215, 1218]:
        test_filename = f"ZL0_{sol}_*_N0560000ZCAM*_1100LMJ01.png"
        print(f"Sol {sol}: {test_filename}")
        print(f"  (Check https://mars.nasa.gov/mars2020/multimedia/raw-images/ for exact filenames)")
        print()
    
    print("=" * 70)
    print(f"Downloaded images saved to: {output_dir.absolute()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
