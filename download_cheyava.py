#!/usr/bin/env python3
"""
Download Cheyava Falls Mastcam-Z images from Mars Perseverance rover.
Sol 1217 - July 21, 2024
"""

import requests
from pathlib import Path
import json

def download_mastcamz_sol(sol, output_dir="data/cheyava_falls"):
    """Download Mastcam-Z images for a specific sol using NASA Mars Photos API."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # NASA Mars Photos API endpoint
    api_url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos"
    
    # Use DEMO_KEY or set your own NASA_API_KEY
    api_key = "DEMO_KEY"
    
    params = {
        "sol": sol,
        "camera": "MAST_LEFT",  # Mastcam-Z Left
        "api_key": api_key
    }
    
    print(f"Fetching Mastcam-Z images for Sol {sol}...")
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        photos = data.get("photos", [])
        
        print(f"Found {len(photos)} Mastcam-Z Left images for Sol {sol}")
        
        # Download each image
        for idx, photo in enumerate(photos[:10]):  # Limit to first 10
            img_url = photo["img_src"]
            img_name = Path(img_url).name
            img_path = output_path / f"sol_{sol:04d}_{img_name}"
            
            print(f"Downloading {img_name}...")
            img_response = requests.get(img_url)
            
            if img_response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"  Saved to {img_path}")
                
                # Save metadata
                metadata_path = img_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(photo, f, indent=2)
            else:
                print(f"  Failed to download {img_name}")
        
        return len(photos)
    else:
        print(f"API request failed: {response.status_code}")
        return 0

if __name__ == "__main__":
    # Download images from Sol 1217 (Cheyava Falls)
    print("Downloading Cheyava Falls images (Sol 1217)")
    print("=" * 60)
    download_mastcamz_sol(1217)
    
    # Also try nearby sols
    for sol in [1212, 1215, 1218]:
        print(f"\n\nDownloading Sol {sol}")
        print("=" * 60)
        download_mastcamz_sol(sol)
