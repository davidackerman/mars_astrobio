#!/usr/bin/env python3
"""Quick downloader for Cheyava Falls (Sol 1217)"""

import requests
import json
from pathlib import Path

sol = 1217
output_dir = Path("data/cheyava_falls_official")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"🔴 Downloading Sol {sol} Mastcam-Z images from NASA API")
print("=" * 70)

# Get Mastcam-Z Left images
for camera in ["MCZ_LEFT", "MCZ_RIGHT"]:
    print(f"\n📷 Camera: {camera}")
    url = f"https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&num=50&sol={sol}&search={camera}"
    
    try:
        response = requests.get(url, timeout=60)
        data = response.json()
        images = data.get("images", [])
        
        print(f"   Found {len(images)} images")
        
        for i, img in enumerate(images, 1):
            filename = img["image_files"]["full_res"].split("/")[-1]
            filepath = output_dir / filename
            
            if filepath.exists():
                print(f"   [{i}/{len(images)}] {filename} - exists, skipping")
                continue
            
            print(f"   [{i}/{len(images)}] {filename}...", end=" ", flush=True)
            
            img_response = requests.get(img["image_files"]["full_res"], timeout=60)
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            
            # Save metadata
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(img, f, indent=2)
            
            size_mb = len(img_response.content) / 1024 / 1024
            print(f"✓ ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print(f"✅ Images saved to: {output_dir.absolute()}")
print("=" * 70)
