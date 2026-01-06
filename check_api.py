#!/usr/bin/env python3
"""Check what cameras and sols are available."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mars_biosig.data.pds_client import PDSClient
from mars_biosig.utils.config import get_config

config = get_config()
client = PDSClient(api_key=config.nasa_api_key)

print("Checking Perseverance rover manifest...")
print()

manifest = client.get_rover_manifest('perseverance')
photo_manifest = manifest['photo_manifest']

print(f"Mission Name: {photo_manifest['name']}")
print(f"Landing Date: {photo_manifest['landing_date']}")
print(f"Total Photos: {photo_manifest['total_photos']:,}")
print(f"Max Sol: {photo_manifest['max_sol']}")
print()

print("Available Cameras:")
for camera in photo_manifest['cameras']:
    print(f"  - {camera}")
print()

# Try to find a sol with WATSON images
print("Checking for WATSON images in recent sols...")
for sol in [100, 200, 300, 400, 500]:
    try:
        photos = client.query_rover_photos(sol=sol, camera='WATSON')
        count = len(photos.get('photos', []))
        if count > 0:
            print(f"  Sol {sol}: {count} WATSON images found!")
            break
    except Exception as e:
        print(f"  Sol {sol}: Error - {e}")
