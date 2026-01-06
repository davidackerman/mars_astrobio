#!/usr/bin/env python3
"""Test the NASA API to see what's available."""

import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mars_biosig.utils.config import get_config

# Load API key from .env (stays local on your machine)
config = get_config()
api_key = config.nasa_api_key

print("Testing NASA Mars Photos API...")
print("(All requests stay on your local machine)")
print()

# Test 1: Check available rovers
print("1. Checking available rovers...")
url = "https://api.nasa.gov/mars-photos/api/v1/rovers"
response = requests.get(url, params={"api_key": api_key})
if response.status_code == 200:
    data = response.json()
    print(f"   Available rovers: {[r['name'] for r in data.get('rovers', [])]}")
else:
    print(f"   Error: {response.status_code}")

print()

# Test 2: Try Curiosity (known to work)
print("2. Testing with Curiosity rover...")
url = "https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos"
response = requests.get(url, params={"api_key": api_key, "sol": 1000})
if response.status_code == 200:
    data = response.json()
    print(f"   Found {len(data['photos'])} Curiosity photos on sol 1000")
    if data['photos']:
        cameras = set(p['camera']['name'] for p in data['photos'])
        print(f"   Cameras: {cameras}")
else:
    print(f"   Error: {response.status_code}")

print()

# Test 3: Check manifest for Curiosity
print("3. Checking Curiosity manifest...")
url = "https://api.nasa.gov/mars-photos/api/v1/manifests/curiosity"
response = requests.get(url, params={"api_key": api_key})
if response.status_code == 200:
    data = response.json()
    manifest = data['photo_manifest']
    print(f"   Total photos: {manifest['total_photos']:,}")
    print(f"   Max sol: {manifest['max_sol']}")
    print(f"   Cameras: {manifest['cameras']}")
else:
    print(f"   Error: {response.status_code}")

print()
print("Note: The Mars Photos API may not have Perseverance/WATSON data yet.")
print("We may need to use the direct PDS archive instead.")
