#!/usr/bin/env python3
"""
Test Mars Photos API without exposing API key in logs.
Your API key is only loaded from .env and used locally.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests
from mars_biosig.utils.config import get_config

# Load config (key stays in memory, not logged)
config = get_config()
key = config.nasa_api_key

print("Testing NASA Mars Photos API")
print(f"Using API key: {key[:4]}...{key[-4:]}")  # Only show first/last 4 chars
print()

# The Mars Photos API documentation says the base URL is different now
# Let's try both the old and new formats

base_urls = [
    "https://api.nasa.gov/mars-photos/api/v1",
    "https://mars-photos.api.nasa.gov/api/v1"
]

for base_url in base_urls:
    print(f"Trying {base_url}...")

    # Test with Curiosity (known rover)
    url = f"{base_url}/rovers/curiosity/photos"
    try:
        r = requests.get(url, params={"sol": 1000}, headers={"X-Api-Key": key}, timeout=10)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✅ SUCCESS! Found {len(data.get('photos', []))} photos")
            break
        elif r.status_code == 404:
            print(f"  404 - Endpoint not found")
        else:
            print(f"  Error: {r.status_code} - {r.text[:100]}")
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {str(e)[:100]}")
    print()

print("\nChecking if we need to access PDS directly instead...")
print("The Mars Photos API may not support Perseverance yet.")
print("For Perseverance/WATSON data, we likely need the PDS Geosciences Node.")
