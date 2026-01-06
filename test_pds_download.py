#!/usr/bin/env python3
"""Test PDS WATSON IMG downloader."""

import logging
from pathlib import Path
from mars_biosig.data.pds_client import PDSClient
from mars_biosig.data.downloaders.watson import WATSONDownloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_list_sols():
    """Test listing available WATSON sols."""
    print("\n" + "="*60)
    print("Testing: List available WATSON sols")
    print("="*60)

    client = PDSClient()
    sols = client.list_watson_sols()

    print(f"\nFound {len(sols)} sols with WATSON data")
    print(f"First 10 sols: {sols[:10]}")
    print(f"Last 10 sols: {sols[-10:]}")
    print(f"Sol range: {min(sols)} - {max(sols)}")

    return sols

def test_list_images(sol=530):
    """Test listing WATSON images for a specific sol."""
    print("\n" + "="*60)
    print(f"Testing: List WATSON images for sol {sol}")
    print("="*60)

    client = PDSClient()
    images = client.list_watson_images(sol=sol)

    print(f"\nFound {len(images)} IMG files for sol {sol}")
    if images:
        print("\nFirst 3 images:")
        for img in images[:3]:
            print(f"  - {img['filename']} ({img['size']})")
            print(f"    URL: {img['url']}")

    return images

def test_download_single_image(sol=530):
    """Test downloading a single WATSON image."""
    print("\n" + "="*60)
    print(f"Testing: Download single WATSON image from sol {sol}")
    print("="*60)

    client = PDSClient()
    images = client.list_watson_images(sol=sol)

    if not images:
        print(f"No images found for sol {sol}")
        return

    # Download first image
    test_dir = Path("data/test_downloads")
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {images[0]['filename']}")
    img_path, label_path = client.download_watson_image(
        images[0],
        test_dir,
        download_label=True,
        overwrite=True
    )

    if img_path and img_path.exists():
        size_mb = img_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded IMG: {img_path} ({size_mb:.2f} MB)")

    if label_path and label_path.exists():
        size_kb = label_path.stat().st_size / 1024
        print(f"✓ Downloaded XML: {label_path} ({size_kb:.1f} KB)")

    return img_path, label_path

def test_watson_downloader(sol=530):
    """Test the full WATSON downloader for a single sol."""
    print("\n" + "="*60)
    print(f"Testing: WATSON Downloader for sol {sol}")
    print("="*60)

    output_dir = Path("data/test_watson")
    downloader = WATSONDownloader(output_dir=output_dir)

    # Download all images for the sol
    files = downloader.download_sol(sol=sol, overwrite=False)

    print(f"\n✓ Downloaded {len(files)} IMG files to {output_dir}")

    # Show stats
    stats = downloader.get_statistics()
    print(f"\nDownloader Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total sols: {stats['total_sols']}")
    print(f"  Sol range: {stats['sol_range']}")

    return files

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PDS WATSON IMG Downloader Test Suite")
    print("="*60)

    try:
        # Test 1: List available sols
        sols = test_list_sols()

        # Use sol 530 (Wildcat Ridge - known biosignature site)
        test_sol = 530

        # Test 2: List images for sol
        images = test_list_images(sol=test_sol)

        # Test 3: Download single image
        if images:
            test_download_single_image(sol=test_sol)

        # Test 4: Full downloader
        # test_watson_downloader(sol=test_sol)  # Commented out to avoid large download

        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Uncomment test_watson_downloader() to download full sol")
        print("2. Install mars-raw-utils for IMG→PNG conversion")
        print("3. Set up calibration pipeline")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
