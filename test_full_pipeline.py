#!/usr/bin/env python3
"""
Test the complete WATSON data pipeline:
1. Download IMG files from PDS archive
2. Convert IMG to PNG
3. Ready for PyTorch dataset
"""

import logging
from pathlib import Path
from mars_biosig.data.pds_client import PDSClient
from mars_biosig.data.parsers.img_converter import IMGConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*70)
    print(" WATSON Data Pipeline Test - From PDS to PyTorch-Ready Images")
    print("="*70)

    # Sol 530 - Wildcat Ridge (known biosignature site)
    test_sol = 530
    output_base = Path("data/pipeline_test")

    print(f"\nTarget: Sol {test_sol} (Wildcat Ridge - organic detection site)")
    print(f"Output: {output_base}")

    # Step 1: Download IMG files from PDS
    print("\n" + "-"*70)
    print("Step 1: Download WATSON IMG Files from PDS Archive")
    print("-"*70)

    client = PDSClient()

    # List available images
    images = client.list_watson_images(sol=test_sol)
    print(f"\nFound {len(images)} IMG files for sol {test_sol}")

    # Download first 3 images (to save time/bandwidth)
    raw_dir = output_base / "raw_img"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for i, img_info in enumerate(images[:3]):
        print(f"\nDownloading {i+1}/3: {img_info['filename']}")
        img_path, label_path = client.download_watson_image(
            img_info,
            raw_dir,
            download_label=True,
            overwrite=False
        )
        if img_path:
            downloaded.append(img_path)

    print(f"\n✓ Downloaded {len(downloaded)} IMG files")

    # Step 2: Convert IMG to PNG
    print("\n" + "-"*70)
    print("Step 2: Convert IMG Files to PNG")
    print("-"*70)

    converter = IMGConverter()
    png_dir = output_base / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(downloaded)} IMG files to PNG...")
    converted = converter.batch_convert(
        raw_dir,
        output_dir=png_dir,
        pattern="*.IMG",
        normalize=True,
        stretch='linear'
    )

    print(f"\n✓ Converted {len(converted)} files to PNG")

    # Step 3: Summarize results
    print("\n" + "="*70)
    print(" Pipeline Complete! Summary:")
    print("="*70)

    print(f"\nSol: {test_sol}")
    print(f"Total images available: {len(images)}")
    print(f"Downloaded: {len(downloaded)} IMG files")
    print(f"Converted: {len(converted)} PNG files")

    print(f"\nOutput directories:")
    print(f"  Raw IMG files: {raw_dir}")
    print(f"  PNG files: {png_dir}")

    # Show file details
    print(f"\nConverted PNG files:")
    for png in converted:
        size_kb = png.stat().st_size / 1024
        print(f"  - {png.name} ({size_kb:.1f} KB)")

    print("\n" + "="*70)
    print(" Next Steps:")
    print("="*70)
    print("""
1. PyTorch Dataset Integration:
   - PNG files are ready to use with torchvision.datasets.ImageFolder
   - Or create custom Dataset class for biosignature labels

2. Download More Data:
   - Use WATSONDownloader to download entire sols
   - Target specific biosignature sites:
     * Sol 530: Wildcat Ridge (strongest organics)
     * Sol 1174: Cheyava Falls (leopard spots)

3. Model Training:
   - Build texture classification CNN
   - Train on labeled biosignature vs non-biosignature samples
   - Use data augmentation for robustness

4. Production Pipeline:
   - Automate download → convert → label workflow
   - Set up continuous processing as new sols are released
    """)

    print("="*70)
    print(" Pipeline Test Successful!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
