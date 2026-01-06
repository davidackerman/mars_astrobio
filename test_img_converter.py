#!/usr/bin/env python3
"""Test IMG converter."""

import logging
from pathlib import Path
from mars_biosig.data.parsers.img_converter import IMGConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_img_converter():
    """Test converting a WATSON IMG file to PNG."""
    print("\n" + "="*60)
    print("Testing IMG to PNG Conversion")
    print("="*60)

    # Find the downloaded IMG file
    test_dir = Path("data/test_downloads")
    img_files = list(test_dir.glob("*.IMG"))

    if not img_files:
        print("No IMG files found. Run test_pds_download.py first.")
        return

    img_file = img_files[0]
    print(f"\nInput: {img_file}")
    print(f"Size: {img_file.stat().st_size / 1024:.1f} KB")

    # Create converter
    converter = IMGConverter()

    # Parse label
    print("\nParsing VICAR label...")
    label = converter.read_vicar_label(img_file)
    print(f"Label parameters: {len(label)}")
    print("\nKey parameters:")
    for key in ['LINES', 'LINE_SAMPLES', 'SAMPLE_BITS', 'LBLSIZE']:
        if key in label:
            print(f"  {key}: {label[key]}")

    # Convert to array
    print("\nConverting to numpy array...")
    arr, label = converter.convert_img_to_array(img_file)
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    print(f"Value range: {arr.min()} - {arr.max()}")

    # Convert to PNG
    print("\nConverting to PNG...")
    output_dir = Path("data/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = converter.convert_img_to_png(
        img_file,
        output_path=output_dir / img_file.with_suffix('.png').name,
        normalize=True,
        stretch='linear'
    )

    print(f"\n✓ Converted to PNG: {png_path}")
    print(f"  Size: {png_path.stat().st_size / 1024:.1f} KB")

    # Test batch conversion
    print("\n" + "="*60)
    print("Testing Batch Conversion")
    print("="*60)

    converted = converter.batch_convert(
        test_dir,
        output_dir=output_dir,
        pattern="*.IMG",
        normalize=True,
        stretch='linear'
    )

    print(f"\n✓ Batch converted {len(converted)} files")
    print("\nConverted files:")
    for png in converted[:5]:  # Show first 5
        print(f"  - {png.name}")

    print("\n" + "="*60)
    print("✓ IMG Converter Test Complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nYou can now use these PNG files for:")
    print("1. PyTorch dataset loading")
    print("2. Model training")
    print("3. Visual inspection")

if __name__ == "__main__":
    try:
        test_img_converter()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
