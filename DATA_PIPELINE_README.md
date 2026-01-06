# Data Pipeline Implementation

Phase 2 of the Mars Biosignature Detection project is now complete! This document describes the data pipeline implementation.

## What's Been Implemented

### Core Modules

1. **[src/mars_biosig/data/pds_client.py](src/mars_biosig/data/pds_client.py)** - NASA PDS API Client
   - Connects to NASA's Mars Photos API and PDS archive
   - Automatic retry logic with exponential backoff
   - Rate limiting to respect API limits
   - Caching support
   - Context manager for clean resource management

2. **[src/mars_biosig/data/downloaders/watson.py](src/mars_biosig/data/downloaders/watson.py)** - WATSON Downloader
   - Downloads WATSON micro-imager images
   - Filters out calibration targets automatically
   - Downloads by sol or sol ranges
   - Tracks download progress with manifest
   - Saves metadata alongside images

3. **[src/mars_biosig/data/parsers/pds4_parser.py](src/mars_biosig/data/parsers/pds4_parser.py)** - PDS4 XML Parser
   - Parses PDS4 XML labels (metadata files)
   - Extracts image metadata, timing, instrument info
   - Converts to clean JSON format

4. **[src/mars_biosig/data/dataset.py](src/mars_biosig/data/dataset.py)** - PyTorch Datasets
   - `WATSONTextureDataset` for texture classification
   - Loads images from directory structure or annotation files
   - Supports train/val/test splits
   - Computes class weights for handling imbalance
   - `MultiModalDataset` placeholder for future SHERLOC/PIXL integration

5. **[src/mars_biosig/data/transforms.py](src/mars_biosig/data/transforms.py)** - Data Augmentation
   - Mars-specific augmentation pipeline
   - Texture-preserving transformations
   - Separate train/val/test transforms
   - Support for both ImageNet and Mars-specific normalization

### Command-Line Tools

**[scripts/download_data.py](scripts/download_data.py)** - Data Download Script
- Download WATSON images by sol range
- Optionally include/exclude calibration targets
- Progress tracking
- Automatic manifest management

### Configuration Files

1. **[configs/default.yaml](configs/default.yaml)** - Default project configuration
2. **[configs/data/watson.yaml](configs/data/watson.yaml)** - WATSON-specific settings
3. **[configs/experiments/exp001_watson_baseline.yaml](configs/experiments/exp001_watson_baseline.yaml)** - Baseline experiment config

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you're in the project directory
cd mars_astrobio

# Install with Pixi
pixi install

# Activate environment
pixi shell
```

### 2. Get NASA API Key

1. Go to https://api.nasa.gov/
2. Generate your API key (free)
3. Set environment variable:

```bash
export NASA_API_KEY="your_key_here"
```

### 3. Download Data

```bash
# Download first 10 sols (small test)
python scripts/download_data.py --instrument watson --sols 0-10

# Download larger range
python scripts/download_data.py --instrument watson --sols 0-100

# Include calibration targets
python scripts/download_data.py --instrument watson --sols 0-10 --include-calibration

# Custom output directory
python scripts/download_data.py --instrument watson --sols 0-50 --output /path/to/data
```

## Usage Examples

### Python API

```python
from pathlib import Path
from mars_biosig.data.downloaders.watson import WATSONDownloader

# Download WATSON images
with WATSONDownloader(output_dir=Path("data/raw/watson")) as downloader:
    # Download single sol
    files = downloader.download_sol(sol=10)
    print(f"Downloaded {len(files)} images")

    # Download range of sols
    results = downloader.download_sol_range(start_sol=0, end_sol=50)

    # Get statistics
    stats = downloader.get_statistics()
    print(f"Total images: {stats['total_images']}")
    print(f"Sol range: {stats['sol_range']}")
```

### PyTorch Dataset

```python
from pathlib import Path
from mars_biosig.data.dataset import WATSONTextureDataset
from mars_biosig.data.transforms import MarsImageTransform

# Create train dataset
train_dataset = WATSONTextureDataset(
    root=Path("data/processed/watson_texture_v1"),
    split="train",
    transform=MarsImageTransform.get_train_transform(image_size=224),
)

# Get a sample
image, label = train_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {train_dataset.idx_to_class[label]}")

# Check class distribution
distribution = train_dataset.get_class_distribution()
print(f"Class distribution: {distribution}")

# Get class weights for handling imbalance
weights = train_dataset.get_class_weights()
print(f"Class weights: {weights}")
```

### Data Transforms

```python
from mars_biosig.data.transforms import (
    MarsImageTransform,
    TexturePreservingAugmentation,
    get_transform_config,
)

# Get train transform
train_transform = MarsImageTransform.get_train_transform(image_size=224)

# Get validation transform (no augmentation)
val_transform = MarsImageTransform.get_val_transform(image_size=224)

# Advanced texture-preserving augmentation
advanced_transform = TexturePreservingAugmentation(image_size=224)

# Or use config-based approach
train_transform = get_transform_config(mode="train", image_size=224)
val_transform = get_transform_config(mode="val", image_size=224)
```

## Data Organization

After downloading, your data directory will look like:

```
data/
├── raw/
│   └── watson/
│       ├── sol_0001/
│       │   ├── NLF_0001_0000000001_001.JPG  # Image file
│       │   ├── NLF_0001_0000000001_001.json # Metadata
│       │   └── ...
│       ├── sol_0002/
│       └── manifest.json  # Download tracking
├── processed/  # After running prepare_dataset.py (Phase 4)
│   └── watson_texture_v1/
│       ├── train/
│       │   ├── laminations/
│       │   ├── porosity/
│       │   ├── halos/
│       │   ├── alteration/
│       │   └── normal/
│       ├── val/
│       └── test/
└── annotations/  # Manual labels (to be created)
    ├── cheyava_falls_labels.json
    └── wildcat_ridge_labels.json
```

## Texture Classes

The system is designed to detect 5 texture classes:

1. **Laminations** - Fine-scale layering in sedimentary rocks
2. **Porosity** - Micro-porous textures and cavities
3. **Halos** - Iron oxide halos (like Cheyava Falls "leopard spots")
4. **Alteration** - Chemical alteration zones
5. **Normal** - Background rock textures

## Known Biosignature Sites

Training will focus on these known sites:

- **Cheyava Falls** (Sol 1174) - Leopard-spot patterns with organic signatures
- **Wildcat Ridge** (Sol 528) - Strongest organic detection in Jezero Crater
- **Delta deposits** - Sedimentary structures in ancient river delta

## Next Steps

Now that the data pipeline is complete, the next phases are:

### Phase 3: Model Architecture
- Implement base model class
- Create ResNet-50 based texture CNN
- Implement Vision Transformer variant
- Create model registry

### Phase 4: Training Pipeline
- Implement Trainer class
- Create loss functions (Focal loss for imbalance)
- Implement metrics and callbacks
- Create training script

### Phase 5: Inference
- Implement prediction pipeline
- Create anomaly detection
- Build visualization tools
- Mission-wide scanning

## Troubleshooting

### API Rate Limits
- Free tier: 1000 requests/hour
- Use caching to avoid re-downloading
- Download in smaller batches if needed

### Large Downloads
- First 100 sols can be several GB
- Use `--sols 0-10` for testing
- Data is gitignored automatically

### Missing NASA_API_KEY
If you see errors about API key:
```bash
export NASA_API_KEY="your_key_here"
# Or add to your ~/.bashrc
echo 'export NASA_API_KEY="your_key_here"' >> ~/.bashrc
```

## Testing

Test the data pipeline:

```bash
# Download a small sample
python scripts/download_data.py --instrument watson --sols 0-2

# Check the output
ls -lh data/raw/watson/sol_0001/

# Verify metadata
cat data/raw/watson/manifest.json
```

## Configuration

All settings can be customized in:
- [configs/default.yaml](configs/default.yaml) - Project-wide defaults
- [configs/data/watson.yaml](configs/data/watson.yaml) - WATSON-specific settings

Modify these files to change:
- Image sizes
- Augmentation parameters
- Quality filters
- Normalization statistics

## Documentation

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

## Support

Questions? Check:
1. The implementation plan at `/.claude/plans/elegant-questing-salamander.md`
2. Source code docstrings (all functions documented)
3. Example configurations in `configs/`

Happy biosignature hunting! 🚀🔬
