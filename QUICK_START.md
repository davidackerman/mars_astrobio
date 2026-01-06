# Quick Start Guide - WATSON Data Pipeline

## Installation

```bash
# Already installed! Just activate the environment
pixi shell
```

## Download WATSON Images

### Option 1: Single Sol

```python
from mars_biosig.data.downloaders.watson import WATSONDownloader

downloader = WATSONDownloader(output_dir="data/raw/watson")
files = downloader.download_sol(sol=530)  # Wildcat Ridge
print(f"Downloaded {len(files)} IMG files")
```

### Option 2: Multiple Sols

```python
# Download biosignature-rich sols
results = downloader.download_sol_range(start_sol=520, end_sol=540)
```

### Option 3: Direct PDS Access

```python
from mars_biosig.data.pds_client import PDSClient

client = PDSClient()

# List available sols
sols = client.list_watson_sols()
print(f"Available: {len(sols)} sols, range {min(sols)}-{max(sols)}")

# List images for a sol
images = client.list_watson_images(sol=530)
print(f"Sol 530 has {len(images)} images")

# Download specific image
img_path, xml_path = client.download_watson_image(
    images[0],
    output_dir="data/downloads"
)
```

## Convert IMG to PNG

### Single File

```python
from mars_biosig.data.parsers.img_converter import IMGConverter

converter = IMGConverter()
png_path = converter.convert_img_to_png(
    'data/raw/watson/sol_0530/image.IMG',
    normalize=True,
    stretch='linear'
)
```

### Batch Conversion

```python
# Convert all IMG files in a directory
converted = converter.batch_convert(
    input_dir='data/raw/watson/sol_0530',
    output_dir='data/processed/watson',
    pattern='*.IMG',
    normalize=True
)
print(f"Converted {len(converted)} files")
```

## Quick Tests

```bash
# Test PDS download (lists sols, downloads 1 image)
pixi run python test_pds_download.py

# Test IMG conversion
pixi run python test_img_converter.py

# Test full pipeline
pixi run python test_full_pipeline.py
```

## Key Biosignature Sites

```python
# Wildcat Ridge - strongest organic detection
downloader.download_sol(530)

# Cheyava Falls - leopard spot biosignature patterns
downloader.download_sol(1174)

# Recent exploration (Margin Unit)
downloader.download_sol_range(600, 655)
```

## Data Locations

After running the pipeline:

```
data/
├── raw/watson/           # Downloaded IMG + XML files
│   ├── sol_0530/
│   │   ├── SI3_*.IMG    # Binary image data
│   │   └── SI3_*.xml    # PDS4 metadata
│   └── manifest.json    # Download tracking
│
└── processed/watson/     # Converted PNG files
    ├── sol_0530/
    │   └── SI3_*.png    # PyTorch-ready images
    └── ...
```

## Next Steps

1. **Load in PyTorch:**
   ```python
   from torchvision.datasets import ImageFolder
   dataset = ImageFolder('data/processed/watson')
   ```

2. **Create Custom Dataset:**
   ```python
   from torch.utils.data import Dataset
   class WATSONDataset(Dataset):
       # Your implementation
   ```

3. **Start Training:**
   - Build CNN for texture classification
   - Label biosignature vs non-biosignature samples
   - Train and evaluate

## Documentation

- [PDS_PIPELINE_COMPLETE.md](PDS_PIPELINE_COMPLETE.md) - Full technical details
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - What was built
- [MARS_RAW_UTILS_SETUP.md](MARS_RAW_UTILS_SETUP.md) - Optional advanced calibration

## Available Data

- **314 sols** with WATSON data
- **~30,000+ images** in PDS archive
- **Sol range:** 4 - 655 (and growing with each release)
- **Quality:** EDR (highest available)
- **Format:** 384×384 to 1648×1214, 8-bit grayscale

Ready to build your biosignature detection model!
