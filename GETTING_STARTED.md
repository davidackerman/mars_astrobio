# Getting Started - Download Your First Mars Images!

You have a NASA API key - let's use it to download some Mars rover data!

## Step 1: Set Your API Key

```bash
# Set your NASA API key as an environment variable
export NASA_API_KEY="your_actual_key_here"

# Optional: Add to your ~/.bashrc to make it permanent
echo 'export NASA_API_KEY="your_actual_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Test the Data Pipeline

Run the quick test script to download a small sample (sols 0-1):

```bash
# Activate Pixi environment
pixi shell

# Run test download
python test_download.py
```

This will download a few images from the first 2 Martian days (sols) to verify everything works.

## Step 3: Download More Data

Once the test works, download a larger sample:

```bash
# Download first 10 sols (~dozens of images)
python scripts/download_data.py --instrument watson --sols 0-10

# Or download more sols
python scripts/download_data.py --instrument watson --sols 0-50

# Or download first 100 sols (as configured in pyproject.toml)
pixi run download-watson
```

## What You'll Get

After downloading, check your data:

```bash
# See what was downloaded
ls -lh data/raw/watson/

# Check sol 1
ls -lh data/raw/watson/sol_0001/

# View the download manifest
cat data/raw/watson/manifest.json
```

Each sol directory contains:
- **JPG files** - WATSON micro-imager images
- **JSON files** - Metadata for each image

## Download Options

### Custom Sol Ranges

```bash
# Download specific sols
python scripts/download_data.py --instrument watson --sols 100-150

# Download single sol
python scripts/download_data.py --instrument watson --sols 500

# Include calibration targets (normally filtered out)
python scripts/download_data.py --instrument watson --sols 0-10 --include-calibration
```

### Verbose Output

```bash
# See detailed debug information
python scripts/download_data.py --instrument watson --sols 0-5 -v
```

### Resume Downloads

If download is interrupted, just run the same command again - it will skip already downloaded files:

```bash
# Interrupted at sol 30? Just re-run:
python scripts/download_data.py --instrument watson --sols 0-50
# It will skip sols 0-30 and continue from 31
```

## Expected Download Times

- **Sol 0-1** (test): ~30 seconds - 1 minute
- **Sol 0-10**: ~2-5 minutes
- **Sol 0-50**: ~10-20 minutes
- **Sol 0-100**: ~30-60 minutes

Times vary based on:
- Number of images per sol (varies widely)
- Your internet speed
- NASA API response time

## Troubleshooting

### "NASA_API_KEY not set"
```bash
# Make sure you exported the key in your current shell
export NASA_API_KEY="your_key"
echo $NASA_API_KEY  # Should print your key
```

### Rate Limit Errors
NASA's free tier allows 1000 requests/hour. If you hit the limit:
- Wait an hour
- Download in smaller batches
- The script will automatically retry with backoff

### Import Errors
Make sure you're in the Pixi environment:
```bash
pixi shell
python test_download.py
```

## Next Steps

Once you have data downloaded:

### 1. Explore the Data

```bash
# Start Jupyter Lab
pixi run notebook

# Create a new notebook and try:
```

```python
from pathlib import Path
from PIL import Image
import json

# Load an image
img_path = Path("data/raw/watson/sol_0001").glob("*.JPG").__next__()
img = Image.open(img_path)
img  # Display in Jupyter

# Load its metadata
metadata_path = img_path.with_suffix(".json")
with open(metadata_path) as f:
    metadata = json.load(f)
print(metadata)
```

### 2. Prepare Training Data (Phase 4)

Once you have ~100+ sols downloaded:
- Manually annotate known biosignature examples
- Create train/val/test splits
- Train your first model!

### 3. Explore Known Biosignature Sites

Download specific sols with known features:
```bash
# Wildcat Ridge (strongest organics)
python scripts/download_data.py --instrument watson --sols 528

# Cheyava Falls (leopard spots)
python scripts/download_data.py --instrument watson --sols 1174
```

## Understanding the Data

### What is WATSON?

WATSON (Wide Angle Topographic Sensor for Operations and eNgineering) is a high-resolution color camera on Perseverance's robotic arm. It takes close-up images of:
- Rock textures
- Mineral grains
- Potential biosignature features
- Surface details at microscopic scale

### Texture Classes We're Looking For

The model will learn to detect 5 texture types:

1. **Laminations** - Fine sedimentary layers
2. **Porosity** - Micro-pores and cavities
3. **Halos** - Iron oxide rings (like Cheyava Falls)
4. **Alteration** - Chemical changes
5. **Normal** - Background rock

## Data Organization

```
data/raw/watson/
├── sol_0001/
│   ├── NLF_0001_0000000001_001.JPG  # Image
│   ├── NLF_0001_0000000001_001.json # Metadata
│   ├── NLF_0001_0000000002_001.JPG
│   └── ...
├── sol_0002/
├── ...
└── manifest.json  # Download tracking
```

## Questions?

- Check [DATA_PIPELINE_README.md](DATA_PIPELINE_README.md) for Python API usage
- Check [QUICKSTART.md](QUICKSTART.md) for general setup
- Look at source code docstrings - everything is documented!

Happy Mars exploring! 🚀🔬
