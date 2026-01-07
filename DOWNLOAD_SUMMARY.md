# PDS Atlas Download Guide - Quick Start

## TL;DR - Get Started Fast

```bash
# Download just the biosignature discovery sites (RECOMMENDED!)
pixi run download-cheyava   # ~few hundred images, minutes to download
pixi run download-wildcat   # ~few hundred images, minutes to download

# Or both at once
pixi run download-biosig
```

## What Are These Sites?

### Cheyava Falls
**The most promising biosignature candidate found on Mars to date**
- Discovered: July 2024 (Sol 1212-1218)
- Features: "Leopard spots" with organic molecules and redox reactions
- Images: ~few hundred from sols 1200-1220
- Download time: Minutes
- Read more: [NASA Announcement](https://science.nasa.gov/resource/perseverance-finds-a-rock-with-leopard-spots/)

### Wildcat Ridge
**Strongest organic molecule detection on the mission**
- Discovered: September 2022 (Sol 495-500)
- Features: Aromatic organic molecules in ancient lake bed
- Images: ~few hundred from sols 490-510
- Download time: Minutes
- Read more: [JPL News](https://www.jpl.nasa.gov/news/nasas-perseverance-rover-investigates-geologically-rich-mars-terrain/)

## Why Start With These Sites?

1. **Scientific relevance**: These are the most interesting targets for biosignature detection
2. **Fast download**: Get started in minutes, not hours
3. **Proof of concept**: Test your pipeline on the most promising data first
4. **Focused training**: Build initial models on high-value targets

## Available Download Commands

| Command | What It Downloads | Data Type | Image Count | Est. Time |
|---------|-------------------|-----------|-------------|-----------|
| `pixi run download-cheyava` | Cheyava Falls (sols 1200-1220) | RDR (processed) | ~few hundred | Minutes |
| `pixi run download-wildcat` | Wildcat Ridge (sols 490-510) | RDR (processed) | ~few hundred | Minutes |
| `pixi run download-biosig` | Both biosignature sites | RDR (processed) | ~1000 | Minutes |
| `pixi run download-atlas` | Full mission archive | RDR (processed) | ~296,000 | 10-15 hrs |

**Data Types:**
- **RDR (Reduced Data Record)**: Processed, focus-merged, best-quality images (default, recommended)
- **EDR (Experimental Data Record)**: Raw, unprocessed images (use `--use-raw` flag)

See [docs/RDR_vs_EDR.md](docs/RDR_vs_EDR.md) for details on the differences.

## Custom Sol Filtering

You can also specify custom sol ranges:

```bash
# Download a specific sol range (RDR processed)
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_browse \
  8 \
  --sols 1000-1100

# Download multiple specific sols
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_browse \
  8 \
  --sols 495,1212,1218

# Download multiple ranges
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_browse \
  8 \
  --sols 490-510,1200-1220

# Download raw EDR data instead of processed RDR
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_raw \
  8 \
  --sols cheyava \
  --use-raw
```

## Features

- **8x faster** than the provided shell script (parallel downloads)
- **Smart resume** - automatically skips already downloaded files
- **Progress tracking** - real-time progress bar
- **Organized output** - files organized by sol number
- **Error handling** - retries failed downloads, logs errors

## File Organization

Downloaded files are automatically organized by sol:

```
data/raw/watson_browse/
├── sol_0490/
│   ├── SI1_0490_0xxx_xxx_xxx.png
│   └── ...
├── sol_1212/
│   ├── SI1_1212_0xxx_xxx_xxx.png  # Cheyava Falls images
│   └── ...
└── sol_1220/
    └── ...
```

## Next Steps

After downloading:

1. **Explore the images** - Check out the downloaded files in your file browser
2. **Run preparation** - `pixi run prepare-data` to organize into train/val/test splits
3. **Train a model** - `pixi run train` to start training on biosignature detection
4. **Expand dataset** - Download more sols as needed with custom filters

## More Information

- See [DOWNLOAD_COMPARISON.md](DOWNLOAD_COMPARISON.md) for detailed comparison of download methods
- See [README.md](README.md) for full project documentation

## References

- [Cheyava Falls - NASA Science](https://science.nasa.gov/resource/perseverance-finds-a-rock-with-leopard-spots/)
- [Wildcat Ridge - JPL News](https://www.jpl.nasa.gov/news/nasas-perseverance-rover-investigates-geologically-rich-mars-terrain/)
- [PDS Imaging Atlas](https://pds-imaging.jpl.nasa.gov/beta/cart)
