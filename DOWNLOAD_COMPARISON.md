# PDS Atlas Download Methods Comparison

## Quick Note: RDR vs EDR Data

The PDS Atlas provides **RDR (Reduced Data Record)** - processed, focus-merged images. These are the recommended choice for most scientific work and ML training.

If you need raw, unprocessed **EDR (Experimental Data Record)** images instead, add the `--use-raw` flag to download from NASA's raw images server. See [docs/RDR_vs_EDR.md](docs/RDR_vs_EDR.md) for details.

## Method 1: Original Shell Script (source pdsimg-atlas-curl_*.bat)

**Pros:**
- Simple to use
- No dependencies beyond curl
- Provided directly by PDS

**Cons:**
- **Sequential downloads only** - downloads one file at a time (~296k files!)
- **No resume capability** - if interrupted, starts over from beginning
- **No progress tracking** - hard to know how much is complete
- **No organization** - all files dumped into one directory
- **Estimated time**: At ~1 second per file = 82+ hours of continuous downloading

## Method 2: Python Script (download_pds_atlas.py)

**Pros:**
- **Parallel downloads** - uses 8 workers by default (configurable)
- **Smart resume** - skips already downloaded files automatically
- **Progress bar** - real-time progress with success/fail counts
- **Better organization** - files organized by sol number (e.g., `sol_1613/`)
- **Retry logic** - automatically retries failed downloads (3 attempts)
- **Error reporting** - generates failed_downloads.log for any issues
- **Estimated time**: With 8 workers = ~10-15 hours (5-8x faster)

**Cons:**
- Requires Python with requests and tqdm packages (already included in this project)

## Recommendation

Use the Python script for much faster and more reliable downloads:

```bash
# Dependencies already installed via pixi!

# Download everything (all sols)
pixi run download-atlas

# Or download only biosignature sites (RECOMMENDED for faster start!)
pixi run download-cheyava   # Cheyava Falls site (sols 1200-1220)
pixi run download-wildcat   # Wildcat Ridge organics (sols 490-510)
pixi run download-biosig    # Both biosignature sites combined

# Or with more workers (if your connection can handle it)
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 16
```

## Usage Examples

### Download specific biosignature sites (FASTEST - recommended to start!)
```bash
# Cheyava Falls discovery site (sols 1200-1220, ~few hundred images)
pixi run download-cheyava

# Wildcat Ridge organics site (sols 490-510, ~few hundred images)
pixi run download-wildcat

# Both biosignature sites combined
pixi run download-biosig
```

### Download everything (all ~296k images)
```bash
pixi run download-atlas
```

### Custom sol filtering
```bash
# Single sol
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 8 --sols 1212

# Sol range
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 8 --sols 1200-1220

# Multiple sols
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 8 --sols 495,1212,1218

# Multiple ranges
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 8 --sols 490-510,1200-1220
```

### Increase parallelism (more workers = faster download)
```bash
pixi run python scripts/download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 16
```

### Resume interrupted download
Just run the same command again - it will skip already downloaded files and continue from where it left off.

## Biosignature Site Presets

The script includes convenient presets for known biosignature discovery sites:

### Cheyava Falls (sols 1200-1220)
- **Discovery**: July 2024 (Sol 1212-1218)
- **Significance**: Contains "leopard spots" - potential biosignature features
- **Features**: Organic molecules, calcium sulfate veins, redox reactions
- **Why important**: Most promising biosignature candidate to date
- **Reference**: [NASA Perseverance Finds Cheyava Falls](https://science.nasa.gov/resource/perseverance-finds-a-rock-with-leopard-spots/)

### Wildcat Ridge (sols 490-510)
- **Discovery**: September 2022 (Sol ~495-500)
- **Significance**: Most abundant organic molecule detection on mission
- **Features**: Aromatic organic molecules, sulfate minerals, ancient lake bed
- **Why important**: Strong evidence of habitable environment
- **Reference**: [NASA Mars Rover Investigates Organics](https://www.jpl.nasa.gov/news/nasas-perseverance-rover-investigates-geologically-rich-mars-terrain/)

## File Organization

The Python script organizes files by sol number:
```
data/raw/watson_browse/
├── sol_0490/
│   ├── SI1_0490_0xxx_xxx_xxx.png  # Wildcat Ridge area
│   └── ...
├── sol_1212/
│   ├── SI1_1212_0xxx_xxx_xxx.png  # Cheyava Falls discovery
│   └── ...
└── sol_1613/
    ├── SIF_1613_0810171983_281RZS_N0790102SRLC08062_0000LMJ01.png
    └── ...
```

This makes it much easier to:
- Find images from a specific sol
- Process images by sol
- Track download progress
- Manage disk space
- Focus on scientifically interesting targets
