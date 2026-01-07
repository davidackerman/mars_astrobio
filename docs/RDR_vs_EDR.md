# Understanding RDR vs EDR Image Data

## What's the Difference?

When downloading Mars rover images, you have two options for data processing levels:

### EDR - Experimental Data Record (Raw Data)
**URL Pattern**: `https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{SOL}/ids/edr/browse/shrlc/SI1_*.png`

- **Processing**: Minimal - straight from the camera with basic decompression
- **Filename prefix**: `SI1_` (SHERLOC/WATSON single focus), `SI3_` (all focuses)
- **Quality**: Raw sensor data, may have:
  - Focus blur (unfocused images)
  - Multiple exposures of same target
  - Dust on lens
  - Uncalibrated colors
- **Size**: More files (one per focus position, exposure)
- **Use case**: When you need complete raw data or want to do your own processing

### RDR - Reduced Data Record (Processed Data)
**URL Pattern**: `https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_imgops/browse/sol/{SOL}/ids/rdr/shrlc/SIF_*.png`

- **Processing**: Focus-merged, calibrated, cleaned
- **Filename prefix**: `SIF_` (SHERLOC/WATSON focus-merged)
- **Quality**: Best-focused, calibrated data:
  - Multiple focus positions merged into single best image
  - Calibrated radiometry
  - Cleaned and validated
- **Size**: Fewer files (merged products)
- **Use case**: Most scientific analysis - you get the best image for each target

## Which Should You Use?

### Use RDR (Default) If:
✅ You want the cleanest, best-focused images
✅ You're doing scientific analysis or ML training
✅ You want fewer files to manage
✅ You trust NASA's processing pipeline
✅ **This is the recommended choice for most users**

### Use EDR (--use-raw) If:
✅ You need complete raw data
✅ You want to implement custom focus-merging algorithms
✅ You're studying instrument artifacts or calibration
✅ You need every single exposure for temporal analysis
✅ You're doing research on image processing methods

## Example Comparison

### Same target, different processing levels:

**EDR (raw)**:
```
SI1_1613_0810132725_121ECM_N0790102SRLC00472_0000LMJ01.png  (focus position 1)
SI1_1613_0810132725_121ECM_N0790102SRLC00472_0000LMJ02.png  (focus position 2)
SI1_1613_0810132725_121ECM_N0790102SRLC00472_0000LMJ03.png  (focus position 3)
...
```
→ Multiple images, some out of focus

**RDR (processed)**:
```
SIF_1613_0810132725_121ECM_N0790102SRLC00472_0000LMJ01.png  (best-focused merge)
```
→ Single best image, optimally focused

## Download Commands

### Download RDR (processed, default):
```bash
pixi run download-cheyava
```

### Download EDR (raw):
```bash
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_raw \
  8 \
  --sols cheyava \
  --use-raw
```

## Technical Details

### URL Conversion (Experimental)
The download script can attempt to convert between formats, but **NASA's URL structure is inconsistent**:

**PDS Atlas (RDR)** → **NASA Raw Images (might be EDR or RDR)**
```
/ids/rdr/shrlc/SIF_1613_...  →  /ids/edr/shrlc/SI1_1613_... (attempt)
```

**⚠️ WARNING**: NASA's raw images server sometimes uses `SIF_` filenames (which typically indicate RDR/processed data) in `/edr/` paths (which typically indicate raw data). The naming is inconsistent.

**Recommendation**: Stick with the default PDS Atlas URLs for reliable, well-documented RDR data. Only use `--use-raw` if you specifically need to access NASA's raw images server and understand the inconsistencies.

### Filename Prefixes

| Prefix | Meaning | Type |
|--------|---------|------|
| `SI1_` | SHERLOC/WATSON single focus EDR | EDR (raw) |
| `SI3_` | SHERLOC/WATSON all focus positions | EDR (raw) |
| `SIF_` | SHERLOC/WATSON focus-merged | RDR (processed) |

### Processing Pipeline

```
Camera → EDR (SI1_) → Focus Merging → RDR (SIF_)
         ↑                             ↑
         Raw data                      Best science product
```

## Recommendation

**For biosignature detection ML:**
Start with **RDR (default)** - you get cleaner data, fewer files, and the best-focused images. This is what scientists use for analysis.

**For custom processing research:**
Use **EDR (--use-raw)** - you get complete control over the processing pipeline.

## References

- [PDS4 Standards](https://pds.nasa.gov/datastandards/documents/)
- [Mars 2020 Archive](https://pds-geosciences.wustl.edu/missions/mars2020/)
- [SHERLOC Instrument](https://mars.nasa.gov/mars2020/spacecraft/instruments/sherloc/)
