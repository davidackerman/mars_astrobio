# Mars Image Data Sources - URL Comparison

## The Problem: Inconsistent Naming

There are **two different servers** for Mars Perseverance images, and they don't always use consistent naming conventions.

## 1. PDS Atlas API (Recommended)

**URL**: `https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_imgops/browse/sol/{SOL}/ids/rdr/shrlc/{FILENAME}.png`

**Example**:
```
https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/
  mars2020_imgops/browse/sol/01613/ids/rdr/shrlc/
  SIF_1613_0810171983_281RZS_N0790102SRLC08062_0000LMJ01.png::14
```

**Characteristics**:
- ✅ **Reliable**: Consistent naming and availability
- ✅ **Well-documented**: Official PDS4 archive
- ✅ **RDR data**: Processed, focus-merged, best-quality images
- ✅ **API access**: Programmatic access through PDS API
- ❌ **Delayed**: Released in batches (not real-time)

**Filename pattern**: `SIF_` (SHERLOC/WATSON Focus-merged)

## 2. NASA Raw Images Server (Inconsistent)

**URL**: `https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{SOL}/ids/edr/browse/shrlc/{FILENAME}.png`

**Example (you found)**:
```
https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01613/ids/edr/
  browse/shrlc/SIF_1613_0810171983_281RAD_N0790102SRLC08062_0000LMJ01.png
                ^^^                                          (note: different suffix)
```

**Characteristics**:
- ⚠️ **Inconsistent naming**: Uses `SIF_` (RDR naming) in `/edr/` (raw) paths!
- ⚠️ **Unpredictable**: Some files exist, others don't
- ✅ **Near real-time**: Updated within days of acquisition
- ✅ **All cameras**: Mastcam-Z, NavCam, HazCam, WATSON, etc.
- ✅ **Browse images**: Web-friendly PNG previews

**Filename pattern**: `SIF_`, `SI1_`, `SI3_` - **mixed and inconsistent**

## The Mismatch You Found

You noticed that:
- PDS Atlas curl script has: `SIF_1613_0810171983_281RZS_...png`
- NASA raw images has: `SIF_1613_0810132725_121ECM_...png` (different timestamp!)

**Why?** These are **different images from the same sol**:
- `0810171983` vs `0810132725` = different acquisition times
- `281RZS` vs `121ECM` = different camera filters/modes
- Same sol (1613), different targets or times

## What This Means for Downloads

### PDS Atlas URLs (Default)
```bash
pixi run download-cheyava
```
→ Downloads **reliable RDR processed images** from PDS Atlas API
→ **Recommended for science and ML**

### NASA Raw Images URLs (Experimental)
```bash
pixi run python scripts/download_pds_atlas.py \
  pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat \
  data/raw/watson_raw \
  8 \
  --sols cheyava \
  --use-raw
```
→ **Attempts** to download from NASA server
→ **May fail** due to inconsistent naming
→ Only use if you specifically need NASA's server

## Recommendation

**Stick with PDS Atlas (default)** because:

1. **Reliable naming**: Files actually exist at the URLs
2. **Best quality**: RDR processed, focus-merged images
3. **Consistent**: Predictable file structure
4. **Well-documented**: Part of official PDS4 archive

The `--use-raw` option is **experimental** and mainly for users who:
- Need to access NASA's near-real-time server for recent sols
- Want to compare PDS vs NASA data sources
- Are researching data pipeline differences

## Finding Specific Images

If you're looking for a specific image you saw on NASA's website:

1. **Find the image ID** on mars.nasa.gov
2. **Look for it in the PDS Atlas** first (more reliable)
3. **Download directly** using the exact URL if needed

For the image you mentioned:
```bash
# Direct download
curl -o image.png "https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01613/ids/edr/browse/shrlc/SI1_1613_0810132725_121ECM_N0790102SRLC00472_0000LMJ01.png"
```

## Critical Difference: RDR Products

**PDS Atlas has many RDR (focus-merged) products that NASA's raw server DOESN'T have:**

PDS Atlas example:
```
/ids/rdr/shrlc/SIF_1613_0810171983_281RZS_N0790102SRLC08062_0000LMJ01.png
```
- `SIF_` prefix = RDR focus-merged product
- In `/rdr/` path (correct)
- **May NOT exist on NASA's raw images server!**

NASA's server focuses on:
- EDR (raw) products in `/edr/` paths
- Some RDR products inconsistently

**This is why `--use-raw` is experimental** - many PDS Atlas RDR products won't have equivalents on NASA's server!

## Summary

| Source | Reliability | Data Type | Coverage | Use Case |
|--------|-------------|-----------|----------|----------|
| **PDS Atlas** | ✅ High | RDR (processed) | **Complete RDR archive** | **Recommended - science & ML** |
| **NASA Raw** | ⚠️ Variable | Mostly EDR, some RDR | **Incomplete RDR, complete EDR** | Near-real-time EDR only |

**Bottom line**:
- Use **PDS Atlas** (default) for reliable, complete RDR processed images ✅
- Use **NASA Raw** only if you need specific EDR products not in your PDS cart ⚠️
- **Not all PDS Atlas URLs will work on NASA's server** - especially RDR products!
