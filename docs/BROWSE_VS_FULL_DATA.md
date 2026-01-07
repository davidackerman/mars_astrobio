# Browse Images vs Full-Resolution Data

## THE CRITICAL ISSUE YOU DISCOVERED

Your PDS Atlas .bat file downloads **browse images** (small PNGs), NOT full-resolution science data!

## What You're Getting

### Current .bat File: Browse Images
```
/browse/sol/01613/ids/rdr/shrlc/SIF_1613_*.png
         ^^^^^^
```

- **Format**: PNG (web-friendly)
- **Size**: ~500 KB per image
- **Resolution**: Low (thumbnail/preview quality)
- **Purpose**: Quick visualization, web browsing
- **Good for**: Previewing data, making mosaics, initial screening
- **Bad for**: Detailed analysis, ML training on fine features

### What You Need: Full-Resolution Data
```
/data/sol/0004/ids/edr/watson/SI0_0004_*.IMG + .xml
      ^^^^
```

- **Format**: IMG + XML (PDS4 format)
- **Size**: ~5-20 MB per image
- **Resolution**: Full camera resolution
- **Purpose**: Scientific analysis
- **Good for**: ML training, biosignature detection, detailed texture analysis
- **Already downloaded**: You have this in `data/raw/watson/`!

## Size Comparison

**Your .bat file (browse images)**:
```bash
$ wc -l pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat
295887 lines  # ~296k browse images, ~14.7 MiB total
```

That's only **50 KB per image on average** - these are tiny previews!

**Full-resolution data (what you need)**:
```bash
$ ls -lh data/raw/watson/sol_0004/SI0_0004_*.IMG
-rw-r--r-- 1 user user 5.2M Jan 01 12:00 SI0_0004_0667297981_035ECM_N0010052SRLC10002_0000LUJ01.IMG
```

That's **5.2 MB** - 100x larger, full camera resolution!

## The Problem

Your PDS Atlas cart selected **"Browse Image"** products:

```
Choose Your Download Method
☑️ Browse Images        295887 items | 14.7 MiB   ← YOU SELECTED THIS
☐ Primary Product/File  295887 items | 0 B       ← YOU NEED THIS!
☐ PDS Label            295887 items | 767.7 MiB
```

## The Solution

### Option 1: Use Your Existing Full-Resolution Data (RECOMMENDED)

You already have full-resolution WATSON data in `data/raw/watson/`!

```bash
ls data/raw/watson/sol_*/
# These are full-resolution IMG files - use these for ML!
```

This was downloaded by your existing pipeline:
```bash
pixi run download-watson
```

### Option 2: Download Full-Resolution for More Sols

To get full-resolution data for Cheyava Falls / Wildcat Ridge:

1. Go to PDS Atlas cart
2. **Uncheck "Browse Image"**
3. **Check "Primary Product/File"** (the full IMG files)
4. Regenerate your .bat file
5. Run the download

**Warning**: Full data is MUCH larger:
- Browse: ~15 MB for 296k images
- Full data: ~1.5 TB for 296k images (100x larger!)

### Option 3: Use Browse Images for Initial Screening

Browse images are actually useful for:
- Quick visual inspection
- Training on gross features (rocks vs sand vs sky)
- Making overview mosaics
- Identifying regions of interest

Then download full-resolution for those specific sols.

## Your Current Situation

✅ **You have**:
- Full-resolution WATSON data (sols 0-100) in `data/raw/watson/`
- Browse images .bat file for sols 0-1600+

❌ **You don't have**:
- Full-resolution data for Cheyava Falls (sols 1200-1220)
- Full-resolution data for Wildcat Ridge (sols 490-510)

## Recommendation for Biosignature Detection

**For training your ML model:**

1. **Use full-resolution data** from `data/raw/watson/` (sols 0-100)
2. **Download full-resolution** for biosignature sites:
   - Regenerate PDS cart with "Primary Product/File" for sols 490-510, 1200-1220
   - Or use your existing download script that gets full IMG files

3. **Use browse images** only for:
   - Quick screening of large numbers of sols
   - Identifying interesting targets
   - Web visualization

## How to Tell What You Have

```bash
# Browse images (small PNGs)
file *.png
# Output: PNG image data, 1024 x 1024, 8-bit/color RGB

# Full-resolution data (large IMG + XML)
file *.IMG
# Output: data

ls -lh *.IMG
# Output: 5-20 MB per file
```

## Updated Download Strategy

**For your biosignature work:**

```bash
# 1. Keep using your existing full-resolution WATSON data (sols 0-100)
# Already have this!

# 2. Download full-resolution for biosignature sites
# Modify your download script or regenerate PDS cart for:
# - Cheyava Falls: sols 1200-1220 (PRIMARY PRODUCT)
# - Wildcat Ridge: sols 490-510 (PRIMARY PRODUCT)

# 3. Use browse images from .bat file for quick screening only
pixi run download-cheyava  # Gets browse PNGs (current)
# But need to get full IMG files for training!
```

## Bottom Line

Your .bat file gives you **low-res previews**, not science data.

For ML biosignature detection, you need **full-resolution IMG files** which you already have for sols 0-100.

For Cheyava Falls / Wildcat Ridge, either:
- Download full IMG files (large!)
- Or train on browse PNGs if gross features are sufficient
- Or use browse PNGs to identify interesting images, then download those specific full-resolution files

**The browse vs full-resolution difference is MORE important than RDR vs EDR or PDS vs NASA!**
