# Mars Raw Images Download Methods

## Two Different Data Sources

### 1. PDS Archive (Current WATSON Pipeline)
**URL Pattern**: `https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_imgops/data_watson_imgops/sol/`

- **What it has**: WATSON camera images (close-up micro-imager on robotic arm)
- **Format**: PDS4 format (.IMG files with .xml metadata)
- **Coverage**: Lower sol numbers (0-314 available currently)
- **Used by**: Your existing `scripts/download_data.py` pipeline

### 2. NASA Raw Images Gallery (New - Used for Cheyava Falls)
**URL**: `https://mars.nasa.gov/mars2020/multimedia/raw-images/`

- **What it has**: ALL camera images from Perseverance, including:
  - Mastcam-Z (ZL0/ZR0) - mast-mounted stereo cameras
  - Navigation cameras (NCL/NCR)
  - Hazard cameras (various)
  - WATSON (also available here)
  - And more...
- **Format**: Browse images (.png) with access to raw (.IMG) files
- **Coverage**: Up-to-date, includes recent sols (1217+)
- **Access Pattern**: 
  ```
  https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{SOL:05d}/ids/edr/browse/zcam/{FILENAME}.png
  ```

## Why Two Different Sources?

### PDS Archive (planetarydata.jpl.nasa.gov)
- ✅ Structured, machine-readable PDS4 format
- ✅ Complete metadata in XML
- ✅ Official science archive
- ❌ Delayed releases (batched every ~4 months)
- ❌ Only has WATSON in `mars2020_imgops` endpoint

### NASA Raw Images (mars.nasa.gov)
- ✅ Near real-time updates (days after acquisition)
- ✅ All cameras available
- ✅ Web-friendly browse images
- ✅ Can access higher sols (1200+)
- ❌ Less structured (requires knowing exact filenames)
- ❌ Browse images are lower resolution

## What I Did for Cheyava Falls

Since Cheyava Falls was discovered at **Sol 1215-1217** and uses **Mastcam-Z** (not WATSON), I:

1. **Found the image ID** from NASA's raw images page:
   ```
   ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ
   ```

2. **Downloaded browse image** from:
   ```
   https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
   ```

3. **Downloaded official processed image** from NASA Photojournal:
   ```
   https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia26/pia26370/PIA26370.jpg
   https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia26/pia26370/PIA26370.tif
   ```

## Camera Prefixes on mars.nasa.gov

- **ZL0/ZR0**: Mastcam-Z Left/Right (high-res color, on mast)
- **SI0/SI1**: SHERLOC (includes WATSON micro-imager)
- **NCL/NCR**: Navigation cameras
- **FCL/FCR**: Front hazard cameras
- **RCL/RCR**: Rear hazard cameras
- **PCL/PCR**: Parachute cameras
- **And more...**

## Recommendation for Your Pipeline

### For Historical Data (Sol 0-300)
✅ Use existing WATSON downloader with PDS archive
```bash
python scripts/download_data.py --instrument watson --sols 0-100
```

### For Recent Data or Other Cameras
✅ Access mars.nasa.gov raw images:
1. Browse images at: https://mars.nasa.gov/mars2020/multimedia/raw-images/
2. Filter by sol, camera type
3. Download using the URL pattern discovered

### For Published Science Images
✅ Use NASA Photojournal (PIA catalog):
- Browse: https://science.nasa.gov/photojournal/
- Direct download pattern: `https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia{SERIES}/pia{NUMBER}/PIA{NUMBER}.jpg`

## Files Created

I created these helper scripts:
1. `download_cheyava_direct.py` - Downloads from mars.nasa.gov raw images
2. `download_leopard_spots.py` - Downloads from NASA Photojournal
3. Downloaded to: `data/cheyava_falls/`

## Next Steps

To extend your pipeline to support Mastcam-Z and other cameras from mars.nasa.gov:

1. Create `src/mars_biosig/data/downloaders/mastcamz.py`
2. Implement sol listing from https://mars.nasa.gov/mars2020/multimedia/raw-images/
3. Parse image listing pages to get filenames
4. Download using discovered URL patterns

