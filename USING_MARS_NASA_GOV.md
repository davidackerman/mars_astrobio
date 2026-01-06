# Using mars.nasa.gov/mars2020/multimedia/raw-images/

## How to Download Images from the Website

The mars.nasa.gov raw images gallery contains ALL Perseverance camera images, updated in near real-time. Here's how to download them:

### Method 1: Using Known URLs (Recommended)

1. **Find images on the website:**
   - Visit: https://mars.nasa.gov/mars2020/multimedia/raw-images/
   - Filter by:
     - Sol number (e.g., 1217 for Cheyava Falls)
     - Camera (e.g., Mastcam-Z for ZL0/ZR0)
   - Click on an image

2. **Get the image URL:**
   - Right-click on the image → "Open image in new tab"
   - Or inspect the page source and find URLs like:
     ```
     https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
     ```

3. **Create a URL list file:**
   ```bash
   # Create image_urls.txt with one URL per line
   cat > image_urls.txt << 'EOL'
   https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
   https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZR0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
   EOL
   ```

4. **Download all images:**
   ```bash
   pixi run python scripts/download_from_mars_nasa_gov.py --url-list image_urls.txt
   ```

### Method 2: Download Specific Known Images

For Cheyava Falls (Sol 1217):
```bash
pixi run python scripts/download_from_mars_nasa_gov.py --sol 1217 --camera zcam
```

Or download a specific file if you know the filename:
```bash
pixi run python scripts/download_from_mars_nasa_gov.py \
  --sol 1217 \
  --camera zcam \
  --filename ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
```

## URL Pattern Explained

```
https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{SOL:05d}/ids/edr/browse/{CAMERA}/{FILENAME}
```

- **SOL**: 5-digit zero-padded sol number (e.g., `01217`)
- **CAMERA**: Camera directory:
  - `zcam` - Mastcam-Z (ZL0/ZR0)
  - `sherlc` - SHERLOC/WATSON (SI0/SI1)
  - `ncam` - Navigation cameras (NCL/NCR)
  - `fcam` - Front hazard cameras
  - `rcam` - Rear hazard cameras
- **FILENAME**: Image filename from the gallery

## Camera Prefixes

| Prefix | Camera | Description |
|--------|--------|-------------|
| ZL0/ZR0 | Mastcam-Z | Left/Right mast-mounted color cameras |
| SI0/SI1 | SHERLOC/WATSON | Close-up micro-imager on arm |
| NCL/NCR | NavCam | Navigation cameras |
| FCL/FCR | FrontHazCam | Front hazard avoidance |
| RCL/RCR | RearHazCam | Rear hazard avoidance |

## Tips for Finding Images

### On the Website
1. Use the Sol filter to narrow down to specific days
2. Use camera filter to select instrument
3. Browse through thumbnails
4. Right-click on interesting images to get URLs

### Browser Developer Tools
1. Open the website in Chrome/Firefox
2. Press F12 to open Developer Tools
3. Go to Network tab
4. Filter images as you browse
5. Click on image requests to see full URLs
6. Copy URLs to your list file

## Example Workflow

**Goal**: Download all Mastcam-Z images from Sol 1217 (Cheyava Falls)

```bash
# 1. Visit the website and collect URLs
# https://mars.nasa.gov/mars2020/multimedia/raw-images/
# Filter: Sol=1217, Camera=Mastcam-Z

# 2. Create URL list (example)
cat > sol_1217_mastcamz.txt << 'EOL'
https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png
https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01217/ids/edr/browse/zcam/ZL0_1217_0774976102_568ECM_N0560000ZCAM09259_1100LMJ02.png
# ... add more URLs
EOL

# 3. Download
pixi run python scripts/download_from_mars_nasa_gov.py --url-list sol_1217_mastcamz.txt

# 4. Check results
ls -lh data/raw/mars_nasa_gov/sol_1217/
```

## Why This Approach?

The mars.nasa.gov website uses JavaScript to dynamically load images, making it difficult to scrape automatically. However:

✅ **Pros:**
- Gets most recent images (days after acquisition)
- All cameras available
- High-quality browse images
- No API key needed

✅ **Simple workflow:**
1. Use website to browse/filter
2. Collect URLs manually or via browser tools
3. Batch download with script

✅ **One-time effort:**
- For specific events (like Cheyava Falls), collect URLs once
- Share URL lists with collaborators
- Reproducible downloads

## Current Limitations

⚠️ **No automatic discovery**: The script cannot automatically list all images for a sol because:
- Directory listing is disabled (403 Forbidden)
- The website uses JavaScript/React to load image lists
- No public API for bulk queries

💡 **Solution**: Manual URL collection from website is currently the most reliable method.

## Future Enhancements

Possible improvements:
1. Selenium/Playwright web scraper to automate URL collection
2. Browser extension to export image URLs
3. Reverse-engineer the website's internal API calls
4. Parse the JavaScript bundle to find data endpoints

## Already Downloaded

The following Cheyava Falls images are in [data/cheyava_falls/](data/cheyava_falls/):
- ✅ PIA26370_leopard_spots.jpg (official processed image)
- ✅ PIA26370_leopard_spots.tif (full resolution)
- ✅ sol_1217_ZL0_*.png (raw browse image)

