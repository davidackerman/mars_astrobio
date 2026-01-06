# Cheyava Falls Images - Mars Perseverance Rover

## About This Discovery

**Cheyava Falls** is a rock discovered by NASA's Perseverance rover on **Sol 1215 (July 21, 2024)** in Mars' Jezero Crater. This discovery represents one of the most compelling potential biosignatures found on Mars to date.

## Key Features

The rock exhibits distinctive "**leopard spots**" - small, irregularly shaped light patches surrounded by thin rims of dark minerals. These features suggest:

- **Chemical redox boundaries** where oxidation and reduction reactions occurred
- **Vivianite** (hydrated iron phosphate) and **greigite** (iron sulfide) minerals
- Conditions that could have supported ancient microbial life
- Organic carbon compounds detected by SHERLOC instrument

## Files in This Directory

### Official NASA Images

1. **PIA26370_leopard_spots.jpg** (217 KB)
   - High-quality JPEG of the drill bit showing leopard spots
   - Image ID: PIA26370
   - Instrument: Mastcam-Z
   - Sol: 1215
   - Date: July 21, 2024

2. **PIA26370_leopard_spots.tif** (5.9 MB)
   - Full-resolution TIFF version (science-grade)
   - Same image as above, uncompressed

### Raw Mastcam-Z Images

3. **sol_1217_ZL0_1217_0774975958_568ECM_N0560000ZCAM09259_1100LMJ01.png** (18 KB)
   - Raw browse image from Sol 1217
   - Left Mastcam-Z camera (ZL0)
   - Date: July 23, 2024

## Scientific Significance

After a year-long peer review process, findings about the "Sapphire Canyon" sample from Cheyava Falls were published in **Nature (September 2025)**, confirming this as the strongest potential evidence for ancient life on Mars discovered to date.

### Key Minerals Found
- **Vivianite**: Hydrated iron phosphate (light patches)
- **Greigite**: Iron sulfide (dark rims)
- **Organic compounds**: Detected by SHERLOC spectroscopy

## Sources

- [NASA JPL Official Announcement](https://www.jpl.nasa.gov/news/nasa-says-mars-rover-discovered-potential-biosignature-last-year/)
- [Nature Publication](https://www.nature.com/articles/s41586-025-09413-0)
- [Planetary Society Analysis](https://www.planetary.org/articles/a-biosignature-on-mars-unpacking-perseverances-cheyava-falls-find)
- [NASA Photojournal PIA26370](https://science.nasa.gov/photojournal/mastcam-z-views-leopard-spots-in-perseverances-drill-bit/)

## How to Use These Images

These images can be used with the Mars biosignature detection pipeline in this repository:

```bash
# View the images
ls -lh data/cheyava_falls/

# Use in your analysis scripts
python scripts/analyze_texture.py --image data/cheyava_falls/PIA26370_leopard_spots.jpg
```

## Download Date

Downloaded: December 31, 2025

## Credit

Images: NASA/JPL-Caltech/ASU
Mission: Mars 2020 Perseverance Rover
