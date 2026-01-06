# Important: Mars Photos API Limitation

## Issue Discovered

The NASA Mars Photos API (`https://api.nasa.gov/mars-photos/`) does **not** currently support Perseverance rover WATSON data. This API was designed for older rovers (Curiosity, Opportunity, Spirit).

## What This Means

Our current downloader won't work with the Mars Photos API for WATSON images. We need to:

1. Access the **PDS Geosciences Node** directly
2. Use a different approach to browse and download Perseverance data

## Solutions

### Option 1: Direct PDS Archive Access (Recommended for Phase 3+)
- Access: https://pds-geosciences.wustl.edu/missions/mars2020/
- Browse by sol and instrument
- Download IMG files and XML labels directly
- **Pros**: Complete data access, all instruments
- **Cons**: More complex, requires parsing directory structure

### Option 2: Use Pre-compiled Datasets
- Many researchers share Perseverance image datasets
- Could download curated collections
- **Pros**: Faster to get started
- **Cons**: Limited to what others have selected

### Option 3: Manual Download + Annotation Focus
- For now, manually download known biosignature examples:
  - Cheyava Falls (Sol 1174)
  - Wildcat Ridge (Sol 528)
- Focus on model architecture and training pipeline
- Automated downloading in future phase

## Recommendation

**For immediate progress:** Let's pivot to Option 3:
1. Manually download a few key images from PDS
2. Focus on Phase 3 (Model Architecture)
3. Implement proper PDS archive crawler later

This lets you:
- Keep momentum on the ML/AI parts (your core interest)
- Test the PyTorch dataset and transforms we built
- Build and train models
- Add automated PDS downloading as a future enhancement

## Your API Key is Still Valid

Your NASA API key works fine - it's just that the Mars Photos API endpoint doesn't support Perseverance yet. Your key will work for:
- Other NASA APIs
- When/if Perseverance is added to Mars Photos API
- We can still use it for rate limiting compliance

## Next Steps?

Would you like to:
A) Manually download some sample WATSON images and move forward with model building?
B) Implement a PDS Geosciences direct browser/downloader?
C) Use Curiosity data as a proxy to test the pipeline?

Let me know how you'd like to proceed!
