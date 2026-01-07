# Why Different Images on Different Servers?

## The Question

Why does an image like `SI1_1613_0810132725_121EBY_N0790102SRLC00472_0000LMJ01.png` appear on NASA's website but not in the PDS Atlas .bat file, and vice versa?

## The Answer: Different Product Types & Processing Pipelines

The two servers serve **different product types** from the same raw acquisition. They're filtering and processing the same underlying data differently.

## Breaking Down the Filename

Let's decode: `SI1_1613_0810132725_121EBY_N0790102SRLC00472_0000LMJ01.png`

```
SI1  = SHERLOC/WATSON camera (S), Instrument 1
1613 = Sol number
0810132725 = Timestamp (Mars time)
121  = Sequence number
EBY  = PRODUCT TYPE / FILTER CODE ← This is the key difference!
N... = Processing level indicator
0790102SRLC00472 = Observation ID
0000LMJ01 = Product version/variant
```

## Product Type Codes Found

### In PDS Atlas .bat file (for timestamp 0810132725):
- `RZS` - Z-stack (focus stack, multiple depths)
- `RAS` - Auto-focus summary
- `RAF` - Auto-focus final
- `RAD` - Auto-focus diagnostic
- `EDR` - Experimental Data Record (raw)

### On NASA Website (your example):
- `EBY` - Unknown specific code (possibly a different processing type)

### Other common codes:
- `SIF` - Science Image Focus-merged
- `ECM` - Extended Color Merge
- Various filter designations

## Why Different Servers Have Different Images

### 1. **PDS Atlas Cart Selection**
When you created your cart on PDS Atlas, you selected specific product types:
- Likely selected: "RDR browse images" or "focus-merged products"
- This filters to specific processing levels
- Your .bat file contains only what you added to your cart

### 2. **NASA Raw Images Server**
The NASA website shows **all** product types for a given sol:
- EDR (raw) images
- Various RDR processing levels
- Thumbnail/browse versions
- Experimental products
- Everything acquired during that sol

### 3. **Different Processing Pipelines**
Same acquisition → Multiple products:
```
Raw Camera Data
    ↓
    ├→ EDR (raw, minimal processing)
    ├→ RDR-RZS (Z-stack focus series)
    ├→ RDR-RAF (Auto-focus final)
    ├→ RDR-SIF (Focus-merged science product)
    └→ EBY/ECM/others (Various processing types)
```

## Example: Same Target, Multiple Products

For sol 1613, timestamp 0810132725, sequence 121:

**PDS Atlas has:**
1. `SI1_1613_0810132725_121RZS_T...` (Z-stack)
2. `SI1_1613_0810132725_121RAS_T...` (Auto-focus summary)
3. `SI1_1613_0810132725_121RAF_T...` (Auto-focus final)
4. `SI1_1613_0810132725_121RAD_T...` (Auto-focus diagnostic)
5. `SI1_1613_0810132725_121EDR_T...` (Raw EDR)

**NASA Website also has:**
- `SI1_1613_0810132725_121EBY_N...` (Different product type)
- Possibly others...

These are **all from the same camera pointing**, just processed differently!

## Why This Matters

### For Your Download Script
The .bat file contains **only what you selected** in your PDS Atlas cart:
- If you selected "RDR browse images" → you get RDR products
- If you selected "focus-merged" → you get SIF products
- If you selected specific product types → that's what's in the .bat

### For Finding Specific Images
If you want a specific image you saw on NASA's website:

**Option 1**: Add it to a new PDS Atlas cart with the right product types
**Option 2**: Download directly from NASA using the exact URL
**Option 3**: Modify your cart selection to include more product types

## How to Get More Complete Coverage

### Regenerate Your PDS Atlas Cart
1. Go to https://pds-imaging.jpl.nasa.gov/beta/cart
2. Select sol range (e.g., 1613)
3. Select instrument: SHERLOC/WATSON
4. **Select multiple product types**:
   - ☑️ RDR (all types)
   - ☑️ EDR (all types)
   - ☑️ Browse images
   - ☑️ All processing levels
5. Generate new .bat file
6. Use with your download script

### Or: Use NASA's Server Directly
If you need specific product types not in PDS Atlas:
```bash
# Direct download from NASA
curl -o image.png "https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01613/ids/edr/browse/shrlc/SI1_1613_0810132725_121EBY_N0790102SRLC00472_0000LMJ01.png"
```

## Recommendation

**For comprehensive biosignature analysis:**

1. **Start with your current .bat file** (RDR products) - best quality, focus-merged
2. **If you need more variants**, regenerate cart with additional product types
3. **For specific missing images**, download directly from NASA using exact URLs

The images aren't "missing" - they're just **different products** from the same acquisition, and your cart only selected certain types!

## Real Example: Sol 1613, Timestamp 0810132725

**PDS Atlas .bat file HAS:**
- `SI1_1613_0810132725_121RZS_...` ✅
- `SI1_1613_0810132725_121RAS_...` ✅
- `SI1_1613_0810132725_121RAF_...` ✅
- `SI1_1613_0810132725_121RAD_...` ✅
- `SI1_1613_0810132725_121EDR_...` ✅
- `SI1_1613_0810132725_121ECM_...` ✅

**PDS Atlas .bat file DOES NOT HAVE:**
- `SI1_1613_0810132725_121EBY_...` ❌ (but has EBY for other timestamps!)

**NASA Website HAS:**
- `SI1_1613_0810132725_121EBY_...` ✅ (the one you found!)

**Why?** PDS Atlas cart has selective filtering - it includes EBY products for some acquisitions but not others. NASA's website shows everything without filtering.

## Summary

| Source | Coverage | Product Types | Completeness |
|--------|----------|---------------|--------------|
| **PDS Atlas .bat** | Only what you selected in cart | Specific RDR/EDR types | **Partial** - filtered |
| **NASA Raw Website** | Everything for the sol | All product types, all processing levels | **Complete** - unfiltered |

**Key insight**: Even when your PDS cart includes a product type (like EBY), it may not include that type for EVERY acquisition. The cart applies additional filters based on quality, processing success, or other criteria.

Different servers, different filtering, same underlying data! 🔬
