# Mars Raw Utils Setup Instructions

The project now downloads IMG files directly from NASA's PDS archive. To process these IMG files, you have two options:

## Option 1: Install mars-raw-utils (Recommended for Production)

Mars-raw-utils is a Rust tool that provides professional-grade calibration for Mars rover images.

### Prerequisites
- Rust toolchain (cargo)
- OpenSSL development libraries

### Installation Steps

1. **Install OpenSSL dev libraries** (requires sudo):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libssl-dev pkg-config

   # Fedora/RHEL
   sudo dnf install openssl-devel pkg-config
   ```

2. **Install mars-raw-utils**:
   ```bash
   cargo install mars_raw_utils
   ```

3. **Set up calibration data**:
   ```bash
   mkdir ~/.marsdata
   # You'll need to clone the repo and copy calibration files:
   git clone https://github.com/MarsRaw/mars-raw-utils.git
   cp mars-raw-utils/mars-raw-utils-data/caldata/* ~/.marsdata/
   ```

### Usage

Convert PDS IMG to PNG:
```bash
mru pds2png -i data/raw/watson/sol_0530/SI3_0530_*.IMG
```

Calibrate images:
```bash
mru calibrate -i *.png -P m20_watson
```

## Option 2: Pure Python Converter (Currently Implemented)

We've implemented a Python-based IMG converter using PIL/Pillow and numpy that works with VICAR/PDS4 IMG files.

### Usage

```python
from mars_biosig.data.parsers.img_converter import IMGConverter

converter = IMGConverter()
png_path = converter.convert_img_to_png('data/raw/watson/sol_0530/image.IMG')
```

This pure Python approach:
- ✓ No external dependencies beyond Python
- ✓ Works immediately without system libraries
- ✓ Good for initial prototyping
- ⚠ Limited calibration compared to mars-raw-utils
- ⚠ May not handle all IMG variants

## Recommendation

- **For development/prototyping**: Use Option 2 (pure Python)
- **For production/research**: Install Option 1 (mars-raw-utils) when you have admin access

Both options are integrated into the pipeline and can be switched via configuration.
