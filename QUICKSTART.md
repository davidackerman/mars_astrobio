# Quick Start Guide

Get up and running with the Mars Biosignature Detection project.

## Step 1: Install Pixi

If you don't have Pixi installed:

```bash
# On Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Restart your shell or source your profile
source ~/.bashrc  # or ~/.zshrc
```

For other installation methods, see https://pixi.sh

## Step 2: Install Dependencies

```bash
cd mars_astrobio

# Install all dependencies (this will take a few minutes)
pixi install

# Activate the environment
pixi shell
```

This will install:
- Python 3.10+
- PyTorch with CUDA support
- All data processing and ML libraries
- Development tools (pytest, black, ruff)

## Step 3: Get a NASA API Key

1. Go to https://api.nasa.gov/
2. Generate your API key (free, instant)
3. Set it as an environment variable:

```bash
export NASA_API_KEY="your_key_here"

# Add to your ~/.bashrc or ~/.zshrc to make it permanent
echo 'export NASA_API_KEY="your_key_here"' >> ~/.bashrc
```

## Step 4: Download Sample Data

```bash
# Download first 10 sols of WATSON images (small test)
python scripts/download_data.py --instrument watson --sols 0-10

# Or use the Pixi task for first 100 sols
pixi run download-watson
```

## Step 5: Explore the Data

```bash
# Start Jupyter Lab
pixi run notebook

# Open notebooks/01_data_exploration.ipynb
```

## Next Steps

### Phase 1: Data Collection
- Download more WATSON images
- Manually annotate known biosignature sites
- Prepare training dataset

### Phase 2: Model Training
- Implement baseline ResNet-50 model
- Train on texture classification
- Evaluate performance

### Phase 3: Discovery
- Run inference on entire mission archive
- Identify new biosignature candidates
- Scientific analysis

## Common Commands

```bash
# Development workflow
pixi run format          # Format code with Black
pixi run lint            # Lint with Ruff
pixi run test            # Run tests
pixi run typecheck       # Type check with mypy

# Data pipeline
pixi run download-watson # Download WATSON images
pixi run prepare-data    # Prepare dataset

# Training
pixi run train           # Train model

# Jupyter
pixi run notebook        # Start Jupyter Lab
```

## Project Structure Overview

```
mars_astrobio/
├── src/mars_biosig/     # Main Python package
├── configs/             # YAML configurations
├── scripts/             # CLI tools
├── notebooks/           # Jupyter notebooks
├── data/                # Downloaded data (gitignored)
├── models/              # Trained models (gitignored)
└── tests/               # Unit and integration tests
```

## Troubleshooting

### Pixi install fails
- Check your internet connection
- Try: `pixi clean` then `pixi install`
- Check Pixi version: `pixi --version` (should be v0.30+)

### CUDA not available
- Verify GPU: `nvidia-smi`
- Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Pixi installs CUDA automatically, but ensure drivers are up to date

### NASA API rate limits
- Free tier: 1000 requests/hour
- Use caching to avoid re-downloading
- Download in smaller batches if needed

## Documentation

- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [Implementation Plan](/.claude/plans/elegant-questing-salamander.md) - Detailed architecture
- [Data README](data/README.md) - Data organization

## Getting Help

- Check the [Issues page](https://github.com/yourusername/mars_astrobio/issues)
- Read the documentation in `docs/`
- Review example notebooks in `notebooks/`

Happy biosignature hunting! 🚀🔬
