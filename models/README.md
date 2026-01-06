# Models Directory

This directory contains trained model checkpoints.

## Structure

- `checkpoints/` - Training checkpoints (gitignored)
  - Organized by experiment ID
  - Includes intermediate epochs for resume

- `production/` - Production-ready models (gitignored)
  - Best performing models
  - Ready for deployment

## Usage

### Saving Models

Models are automatically saved during training:

```python
# Training automatically saves checkpoints
pixi run train
```

### Loading Models

```python
from mars_biosig.inference.predictor import BiosignaturePredictor

predictor = BiosignaturePredictor()
predictor.load_model("models/production/watson_texture_v1.pt")
```

## Checkpoint Format

Checkpoints include:
- Model state_dict
- Optimizer state
- Training configuration
- Epoch number
- Best validation metrics

## Notes

- Model files are large (>100MB) and are gitignored
- Keep notes about model versions in this README
- Tag production models with version numbers
