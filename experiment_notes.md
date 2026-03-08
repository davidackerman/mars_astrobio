# Experiment Notes

## Recent Code Updates

- `scripts/infer_crop_heatmap.py`: accepts `--data-dir` pointing directly at a subjects folder; GIF duration slowed and written via `imageio.get_writer`; output files now prepend peak heatmap score; writes `heatmap_peaks.csv` sorted by peak score.
- `scripts/generate_synthetic_groundtruth.py`: added mover size controls (`--min-mover-radius`, `--max-mover-radius`); added blinkers (stationary on/off bright spots) with size controls; background bright add-ons default off (`--background-bright-prob 0.0`).
- `src/scientific_pipelines/core/training/datasets/temporal_crop_dataset.py`: added bright-negative sampling for background crops.
- `scripts/train_crop_classifier.py`: added CLI flags to control bright-negative sampling.
- `scripts/sample_training_crops.py`: new utility to save training crops as GIFs grouped by category.

## Inference Commands (Recent)

```bash
pixi run python scripts/infer_crop_heatmap.py \
  --data-dir data/backyard_worlds/subjects \
  --checkpoint outputs/checkpoints/crop_20260114_143212/split.pt \
  --crop-size 128,128 \
  --stride 32 \
  --threshold 0.995 \
  --any-object \
  --normalize \
  --overlay-alpha 0.45 \
  --contour-level 0.995 \
  --out-dir outputs/crop_inference_temporal_20260114_movers_only
```

```bash
pixi run python scripts/infer_crop_heatmap.py \
  --data-dir data/backyard_worlds/subjects \
  --checkpoint outputs/checkpoints/crop_20260114_165853/split.pt \
  --crop-size 128,128 \
  --stride 32 \
  --threshold 0.995 \
  --any-object \
  --normalize \
  --overlay-alpha 0.45 \
  --contour-level 0.995 \
  --out-dir outputs/crop_inference_temporal_20260114_movers_blinkers
```

## Synthetic Data Generation (Recent)

```bash
pixi run python scripts/generate_synthetic_groundtruth.py \
  --background-dir data/backyard_worlds/subjects \
  --output-dir data/backyard_worlds/synthetic_movers_blinkers_brightbg \
  --num-sequences 200 \
  --min-movers 1 --max-movers 2 \
  --min-dipoles 0 --max-dipoles 0 \
  --ensure-object \
  --min-mover-radius 7.0 \
  --min-blinkers 1 --max-blinkers 3 \
  --min-blinker-radius 6.0 --max-blinker-radius 30.0
```

## Training Commands (Recent)

```bash
nohup pixi run python scripts/train_crop_classifier.py \
  --data-dir data/backyard_worlds/synthetic_movers_blinkers \
  --annotations-path data/backyard_worlds/synthetic_movers_blinkers/annotations.json \
  --crop-size 64,64 \
  --samples-per-subject 10 \
  --positive-fraction 0.5 \
  --val-fraction 0.25 \
  --positive-jitter 48,48 \
  --no-crop \
  --no-noise \
  --no-blur \
  --any-object \
  --num-epochs 1000 \
  --checkpoint-every 10 \
  > outputs/crop_classifier_movers_blinkers_64_nohup.log 2>&1 &
```

```bash
nohup pixi run python scripts/train_crop_classifier.py \
  --data-dir data/backyard_worlds/synthetic_movers_blinkers_bright \
  --annotations-path data/backyard_worlds/synthetic_movers_blinkers_bright/annotations.json \
  --crop-size 64,64 \
  --samples-per-subject 20 \
  --positive-fraction 0.5 \
  --val-fraction 0.25 \
  --positive-jitter 32,32 \
  --no-crop \
  --no-noise \
  --no-blur \
  --any-object \
  --negative-bright-fraction 0.5 \
  --negative-bright-percentile 85 \
  --negative-bright-samples 200 \
  --num-epochs 1000 \
  --checkpoint-every 10 \
  > outputs/crop_classifier_movers_blinkers_bright_64_nohup.log 2>&1 &
```

## Crop Sampling GIFs

```bash
pixi run python scripts/sample_training_crops.py \
  --data-dir data/backyard_worlds/synthetic_movers_blinkers_brightbg \
  --annotations-path data/backyard_worlds/synthetic_movers_blinkers_brightbg/annotations.json \
  --crop-size 64,64 \
  --samples-per-subject 10 \
  --positive-fraction 0.5 \
  --positive-jitter 32,32 \
  --no-crop \
  --negative-bright-fraction 0.5 \
  --negative-bright-percentile 85 \
  --negative-bright-samples 200 \
  --num-per-class 50 \
  --out-dir outputs/training_crop_samples_64
```
