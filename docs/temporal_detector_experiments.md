# Temporal Detector Experiment Ideas

## Model Variants
- Frame-stack CNN: stack 4 frames as 12-channel input into ResNet18/34.
- 3D CNN: try R(2+1)D-18 or X3D-S for explicit temporal modeling.
- CNN + GRU: ResNet encoder per frame, GRU over time, heatmap head on pooled features.
- Difference-stream: feed frame differences (t1-t0, t2-t1, t3-t2) alongside raw frames.

## Noise-Robust Training
- Focal loss gamma sweep: 1.5, 2.0, 3.0.
- Label smoothing on class labels: 0.05-0.1.
- Temporal dropout: randomly drop 1 frame per sequence during training.
- MixUp on sequences with the same label set to reduce overfitting.

## Heatmap / Detection Tuning
- Heatmap loss weight sweep: 3.0, 5.0, 7.5.
- Peak threshold sweep for decoding: 0.05, 0.1, 0.2.
- Increase heatmap resolution: 8x8 -> 16x16 (adjust decoder thresholds).
