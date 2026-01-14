Temporal crop classifier sketch

Goal
Train a lightweight model on fixed-size 4-frame crops to predict whether a mover
and/or dipole appears in that crop. This reduces memory cost and helps balance
positives with background-only samples.

Data strategy
- Build multiple crops per subject with a mix of positive-centered and random crops.
- Label each crop based on which annotated centers fall inside the crop.
- Split by subject ID (leave-one-subject-out) to avoid leakage.
- Frames are cropped to remove the left border and bottom legend before sampling.

Implementation
- Dataset: `BackyardWorldsTemporalCropDataset` in
  `src/scientific_pipelines/core/training/datasets/temporal_crop_dataset.py`.
  It samples crop centers once and reuses them for train/val splits.
- Model: `TemporalCropClassifier` in
  `src/scientific_pipelines/core/training/models/temporal_crop_classifier.py`.
  A small 3D CNN with global pooling and multi-label output.
- Training: `scripts/train_crop_classifier.py` runs LOOCV and reports micro-F1.
- Visualization: `scripts/visualize_crop_samples.py` exports crop strips for
  manual inspection.
- Inference: `scripts/infer_crop_heatmap.py` slides the crop classifier over
  full frames and writes heatmap overlays.

Training options
- Use `--balanced-sampling` to balance training crops across classes.
- Use `--any-object` to collapse mover/dipole into a single label.
- Use `--early-stopping-patience` to stop when val loss stalls.
- Use `--save-checkpoints` to export fold checkpoints for inference.

Next steps
- Tune crop size and positive fraction (e.g., 96-192px, 0.5-0.8 positive).
- Add hard-negative mining (crops near but excluding keypoints).
- If localization is needed, switch to a crop-level U-Net and keep this as a
  proposal stage.
