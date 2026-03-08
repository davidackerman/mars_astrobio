#!/usr/bin/env bash
set -euo pipefail

csv_path="/mnt/bigdisk/Programming/mars_astrobio/outputs/crop_inference_2026115_233523_5000_samples/heatmap_peaks.csv"
subjects_src="/mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/subjects"
selected_dir="/mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/subjects_toprand_1000"
synthetic_dir="/mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/synthetic_toprand_1000"

# Selection settings
TOP_N=500
RAND_N=500
SEED=42

# Synthetic generation settings (movers+blinkers + bright backgrounds)
NUM_SEQUENCES=1000
BRIGHT_PROB=0.9
MIN_BRIGHTS=2
MAX_BRIGHTS=6
MIN_BRIGHT_RADIUS=6.0
MAX_BRIGHT_RADIUS=30.0

python - <<PY
import csv, random
from pathlib import Path

csv_path = Path("$csv_path")
top_n = $TOP_N
rand_n = $RAND_N
seed = $SEED

scores = {}
with csv_path.open() as f:
    for row in csv.DictReader(f):
        sid = row["subject_id"]
        score = float(row["peak_score"])
        scores[sid] = max(score, scores.get(sid, float("-inf")))

sorted_sids = sorted(scores, key=lambda s: scores[s], reverse=True)
top = sorted_sids[:top_n]
rest = sorted_sids[top_n:]
random.seed(seed)
rand = random.sample(rest, rand_n)

selected = top + rand
out_path = Path("/mnt/bigdisk/Programming/mars_astrobio/outputs/toprand_1000_subjects.txt")
out_path.write_text("\n".join(selected) + "\n")
print(f"Wrote {len(selected)} subject_ids -> {out_path}")
PY

mkdir -p "$selected_dir"
while read -r sid; do
  ln -sfn "$subjects_src/$sid" "$selected_dir/$sid"
done < /mnt/bigdisk/Programming/mars_astrobio/outputs/toprand_1000_subjects.txt

pixi run python /mnt/bigdisk/Programming/mars_astrobio/scripts/generate_synthetic_groundtruth.py \
  --background-dir "$selected_dir" \
  --output-dir "$synthetic_dir" \
  --num-sequences $NUM_SEQUENCES \
  --min-movers 1 --max-movers 2 \
  --min-dipoles 0 --max-dipoles 0 \
  --ensure-object \
  --min-mover-radius 5.0 \
  --min-blinkers 1 --max-blinkers 3 \
  --min-blinker-radius 5.0 --max-blinker-radius 30.0 \
  --background-bright-prob $BRIGHT_PROB \
  --min-background-brights $MIN_BRIGHTS --max-background-brights $MAX_BRIGHTS \
  --min-background-bright-radius $MIN_BRIGHT_RADIUS --max-background-bright-radius $MAX_BRIGHT_RADIUS

nohup pixi run python scripts/train_crop_classifier.py \
  --data-dir /mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/synthetic_toprand_1000 \
  --annotations-path /mnt/bigdisk/Programming/mars_astrobio/data/backyard_worlds/synthetic_toprand_1000/annotations.json \
  --crop-size 64,64 \
  --samples-per-subject 10 \
  --positive-fraction 0.5 \
  --val-fraction 0.25 \
  --positive-jitter 32,32 \
  --no-crop \
  --any-object \
  --negative-bright-fraction 0.75 \
  --negative-bright-percentile 95 \
  --negative-bright-samples 200 \
  --loss focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --threshold-sweep \
  --num-epochs 1000 \
  --checkpoint-every 5 \
  --positive-mover-min-frames 3 \
  --threshold-low --threshold-value 125 \
  --noise-prob 0.6 --adaptive-noise \
  --denoise --denoise-prob 0.2 --denoise-strength 7 \
  --batch-size 64 \
  --persistent-workers \
  --prefetch-factor 4 \
  --num-workers 8 \
  > outputs/crop_classifier_toprand_1000_64_focal_nohup.log 2>&1 &
