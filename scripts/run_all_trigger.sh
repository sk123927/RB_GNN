#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "trigger/er1500"
  "trigger/sf1750"
  "trigger/wiki7115"
  "trigger/dblp12591"
)

for dataset in "${DATASETS[@]}"; do
  python scripts/train.py \
    --dataset "$dataset" \
    --model rbgnn \
    --epochs 200 \
    --batch-size 4 \
    --hidden-dim 128 \
    --num-layers 6
done

