#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-trigger/er1500}"
MODELS=(mlp gcn gin gat graph_transformer appnp gcnii rgcn)

for model in "${MODELS[@]}"; do
  layers=2
  if [[ "$model" == "gcnii" ]]; then
    layers=6
  fi
  python scripts/train.py \
    --dataset "$DATASET" \
    --model "$model" \
    --epochs 200 \
    --batch-size 4 \
    --hidden-dim 128 \
    --num-layers "$layers"
done

