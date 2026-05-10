#!/usr/bin/env python3
"""Evaluate a trained neural model by protector recall at the gold budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rbgnn.datasets import RumorBlockingDataset
from rbgnn.factory import build_model
from rbgnn.training import choose_device, evaluate_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    model_config = dict(checkpoint["model_config"])
    dataset_key = args.dataset or checkpoint.get("dataset", "trigger/er1500")
    is_mask = model_name.lower() == "rbgnn-mask"

    dataset = RumorBlockingDataset.from_registry(
        args.data_root,
        dataset_key,
        args.split,
        limit=args.limit,
        include_protector_indicator=is_mask,
        mask_training=False,
    )
    model = build_model(model_name, **model_config)
    model.load_state_dict(checkpoint["model_state"])

    device = choose_device(args.device)
    model = model.to(device)
    metrics = evaluate_predictor(model, dataset, device, use_mask_iterative=is_mask)
    print(json.dumps({"dataset": dataset_key, "split": args.split, **metrics}, indent=2))


if __name__ == "__main__":
    main()

