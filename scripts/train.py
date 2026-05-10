#!/usr/bin/env python3
"""Train RB-GNN or a neural baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from rbgnn.datasets import RumorBlockingDataset
from rbgnn.factory import build_model
from rbgnn.training import TrainConfig, fit


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open() as fp:
        return yaml.safe_load(fp) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", default="trigger/er1500")
    parser.add_argument("--model", default="rbgnn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--aggregation", default=None, choices=["attention", "sum", "mean", "max"])
    parser.add_argument("--jumping", default=None, choices=["last", "lstm", "max"])
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask-all-prob", type=float, default=None)
    parser.add_argument("--mask-partial-ratio", type=float, default=None)
    return parser.parse_args()


def merged_options(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("training", {}))
    data_cfg = dict(config.get("data", {}))

    for key in ("dataset", "data_root", "train_limit", "validation_limit", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            data_cfg[key] = value
    for key in ("model", "hidden_dim", "num_layers", "dropout", "aggregation", "jumping"):
        value = getattr(args, key, None)
        if value is not None:
            model_cfg[key] = value
    if args.no_reverse:
        model_cfg["use_reverse"] = False
    for key in (
        "epochs",
        "batch_size",
        "lr",
        "device",
        "checkpoint",
        "mask_all_prob",
        "mask_partial_ratio",
    ):
        value = getattr(args, key, None)
        if value is not None:
            train_cfg[key] = value

    return {"data": data_cfg, "model": model_cfg, "training": train_cfg}


def main() -> None:
    options = merged_options(parse_args())
    data_cfg = options["data"]
    model_cfg = options["model"]
    train_cfg = options["training"]

    model_name = model_cfg.get("model", "rbgnn")
    is_mask = model_name.lower() == "rbgnn-mask"
    input_dim = 4 if is_mask else 3

    train_dataset = RumorBlockingDataset.from_registry(
        data_cfg.get("data_root", ROOT / "data"),
        data_cfg.get("dataset", "trigger/er1500"),
        "train",
        limit=data_cfg.get("train_limit"),
        include_protector_indicator=is_mask,
        mask_training=is_mask,
        mask_all_prob=train_cfg.get("mask_all_prob", 0.1),
        mask_partial_ratio=train_cfg.get("mask_partial_ratio", 0.7),
        seed=data_cfg.get("seed", 123),
    )
    validation_dataset = RumorBlockingDataset.from_registry(
        data_cfg.get("data_root", ROOT / "data"),
        data_cfg.get("dataset", "trigger/er1500"),
        "validation",
        limit=data_cfg.get("validation_limit"),
        include_protector_indicator=is_mask,
        mask_training=False,
        seed=data_cfg.get("seed", 123),
    )

    model = build_model(
        model_name,
        input_dim=input_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 6 if model_name.lower().startswith("rbgnn") else 2),
        dropout=model_cfg.get("dropout", 0.3),
        aggregation=model_cfg.get("aggregation", "attention"),
        jumping=model_cfg.get("jumping", "last"),
        use_reverse=model_cfg.get("use_reverse", True),
    )
    config = TrainConfig(
        epochs=train_cfg.get("epochs", 200),
        batch_size=train_cfg.get("batch_size", 4),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.9),
        focal_gamma=train_cfg.get("focal_gamma", 4.0),
        device=train_cfg.get("device", "auto"),
        num_workers=train_cfg.get("num_workers", 0),
    )

    checkpoint = Path(
        train_cfg.get(
            "checkpoint",
            ROOT / "checkpoints" / f"{data_cfg.get('dataset', 'dataset').replace('/', '_')}_{model_name}.pt",
        )
    )
    result = fit(
        model,
        train_dataset,
        validation_dataset,
        config,
        checkpoint_path=checkpoint,
        use_mask_iterative=is_mask,
    )

    payload = torch.load(checkpoint, map_location="cpu") if checkpoint.exists() else {}
    payload.update(
        {
            "model_name": model_name,
            "model_config": {
                "input_dim": input_dim,
                "hidden_dim": model_cfg.get("hidden_dim", 128),
                "num_layers": model_cfg.get(
                    "num_layers", 6 if model_name.lower().startswith("rbgnn") else 2
                ),
                "dropout": model_cfg.get("dropout", 0.3),
                "aggregation": model_cfg.get("aggregation", "attention"),
                "jumping": model_cfg.get("jumping", "last"),
                "use_reverse": model_cfg.get("use_reverse", True),
            },
            "dataset": data_cfg.get("dataset", "trigger/er1500"),
            "data_root": str(data_cfg.get("data_root", ROOT / "data")),
        }
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)

    history_path = ROOT / "outputs" / f"{checkpoint.stem}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(result["history"], indent=2))

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "history": str(history_path),
                "best_validation_recall": result["best_validation_recall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
