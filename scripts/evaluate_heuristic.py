#!/usr/bin/env python3
"""Evaluate high-degree or neighbor heuristic selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbgnn.data import read_diffusion_graph, read_pair_file, resolve_dataset_entry
from rbgnn.metrics import summarize_predictions
from rbgnn.models.heuristics import HeuristicSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", default="trigger/er1500")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--method", default="high_degree", choices=["high_degree", "neighbors"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = resolve_dataset_entry(args.data_root, args.dataset)
    graph = read_diffusion_graph(entry["graph"], num_nodes=entry["num_nodes"], directed=entry["directed"])
    samples = read_pair_file(entry[args.split])
    if args.limit is not None:
        samples = samples[: args.limit]

    selector = HeuristicSelector(args.method, seed=args.seed)
    predictions = [
        selector.select(graph, sample.rumor_seeds, budget=len(sample.protectors))
        for sample in samples
    ]
    labels = [sample.protectors for sample in samples]
    metrics = summarize_predictions(predictions, labels)
    print(json.dumps({"dataset": args.dataset, "split": args.split, **metrics.__dict__}, indent=2))


if __name__ == "__main__":
    main()
