#!/usr/bin/env python3
"""Print dataset statistics without requiring PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbgnn.data import read_diffusion_graph, read_pair_file, resolve_dataset_entry, validate_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", default="trigger/er1500")
    return parser.parse_args()


def summarize_pair_file(path: str, num_nodes: int) -> dict:
    samples = read_pair_file(path)
    validate_samples(samples, num_nodes)
    budgets = [sample.budget for sample in samples]
    return {
        "samples": len(samples),
        "min_budget": min(budgets) if budgets else 0,
        "max_budget": max(budgets) if budgets else 0,
        "mean_budget": sum(budgets) / max(len(budgets), 1),
    }


def main() -> None:
    args = parse_args()
    entry = resolve_dataset_entry(args.data_root, args.dataset)
    graph = read_diffusion_graph(entry["graph"], num_nodes=entry["num_nodes"], directed=entry["directed"])
    result = {
        "dataset": args.dataset,
        "name": entry["name"],
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "edge_attributes": graph.attribute_names,
        "splits": {
            split: summarize_pair_file(entry[split], graph.num_nodes)
            for split in ("train", "validation", "test")
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

