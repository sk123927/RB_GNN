"""Pure-Python heuristic protector selectors."""

from __future__ import annotations

import random
from typing import Iterable, List


class HeuristicSelector:
    """Non-neural protector selectors: high-degree and neighbor candidates."""

    def __init__(self, method: str = "high_degree", seed: int = 123) -> None:
        if method not in {"high_degree", "neighbors"}:
            raise ValueError("method must be one of: high_degree, neighbors")
        self.method = method
        self.rng = random.Random(seed)

    def select(self, graph, rumor_seeds: Iterable[int], budget: int) -> List[int]:
        rumor_set = set(int(node) for node in rumor_seeds)
        if self.method == "high_degree":
            out_degree = graph.out_degrees()
            candidates = [node for node in range(graph.num_nodes) if node not in rumor_set]
            return sorted(candidates, key=lambda node: (-out_degree[node], node))[:budget]

        neighbors = graph.out_neighbors()
        candidate_set = set()
        for rumor in rumor_set:
            candidate_set.update(neighbors[rumor])
        candidate_set.difference_update(rumor_set)
        candidates = sorted(candidate_set)
        self.rng.shuffle(candidates)

        if len(candidates) < budget:
            fallback = [node for node in range(graph.num_nodes) if node not in rumor_set | set(candidates)]
            candidates.extend(fallback)
        return candidates[:budget]

