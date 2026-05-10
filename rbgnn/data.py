"""File-format utilities for RB-GNN datasets.

This module intentionally has no PyTorch dependency. It defines the canonical
data protocol used by all neural models and baselines in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RumorProtectorSample:
    """One supervised rumor-blocking sample."""

    rumor_seeds: List[int]
    protectors: List[int]
    rumor_influence: float
    protected_influence: float

    @property
    def budget(self) -> int:
        return len(self.protectors)


@dataclass(frozen=True)
class DiffusionEdge:
    """Directed edge with optional diffusion attributes."""

    source: int
    target: int
    attributes: Tuple[float, ...] = ()


@dataclass
class DiffusionGraph:
    """Lightweight directed graph representation shared by loaders and baselines."""

    num_nodes: int
    edges: List[DiffusionEdge]
    directed: bool = True
    attribute_names: Tuple[str, ...] = ()

    def out_neighbors(self) -> List[List[int]]:
        neighbors = [[] for _ in range(self.num_nodes)]
        for edge in self.edges:
            neighbors[edge.source].append(edge.target)
        return neighbors

    def in_degrees(self) -> List[int]:
        degrees = [0] * self.num_nodes
        for edge in self.edges:
            degrees[edge.target] += 1
        return degrees

    def out_degrees(self) -> List[int]:
        degrees = [0] * self.num_nodes
        for edge in self.edges:
            degrees[edge.source] += 1
        return degrees


def parse_int_list(line: str) -> List[int]:
    """Parse a space-separated list of node ids."""

    stripped = line.strip()
    if not stripped:
        return []
    return [int(item) for item in stripped.split()]


def read_pair_file(path: Path | str) -> List[RumorProtectorSample]:
    """Read 5-line rumor/protector blocks.

    The parser ignores blank separator lines, so the final blank line is optional.
    Every sample must contribute four non-empty records: rumor seeds, protectors,
    rumor-only influence, and influence after applying the gold protectors.
    """

    path = Path(path)
    records = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(records) % 4 != 0:
        raise ValueError(
            f"{path} has {len(records)} non-empty lines; expected a multiple of 4."
        )

    samples: List[RumorProtectorSample] = []
    for offset in range(0, len(records), 4):
        rumor_seeds = parse_int_list(records[offset])
        protectors = parse_int_list(records[offset + 1])
        samples.append(
            RumorProtectorSample(
                rumor_seeds=rumor_seeds,
                protectors=protectors,
                rumor_influence=float(records[offset + 2]),
                protected_influence=float(records[offset + 3]),
            )
        )
    return samples


def read_diffusion_graph(
    path: Path | str,
    num_nodes: Optional[int] = None,
    directed: bool = True,
) -> DiffusionGraph:
    """Read a diffusion-model edge list.

    The first two columns are `source target`. Any remaining columns are stored as
    floating-point edge attributes. For the bundled Triggering-model datasets the
    attribute columns are `weibull_scale weibull_shape propagation_probability`.
    """

    path = Path(path)
    edges: List[DiffusionEdge] = []
    max_node = -1
    attr_count: Optional[int] = None

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number} has fewer than two columns.")

        source, target = int(parts[0]), int(parts[1])
        attrs = tuple(float(item) for item in parts[2:])
        if attr_count is None:
            attr_count = len(attrs)
        elif attr_count != len(attrs):
            raise ValueError(
                f"{path}:{line_number} has {len(attrs)} attributes; expected {attr_count}."
            )

        max_node = max(max_node, source, target)
        edges.append(DiffusionEdge(source=source, target=target, attributes=attrs))

    inferred_nodes = max_node + 1
    if num_nodes is None:
        num_nodes = inferred_nodes
    elif inferred_nodes > num_nodes:
        raise ValueError(
            f"{path} references node {max_node}, but registry declares {num_nodes} nodes."
        )

    if attr_count == 3:
        names = ("weibull_scale", "weibull_shape", "propagation_probability")
    else:
        names = tuple(f"edge_attr_{i}" for i in range(attr_count or 0))
    return DiffusionGraph(num_nodes=num_nodes, edges=edges, directed=directed, attribute_names=names)


def load_registry(data_root: Path | str) -> Dict[str, dict]:
    """Load `data/registry.json`."""

    data_root = Path(data_root)
    with (data_root / "registry.json").open() as fp:
        return json.load(fp)


def resolve_dataset_entry(data_root: Path | str, dataset: str) -> dict:
    """Resolve one dataset entry from the registry and convert paths to absolute paths."""

    data_root = Path(data_root)
    registry = load_registry(data_root)
    if dataset not in registry:
        choices = ", ".join(sorted(registry))
        raise KeyError(f"Unknown dataset '{dataset}'. Available datasets: {choices}")

    entry = dict(registry[dataset])
    for key in ("graph", "train", "validation", "test"):
        entry[key] = str(data_root / entry[key])
    return entry


def validate_samples(samples: Sequence[RumorProtectorSample], num_nodes: int) -> None:
    """Validate sample node ids.

    Some generated files contain a few samples where the number of rumor seeds and
    gold protectors differ. That is valid for supervised learning: the protector
    line defines the prediction budget.
    """

    for sample_index, sample in enumerate(samples):
        for node in [*sample.rumor_seeds, *sample.protectors]:
            if node < 0 or node >= num_nodes:
                raise ValueError(
                    f"Sample {sample_index} references node {node}, outside [0, {num_nodes})."
                )


def iter_sample_blocks(samples: Iterable[RumorProtectorSample]) -> Iterable[str]:
    """Serialize samples back to the canonical 5-line block format."""

    for sample in samples:
        yield " ".join(str(node) for node in sample.rumor_seeds)
        yield " ".join(str(node) for node in sample.protectors)
        yield f"{sample.rumor_influence:.6f}"
        yield f"{sample.protected_influence:.6f}"
        yield ""
