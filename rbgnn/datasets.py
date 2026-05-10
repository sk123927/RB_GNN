"""PyTorch Geometric datasets for RB-GNN."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable, Optional, Sequence, Set

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from rbgnn.data import (
    DiffusionGraph,
    RumorProtectorSample,
    read_diffusion_graph,
    read_pair_file,
    resolve_dataset_entry,
    validate_samples,
)


class RumorBlockingDataset(Dataset):
    """Build one PyG `Data` object per rumor/protector sample.

    Node features are deliberately simple and match the paper:

    1. rumor-seed indicator
    2. normalized in-degree
    3. normalized out-degree
    4. optional observed-protector indicator for RB-GNN-Mask
    """

    def __init__(
        self,
        graph: DiffusionGraph,
        samples: Sequence[RumorProtectorSample],
        *,
        include_protector_indicator: bool = False,
        mask_training: bool = False,
        mask_all_prob: float = 0.1,
        mask_partial_ratio: float = 0.7,
        seed: int = 123,
    ) -> None:
        self.graph = graph
        self.samples = list(samples)
        self.include_protector_indicator = include_protector_indicator
        self.mask_training = mask_training
        self.mask_all_prob = mask_all_prob
        self.mask_partial_ratio = mask_partial_ratio
        self.rng = random.Random(seed)

        validate_samples(self.samples, self.graph.num_nodes)

        self.edge_index = self._build_edge_index(self.graph)
        self.reversed_edge_index = torch.flip(self.edge_index, dims=[0])
        self.edge_attr = self._build_edge_attr(self.graph)
        self.degree_features = self._build_degree_features(self.graph)

    @classmethod
    def from_registry(
        cls,
        data_root: Path | str,
        dataset: str,
        split: str,
        *,
        limit: Optional[int] = None,
        include_protector_indicator: bool = False,
        mask_training: bool = False,
        mask_all_prob: float = 0.1,
        mask_partial_ratio: float = 0.7,
        seed: int = 123,
    ) -> "RumorBlockingDataset":
        entry = resolve_dataset_entry(data_root, dataset)
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")

        graph = read_diffusion_graph(
            entry["graph"],
            num_nodes=int(entry["num_nodes"]),
            directed=bool(entry.get("directed", True)),
        )
        samples = read_pair_file(entry[split])
        if limit is not None:
            samples = samples[:limit]

        return cls(
            graph,
            samples,
            include_protector_indicator=include_protector_indicator,
            mask_training=mask_training,
            mask_all_prob=mask_all_prob,
            mask_partial_ratio=mask_partial_ratio,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        sample = self.samples[index]
        observed = self._sample_observed_protectors(sample)
        return self.to_data(sample, sample_id=index, observed_protectors=observed)

    def to_data(
        self,
        sample: RumorProtectorSample,
        *,
        sample_id: int = -1,
        observed_protectors: Optional[Iterable[int]] = None,
    ) -> Data:
        num_nodes = self.graph.num_nodes
        x = torch.zeros((num_nodes, 1), dtype=torch.float32)
        x[sample.rumor_seeds, 0] = 1.0
        x = torch.cat([x, self.degree_features], dim=1)

        if self.include_protector_indicator:
            indicator = torch.zeros((num_nodes, 1), dtype=torch.float32)
            if observed_protectors is not None:
                observed_list = list(observed_protectors)
                if observed_list:
                    indicator[observed_list, 0] = 1.0
            x = torch.cat([x, indicator], dim=1)
            loss_mask = indicator.squeeze(-1) == 0
        else:
            loss_mask = torch.ones(num_nodes, dtype=torch.bool)

        y = torch.zeros(num_nodes, dtype=torch.float32)
        y[sample.protectors] = 1.0

        data = Data(
            x=x,
            y=y,
            edge_index=self.edge_index,
            reversed_edge_index=self.reversed_edge_index,
            edge_attr=self.edge_attr,
            loss_mask=loss_mask,
            rumor_seed_index=torch.tensor(sample.rumor_seeds, dtype=torch.long),
            protector_seed_index=torch.tensor(sample.protectors, dtype=torch.long),
            rumor_influence=torch.tensor([sample.rumor_influence], dtype=torch.float32),
            protected_influence=torch.tensor([sample.protected_influence], dtype=torch.float32),
            num_nodes=num_nodes,
            sample_id=torch.tensor([sample_id], dtype=torch.long),
        )
        return data

    def _sample_observed_protectors(self, sample: RumorProtectorSample) -> Set[int]:
        if not self.include_protector_indicator:
            return set()
        if not self.mask_training:
            return set()
        if self.rng.random() < self.mask_all_prob:
            return set()
        return {
            node
            for node in sample.protectors
            if self.rng.random() > self.mask_partial_ratio
        }

    @staticmethod
    def _build_edge_index(graph: DiffusionGraph) -> torch.Tensor:
        if not graph.edges:
            return torch.empty((2, 0), dtype=torch.long)
        rows = [[edge.source for edge in graph.edges], [edge.target for edge in graph.edges]]
        return torch.tensor(rows, dtype=torch.long)

    @staticmethod
    def _build_edge_attr(graph: DiffusionGraph) -> Optional[torch.Tensor]:
        if not graph.edges or not graph.edges[0].attributes:
            return None
        return torch.tensor([edge.attributes for edge in graph.edges], dtype=torch.float32)

    @staticmethod
    def _build_degree_features(graph: DiffusionGraph) -> torch.Tensor:
        in_degrees = torch.tensor(graph.in_degrees(), dtype=torch.float32)
        out_degrees = torch.tensor(graph.out_degrees(), dtype=torch.float32)
        in_denom = torch.clamp(in_degrees.max(), min=1.0)
        out_denom = torch.clamp(out_degrees.max(), min=1.0)
        return torch.stack([in_degrees / in_denom, out_degrees / out_denom], dim=1)

