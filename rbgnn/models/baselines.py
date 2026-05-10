"""Neural and heuristic baselines used in the RB-GNN experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GCN2Conv,
    GCNConv,
    GINConv,
    RGCNConv,
    TransformerConv,
)

from rbgnn.models.heuristics import HeuristicSelector


class GNNBaseline(nn.Module):
    """Shared implementation for MLP, GCN, GIN, GAT, Transformer, APPNP, GCNII, R-GCN."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        *,
        backbone: str = "gcn",
        dropout: float = 0.3,
        appnp_k: int = 4,
        appnp_alpha: float = 0.2,
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone.lower()
        self.dropout = dropout
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layers = nn.ModuleList()

        if self.backbone == "mlp":
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        elif self.backbone == "gcn":
            self.layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        elif self.backbone == "gin":
            self.layers = nn.ModuleList(
                [
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        train_eps=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif self.backbone == "gat":
            self.layers = nn.ModuleList(
                [
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif self.backbone == "graph_transformer":
            self.layers = nn.ModuleList(
                [
                    TransformerConv(
                        hidden_dim,
                        hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif self.backbone == "appnp":
            self.prelogit = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.propagation = APPNP(K=appnp_k, alpha=appnp_alpha, dropout=0.0)
        elif self.backbone == "gcnii":
            self.layers = nn.ModuleList(
                [
                    GCN2Conv(
                        channels=hidden_dim,
                        alpha=gcnii_alpha,
                        theta=gcnii_theta,
                        layer=layer + 1,
                        shared_weights=False,
                    )
                    for layer in range(num_layers)
                ]
            )
        elif self.backbone == "rgcn":
            self.layers = nn.ModuleList()
            self.layers.append(RGCNConv(input_dim, hidden_dim, num_relations=2))
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=2))
        else:
            raise ValueError(
                "Unknown backbone. Choose from mlp, gcn, gin, gat, graph_transformer, "
                "appnp, gcnii, rgcn."
            )

    def forward(self, data) -> torch.Tensor:
        if self.backbone == "appnp":
            logits = self.prelogit(data.x).squeeze(-1)
            return self.propagation(logits.unsqueeze(-1), data.edge_index).squeeze(-1)

        if self.backbone == "rgcn":
            reverse_edge_index = getattr(data, "reversed_edge_index", torch.flip(data.edge_index, dims=[0]))
            edge_index = torch.cat([data.edge_index, reverse_edge_index], dim=1)
            edge_type = torch.cat(
                [
                    torch.zeros(data.edge_index.size(1), dtype=torch.long, device=data.edge_index.device),
                    torch.ones(reverse_edge_index.size(1), dtype=torch.long, device=data.edge_index.device),
                ]
            )
            h = F.relu(self.layers[0](data.x, edge_index, edge_type))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[1](h, edge_index, edge_type)
            return self.decoder(h).squeeze(-1)

        h = self.encoder(data.x)
        h0 = h
        for index, layer in enumerate(self.layers):
            if self.backbone == "gcnii":
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = layer(h, h0, data.edge_index)
                h = F.relu(h)
                h = self.norms[index](h)
                continue

            residual = h
            h = self.norms[index](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.backbone == "mlp":
                h = layer(h)
            else:
                h = layer(h, data.edge_index)
            h = h + residual
        return self.decoder(h).squeeze(-1)

    @torch.no_grad()
    def predict_topk(self, data, *, budget: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(data).clone()
        if budget is None:
            budget = int(data.protector_seed_index.numel())
        logits[data.rumor_seed_index] = -torch.inf
        selected = torch.topk(logits, k=budget).indices
        indicator = torch.zeros_like(logits, dtype=torch.long)
        indicator[selected] = 1
        return selected, indicator


def build_baseline(name: str, **kwargs) -> GNNBaseline:
    return GNNBaseline(backbone=name, **kwargs)


