"""Custom RB-GNN layers."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class DeepSetAttentionConv(MessagePassing):
    """Deep Sets style message passing with optional neighbor attention.

    One layer has the paper form

    h_v' = rho(sum_{u in N(v) union {v}} alpha_uv * phi(h_u)).

    `phi` is a linear map, `rho` is a two-layer MLP, and `alpha` can be learned
    attention, uniform sum, mean, or max aggregation for ablation studies.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        aggregation: str = "attention",
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        if aggregation not in {"attention", "sum", "mean", "max"}:
            raise ValueError("aggregation must be one of: attention, sum, mean, max")
        aggr = "add" if aggregation in {"attention", "sum"} else aggregation
        super().__init__(aggr=aggr, node_dim=0)

        self.aggregation = aggregation
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.phi = nn.Linear(in_channels, out_channels)
        self.attention = nn.Parameter(torch.empty(2 * out_channels, 1))
        self.rho = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.phi.weight)
        nn.init.zeros_(self.phi.bias)
        nn.init.xavier_uniform_(self.attention)
        for module in self.rho:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.phi(x)
        aggregated = self.propagate(edge_index, x=x)
        return self.rho(aggregated)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        if self.aggregation != "attention":
            return x_j

        pair = torch.cat([x_i, x_j], dim=-1)
        score = F.leaky_relu(pair @ self.attention, negative_slope=self.negative_slope)
        alpha = softmax(score, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

