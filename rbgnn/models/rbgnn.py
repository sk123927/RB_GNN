"""RB-GNN and RB-GNN-Mask."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge

from rbgnn.models.layers import DeepSetAttentionConv


class _ResidualBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        *,
        dropout: float,
        aggregation: str,
        jumping: str,
    ) -> None:
        super().__init__()
        if jumping not in {"last", "lstm", "max"}:
            raise ValueError("jumping must be one of: last, lstm, max")

        self.dropout = dropout
        self.jumping = jumping
        self.convs = nn.ModuleList(
            [
                DeepSetAttentionConv(
                    hidden_dim,
                    hidden_dim,
                    aggregation=aggregation,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.jump = (
            JumpingKnowledge(mode=jumping, channels=hidden_dim, num_layers=num_layers)
            if jumping != "last"
            else None
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        layer_outputs = []
        for norm, conv in zip(self.norms, self.convs):
            update = norm(h)
            update = F.relu(update)
            update = F.dropout(update, p=self.dropout, training=self.training)
            update = conv(update, edge_index)
            h = h + update
            layer_outputs.append(h)
        if self.jump is None:
            return layer_outputs[-1]
        return self.jump(layer_outputs)


class RBGNN(nn.Module):
    """Rumor Blocking Graph Neural Network.

    The model has a forward branch on the original directed graph, a reverse
    branch on the transposed graph, and a scalar gated fusion module.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
        *,
        dropout: float = 0.3,
        aggregation: str = "attention",
        jumping: str = "last",
        use_reverse: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_reverse = use_reverse

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.forward_branch = _ResidualBranch(
            hidden_dim,
            num_layers,
            dropout=dropout,
            aggregation=aggregation,
            jumping=jumping,
        )
        if use_reverse:
            self.reverse_branch = _ResidualBranch(
                hidden_dim,
                num_layers,
                dropout=dropout,
                aggregation=aggregation,
                jumping=jumping,
            )
            self.gate = nn.Linear(2 * hidden_dim, 1)
        else:
            self.reverse_branch = None
            self.gate = None
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        x = self.encoder(data.x)
        forward = self.forward_branch(x, data.edge_index)

        if self.reverse_branch is not None:
            reverse_edge_index = getattr(data, "reversed_edge_index", None)
            if reverse_edge_index is None:
                reverse_edge_index = torch.flip(data.edge_index, dims=[0])
            reverse = self.reverse_branch(x.clone(), reverse_edge_index)
            gamma = torch.sigmoid(self.gate(torch.cat([forward, reverse], dim=-1)))
            fused = gamma * forward + (1.0 - gamma) * reverse
        else:
            fused = forward

        return self.decoder(fused).squeeze(-1)

    @torch.no_grad()
    def predict_topk(
        self,
        data,
        *,
        budget: Optional[int] = None,
        extra_exclude: Optional[Iterable[int] | torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(data).clone()
        if budget is None:
            budget = int(getattr(data, "protector_seed_index").numel())

        exclude = [int(node) for node in data.rumor_seed_index.view(-1).tolist()]
        if extra_exclude is not None:
            if isinstance(extra_exclude, torch.Tensor):
                exclude.extend(int(node) for node in extra_exclude.view(-1).tolist())
            else:
                exclude.extend(int(node) for node in extra_exclude)
        if exclude:
            logits[torch.tensor(exclude, dtype=torch.long, device=logits.device)] = -torch.inf

        selected = torch.topk(logits, k=budget).indices
        indicator = torch.zeros_like(logits, dtype=torch.long)
        indicator[selected] = 1
        return selected, indicator


class RBGNNMask(RBGNN):
    """RB-GNN with an observed-protector indicator and iterative inference."""

    def __init__(self, *args, input_dim: int = 4, **kwargs) -> None:
        super().__init__(*args, input_dim=input_dim, **kwargs)

    @torch.no_grad()
    def predict_iterative(self, data, *, budget: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        if budget is None:
            budget = int(getattr(data, "protector_seed_index").numel())

        work = data.clone()
        if work.x.size(-1) == 3:
            zeros = torch.zeros((work.x.size(0), 1), dtype=work.x.dtype, device=work.x.device)
            work.x = torch.cat([work.x, zeros], dim=-1)
        if work.x.size(-1) != self.input_dim:
            raise ValueError(f"RBGNNMask expects {self.input_dim} input features.")

        selected = []
        for _ in range(budget):
            logits = self.forward(work).clone()
            excluded = [int(node) for node in work.rumor_seed_index.view(-1).tolist()]
            excluded.extend(selected)
            if excluded:
                logits[torch.tensor(excluded, dtype=torch.long, device=logits.device)] = -torch.inf
            node = int(torch.argmax(logits).item())
            selected.append(node)
            work.x[node, -1] = 1.0

        selected_tensor = torch.tensor(selected, dtype=torch.long, device=work.x.device)
        indicator = torch.zeros(work.x.size(0), dtype=torch.long, device=work.x.device)
        indicator[selected_tensor] = 1
        return selected_tensor, indicator

