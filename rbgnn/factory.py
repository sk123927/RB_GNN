"""Model factory used by scripts."""

from __future__ import annotations

from rbgnn.models.baselines import GNNBaseline
from rbgnn.models.rbgnn import RBGNN, RBGNNMask


BASELINE_NAMES = {"mlp", "gcn", "gin", "gat", "graph_transformer", "appnp", "gcnii", "rgcn"}


def build_model(
    name: str,
    *,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    aggregation: str = "attention",
    jumping: str = "last",
    use_reverse: bool = True,
):
    model_name = name.lower()
    if model_name == "rbgnn":
        return RBGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregation=aggregation,
            jumping=jumping,
            use_reverse=use_reverse,
        )
    if model_name == "rbgnn-mask":
        return RBGNNMask(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregation=aggregation,
            jumping=jumping,
            use_reverse=use_reverse,
        )
    if model_name in BASELINE_NAMES:
        return GNNBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            backbone=model_name,
        )
    choices = ", ".join(["rbgnn", "rbgnn-mask", *sorted(BASELINE_NAMES)])
    raise ValueError(f"Unknown model '{name}'. Choices: {choices}")

