"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.9,
    gamma: float = 4.0,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary focal loss on logits.

    `targets` must contain 0/1 labels with the same shape as `logits`.
    """

    targets = targets.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t).clamp(min=1e-8) ** gamma) * loss

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=loss.device)
        loss = loss[mask]

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    if loss.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return loss.mean()

