"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from rbgnn.losses import sigmoid_focal_loss
from rbgnn.metrics import summarize_predictions
from rbgnn.models.rbgnn import RBGNNMask


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    focal_alpha: float = 0.9
    focal_gamma: float = 4.0
    device: str = "auto"
    num_workers: int = 0


def choose_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, config: TrainConfig, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = sigmoid_focal_loss(
            logits,
            batch.y,
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            mask=getattr(batch, "loss_mask", None),
        )
        loss.backward()
        optimizer.step()

        examples = int(batch.y.numel())
        total_loss += float(loss.detach().cpu()) * examples
        total_examples += examples
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_predictor(model, dataset, device: torch.device, *, use_mask_iterative: bool = False) -> Dict:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    labels = []
    model.eval()
    for data in loader:
        data = data.to(device)
        if use_mask_iterative and isinstance(model, RBGNNMask):
            selected, _ = model.predict_iterative(data)
        else:
            selected, _ = model.predict_topk(data)
        predictions.append(selected.detach().cpu().tolist())
        labels.append(data.protector_seed_index.view(-1).detach().cpu().tolist())
    return asdict(summarize_predictions(predictions, labels))


def fit(
    model,
    train_dataset,
    validation_dataset,
    config: TrainConfig,
    *,
    checkpoint_path: Optional[Path] = None,
    use_mask_iterative: bool = False,
) -> Dict:
    device = choose_device(config.device)
    model = model.to(device)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_metric = -1.0
    best_state = None
    history = []
    progress = tqdm(range(1, config.epochs + 1), desc="training", leave=False)
    for epoch in progress:
        loss = train_one_epoch(model, loader, optimizer, config, device)
        validation = evaluate_predictor(
            model,
            validation_dataset,
            device,
            use_mask_iterative=use_mask_iterative,
        )
        metric = validation["recall_at_budget"]
        history.append({"epoch": epoch, "loss": loss, **validation})
        progress.set_postfix(loss=f"{loss:.4f}", recall=f"{metric:.4f}")

        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": best_state,
                        "epoch": epoch,
                        "validation": validation,
                        "train_config": asdict(config),
                    },
                    checkpoint_path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_validation_recall": best_metric, "history": history}

