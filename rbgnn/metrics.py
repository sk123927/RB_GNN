"""Evaluation metrics for supervised protector prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class PredictionMetrics:
    samples: int
    recall_at_budget: float
    precision_at_budget: float
    mean_budget: float


def recall_at_budget(predicted: Iterable[int], gold: Iterable[int]) -> float:
    gold_set = set(int(node) for node in gold)
    if not gold_set:
        return 0.0
    pred_set = set(int(node) for node in predicted)
    return len(pred_set & gold_set) / len(gold_set)


def precision_at_budget(predicted: Iterable[int], gold: Iterable[int]) -> float:
    pred_list = list(int(node) for node in predicted)
    if not pred_list:
        return 0.0
    gold_set = set(int(node) for node in gold)
    return len(set(pred_list) & gold_set) / len(pred_list)


def summarize_predictions(predictions: List[Iterable[int]], labels: List[Iterable[int]]) -> PredictionMetrics:
    recalls = [recall_at_budget(pred, gold) for pred, gold in zip(predictions, labels)]
    precisions = [precision_at_budget(pred, gold) for pred, gold in zip(predictions, labels)]
    budgets = [len(list(gold)) for gold in labels]
    count = len(labels)
    return PredictionMetrics(
        samples=count,
        recall_at_budget=sum(recalls) / max(count, 1),
        precision_at_budget=sum(precisions) / max(count, 1),
        mean_budget=sum(budgets) / max(count, 1),
    )

