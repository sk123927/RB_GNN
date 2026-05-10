"""Model builders.

Neural modules are imported lazily so pure data/heuristic scripts can run without
importing PyTorch.
"""

from rbgnn.models.heuristics import HeuristicSelector

__all__ = [
    "GNNBaseline",
    "HeuristicSelector",
    "RBGNN",
    "RBGNNMask",
    "build_baseline",
]


def __getattr__(name):
    if name in {"GNNBaseline", "build_baseline"}:
        from rbgnn.models.baselines import GNNBaseline, build_baseline

        return {"GNNBaseline": GNNBaseline, "build_baseline": build_baseline}[name]
    if name in {"RBGNN", "RBGNNMask"}:
        from rbgnn.models.rbgnn import RBGNN, RBGNNMask

        return {"RBGNN": RBGNN, "RBGNNMask": RBGNNMask}[name]
    raise AttributeError(name)
