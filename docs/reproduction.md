# Reproduction Guide

## Dataset Inspection

```bash
python scripts/inspect_dataset.py --dataset trigger/er1500
python scripts/inspect_dataset.py --dataset trigger/sf1750
python scripts/inspect_dataset.py --dataset trigger/wiki7115
python scripts/inspect_dataset.py --dataset trigger/dblp12591
```

## Main RB-GNN Runs

```bash
python scripts/train.py --dataset trigger/er1500 --model rbgnn --epochs 200
python scripts/train.py --dataset trigger/sf1750 --model rbgnn --epochs 200
python scripts/train.py --dataset trigger/wiki7115 --model rbgnn --epochs 200
python scripts/train.py --dataset trigger/dblp12591 --model rbgnn --epochs 200
```

Or:

```bash
bash scripts/run_all_trigger.sh
```

## Baselines

```bash
bash scripts/run_baselines.sh trigger/er1500
```

Heuristic selectors do not require PyTorch:

```bash
python scripts/evaluate_heuristic.py --dataset trigger/er1500 --method high_degree
python scripts/evaluate_heuristic.py --dataset trigger/er1500 --method neighbors
```

## Notes

The original paper reports blocking effect using Monte Carlo diffusion simulation.
This clean repository exposes recall-at-budget as the default fast supervised metric.
The pair files include rumor-only and gold-protector influence values so an
influence simulator can be added without changing the training data format.

