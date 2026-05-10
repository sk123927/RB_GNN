# RB-GNN

Clean research code for **RB-GNN: Rumor Blocking without Knowing the Diffusion Model**.

This repository is a curated implementation of the paper codebase. It keeps the
research idea, data protocol, models, and experiment scripts, while removing old
experiments, backup files, hard-coded machine paths, and deprecated feature branches.

## What Is Included

- `rbgnn/`: reusable Python package
  - canonical data parser for rumor/protector pair files
  - PyTorch Geometric dataset wrapper
  - RB-GNN and RB-GNN-Mask
  - neural baselines: MLP, GCN, GIN, GAT, GraphTransformer, APPNP, GCNII, R-GCN
  - heuristic baselines: high-degree and neighbor selection
- `data/trigger/`: four bundled Triggering-model datasets
  - ER1500
  - SF1750
  - Wiki-Vote
  - DBLP
- `scripts/`: training, evaluation, heuristic evaluation, and dataset inspection
- `configs/`: ready-to-run experiment configs
- `docs/`: data format, model notes, and reproduction guide
- `tests/`: parser-level tests that run without GPU or PyTorch

## Data Format

All supervised neural training reads the same 5-line block format:

```text
237 525 617 ...
1583 1608 1619 ...
5416.000000
3214.000000

```

Line 1 is the rumor seed set. Line 2 is the gold protector set. Line 3 is the
rumor-only influence. Line 4 is the influence after using the gold protectors.
Line 5 is a blank separator.

Diffusion-model files are directed edge lists. The first two columns are source
and target node ids; the remaining columns are preserved as edge attributes.

## Installation

Create an environment with PyTorch and PyTorch Geometric installed for your
hardware, then install this package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

For CPU-only smoke tests, install CPU PyTorch following the official PyTorch
instructions, then install `torch-geometric`.

## Quick Checks

Inspect the bundled ER dataset without importing PyTorch:

```bash
python scripts/inspect_dataset.py --dataset trigger/er1500
```

Run parser tests:

```bash
python -m unittest discover -s tests
```

## Train RB-GNN

```bash
python scripts/train.py --config configs/rbgnn_er1500.yaml
```

Train RB-GNN-Mask:

```bash
python scripts/train.py --config configs/rbgnn_mask_er1500.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/trigger_er1500_rbgnn.pt \
  --dataset trigger/er1500 \
  --split test
```

Run heuristic baselines:

```bash
python scripts/evaluate_heuristic.py --dataset trigger/er1500 --method high_degree
python scripts/evaluate_heuristic.py --dataset trigger/er1500 --method neighbors
```

## Paper-Default Hyperparameters

- Node features: rumor indicator, normalized in-degree, normalized out-degree
- RB-GNN/RB-GNN-Mask: 6 forward layers, 6 reverse layers, hidden dimension 128
- Baselines: 2 layers for MLP/GIN/GAT/GraphTransformer/R-GCN/APPNP, 6 layers for GCNII
- Optimizer: AdamW, learning rate 0.001
- Loss: sigmoid focal loss with alpha = 0.9 and gamma = 4
- Batch size: 4
- Epochs: 200

## Repository Design

The code is intentionally boring in the best way: one data protocol, one model
factory, one training loop, one evaluation path. That makes it easier for reviewers,
collaborators, and hiring committees to see the research contribution without
digging through temporary experiments.

