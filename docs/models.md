# Models

## RB-GNN

RB-GNN has three parts:

1. A forward branch over the original directed graph.
2. A reverse branch over the transposed graph.
3. A scalar gated fusion module that combines forward and reverse node states.

Each branch stacks residual DeepSet-attention message-passing layers:

```text
h_v' = rho(sum alpha_uv * phi(h_u))
```

The output is one logit per node. At inference time, rumor seeds are excluded and
the top-k nodes are selected as protectors, where k is the gold budget for the
sample.

## RB-GNN-Mask

RB-GNN-Mask adds an observed-protector indicator as a fourth node feature. During
training, part of the gold protector set is visible and part is masked. Loss is
computed only on nodes whose protector indicator is zero.

At inference time, protectors are selected iteratively:

1. Start with all protector indicators set to zero.
2. Predict one protector.
3. Mark that protector as observed.
4. Repeat until the budget is filled.

## Baselines

The package includes:

- MLP
- GCN
- GIN
- GAT
- GraphTransformer
- APPNP
- GCNII
- R-GCN
- High-Degree heuristic
- Neighbor heuristic

All neural baselines use the same input features and focal-loss training loop as
RB-GNN unless the architecture requires a standard variant, such as APPNP
propagation.

