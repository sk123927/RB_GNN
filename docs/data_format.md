# Data Format

## Pair Files

Every model and baseline consumes the same supervised sample format. One sample
is stored as:

```text
rumor_seed_1 rumor_seed_2 ...
gold_protector_1 gold_protector_2 ...
rumor_only_influence
gold_protector_influence

```

The blank separator is optional for the final sample. The loader ignores blank
lines and parses every four non-empty lines as one sample.

The number of gold protectors is the selection budget. Most generated samples use
the same number of rumor seeds and protectors, but the loader does not require
that equality; the second line is authoritative for training and evaluation.

## Diffusion-Model Files

The bundled Triggering-model diffusion files use:

```text
source target weibull_scale weibull_shape propagation_probability
```

The learning models use the first two columns as the graph structure. Extra
columns are kept as `edge_attr` so future diffusion-aware experiments can use them
without changing the parser.

## Registry

`data/registry.json` declares:

- display name
- number of nodes
- graph path
- train/validation/test pair paths

The explicit node count is important because edge lists can omit isolated nodes.
