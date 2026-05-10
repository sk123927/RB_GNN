# Data

The default datasets are the Triggering-model rumor-protector pairs used by the
RB-GNN experiments.

Each supervised sample is stored as a 5-line block:

```text
rumor_seed_1 rumor_seed_2 ...
gold_protector_1 gold_protector_2 ...
rumor_only_influence
gold_protector_influence

```

The fifth line is blank and separates samples. The loader is tolerant of a missing
final blank line.

Diffusion-model files are directed edge lists. The first two columns are source and
target node ids. Additional columns are preserved as edge attributes; for the
provided Triggering-model files they are:

```text
source target weibull_scale weibull_shape propagation_probability
```

Dataset metadata, including node counts needed for isolated nodes, is defined in
`registry.json`. The protector line defines the prediction budget; it does not need
to have the same length as the rumor-seed line.
