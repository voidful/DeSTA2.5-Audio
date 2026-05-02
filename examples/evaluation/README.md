# Evaluation Scripts Configuration Guide

## Overview

The active model path now supports the groupwise-orthogonal Q-Former connector
only. Older experimental evaluation flows are preserved under
`examples/evaluation/legacy/`.

## Expected Configuration

```yaml
connector:
  mode: groupwise_ortho

groupwise_ortho:
  num_groups: 8
  queries_per_group: 8
  inter_group_weight: 0.1
  intra_group_weight: 0.01
```

The legacy aliases `orca_desta` and `orca_r1` still load as
`groupwise_ortho` for checkpoint compatibility.

## Usage

Use the active evaluation scripts in this directory for current checkpoints.
For older experiments, run the scripts from `examples/evaluation/legacy/`.

```bash
cd examples/evaluation
python mmau_pro_eval.py --model_id /path/to/your/checkpoint
```

## Troubleshooting

```python
from desta.models.modeling_desta25 import DeSTA25AudioModel

model = DeSTA25AudioModel.from_pretrained("your_model_id")
print(f"Connector mode: {model.config.connector_mode}")
print(f"Groups: {model.config.orca_r1_num_groups}")
print(f"Queries/group: {model.config.orca_r1_queries_per_group}")
```
