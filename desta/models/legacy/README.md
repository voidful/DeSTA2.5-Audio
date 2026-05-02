Legacy model experiments
========================

This directory preserves pre-paper experiments that are not part of the
active groupwise-orthogonal method:

- `qformer_1` baseline connector
- `orca_hybrid`
- stochastic / variational grouping experiments
- ACP / Modality-DPO experiments
- deep-injection / gated cross-attention experiments
- Acoustic-Contrastive Decoding (ACD)
- auxiliary alignment and erasure probes

The main package path in `desta/models/modeling_desta25.py` intentionally keeps
only `groupwise_ortho`. The old `orca_desta` and `orca_r1` names are accepted
as aliases for checkpoint compatibility.
