Legacy model experiments
========================

This directory preserves pre-paper experiments that are not part of the
ORCA-DeSTA method:

- `orca_hybrid`
- deep-injection / gated cross-attention experiments
- Acoustic-Contrastive Decoding (ACD)
- auxiliary alignment and erasure probes

The main package path in `desta/models/modeling_desta25.py` intentionally keeps
only the DeSTA Q-Former baseline (`qformer_1`) plus the ORCA-DeSTA paper method
(`orca_desta`, with `orca_r1` accepted as a backward-compatible alias).
