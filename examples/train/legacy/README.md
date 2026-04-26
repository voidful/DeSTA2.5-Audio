Legacy training experiments
===========================

This directory keeps exploratory sbatch launchers and ablation scripts that are
not the current paper method. Main training entry points should use:

- `examples/train/config/*_Qformer6L.yaml` for the DeSTA baseline
- `examples/train/config/*_ORCA.yaml` for ORCA-DeSTA

The canonical ORCA-DeSTA connector mode is `orca_desta`. Older scripts may use
`orca_r1`; that name is kept as a compatibility alias only.
