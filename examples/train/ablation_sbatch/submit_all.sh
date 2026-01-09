#!/bin/bash
# Minimal ablation: prove each component adds value
# 4 experiments total

set -e

echo "Submitting minimal ablation experiments..."

# 1. Baseline (no ORCA)
sbatch A0_baseline_qformer.sbatch

# 2. +Grouping (8×8)
sbatch A2_grouped_8x8.sbatch

# 3. +Inter-group orthogonality
sbatch B2_inter_group_0.1.sbatch

# 4. +IV Disentanglement (Full ORCA)
sbatch C2_iv_0.1_full.sbatch

echo ""
echo "4 experiments submitted!"
echo "A0 → A2 → B2 → C2"
echo "Monitor: squeue -u \$USER"
