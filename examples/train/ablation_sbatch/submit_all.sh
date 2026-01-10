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

# 4. +Contrastive Alignment Loss
sbatch C2_align_loss_0.1.sbatch

echo ""
echo "4 experiments submitted!"
echo "A0 → A2 → B2 → C2_align"
echo "Monitor: squeue -u \$USER"
