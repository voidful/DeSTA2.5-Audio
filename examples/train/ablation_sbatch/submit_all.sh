#!/bin/bash
# Submit all ablation experiments in order
# Usage: ./submit_all.sh

set -e

# Phase 1: Architecture
echo "Submitting Group A: Architecture experiments..."
sbatch A0_baseline_qformer.sbatch
sbatch A1_flat_global_64.sbatch
sbatch A2_grouped_8x8.sbatch

# Phase 2: Group Losses
echo "Submitting Group B: Group Loss experiments..."
sbatch B1_inter_group_0.05.sbatch
sbatch B2_inter_group_0.1.sbatch
sbatch B3_both_losses.sbatch

# Phase 3: IV Disentanglement
echo "Submitting Group C: IV experiments..."
sbatch C1_iv_0.05.sbatch
sbatch C2_iv_0.1_full.sbatch

echo ""
echo "All 8 experiments submitted!"
echo "Monitor with: squeue -u \$USER"
