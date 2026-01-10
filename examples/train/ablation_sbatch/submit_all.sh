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

# C2: Contrastive Alignment (weight=0.1)
sbatch examples/train/ablation_sbatch/C2_align_loss_0.1.sbatch

# D1: Orthogonal Projection (Method 2)
sbatch examples/train/ablation_sbatch/D1_ortho_proj.sbatch

# D2: Adversarial Erasure (Method 1)
sbatch examples/train/ablation_sbatch/D2_adv_erasure.sbatch

echo ""
echo "6 experiments submitted!"
echo "A0 → A2 → B2 → C2 → D1 → D2"
echo "Monitor: squeue -u \$USER"
