#!/bin/bash

# DeSTA 2.5 Audio Additive Ablation Study Sbatch Launcher
# Launch experiments building up from corrected ORCA base to full system.

SBATCH_DIR="examples/train/ablation_sbatch"

EXPERIMENTS=(
    "baseline_desta2.0"
    "orca_base"
    "add_local_branch"
    "add_global_xattn"
    "add_deep_injection"
    "add_align_loss"
    "desta2.5_full"
)

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    FILE="${SBATCH_DIR}/run_${EXP_NAME}.sbatch"
    if [ -f "$FILE" ]; then
        echo "Submitting $EXP_NAME..."
        sbatch "$FILE"
    else
        echo "Warning: $FILE not found, skipping."
    fi
done
