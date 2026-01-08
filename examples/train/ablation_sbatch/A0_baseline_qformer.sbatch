#!/bin/bash
#SBATCH --job-name=A0_baseline
#SBATCH --output=logs/A0_baseline_%j.out
#SBATCH --error=logs/A0_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# ==========================================
# A0: Baseline QFormer (No ORCA Components)
# ==========================================
# This is the DeSTA2.5 baseline with standard QFormer connector.
# No grouping, no orthogonality losses, no discriminator.

set -e
cd /path/to/DeSTA2.5-Audio

# Activate environment
source ~/.bashrc
conda activate desta

# Export environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME=/path/to/hf_cache
export WANDB_PROJECT="struct_orca_ablation"

# Create output directory
EXP_DIR="outputs/ablation/A0_baseline_qformer"
mkdir -p $EXP_DIR/logs

# Run training with baseline QFormer config
torchrun --nproc_per_node=4 \
    examples/train/train_desta.py \
    --config-name desta25_qwen3-0.6b_Qformer6L \
    dataset=DestaAQA-5M_0.6b \
    exp_dir=$EXP_DIR \
    name=A0_baseline_qformer \
    wandb.project=struct_orca_ablation
