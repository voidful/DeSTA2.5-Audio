#!/bin/bash
#SBATCH --job-name=A1_flat64
#SBATCH --output=logs/A1_flat64_%j.out
#SBATCH --error=logs/A1_flat64_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# ==========================================
# A1: Flat Global 64 (ORCA without Grouping)
# ==========================================
# Uses struct_orca connector but with 1 group × 64 queries.
# No grouping structure, just flat 64 learnable queries.

set -e
cd /path/to/DeSTA2.5-Audio

source ~/.bashrc
conda activate desta

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME=/path/to/hf_cache
export WANDB_PROJECT="struct_orca_ablation"

EXP_DIR="outputs/ablation/A1_flat_global_64"
mkdir -p $EXP_DIR/logs

torchrun --nproc_per_node=4 \
    examples/train/train_desta.py \
    --config-name desta25_qwen3-0.6b_ORCA \
    dataset=DestaAQA-5M_0.6b \
    exp_dir=$EXP_DIR \
    name=A1_flat_global_64 \
    model.struct_orca.num_groups=1 \
    model.struct_orca.queries_per_group=64 \
    model.struct_orca.inter_group_weight=0 \
    model.struct_orca.intra_group_weight=0 \
    model.struct_orca.iv_weight=0 \
    wandb.project=struct_orca_ablation
