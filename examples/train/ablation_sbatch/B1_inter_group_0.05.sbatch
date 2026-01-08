#!/bin/bash
#SBATCH --job-name=B1_inter05
#SBATCH --output=logs/B1_inter05_%j.out
#SBATCH --error=logs/B1_inter05_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# ==========================================
# B1: Inter-Group Orthogonality (weight=0.05)
# ==========================================
# Adds inter-group orthogonality loss to push group centroids apart.

set -e
cd /path/to/DeSTA2.5-Audio

source ~/.bashrc
conda activate desta

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME=/path/to/hf_cache
export WANDB_PROJECT="struct_orca_ablation"

EXP_DIR="outputs/ablation/B1_inter_group_0.05"
mkdir -p $EXP_DIR/logs

torchrun --nproc_per_node=4 \
    examples/train/train_desta.py \
    --config-name desta25_qwen3-0.6b_ORCA \
    dataset=DestaAQA-5M_0.6b \
    exp_dir=$EXP_DIR \
    name=B1_inter_group_0.05 \
    model.struct_orca.num_groups=8 \
    model.struct_orca.queries_per_group=8 \
    model.struct_orca.inter_group_weight=0.05 \
    model.struct_orca.intra_group_weight=0 \
    model.struct_orca.iv_weight=0 \
    wandb.project=struct_orca_ablation
