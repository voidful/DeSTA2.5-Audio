#!/bin/bash

#SBATCH --job-name=desta_orca_base
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=./slurm-report/ablation_orca_base_%j.out
#SBATCH --error=./slurm-report/ablation_orca_base_%j.out

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# ===== 實驗設定 =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-0.6b_ORCAHybrid
dataset_config=DestaAQA-5M_0.6b_orca
project=desta25_ablation

# Use provided argument as exp_dir, or create new one
if [ -n "$1" ]; then
    exp_dir="${1%/}" # Remove trailing slash
    
    # Find latest numeric checkpoint (e.g., checkpoint-94260)
    # Filter for directories ending in numbers to avoid checkpoint-initial
    latest_ckpt=$(ls -d ${exp_dir}/checkpoint-* 2>/dev/null | grep -E "checkpoint-[0-9]+$" | sort -V | tail -n 1)
    
    if [ -n "$latest_ckpt" ]; then
        RESUME="$latest_ckpt"
        echo "Found latest checkpoint: $RESUME"
    else
        # Fallback to boolean true if no numeric checkpoint found (let Trainer decide)
        RESUME="true"
        echo "No numeric checkpoint found, trying 'true'"
    fi
else
    exp_name_dated=$(date +%y%m%d-%H%M)_orca_base
    exp_dir="/work/voidful2nlp/desta/outputs/ablation/${exp_name_dated}"
    RESUME="null"
    echo "Starting new experiment: ${exp_dir}"
fi

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=4
MASTER_PORT=$((29500 + RANDOM % 1000))

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    ++resume_from_checkpoint=${RESUME} \
    project=${project} \
    name=orca_base \
    model.orca.enabled=true model.orca.local_enabled=false model.orca.global_cross_attn=false model.orca.deep_injection_enabled=false model.orca.ortho_weight_global=0.05 model.orca.ortho_diversity_weight=0.05 model.orca.ortho_weight_qformer_local=0.05 model.orca.align_weight_local=0
