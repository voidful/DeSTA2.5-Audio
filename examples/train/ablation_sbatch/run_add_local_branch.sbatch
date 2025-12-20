#!/bin/bash

#SBATCH --job-name=desta_add_local_branch
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-report/ablation_add_local_branch_%j.out
#SBATCH --error=./slurm-report/ablation_add_local_branch_%j.out

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

exp_suffix="add_local_branch"
output_root="/work/voidful2nlp/desta/outputs/ablation"

# Default to new experiment
exp_name_dated=$(date +%y%m%d-%H%M)_${exp_suffix}
exp_dir="${output_root}/${exp_name_dated}"
resume_args=""

# Check for existing runs to resume
latest_dir=$(ls -td ${output_root}/*_${exp_suffix} 2>/dev/null | head -n 1)

if [ -n "$latest_dir" ]; then
    # Find latest checkpoint in that directory
    latest_ckpt=$(ls -td ${latest_dir}/checkpoint-* 2>/dev/null | head -n 1)
    
    if [ -n "$latest_ckpt" ]; then
        echo "Found existing checkpoint, resuming from: ${latest_ckpt}"
        exp_dir=${latest_dir}
        resume_args="++resume_from_checkpoint=${latest_ckpt}"
    else
        echo "Found experiment directory but no checkpoint, starting new run."
    fi
else
    echo "No previous experiment found, starting new run."
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
    project=${project} \
    name=add_local_branch \
    model.orca.enabled=true model.orca.local_enabled=false model.orca.global_cross_attn=false model.orca.deep_injection_enabled=false model.orca.ortho_weight_global=0.05 model.orca.ortho_diversity_weight=0.05 model.orca.ortho_weight_qformer_local=0.05 model.orca.align_weight_local=0 model.orca.local_enabled=true \
    ${resume_args}
