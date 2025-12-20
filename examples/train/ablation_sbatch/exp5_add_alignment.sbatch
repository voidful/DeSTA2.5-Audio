#!/bin/bash

#SBATCH --job-name=ablation_exp5_add_alignment
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-report/ablation_exp5_add_alignment_%j.out
#SBATCH --error=./slurm-report/ablation_exp5_add_alignment_%j.out

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
project=desta25_ablation_v2

exp_suffix="exp5_add_alignment"
output_root="/work/voidful2nlp/desta/outputs/ablation_v2"

exp_name_dated=$(date +%y%m%d-%H%M)_${exp_suffix}
exp_dir="${output_root}/${exp_name_dated}"
resume_args=""

latest_dir=$(ls -td ${output_root}/*_${exp_suffix} 2>/dev/null | head -n 1)
if [ -d "$latest_dir" ] && [ -d "$latest_dir/checkpoint-latest" ]; then
    exp_dir="$latest_dir"
    resume_args="resume_from_checkpoint=$latest_dir/checkpoint-latest"
fi

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=4
MASTER_PORT=$((29500 + RANDOM % 1000))

# Exp 5: + Alignment Loss (Full ORCA with simplified losses)
torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    project=${project} \
    name=exp5_add_alignment \
    model.orca.enabled=true \
    model.orca.local_enabled=true \
    model.orca.global_cross_attn=true \
    model.orca.deep_injection_enabled=true \
    model.orca.local_downsample=2 \
    model.orca.local_kernel_size=5 \
    model.orca.audio_position_scale=5.0 \
    model.orca.ortho_diversity_weight=0.05 \
    model.orca.align_weight_local=0.05 \
    ${resume_args}
