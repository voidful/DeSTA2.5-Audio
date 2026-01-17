#!/bin/bash
#SBATCH --job-name=P2_ortho_only
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/P2_ortho_only_%j.out
#SBATCH --error=./slurm-report/P2_ortho_only_%j.out

# ==========================================
# P2-1a: Orthogonal Encoder Only
# 
# Ablation: Test ONLY Orthogonal Encoder
# - NO ASR Dropout
# - NO Modality-DPO
# - NO Variational Grouping
#
# Purpose: Isolate contribution of orthogonal disentanglement
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true
pip install torchaudio soundfile nvidia-nvjitlink-cu12 -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-4B_ORCA
dataset_config=DestaAQA-5M_local

project=orca_r1_ablation
name="P2_ortho_only"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="/work/voidful2nlp/desta/experiment_results/${exp_name}"

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=2
MASTER_PORT=$((29500 + RANDOM % 1000))

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    ++model.orca_r1.num_groups=8 \
    ++model.orca_r1.queries_per_group=8 \
    ++model.orca_r1.inter_group_weight=0.1 \
    ++model.orca_r1.intra_group_weight=0.03 \
    ++model.modality_dpo_enabled=false \
    ++model.variational_grouping_enabled=false \
    ++model.asr_dropout_prob=0.0

# ===== 評估 =====
echo "Starting MMAU Evaluation..."
python examples/evaluation/mmau_eval.py \
    --model_id ${exp_dir} \
    --output_dir ${exp_dir}/eval_results/mmau \
    --split test_mini

echo "Starting MMSU Evaluation..."
python examples/evaluation/mmsu_eval.py \
    --model_id ${exp_dir} \
    --output_dir ${exp_dir}/eval_results/mmsu \
    --split test

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_P2_ortho_only"
