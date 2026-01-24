#!/bin/bash
#SBATCH --job-name=baseline_full
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/baseline_full_%j.out
#SBATCH --error=./slurm-report/baseline_full_%j.out

# ==========================================
# Baseline: Standard Q-Former + Full Dataset
# 
# 標準 Q-Former 連接器，無任何額外技巧
# 數據集: qwen3_4b_desta_fixed.jsonl (全部資料)
#
# 移除所有技巧:
# ❌ G1 (Modality-DPO): 關閉
# ❌ H1 (Variational Grouping): 關閉
# ❌ P1 (ASR Dropout): 關閉
# ❌ ORCA-R1: 使用標準 qformer_1
#
# 作為對照組，用於驗證各種技巧的效果
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-4B_Qformer6L  # 標準 Q-Former 6層配置
dataset_config=DestaAQA-5M_4b_orca_full  # 使用完整數據集

project=orca_r1_ablation
name="baseline_full"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="/work/voidful2nlp/desta/outputs/${project}/${exp_name}"

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
    name=${name}

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_baseline_full"
