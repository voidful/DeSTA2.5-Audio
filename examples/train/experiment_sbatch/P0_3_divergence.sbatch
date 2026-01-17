#!/bin/bash
#SBATCH --job-name=P0_3_divergence
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./slurm-report/P0_3_divergence_%j.out
#SBATCH --error=./slurm-report/P0_3_divergence_%j.out

# ==========================================
# P0-3: Divergence Rate Analysis
# 
# 計算訓練資料中 y_w ≠ y_l 的比例
# 預期: ~34% divergence
# 預計時間: 1 天 (5M samples)
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true
pip install "numpy<2.0" -q 2>/dev/null || true
pip install torchaudio soundfile nvidia-nvjitlink-cu12 -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
MANIFEST="/work/voidful2nlp/desta/qwen3_4b_desta_fixed.jsonl"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P0_3_divergence"
SAMPLES=50000  # Start with 50k, can increase

mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行實驗 =====
echo "Starting P0-3: Divergence Rate Analysis"
echo "Manifest: ${MANIFEST}"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"

python experiments/diagnosis/divergence_analysis.py \
    --manifest ${MANIFEST} \
    --model ${MODEL} \
    --samples ${SAMPLES} \
    --batch-size 64 \
    --output-dir ${OUTPUT_DIR} \
    --device cuda

echo "P0-3 Complete!"
