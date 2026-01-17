#!/bin/bash
#SBATCH --job-name=P1_1_linear_probe
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=./slurm-report/P1_1_linear_probe_%j.out
#SBATCH --error=./slurm-report/P1_1_linear_probe_%j.out

# ==========================================
# P1-1: Linear Probing Analysis
# 
# 測試 frozen features 是否包含 speaker/emotion 資訊
# 使用 logistic regression 在 CommonVoice (gender) 
# 和 IEMOCAP (emotion) 上訓練
# 預期: 70-90% accuracy
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
MODEL_DESTA="voidful/desta25-qwen3-4b"
MODEL_ORCA="voidful/desta25-4b-R2-full"  # 修改為實際的 ORCA 模型路徑
SAMPLES=1000
CV_FOLDS=5

OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P1_1_linear_probe"
mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行實驗 =====
echo "Starting P1-1: Linear Probing Analysis"

# 測試 DeSTA baseline
echo "=== Testing DeSTA Baseline ==="
python experiments/ablation/linear_probing.py \
    --model ${MODEL_DESTA} \
    --task both \
    --samples ${SAMPLES} \
    --cv-folds ${CV_FOLDS} \
    --batch-size 64 \
    --output-dir ${OUTPUT_DIR}/desta \
    --device cuda

# 測試 ORCA
echo "=== Testing ORCA ==="
python experiments/ablation/linear_probing.py \
    --model ${MODEL_ORCA} \
    --task both \
    --samples ${SAMPLES} \
    --cv-folds ${CV_FOLDS} \
    --batch-size 64 \
    --output-dir ${OUTPUT_DIR}/orca \
    --device cuda

echo "P1-1 Complete!"
echo "Results in: ${OUTPUT_DIR}"
