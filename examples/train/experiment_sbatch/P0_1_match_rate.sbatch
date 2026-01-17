#!/bin/bash
#SBATCH --job-name=P0_1_match_rate
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm-report/P0_1_match_rate_%j.out
#SBATCH --error=./slurm-report/P0_1_match_rate_%j.out

# ==========================================
# P0-1: Match Rate Quantification
# 
# 比較有/無音訊時的預測匹配率
# 預期: DeSTA ~78%, ORCA ~60%
# 預計時間: 2-3 天
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true
pip install "numpy<2.0" -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
# 修改這裡的 MODEL 來測試不同模型
MODEL_DESTA="voidful/desta25-qwen3-4b"
MODEL_ORCA="voidful/desta25-4b-R2-full"  # 修改為實際的 ORCA 模型路徑
SAMPLES_PER_TASK=250

OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P0_1_match_rate"
mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行實驗 =====
echo "Starting P0-1: Match Rate Analysis"

# 測試 DeSTA baseline
echo "=== Testing DeSTA Baseline ==="
python experiments/diagnosis/match_rate_analysis.py \
    --model ${MODEL_DESTA} \
    --samples-per-task ${SAMPLES_PER_TASK} \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}/desta \
    --device cuda

# 測試 ORCA
echo "=== Testing ORCA ==="
python experiments/diagnosis/match_rate_analysis.py \
    --model ${MODEL_ORCA} \
    --samples-per-task ${SAMPLES_PER_TASK} \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}/orca \
    --device cuda

echo "P0-1 Complete!"
echo "Results in: ${OUTPUT_DIR}"
