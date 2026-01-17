#!/bin/bash
#SBATCH --job-name=P1_2_refusal
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=./slurm-report/P1_2_refusal_%j.out
#SBATCH --error=./slurm-report/P1_2_refusal_%j.out

# ==========================================
# P1-2: Refusal Rate Analysis
# 
# 分析模型在 audio-dependent 問題上的拒絕率
# 計算 "cannot determine" 類型的回應比例
# 預期: DeSTA ~40%, ORCA ~18%
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true
pip install torch torchaudio soundfile --upgrade --force-reinstall -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
MODEL_DESTA="voidful/desta25-qwen3-4b"
MODEL_ORCA="voidful/desta25-4b-R2-full"  # 修改為實際的 ORCA 模型路徑
SAMPLES=200

OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P1_2_refusal"
mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行實驗 =====
echo "Starting P1-2: Refusal Rate Analysis"

# 測試 DeSTA baseline
echo "=== Testing DeSTA Baseline ==="
python experiments/diagnosis/refusal_analysis.py \
    --model ${MODEL_DESTA} \
    --samples ${SAMPLES} \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}/desta \
    --device cuda

# 測試 ORCA
echo "=== Testing ORCA ==="
python experiments/diagnosis/refusal_analysis.py \
    --model ${MODEL_ORCA} \
    --samples ${SAMPLES} \
    --batch-size 32 \
    --output-dir ${OUTPUT_DIR}/orca \
    --device cuda

echo "P1-2 Complete!"
echo "Results in: ${OUTPUT_DIR}"
