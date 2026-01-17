#!/bin/bash
#SBATCH --job-name=P0_2_liar_eval
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=./slurm-report/P0_2_liar_eval_%j.out
#SBATCH --error=./slurm-report/P0_2_liar_eval_%j.out

# ==========================================
# P0-2b: Liar Transcript Evaluation
# 
# 測試模型是否能正確使用音訊資訊
# 當 transcript 刻意錯誤時
# 預期: DeSTA ~35%, ORCA ~70%
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

LIAR_DATA="/work/voidful2nlp/desta/experiment_results/P0_2_liar/data/liar_samples.jsonl"
OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P0_2_liar/eval"

mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行評估 =====
echo "Starting P0-2b: Liar Transcript Evaluation"

# 測試 DeSTA baseline
echo "=== Testing DeSTA Baseline ==="
python experiments/ablation/liar_transcript/liar_eval.py \
    --model ${MODEL_DESTA} \
    --liar-data ${LIAR_DATA} \
    --output-dir ${OUTPUT_DIR}/desta \
    --device cuda

# 測試 ORCA
echo "=== Testing ORCA ==="
python experiments/ablation/liar_transcript/liar_eval.py \
    --model ${MODEL_ORCA} \
    --liar-data ${LIAR_DATA} \
    --output-dir ${OUTPUT_DIR}/orca \
    --device cuda

echo "P0-2b Complete!"
echo "Results in: ${OUTPUT_DIR}"
