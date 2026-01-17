#!/bin/bash
#SBATCH --job-name=P0_2_liar_gen
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=./slurm-report/P0_2_liar_gen_%j.out
#SBATCH --error=./slurm-report/P0_2_liar_gen_%j.out

# ==========================================
# P0-2a: Liar Transcript Generation
# 
# 生成帶有矛盾 transcript 的測試資料
# - Emotion: sad audio + happy text
# - Gender: male voice + female text
# - Animal: dog bark + cat text
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true
pip install torch torchaudio soundfile "numpy<2.0" --upgrade --ignore-installed

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
SAMPLES_PER_TASK=150  # 150 per task = ~450 total
OUTPUT_DIR="/work/voidful2nlp/desta/experiment_results/P0_2_liar/data"

mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== 執行生成 =====
echo "Starting P0-2a: Liar Transcript Generation"

python experiments/ablation/liar_transcript/liar_generator.py \
    --samples-per-task ${SAMPLES_PER_TASK} \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

echo "P0-2a Complete!"
echo "Generated liar data in: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run P0_2_liar_eval.sbatch to evaluate models"
