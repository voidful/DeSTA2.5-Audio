#!/bin/bash
#SBATCH --job-name=P3_benchmark
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/P3_benchmark_%j.out
#SBATCH --error=./slurm-report/P3_benchmark_%j.out

# ==========================================
# P3: Comprehensive Benchmark (MMAU & MMSU)
# 
# Runs standard benchmarks on the specified model.
# ==========================================

# ===== Environment =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# ===== Variables =====
# MODEL_PATH passed from submit_all.sh or command line
: "${MODEL_PATH:=voidful/QAQ_0.6b_orca_all}"  # Default to baseline if not set
: "${OUTPUT_DIR:=benchmarks}"

echo "Benchmarking Model: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ./slurm-report

# ===== MMAU Evaluation =====
echo "Starting MMAU Evaluation..."
python examples/evaluation/mmau_eval.py \
    --model_id ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/mmau \
    --split test_mini

# ===== MMSU Evaluation =====
echo "Starting MMSU Evaluation..."
python examples/evaluation/mmsu_eval.py \
    --model_id ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/mmsu \
    --split test

echo "P3 Benchmarking Complete!"
