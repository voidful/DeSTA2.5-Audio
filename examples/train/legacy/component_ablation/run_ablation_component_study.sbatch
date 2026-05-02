#!/bin/bash
#
# ORCA-DeSTA Component Ablation Study — SLURM Batch Script
#
# Runs 5 progressive ablation experiments (Table 5 in the paper):
#   Exp 0: Baseline (standard DeSTA Q-Former)
#   Exp 1: + Groupwise Orthogonality
#   Exp 2: + ACP (Acoustic-Contrastive Preference)
#   Exp 3: + Stochastic Perturbation Encoding
#   Exp 4: Full Model (+ ASR Dropout)
#
# Usage:
#   sbatch examples/train/run_ablation_component_study.sbatch
#
# Or run specific experiments:
#   ABLATION_EXPS="0 1 2" sbatch examples/train/run_ablation_component_study.sbatch

#SBATCH --job-name=orca_ablation
#SBATCH --partition=dev
#SBATCH --account=MST115022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=./slurm-report/orca_ablation_%j.out
#SBATCH --error=./slurm-report/orca_ablation_%j.out

set -e

# ===== Environment Setup =====
# Initialize conda
__conda_setup_done=0

# 1. If CONDA_EXE is inherited from the submitting shell
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
    __conda_setup_done=1
    echo "[env] conda initialised via CONDA_EXE (${CONDA_EXE})"
fi

# 2. Try module load
if [ "$__conda_setup_done" -eq 0 ]; then
    if ! type module >/dev/null 2>&1; then
        source /etc/profile.d/modules.sh 2>/dev/null || true
    fi
    module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || true
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        __conda_setup_done=1
        echo "[env] conda initialised via module load"
    fi
fi

# 3. Try common base directories
if [ "$__conda_setup_done" -eq 0 ]; then
    for __try_base in \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "${HOME}/.local/miniconda3" \
        "/work/${USER:-$(whoami)}/miniconda3" \
        "/work/${USER:-$(whoami)}/anaconda3" \
        "/work/voidful2nlp/miniconda3" \
        "/work/voidful2nlp/anaconda3" \
        "/opt/conda" \
        "/pkg/miniconda3" \
        "/pkg/anaconda3"; do
        if [ -f "${__try_base}/etc/profile.d/conda.sh" ]; then
            source "${__try_base}/etc/profile.d/conda.sh"
            __conda_setup_done=1
            echo "[env] conda initialised from ${__try_base}"
            break
        fi
    done
fi

if [ "$__conda_setup_done" -eq 0 ]; then
    echo "[env] ERROR: cannot find conda installation. Aborting." >&2
    exit 1
fi
unset __conda_setup_done __try_base

# Target environment name
CONDA_ENV_NAME="${CONDA_ENV_NAME:-desta25-qwen3}"

# Auto-create environment if it doesn't exist
if ! conda env list | grep -qw "$CONDA_ENV_NAME"; then
    echo "[env] Environment '$CONDA_ENV_NAME' not found. Creating..."
    ENV_YML="/work/voidful2nlp/DeSTA2.5-Audio/environment.yml"
    if [ -f "$ENV_YML" ]; then
        conda env create -n "$CONDA_ENV_NAME" -f "$ENV_YML" -y
    else
        echo "[env] No environment.yml found — creating minimal env"
        conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    fi
fi

# Activate
conda activate "$CONDA_ENV_NAME"
echo "[env] active env: $(conda info --envs | grep '*')"

# Working directory
cd /work/voidful2nlp/DeSTA2.5-Audio

# Dependencies
pip install "numpy<2.0" -q 2>/dev/null || true

# Environment variables
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== Experiment Configuration =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
DATA_ROOT="/work/voidful2nlp/desta"
OUTPUT_BASE="/work/voidful2nlp/desta/outputs/desta25_ablation"
DATASET_CONFIG="DestaAQA-5M_4b_ablation"
NUM_GPUS=4
MASTER_PORT=$((29500 + RANDOM % 1000))
TIMESTAMP=$(date +%y%m%d-%H%M)

# Which experiments to run (default: all 5)
ABLATION_EXPS="${ABLATION_EXPS:-0 1 2 3 4}"

# Ablation experiment definitions
# Format: config_name|experiment_label|description
declare -A ABLATION_CONFIGS
ABLATION_CONFIGS[0]="ablation_0_baseline|baseline_qformer|Baseline (standard DeSTA Q-Former)"
ABLATION_CONFIGS[1]="ablation_1_groupwise_ortho|plus_ortho|+ Groupwise Orthogonality"
ABLATION_CONFIGS[2]="ablation_2_plus_acp|plus_acp|+ ACP"
ABLATION_CONFIGS[3]="ablation_3_plus_stochastic|plus_stoch|+ Stochastic Perturbation"
ABLATION_CONFIGS[4]="ablation_4_full_model|full|Full Model (+ ASR Dropout)"

# Create output directories
mkdir -p "${OUTPUT_BASE}"
mkdir -p ./slurm-report

# Record job info
echo "========================================" | tee "${OUTPUT_BASE}/ablation_run_info.txt"
echo "ORCA-DeSTA Component Ablation Study"     | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "========================================"| tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "Job ID: ${SLURM_JOB_ID:-local}"         | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "Start Time: $(date)"                     | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "Experiments: ${ABLATION_EXPS}"           | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "Dataset: ${DATASET_CONFIG}"              | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "GPUs: ${NUM_GPUS}"                       | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"
echo "========================================" | tee -a "${OUTPUT_BASE}/ablation_run_info.txt"

git diff > "${OUTPUT_BASE}/git-diff.txt" 2>/dev/null || true
nvidia-smi > "${OUTPUT_BASE}/nvidia-smi.txt" 2>/dev/null || true


# ===== Run Ablation Experiments =====
for EXP_ID in ${ABLATION_EXPS}; do
    IFS="|" read -r CONFIG_NAME EXP_LABEL DESCRIPTION <<< "${ABLATION_CONFIGS[$EXP_ID]}"

    if [ -z "$CONFIG_NAME" ]; then
        echo "ERROR: Unknown experiment ID: ${EXP_ID}. Skipping."
        continue
    fi

    EXP_NAME="${TIMESTAMP}_${EXP_LABEL}"
    EXP_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    mkdir -p "${EXP_DIR}"

    echo ""
    echo "============================================================"
    echo "  Experiment ${EXP_ID}: ${DESCRIPTION}"
    echo "  Config: ${CONFIG_NAME}"
    echo "  Output: ${EXP_DIR}"
    echo "============================================================"

    # Record per-experiment config
    cat > "${EXP_DIR}/experiment_info.json" << EOF
{
    "experiment_id": ${EXP_ID},
    "label": "${EXP_LABEL}",
    "description": "${DESCRIPTION}",
    "config": "${CONFIG_NAME}",
    "dataset": "${DATASET_CONFIG}",
    "start_time": "$(date -Iseconds)",
    "num_gpus": ${NUM_GPUS},
    "slurm_job_id": "${SLURM_JOB_ID:-local}"
}
EOF

    # Increment master port for each experiment to avoid conflicts
    CURRENT_PORT=$((MASTER_PORT + EXP_ID))

    # Launch training
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${CURRENT_PORT} \
        ${ROOT_DIR}/examples/train/train_desta.py \
        --config-path=config \
        --config-name=${CONFIG_NAME} \
        +dataset=${DATASET_CONFIG} \
        ++exp_dir=${EXP_DIR} \
        project=desta25_ablation \
        name="${EXP_LABEL}" \
        ++dataset.train_ds.data_root=${DATA_ROOT} \
        ++dataset.validation_ds.data_root=${DATA_ROOT} \
        ++resume_from_checkpoint=null \
        ++init_from_pretrained_weights=null

    echo "  Experiment ${EXP_ID} (${EXP_LABEL}) finished at $(date)"
    echo "${EXP_ID}|${EXP_LABEL}|done|$(date -Iseconds)" >> "${OUTPUT_BASE}/ablation_progress.log"
done

echo ""
echo "============================================================"
echo "  All ablation experiments completed at $(date)"
echo "============================================================"
echo "Training finished at $(date)" >> "${OUTPUT_BASE}/ablation_run_info.txt"
