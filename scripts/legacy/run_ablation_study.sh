#!/bin/bash
#
# ===================================================================
# ORCA-DeSTA Component Ablation Study — Master Script
# ===================================================================
#
# This script manages the complete ablation study pipeline for the
# three ORCA-DeSTA components described in the paper:
#
#   Component 1: Groupwise Orthogonality (§4.2)
#                → Reduces query redundancy (cosine sim 0.923 → 0.077)
#
#   Component 2: Stochastic Perturbation Encoding (§4.3)
#                → Reduces acoustic collapse (variance 0.0015 → 0.113)
#
#   Component 3: Acoustic-Contrastive Preference / ACP (§4.4)
#                → Reduces audio-insensitive decoding (APV VR 41% → 14%)
#
# Progressive ablation (Table 5 in paper):
#   Exp 0: Baseline (ORCA connector, no aux losses)
#   Exp 1: + Groupwise Orthogonality
#   Exp 2: + ACP
#   Exp 3: + Stochastic Perturbation Encoding
#   Exp 4: Full Model (+ ASR Dropout)
#
# ===================================================================
#
# Usage (local):
#   bash scripts/run_ablation_study.sh                      # Train all
#   bash scripts/run_ablation_study.sh --eval-only          # Eval only
#   ABLATION_EXPS="0 2 4" bash scripts/run_ablation_study.sh  # Subset
#
# Usage (SLURM):
#   sbatch examples/train/run_ablation_component_study.sbatch
#
# ===================================================================

set -e

# ===== Parse Arguments =====
EVAL_ONLY=false
EVAL_SAKURA=false
for arg in "$@"; do
    case $arg in
        --eval-only) EVAL_ONLY=true ;;
        --sakura)    EVAL_SAKURA=true ;;
    esac
done

# ===== Configuration =====
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/work/voidful2nlp/desta}"
OUTPUT_BASE="${OUTPUT_BASE:-/work/voidful2nlp/desta/outputs/desta25_ablation}"
DATASET_CONFIG="DestaAQA-5M_4b_ablation"
NUM_GPUS="${NUM_GPUS:-4}"
TIMESTAMP=$(date +%y%m%d-%H%M)

# Which experiments to run (default: all 5)
ABLATION_EXPS="${ABLATION_EXPS:-0 1 2 3 4}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# GPU setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# ===== Experiment Definitions =====
# Progressive ablation matching Table 5 in the paper
declare -A ABLATION_CONFIGS
ABLATION_CONFIGS[0]="ablation_0_baseline"
ABLATION_CONFIGS[1]="ablation_1_groupwise_ortho"
ABLATION_CONFIGS[2]="ablation_2_plus_acp"
ABLATION_CONFIGS[3]="ablation_3_plus_stochastic"
ABLATION_CONFIGS[4]="ablation_4_full_model"

declare -A ABLATION_NAMES
ABLATION_NAMES[0]="baseline"
ABLATION_NAMES[1]="plus_ortho"
ABLATION_NAMES[2]="plus_acp"
ABLATION_NAMES[3]="plus_stoch"
ABLATION_NAMES[4]="full"

declare -A ABLATION_DESC
ABLATION_DESC[0]="Baseline (Contrastive Alignment Only)"
ABLATION_DESC[1]="+ Groupwise Orthogonality"
ABLATION_DESC[2]="+ ACP (Acoustic-Contrastive Preference)"
ABLATION_DESC[3]="+ Stochastic Perturbation Encoding"
ABLATION_DESC[4]="Full Model (w/ ASR Dropout)"

# Create directories
mkdir -p "${OUTPUT_BASE}"

# ===== Print Header =====
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ORCA-DeSTA Component Ablation Study                    ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Paper: Structured Connector Geometry for Data-Efficient ║"
echo "║         Audio-Language Models                            ║"
echo "║                                                          ║"
echo "║  Three Components:                                       ║"
echo "║    C1: Groupwise Orthogonality      (§4.2)               ║"
echo "║    C2: Stochastic Perturbation      (§4.3)               ║"
echo "║    C3: Acoustic-Contrastive Pref    (§4.4)               ║"
echo "║                                                          ║"
echo "╠════════════════════════════════════════════════════════════╣"
printf "║  Experiments: %-43s ║\n" "${ABLATION_EXPS}"
printf "║  GPUs:        %-43s ║\n" "${NUM_GPUS}"
printf "║  Output:      %-43s ║\n" "${OUTPUT_BASE}"
printf "║  Eval Only:   %-43s ║\n" "${EVAL_ONLY}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""


# ===== Training Phase =====
if [ "$EVAL_ONLY" = false ]; then
    echo "═══════════════════════════════════════════"
    echo "  Phase 1: Training Ablation Experiments"
    echo "═══════════════════════════════════════════"

    for EXP_ID in ${ABLATION_EXPS}; do
        CONFIG="${ABLATION_CONFIGS[$EXP_ID]}"
        NAME="${ABLATION_NAMES[$EXP_ID]}"
        DESC="${ABLATION_DESC[$EXP_ID]}"

        if [ -z "$CONFIG" ]; then
            echo "  ERROR: Unknown experiment ID: ${EXP_ID}"
            continue
        fi

        EXP_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${NAME}"
        mkdir -p "${EXP_DIR}"

        echo ""
        echo "────────────────────────────────────────────"
        echo "  Exp ${EXP_ID}: ${DESC}"
        echo "  Config: ${CONFIG}"
        echo "  Output: ${EXP_DIR}"
        echo "────────────────────────────────────────────"

        # Save experiment metadata
        cat > "${EXP_DIR}/experiment_info.json" << EOF
{
    "experiment_id": ${EXP_ID},
    "label": "${NAME}",
    "description": "${DESC}",
    "config": "${CONFIG}",
    "dataset": "${DATASET_CONFIG}",
    "timestamp": "${TIMESTAMP}",
    "num_gpus": ${NUM_GPUS},
    "components": {
        "groupwise_orthogonality": $([ $EXP_ID -ge 1 ] && echo "true" || echo "false"),
        "acp": $([ $EXP_ID -ge 2 ] && echo "true" || echo "false"),
        "stochastic_perturbation": $([ $EXP_ID -ge 3 ] && echo "true" || echo "false"),
        "asr_dropout": $([ $EXP_ID -ge 4 ] && echo "true" || echo "false")
    }
}
EOF

        # Record git state
        git diff > "${EXP_DIR}/git-diff.txt" 2>/dev/null || true

        # Launch training
        MASTER_PORT=$((29500 + RANDOM % 1000))

        torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
            ${ROOT_DIR}/examples/train/train_desta.py \
            --config-path=config \
            --config-name=${CONFIG} \
            +dataset=${DATASET_CONFIG} \
            ++exp_dir=${EXP_DIR} \
            project=desta25_ablation \
            name="${NAME}" \
            ++dataset.train_ds.data_root=${DATA_ROOT} \
            ++dataset.validation_ds.data_root=${DATA_ROOT} \
            ++resume_from_checkpoint=null \
            ++init_from_pretrained_weights=null

        echo "  ✓ Exp ${EXP_ID} (${NAME}) finished at $(date)"
    done

    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Phase 1 Complete: All training finished"
    echo "═══════════════════════════════════════════"
fi


# ===== Evaluation Phase =====
echo ""
echo "═══════════════════════════════════════════"
echo "  Phase 2: Evaluating Ablation Checkpoints"
echo "═══════════════════════════════════════════"

SAKURA_FLAG=""
if [ "$EVAL_SAKURA" = true ]; then
    SAKURA_FLAG="--sakura"
fi

bash "${ROOT_DIR}/scripts/eval_ablation_study.sh" "${OUTPUT_BASE}" ${SAKURA_FLAG}


echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ablation Study Complete!                                ║"
echo "║  Results: ${OUTPUT_BASE}/eval_results"
echo "╚════════════════════════════════════════════════════════════╝"
