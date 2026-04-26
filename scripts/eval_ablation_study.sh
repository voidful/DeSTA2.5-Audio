#!/bin/bash
#
# ORCA-DeSTA Component Ablation — Evaluation Script
#
# After training the 5 ablation experiments, this script:
#   1. Evaluates each checkpoint on MMAU-mini (same as Table 5)
#   2. Optionally evaluates on SAKURA for multi-hop reasoning
#   3. Generates a consolidated comparison table
#
# Usage:
#   bash scripts/eval_ablation_study.sh /path/to/ablation/outputs
#   bash scripts/eval_ablation_study.sh /path/to/ablation/outputs --sakura
#
# SLURM usage:
#   sbatch --wrap="bash scripts/eval_ablation_study.sh /path/to/ablation/outputs"

set -e

# ===== Configuration =====
ABLATION_DIR="${1:?Usage: $0 <ablation_output_dir> [--sakura]}"
EVAL_SAKURA=false
if [[ "$2" == "--sakura" ]]; then
    EVAL_SAKURA=true
fi

ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
RESULT_DIR="${ABLATION_DIR}/eval_results"
TIMESTAMP=$(date +%y%m%d-%H%M)

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/voidful2nlp/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "${RESULT_DIR}"

# Ablation labels (must match training script)
declare -A ABLATION_LABELS
ABLATION_LABELS[baseline]="Baseline (no aux losses)"
ABLATION_LABELS[plus_ortho]="+ Groupwise Orthogonality"
ABLATION_LABELS[plus_acp]="+ ACP"
ABLATION_LABELS[plus_stoch]="+ Stochastic Perturbation"
ABLATION_LABELS[full]="Full Model (+ ASR Dropout)"

echo "============================================================"
echo "  ORCA-DeSTA Component Ablation — Evaluation"
echo "  Ablation dir: ${ABLATION_DIR}"
echo "  Results dir: ${RESULT_DIR}"
echo "  Eval SAKURA: ${EVAL_SAKURA}"
echo "============================================================"


# ===== Discover checkpoints =====
echo ""
echo "Discovering trained checkpoints..."

declare -A CHECKPOINT_PATHS
for exp_dir in "${ABLATION_DIR}"/*/; do
    if [ ! -d "${exp_dir}" ]; then continue; fi

    dir_name=$(basename "${exp_dir}")

    # Extract label from directory name (format: YYMMDD-HHMM_label)
    label=$(echo "${dir_name}" | sed 's/^[0-9]\{6\}-[0-9]\{4\}_//')

    if [ -z "${ABLATION_LABELS[$label]+isset}" ]; then
        echo "  Skipping unknown experiment: ${dir_name}"
        continue
    fi

    # Find best checkpoint (last epoch or safetensors)
    ckpt=""
    # Try safetensors format first
    for candidate in "${exp_dir}"/checkpoint-*/model.safetensors; do
        if [ -f "$candidate" ]; then
            ckpt=$(dirname "$candidate")
        fi
    done
    # Fallback to pytorch_model.bin
    if [ -z "$ckpt" ]; then
        for candidate in "${exp_dir}"/checkpoint-*/pytorch_model.bin; do
            if [ -f "$candidate" ]; then
                ckpt=$(dirname "$candidate")
            fi
        done
    fi

    if [ -n "$ckpt" ]; then
        echo "  Found: ${label} -> ${ckpt}"
        CHECKPOINT_PATHS[$label]="${ckpt}"
    else
        echo "  WARNING: No checkpoint found for ${label} in ${exp_dir}"
    fi
done

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in ${ABLATION_DIR}"
    exit 1
fi


# ===== Run MMAU-mini evaluation =====
echo ""
echo "============================================================"
echo "  Running MMAU-mini evaluation..."
echo "============================================================"

MMAU_RESULTS_FILE="${RESULT_DIR}/mmau_mini_results_${TIMESTAMP}.jsonl"
> "${MMAU_RESULTS_FILE}"

for label in baseline plus_ortho plus_acp plus_stoch full; do
    ckpt="${CHECKPOINT_PATHS[$label]:-}"
    if [ -z "$ckpt" ]; then
        echo "  Skipping ${label}: no checkpoint"
        continue
    fi

    desc="${ABLATION_LABELS[$label]}"
    mmau_out="${RESULT_DIR}/mmau_${label}"
    mkdir -p "${mmau_out}"

    echo ""
    echo "  Evaluating: ${desc}"
    echo "  Checkpoint: ${ckpt}"

    python "${ROOT_DIR}/examples/evaluation/mmau_eval.py" \
        --model_id "${ckpt}" \
        --output_dir "${mmau_out}" \
        --split test_mini \
        2>&1 | tee "${mmau_out}/eval.log"

    # Extract results from output
    echo "{\"label\": \"${label}\", \"description\": \"${desc}\", \"checkpoint\": \"${ckpt}\", \"output_dir\": \"${mmau_out}\"}" >> "${MMAU_RESULTS_FILE}"

    echo "  Done: ${label}"
done


# ===== Run SAKURA evaluation (optional) =====
if [ "$EVAL_SAKURA" = true ]; then
    echo ""
    echo "============================================================"
    echo "  Running SAKURA evaluation..."
    echo "============================================================"

    SAKURA_RESULTS_FILE="${RESULT_DIR}/sakura_results_${TIMESTAMP}.jsonl"
    > "${SAKURA_RESULTS_FILE}"

    for label in baseline plus_ortho plus_acp plus_stoch full; do
        ckpt="${CHECKPOINT_PATHS[$label]:-}"
        if [ -z "$ckpt" ]; then
            echo "  Skipping ${label}: no checkpoint"
            continue
        fi

        desc="${ABLATION_LABELS[$label]}"
        sakura_out="${RESULT_DIR}/sakura_${label}"
        mkdir -p "${sakura_out}"

        echo ""
        echo "  Evaluating SAKURA: ${desc}"

        python "${ROOT_DIR}/examples/evaluation/sakura_eval.py" \
            --model_id "${ckpt}" \
            --output_dir "${sakura_out}" \
            2>&1 | tee "${sakura_out}/eval.log"

        echo "{\"label\": \"${label}\", \"description\": \"${desc}\", \"checkpoint\": \"${ckpt}\", \"output_dir\": \"${sakura_out}\"}" >> "${SAKURA_RESULTS_FILE}"
        echo "  Done: ${label}"
    done
fi


# ===== Generate summary table =====
echo ""
echo "============================================================"
echo "  Generating ablation summary table..."
echo "============================================================"

python3 << 'PYEOF'
import os
import json
import glob
import sys

result_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULT_DIR", ".")

labels_order = ["baseline", "plus_ortho", "plus_acp", "plus_stoch", "full"]
labels_desc = {
    "baseline":    "Baseline (no aux losses)",
    "plus_ortho":  "+ Groupwise Orthogonality",
    "plus_acp":    "+ ACP",
    "plus_stoch":  "+ Stochastic Perturbation",
    "full":        "Full Model (+ ASR Dropout)",
}

print("\n" + "=" * 80)
print("ORCA-DeSTA Component Ablation — Results Summary")
print("=" * 80)

# Collect MMAU results
print("\n--- MMAU-mini Results ---")
print(f"{'Configuration':<38s} {'Total':>7s} {'Sound':>7s} {'Speech':>7s} {'Music':>7s}")
print("-" * 68)

for label in labels_order:
    mmau_dir = os.path.join(result_dir, f"mmau_{label}")
    if not os.path.isdir(mmau_dir):
        print(f"{labels_desc.get(label, label):<38s}  (not evaluated)")
        continue

    # Find the summary JSON
    summary_files = glob.glob(os.path.join(mmau_dir, "*summary*.json")) + \
                    glob.glob(os.path.join(mmau_dir, "*report*.json"))
    if not summary_files:
        # Try to parse from eval.log
        log_file = os.path.join(mmau_dir, "eval.log")
        if os.path.exists(log_file):
            with open(log_file) as f:
                content = f.read()
            # Try to extract accuracy from log output
            print(f"{labels_desc.get(label, label):<38s}  (see {log_file})")
        else:
            print(f"{labels_desc.get(label, label):<38s}  (no results found)")
        continue

    with open(summary_files[-1]) as f:
        data = json.load(f)

    total = data.get("total_accuracy", data.get("accuracy", "N/A"))
    sound = data.get("sound_accuracy", data.get("Sound", "N/A"))
    speech = data.get("speech_accuracy", data.get("Speech", "N/A"))
    music = data.get("music_accuracy", data.get("Music", "N/A"))

    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v*100:.2f}%" if v <= 1.0 else f"{v:.2f}%"
        return str(v)

    print(f"{labels_desc.get(label, label):<38s} {fmt(total):>7s} {fmt(sound):>7s} {fmt(speech):>7s} {fmt(music):>7s}")

# Collect SAKURA results
sakura_found = False
for label in labels_order:
    sakura_dir = os.path.join(result_dir, f"sakura_{label}")
    if os.path.isdir(sakura_dir):
        sakura_found = True
        break

if sakura_found:
    print("\n--- SAKURA Results ---")
    print(f"{'Configuration':<38s} {'Animal S':>9s} {'Animal M':>9s} {'Gender S':>9s} {'Gender M':>9s} {'Emotion S':>10s} {'Emotion M':>10s} {'Lang S':>7s} {'Lang M':>7s}")
    print("-" * 120)
    for label in labels_order:
        sakura_dir = os.path.join(result_dir, f"sakura_{label}")
        if not os.path.isdir(sakura_dir):
            print(f"{labels_desc.get(label, label):<38s}  (not evaluated)")
            continue
        print(f"{labels_desc.get(label, label):<38s}  (see {sakura_dir})")

print("\n" + "=" * 80)
PYEOF

echo ""
echo "All evaluations complete. Results saved to: ${RESULT_DIR}"
echo "Finished at $(date)"
