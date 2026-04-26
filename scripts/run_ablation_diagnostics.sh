#!/bin/bash
#
# ORCA-DeSTA Component Ablation — Diagnostic Validation Script
#
# Runs the three diagnostic experiments from Section 3 on each ablation
# checkpoint to verify that each component reduces its targeted failure mode:
#
#   Diagnostic 1: Query Redundancy (mean off-diagonal cosine similarity)
#                 → Targeted by Component 1: Groupwise Orthogonality
#
#   Diagnostic 2: Acoustic Information Loss (cross-speaker variance)
#                 → Targeted by Component 2: Stochastic Perturbation
#
#   Diagnostic 3: Acoustic Preference Violation (APV margin & violation rate)
#                 → Targeted by Component 3: ACP
#
# Usage:
#   bash scripts/run_ablation_diagnostics.sh /path/to/ablation/outputs
#
# Requires CREMA-D dataset to be accessible.

set -e

ABLATION_DIR="${1:?Usage: $0 <ablation_output_dir>}"
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RESULT_DIR="${ABLATION_DIR}/diagnostic_results"
TIMESTAMP=$(date +%y%m%d-%H%M)

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/voidful2nlp/.cache/huggingface}"

mkdir -p "${RESULT_DIR}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ORCA-DeSTA Diagnostic Validation (Section 3)            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  D1: Query Redundancy Analysis     (§3.1)                ║"
echo "║  D2: Acoustic Information Loss     (§3.2)                ║"
echo "║  D3: Acoustic Preference Violation (§3.3)                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ===== Discover checkpoints =====
declare -A CHECKPOINT_PATHS
declare -A ABLATION_DESC
ABLATION_DESC[baseline]="Exp 0: Baseline"
ABLATION_DESC[plus_ortho]="Exp 1: + Groupwise Orthogonality"
ABLATION_DESC[plus_acp]="Exp 2: + ACP"
ABLATION_DESC[plus_stoch]="Exp 3: + Stochastic Perturbation"
ABLATION_DESC[full]="Exp 4: Full Model"

echo "Discovering checkpoints..."
for exp_dir in "${ABLATION_DIR}"/*/; do
    [ -d "${exp_dir}" ] || continue
    dir_name=$(basename "${exp_dir}")
    label=$(echo "${dir_name}" | sed 's/^[0-9]\{6\}-[0-9]\{4\}_//')

    [ -n "${ABLATION_DESC[$label]+isset}" ] || continue

    ckpt=""
    for candidate in "${exp_dir}"/checkpoint-*/model.safetensors; do
        [ -f "$candidate" ] && ckpt=$(dirname "$candidate")
    done
    if [ -z "$ckpt" ]; then
        for candidate in "${exp_dir}"/checkpoint-*/pytorch_model.bin; do
            [ -f "$candidate" ] && ckpt=$(dirname "$candidate")
        done
    fi

    if [ -n "$ckpt" ]; then
        echo "  ✓ ${label} → ${ckpt}"
        CHECKPOINT_PATHS[$label]="${ckpt}"
    fi
done

echo ""

# ===== Run diagnostics on each checkpoint =====
for label in baseline plus_ortho plus_acp plus_stoch full; do
    ckpt="${CHECKPOINT_PATHS[$label]:-}"
    [ -z "$ckpt" ] && continue

    desc="${ABLATION_DESC[$label]}"
    out_dir="${RESULT_DIR}/${label}"
    mkdir -p "${out_dir}"

    echo "════════════════════════════════════════"
    echo "  ${desc}"
    echo "  Checkpoint: ${ckpt}"
    echo "════════════════════════════════════════"

    python3 "${ROOT_DIR}/scripts/ablation_diagnostics.py" \
        --model_id "${ckpt}" \
        --output_dir "${out_dir}" \
        --label "${label}" \
        2>&1 | tee "${out_dir}/diagnostics.log"

    echo "  ✓ Done: ${label}"
    echo ""
done


# ===== Generate consolidated comparison =====
echo "════════════════════════════════════════"
echo "  Generating consolidated diagnostic table..."
echo "════════════════════════════════════════"

python3 << 'PYEOF'
import os, json, sys, glob

result_dir = os.environ.get("RESULT_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")

labels = ["baseline", "plus_ortho", "plus_acp", "plus_stoch", "full"]
descs = {
    "baseline":    "Baseline",
    "plus_ortho":  "+ Orthogonality",
    "plus_acp":    "+ ACP",
    "plus_stoch":  "+ Stochastic",
    "full":        "Full Model",
}

print("\n" + "=" * 90)
print("Diagnostic Validation: Component Ablation")
print("=" * 90)
print(f"{'Configuration':<22s} │ {'Cosine Sim':>10s} │ {'X-Speaker Var':>13s} │ {'APV Mean Δ':>10s} │ {'APV VR':>7s}")
print("─" * 22 + "─┼─" + "─" * 10 + "─┼─" + "─" * 13 + "─┼─" + "─" * 10 + "─┼─" + "─" * 7)

for label in labels:
    diag_file = os.path.join(result_dir, label, "diagnostics.json")
    if not os.path.exists(diag_file):
        print(f"{descs[label]:<22s} │ {'N/A':>10s} │ {'N/A':>13s} │ {'N/A':>10s} │ {'N/A':>7s}")
        continue

    with open(diag_file) as f:
        d = json.load(f)

    cosine = d.get("query_cosine_sim", "N/A")
    xvar = d.get("cross_speaker_var", "N/A")
    apv_mean = d.get("apv_mean_margin", "N/A")
    apv_vr = d.get("apv_violation_rate", "N/A")

    def fmt(v, pct=False):
        if isinstance(v, (int, float)):
            return f"{v:.4f}" if not pct else f"{v:.2f}%"
        return str(v)

    print(f"{descs[label]:<22s} │ {fmt(cosine):>10s} │ {fmt(xvar):>13s} │ {fmt(apv_mean):>10s} │ {fmt(apv_vr, True):>7s}")

print("=" * 90)
print("Lower cosine sim = less redundancy  |  Higher variance = better acoustic preservation")
print("Higher APV mean = stronger audio preference  |  Lower VR = fewer violations")
print()
PYEOF

echo ""
echo "Diagnostic validation complete. Results: ${RESULT_DIR}"
