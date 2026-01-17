#!/bin/bash
# ==========================================
# ORCA Paper Experiments - Submit All
# 
# 用法:
#   ./submit_all.sh           # 提交所有實驗
#   ./submit_all.sh p0        # 只提交 P0 (Critical)
#   ./submit_all.sh p1        # 只提交 P1 (Validation)
#   ./submit_all.sh p2        # 只提交 P2 (Ablation Training)
#   ./submit_all.sh eval      # 只提交評估實驗
#   ./submit_all.sh train     # 只提交訓練實驗
# ==========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

# 確保 slurm-report 目錄存在
mkdir -p ../../slurm-report

MODE=${1:-all}

echo "========================================"
echo "ORCA Paper Experiments Submission"
echo "Mode: ${MODE}"
echo "========================================"

submit_p0() {
    echo ""
    echo "=== P0: Critical Experiments ==="
    
    echo "Submitting P0-3: Divergence Rate..."
    sbatch P0_3_divergence.sbatch
    
    echo "Submitting P0-1: Match Rate..."
    sbatch P0_1_match_rate.sbatch
    
    echo "Submitting P0-2a: Liar Generation..."
    JOB_LIAR_GEN=$(sbatch --parsable P0_2_liar_gen.sbatch)
    
    echo "Submitting P0-2b: Liar Eval (depends on P0-2a)..."
    sbatch --dependency=afterok:${JOB_LIAR_GEN} P0_2_liar_eval.sbatch
}

submit_p1() {
    echo ""
    echo "=== P1: Validation Experiments ==="
    
    echo "Submitting P1-1: Linear Probing..."
    sbatch P1_1_linear_probe.sbatch
    
    echo "Submitting P1-2: Refusal Rate..."
    sbatch P1_2_refusal.sbatch
}

submit_p2() {
    echo ""
    echo "=== P2: Ablation Training ==="
    
    echo "Submitting P2-1a: Ortho Only..."
    sbatch P2_ortho_only.sbatch
    
    echo "Submitting P2-1b: ASR Dropout Only..."
    sbatch P2_dropout_only.sbatch
    
    echo "Submitting P2-1c: Modality-DPO Only..."
    sbatch P2_dpo_only.sbatch
}

case ${MODE} in
    p0)
        submit_p0
        ;;
    p1)
        submit_p1
        ;;
    p2)
        submit_p2
        ;;
    eval)
        submit_p0
        submit_p1
        ;;
    train)
        submit_p2
        ;;
    all)
        submit_p0
        submit_p1
        submit_p2
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [all|p0|p1|p2|eval|train]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "========================================"
