#!/bin/bash
# Quick submission script for all minimal ablation experiments

cd /work/voidful2nlp/DeSTA2.5-Audio/examples/train/ablation_sbatch

echo "========================================="
echo "Submitting Minimal ORCA Ablation Study"
echo "========================================="
echo ""

echo "[1/4] Submitting Exp 0: DeSTA2.5 Baseline..."
sbatch exp0_baseline.sbatch
echo ""

echo "[2/4] Submitting Exp 1: ORCA Architecture..."
sbatch exp1_orca_architecture.sbatch
echo ""

echo "[3/4] Submitting Exp 2: + Orthogonality..."
sbatch exp2_add_orthogonality.sbatch
echo ""

echo "[4/4] Submitting Exp 3: Full ORCA..."
sbatch exp3_full_orca.sbatch
echo ""

echo "========================================="
echo "All experiments submitted!"
echo "========================================="
echo ""
echo "Check status with: squeue -u \$USER"
echo "View logs in: ./slurm-report/"
echo ""
