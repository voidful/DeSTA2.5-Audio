#!/bin/bash
# Quick submission script for loss-only ablation experiments

cd /work/voidful2nlp/DeSTA2.5-Audio/examples/train/ablation_sbatch

echo "========================================="
echo "Submitting Loss-Only Ablation Study"
echo "========================================="
echo ""

echo "[1/3] Submitting Exp 0: DeSTA2.5 Baseline..."
sbatch exp0_baseline.sbatch
echo ""

echo "[2/3] Submitting Exp 1: + Diversity Loss..."
sbatch exp1_add_diversity.sbatch
echo ""

echo "[3/3] Submitting Exp 2: + Alignment Loss..."
sbatch exp2_add_alignment.sbatch
echo ""

echo "========================================="
echo "All experiments submitted!"
echo "========================================="
echo ""
echo "Check status with: squeue -u \$USER"
echo "View logs in: ./slurm-report/"
echo ""
