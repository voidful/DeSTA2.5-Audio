#!/bin/bash
# Verification script to check experiment configurations

echo "========================================="
echo "Verifying Minimal Ablation Study Setup"
echo "========================================="
echo ""

# Check if all required files exist
echo "Checking experiment scripts..."
files=(
    "exp0_baseline.sbatch"
    "exp1_orca_architecture.sbatch"
    "exp2_add_orthogonality.sbatch"
    "exp3_full_orca.sbatch"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING!)"
        all_exist=false
    fi
done
echo ""

# Check if archive directory exists
if [ -d "archive" ]; then
    echo "✓ Archive directory exists"
    echo "  Archived files:"
    ls -1 archive/ | sed 's/^/    /'
else
    echo "✗ Archive directory missing"
fi
echo ""

# Verify key configuration differences
echo "Verifying experiment configurations..."
echo ""

echo "Exp 0 (Baseline):"
grep -E "connector_mode=|model.orca.enabled=" exp0_baseline.sbatch | sed 's/^/  /'
echo ""

echo "Exp 1 (Architecture):"
grep -E "model.orca.enabled=|model.orca.ortho_diversity_weight=" exp1_orca_architecture.sbatch | sed 's/^/  /'
echo ""

echo "Exp 2 (Orthogonality):"
grep -E "model.orca.ortho_diversity_weight=|model.orca.ortho_weight_qformer_local=" exp2_add_orthogonality.sbatch | sed 's/^/  /'
echo ""

echo "Exp 3 (Full ORCA):"
echo "  (Uses default config - no overrides)"
echo ""

# Summary
echo "========================================="
if [ "$all_exist" = true ]; then
    echo "✓ All experiment scripts are present"
    echo "✓ Ready to submit experiments"
    echo ""
    echo "To submit all experiments, run:"
    echo "  bash submit_all.sh"
else
    echo "✗ Some files are missing"
    echo "Please check the setup"
fi
echo "========================================="
