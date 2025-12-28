#!/bin/bash
# Verification script to check experiment configurations

echo "========================================="
echo "Verifying Loss-Only Ablation Study Setup"
echo "========================================="
echo ""

# Check if all required files exist
echo "Checking experiment scripts..."
files=(
    "exp0_baseline.sbatch"
    "exp1_add_diversity.sbatch"
    "exp2_add_alignment.sbatch"
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

# Verify key configuration differences
echo "Verifying experiment configurations..."
echo ""

echo "Exp 0 (Baseline):"
grep -E "connector_mode=|model.orca.enabled=" exp0_baseline.sbatch | sed 's/^/  /'
echo ""

echo "Exp 1 (+ Diversity):"
grep -E "model.orca.ortho_diversity_weight=|model.orca.align_weight_local=" exp1_add_diversity.sbatch | sed 's/^/  /'
echo ""

echo "Exp 2 (+ Alignment):"
grep -E "model.orca.ortho_diversity_weight=|model.orca.align_weight_local=" exp2_add_alignment.sbatch | sed 's/^/  /'
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
