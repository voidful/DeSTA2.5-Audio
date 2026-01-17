# ORCA-R1 Ablation Study

Progressive ablation from baseline DeSTA2.5 to full ORCA-R1.

## Experiment Progression

```
A0: Baseline QFormer
 ↓
A1: Flat 64 queries (no grouping)
 ↓
A2: Grouped 8×8 (no losses)
 ↓
B1: +Inter-group (0.05) → B2: (0.1)
 ↓
B3: +Intra-group (0.01)
 ↓
C1: +IV discriminator (0.05) → C2: (0.1) = Full ORCA
```

## Quick Start

```bash
# Edit paths in scripts first
vim *.sbatch  # Update /path/to/DeSTA2.5-Audio

# Submit all
./submit_all.sh

# Or submit individually
sbatch A0_baseline_qformer.sbatch
```

## ACD Evaluation (Inference)

After training C2, test different ACD alphas:
```bash
python examples/evaluation/eval.py \
    --checkpoint outputs/ablation/C2_iv_0.1_full/checkpoint-final \
    --acd_alpha 0.5  # Try: 0.3, 0.5, 0.7, 1.0
```
