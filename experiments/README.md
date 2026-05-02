# Experiments Directory

This directory contains experimental scripts for the ORCA paper.

## Directory Structure

```
experiments/
├── diagnosis/         # Diagnostic analysis scripts
│   ├── match_rate_analysis.py     # P0-1: Audio vs text-only prediction match
│   ├── refusal_analysis.py        # P1-2: Model refusal rate analysis
│   ├── feature_analysis.py        # PCA, effective dimensionality metrics
│   ├── group_probing.py           # Group-wise linear probing
│   ├── intervention_experiments.py # Audio swap/mismatch experiments
│   ├── mutual_information.py      # CMI estimation (MINE)
│   ├── run_diagnosis.py           # Main diagnosis runner
│   ├── run_mmau_observations.py   # MMAU feature extraction & analysis
│   ├── text_probe.py              # Text representation probing
│   └── visualizations.py          # Plotting utilities
│
├── ablation/          # Ablation study scripts
│   ├── linear_probing.py          # P1-1: Linear probe for gender/emotion
│   ├── cross_dataset_eval.py      # Cross-dataset generalization
│   └── sensitivity_analysis.py    # Hyperparameter sensitivity
│
└── evaluation/        # (empty - use examples/evaluation/)
```

## Quick Start

### P0: Critical Experiments (Must Do)

```bash
# P0-1: Match Rate (2-3 days)  
python experiments/diagnosis/match_rate_analysis.py --model voidful/desta25-qwen3-4b
```

### P1: Validation Experiments

```bash
# P1-1: Linear Probing
python experiments/ablation/linear_probing.py --model <model_path> --task both

# P1-2: Refusal Rate
python experiments/diagnosis/refusal_analysis.py --model <model_path> --samples 200
```

### Feature Analysis

```bash
# Run comprehensive diagnosis
python experiments/diagnosis/run_diagnosis.py --model <model_path> --output-dir results/

# MMAU observations with t-SNE
python experiments/diagnosis/run_mmau_observations.py --model <model_path>
```

## File Descriptions

| File | Purpose | Paper Section |
|------|---------|---------------|
| `match_rate_analysis.py` | Compare predictions with/without audio | §4.2 |
| `linear_probing.py` | Probe frozen features for attributes | §4.4 |
| `refusal_analysis.py` | Count "cannot determine" responses | §4.5 |
| `feature_analysis.py` | PCA, effective dimensionality | §3 |
| `group_probing.py` | Per-group attribute probing | §4 |
| `intervention_experiments.py` | Audio swap, mismatch tests | §4 |
| `mutual_information.py` | MINE-based CMI estimation | §5 |
