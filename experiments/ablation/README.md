# Ablation Scripts

Ablation study and comparative analysis scripts.

## Directory Structure

```
ablation/
├── linear_probing.py      # P1-1: Linear probe for gender/emotion
├── cross_dataset_eval.py  # Cross-dataset generalization test
└── sensitivity_analysis.py # Hyperparameter sensitivity analysis
```

## Scripts

### P1-1: Linear Probing

Test if frozen features contain speaker/emotion information.

```bash
python linear_probing.py --model <model_path> --task both --samples 1000
```

### Other Ablations

```bash
# Cross-dataset generalization
python cross_dataset_eval.py --checkpoint <checkpoint> --datasets iemocap meld

# Sensitivity analysis
python sensitivity_analysis.py --analysis acd_alpha --checkpoint <checkpoint>
```
