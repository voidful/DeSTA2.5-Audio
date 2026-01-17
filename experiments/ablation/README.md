# Ablation Scripts

Ablation study and comparative analysis scripts.

## Directory Structure

```
ablation/
├── liar_transcript/       # P0-2: Liar transcript test
│   ├── liar_generator.py  # Generate contradictory transcripts
│   └── liar_eval.py       # Evaluate model on liar data
├── linear_probing.py      # P1-1: Linear probe for gender/emotion
├── cross_dataset_eval.py  # Cross-dataset generalization test
└── sensitivity_analysis.py # Hyperparameter sensitivity analysis
```

## Scripts

### P0-2: Liar Transcript Test

Tests if model follows audio truth when transcript is intentionally wrong.

```bash
# Step 1: Generate liar data
python liar_transcript/liar_generator.py \
    --samples-per-task 150 \
    --output-dir liar_data

# Step 2: Evaluate
python liar_transcript/liar_eval.py \
    --model <model_path> \
    --liar-data liar_data/liar_samples.jsonl
```

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
