# Diagnosis Scripts

Diagnostic analysis scripts for understanding and validating ORCA model behavior.

## Scripts

### Core Paper Experiments

| Script | Priority | Purpose |
|--------|----------|---------|
| `match_rate_analysis.py` | P0-1 | Compare predictions with/without audio |
| `refusal_analysis.py` | P1-2 | Count "cannot determine" responses |

### Feature Analysis

| Script | Purpose |
|--------|---------|
| `feature_analysis.py` | PCA variance, effective dimensionality, group metrics |
| `group_probing.py` | Per-group linear probing for attribute specialization |
| `mutual_information.py` | MINE-based conditional mutual information estimation |
| `text_probe.py` | Probe transcription representations |

### Visualization & Runners

| Script | Purpose |
|--------|---------|
| `run_diagnosis.py` | Main entry point for running all diagnostics |
| `run_mmau_observations.py` | MMAU feature extraction with t-SNE visualization |
| `visualizations.py` | Plotting utilities |
| `intervention_experiments.py` | Audio swap, text paraphrase, mismatch tests |

## Usage

```bash
# P0-1: Match rate analysis
python match_rate_analysis.py --model <model_path> --samples-per-task 250

# P1-2: Refusal rate
python refusal_analysis.py --model <model_path> --samples 200

# Full diagnosis
python run_diagnosis.py --model <model_path> --output-dir ./results
```
