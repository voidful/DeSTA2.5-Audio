# Evaluation Scripts Configuration Guide

## Overview

The evaluation scripts (`mmau_eval.py` and `sakura_eval.py`) automatically load model configuration from the pretrained checkpoint using `DeSTA25AudioModel.from_pretrained()`.

## Expected ORCA-DeSTA Configuration

When evaluating models trained with the paper method, ensure your checkpoint includes:

### **Model Components**

```yaml
encoder:
  model_id: openai/whisper-large-v3  # Standard version, not turbo
  freeze: true

llm:
  model_id: Qwen/Qwen3-0.6B
  freeze: true

connector:
  mode: orca_desta
```

### **ORCA-DeSTA Settings**

```yaml
orca_desta:
  num_groups: 8
  queries_per_group: 8
  inter_group_weight: 0.1
  intra_group_weight: 0.01

variational_grouping:
  enabled: true
  kl_weight: 0.01

modality_dpo:
  enabled: true
  beta: 0.1

asr_dropout:
  prob: 0.2
```

## Usage

### **MMAU Evaluation**

```bash
cd examples/evaluation

# Evaluate on test_mini split
python mmau_eval.py --model_id voidful/DeSTA2.5-Qwen3-0.6B-ORCA

# Evaluate on full test split
python mmau_eval.py --model_id voidful/DeSTA2.5-Qwen3-0.6B-ORCA --split test

# Limit samples for quick testing
python mmau_eval.py --model_id voidful/DeSTA2.5-Qwen3-0.6B-ORCA --max_samples 100
```

### **Sakura Evaluation**

```bash
cd examples/evaluation

# Evaluate all 4 datasets × 2 hop types
python sakura_eval.py

# Results will be saved to desta_sakura_results/
```

## Model ID

Update the default model ID in the scripts to point to your trained checkpoint:

```python
# mmau_eval.py
DEFAULT_MODEL_ID = "voidful/DeSTA2.5-Qwen3-0.6B-ORCA"

# sakura_eval.py
DESTA_MODEL_ID = "voidful/DeSTA2.5-Qwen3-0.6B-ORCA"
```

Or specify via command line:

```bash
python mmau_eval.py --model_id /path/to/your/checkpoint
```

## Expected Performance

Based on Sakura benchmark analysis with current ORCA configuration:

| Metric | Expected | Notes |
|--------|----------|-------|
| Multi-speaker | 42~43 | Improved with 4x downsample |
| Language-Single | 68~70 | +3~5 from losses |
| Language-Multi | 42~44 | +3~6 improvement |
| Overall Hmean | 49~50 | +2~3 overall |

## Troubleshooting

### Configuration Mismatch

If you see unexpected results, verify the loaded configuration:

```python
from desta import DeSTA25AudioModel

model = DeSTA25AudioModel.from_pretrained("your_model_id")
print(f"Whisper: {model.config.encoder_model_id}")
print(f"Connector mode: {model.config.connector_mode}")
print(f"Groups: {model.config.orca_r1_num_groups}")
print(f"Queries/group: {model.config.orca_r1_queries_per_group}")
print(f"Variational: {model.config.variational_grouping_enabled}")
print(f"ACP: {model.config.modality_dpo_enabled}")
```

### OOM Issues

If evaluation runs out of memory:

- Reduce batch size in the evaluation script
- Use gradient checkpointing (already enabled in model)
- Evaluate on smaller splits first

## Notes

- Both scripts use LLM judge (Qwen3-4B) for answer evaluation
- Results are saved as JSONL files for detailed analysis
- The scripts automatically handle audio preprocessing and format conversion
