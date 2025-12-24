# Evaluation Scripts Configuration Guide

## Overview

The evaluation scripts (`mmau_eval.py` and `sakura_eval.py`) automatically load model configuration from the pretrained checkpoint using `DeSTA25AudioModel.from_pretrained()`.

## Expected ORCA Configuration

When evaluating models trained with the current ORCA architecture, ensure your checkpoint includes:

### **Model Components**

```yaml
encoder:
  model_id: openai/whisper-large-v3  # Standard version, not turbo
  freeze: true

llm:
  model_id: Qwen/Qwen3-0.6B
  freeze: true

connector:
  mode: orca_hybrid
```

### **ORCA Settings**

```yaml
orca:
  enabled: true
  
  # Global Branch
  global_num_tokens: 64
  global_cross_attn: true
  target_layer_ids: [7, 15, 23, 31]  # 4 selected layers
  
  # Local Branch
  local_enabled: true
  local_downsample: 4                # 4x downsample for efficiency
  local_kernel_size: 5
  
  # Deep Injection
  deep_injection_enabled: true
  gate_init: 0.1
  audio_position_scale: 2.5          # Adjusted for 4x downsample
  
  # Losses (with Global-Local orthogonality)
  ortho_diversity_weight: 0.05       # L_ortho_diversity
  ortho_weight_qformer_local: 0.05   # L_ortho_qformer_local (new!)
  align_weight_local: 0.05           # L_align_layerwise
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
print(f"Local downsample: {model.config.orca_local_downsample}")
print(f"Target layers: {model.connector.target_layer_ids}")
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
