# P0-2: Liar Transcript Test

Tests if ORCA correctly uses audio information when given intentionally wrong transcripts.

## Concept

- **Emotion**: Audio shows sad speaker, transcript says "I'm so happy!"
- **Gender**: Male voice audio, transcript says "As a woman, I..."  
- **Animal**: Dog barking audio, transcript says "The cat is meowing"

## Expected Results

| Model | Accuracy (follows audio) |
|-------|--------------------------|
| DeSTA | ~35% (follows wrong text) |
| ORCA  | ~70% (uses audio truth) |

## Usage

```bash
# Step 1: Generate liar samples
python liar_generator.py \
    --samples-per-task 150 \
    --output-dir liar_transcript_data

# Step 2: Evaluate DeSTA baseline
python liar_eval.py \
    --model voidful/desta25-qwen3-4b \
    --liar-data liar_transcript_data/liar_samples.jsonl \
    --output-dir liar_eval_desta

# Step 3: Evaluate ORCA-R1
python liar_eval.py \
    --model voidful/desta25-4b-orca-r1 \
    --liar-data liar_transcript_data/liar_samples.jsonl \
    --output-dir liar_eval_orca
```

## Output

- `liar_samples.jsonl`: Generated samples with liar transcripts
- `liar_eval_summary.json`: Accuracy and fooled rate statistics
- `liar_eval_details.jsonl`: Per-sample predictions and correctness
