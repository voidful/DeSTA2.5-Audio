"""
P0-3: Divergence Rate Analysis (Optimized for H100)

For DPO training to be effective, we need samples where the audio-conditioned
response differs from the text-only response. This script analyzes what
percentage of training samples have this property.

The "divergence rate" measures: What fraction of samples require audio 
information to get the correct answer?

Method:
1. Load training data (JSONL with prompt + response)
2. Use text-only LLM to predict answer (BATCHED INFERENCE)
3. Compare text-only prediction with ground truth response
4. Calculate divergence rate: % where text-only != ground-truth

Expected: ~34% divergence (meaning 34% of samples need audio)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import re
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_task_category(sample_id: str) -> str:
    """Extract task category from sample ID."""
    sample_id_lower = sample_id.lower()
    
    # Try to extract from path - common in DESTA data
    # e.g. "voidful/desta/data/emotion_recognition/sample_001"
    if "/" in sample_id:
        parts = sample_id.split("/")
        # usually task is at the index before filename or at start
        for part in parts:
            if any(k in part.lower() for k in ["emotion", "gender", "animal", "sound", "speech", "music"]):
                return part.lower()

    # Pattern matching for common task types
    task_patterns = {
        "emotion": ["emotion", "emo", "sentiment", "feeling"],
        "gender": ["gender", "sex", "male", "female", "speaker"],
        "animal": ["animal", "creature", "wildlife", "species"],
        "sound": ["sound", "audio", "noise", "acoustic", "environment"],
        "speech": ["speech", "asr", "transcript", "spoken"],
        "music": ["music", "song", "melody", "instrument"],
    }
    
    for task, patterns in task_patterns.items():
        for pattern in patterns:
            if pattern in sample_id_lower:
                return task
    
    return "other"


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase, strip whitespace
    answer = answer.lower().strip()
    # Remove punctuation at the end and typical generated artifacts
    answer = re.sub(r'[.!?]+$', '', answer)
    # Remove common prefixes
    for prefix in ["the answer is", "i think", "it sounds like", "this is", "predicted answer:", "answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    return answer


def answers_match(pred: str, gt: str) -> bool:
    """Check if prediction matches ground truth (flexible matching)."""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # One contains the other (robustness for short/long answers)
    if pred_norm in gt_norm or gt_norm in pred_norm:
        return True
    
    # First word match (for single-word answers like "male", "happy", "dog")
    pred_first = pred_norm.split()[0] if pred_norm.split() else ""
    gt_first = gt_norm.split()[0] if gt_norm.split() else ""
    if pred_first and gt_first and pred_first == gt_first:
        return True
    
    return False


class DivergenceDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample.get('prompt', '')
        
        # Determine task early for tracking
        sample_id = sample.get('id', str(idx))
        task = extract_task_category(sample_id)
        
        # Clean prompt (remove audio tokens)
        prompt_clean = re.sub(r'<\|AUDIO\|>', '[AUDIO]', prompt)
        prompt_clean = re.sub(r'<start_audio>.*?<end_audio>', '[AUDIO]', prompt_clean, flags=re.DOTALL)
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt_clean}]
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                input_text = prompt_clean
        else:
            input_text = prompt_clean
            
        return {
            "id": sample_id,
            "input_text": input_text,
            "ground_truth": sample.get('response', ''),
            "task": task,
            "prompt_raw": prompt
        }


def analyze_divergence(
    manifest_path: str,
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    num_samples: int = None,
    device: str = "cuda",
    batch_size: int = 32,
    output_dir: str = "divergence_results",
    dry_run: bool = False
) -> Dict:
    """Analyze divergence rate of training data (optimized)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    logger.info(f"Loading samples from: {manifest_path}")
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    total_samples = len(samples)
    logger.info(f"Total samples in manifest: {total_samples}")
    
    if num_samples:
        samples = samples[:num_samples]
        logger.info(f"Analyzing subset of {len(samples)} samples")
    
    if dry_run:
        logger.info("Dry run mode - counting task distribution only")
        task_counts = defaultdict(int)
        for sample in samples:
            sample_id = sample.get('id', '')
            task = extract_task_category(sample_id)
            task_counts[task] += 1
        
        return {"mode": "dry_run", "task_counts": dict(task_counts)}

    # Load Model & Tokenizer
    logger.info(f"Loading text-only model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    model.eval()
    
    # Compile model for faster generation if supported (optional, can be tricky with dynamic shapes)
    # if hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model)
    #         logger.info("Model compiled with torch.compile")
    #     except Exception as e:
    #         logger.warning(f"Failed to compile model: {e}")

    # Create Dataset & DataLoader
    dataset = DivergenceDataset(samples, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda x: x # Simple list collation
    )
    
    results = []
    task_stats = defaultdict(lambda: {"total": 0, "divergent": 0})
    
    logger.info(f"Starting batched inference with batch_size={batch_size}...")
    
    # Iterate through batches
    for batch_items in tqdm(dataloader, desc="Processing batches"):
        # Prepare batch inputs
        input_texts = [item["input_text"] for item in batch_items]
        ground_truths = [item["ground_truth"] for item in batch_items]
        ids = [item["id"] for item in batch_items]
        tasks = [item["task"] for item in batch_items]
        
        # Tokenize batch
        inputs = tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(model.device)
        
        input_len = inputs['input_ids'].shape[1]
        
        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
        
        # Process outputs
        for i, output in enumerate(outputs):
            # Decode only new tokens
            new_tokens = output[input_len:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            gt = ground_truths[i]
            task = tasks[i]
            sample_id = ids[i]
            
            if not gt:
                continue
                
            is_match = answers_match(prediction, gt)
            is_divergent = not is_match
            
            task_stats[task]["total"] += 1
            if is_divergent:
                task_stats[task]["divergent"] += 1
            
            results.append({
                "id": sample_id,
                "task": task,
                "ground_truth": gt,
                "text_only_pred": prediction,
                "is_divergent": is_divergent
            })

    # Calculate statistics
    total_analyzed = len(results)
    total_divergent = sum(1 for r in results if r["is_divergent"])
    divergence_rate = total_divergent / total_analyzed if total_analyzed > 0 else 0
    
    # Per-task summary
    task_rates = {}
    for task, stats in task_stats.items():
        if stats["total"] > 0:
            task_rates[task] = {
                "total": stats["total"],
                "divergent": stats["divergent"],
                "rate": stats["divergent"] / stats["total"]
            }
            
    summary = {
        "model_id": model_id,
        "total_samples": total_analyzed,
        "total_divergent": total_divergent,
        "divergence_rate": divergence_rate,
        "divergence_rate_pct": f"{divergence_rate * 100:.2f}%",
        "per_task_stats": task_rates
    }
    
    # Save results
    results_file = output_path / "divergence_details.jsonl"
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    summary_file = output_path / "divergence_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info("=" * 60)
    logger.info(f"Overall Divergence Rate: {divergence_rate*100:.2f}%")
    logger.info("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="P0-3: Divergence Rate (Batched)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Model ID")
    parser.add_argument("--samples", type=int, default=None, help="Num samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="divergence_results")
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    analyze_divergence(
        manifest_path=args.manifest,
        model_id=args.model,
        num_samples=args.samples,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
