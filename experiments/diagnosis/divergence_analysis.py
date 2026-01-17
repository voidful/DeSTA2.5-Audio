"""
P0-3: Divergence Rate Analysis

For DPO training to be effective, we need samples where the audio-conditioned
response differs from the text-only response. This script analyzes what
percentage of training samples have this property.

The "divergence rate" measures: What fraction of samples require audio 
information to get the correct answer?

Method:
1. Load training data (JSONL with prompt + response)
2. For each sample, use a text-only LLM to predict the answer
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_task_category(sample_id: str) -> str:
    """Extract task category from sample ID.
    
    Sample ID format varies, but typically contains task info.
    Examples:
    - "emotion_recognition/sample_001" -> "emotion_recognition"
    - "gender_det_sample123" -> "gender"
    - "animal_sounds_0042" -> "animal"
    """
    sample_id_lower = sample_id.lower()
    
    # Try to extract from path
    if "/" in sample_id:
        parts = sample_id.split("/")
        return parts[0]
    
    # Pattern matching for common task types
    task_patterns = {
        "emotion": ["emotion", "emo", "sentiment", "feeling"],
        "gender": ["gender", "sex", "male", "female", "speaker"],
        "animal": ["animal", "creature", "wildlife", "species"],
        "sound": ["sound", "audio", "noise", "acoustic"],
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
    # Remove punctuation at the end
    answer = re.sub(r'[.!?]+$', '', answer)
    # Remove common prefixes
    for prefix in ["the answer is", "i think", "it sounds like", "this is"]:
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
    
    # One contains the other
    if pred_norm in gt_norm or gt_norm in pred_norm:
        return True
    
    # First word match (for single-word answers like "male", "happy", "dog")
    pred_first = pred_norm.split()[0] if pred_norm.split() else ""
    gt_first = gt_norm.split()[0] if gt_norm.split() else ""
    if pred_first and gt_first and pred_first == gt_first:
        return True
    
    return False


def load_text_only_model(model_id: str, device: str = "cuda"):
    """Load a text-only LLM for generating predictions."""
    logger.info(f"Loading text-only model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    return model, tokenizer


def generate_text_only_prediction(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50
) -> str:
    """Generate prediction using text-only model."""
    # Remove audio markers from prompt
    prompt_clean = re.sub(r'<\|AUDIO\|>', '[AUDIO]', prompt)
    prompt_clean = re.sub(r'<start_audio>.*?<end_audio>', '[AUDIO]', prompt_clean, flags=re.DOTALL)
    
    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt_clean}]
        try:
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            input_text = prompt_clean
    else:
        input_text = prompt_clean
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return prediction.strip()


def analyze_divergence(
    manifest_path: str,
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    num_samples: int = None,
    device: str = "cuda",
    output_dir: str = "divergence_results",
    dry_run: bool = False
) -> Dict:
    """Analyze divergence rate of training data.
    
    Args:
        manifest_path: Path to JSONL training manifest
        model_id: Text-only model for predictions
        num_samples: Number of samples to analyze (None = all)
        device: Device for model
        output_dir: Directory to save results
        dry_run: If True, only count samples without running model
    
    Returns:
        Dictionary with divergence statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    logger.info(f"Loading samples from: {manifest_path}")
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    total_samples = len(samples)
    logger.info(f"Total samples: {total_samples}")
    
    if num_samples:
        samples = samples[:num_samples]
        logger.info(f"Analyzing first {len(samples)} samples")
    
    if dry_run:
        logger.info("Dry run mode - counting task distribution only")
        task_counts = defaultdict(int)
        for sample in samples:
            sample_id = sample.get('id', '')
            task = extract_task_category(sample_id)
            task_counts[task] += 1
        
        return {
            "total_samples": len(samples),
            "task_distribution": dict(task_counts),
            "mode": "dry_run"
        }
    
    # Load model
    model, tokenizer = load_text_only_model(model_id, device)
    
    # Analyze samples
    results = []
    task_stats = defaultdict(lambda: {"total": 0, "divergent": 0})
    
    for sample in tqdm(samples, desc="Analyzing divergence"):
        sample_id = sample.get('id', '')
        prompt = sample.get('prompt', '')
        ground_truth = sample.get('response', '')
        task = extract_task_category(sample_id)
        
        if not prompt or not ground_truth:
            continue
        
        # Generate text-only prediction
        prediction = generate_text_only_prediction(model, tokenizer, prompt)
        
        # Check if divergent
        is_match = answers_match(prediction, ground_truth)
        is_divergent = not is_match
        
        task_stats[task]["total"] += 1
        if is_divergent:
            task_stats[task]["divergent"] += 1
        
        results.append({
            "id": sample_id,
            "task": task,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "ground_truth": ground_truth,
            "text_only_pred": prediction,
            "is_divergent": is_divergent
        })
    
    # Calculate statistics
    total_analyzed = len(results)
    total_divergent = sum(1 for r in results if r["is_divergent"])
    divergence_rate = total_divergent / total_analyzed if total_analyzed > 0 else 0
    
    # Per-task rates
    task_rates = {}
    for task, stats in task_stats.items():
        if stats["total"] > 0:
            task_rates[task] = {
                "total": stats["total"],
                "divergent": stats["divergent"],
                "rate": stats["divergent"] / stats["total"]
            }
    
    # Summary
    summary = {
        "total_samples_analyzed": total_analyzed,
        "total_divergent": total_divergent,
        "divergence_rate": divergence_rate,
        "divergence_rate_pct": f"{divergence_rate * 100:.2f}%",
        "per_task_rates": task_rates,
        "model_id": model_id
    }
    
    # Save detailed results
    results_file = output_path / "divergence_details.jsonl"
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    summary_file = output_path / "divergence_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("DIVERGENCE ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples analyzed: {total_analyzed}")
    logger.info(f"Divergent samples (text-only != ground-truth): {total_divergent}")
    logger.info(f"Overall divergence rate: {divergence_rate * 100:.2f}%")
    logger.info("-" * 60)
    logger.info("Per-Task Divergence Rates:")
    for task, stats in sorted(task_rates.items()):
        logger.info(f"  {task}: {stats['rate']*100:.1f}% ({stats['divergent']}/{stats['total']})")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="P0-3: Divergence Rate Analysis")
    parser.add_argument(
        "--manifest", 
        type=str, 
        default="/work/voidful2nlp/desta/qwen3_4b_desta_fixed.jsonl",
        help="Path to training JSONL manifest"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Text-only model for predictions"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None,
        help="Number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="divergence_results",
        help="Output directory"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Only count samples, don't run model"
    )
    
    args = parser.parse_args()
    
    analyze_divergence(
        manifest_path=args.manifest,
        model_id=args.model,
        num_samples=args.samples,
        device=args.device,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
