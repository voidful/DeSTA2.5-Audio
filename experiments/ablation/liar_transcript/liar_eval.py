"""
P0-2: Liar Transcript Evaluation

Evaluates model performance when given intentionally contradictory transcripts.

The test: Model receives audio with a WRONG transcript and must answer correctly
based on what it HEARS, not what the transcript says.

Expected results:
- DeSTA (ASR-reliant): ~35% accuracy (follows wrong transcript)
- ORCA (audio-aware): ~70% accuracy (uses actual audio info)
"""

import os
import json
import argparse
import logging
import wave
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np

import torch
from desta import DeSTA25AudioModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    import re
    answer = answer.lower().strip()
    answer = re.sub(r'^(the |a |an )', '', answer)
    answer = re.sub(r'[.!?,;:]+$', '', answer)
    return answer


def answer_matches_audio_truth(prediction: str, audio_truth: str) -> bool:
    """Check if prediction matches the audio ground truth."""
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(audio_truth)
    
    # Exact match
    if pred_norm == truth_norm:
        return True
    
    # One contains the other
    if truth_norm in pred_norm or pred_norm in truth_norm:
        return True
    
    # First word match (for categorical answers)
    if pred_norm.split() and truth_norm.split():
        if pred_norm.split()[0] == truth_norm.split()[0]:
            return True
    
    return False


def answer_matches_liar_transcript(prediction: str, liar_transcript: str) -> bool:
    """Check if prediction was fooled by the liar transcript."""
    pred_norm = normalize_answer(prediction)
    liar_norm = normalize_answer(liar_transcript)
    
    # Check for key contradiction words
    # E.g., if liar says "happy" but audio is "sad", check if model said "happy"
    liar_words = set(liar_norm.split())
    pred_words = set(pred_norm.split())
    
    # Intersection of key words
    key_words = {"happy", "sad", "angry", "calm", "male", "female", 
                 "man", "woman", "dog", "cat", "bird"}
    
    liar_keys = liar_words & key_words
    pred_keys = pred_words & key_words
    
    # If prediction shares key words with liar, it was fooled
    if liar_keys & pred_keys:
        return True
    
    return False


def run_liar_evaluation(
    model_path: str,
    liar_data_path: str,
    output_dir: str = "liar_eval_results",
    device: str = "cuda",
    inject_transcript: bool = True
) -> Dict:
    """Evaluate model on liar transcript test.
    
    Args:
        model_path: Path to DeSTA/ORCA model
        liar_data_path: Path to liar_samples.jsonl from generator
        output_dir: Output directory
        device: Device
        inject_transcript: If True, inject liar transcript into prompt
    
    Returns:
        Evaluation statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    
    # Load liar samples
    logger.info(f"Loading liar samples from: {liar_data_path}")
    samples = []
    with open(liar_data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples")
    
    # Evaluation
    results = []
    task_stats = {}
    
    for sample in tqdm(samples, desc="Evaluating"):
        audio_path = sample.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Audio not found: {audio_path}")
            continue
        
        question = sample["question"]
        audio_truth = sample["audio_ground_truth"]
        liar_transcript = sample["liar_transcript"]
        task_type = sample["task_type"]
        
        # Build prompt with liar transcript if requested
        if inject_transcript:
            prompt = f"{question}\n\n[Transcription]: {liar_transcript}"
        else:
            prompt = question
        
        # Get prediction
        try:
            prediction = model.chat(
                wav=audio_path,
                prompt=prompt,
                max_new_tokens=64,
                do_sample=False
            )
        except Exception as e:
            logger.warning(f"Inference failed for {audio_path}: {e}")
            continue
        
        # Evaluate
        correct = answer_matches_audio_truth(prediction, audio_truth)
        fooled = answer_matches_liar_transcript(prediction, liar_transcript)
        
        # Track stats
        if task_type not in task_stats:
            task_stats[task_type] = {"total": 0, "correct": 0, "fooled": 0}
        task_stats[task_type]["total"] += 1
        if correct:
            task_stats[task_type]["correct"] += 1
        if fooled:
            task_stats[task_type]["fooled"] += 1
        
        results.append({
            "sample_id": sample["sample_id"],
            "task_type": task_type,
            "question": question,
            "audio_truth": audio_truth,
            "liar_transcript": liar_transcript,
            "prediction": prediction,
            "correct": correct,
            "fooled_by_transcript": fooled
        })
    
    # Calculate overall stats
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    total_fooled = sum(1 for r in results if r["fooled_by_transcript"])
    
    summary = {
        "model_path": model_path,
        "inject_transcript": inject_transcript,
        "total_samples": total,
        "correct_count": total_correct,
        "accuracy": total_correct / total if total > 0 else 0,
        "accuracy_pct": f"{(total_correct / total * 100) if total > 0 else 0:.1f}%",
        "fooled_count": total_fooled,
        "fooled_rate": total_fooled / total if total > 0 else 0,
        "fooled_rate_pct": f"{(total_fooled / total * 100) if total > 0 else 0:.1f}%",
        "per_task": {}
    }
    
    for task, stats in task_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        fooled = stats["fooled"] / stats["total"] if stats["total"] > 0 else 0
        summary["per_task"][task] = {
            "total": stats["total"],
            "accuracy": f"{acc * 100:.1f}%",
            "fooled_rate": f"{fooled * 100:.1f}%"
        }
    
    # Save results
    results_file = output_path / "liar_eval_details.jsonl"
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    summary_file = output_path / "liar_eval_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LIAR TRANSCRIPT EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Transcript injection: {inject_transcript}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Overall accuracy (audio-based): {summary['accuracy_pct']}")
    logger.info(f"Fooled rate (followed liar): {summary['fooled_rate_pct']}")
    logger.info("-" * 60)
    logger.info("Per-Task Results:")
    for task, stats in summary["per_task"].items():
        logger.info(f"  {task}: Accuracy={stats['accuracy']}, Fooled={stats['fooled_rate']}")
    logger.info("-" * 60)
    logger.info("Interpretation:")
    logger.info("  Higher accuracy = Better audio utilization (less ASR reliance)")
    logger.info("  Expected: DeSTA ~35%, ORCA ~70%")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P0-2: Evaluate model on liar transcript test"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of model"
    )
    parser.add_argument(
        "--liar-data",
        type=str,
        default="liar_transcript_data/liar_samples.jsonl",
        help="Path to liar samples JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="liar_eval_results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--no-inject",
        action="store_true",
        help="Don't inject liar transcript (baseline)"
    )
    
    args = parser.parse_args()
    
    run_liar_evaluation(
        model_path=args.model,
        liar_data_path=args.liar_data,
        output_dir=args.output_dir,
        device=args.device,
        inject_transcript=not args.no_inject
    )


if __name__ == "__main__":
    main()
