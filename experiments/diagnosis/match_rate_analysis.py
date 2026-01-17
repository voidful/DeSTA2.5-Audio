"""
P0-1: Match Rate Quantification

This script measures how much a model relies on ASR by comparing predictions
with and without audio input.

Hypothesis: If a model relies heavily on ASR, predictions will be SIMILAR 
whether audio is present or not (high match rate). ORCA should have LOWER
match rate (more audio-dependent predictions).

Method:
1. Load SAKURA test samples (1000 samples, 250 per task)
2. For each sample, run model:
   - With full audio + transcript → prediction_A
   - With ONLY text (no audio) → prediction_B
3. Calculate match rate: % where prediction_A == prediction_B
4. Lower match rate = better audio utilization

Expected results:
- DeSTA baseline: ~78% match rate (high ASR reliance)
- ORCA-R1: ~60% match rate (better audio utilization)
"""

import os
import json
import argparse
import logging
import wave
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

import torch
from datasets import load_dataset
from desta import DeSTA25AudioModel
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SAKURA dataset and task configuration
SAKURA_HUB_ID = "MERLab/SAKURA"
SAKURA_TASKS = {
    "single_hop": [
        "AnimalRecognition",
        "EmotionRecognition", 
        "Gender",
        "SoundRecognition"
    ],
    "multi_hop": [
        "MultiAnimalRecognition",
        "MultiEmotionRecognition",
        "MultiGender", 
        "MultiSoundRecognition"
    ]
}

TMP_WAV_PATH = "tmp_match_rate_audio.wav"


def write_wav_from_array(audio_array, sample_rate: int, wav_path: str):
    """Write float audio array to WAV file."""
    audio_array = np.array(audio_array, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def write_wav_from_dataset_item(item, wav_path: str):
    """Extract and write WAV from SAKURA dataset item."""
    audio_data = item["audio"]
    audio_array = audio_data["array"]
    sample_rate = audio_data["sampling_rate"]
    write_wav_from_array(audio_array, sample_rate, wav_path)
    return wav_path


def get_prediction_with_audio(
    model: DeSTA25AudioModel,
    item: dict,
    wav_path: str = TMP_WAV_PATH,
    max_new_tokens: int = 64
) -> str:
    """Get model prediction WITH audio input."""
    write_wav_from_dataset_item(item, wav_path)
    question = item["question"]
    
    response = model.chat(
        wav=wav_path,
        prompt=question,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return response.strip()


def get_prediction_text_only(
    tokenizer,
    llm_model,
    item: dict,
    max_new_tokens: int = 64,
    device: str = "cuda"
) -> str:
    """Get model prediction WITHOUT audio (text-only LLM backbone).
    
    This simulates what the model would predict based purely on the question
    text, without any audio features.
    """
    question = item["question"]
    
    # Apply chat template
    messages = [{"role": "user", "content": question}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        input_text = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return prediction.strip()


def normalize_for_comparison(answer: str) -> str:
    """Normalize answer for fuzzy comparison."""
    import re
    answer = answer.lower().strip()
    # Remove common variations
    answer = re.sub(r'^(the answer is|i think|it is|the|a|an)\s+', '', answer)
    answer = re.sub(r'[.!?,;:\'"]+$', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def predictions_match(pred_audio: str, pred_text: str) -> bool:
    """Check if two predictions are semantically equivalent."""
    norm_audio = normalize_for_comparison(pred_audio)
    norm_text = normalize_for_comparison(pred_text)
    
    # Exact match
    if norm_audio == norm_text:
        return True
    
    # One contains the other
    if norm_audio in norm_text or norm_text in norm_audio:
        return True
    
    # First word match (for categorical answers)
    words_audio = norm_audio.split()
    words_text = norm_text.split()
    if words_audio and words_text and words_audio[0] == words_text[0]:
        return True
    
    return False


def run_match_rate_analysis(
    model_path: str,
    samples_per_task: int = 250,
    output_dir: str = "match_rate_results",
    device: str = "cuda",
    hop_type: str = "single_hop"
) -> Dict:
    """Run match rate analysis on SAKURA dataset.
    
    Args:
        model_path: Path or HuggingFace ID of DeSTA model
        samples_per_task: Number of samples per task (default 250, total 1000)
        output_dir: Output directory
        device: Device
        hop_type: "single_hop" or "multi_hop"
    
    Returns:
        Dictionary with match rate statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load DeSTA model
    logger.info(f"Loading DeSTA model: {model_path}")
    desta_model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    
    # Load text-only LLM backbone for comparison
    # Get the LLM ID from the DeSTA model config
    llm_model_id = desta_model.config.llm_model_name_or_path
    logger.info(f"Loading LLM backbone for text-only comparison: {llm_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    llm_model.eval()
    
    # Task list
    tasks = SAKURA_TASKS[hop_type]
    hop_prefix = "single_" if hop_type == "single_hop" else "multi_"
    
    results = []
    task_stats = {}
    
    for task_name in tasks:
        dataset_name = f"{hop_prefix}{task_name}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}")
        
        try:
            dataset = load_dataset(SAKURA_HUB_ID, dataset_name, split="test")
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
        
        # Sample subset
        num_samples = min(samples_per_task, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        task_results = []
        matches = 0
        
        for idx in tqdm(indices, desc=f"{task_name}"):
            item = dataset[int(idx)]
            
            # Get predictions
            pred_with_audio = get_prediction_with_audio(desta_model, item)
            pred_text_only = get_prediction_text_only(tokenizer, llm_model, item)
            
            is_match = predictions_match(pred_with_audio, pred_text_only)
            if is_match:
                matches += 1
            
            task_results.append({
                "task": task_name,
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "pred_with_audio": pred_with_audio,
                "pred_text_only": pred_text_only,
                "is_match": is_match
            })
        
        match_rate = matches / len(task_results) if task_results else 0
        task_stats[task_name] = {
            "total": len(task_results),
            "matches": matches,
            "match_rate": match_rate,
            "match_rate_pct": f"{match_rate * 100:.1f}%"
        }
        
        results.extend(task_results)
        logger.info(f"{task_name}: Match rate = {match_rate * 100:.1f}% ({matches}/{len(task_results)})")
    
    # Overall statistics
    total_samples = len(results)
    total_matches = sum(1 for r in results if r["is_match"])
    overall_match_rate = total_matches / total_samples if total_samples > 0 else 0
    
    summary = {
        "model_path": model_path,
        "hop_type": hop_type,
        "total_samples": total_samples,
        "total_matches": total_matches,
        "overall_match_rate": overall_match_rate,
        "overall_match_rate_pct": f"{overall_match_rate * 100:.1f}%",
        "per_task_stats": task_stats,
        "interpretation": (
            "Lower match rate = better audio utilization (less ASR reliance). "
            f"Expected: DeSTA ~78%, ORCA ~60%. "
            f"This model: {overall_match_rate * 100:.1f}%"
        )
    }
    
    # Save results
    results_file = output_path / "match_rate_details.jsonl"
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    summary_file = output_path / "match_rate_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MATCH RATE ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Overall match rate: {overall_match_rate * 100:.1f}%")
    logger.info("-" * 60)
    logger.info("Per-Task Match Rates:")
    for task, stats in task_stats.items():
        logger.info(f"  {task}: {stats['match_rate_pct']}")
    logger.info("-" * 60)
    logger.info(summary["interpretation"])
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    # Cleanup temp file
    if os.path.exists(TMP_WAV_PATH):
        os.remove(TMP_WAV_PATH)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P0-1: Match Rate Quantification - Measure ASR reliance"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of DeSTA model"
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=250,
        help="Number of samples per task (default: 250, total ~1000)"
    )
    parser.add_argument(
        "--hop-type",
        type=str,
        default="single_hop",
        choices=["single_hop", "multi_hop"],
        help="SAKURA hop type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="match_rate_results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    
    args = parser.parse_args()
    
    np.random.seed(42)  # Reproducibility
    
    run_match_rate_analysis(
        model_path=args.model,
        samples_per_task=args.samples_per_task,
        output_dir=args.output_dir,
        device=args.device,
        hop_type=args.hop_type
    )


if __name__ == "__main__":
    main()
