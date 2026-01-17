"""
P1-2: Refusal Rate Analysis

Analyzes how often models refuse to answer audio-dependent questions.

Hypothesis: DeSTA (ASR-reliant) refuses more because it can't extract info
from audio. ORCA should refuse less (better audio understanding).

Method:
1. Select audio samples where answer requires audio analysis
2. Count "cannot determine", "unclear", "unable to" responses
3. Compare refusal rates between DeSTA and ORCA

Expected:
- DeSTA: ~40% refusal rate
- ORCA: ~18% refusal rate
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import wave

from datasets import load_dataset
from desta import DeSTA25AudioModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patterns that indicate refusal/uncertainty
REFUSAL_PATTERNS = [
    r"cannot determine",
    r"unable to (identify|determine|recognize|classify)",
    r"cannot (identify|tell|recognize|classify)",
    r"not (possible|able) to",
    r"unclear",
    r"(can't|cannot) be (determined|identified)",
    r"i('m| am) not sure",
    r"i don't know",
    r"difficult to (tell|determine|identify)",
    r"impossible to (tell|determine|identify)",
    r"no (clear|definitive) (answer|indication)",
    r"insufficient (information|audio|data)",
    r"hard to (tell|say|determine)",
    r"(audio|sound) (is )?(too )?(unclear|noisy|distorted)",
]


def is_refusal(response: str) -> bool:
    """Check if response indicates refusal/uncertainty."""
    response_lower = response.lower()
    
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    
    return False


def write_wav_from_item(item, wav_path: str):
    """Write SAKURA audio to WAV file."""
    audio_data = item["audio"]
    audio_array = np.array(audio_data["array"], dtype=np.float32)
    sample_rate = audio_data["sampling_rate"]
    
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def run_refusal_analysis(
    model_path: str,
    num_samples: int = 200,
    output_dir: str = "refusal_analysis_results",
    device: str = "cuda"
) -> Dict:
    """Run refusal rate analysis on Animal Recognition task.
    
    Args:
        model_path: Path to model
        num_samples: Number of samples
        output_dir: Output directory
        device: Device
    
    Returns:
        Analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    
    # Load Animal Recognition dataset
    logger.info("Loading SAKURA Animal Recognition dataset...")
    try:
        dataset = load_dataset("MERLab/SAKURA", "single_AnimalRecognition", split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {}
    
    # Sample
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    tmp_wav = "tmp_refusal_audio.wav"
    results = []
    refusal_count = 0
    
    for idx in tqdm(indices, desc="Analyzing refusals"):
        item = dataset[int(idx)]
        
        write_wav_from_item(item, tmp_wav)
        question = item["question"]
        ground_truth = item.get("answer", "")
        
        try:
            response = model.chat(
                wav=tmp_wav,
                prompt=question,
                max_new_tokens=100,
                do_sample=False
            )
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            continue
        
        is_refused = is_refusal(response)
        if is_refused:
            refusal_count += 1
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "is_refusal": is_refused
        })
    
    # Cleanup
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    
    # Calculate stats
    total = len(results)
    refusal_rate = refusal_count / total if total > 0 else 0
    
    # Find example refusals
    example_refusals = [r for r in results if r["is_refusal"]][:5]
    example_answers = [r for r in results if not r["is_refusal"]][:5]
    
    summary = {
        "model_path": model_path,
        "total_samples": total,
        "refusal_count": refusal_count,
        "refusal_rate": refusal_rate,
        "refusal_rate_pct": f"{refusal_rate * 100:.1f}%",
        "example_refusals": example_refusals,
        "example_answers": example_answers,
        "interpretation": (
            f"Refusal rate: {refusal_rate * 100:.1f}%. "
            f"Expected: DeSTA ~40%, ORCA ~18%. "
            f"Lower refusal = better audio understanding."
        )
    }
    
    # Save results
    results_file = output_path / "refusal_details.jsonl"
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    summary_file = output_path / "refusal_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("REFUSAL RATE ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Refusals: {refusal_count}")
    logger.info(f"Refusal rate: {summary['refusal_rate_pct']}")
    logger.info("-" * 60)
    if example_refusals:
        logger.info("Example refusals:")
        for ex in example_refusals[:2]:
            logger.info(f"  Q: {ex['question'][:50]}...")
            logger.info(f"  R: {ex['response'][:80]}...")
    logger.info("-" * 60)
    logger.info(summary["interpretation"])
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P1-2: Refusal Rate Analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="refusal_analysis_results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    
    args = parser.parse_args()
    
    np.random.seed(42)
    
    run_refusal_analysis(
        model_path=args.model,
        num_samples=args.samples,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
