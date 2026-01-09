"""
Intervention Experiments for ORCA-DeSTA

Implements causal verification experiments from the paper:
- Audio Swap: Same text, different audio
- Text Paraphrase: Same audio, different wording
- Synthetic Mismatch: Positive text + negative audio

These experiments verify that ORCA genuinely uses audio information.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass
import json


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    original_pred: str
    intervened_pred: str
    original_label: str
    intervention_type: str
    prediction_changed: bool
    follows_intervention: bool
    metadata: Dict[str, Any]


def find_matching_pairs(
    dataset: List[Dict],
    group_by: str = "transcription",
    differ_by: str = "emotion"
) -> List[Tuple[Dict, Dict]]:
    """
    Find pairs of samples that match on one attribute but differ on another.
    
    Args:
        dataset: List of sample dicts
        group_by: Attribute that should be the same
        differ_by: Attribute that should be different
        
    Returns:
        List of (sample1, sample2) pairs
    """
    # Group samples by the grouping attribute
    groups = {}
    for sample in dataset:
        key = sample.get(group_by, "")
        if key not in groups:
            groups[key] = []
        groups[key].append(sample)
    
    # Find pairs within each group that differ on the other attribute
    pairs = []
    for key, samples in groups.items():
        if len(samples) < 2:
            continue
        
        # Get unique values of the differing attribute
        values = {}
        for s in samples:
            val = s.get(differ_by, "")
            if val not in values:
                values[val] = []
            values[val].append(s)
        
        # Create pairs from different values
        value_list = list(values.keys())
        for i, v1 in enumerate(value_list):
            for v2 in value_list[i+1:]:
                for s1 in values[v1][:3]:  # Limit pairs per group
                    for s2 in values[v2][:3]:
                        pairs.append((s1, s2))
    
    return pairs


def audio_swap_experiment(
    model,
    dataset: List[Dict],
    max_pairs: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Audio Swap Experiment: Keep text, swap audio.
    
    If model genuinely uses audio, predictions should change when audio changes.
    
    Args:
        model: DeSTA25AudioModel
        dataset: List of samples with 'audio', 'transcription', and 'label' keys
        max_pairs: Maximum number of pairs to test
        device: Device to run inference
        
    Returns:
        Dict with metrics: prediction_change_rate, follows_swap_rate
    """
    model.eval()
    model = model.to(device)
    
    # Find pairs with same transcription, different emotion/label
    pairs = find_matching_pairs(dataset, group_by="transcription", differ_by="label")
    pairs = pairs[:max_pairs]
    
    if not pairs:
        return {"error": "No valid pairs found"}
    
    results = []
    
    for sample1, sample2 in tqdm(pairs, desc="Audio swap experiment"):
        # Original prediction for sample1
        with torch.no_grad():
            pred1 = _get_prediction(model, sample1, device)
        
        # Swapped prediction: sample1's text with sample2's audio
        swapped_sample = {
            **sample1,
            "audio": sample2["audio"]
        }
        with torch.no_grad():
            pred_swapped = _get_prediction(model, swapped_sample, device)
        
        # Did prediction change?
        changed = pred1 != pred_swapped
        
        # Does it follow the swapped audio's label?
        follows_swap = pred_swapped == sample2.get("label", "")
        
        results.append(InterventionResult(
            original_pred=pred1,
            intervened_pred=pred_swapped,
            original_label=sample1.get("label", ""),
            intervention_type="audio_swap",
            prediction_changed=changed,
            follows_intervention=follows_swap,
            metadata={"sample1": sample1.get("id"), "sample2": sample2.get("id")}
        ))
    
    # Compute metrics
    n_total = len(results)
    n_changed = sum(1 for r in results if r.prediction_changed)
    n_follows = sum(1 for r in results if r.follows_intervention)
    
    return {
        "total_pairs": n_total,
        "prediction_change_rate": n_changed / n_total if n_total > 0 else 0,
        "follows_swap_rate": n_follows / n_total if n_total > 0 else 0,
        "follows_given_change_rate": n_follows / n_changed if n_changed > 0 else 0
    }


def text_paraphrase_experiment(
    model,
    dataset: List[Dict],
    paraphrases: Dict[str, str],
    max_samples: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Text Paraphrase Experiment: Keep audio, change text wording.
    
    If model relies on exact text, predictions will change with paraphrases.
    Robust model should give same predictions.
    
    Args:
        model: DeSTA25AudioModel
        dataset: List of samples
        paraphrases: Dict mapping original text to paraphrased text
        max_samples: Maximum samples to test
        device: Device to run inference
        
    Returns:
        Dict with metrics: prediction_change_rate (lower is better)
    """
    model.eval()
    model = model.to(device)
    
    results = []
    tested = 0
    
    for sample in tqdm(dataset, desc="Text paraphrase experiment"):
        if tested >= max_samples:
            break
            
        original_text = sample.get("transcription", "")
        if original_text not in paraphrases:
            continue
        
        paraphrased_text = paraphrases[original_text]
        
        # Original prediction
        with torch.no_grad():
            pred_original = _get_prediction(model, sample, device)
        
        # Paraphrased prediction
        paraphrased_sample = {
            **sample,
            "transcription": paraphrased_text
        }
        with torch.no_grad():
            pred_paraphrased = _get_prediction(model, paraphrased_sample, device)
        
        changed = pred_original != pred_paraphrased
        
        results.append(InterventionResult(
            original_pred=pred_original,
            intervened_pred=pred_paraphrased,
            original_label=sample.get("label", ""),
            intervention_type="text_paraphrase",
            prediction_changed=changed,
            follows_intervention=not changed,  # For paraphrase, same pred is desired
            metadata={"original_text": original_text, "paraphrased_text": paraphrased_text}
        ))
        tested += 1
    
    n_total = len(results)
    n_changed = sum(1 for r in results if r.prediction_changed)
    
    return {
        "total_samples": n_total,
        "prediction_change_rate": n_changed / n_total if n_total > 0 else 0,
        "stability_rate": 1 - (n_changed / n_total) if n_total > 0 else 0
    }


def synthetic_mismatch_experiment(
    model,
    positive_text_samples: List[Dict],
    negative_audio_samples: List[Dict],
    max_pairs: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Synthetic Mismatch Experiment: Positive text + negative audio.
    
    Creates samples where text sentiment (e.g., "I love this") conflicts with
    audio sentiment (e.g., angry/sad tone). Tests if model can detect mismatch.
    
    Args:
        model: DeSTA25AudioModel
        positive_text_samples: Samples with positive text
        negative_audio_samples: Samples with negative audio tone
        max_pairs: Maximum pairs
        device: Device
        
    Returns:
        Dict with correct_detection_rate (predicts negative = follows audio)
    """
    model.eval()
    model = model.to(device)
    
    n_pairs = min(len(positive_text_samples), len(negative_audio_samples), max_pairs)
    
    results = []
    
    for i in tqdm(range(n_pairs), desc="Synthetic mismatch experiment"):
        pos_sample = positive_text_samples[i]
        neg_sample = negative_audio_samples[i]
        
        # Create mismatched sample
        mismatched = {
            **pos_sample,
            "audio": neg_sample["audio"],
            "expected_if_audio": neg_sample.get("label", "negative")
        }
        
        with torch.no_grad():
            pred = _get_prediction(model, mismatched, device)
        
        # Does it detect the mismatch (predict based on audio)?
        follows_audio = pred.lower() in ["negative", "sad", "angry", "sarcastic"]
        
        results.append(InterventionResult(
            original_pred=pred,
            intervened_pred=pred,
            original_label=neg_sample.get("label", "negative"),
            intervention_type="synthetic_mismatch",
            prediction_changed=True,  # Always a mismatch case
            follows_intervention=follows_audio,
            metadata={"positive_text": pos_sample.get("transcription"),
                     "negative_audio_label": neg_sample.get("label")}
        ))
    
    n_total = len(results)
    n_correct = sum(1 for r in results if r.follows_intervention)
    
    return {
        "total_pairs": n_total,
        "correct_detection_rate": n_correct / n_total if n_total > 0 else 0,
        "text_shortcut_rate": 1 - (n_correct / n_total) if n_total > 0 else 0
    }


def _get_prediction(
    model,
    sample: Dict,
    device: str
) -> str:
    """
    Helper to get model prediction for a sample.
    
    This needs to be customized based on your model's interface.
    """
    # Prepare input
    audio_path = sample.get("audio", sample.get("audio_path"))
    transcription = sample.get("transcription", "")
    instruction = sample.get("instruction", "What is the emotion in this audio?")
    
    messages = [
        {
            "role": "user",
            "content": f"<|AUDIO|>\nTranscription: {transcription}\n{instruction}",
            "audios": [{"audio": audio_path}]
        }
    ]
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                max_new_tokens=64
            )
        
        pred = outputs.text if hasattr(outputs, 'text') else str(outputs)
        if isinstance(pred, list):
            pred = pred[0]
        return pred.strip()
    except Exception as e:
        return f"ERROR: {e}"


def run_all_interventions(
    model,
    dataset: List[Dict],
    paraphrases: Optional[Dict[str, str]] = None,
    max_samples: int = 100,
    device: str = "cuda",
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run all intervention experiments.
    
    Args:
        model: DeSTA25AudioModel
        dataset: Full dataset
        paraphrases: Optional paraphrase dict for text_paraphrase experiment
        max_samples: Max samples per experiment
        device: Device
        output_path: Path to save results
        
    Returns:
        Dict with results from all experiments
    """
    results = {}
    
    # Audio Swap
    print("\n=== Audio Swap Experiment ===")
    results["audio_swap"] = audio_swap_experiment(model, dataset, max_samples, device)
    print(f"Prediction change rate: {results['audio_swap'].get('prediction_change_rate', 0):.2%}")
    print(f"Follows swap rate: {results['audio_swap'].get('follows_swap_rate', 0):.2%}")
    
    # Text Paraphrase (if paraphrases provided)
    if paraphrases:
        print("\n=== Text Paraphrase Experiment ===")
        results["text_paraphrase"] = text_paraphrase_experiment(
            model, dataset, paraphrases, max_samples, device
        )
        print(f"Prediction change rate: {results['text_paraphrase'].get('prediction_change_rate', 0):.2%}")
    
    # Synthetic Mismatch
    print("\n=== Synthetic Mismatch Experiment ===")
    positive_samples = [s for s in dataset if s.get("sentiment") == "positive"]
    negative_samples = [s for s in dataset if s.get("sentiment") == "negative"]
    
    if positive_samples and negative_samples:
        results["synthetic_mismatch"] = synthetic_mismatch_experiment(
            model, positive_samples, negative_samples, max_samples, device
        )
        print(f"Correct detection rate: {results['synthetic_mismatch'].get('correct_detection_rate', 0):.2%}")
    else:
        print("Skipping: No positive/negative samples found")
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    print("Intervention Experiments Module")
    print("This module requires a model and dataset to run.")
    print("Use run_all_interventions() with your model and data.")
