#!/usr/bin/env python
"""
Cross-Dataset Generalization Evaluation for ORCA-DeSTA

Tests model generalization to unseen emotion recognition datasets:
- IEMOCAP
- MELD
- RAVDESS

Usage:
    python experiments/ablation/cross_dataset_eval.py \
        --checkpoint /path/to/checkpoint \
        --datasets iemocap meld ravdess \
        --output_dir ./cross_dataset_results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset


# Dataset configurations
EMOTION_DATASETS = {
    "iemocap": {
        "dataset_id": "Zahra99/IEMOCAP_Text_Audio",
        "audio_column": "audio",
        "label_column": "label",
        "text_column": "text",
        "split": "test"
    },
    "meld": {
        "dataset_id": "declare-lab/MELD",
        "audio_column": "audio",
        "label_column": "Emotion",
        "text_column": "Utterance",
        "split": "test"
    },
    "ravdess": {
        "dataset_id": "ylacombe/ravdess",
        "audio_column": "audio",
        "label_column": "emotion",
        "text_column": None,  # No transcript
        "split": "test"
    }
}


def evaluate_emotion_recognition(
    model,
    dataset_config: Dict,
    max_samples: int = 500,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate emotion recognition on a single dataset.
    
    Args:
        model: DeSTA25AudioModel
        dataset_config: Dataset configuration dict
        max_samples: Maximum samples to evaluate
        device: Device
        
    Returns:
        Dict with accuracy and per-class metrics
    """
    import wave
    import tempfile
    import numpy as np
    
    # Load dataset
    print(f"Loading dataset: {dataset_config['dataset_id']}")
    
    try:
        ds = load_dataset(dataset_config["dataset_id"], split=dataset_config["split"])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"error": str(e)}
    
    # Limit samples
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    
    correct = 0
    total = 0
    predictions = []
    
    for sample in tqdm(ds, desc="Evaluating"):
        try:
            # Get audio
            audio_data = sample[dataset_config["audio_column"]]
            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                continue
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                
                # Convert to int16
                audio_int16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
                
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(sample_rate))
                    wf.writeframes(audio_int16.tobytes())
            
            # Get transcript if available
            transcript = ""
            if dataset_config.get("text_column"):
                transcript = sample.get(dataset_config["text_column"], "")
            
            # Build message
            instruction = "What is the emotion expressed in this audio? Answer with one word: happy, sad, angry, neutral, fear, disgust, or surprise."
            
            messages = [
                {"role": "system", "content": "You are an emotion recognition assistant."},
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{instruction}",
                    "audios": [{"audio": temp_path, "text": transcript if transcript else None}]
                }
            ]
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    messages=messages,
                    do_sample=False,
                    max_new_tokens=16
                )
            
            pred = outputs.text
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.strip().lower()
            
            # Get label
            label = sample[dataset_config["label_column"]]
            if isinstance(label, int):
                # Map numeric to string if needed
                emotion_map = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fear"}
                label = emotion_map.get(label, str(label))
            label = str(label).lower()
            
            # Check correctness
            is_correct = label in pred or pred in label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                "prediction": pred,
                "label": label,
                "correct": is_correct
            })
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions
    }


def run_cross_dataset_evaluation(
    checkpoint: str,
    datasets: List[str],
    output_dir: str,
    max_samples_per_dataset: int = 500
) -> Dict[str, Any]:
    """
    Run cross-dataset generalization evaluation.
    
    Args:
        checkpoint: Path to trained model
        datasets: List of dataset names to evaluate
        output_dir: Output directory
        max_samples_per_dataset: Max samples per dataset
        
    Returns:
        Dict with results for each dataset
    """
    from desta import DeSTA25AudioModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading model from {checkpoint}")
    model = DeSTA25AudioModel.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    
    results = {}
    
    for dataset_name in datasets:
        if dataset_name not in EMOTION_DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        print(f"\n=== Evaluating on {dataset_name} ===")
        
        config = EMOTION_DATASETS[dataset_name]
        result = evaluate_emotion_recognition(
            model, config, max_samples_per_dataset, device
        )
        
        results[dataset_name] = result
        print(f"Accuracy: {result.get('accuracy', 0):.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        "checkpoint": checkpoint,
        "datasets": {
            name: {
                "accuracy": res.get("accuracy", 0),
                "correct": res.get("correct", 0),
                "total": res.get("total", 0)
            }
            for name, res in results.items()
        }
    }
    
    with open(os.path.join(output_dir, "cross_dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed predictions
    for name, res in results.items():
        if "predictions" in res:
            with open(os.path.join(output_dir, f"{name}_predictions.jsonl"), "w") as f:
                for pred in res["predictions"]:
                    f.write(json.dumps(pred) + "\n")
    
    print("\n=== Summary ===")
    for name, res in results.items():
        print(f"{name}: {res.get('accuracy', 0):.4f} ({res.get('correct', 0)}/{res.get('total', 0)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["iemocap", "meld", "ravdess"],
                        help="Datasets to evaluate")
    parser.add_argument("--output_dir", type=str, default="./cross_dataset_results",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per dataset")
    
    args = parser.parse_args()
    
    run_cross_dataset_evaluation(
        checkpoint=args.checkpoint,
        datasets=args.datasets,
        output_dir=args.output_dir,
        max_samples_per_dataset=args.max_samples
    )


if __name__ == "__main__":
    main()
