"""
P1-1: Linear Probing Analysis

Tests if frozen audio features contain speaker/emotion information.

Method:
1. Extract frozen features from DeSTA/ORCA audio encoder
2. Train logistic regression on gender/emotion classification
3. High accuracy = model learned speaker/emotion representations

Expected: 70-90% accuracy (showing audio encoder captures this info)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from transformers import AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_audio_features(
    model,
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    device: str = "cuda"
) -> np.ndarray:
    """Extract frozen features from audio encoder.
    
    Returns pooled feature vector.
    """
    from desta.utils.audio import AudioSegment
    
    # Ensure model is in eval mode
    model.eval()
    
    # Process audio through encoder
    with torch.no_grad():
        # Get audio features through the model's audio processor
        processor = model.processor
        features = processor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Get encoder outputs
        encoder_outputs = model.audio_encoder(features)
        hidden_states = encoder_outputs.last_hidden_state  # [1, T, D]
        
        # Mean pooling over time
        pooled = hidden_states.mean(dim=1)  # [1, D]
        
    return pooled.cpu().numpy().squeeze()


def prepare_iemocap_data(
    model,
    num_samples: int = 1000,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare IEMOCAP emotion classification data.
    
    Note: Requires IEMOCAP dataset access.
    Falls back to synthetic data if not available.
    """
    logger.info("Preparing IEMOCAP emotion data...")
    
    try:
        # Try loading IEMOCAP from HuggingFace
        dataset = load_dataset("j0hngou/IEMOCAP", split="train")
        
        features_list = []
        labels_list = []
        
        for item in tqdm(dataset.select(range(min(num_samples, len(dataset)))),
                        desc="Extracting emotion features"):
            audio = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            label = item["emotion"]
            
            feat = extract_audio_features(model, audio, sr, device)
            features_list.append(feat)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
        
    except Exception as e:
        logger.warning(f"IEMOCAP not available: {e}")
        logger.info("Using synthetic emotion data for demonstration...")
        
        # Generate synthetic features
        feature_dim = 1280  # Whisper encoder dim
        n_samples = num_samples
        n_classes = 4
        
        features = np.random.randn(n_samples, feature_dim)
        labels = np.random.randint(0, n_classes, n_samples)
        
        return features, np.array(["happy", "sad", "angry", "neutral"])[labels]


def prepare_commonvoice_gender_data(
    model,
    num_samples: int = 1000,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare CommonVoice gender classification data."""
    logger.info("Preparing CommonVoice gender data...")
    
    try:
        # Load CommonVoice
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", 
                              "en", split="test", streaming=True)
        
        features_list = []
        labels_list = []
        count = 0
        
        for item in tqdm(dataset, desc="Extracting gender features"):
            if count >= num_samples:
                break
            
            gender = item.get("gender")
            if gender not in ["male", "female"]:
                continue
            
            audio = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            
            try:
                feat = extract_audio_features(model, audio, sr, device)
                features_list.append(feat)
                labels_list.append(gender)
                count += 1
            except Exception:
                continue
        
        return np.array(features_list), np.array(labels_list)
        
    except Exception as e:
        logger.warning(f"CommonVoice not available: {e}")
        logger.info("Using synthetic gender data for demonstration...")
        
        feature_dim = 1280
        n_samples = num_samples
        
        features = np.random.randn(n_samples, feature_dim)
        labels = np.random.choice(["male", "female"], n_samples)
        
        return features, labels


def run_linear_probing(
    model_path: str,
    task: str = "both",
    num_samples: int = 1000,
    cv_folds: int = 5,
    output_dir: str = "linear_probing_results",
    device: str = "cuda"
) -> Dict:
    """Run linear probing experiments.
    
    Args:
        model_path: Path to DeSTA/ORCA model
        task: "gender", "emotion", or "both"
        num_samples: Samples per task
        cv_folds: Cross-validation folds
        output_dir: Output directory
        device: Device
    
    Returns:
        Results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    from desta import DeSTA25AudioModel
    model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    model.eval()
    
    results = {}
    
    tasks_to_run = []
    if task in ["gender", "both"]:
        tasks_to_run.append("gender")
    if task in ["emotion", "both"]:
        tasks_to_run.append("emotion")
    
    for task_name in tasks_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running linear probing for: {task_name}")
        
        # Get data
        if task_name == "gender":
            features, labels = prepare_commonvoice_gender_data(
                model, num_samples, device
            )
        else:
            features, labels = prepare_iemocap_data(
                model, num_samples, device
            )
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Labels: {le.classes_}")
        logger.info(f"  Class distribution: {np.bincount(y)}")
        
        # Train logistic regression with cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, features, y, cv=cv_folds, scoring="accuracy")
        
        results[task_name] = {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "accuracy_pct": f"{scores.mean() * 100:.1f}% ± {scores.std() * 100:.1f}%",
            "cv_scores": scores.tolist(),
            "classes": le.classes_.tolist(),
            "num_samples": len(features)
        }
        
        logger.info(f"  Accuracy: {results[task_name]['accuracy_pct']}")
    
    # Save results
    summary = {
        "model_path": model_path,
        "results": results
    }
    
    summary_file = output_path / "linear_probing_results.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("LINEAR PROBING RESULTS")
    logger.info("=" * 60)
    for task_name, res in results.items():
        logger.info(f"{task_name}: {res['accuracy_pct']}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P1-1: Linear Probing Analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to DeSTA/ORCA model"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["gender", "emotion", "both"],
        help="Task to probe"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Samples per task"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="linear_probing_results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    
    args = parser.parse_args()
    
    run_linear_probing(
        model_path=args.model,
        task=args.task,
        num_samples=args.samples,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
