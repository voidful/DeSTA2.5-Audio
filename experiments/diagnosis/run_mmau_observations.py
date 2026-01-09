#!/usr/bin/env python
"""
Observation Analysis on MMAU Dataset

Runs the three observations from the ORCA-DeSTA paper introduction on MMAU:
- Observation 1: Feature Collapse (PCA explained variance, effective dimensionality)
- Observation 2: Content Redundancy (Mutual Information with text)
- Observation 3: Entanglement (t-SNE visualization by task type)

Usage:
    python experiments/diagnosis/run_mmau_observations.py \
        --model_id voidful/DeSTA2.5-Qwen3-0.6B \
        --max_samples 500 \
        --output_dir ./mmau_observation_results
"""

import os
import sys
import json
import argparse
import wave
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import (
    compute_pca_explained_variance,
    compute_effective_dimensionality,
    compute_feature_statistics,
    analyze_feature_collapse
)
from mutual_information import estimate_mutual_information, compute_linear_cka
from visualizations import plot_tsne_by_attribute, plot_pca_variance_curve, plot_group_similarity_heatmap

# Constants
DATASET_ID = "lmms-lab/mmau"
DEFAULT_SPLIT = "test_mini"


def write_wav_from_array(audio_array, sample_rate, wav_path):
    """Write audio array to WAV file."""
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())

    return wav_path


def extract_representations_from_mmau(
    model,
    dataset,
    max_samples: int = 500,
    device: str = "cuda"
):
    """
    Extract audio token representations from MMAU dataset samples.
    
    Returns:
        Dict with audio_tokens, text_embeddings, and labels (task, difficulty)
    """
    from transformers import AutoProcessor
    
    model.eval()
    
    # Initialize processor (model.processor is lazily loaded, so we load it directly)
    processor = AutoProcessor.from_pretrained(model.config.encoder_model_id)
    
    # Initialize tokenizer if not already done
    if not hasattr(model, 'tokenizer'):
        model._setup_generation()
    
    all_audio_tokens = []
    all_text_embeddings = []
    labels = {
        "task": [],
        "difficulty": [],
        "sub_category": []
    }
    
    print(f"Extracting representations from {min(len(dataset), max_samples)} samples...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "temp_audio.wav")
        
        for i, item in enumerate(tqdm(dataset, desc="Extracting")):
            if i >= max_samples:
                break
            
            try:
                # Get audio
                audio_obj = item["audio"]
                audio_array = audio_obj["array"]
                sample_rate = audio_obj.get("sampling_rate", 16000) or 16000
                
                # Write to temp file
                write_wav_from_array(audio_array, sample_rate, wav_path)
                
                # Process through processor
                feature = processor(
                    audio_array, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device)
                
                # Get audio representations through perception module
                # Returns (global_tokens, speech_feature_lengths) tuple
                with torch.no_grad():
                    outputs = model.perception(input_features=feature)
                    # perception returns tuple: (audio_tokens, lengths)
                    if isinstance(outputs, tuple):
                        audio_tokens = outputs[0].cpu().numpy()  # [1, num_tokens, H]
                    else:
                        audio_tokens = outputs.audio_global.cpu().numpy()
                
                all_audio_tokens.append(audio_tokens[0])
                
                # Get text embedding for question
                question = item.get("question", "")
                text_ids = model.tokenizer(question, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    text_emb = model.llm_model.model.embed_tokens(text_ids).mean(dim=1).cpu().numpy()
                all_text_embeddings.append(text_emb[0])
                
                # Collect labels
                labels["task"].append(item.get("task", "unknown"))
                labels["difficulty"].append(item.get("difficulty", "unknown"))
                labels["sub_category"].append(item.get("sub-category", "unknown"))
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    
    # Stack arrays
    audio_tokens = np.stack(all_audio_tokens)
    text_embeddings = np.stack(all_text_embeddings)
    
    # Convert labels to numeric
    label_mappings = {}
    for key in labels:
        unique_vals = sorted(set(labels[key]))
        label_mappings[key] = {v: i for i, v in enumerate(unique_vals)}
        labels[key] = np.array([label_mappings[key][v] for v in labels[key]])
    
    return {
        "audio_tokens": audio_tokens,
        "text_embeddings": text_embeddings,
        "labels": labels,
        "label_mappings": label_mappings
    }


def run_observation1_feature_collapse(audio_tokens, output_dir):
    """
    Observation 1: Feature Collapse Analysis
    
    Measures how much of the representation space is actually utilized.
    High PCA concentration = feature collapse (bad).
    """
    print("\n" + "="*60)
    print("OBSERVATION 1: Feature Collapse Analysis")
    print("="*60)
    
    # Flatten audio tokens
    audio_flat = audio_tokens.reshape(audio_tokens.shape[0], -1)
    
    # PCA analysis
    explained_var, cumulative_var = compute_pca_explained_variance(audio_flat)
    
    # Effective dimensionality
    eff_dim = compute_effective_dimensionality(audio_flat)
    
    # Feature statistics
    stats = compute_feature_statistics(audio_flat)
    
    results = {
        "effective_dimensionality": float(eff_dim),
        "pca_variance": {
            "pc1": float(explained_var[0]),
            "pc2": float(explained_var[1]) if len(explained_var) > 1 else 0,
            "pc3": float(explained_var[2]) if len(explained_var) > 2 else 0,
            "cumulative_3": float(cumulative_var[2]) if len(cumulative_var) > 2 else 0,
            "cumulative_5": float(cumulative_var[4]) if len(cumulative_var) > 4 else 0,
            "cumulative_10": float(cumulative_var[9]) if len(cumulative_var) > 9 else 0,
        },
        "feature_stats": stats,
        "raw_explained_variance": explained_var[:20].tolist(),
        "raw_cumulative_variance": cumulative_var[:20].tolist(),
    }
    
    print(f"\nEffective Dimensionality: {eff_dim:.2f}")
    print(f"  (Higher is better - more dimensions utilized)")
    print(f"\nPCA Explained Variance:")
    print(f"  PC1: {results['pca_variance']['pc1']:.2%}")
    print(f"  PC1-3 cumulative: {results['pca_variance']['cumulative_3']:.2%}")
    print(f"  PC1-5 cumulative: {results['pca_variance']['cumulative_5']:.2%}")
    print(f"  PC1-10 cumulative: {results['pca_variance']['cumulative_10']:.2%}")
    print(f"\nInterpretation:")
    if results['pca_variance']['cumulative_3'] > 0.8:
        print("  ⚠️  HIGH COLLAPSE: Top 3 PCs explain >80% variance")
    elif results['pca_variance']['cumulative_3'] > 0.5:
        print("  ⚡ MODERATE COLLAPSE: Top 3 PCs explain 50-80% variance")
    else:
        print("  ✓ LOW COLLAPSE: Representations well distributed")
    
    # Plot PCA curve
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_pca_variance_curve(
        {"MMAU": np.array(results['raw_cumulative_variance'])},
        save_path=os.path.join(figures_dir, "obs1_pca_variance.png")
    )
    
    return results


def run_observation2_content_redundancy(audio_tokens, text_embeddings, output_dir, device="cuda"):
    """
    Observation 2: Content Redundancy Analysis
    
    Measures how much text/linguistic information is encoded in audio representations.
    High MI = high redundancy (bad - audio tokens leak text content).
    """
    print("\n" + "="*60)
    print("OBSERVATION 2: Content Redundancy (Text Information Leakage)")
    print("="*60)
    
    # Flatten
    audio_flat = audio_tokens.reshape(audio_tokens.shape[0], -1)
    
    # Limit dimensions for computational efficiency
    max_dim = 256
    if audio_flat.shape[1] > max_dim:
        # Use PCA to reduce dimensions
        from sklearn.decomposition import PCA
        pca_audio = PCA(n_components=max_dim)
        audio_reduced = pca_audio.fit_transform(audio_flat)
    else:
        audio_reduced = audio_flat
    
    if text_embeddings.shape[1] > max_dim:
        from sklearn.decomposition import PCA
        pca_text = PCA(n_components=max_dim)
        text_reduced = pca_text.fit_transform(text_embeddings)
    else:
        text_reduced = text_embeddings
    
    # Estimate MI
    print("\nEstimating Mutual Information (this may take a few minutes)...")
    mi_estimate = estimate_mutual_information(
        audio_reduced, 
        text_reduced,
        num_epochs=50,
        batch_size=128,
        device=device,
        verbose=True
    )
    
    # Compute CKA
    cka = compute_linear_cka(audio_reduced, text_reduced)
    
    results = {
        "mutual_information_nats": float(mi_estimate),
        "mutual_information_bits": float(mi_estimate / np.log(2)),
        "cka_similarity": float(cka)
    }
    
    print(f"\nMutual Information Estimate:")
    print(f"  MI(Audio, Text): {results['mutual_information_nats']:.4f} nats")
    print(f"                   {results['mutual_information_bits']:.4f} bits")
    print(f"\nCentered Kernel Alignment (CKA):")
    print(f"  CKA(Audio, Text): {results['cka_similarity']:.4f}")
    print(f"\nInterpretation:")
    if results['cka_similarity'] > 0.5:
        print("  ⚠️  HIGH REDUNDANCY: Audio strongly resembles text embeddings")
    elif results['cka_similarity'] > 0.3:
        print("  ⚡ MODERATE REDUNDANCY: Some text information in audio")
    else:
        print("  ✓ LOW REDUNDANCY: Audio representations are distinct from text")
    
    return results


def run_observation3_entanglement(audio_tokens, labels, label_mappings, output_dir, max_samples=1000):
    """
    Observation 3: Content-Style Entanglement Analysis
    
    Uses t-SNE to visualize whether different task types cluster separately.
    Well-disentangled = clear clusters by task type.
    """
    print("\n" + "="*60)
    print("OBSERVATION 3: Content-Style Entanglement (t-SNE)")
    print("="*60)
    
    # Flatten and limit samples for t-SNE
    audio_flat = audio_tokens.reshape(audio_tokens.shape[0], -1)
    n_samples = min(len(audio_flat), max_samples)
    audio_subset = audio_flat[:n_samples]
    
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    results = {}
    
    # t-SNE by task type
    print("\nGenerating t-SNE visualization by task type...")
    task_labels = labels["task"][:n_samples]
    
    fig = plot_tsne_by_attribute(
        audio_subset,
        task_labels,
        "Task Type",
        save_path=os.path.join(figures_dir, "obs3_tsne_by_task.png"),
        perplexity=min(30, n_samples // 5),
        title="Audio Representation t-SNE by Task Type"
    )
    
    # Reverse mapping for readability
    task_names = {v: k for k, v in label_mappings["task"].items()}
    results["task_mapping"] = task_names
    
    # t-SNE by difficulty
    print("Generating t-SNE visualization by difficulty...")
    diff_labels = labels["difficulty"][:n_samples]
    
    fig = plot_tsne_by_attribute(
        audio_subset,
        diff_labels,
        "Difficulty",
        save_path=os.path.join(figures_dir, "obs3_tsne_by_difficulty.png"),
        perplexity=min(30, n_samples // 5),
        title="Audio Representation t-SNE by Difficulty"
    )
    
    diff_names = {v: k for k, v in label_mappings["difficulty"].items()}
    results["difficulty_mapping"] = diff_names
    
    print(f"\nVisualization saved to: {figures_dir}")
    print("\nInterpretation:")
    print("  - Clear clusters by task = good disentanglement of task-specific features")
    print("  - Mixed/overlapping clusters = entangled representations")
    print("  - Check the saved figures to evaluate visually")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Observations on MMAU Dataset")
    parser.add_argument("--model_id", type=str, default="voidful/QAQ_4b",
                        help="Model ID or checkpoint path")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        help="Dataset split (test_mini, test)")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples to analyze")
    parser.add_argument("--output_dir", type=str, default="./mmau_observation_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_id}")
    print(f"{'='*60}")
    
    from desta import DeSTA25AudioModel
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()
    
    # Print model config info
    print(f"\nModel Configuration:")
    print(f"  Connector mode: {model.config.connector_mode}")
    if model.config.connector_mode == "struct_orca":
        print(f"  Num groups: {model.config.struct_orca_num_groups}")
        print(f"  Queries per group: {model.config.struct_orca_queries_per_group}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading MMAU dataset: {DATASET_ID} [{args.split}]")
    print(f"{'='*60}")
    
    dataset = load_dataset(DATASET_ID, split=args.split)
    print(f"Dataset size: {len(dataset)}")
    
    # Extract representations
    data = extract_representations_from_mmau(
        model, dataset, max_samples=args.max_samples, device=device
    )
    
    print(f"\nExtracted {len(data['audio_tokens'])} samples")
    print(f"Audio token shape: {data['audio_tokens'].shape}")
    print(f"Text embedding shape: {data['text_embeddings'].shape}")
    
    # Run all three observations
    results = {
        "model_id": args.model_id,
        "dataset": DATASET_ID,
        "split": args.split,
        "num_samples": len(data['audio_tokens']),
        "timestamp": datetime.now().isoformat()
    }
    
    # Observation 1: Feature Collapse
    results["observation1_feature_collapse"] = run_observation1_feature_collapse(
        data['audio_tokens'], args.output_dir
    )
    
    # Observation 2: Content Redundancy
    results["observation2_content_redundancy"] = run_observation2_content_redundancy(
        data['audio_tokens'], data['text_embeddings'], args.output_dir, device
    )
    
    # Observation 3: Entanglement
    results["observation3_entanglement"] = run_observation3_entanglement(
        data['audio_tokens'], data['labels'], data['label_mappings'], args.output_dir
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "observation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("OBSERVATION ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {os.path.join(args.output_dir, 'figures')}")
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"Observation 1 - Effective Dimensionality: {results['observation1_feature_collapse']['effective_dimensionality']:.2f}")
    print(f"Observation 1 - PCA-3 Cumulative: {results['observation1_feature_collapse']['pca_variance']['cumulative_3']:.2%}")
    print(f"Observation 2 - MI(Audio,Text): {results['observation2_content_redundancy']['mutual_information_nats']:.4f} nats")
    print(f"Observation 2 - CKA: {results['observation2_content_redundancy']['cka_similarity']:.4f}")
    print(f"Observation 3 - t-SNE plots saved for visual inspection")
    
    return results


if __name__ == "__main__":
    main()
