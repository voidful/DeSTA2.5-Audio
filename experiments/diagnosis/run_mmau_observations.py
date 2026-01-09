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
        Dict with audio_tokens, text_embeddings, transcription_embeddings, and labels
    """
    from transformers import AutoProcessor
    
    model.eval()
    
    # Initialize processor (model.processor is lazily loaded, so we load it directly)
    processor = AutoProcessor.from_pretrained(model.config.encoder_model_id)
    
    # Initialize tokenizer if not already done
    if not hasattr(model, 'tokenizer'):
        model._setup_generation()
    
    all_audio_tokens = []
    all_question_embeddings = []
    all_answer_embeddings = []  # Use answer as proxy for "expected content"
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
                        # Convert to float32 before numpy (bfloat16 not supported by numpy)
                        audio_tokens = outputs[0].float().cpu().numpy()  # [1, num_tokens, H]
                    else:
                        audio_tokens = outputs.audio_global.float().cpu().numpy()
                
                all_audio_tokens.append(audio_tokens[0])
                
                # Get text embedding for question (for context comparison)
                question = item.get("question", "")
                question_ids = model.tokenizer(question, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    question_emb = model.llm_model.model.embed_tokens(question_ids).float().mean(dim=1).cpu().numpy()
                all_question_embeddings.append(question_emb[0])
                
                # Get answer embedding (as proxy for audio content)
                answer = item.get("answer", "")
                answer_ids = model.tokenizer(answer, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    answer_emb = model.llm_model.model.embed_tokens(answer_ids).float().mean(dim=1).cpu().numpy()
                all_answer_embeddings.append(answer_emb[0])
                
                # Collect labels
                labels["task"].append(item.get("task", "unknown"))
                labels["difficulty"].append(item.get("difficulty", "unknown"))
                labels["sub_category"].append(item.get("sub-category", "unknown"))
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    
    # Stack arrays
    audio_tokens = np.stack(all_audio_tokens)
    question_embeddings = np.stack(all_question_embeddings)
    answer_embeddings = np.stack(all_answer_embeddings)
    
    # Convert labels to numeric
    label_mappings = {}
    for key in labels:
        unique_vals = sorted(set(labels[key]))
        label_mappings[key] = {v: i for i, v in enumerate(unique_vals)}
        labels[key] = np.array([label_mappings[key][v] for v in labels[key]])
    
    return {
        "audio_tokens": audio_tokens,
        "text_embeddings": question_embeddings,  # Keep for backward compat
        "question_embeddings": question_embeddings,
        "answer_embeddings": answer_embeddings,
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


def run_observation2_content_redundancy(audio_tokens, question_embeddings, answer_embeddings, output_dir, device="cuda"):
    """
    Observation 2: Content Redundancy & Token Orthogonality Analysis
    
    Measures:
    1. How much text/linguistic information is encoded in audio representations (MI, CKA)
    2. Token-level orthogonality (how decorrelated are the audio tokens)
    
    High MI with answer = audio captures content (good for this task)
    Low token orthogonality = tokens are redundant (bad - feature collapse)
    """
    print("\n" + "="*60)
    print("OBSERVATION 2: Content Redundancy & Token Orthogonality")
    print("="*60)
    
    from sklearn.decomposition import PCA
    
    # Flatten audio tokens for MI/CKA
    audio_flat = audio_tokens.reshape(audio_tokens.shape[0], -1)
    
    # Limit dimensions for computational efficiency
    max_dim = 256
    if audio_flat.shape[1] > max_dim:
        pca_audio = PCA(n_components=max_dim)
        audio_reduced = pca_audio.fit_transform(audio_flat)
    else:
        audio_reduced = audio_flat
    
    # Reduce question embeddings
    if question_embeddings.shape[1] > max_dim:
        pca_q = PCA(n_components=max_dim)
        question_reduced = pca_q.fit_transform(question_embeddings)
    else:
        question_reduced = question_embeddings
    
    # Reduce answer embeddings
    if answer_embeddings.shape[1] > max_dim:
        pca_a = PCA(n_components=max_dim)
        answer_reduced = pca_a.fit_transform(answer_embeddings)
    else:
        answer_reduced = answer_embeddings
    
    results = {}
    
    # ====== Part A: MI and CKA with Question (context) ======
    print("\n[A] Audio vs Question (Context) Analysis:")
    mi_question = estimate_mutual_information(
        audio_reduced, question_reduced,
        num_epochs=50, batch_size=128, device=device, verbose=False
    )
    cka_question = compute_linear_cka(audio_reduced, question_reduced)
    
    results["question"] = {
        "mi_nats": float(mi_question),
        "mi_bits": float(mi_question / np.log(2)),
        "cka": float(cka_question)
    }
    print(f"  MI(Audio, Question): {mi_question:.4f} nats")
    print(f"  CKA(Audio, Question): {cka_question:.4f}")
    
    # ====== Part B: MI and CKA with Answer (expected content) ======
    print("\n[B] Audio vs Answer (Expected Content) Analysis:")
    mi_answer = estimate_mutual_information(
        audio_reduced, answer_reduced,
        num_epochs=50, batch_size=128, device=device, verbose=False
    )
    cka_answer = compute_linear_cka(audio_reduced, answer_reduced)
    
    results["answer"] = {
        "mi_nats": float(mi_answer),
        "mi_bits": float(mi_answer / np.log(2)),
        "cka": float(cka_answer)
    }
    print(f"  MI(Audio, Answer): {mi_answer:.4f} nats")
    print(f"  CKA(Audio, Answer): {cka_answer:.4f}")
    
    # ====== Part C: Token-Level Orthogonality Analysis ======
    print("\n[C] Token-Level Orthogonality Analysis:")
    
    # For each sample, compute average pairwise cosine similarity between tokens
    n_samples = audio_tokens.shape[0]
    n_tokens = audio_tokens.shape[1]
    
    avg_cosine_sims = []
    for i in range(min(n_samples, 500)):  # Limit for speed
        tokens = audio_tokens[i]  # [n_tokens, hidden_dim]
        # Normalize tokens
        norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
        tokens_normalized = tokens / norms
        # Compute pairwise cosine similarity
        sim_matrix = tokens_normalized @ tokens_normalized.T
        # Get upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices(n_tokens, k=1)]
        avg_cosine_sims.append(np.mean(np.abs(upper_tri)))
    
    avg_token_sim = float(np.mean(avg_cosine_sims))
    std_token_sim = float(np.std(avg_cosine_sims))
    
    # Also compute correlation matrix eigenvalue analysis
    # Stack all tokens and compute correlation
    all_tokens_flat = audio_tokens.reshape(-1, audio_tokens.shape[-1])
    # Sample for efficiency
    sample_idx = np.random.choice(len(all_tokens_flat), min(5000, len(all_tokens_flat)), replace=False)
    tokens_sample = all_tokens_flat[sample_idx]
    
    # Compute covariance eigenvalues
    tokens_centered = tokens_sample - tokens_sample.mean(axis=0)
    cov = tokens_centered.T @ tokens_centered / len(tokens_sample)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues / eigenvalues.sum()  # Normalize
    
    # Effective rank (using entropy-based measure)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    effective_rank = float(np.exp(-np.sum(eigenvalues_pos * np.log(eigenvalues_pos))))
    
    results["token_orthogonality"] = {
        "avg_pairwise_cosine_sim": avg_token_sim,
        "std_pairwise_cosine_sim": std_token_sim,
        "effective_rank": effective_rank,
        "top1_eigenvalue_ratio": float(eigenvalues[0]),
        "top3_eigenvalue_ratio": float(eigenvalues[:3].sum()),
        "top10_eigenvalue_ratio": float(eigenvalues[:10].sum())
    }
    
    print(f"  Avg pairwise token cosine sim: {avg_token_sim:.4f} (±{std_token_sim:.4f})")
    print(f"  Token effective rank: {effective_rank:.2f}")
    print(f"  Top-1 eigenvalue ratio: {eigenvalues[0]:.2%}")
    print(f"  Top-3 eigenvalue ratio: {eigenvalues[:3].sum():.2%}")
    
    # ====== Interpretation ======
    print("\n[Interpretation]")
    
    if avg_token_sim > 0.7:
        print("  ⚠️  HIGH TOKEN REDUNDANCY: Tokens are highly correlated (collapse)")
    elif avg_token_sim > 0.4:
        print("  ⚡ MODERATE TOKEN DIVERSITY: Some redundancy in tokens")
    else:
        print("  ✓ GOOD TOKEN DIVERSITY: Tokens capture different aspects")
    
    if cka_answer > 0.3:
        print("  ✓ Audio captures answer-relevant information")
    else:
        print("  ⚠️ Audio may not capture task-relevant content")
    
    return results


def run_observation3_entanglement(audio_tokens, labels, label_mappings, output_dir, max_samples=1000):
    """
    Observation 3: Content-Style Entanglement Analysis
    
    Uses t-SNE to visualize whether different task types cluster separately.
    Well-disentangled = clear clusters by task type (high silhouette score).
    Entangled = mixed clusters (low silhouette score).
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
    
    # Create label name mappings (numeric -> string)
    task_names = {v: k for k, v in label_mappings["task"].items()}
    diff_names = {v: k for k, v in label_mappings["difficulty"].items()}
    
    # t-SNE by task type
    print("\nGenerating t-SNE visualization by task type...")
    task_labels = labels["task"][:n_samples]
    
    fig, task_sil_score = plot_tsne_by_attribute(
        audio_subset,
        task_labels,
        "Task Type",
        save_path=os.path.join(figures_dir, "obs3_tsne_by_task.png"),
        perplexity=min(30, n_samples // 5),
        title="Baseline: Audio Representation t-SNE by Task Type",
        label_names=task_names,
        add_metrics=True
    )
    
    results["task_mapping"] = task_names
    results["task_silhouette_score"] = float(task_sil_score) if task_sil_score else None
    print(f"  Task Silhouette Score: {task_sil_score:.3f}" if task_sil_score else "  Could not compute silhouette")
    
    # t-SNE by difficulty
    print("\nGenerating t-SNE visualization by difficulty...")
    diff_labels = labels["difficulty"][:n_samples]
    
    fig, diff_sil_score = plot_tsne_by_attribute(
        audio_subset,
        diff_labels,
        "Difficulty",
        save_path=os.path.join(figures_dir, "obs3_tsne_by_difficulty.png"),
        perplexity=min(30, n_samples // 5),
        title="Baseline: Audio Representation t-SNE by Difficulty",
        label_names=diff_names,
        add_metrics=True
    )
    
    results["difficulty_mapping"] = diff_names
    results["difficulty_silhouette_score"] = float(diff_sil_score) if diff_sil_score else None
    print(f"  Difficulty Silhouette Score: {diff_sil_score:.3f}" if diff_sil_score else "  Could not compute silhouette")
    
    print(f"\nVisualization saved to: {figures_dir}")
    print("\n[Interpretation]")
    
    # Interpret silhouette scores
    if task_sil_score is not None:
        if task_sil_score < 0.1:
            print("  ⚠️ HIGHLY ENTANGLED: Audio tokens do not separate by task type")
        elif task_sil_score < 0.25:
            print("  ⚡ PARTIALLY ENTANGLED: Weak task-type separation")
        else:
            print("  ✓ GOOD SEPARATION: Audio tokens cluster by task type")
    
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
    
    # Observation 2: Content Redundancy & Token Orthogonality
    results["observation2_content_redundancy"] = run_observation2_content_redundancy(
        data['audio_tokens'], 
        data['question_embeddings'], 
        data['answer_embeddings'],
        args.output_dir, 
        device
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
    obs1 = results['observation1_feature_collapse']
    obs2 = results['observation2_content_redundancy']
    
    print(f"Observation 1 - Effective Dimensionality: {obs1['effective_dimensionality']:.2f}")
    print(f"Observation 1 - PCA-3 Cumulative: {obs1['pca_variance']['cumulative_3']:.2%}")
    print(f"Observation 2 - Token Avg Cosine Sim: {obs2['token_orthogonality']['avg_pairwise_cosine_sim']:.4f}")
    print(f"Observation 2 - CKA(Audio,Answer): {obs2['answer']['cka']:.4f}")
    print(f"Observation 3 - t-SNE plots saved for visual inspection")
    
    return results


if __name__ == "__main__":
    main()

