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
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import (
    compute_pca_explained_variance,
    compute_effective_dimensionality,
    compute_feature_statistics,
    analyze_feature_collapse,
    compute_group_aware_metrics
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
                    # perception returns:
                    # - (tokens, lengths) for struct_orca
                    # - (tokens, lengths) for orca_hybrid (now potentially with losses in tuple)
                    # - (tokens, lengths) for qformer_1
                    
                    if isinstance(outputs, tuple):
                        # outputs could be (tokens, lengths) OR ((tokens, loss), lengths)
                        # We need to inspect the first element
                        first_elem = outputs[0]
                        
                        if isinstance(first_elem, tuple):
                             # Case: ((tokens, loss), lengths) - from modified ORCAHybridConnector
                             audio_tokens = first_elem[0].float().cpu().numpy()
                        else:
                             # Case: (tokens, lengths) - standard
                             audio_tokens = first_elem.float().cpu().numpy()
                    else:
                        # Fallback for other return types
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


def run_observation1_feature_collapse(audio_tokens, output_dir, num_groups=8, queries_per_group=8):
    """
    Observation 1: Feature Collapse Analysis
    
    Measures how much of the representation space is actually utilized.
    High PCA concentration = feature collapse (bad).
    
    Also includes Group-Aware metrics for Struct-ORCA:
    - Group Independence Score (GIS)
    - Intra-Group Diversity (IGD)
    - Token Utilization Variance (TUV)
    - Centroid Orthogonality
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
    
    # Add Group-Aware Metrics (for Struct-ORCA)
    total_tokens = num_groups * queries_per_group
    if len(audio_tokens.shape) >= 2 and (audio_tokens.shape[1] == total_tokens or 
                                         audio_tokens.shape[-1] % total_tokens == 0):
        print("\n--- Group-Aware Metrics (Struct-ORCA) ---")
        group_metrics = compute_group_aware_metrics(audio_tokens, num_groups, queries_per_group)
        results["group_aware"] = group_metrics
        
        print(f"  Group Independence Score (GIS): {group_metrics['group_independence_score']:.4f}")
        print(f"    (Higher is better - groups encode distinct info)")
        print(f"  Intra-Group Diversity Mean: {group_metrics['intra_group_diversity']['mean']:.4f}")
        print(f"    (Moderate is ideal - too low=redundancy, too high=no focus)")
        print(f"  Token Utilization Variance (TUV): {group_metrics['token_utilization_variance']:.4f}")
        print(f"    (Higher is better - no dead tokens)")
        print(f"  Centroid Orthogonality: {group_metrics['centroid_orthogonality']:.4f}")
        print(f"    (Higher is better - groups in distinct subspaces)")
    
    print(f"\n--- Traditional Metrics ---")
    print(f"Effective Dimensionality: {eff_dim:.2f}")
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
    
def run_single_model_analysis(model, model_name, dataset, args, device):
    """Run all observations for a single model."""
    # Extract representations
    data = extract_representations_from_mmau(
        model, dataset, max_samples=args.max_samples, device=device
    )
    
    print(f"\nExtracted {len(data['audio_tokens'])} samples")
    print(f"Audio token shape: {data['audio_tokens'].shape}")
    
    results = {
        "model_id": model_name,
        "num_samples": len(data['audio_tokens']),
    }
    
    # Observation 1
    results["observation1_feature_collapse"] = run_observation1_feature_collapse(
        data['audio_tokens'], args.output_dir
    )
    
    # Observation 2
    results["observation2_content_redundancy"] = run_observation2_content_redundancy(
        data['audio_tokens'], 
        data['question_embeddings'], 
        data['answer_embeddings'],
        args.output_dir, 
        device
    )
    
    # Observation 3
    results["observation3_entanglement"] = run_observation3_entanglement(
        data['audio_tokens'], data['labels'], data['label_mappings'], args.output_dir
    )
    
    return results, data


def run_comparison_analysis(model1, model2, name1, name2, dataset, args, device):
    """Run comparison analysis for two models with side-by-side visualizations."""
    from visualizations import plot_tsne_comparison, plot_pca_variance_curve, plot_metrics_comparison_table
    
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract representations for both models
    print(f"\n{'='*60}")
    print(f"Extracting representations for: {name1}")
    print(f"{'='*60}")
    data1 = extract_representations_from_mmau(model1, dataset, args.max_samples, device)
    
    print(f"\n{'='*60}")
    print(f"Extracting representations for: {name2}")
    print(f"{'='*60}")
    data2 = extract_representations_from_mmau(model2, dataset, args.max_samples, device)
    
    results = {
        "comparison_mode": True,
        "models": [name1, name2],
        "dataset": DATASET_ID,
        "split": args.split,
        "num_samples": len(data1['audio_tokens']),
        "timestamp": datetime.now().isoformat()
    }
    
    # ========== Observation 1: PCA Comparison ==========
    print("\n" + "="*60)
    print("OBSERVATION 1: Feature Collapse Comparison")
    print("="*60)
    
    from feature_analysis import compute_pca_explained_variance, compute_effective_dimensionality, compute_group_aware_metrics
    
    audio1_flat = data1['audio_tokens'].reshape(data1['audio_tokens'].shape[0], -1)
    audio2_flat = data2['audio_tokens'].reshape(data2['audio_tokens'].shape[0], -1)
    
    var1, cumvar1 = compute_pca_explained_variance(audio1_flat)
    var2, cumvar2 = compute_pca_explained_variance(audio2_flat)
    
    eff_dim1 = compute_effective_dimensionality(audio1_flat)
    eff_dim2 = compute_effective_dimensionality(audio2_flat)
    
    # Compute group-aware metrics for both models
    group_metrics1 = compute_group_aware_metrics(data1['audio_tokens'], num_groups=8, queries_per_group=8)
    group_metrics2 = compute_group_aware_metrics(data2['audio_tokens'], num_groups=8, queries_per_group=8)
    
    results["observation1"] = {
        name1: {
            "effective_dim": float(eff_dim1),
            "pc1": float(var1[0]),
            "cumulative_3": float(cumvar1[2]),
            "cumulative_variance": cumvar1[:20].tolist(),
            "group_independence_score": group_metrics1["group_independence_score"],
            "intra_group_diversity_mean": group_metrics1["intra_group_diversity"]["mean"],
            "token_utilization_variance": group_metrics1["token_utilization_variance"],
            "centroid_orthogonality": group_metrics1["centroid_orthogonality"],
        },
        name2: {
            "effective_dim": float(eff_dim2),
            "pc1": float(var2[0]),
            "cumulative_3": float(cumvar2[2]),
            "cumulative_variance": cumvar2[:20].tolist(),
            "group_independence_score": group_metrics2["group_independence_score"],
            "intra_group_diversity_mean": group_metrics2["intra_group_diversity"]["mean"],
            "token_utilization_variance": group_metrics2["token_utilization_variance"],
            "centroid_orthogonality": group_metrics2["centroid_orthogonality"],
        }
    }
    
    print(f"\n{name1}:")
    print(f"  Effective Dim: {eff_dim1:.2f}, PC1: {var1[0]:.2%}, PC1-3: {cumvar1[2]:.2%}")
    print(f"  GIS: {group_metrics1['group_independence_score']:.4f}, IGD: {group_metrics1['intra_group_diversity']['mean']:.4f}")
    print(f"  TUV: {group_metrics1['token_utilization_variance']:.4f}, CentroidOrth: {group_metrics1['centroid_orthogonality']:.4f}")
    print(f"\n{name2}:")
    print(f"  Effective Dim: {eff_dim2:.2f}, PC1: {var2[0]:.2%}, PC1-3: {cumvar2[2]:.2%}")
    print(f"  GIS: {group_metrics2['group_independence_score']:.4f}, IGD: {group_metrics2['intra_group_diversity']['mean']:.4f}")
    print(f"  TUV: {group_metrics2['token_utilization_variance']:.4f}, CentroidOrth: {group_metrics2['centroid_orthogonality']:.4f}")
    
    # Plot comparison PCA
    plot_pca_variance_curve(
        {name1: cumvar1[:20], name2: cumvar2[:20]},
        save_path=os.path.join(figures_dir, "comparison_pca_variance.png")
    )
    
    # ========== Observation 2: Token Orthogonality Comparison ==========
    print("\n" + "="*60)
    print("OBSERVATION 2: Token Orthogonality Comparison")
    print("="*60)
    
    def compute_token_stats(audio_tokens):
        n_samples, n_tokens = audio_tokens.shape[:2]
        avg_sims = []
        for i in range(min(n_samples, 500)):
            tokens = audio_tokens[i]
            norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
            tokens_norm = tokens / norms
            sim = tokens_norm @ tokens_norm.T
            upper = sim[np.triu_indices(n_tokens, k=1)]
            avg_sims.append(np.mean(np.abs(upper)))
        return float(np.mean(avg_sims))
    
    sim1 = compute_token_stats(data1['audio_tokens'])
    sim2 = compute_token_stats(data2['audio_tokens'])
    
    results["observation2"] = {
        name1: {"avg_token_cosine_sim": sim1},
        name2: {"avg_token_cosine_sim": sim2}
    }
    
    print(f"\n{name1}: Avg Token Cosine Sim = {sim1:.4f}")
    print(f"{name2}: Avg Token Cosine Sim = {sim2:.4f}")
    
    if sim2 < sim1:
        print(f"\n✓ {name2} shows {(1 - sim2/sim1)*100:.1f}% reduction in token correlation")
    
    # ========== Observation 3: t-SNE Comparison ==========
    print("\n" + "="*60)
    print("OBSERVATION 3: Entanglement Comparison (t-SNE)")
    print("="*60)
    
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import TSNE
    
    # Use same subset for fair comparison
    n = min(len(data1['audio_tokens']), len(data2['audio_tokens']), 1000)
    labels = data1['labels']['task'][:n]
    label_names = {v: k for k, v in data1['label_mappings']['task'].items()}
    
    # t-SNE for both
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    
    emb1 = tsne.fit_transform(audio1_flat[:n])
    sil1 = silhouette_score(emb1, labels)
    
    emb2 = tsne.fit_transform(audio2_flat[:n])
    sil2 = silhouette_score(emb2, labels)
    
    results["observation3"] = {
        name1: {"task_silhouette": float(sil1)},
        name2: {"task_silhouette": float(sil2)}
    }
    
    print(f"\n{name1}: Task Silhouette = {sil1:.4f}")
    print(f"{name2}: Task Silhouette = {sil2:.4f}")
    
    if sil2 > sil1:
        print(f"\n✓ {name2} shows {(sil2 - sil1):.3f} improvement in task separation")
    
    # Create side-by-side t-SNE figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for ax, emb, name, sil in [(axes[0], emb1, name1, sil1), (axes[1], emb2, name2, sil2)]:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            display_name = label_names.get(label, str(label))
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], label=display_name,
                      alpha=0.7, s=25, edgecolors='white', linewidth=0.3)
        
        ax.set_title(f"{name}\nSilhouette: {sil:.3f}", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(title="Task Type", loc='upper right', fontsize=9)
    
    plt.suptitle("Audio Representation t-SNE by Task Type: Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_tsne_task.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== Summary Metrics Table ==========
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Count wins for each model
    wins = {name1: 0, name2: 0}
    
    # Define metrics with evaluations (lower_is_better flag)
    metrics = [
        ("Token Cosine Sim (↓)", sim1, sim2, True),  # Lower is better
        ("Group Indep. Score (↑)", group_metrics1['group_independence_score'], group_metrics2['group_independence_score'], False),
        ("Token Util. Var. (↑)", group_metrics1['token_utilization_variance'], group_metrics2['token_utilization_variance'], False),
        ("Centroid Orthogonality (↑)", group_metrics1['centroid_orthogonality'], group_metrics2['centroid_orthogonality'], False),
        ("Intra-Group Div. (IGD)", group_metrics1['intra_group_diversity']['mean'], group_metrics2['intra_group_diversity']['mean'], None),  # No clear better
    ]
    
    print(f"\n    | {'Metric':<28} | {name1:<12} | {name2:<12} | Better       |")
    print(f"    |{'-'*30}|{'-'*14}|{'-'*14}|{'-'*14}|")
    
    for metric_name, val1, val2, lower_is_better in metrics:
        if lower_is_better is None:
            better = "-"
        elif lower_is_better:
            better = name2 if val2 < val1 else name1
            if val2 < val1:
                wins[name2] += 1
            else:
                wins[name1] += 1
        else:
            better = name2 if val2 > val1 else name1
            if val2 > val1:
                wins[name2] += 1
            else:
                wins[name1] += 1
        print(f"    | {metric_name:<28} | {val1:<12.4f} | {val2:<12.4f} | {better:<12} |")
    
    print(f"    |{'-'*30}|{'-'*14}|{'-'*14}|{'-'*14}|")
    print(f"    | {'TOTAL WINS':<28} | {wins[name1]:<12} | {wins[name2]:<12} | {'✓ ' + (name2 if wins[name2] > wins[name1] else name1):<12} |")
    print()
    
    # ========== Visualization: Group-Aware Metrics Bar Chart ==========
    print("\nGenerating comparison charts...")
    
    # Bar chart for group-aware metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Group-Aware Metrics (higher is better)
    metric_names = ['GIS', 'TUV', 'Centroid\nOrth']
    values1 = [group_metrics1['group_independence_score'], 
               group_metrics1['token_utilization_variance'],
               group_metrics1['centroid_orthogonality']]
    values2 = [group_metrics2['group_independence_score'],
               group_metrics2['token_utilization_variance'],
               group_metrics2['centroid_orthogonality']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, values1, width, label=name1, color='#4ECDC4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, values2, width, label=name2, color='#FF6B6B', alpha=0.8)
    
    ax1.set_ylabel('Score (higher = better)')
    ax1.set_title('Group-Aware Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Right: Token Orthogonality (lower is better)
    ax2 = axes[1]
    metric_names2 = ['Token\nCosine Sim', 'Intra-Group\nDiversity']
    values1_2 = [sim1, group_metrics1['intra_group_diversity']['mean']]
    values2_2 = [sim2, group_metrics2['intra_group_diversity']['mean']]
    
    x2 = np.arange(len(metric_names2))
    bars1_2 = ax2.bar(x2 - width/2, values1_2, width, label=name1, color='#4ECDC4', alpha=0.8)
    bars2_2 = ax2.bar(x2 + width/2, values2_2, width, label=name2, color='#FF6B6B', alpha=0.8)
    
    ax2.set_ylabel('Score')
    ax2.set_title('Token Similarity Metrics')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_names2)
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1_2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2_2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_group_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(figures_dir, 'comparison_group_metrics.png')}")
    
    # ========== Radar Chart for Overall Comparison ==========
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    categories = ['GIS', 'TUV', 'Centroid Orth', '1 - Token Sim', 'IGD']
    # Normalize to 0-1 scale, invert Token Sim so higher is better
    values1_radar = [
        group_metrics1['group_independence_score'],
        group_metrics1['token_utilization_variance'],
        group_metrics1['centroid_orthogonality'],
        1 - sim1,  # Invert so higher is better
        group_metrics1['intra_group_diversity']['mean'],
    ]
    values2_radar = [
        group_metrics2['group_independence_score'],
        group_metrics2['token_utilization_variance'],
        group_metrics2['centroid_orthogonality'],
        1 - sim2,
        group_metrics2['intra_group_diversity']['mean'],
    ]
    
    # Close the radar chart
    values1_radar += values1_radar[:1]
    values2_radar += values2_radar[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values1_radar, 'o-', linewidth=2, label=name1, color='#4ECDC4')
    ax.fill(angles, values1_radar, alpha=0.25, color='#4ECDC4')
    ax.plot(angles, values2_radar, 'o-', linewidth=2, label=name2, color='#FF6B6B')
    ax.fill(angles, values2_radar, alpha=0.25, color='#FF6B6B')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('ORCA Metrics Comparison\n(All metrics: higher = better)', size=14, pad=20)
    
    plt.savefig(os.path.join(figures_dir, "comparison_radar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(figures_dir, 'comparison_radar.png')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Observations on MMAU Dataset")
    parser.add_argument("--model_id", type=str, nargs='+', default=["voidful/QAQ_4b"],
                        help="Model ID(s). Provide two for comparison mode.")
    parser.add_argument("--model_names", type=str, nargs='+', default=None,
                        help="Display names for models (optional, defaults to model IDs)")
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset once
    print(f"\n{'='*60}")
    print(f"Loading MMAU dataset: {DATASET_ID} [{args.split}]")
    print(f"{'='*60}")
    dataset = load_dataset(DATASET_ID, split=args.split)
    print(f"Dataset size: {len(dataset)}")
    
    from desta import DeSTA25AudioModel
    
    # Comparison mode: two models
    if len(args.model_id) == 2:
        print("\n" + "="*60)
        print("COMPARISON MODE: Analyzing two models")
        print("="*60)
        
        model1_id, model2_id = args.model_id
        if args.model_names and len(args.model_names) == 2:
            name1, name2 = args.model_names
        else:
            name1 = model1_id.split('/')[-1]
            name2 = model2_id.split('/')[-1]
        
        print(f"\nModel 1: {model1_id} ({name1})")
        print(f"Model 2: {model2_id} ({name2})")
        
        # Load models
        model1 = DeSTA25AudioModel.from_pretrained(model1_id)
        model1.to(device).eval()
        
        model2 = DeSTA25AudioModel.from_pretrained(model2_id)
        model2.to(device).eval()
        
        results = run_comparison_analysis(model1, model2, name1, name2, dataset, args, device)
        
        # Clean up
        del model1, model2
        torch.cuda.empty_cache()
        
    else:
        # Single model mode
        model_id = args.model_id[0]
        model_name = args.model_names[0] if args.model_names else model_id.split('/')[-1]
        
        print(f"\n{'='*60}")
        print(f"Loading model: {model_id}")
        print(f"{'='*60}")
        
        model = DeSTA25AudioModel.from_pretrained(model_id)
        model.to(device).eval()
        
        print(f"\nModel Configuration:")
        print(f"  Connector mode: {model.config.connector_mode}")
        if model.config.connector_mode == "struct_orca":
            print(f"  Num groups: {model.config.struct_orca_num_groups}")
            print(f"  Queries per group: {model.config.struct_orca_queries_per_group}")
        
        results, _ = run_single_model_analysis(model, model_name, dataset, args, device)
        results["dataset"] = DATASET_ID
        results["split"] = args.split
        results["timestamp"] = datetime.now().isoformat()
        
        del model
        torch.cuda.empty_cache()
        
        # Summary for single model
        print("\n" + "="*60)
        print("OBSERVATION ANALYSIS COMPLETE")
        print("="*60)
        
        obs1 = results['observation1_feature_collapse']
        obs2 = results['observation2_content_redundancy']
        obs3 = results['observation3_entanglement']
        
        print(f"\nSUMMARY for {model_name}:")
        print(f"  Effective Dim: {obs1['effective_dimensionality']:.2f}")
        print(f"  PC1-3 Cumulative: {obs1['pca_variance']['cumulative_3']:.2%}")
        print(f"  Token Avg Cosine Sim: {obs2['token_orthogonality']['avg_pairwise_cosine_sim']:.4f}")
        print(f"  Task Silhouette: {obs3.get('task_silhouette_score', 'N/A')}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "observation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {os.path.join(args.output_dir, 'figures')}")
    
    return results


if __name__ == "__main__":
    main()


