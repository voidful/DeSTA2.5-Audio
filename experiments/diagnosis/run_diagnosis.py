#!/usr/bin/env python
"""
Main Diagnostic Runner for ORCA-DeSTA

Run all diagnostic analyses on a trained checkpoint.

Usage:
    python experiments/diagnosis/run_diagnosis.py \
        --checkpoint voidful/QAQ_0.6b_orca_all \
        --data_path /path/to/sakura_data.jsonl \
        --output_dir ./diagnosis_results \
        --num_samples 5000
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Local imports
from feature_analysis import (
    analyze_feature_collapse,
    compute_effective_dimensionality,
    compute_pca_explained_variance
)
from mutual_information import estimate_mutual_information, compute_linear_cka
from text_probe import train_text_probe, train_attribute_probes
from group_probing import (
    compute_group_centroids,
    compute_group_similarity_matrix,
    train_group_probes,
    print_specialization_summary
)
from visualizations import (
    plot_tsne_by_attribute,
    plot_pca_variance_curve,
    plot_group_similarity_heatmap,
    plot_metrics_comparison_table
)


def load_model(checkpoint: str, device: str = "cuda"):
    """Load DeSTA25AudioModel from checkpoint."""
    from desta import DeSTA25AudioModel
    
    print(f"Loading model from {checkpoint}...")
    model = DeSTA25AudioModel.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model


def load_dataset(data_path: str, num_samples: int = 5000):
    """Load dataset for analysis."""
    from datasets import load_dataset
    
    if data_path.endswith('.jsonl'):
        # Load from JSONL file
        with open(data_path) as f:
            data = [json.loads(line) for line in f][:num_samples]
        return data
    else:
        # Assume HuggingFace dataset
        ds = load_dataset(data_path, split="test")
        return ds.select(range(min(num_samples, len(ds))))


def extract_representations(
    model,
    dataset,
    device: str = "cuda",
    batch_size: int = 16,
    max_samples: int = 5000
) -> dict:
    """
    Extract audio token representations from model.
    
    Returns dict with:
        - audio_tokens: [N, num_tokens, hidden_dim]
        - text_embeddings: [N, text_dim] (from LLM embedding layer)
        - labels: dict of attribute labels
    """
    model.eval()
    
    all_audio_tokens = []
    all_text_embeddings = []
    
    labels = {
        "emotion": [],
        "gender": [],
        "sentence_id": []
    }
    
    print("Extracting representations...")
    
    for i, sample in enumerate(tqdm(dataset, desc="Extracting")):
        if i >= max_samples:
            break
        
        try:
            # Prepare input
            audio_path = sample.get("audio", {})
            if isinstance(audio_path, dict):
                audio_array = audio_path.get("array")
                sample_rate = audio_path.get("sampling_rate", 16000)
            else:
                # Load from path
                import librosa
                audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Process audio through model's processor
            input_features = model.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Get audio representations
            with torch.no_grad():
                outputs = model.perception(input_features=input_features)
                audio_tokens = outputs.audio_global.cpu().numpy()  # [1, num_tokens, H]
            
            all_audio_tokens.append(audio_tokens[0])
            
            # Get text embedding (for MI calculation)
            text = sample.get("transcription", sample.get("text", ""))
            text_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                text_emb = model.llm.model.embed_tokens(text_ids).mean(dim=1).cpu().numpy()
            all_text_embeddings.append(text_emb[0])
            
            # Collect labels
            labels["emotion"].append(sample.get("emotion", sample.get("label", "unknown")))
            labels["gender"].append(sample.get("gender", "unknown"))
            labels["sentence_id"].append(sample.get("sentence_id", i))
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Stack arrays
    audio_tokens = np.stack(all_audio_tokens)
    text_embeddings = np.stack(all_text_embeddings)
    
    # Convert labels to numeric
    for key in labels:
        unique_vals = list(set(labels[key]))
        labels[key] = np.array([unique_vals.index(v) for v in labels[key]])
    
    return {
        "audio_tokens": audio_tokens,
        "text_embeddings": text_embeddings,
        "labels": labels
    }


def run_full_diagnosis(
    model,
    data: dict,
    output_dir: str,
    model_name: str = "orca"
) -> dict:
    """
    Run all diagnostic analyses.
    
    Args:
        model: DeSTA25AudioModel
        data: Dict from extract_representations
        output_dir: Directory to save results
        model_name: Name for labeling
        
    Returns:
        Dict with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(data["audio_tokens"])
    }
    
    audio_flat = data["audio_tokens"].reshape(data["audio_tokens"].shape[0], -1)
    text_flat = data["text_embeddings"]
    
    # === 1. Feature Collapse Analysis ===
    print("\n=== Feature Collapse Analysis ===")
    collapse_results = analyze_feature_collapse(audio_flat, model_name)
    results["feature_collapse"] = collapse_results
    
    print(f"Effective dimensionality: {collapse_results['effective_dimensionality']:.2f}")
    print(f"PCA-3 variance: {collapse_results['pca_variance']['cumulative_3']:.2%}")
    print(f"PCA-5 variance: {collapse_results['pca_variance']['cumulative_5']:.2%}")
    
    # PCA variance curve
    plot_pca_variance_curve(
        {model_name: np.array(collapse_results["raw_cumulative_variance"])},
        save_path=os.path.join(figures_dir, "pca_variance.png")
    )
    
    # === 2. Mutual Information Analysis ===
    print("\n=== Mutual Information Analysis ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mi_estimate = estimate_mutual_information(
        audio_flat, text_flat,
        num_epochs=100,
        device=device
    )
    results["mutual_information"] = {
        "mi_estimate_nats": mi_estimate,
        "mi_estimate_bits": mi_estimate / np.log(2)
    }
    print(f"MI(Audio, Text): {mi_estimate:.4f} nats")
    
    # CKA
    cka = compute_linear_cka(audio_flat, text_flat)
    results["cka"] = cka
    print(f"CKA(Audio, Text): {cka:.4f}")
    
    # === 3. Text Probe ===
    print("\n=== Text Probe Analysis ===")
    
    # Use bag-of-words encoding for text classification
    from text_probe import BagOfWordsEncoder
    
    # Create simple text labels (cluster text embeddings)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    text_clusters = kmeans.fit_predict(text_flat)
    
    probe_result = train_text_probe(
        audio_flat, text_clusters,
        device=device,
        epochs=50
    )
    results["text_probe"] = probe_result
    print(f"Text probe accuracy: {probe_result['test_accuracy']:.4f}")
    
    # === 4. Attribute Probes ===
    print("\n=== Attribute Probes ===")
    attr_probe_results = train_attribute_probes(
        audio_flat,
        {k: v for k, v in data["labels"].items() if len(np.unique(v)) > 1},
        device=device,
        epochs=50
    )
    results["attribute_probes"] = attr_probe_results
    
    # === 5. Group Probing (if ORCA-R1) ===
    num_groups = getattr(model.config, 'orca_r1_num_groups', 8)
    queries_per_group = getattr(model.config, 'orca_r1_queries_per_group', 8)
    
    if data["audio_tokens"].shape[1] == num_groups * queries_per_group:
        print("\n=== Group Specialization Analysis ===")
        
        centroids = compute_group_centroids(
            data["audio_tokens"], num_groups, queries_per_group
        )
        
        # Similarity matrix
        similarity = compute_group_similarity_matrix(centroids)
        results["group_similarity"] = similarity.tolist()
        
        avg_off_diag = (similarity.sum() - np.trace(similarity)) / (num_groups * (num_groups - 1))
        print(f"Average inter-group similarity: {avg_off_diag:.4f}")
        
        # Group probes
        group_probe_results = train_group_probes(
            centroids, 
            {k: v for k, v in data["labels"].items() if len(np.unique(v)) > 1}
        )
        results["group_probes"] = group_probe_results
        
        # Plot heatmap
        plot_group_similarity_heatmap(
            similarity,
            save_path=os.path.join(figures_dir, "group_similarity.png")
        )
    
    # === 6. t-SNE Visualization ===
    print("\n=== t-SNE Visualizations ===")
    
    for attr_name, attr_labels in data["labels"].items():
        if len(np.unique(attr_labels)) < 10:  # Only for categorical
            plot_tsne_by_attribute(
                audio_flat[:1000],  # Limit for speed
                attr_labels[:1000],
                attr_name,
                save_path=os.path.join(figures_dir, f"tsne_{attr_name}.png")
            )
    
    # === Save Results ===
    results_path = os.path.join(output_dir, "diagnosis_results.json")
    
    # Convert numpy to list for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Figures saved to {figures_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ORCA-DeSTA Diagnostic Analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path or HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data (JSONL or HuggingFace dataset)")
    parser.add_argument("--output_dir", type=str, default="./diagnosis_results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for labeling (defaults to checkpoint name)")
    
    args = parser.parse_args()
    
    # Set model name
    if args.model_name is None:
        args.model_name = Path(args.checkpoint).name
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load dataset
    dataset = load_dataset(args.data_path, args.num_samples)
    
    # Extract representations
    data = extract_representations(model, dataset, args.device, max_samples=args.num_samples)
    
    # Run diagnosis
    results = run_full_diagnosis(model, data, args.output_dir, args.model_name)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Samples analyzed: {results['num_samples']}")
    print(f"\nFeature Collapse:")
    print(f"  Effective dimensionality: {results['feature_collapse']['effective_dimensionality']:.2f}")
    print(f"  PCA-3 variance: {results['feature_collapse']['pca_variance']['cumulative_3']:.2%}")
    print(f"\nRedundancy with Text:")
    print(f"  MI(Audio, Text): {results['mutual_information']['mi_estimate_nats']:.4f} nats")
    print(f"  Text probe accuracy: {results['text_probe']['test_accuracy']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
