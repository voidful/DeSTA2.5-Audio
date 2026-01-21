#!/usr/bin/env python3
"""
Mechanistic Analysis: Demonstrating the Procrustean Alignment Problem

This script performs 4 analyses to motivate the Delocalization framework:
1. Representation Collapse Analysis → motivates Latent Delocalization
2. Acoustic Information Loss → motivates Variational (μ, σ)
3. Audio Token Underutilization → motivates ASR Dropout + DPO
4. Semantic Redundancy → motivates Orthogonal Grouping

Usage:
    python procrustean_analysis.py --baseline_model <baseline_id> --orca_model <orca_id>
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


class ProcrusteanAnalyzer:
    """Analyze the Procrustean Alignment problem in Audio-LLMs."""
    
    def __init__(self, baseline_model_id: str, orca_model_id: str, output_dir: str):
        self.baseline_model_id = baseline_model_id
        self.orca_model_id = orca_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
    
    def load_model(self, model_id: str):
        """Load a DeSTA model."""
        from desta.models.modeling_desta25 import DeSTA25AudioModel
        print(f"Loading model: {model_id}")
        model = DeSTA25AudioModel.from_pretrained(model_id)
        model.to(self.device)
        model.eval()
        return model
    
    def extract_audio_embeddings(self, model, audio_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract audio embeddings and intermediate representations.
        
        Returns:
            Dict with keys: 'audio_tokens', 'mu', 'sigma', 'group_centroids'
        """
        embeddings = {
            'audio_tokens': [],
            'mu': [],
            'sigma': [],
            'group_centroids': []
        }
        
        for audio_path in tqdm(audio_paths, desc="Extracting embeddings"):
            try:
                with torch.no_grad():
                    # Load and process audio
                    from desta.utils.audio import AudioSegment
                    audio = AudioSegment.from_file(audio_path, target_sr=16000)
                    features = model.processor(
                        audio.samples.tolist(), 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_features.to(self.device)
                    
                    # Forward through perception
                    result = model.perception.forward_whisper(features)
                    
                    if isinstance(result, tuple):
                        tokens, _ = result
                    else:
                        tokens = result
                    
                    embeddings['audio_tokens'].append(tokens.cpu())
                    
                    # Extract μ and σ if variational
                    connector = model.perception.connector
                    if hasattr(connector, 'mu_proj') and hasattr(connector, '_cached_mu'):
                        if connector._cached_mu is not None:
                            mu = connector._cached_mu.cpu()
                            embeddings['mu'].append(mu)
                            
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        # Stack tensors
        for key in embeddings:
            if embeddings[key]:
                embeddings[key] = torch.cat(embeddings[key], dim=0)
            else:
                embeddings[key] = None
        
        return embeddings
    
    # =========================================================================
    # Analysis 1: Representation Collapse
    # =========================================================================
    def analyze_representation_collapse(self, 
                                         baseline_embs: Dict, 
                                         orca_embs: Dict) -> Dict:
        """
        Analyze representation collapse via:
        1. Token Utilization Variance (TUV)
        2. t-SNE visualization
        
        Motivates: Latent Delocalization (Variational)
        """
        print("\n" + "="*60)
        print("Analysis 1: Representation Collapse")
        print("="*60)
        
        results = {}
        
        # Token Utilization Variance
        baseline_tokens = baseline_embs['audio_tokens']
        orca_tokens = orca_embs['audio_tokens']
        
        if baseline_tokens is not None:
            # Variance across tokens
            baseline_var = baseline_tokens.var(dim=1).mean().item()
            results['baseline_token_variance'] = baseline_var
            print(f"Baseline Token Variance: {baseline_var:.4f}")
        
        if orca_tokens is not None:
            orca_var = orca_tokens.var(dim=1).mean().item()
            results['orca_token_variance'] = orca_var
            print(f"ORCA Token Variance: {orca_var:.4f}")
        
        # t-SNE Visualization
        self._plot_tsne_comparison(baseline_tokens, orca_tokens, "representation_collapse")
        
        # Interpretation
        if baseline_tokens is not None and orca_tokens is not None:
            improvement = (orca_var / baseline_var - 1) * 100
            results['variance_improvement'] = improvement
            print(f"\n→ ORCA increases token variance by {improvement:.1f}%")
            print("→ This motivates: Latent Delocalization (spread info across μ+σ)")
        
        return results
    
    def _plot_tsne_comparison(self, baseline_tokens, orca_tokens, name):
        """Create t-SNE visualization comparing baseline vs ORCA."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, tokens, title in zip(
            axes, 
            [baseline_tokens, orca_tokens], 
            ['Baseline (Procrustean)', 'ORCA (Delocalized)']
        ):
            if tokens is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(title)
                continue
            
            # Flatten and sample
            flat = tokens.reshape(-1, tokens.shape[-1]).numpy()
            n_samples = min(2000, flat.shape[0])
            indices = np.random.choice(flat.shape[0], n_samples, replace=False)
            sampled = flat[indices]
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = tsne.fit_transform(sampled)
            
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=5)
            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}_tsne.png", dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir / f'{name}_tsne.png'}")
    
    # =========================================================================
    # Analysis 2: Acoustic Information Loss
    # =========================================================================
    def analyze_acoustic_info_loss(self, 
                                    baseline_embs: Dict, 
                                    orca_embs: Dict,
                                    labels: Dict[str, List]) -> Dict:
        """
        Analyze acoustic information loss via linear probing.
        
        Tasks:
        - Speaker ID prediction
        - Emotion prediction
        
        Motivates: Variational (μ for semantic, σ for acoustic)
        """
        print("\n" + "="*60)
        print("Analysis 2: Acoustic Information Loss (Linear Probe)")
        print("="*60)
        
        results = {}
        
        for task_name, task_labels in labels.items():
            print(f"\nTask: {task_name}")
            
            le = LabelEncoder()
            y = le.fit_transform(task_labels)
            
            for model_name, embs in [('Baseline', baseline_embs), ('ORCA', orca_embs)]:
                tokens = embs['audio_tokens']
                if tokens is None:
                    continue
                
                # Pool tokens to get sample-level representation
                X = tokens.mean(dim=1).numpy()  # [N, D]
                
                # Train/test split (80/20)
                n_train = int(0.8 * len(X))
                X_train, X_test = X[:n_train], X[n_train:]
                y_train, y_test = y[:n_train], y[n_train:]
                
                # Linear probe
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                key = f"{model_name.lower()}_{task_name}_acc"
                results[key] = acc
                print(f"  {model_name}: {acc*100:.2f}%")
        
        # Interpretation
        print("\n→ Higher accuracy = more acoustic info preserved")
        print("→ This motivates: Variational separation (σ stores acoustic details)")
        
        return results
    
    # =========================================================================
    # Analysis 3: Audio Token Underutilization
    # =========================================================================
    def analyze_audio_underutilization(self, 
                                        model, 
                                        audio_text_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Analyze whether model ignores audio tokens.
        
        Method:
        1. Compare output with audio vs without audio
        2. Measure "Match Rate" between predictions
        
        Motivates: ASR Dropout + Modality DPO
        """
        print("\n" + "="*60)
        print("Analysis 3: Audio Token Underutilization")
        print("="*60)
        
        results = {'match_count': 0, 'total': 0}
        
        print("Comparing predictions: with audio vs without audio (blind)")
        
        for audio_path, prompt in tqdm(audio_text_pairs[:100], desc="Analyzing"):
            try:
                # Generate with audio
                output_with_audio = model.generate(
                    audio_paths=[audio_path],
                    prompts=[prompt],
                    max_new_tokens=50
                )
                
                # Generate without audio (blind mode)
                # This requires model support for blind generation
                output_blind = model.generate(
                    audio_paths=[],  # No audio
                    prompts=[prompt],
                    max_new_tokens=50
                )
                
                # Check if outputs match
                if output_with_audio[0].strip() == output_blind[0].strip():
                    results['match_count'] += 1
                results['total'] += 1
                
            except Exception as e:
                continue
        
        if results['total'] > 0:
            match_rate = results['match_count'] / results['total']
            results['match_rate'] = match_rate
            print(f"\nMatch Rate (with audio == without audio): {match_rate*100:.1f}%")
            print("→ High match rate = model ignores audio!")
            print("→ This motivates: ASR Dropout + Modality DPO")
        
        return results
    
    # =========================================================================
    # Analysis 4: Semantic Redundancy (Inter-Group Similarity)
    # =========================================================================
    def analyze_semantic_redundancy(self, 
                                     baseline_embs: Dict, 
                                     orca_embs: Dict,
                                     num_groups: int = 8,
                                     queries_per_group: int = 8) -> Dict:
        """
        Analyze semantic redundancy across groups.
        
        Method: Compute inter-group cosine similarity
        
        Motivates: Orthogonal Grouping
        """
        print("\n" + "="*60)
        print("Analysis 4: Semantic Redundancy (Inter-Group Similarity)")
        print("="*60)
        
        results = {}
        
        for model_name, embs in [('Baseline', baseline_embs), ('ORCA', orca_embs)]:
            tokens = embs['audio_tokens']
            if tokens is None:
                continue
            
            # Reshape to groups: [B, num_groups, queries_per_group, D]
            B, total_tokens, D = tokens.shape
            if total_tokens != num_groups * queries_per_group:
                print(f"  {model_name}: Token count mismatch, skipping")
                continue
            
            grouped = tokens.reshape(B, num_groups, queries_per_group, D)
            
            # Compute group centroids: [B, num_groups, D]
            centroids = grouped.mean(dim=2)
            
            # Compute pairwise cosine similarity
            centroids_norm = F.normalize(centroids, dim=-1)
            # [B, num_groups, num_groups]
            sim_matrix = torch.bmm(centroids_norm, centroids_norm.transpose(1, 2))
            
            # Get off-diagonal similarities (inter-group)
            mask = ~torch.eye(num_groups, dtype=torch.bool).unsqueeze(0)
            inter_group_sim = sim_matrix[mask.expand(B, -1, -1)].mean().item()
            
            results[f'{model_name.lower()}_inter_group_sim'] = inter_group_sim
            print(f"  {model_name} Inter-Group Similarity: {inter_group_sim:.4f}")
            
            # Visualize similarity matrix
            self._plot_similarity_matrix(
                sim_matrix[0].numpy(), 
                model_name, 
                num_groups
            )
        
        # Interpretation
        if 'baseline_inter_group_sim' in results and 'orca_inter_group_sim' in results:
            reduction = (1 - results['orca_inter_group_sim'] / results['baseline_inter_group_sim']) * 100
            results['similarity_reduction'] = reduction
            print(f"\n→ ORCA reduces inter-group similarity by {reduction:.1f}%")
        print("→ Lower similarity = less redundancy, more specialization")
        print("→ This motivates: Orthogonal Grouping")
        
        return results
    
    def _plot_similarity_matrix(self, sim_matrix: np.ndarray, model_name: str, num_groups: int):
        """Plot inter-group similarity matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            sim_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            xticklabels=[f"G{i}" for i in range(num_groups)],
            yticklabels=[f"G{i}" for i in range(num_groups)]
        )
        plt.title(f"{model_name}: Inter-Group Cosine Similarity")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"inter_group_sim_{model_name.lower()}.png", dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir / f'inter_group_sim_{model_name.lower()}.png'}")
    
    # =========================================================================
    # Main Analysis
    # =========================================================================
    def run_all_analyses(self, audio_paths: List[str], labels: Dict = None):
        """Run all 4 analyses."""
        print("\n" + "="*60)
        print("PROCRUSTEAN ALIGNMENT ANALYSIS")
        print("="*60)
        
        # Load models
        baseline_model = self.load_model(self.baseline_model_id)
        orca_model = self.load_model(self.orca_model_id)
        
        # Extract embeddings
        print("\nExtracting baseline embeddings...")
        baseline_embs = self.extract_audio_embeddings(baseline_model, audio_paths)
        
        print("\nExtracting ORCA embeddings...")
        orca_embs = self.extract_audio_embeddings(orca_model, audio_paths)
        
        # Run analyses
        self.results['representation_collapse'] = self.analyze_representation_collapse(
            baseline_embs, orca_embs
        )
        
        if labels:
            self.results['acoustic_info_loss'] = self.analyze_acoustic_info_loss(
                baseline_embs, orca_embs, labels
            )
        
        self.results['semantic_redundancy'] = self.analyze_semantic_redundancy(
            baseline_embs, orca_embs
        )
        
        # Summary
        self._print_summary()
        
        # Save results
        with open(self.output_dir / "analysis_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {self.output_dir / 'analysis_results.json'}")
        
        return self.results
    
    def _print_summary(self):
        """Print summary connecting observations to solutions."""
        print("\n" + "="*60)
        print("SUMMARY: Procrustean Problem → Delocalization Solution")
        print("="*60)
        
        summary = """
┌─────────────────────────────────────────────────────────────────┐
│                    Procrustean Alignment                        │
│                                                                 │
│  Problem: Audio forced to fit text embedding space              │
│           → Acoustic information is lost                        │
│           → Representations collapse                            │
│           → Audio tokens underutilized                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ↓
                        Delocalization
                                │
    ┌───────────┬───────────────┼───────────────┬──────────────┐
    │           │               │               │              │
    ↓           ↓               ↓               ↓              
┌─────────┐ ┌─────────┐ ┌───────────────┐ ┌─────────────┐
│  Input  │ │  Space  │ │    Latent     │ │   Output    │
│ Deloc.  │ │ Deloc.  │ │   Deloc.      │ │   Deloc.    │
├─────────┤ ├─────────┤ ├───────────────┤ ├─────────────┤
│   ASR   │ │Orthogonal│ │ Variational  │ │  Modality   │
│ Dropout │ │Grouping │ │    (μ, σ)     │ │    DPO      │
├─────────┤ ├─────────┤ ├───────────────┤ ├─────────────┤
│Prevents │ │Prevents │ │ Separates    │ │Forces model │
│text-only│ │semantic │ │ semantic(μ)  │ │to use audio │
│reliance │ │redundancy│ │ acoustic(σ)  │ │not just text│
└─────────┘ └─────────┘ └───────────────┘ └─────────────┘
"""
        print(summary)


def main():
    parser = argparse.ArgumentParser(description="Procrustean Alignment Analysis")
    parser.add_argument("--baseline_model", type=str, required=True,
                        help="HuggingFace model ID for baseline")
    parser.add_argument("--orca_model", type=str, required=True,
                        help="HuggingFace model ID for ORCA")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./procrustean_analysis",
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of audio samples to analyze")
    args = parser.parse_args()
    
    # Get audio files
    if args.audio_dir and os.path.isdir(args.audio_dir):
        audio_paths = list(Path(args.audio_dir).glob("*.wav"))[:args.num_samples]
    else:
        # Use a sample dataset
        print("No audio_dir specified, using sample from LibriSpeech...")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_paths = [item["audio"]["path"] for item in ds[:args.num_samples]]
    
    # Run analysis
    analyzer = ProcrusteanAnalyzer(
        baseline_model_id=args.baseline_model,
        orca_model_id=args.orca_model,
        output_dir=args.output_dir
    )
    
    results = analyzer.run_all_analyses(audio_paths)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
