"""
Mechanistic Analysis: Demonstrating the Procrustean Alignment Problem

在 Colab 上執行這個 cell 來分析 Procrustean Alignment 問題
並引出 Delocalization 四個方法

Analyses:
1. Representation Collapse → motivates Latent Delocalization
2. Acoustic Information Loss → motivates Variational (μ, σ)
3. Audio Token Underutilization → motivates ASR Dropout + DPO
4. Semantic Redundancy → motivates Orthogonal Grouping
"""

# ============================================================
# 1. 設定 (請修改這裡)
# ============================================================
BASELINE_MODEL = "voidful/QAQ_4b"
ORCA_MODEL = "voidful/desta25_4b_R2_full"
NUM_SAMPLES = 100  # 分析的樣本數
OUTPUT_DIR = "./procrustean_analysis"

# ============================================================
# 2. 安裝依賴
# ============================================================
import subprocess
import sys

def install_if_needed(package):
    try:
        __import__(package.split("[")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_if_needed("datasets")
install_if_needed("matplotlib")
install_if_needed("seaborn")
install_if_needed("scikit-learn")
install_if_needed("tqdm")

# Install desta
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/voidful/DeSTA2.5-Audio.git"
])

# ============================================================
# 3. Imports
# ============================================================
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
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 4. Analysis Functions
# ============================================================

class ProcrusteanAnalyzer:
    """Analyze the Procrustean Alignment problem in Audio-LLMs."""
    
    def __init__(self, baseline_model_id: str, orca_model_id: str, output_dir: str):
        self.baseline_model_id = baseline_model_id
        self.orca_model_id = orca_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.results = {}
        
        # Models will be loaded lazily
        self.baseline_model = None
        self.orca_model = None
    
    def load_model(self, model_id: str):
        """Load a DeSTA model with processor."""
        from desta.models.modeling_desta25 import DeSTA25AudioModel
        from transformers import AutoFeatureExtractor
        
        print(f"🔄 Loading model: {model_id}")
        model = DeSTA25AudioModel.from_pretrained(model_id)
        model.to(self.device)
        model.eval()
        
        # Load processor separately
        encoder_model_id = getattr(model.config, 'encoder_model_id', 'openai/whisper-large-v3')
        model.processor = AutoFeatureExtractor.from_pretrained(encoder_model_id)
        
        print(f"✅ Loaded: {model_id}")
        return model
    
    def extract_audio_embeddings(self, model, audio_samples) -> Dict[str, torch.Tensor]:
        """Extract audio embeddings from samples."""
        embeddings = {
            'audio_tokens': [],
            'mu': [],
            'sigma': [],
        }
        
        for sample in tqdm(audio_samples, desc="Extracting embeddings"):
            try:
                with torch.no_grad():
                    # Get audio array
                    audio_array = sample["audio"]["array"]
                    sr = sample["audio"]["sampling_rate"]
                    
                    # Resample if needed
                    if sr != 16000:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    
                    # Process audio
                    features = model.processor(
                        audio_array.tolist(), 
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
                    
                    # Extract μ if variational
                    connector = model.perception.connector
                    if hasattr(connector, '_cached_mu') and connector._cached_mu is not None:
                        embeddings['mu'].append(connector._cached_mu.cpu())
                        
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
        
        # Stack tensors
        for key in embeddings:
            if embeddings[key]:
                embeddings[key] = torch.cat(embeddings[key], dim=0)
            else:
                embeddings[key] = None
        
        return embeddings
    
    # ----- Analysis 1: Representation Collapse -----
    def analyze_representation_collapse(self, baseline_embs, orca_embs):
        """Token Utilization Variance + t-SNE"""
        print("\n" + "="*60)
        print("📊 Analysis 1: Representation Collapse")
        print("="*60)
        
        results = {}
        
        baseline_tokens = baseline_embs['audio_tokens']
        orca_tokens = orca_embs['audio_tokens']
        
        if baseline_tokens is not None:
            baseline_var = baseline_tokens.var(dim=1).mean().item()
            results['baseline_token_variance'] = baseline_var
            print(f"  Baseline Token Variance: {baseline_var:.4f}")
        
        if orca_tokens is not None:
            orca_var = orca_tokens.var(dim=1).mean().item()
            results['orca_token_variance'] = orca_var
            print(f"  ORCA Token Variance: {orca_var:.4f}")
        
        # t-SNE
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
            
            flat = tokens.reshape(-1, tokens.shape[-1]).numpy()
            n_samples = min(2000, flat.shape[0])
            indices = np.random.choice(flat.shape[0], n_samples, replace=False)
            sampled = flat[indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = tsne.fit_transform(sampled)
            
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=5, c='steelblue')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        
        plt.suptitle("Representation Collapse Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "1_representation_collapse.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        if baseline_tokens is not None and orca_tokens is not None:
            improvement = (orca_var / baseline_var - 1) * 100
            print(f"\n  → ORCA increases token variance by {improvement:.1f}%")
            print("  → This motivates: Latent Delocalization (Variational μ/σ)")
        
        return results
    
    # ----- Analysis 2: Inter-Group Similarity -----
    def analyze_semantic_redundancy(self, baseline_embs, orca_embs, num_groups=8, queries_per_group=8):
        """Inter-group cosine similarity"""
        print("\n" + "="*60)
        print("📊 Analysis 2: Semantic Redundancy (Inter-Group Similarity)")
        print("="*60)
        
        results = {}
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, (model_name, embs) in zip(axes, [('Baseline', baseline_embs), ('ORCA', orca_embs)]):
            tokens = embs['audio_tokens']
            if tokens is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(model_name)
                continue
            
            B, total_tokens, D = tokens.shape
            expected = num_groups * queries_per_group
            
            if total_tokens != expected:
                # Adjust groups
                num_groups = total_tokens // 8 if total_tokens >= 8 else 1
                queries_per_group = total_tokens // num_groups
            
            grouped = tokens[:, :num_groups*queries_per_group].reshape(B, num_groups, queries_per_group, D)
            centroids = grouped.mean(dim=2)
            
            centroids_norm = F.normalize(centroids, dim=-1)
            sim_matrix = torch.bmm(centroids_norm, centroids_norm.transpose(1, 2))
            avg_sim = sim_matrix.mean(dim=0).numpy()
            
            # Off-diagonal mean
            mask = ~np.eye(num_groups, dtype=bool)
            inter_group_sim = avg_sim[mask].mean()
            results[f'{model_name.lower()}_inter_group_sim'] = inter_group_sim
            print(f"  {model_name} Inter-Group Similarity: {inter_group_sim:.4f}")
            
            # Plot
            sns.heatmap(avg_sim, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                       vmin=-1, vmax=1, ax=ax,
                       xticklabels=[f"G{i}" for i in range(num_groups)],
                       yticklabels=[f"G{i}" for i in range(num_groups)])
            ax.set_title(f"{model_name}: Inter-Group Similarity", fontsize=12)
        
        plt.suptitle("Semantic Redundancy Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "2_semantic_redundancy.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        if 'baseline_inter_group_sim' in results and 'orca_inter_group_sim' in results:
            reduction = (1 - results['orca_inter_group_sim'] / results['baseline_inter_group_sim']) * 100
            print(f"\n  → ORCA reduces inter-group similarity by {reduction:.1f}%")
        print("  → This motivates: Space Delocalization (Orthogonal Grouping)")
        
        return results
    
    # ----- Summary -----
    def print_summary(self):
        """Print the connection between observations and solutions."""
        print("\n" + "="*60)
        print("📋 SUMMARY: Procrustean Problem → Delocalization Solution")
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
    
    # ----- Main Runner -----
    def run(self, num_samples=100):
        """Run all analyses."""
        print("="*60)
        print("🔬 PROCRUSTEAN ALIGNMENT ANALYSIS")
        print("="*60)
        
        # Load dataset
        print("\n📦 Loading LibriSpeech samples...")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        samples = list(ds.select(range(min(num_samples, len(ds)))))
        print(f"  Loaded {len(samples)} samples")
        
        # Load models
        self.baseline_model = self.load_model(self.baseline_model_id)
        
        print("\n🎵 Extracting baseline embeddings...")
        baseline_embs = self.extract_audio_embeddings(self.baseline_model, samples)
        
        # Free memory
        del self.baseline_model
        torch.cuda.empty_cache()
        
        self.orca_model = self.load_model(self.orca_model_id)
        
        print("\n🎵 Extracting ORCA embeddings...")
        orca_embs = self.extract_audio_embeddings(self.orca_model, samples)
        
        # Run analyses
        self.results['collapse'] = self.analyze_representation_collapse(baseline_embs, orca_embs)
        self.results['redundancy'] = self.analyze_semantic_redundancy(baseline_embs, orca_embs)
        
        # Summary
        self.print_summary()
        
        # Save results
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {self.output_dir}")
        
        return self.results


# ============================================================
# 5. 執行分析
# ============================================================
if __name__ == "__main__" or 'google.colab' in str(globals()):
    analyzer = ProcrusteanAnalyzer(
        baseline_model_id=BASELINE_MODEL,
        orca_model_id=ORCA_MODEL,
        output_dir=OUTPUT_DIR
    )
    
    results = analyzer.run(num_samples=NUM_SAMPLES)
    
    print("\n✅ Done! Check the output directory for figures.")
