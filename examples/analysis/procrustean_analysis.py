"""
Mechanistic Analysis: Demonstrating the Procrustean Alignment Problem

在 Colab 上執行這個 cell 來分析 Procrustean Alignment 問題
並引出 Delocalization 四個方法

支援模型:
- DeSTA25 (baseline and ORCA)
- Qwen2-Audio

Analyses:
1. Representation Collapse → motivates Latent Delocalization
2. Semantic Redundancy → motivates Orthogonal Grouping
"""

# ============================================================
# 1. 設定 (請修改這裡)
# ============================================================
MODELS_TO_ANALYZE = {
    "DeSTA-Baseline": "voidful/QAQ_4b",
    "DeSTA-ORCA": "voidful/desta25_4b_R2_full",
    "Qwen2-Audio": "Qwen/Qwen2-Audio-7B",  # 加入 Qwen2-Audio
}
NUM_SAMPLES = 100
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
install_if_needed("librosa")

# ============================================================
# 3. Imports
# ============================================================
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 4. Model Loaders
# ============================================================

def load_desta_model(model_id: str, device):
    """Load a DeSTA model."""
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    from transformers import AutoFeatureExtractor
    
    model = DeSTA25AudioModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    encoder_model_id = getattr(model.config, 'encoder_model_id', 'openai/whisper-large-v3')
    model.processor = AutoFeatureExtractor.from_pretrained(encoder_model_id)
    model.model_type = "desta"
    
    return model


def load_qwen2_audio_model(model_id: str, device):
    """Load Qwen2-Audio model."""
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    model.processor = AutoProcessor.from_pretrained(model_id)
    model.model_type = "qwen2_audio"
    
    return model


def load_model(model_id: str, device):
    """Load model based on model ID."""
    print(f"🔄 Loading model: {model_id}")
    
    if "qwen" in model_id.lower() and "audio" in model_id.lower():
        model = load_qwen2_audio_model(model_id, device)
    else:
        model = load_desta_model(model_id, device)
    
    print(f"✅ Loaded: {model_id}")
    return model

# ============================================================
# 5. Embedding Extractors
# ============================================================

def extract_desta_embeddings(model, audio_samples, device) -> Dict[str, torch.Tensor]:
    """Extract embeddings from DeSTA model."""
    embeddings = {'audio_tokens': []}
    
    for sample in tqdm(audio_samples, desc="Extracting (DeSTA)"):
        try:
            with torch.no_grad():
                audio_array = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                
                if sr != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                
                features = model.processor(
                    audio_array.tolist(), 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device)
                
                result = model.perception.forward_whisper(features)
                tokens = result[0] if isinstance(result, tuple) else result
                embeddings['audio_tokens'].append(tokens.cpu())
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue
    
    if embeddings['audio_tokens']:
        embeddings['audio_tokens'] = torch.cat(embeddings['audio_tokens'], dim=0)
    else:
        embeddings['audio_tokens'] = None
    
    return embeddings


def extract_qwen2_audio_embeddings(model, audio_samples, device) -> Dict[str, torch.Tensor]:
    """Extract embeddings from Qwen2-Audio model."""
    embeddings = {'audio_tokens': []}
    
    for sample in tqdm(audio_samples, desc="Extracting (Qwen2-Audio)"):
        try:
            with torch.no_grad():
                audio_array = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                
                if sr != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                
                # Qwen2-Audio requires both text and audio
                # Use a dummy conversation format
                conversation = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": "dummy"},
                        {"type": "text", "text": "Describe this audio."},
                    ]}
                ]
                text = model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                
                inputs = model.processor(
                    text=text,
                    audios=[audio_array],
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Get audio encoder outputs directly
                if hasattr(model, 'audio_tower') or hasattr(model.model, 'audio_tower'):
                    audio_tower = getattr(model, 'audio_tower', None) or model.model.audio_tower
                    # Get input_features for audio
                    input_features = inputs.get("input_features")
                    if input_features is not None:
                        audio_embeds = audio_tower(input_features)
                        if hasattr(audio_embeds, 'last_hidden_state'):
                            audio_embeds = audio_embeds.last_hidden_state
                        embeddings['audio_tokens'].append(audio_embeds.float().cpu())
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue
    
    if embeddings['audio_tokens']:
        embeddings['audio_tokens'] = torch.cat(embeddings['audio_tokens'], dim=0)
    else:
        embeddings['audio_tokens'] = None
    
    return embeddings


def extract_embeddings(model, audio_samples, device) -> Dict[str, torch.Tensor]:
    """Extract embeddings based on model type."""
    model_type = getattr(model, 'model_type', 'desta')
    
    if model_type == "qwen2_audio":
        return extract_qwen2_audio_embeddings(model, audio_samples, device)
    else:
        return extract_desta_embeddings(model, audio_samples, device)

# ============================================================
# 6. Analysis Functions
# ============================================================

def analyze_multiple_models(models_embs: Dict[str, Dict], output_dir: Path):
    """Analyze multiple models and create comparison plots."""
    results = {}
    
    # Analysis 1: Token Variance
    print("\n" + "="*60)
    print("📊 Analysis 1: Representation Collapse (Token Variance)")
    print("="*60)
    
    variances = {}
    for name, embs in models_embs.items():
        tokens = embs.get('audio_tokens')
        if tokens is not None:
            var = tokens.var(dim=1).mean().item()
            variances[name] = var
            print(f"  {name}: {var:.4f}")
    
    results['token_variance'] = variances
    
    # Plot variance comparison
    if variances:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(variances.keys())
        values = list(variances.values())
        colors = ['#ff6b6b' if 'ORCA' not in n else '#4ecdc4' for n in names]
        
        bars = ax.bar(names, values, color=colors)
        ax.set_ylabel("Token Variance")
        ax.set_title("Representation Collapse Analysis\n(Higher = Less Collapse)")
        ax.set_ylim(0, max(values) * 1.2)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=12)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "1_token_variance_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # Analysis 2: t-SNE Visualization
    print("\n" + "="*60)
    print("📊 Analysis 2: t-SNE Visualization")
    print("="*60)
    
    n_models = len(models_embs)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, embs) in zip(axes, models_embs.items()):
        tokens = embs.get('audio_tokens')
        if tokens is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name)
            continue
        
        flat = tokens.reshape(-1, tokens.shape[-1]).numpy()
        n_samples = min(2000, flat.shape[0])
        indices = np.random.choice(flat.shape[0], n_samples, replace=False)
        sampled = flat[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(sampled)
        
        color = '#4ecdc4' if 'ORCA' in name else '#ff6b6b'
        ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=5, c=color)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    
    plt.suptitle("Token Distribution Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "2_tsne_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print("\nToken Variance (higher = less collapse):")
    for name, var in sorted(variances.items(), key=lambda x: x[1], reverse=True):
        marker = "✅" if "ORCA" in name else "❌"
        print(f"  {marker} {name}: {var:.4f}")
    
    if "DeSTA-Baseline" in variances and "DeSTA-ORCA" in variances:
        improvement = (variances["DeSTA-ORCA"] / variances["DeSTA-Baseline"] - 1) * 100
        print(f"\n  → ORCA improves variance by {improvement:.1f}%")
    
    return results


def print_framework_summary():
    """Print the Delocalization framework summary."""
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

# ============================================================
# 7. Main Runner
# ============================================================

def main():
    """Run analysis on multiple models."""
    print("="*60)
    print("🔬 PROCRUSTEAN ALIGNMENT ANALYSIS")
    print("    Comparing Multiple Audio-LLMs")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n📦 Loading LibriSpeech samples...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    samples = list(ds.select(range(min(NUM_SAMPLES, len(ds)))))
    print(f"  Loaded {len(samples)} samples")
    
    # Extract embeddings from each model
    all_embeddings = {}
    
    for model_name, model_id in MODELS_TO_ANALYZE.items():
        try:
            model = load_model(model_id, device)
            embs = extract_embeddings(model, samples, device)
            all_embeddings[model_name] = embs
            
            # Free memory
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
            continue
    
    # Run analysis
    results = analyze_multiple_models(all_embeddings, output_dir)
    
    # Print framework summary
    print_framework_summary()
    
    # Save results
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return obj
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(convert_floats(results), f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("\n✅ Done!")
    
    return results


# ============================================================
# 8. 執行
# ============================================================
if __name__ == "__main__" or 'google.colab' in str(globals()):
    results = main()
