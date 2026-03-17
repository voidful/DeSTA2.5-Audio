#!/usr/bin/env python3
"""
Text Dominance Analysis
========================
Tests whether the model **blindly follows ASR transcripts** and ignores
audio features.

Method
------
For each sample, run generation twice:
  A) **Normal**        : real audio + actual ASR transcript
  B) **Mismatched**    : real audio + INCORRECT (swapped) transcript

If the outputs are (near-)identical to the mismatched text's semantics,
the model is dominated by text and ignores the audio.

We measure:
  - Exact match rate (between normal and mismatched)
  - Token-level overlap rate

Dataset : CREMA-D (has emotion variation with same text → audio should matter)

Usage
-----
python study/text_dominance_analysis.py \\
    --model_id <hf_or_local_desta_model_path> \\
    --out_dir study/output_text_dominance \\
    --num_samples 200
"""

import argparse
import gc
import json
import os
import wave
import tempfile
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Plotting (headless-safe)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from datasets import load_dataset


# ===================================================================== #
#                     Semantic Similarity (SBERT)                       #
# ===================================================================== #

def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    print(f"📐 Loading SBERT model: {model_name}")
    return SentenceTransformer(model_name)


def compute_semantic_similarities(sbert, texts_a: list, texts_b: list) -> list:
    """
    Compute cosine similarity between paired lists of texts.
    Returns list of float similarities in [-1, 1].
    """
    if sbert is None:
        return [None] * len(texts_a)
    import torch as _torch
    embs_a = sbert.encode(texts_a, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    embs_b = sbert.encode(texts_b, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    import torch.nn.functional as _F
    sims = _F.cosine_similarity(embs_a, embs_b, dim=-1).cpu().tolist()
    return sims


# ===================================================================== #
#                           Configuration                               #
# ===================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Text Dominance Analysis")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="myleslinder/crema-d")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=200,
                   help="Number of samples to evaluate (default: 200).")
    p.add_argument("--max_audio_sec", type=float, default=4.0)
    p.add_argument("--out_dir", type=str, default="study/output_text_dominance")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--question", type=str,
                   default="Describe the Audio",
                   help="Question to ask the model about each audio.")
    p.add_argument("--sbert_model", type=str,
                   default="all-MiniLM-L6-v2",
                   help="Sentence-transformers model for semantic similarity.")
    return p.parse_args()


# ===================================================================== #
#                          Helpers                                      #
# ===================================================================== #

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_wav(audio_array, sr, path):
    """Write float audio to 16-bit PCM WAV."""
    arr = np.clip(np.asarray(audio_array, dtype=np.float32), -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def token_overlap(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ===================================================================== #
#                        Load Data                                      #
# ===================================================================== #

def load_cremad_paired(dataset_name, dataset_split, num_samples, seed):
    print(f"📦 Loading dataset: {dataset_name} [{dataset_split}]")
    ds = load_dataset(dataset_name, split=dataset_split)
    
    # Group by text to allow swapping audio for the same transcript
    from collections import defaultdict
    text_to_items = defaultdict(list)
    
    for ex in ds:
        audio_obj = ex.get("audio")
        if audio_obj is None:
            continue
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        text = (ex.get("text") or ex.get("sentence")
                or ex.get("transcription") or str(ex.get("sentence_id", "")))
        emotion = str(ex.get("emotion") or ex.get("label") or "unknown")

        text_to_items[str(text).strip()].append(dict(
            audio=arr, sr=sr, text=str(text).strip(), emotion=emotion
        ))

    paired_items = []
    # Shuffle text keys for randomness
    import random
    random.seed(seed)
    texts = list(text_to_items.keys())
    random.shuffle(texts)

    for txt in texts:
        group = text_to_items[txt]
        if len(group) < 2:
            continue
        
        # Shuffle group to get random pairs
        random.shuffle(group)
        for i in range(0, len(group) - 1, 2):
            item_a = group[i]
            item_b = group[i+1]
            
            # We will use item_a["text"] for both, but swap the audio
            paired_items.append(dict(
                text=item_a["text"],
                audio_a=item_a["audio"],
                audio_b=item_b["audio"],
                sr=item_a["sr"],
                emotion_a=item_a["emotion"],
                emotion_b=item_b["emotion"]
            ))
            if num_samples > 0 and len(paired_items) >= num_samples:
                break
        if num_samples > 0 and len(paired_items) >= num_samples:
            break

    print(f"  Generated {len(paired_items)} pairs (Fixed Transcript, Swapped Audio)")
    return paired_items


# ===================================================================== #
#                     Generate with normal / swapped audio              #
# ===================================================================== #

@torch.inference_mode()
def generate_response(model, wav_path, question, provided_text=None):
    """Generate a text response from the DeSTA model given audio + question."""
    audio_entry = {"audio": wav_path}
    if provided_text is not None:
        audio_entry["text"] = provided_text

    messages = [
        {
            "role": "system",
            "content": "Focus on the audio clips and instructions.",
        },
        {
            "role": "user",
            "content": f"<|AUDIO|> {question}",
            "audios": [audio_entry],
        },
    ]

    outputs = model.generate(
        messages=messages,
        do_sample=False,
        max_new_tokens=512,
    )

    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    return pred.strip() if isinstance(pred, str) else str(pred)

# ===================================================================== #
#                        Plotting                                       #
# ===================================================================== #

def plot_results(results, out_dir, has_semantic: bool = False):
    """Generate summary plots."""
    overlaps = [r["token_overlap"] for r in results]
    semantic_sims = [r["semantic_sim"] for r in results if r["semantic_sim"] is not None]

    ncols = 2 if (not has_semantic or not semantic_sims) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # ---- Semantic similarity histogram (Primary Metric) ----
    if has_semantic and semantic_sims:
        ax = axes[0]
        ax.hist(semantic_sims, bins=20, color="#9b59b6", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(semantic_sims), color="black", ls="--", lw=1.5,
                   label=f"mean = {np.mean(semantic_sims):.3f}")
        ax.set_xlabel("Cosine Similarity (SBERT)")
        ax.set_ylabel("Count")
        ax.set_title("Audio-A vs Audio-B (Fixed Text): Semantic Sim")
        ax.legend()
        ax_emo = axes[1]
    else:
        ax = axes[0]
        ax.hist(overlaps, bins=20, color="#e74c3c", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(overlaps), color="black", ls="--", lw=1.5,
                   label=f"mean = {np.mean(overlaps):.3f}")
        ax.set_xlabel("Token Overlap (Jaccard)")
        ax.set_ylabel("Count")
        ax.set_title("Audio-A vs Audio-B (Fixed Text): Token Overlap")
        ax.legend()
        ax_emo = axes[1]

    # ---- Per-emotion audio sensitivity ----
    # (1 - similarity) represents how much the model changed its output when audio changed
    emotion_groups = {}
    for r in results:
        emo_pair = f"{r['emotion_a']}->{r['emotion_b']}"
        if emo_pair not in emotion_groups:
            emotion_groups[emo_pair] = []
        val = r["semantic_sim"] if r["semantic_sim"] is not None else r["token_overlap"]
        emotion_groups[emo_pair].append(1.0 - val)

    # Sort by sensitivity
    sorted_emos = sorted(emotion_groups.keys(), key=lambda x: np.mean(emotion_groups[x]), reverse=True)
    top_emos = sorted_emos[:8] # Show top 8 transitions
    
    emo_vals = [np.mean(emotion_groups[e]) for e in top_emos]
    
    ax_emo.bar(top_emos, emo_vals, color="#3498db", alpha=0.85)
    ax_emo.set_ylabel("Audio Sensitivity (1 - Similarity)")
    ax_emo.set_xlabel("Emotion Transition")
    ax_emo.set_title("Top 8 Audio-Sensitive Emotion Pairs")
    plt.setp(ax_emo.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "audio_sensitivity_analysis.png"), dpi=150)
    plt.close(fig)


# ===================================================================== #
#                             Main                                      #
# ===================================================================== #

def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"🖥️  device={device}  |  out_dir={out_dir}\n")

    # ---- 1. Load data ----
    pairs = load_cremad_paired(
        args.dataset_name, args.dataset_split, args.num_samples, args.seed,
    )

    # ---- 2. Load model ----
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    print(f"🔄 Loading DeSTA model: {args.model_id}")
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device).eval()

    # ---- 2b. Load SBERT ----
    sbert = load_sbert(args.sbert_model)

    # ---- 3. Run paired generation ----
    results = []
    tmp_wav_a = os.path.join(out_dir, "_tmp_audio_a.wav")
    tmp_wav_b = os.path.join(out_dir, "_tmp_audio_b.wav")

    print(f"\n🔬 Running Audio Sensitivity test on {len(pairs)} pairs …")
    print(f"   (Fixed Transcript, Swapped Audio)")
    print(f"   Question: \"{args.question}\"\n")

    for i, pair in enumerate(pairs):
        write_wav(pair["audio_a"], pair["sr"], tmp_wav_a)
        write_wav(pair["audio_b"], pair["sr"], tmp_wav_b)

        # 1) Generation with Audio A + Fixed Transcript
        pred_a = generate_response(model, tmp_wav_a, args.question, provided_text=pair["text"])

        # 2) Generation with Audio B + Fixed Transcript
        pred_b = generate_response(model, tmp_wav_b, args.question, provided_text=pair["text"])

        overlap = token_overlap(pred_a, pred_b)
        exact = (pred_a == pred_b)

        results.append(dict(
            idx=i,
            text=pair["text"],
            emotion_a=pair["emotion_a"],
            emotion_b=pair["emotion_b"],
            pred_a=pred_a,
            pred_b=pred_b,
            token_overlap=overlap,
            exact_match=exact,
            semantic_sim=None,  # Filled later in batch
        ))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"   [{i+1}/{len(pairs)}]  overlap={overlap:.3f}  "
                  f"diff={'✓' if not exact else '✗'}")
            if i == 0:
                print(f"     Transcript: \"{pair['text']}\"")
                print(f"     A ({pair['emotion_a']}): {pred_a[:100]}...")
                print(f"     B ({pair['emotion_b']}): {pred_b[:100]}...")

        if (i + 1) % 50 == 0:
            free_mem()

    # Cleanup
    for p in [tmp_wav_a, tmp_wav_b]:
        if os.path.exists(p):
            os.remove(p)

    # ---- 3b. Compute semantic similarity (batched) ----
    print("\n📐 Computing semantic similarity (SBERT) …")
    sims = compute_semantic_similarities(
        sbert,
        [r["pred_a"] for r in results],
        [r["pred_b"] for r in results],
    )
    for r, s in zip(results, sims):
        r["semantic_sim"] = s

    has_semantic = any(r["semantic_sim"] is not None for r in results)

    # ---- 4. Summary ----
    exact_rate = np.mean([r["exact_match"] for r in results])
    avg_overlap = np.mean([r["token_overlap"] for r in results])
    valid_sems = [r["semantic_sim"] for r in results if r["semantic_sim"] is not None]
    avg_semantic = float(np.mean(valid_sems)) if valid_sems else None

    # AUDIO SENSITIVITY = 1.0 - SIMILARITY
    # If the model is text-dominant, similarity remains HIGH (near 1.0), so sensitivity is LOW.
    # If the model is audio-aware, similarity should DROP when audio changes, so sensitivity is HIGH.
    sensitivity_score = 1.0 - (avg_semantic if avg_semantic is not None else avg_overlap)

    print(f"\n{'='*55}")
    print(f"  AUDIO SENSITIVITY ANALYSIS (Fixed Text, Swapped Audio)")
    print(f"{'='*55}")
    print(f"  Model              : {args.model_id}")
    print(f"  #Pairs             : {len(results)}")
    print(f"  Question           : {args.question}")
    print(f"{'='*55}")
    print(f"  Avg Semantic Sim   : {avg_semantic:.4f}" if avg_semantic is not None else f"  Avg Token Overlap  : {avg_overlap:.4f}")
    print(f"  Audio Sensitivity  : {sensitivity_score:.4f} (Goal: High)")
    print(f"  Identical Response : {exact_rate*100:.2f}% (Goal: Low)")
    print(f"{'='*55}")

    if sensitivity_score < 0.15:
        print("  ❌ SEVERE: Model is 100% TEXT DOMINANT. Ignoring all audio changes.")
    elif sensitivity_score < 0.35:
        print("  ⚠️  WARNING: Model largely follows text; audio has minimal impact.")
    elif sensitivity_score < 0.55:
        print("  ⚡ MODERATE: Model shows some awareness of acoustic cues.")
    else:
        print("  ✅ GOOD: Model is highly sensitive to audio changes.")

    # ---- 5. Plots ----
    plot_results(results, out_dir, has_semantic=has_semantic)

    # ---- 6. Save report ----
    report = dict(
        config=dict(
            model_id=args.model_id,
            num_samples=len(results),
            question=args.question,
            seed=args.seed,
        ),
        avg_semantic_sim=avg_semantic,
        audio_sensitivity=sensitivity_score,
        identical_response_rate=exact_rate,
        results=results,
    )
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report saved → {report_path}")

    # Show a few examples
    print(f"\n📋 Example sensitivity:")
    for r in results[:5]:
        sem_str = f"{r['semantic_sim']:.3f}" if r["semantic_sim"] is not None else "N/A"
        print(f"  [Text: \"{r['text']}\"]")
        print(f"    Audio A ({r['emotion_a']}): {r['pred_a'][:150]}")
        print(f"    Audio B ({r['emotion_b']}): {r['pred_b'][:150]}")
        print(f"    Similarity: {sem_str}  Exact: {r['exact_match']}")
        print()


if __name__ == "__main__":
    main()
