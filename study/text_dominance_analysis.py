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

def load_cremad(dataset_name, dataset_split, num_samples, seed):
    print(f"📦 Loading dataset: {dataset_name} [{dataset_split}]")
    ds = load_dataset(dataset_name, split=dataset_split)
    ds = ds.shuffle(seed=seed)
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    items = []
    for ex in ds:
        audio_obj = ex.get("audio")
        if audio_obj is None:
            continue
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        text = (ex.get("text") or ex.get("sentence")
                or ex.get("transcription") or str(ex.get("sentence_id", "")))
        emotion = str(ex.get("emotion") or ex.get("label") or "unknown")

        items.append(dict(
            audio=arr, sr=sr, text=str(text), emotion=emotion,
            swapped_text="" # Will be filled below
        ))

    # Assign a completely mismatched text to each item
    for i in range(len(items)):
        j = (i + len(items) // 2) % len(items)
        items[i]["swapped_text"] = items[j]["text"]

    print(f"  Loaded {len(items)} samples")
    return items


# ===================================================================== #
#                     Generate with normal / zeroed audio               #
# ===================================================================== #

def generate_response(model, wav_path, question, provided_text=None, do_sample=False):
    """Run model.generate() and return text output."""
    audio_node = {"audio": wav_path}
    if provided_text is not None:
        audio_node["text"] = provided_text
    
    messages = [
        {"role": "system", "content": "You are an audio assistant."},
        {"role": "user",
         "content": f"<|AUDIO|>\n\n{question}",
         "audios": [audio_node]},
    ]
    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=do_sample,
            max_new_tokens=256,
        )
    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    return pred.strip()


# (Removed monkey-patching logic as we now test text swapping)


# ===================================================================== #
#                        Plotting                                       #
# ===================================================================== #

def plot_results(results, out_dir, has_semantic: bool = False):
    """Generate summary plots."""
    overlaps = [r["token_overlap"] for r in results]
    semantic_sims = [r["semantic_sim"] for r in results if r["semantic_sim"] is not None]

    ncols = 2 if (not has_semantic or not semantic_sims) else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # ---- Token overlap histogram ----
    ax = axes[0]
    ax.hist(overlaps, bins=20, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(overlaps), color="black", ls="--", lw=1.5,
               label=f"mean = {np.mean(overlaps):.3f}")
    ax.set_xlabel("Token Overlap (Jaccard)")
    ax.set_ylabel("Count")
    ax.set_title("Normal vs Swapped-Text: Token Overlap")
    ax.legend()

    # ---- Semantic similarity histogram ----
    if has_semantic and semantic_sims:
        ax = axes[1]
        ax.hist(semantic_sims, bins=20, color="#9b59b6", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(semantic_sims), color="black", ls="--", lw=1.5,
                   label=f"mean = {np.mean(semantic_sims):.3f}")
        ax.set_xlabel("Cosine Similarity (SBERT)")
        ax.set_ylabel("Count")
        ax.set_title("Normal vs Swapped-Text: Semantic Similarity")
        ax.legend()
        ax_emo = axes[2]
    else:
        ax_emo = axes[1]

    # ---- Per-emotion semantic sim ----
    emotion_groups = {}
    for r in results:
        emo = r["emotion"]
        if emo not in emotion_groups:
            emotion_groups[emo] = {"overlap": [], "semantic": []}
        emotion_groups[emo]["overlap"].append(r["token_overlap"])
        if r["semantic_sim"] is not None:
            emotion_groups[emo]["semantic"].append(r["semantic_sim"])

    emotions = sorted(emotion_groups.keys())
    if has_semantic and semantic_sims:
        emo_vals = [np.mean(emotion_groups[e]["semantic"]) if emotion_groups[e]["semantic"] else 0.0
                    for e in emotions]
        y_label = "Avg Semantic Sim (SBERT)"
        title = "Semantic Similarity by Emotion"
    else:
        emo_vals = [np.mean(emotion_groups[e]["overlap"]) for e in emotions]
        y_label = "Avg Token Overlap"
        title = "Token Overlap by Emotion"

    ax_emo.bar(emotions, emo_vals, color="#3498db", alpha=0.85)
    ax_emo.set_ylabel(y_label)
    ax_emo.set_xlabel("Emotion")
    ax_emo.set_title(title)
    ax_emo.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "similarity_analysis.png"), dpi=150)
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
    items = load_cremad(
        args.dataset_name, args.dataset_split, args.num_samples, args.seed,
    )

    # ---- 2. Load model ----
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    print(f"🔄 Loading DeSTA model: {args.model_id}")
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device).eval()

    # ---- 2b. Load SBERT ----
    sbert = load_sbert(args.sbert_model)

    # ---- 3. Run A/B generation ----
    results = []
    tmp_wav = os.path.join(out_dir, "_tmp_audio.wav")

    print(f"\n🔬 Running text dominance test on {len(items)} samples …")
    print(f"   Question: \"{args.question}\"\n")

    for i, item in enumerate(items):
        write_wav(item["audio"], item["sr"], tmp_wav)

        # A) Normal generation (uses actual text or ASR)
        pred_normal = generate_response(model, tmp_wav, args.question, provided_text=item["text"])

        # B) Mismatched-text generation (uses wrong text)
        pred_mismatched = generate_response(model, tmp_wav, args.question, provided_text=item["swapped_text"])

        overlap = token_overlap(pred_normal, pred_mismatched)
        exact = (pred_normal == pred_mismatched)

        results.append(dict(
            idx=i,
            text_real=item["text"],
            text_swapped=item["swapped_text"],
            emotion=item["emotion"],
            pred_normal=pred_normal,
            pred_mismatched=pred_mismatched,
            token_overlap=overlap,
            exact_match=exact,
            semantic_sim=None,  # Filled later in batch
        ))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"   [{i+1}/{len(items)}]  overlap={overlap:.3f}  "
                  f"exact={'✓' if exact else '✗'}")
            if i == 0:
                print(f"     Normal (\"{item['text'][:30]}...\") : {pred_normal[:120]}")
                print(f"     Swapped (\"{item['swapped_text'][:30]}...\") : {pred_mismatched[:120]}")

        if (i + 1) % 50 == 0:
            free_mem()

    # Cleanup
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    # ---- 3b. Compute semantic similarity (batched) ----
    print("\n📐 Computing semantic similarity (SBERT) …")
    sims = compute_semantic_similarities(
        sbert,
        [r["pred_normal"] for r in results],
        [r["pred_mismatched"] for r in results],
    )
    for r, s in zip(results, sims):
        r["semantic_sim"] = s

    has_semantic = any(r["semantic_sim"] is not None for r in results)

    # ---- 4. Summary ----
    exact_rate = np.mean([r["exact_match"] for r in results])
    avg_overlap = np.mean([r["token_overlap"] for r in results])
    valid_sems = [r["semantic_sim"] for r in results if r["semantic_sim"] is not None]
    avg_semantic = float(np.mean(valid_sems)) if valid_sems else None

    print(f"\n{'='*55}")
    print(f"  Text Dominance Analysis")
    print(f"{'='*55}")
    print(f"  Model              : {args.model_id}")
    print(f"  #Samples           : {len(results)}")
    print(f"  Question           : {args.question}")
    print(f"{'='*55}")
    print(f"  Exact match rate   : {exact_rate:.4f}  ({sum(r['exact_match'] for r in results)}/{len(results)})")
    print(f"  Avg token overlap  : {avg_overlap:.4f}")
    if avg_semantic is not None:
        print(f"  Avg semantic sim   : {avg_semantic:.4f}  (1.0 = identical meaning)")
    print(f"{'='*55}")

    # Threshold on semantic similarity if available, else fall back to token overlap
    dominance_score = avg_semantic if avg_semantic is not None else avg_overlap
    if dominance_score > 0.85:
        print("  ❌ SEVERE: Model almost 100% blindly follows text (ignores audio)")
    elif dominance_score > 0.65:
        print("  ⚠️  WARNING: Model largely follows text with minimal audio influence")
    elif dominance_score > 0.45:
        print("  ⚡ MODERATE: Model partially uses audio features")
    else:
        print("  ✅ GOOD: Model meaningfully uses audio features")

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
        exact_match_rate=exact_rate,
        avg_token_overlap=avg_overlap,
        avg_semantic_sim=avg_semantic,
        results=results,
    )
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report saved → {report_path}")

    # Show a few examples
    print(f"\n📋 Example comparisons:")
    for r in results[:5]:
        sem_str = f"{r['semantic_sim']:.3f}" if r["semantic_sim"] is not None else "N/A"
        print(f"  [{r['emotion']}] Audio Text: \"{r['text_real']}\"")
        print(f"    Normal  : {r['pred_normal'][:150]}")
        print(f"    Swapped text given: \"{r['text_swapped']}\"")
        print(f"    Swapped : {r['pred_mismatched'][:150]}")
        print(f"    Overlap: {r['token_overlap']:.3f}  SemanticSim: {sem_str}  Exact: {r['exact_match']}")
        print()


if __name__ == "__main__":
    main()
