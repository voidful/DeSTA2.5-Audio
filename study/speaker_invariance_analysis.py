#!/usr/bin/env python3
"""
Speaker Invariance Analysis — Cross-speaker Cosine Similarity
=============================================================
For each sentence in CREMA-D, many different speakers say the same text.
If the connector learns a speaker-invariant semantic representation,
then query vectors for the same sentence (different speakers) should
cluster tightly → **high cross-speaker cosine similarity (S_same)**.

Because different LLM backbones exhibit different representation
anisotropies, absolute cosine similarities cannot be compared directly
across models.  We therefore also compute a *random baseline* S_random:
the average cosine similarity between queries from randomly paired samples
that share neither sentence, speaker, nor emotion.

The key metric reported is the *Relative / Corrected Cosine Similarity*:
    ΔS = S_same − S_random

This script:
1. Extracts query vectors from a DeSTA model.
2. Groups samples by sentence (text).
3. Computes S_same  — cross-speaker cosine similarity (same sentence).
4. Computes S_random — null-hypothesis baseline (random pairs).
5. Computes ΔS = S_same − S_random.
6. Visualises all three metrics per query index and per sentence.
7. For ORCA models, aggregates by group.

Outputs
-------
- cosine_sim_per_query.png        : bar chart of S_same / S_random / ΔS per query slot
- cosine_sim_heatmap.png          : heatmap (rows=sentence, cols=query index)
- cosine_sim_by_group.png         : (ORCA only) per-group summary
- report.json                     : numeric summary

Usage
-----
python study/speaker_invariance_analysis.py \\
    --model_id <hf_or_local_desta_model_path> \\
    --out_dir study/output_invariance
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

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
#                           Configuration                               #
# ===================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Speaker Invariance Analysis — Cross-speaker Cosine Similarity"
    )
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="myleslinder/crema-d")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=0, help="0 = ALL")
    p.add_argument("--max_audio_sec", type=float, default=4.0)
    p.add_argument("--trim_strategy", type=str, default="end",
                   choices=["head", "end", "random"])
    p.add_argument("--out_dir", type=str, default="study/output_invariance")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_projected", action="store_true",
                   help="Analyse queries after LLM projection (default: before).")
    p.add_argument("--n_random_pairs", type=int, default=2000,
                   help="Number of random pairs used to estimate S_random.")
    return p.parse_args()


# ===================================================================== #
#                          Utility helpers                              #
# ===================================================================== #

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trim_audio(arr, sr, max_seconds, strategy, seed):
    max_len = int(max_seconds * sr)
    if arr.shape[0] <= max_len:
        return arr.astype(np.float32)
    if strategy == "head":
        return arr[:max_len].astype(np.float32)
    if strategy == "random":
        rng = np.random.RandomState(seed)
        start = int(rng.randint(0, arr.shape[0] - max_len + 1))
        return arr[start : start + max_len].astype(np.float32)
    return arr[-max_len:].astype(np.float32)


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cosine_similarity_matrix_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each pair (a[i], b[i]).
    a, b : [N, D]
    Returns : [N]
    """
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=-1)


# ===================================================================== #
#                        1. Load CREMA-D Data                           #
# ===================================================================== #

def load_cremad(dataset_name, dataset_split, num_samples,
                max_audio_seconds, trim_strategy, seed):
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
        arr = trim_audio(arr, sr, max_audio_seconds, trim_strategy, seed)

        text = (ex.get("text") or ex.get("sentence")
                or ex.get("transcription") or str(ex.get("sentence_id", "")))
        emotion = str(ex.get("emotion") or ex.get("label") or "unknown")
        speaker = str(ex.get("actor_id") or ex.get("speaker_id")
                      or ex.get("speaker") or "unknown")

        items.append(dict(audio=arr, sr=sr, text=str(text),
                          emotion=emotion, speaker=speaker))

    print(f"  Loaded {len(items)} samples  |  "
          f"unique_texts={len(set(i['text'] for i in items))}  |  "
          f"speakers={len(set(i['speaker'] for i in items))}")
    return items


# ===================================================================== #
#         2. Load Model & Extract Query Vectors                         #
# ===================================================================== #

def load_model(model_id, device):
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    from transformers import AutoFeatureExtractor

    print(f"🔄 Loading DeSTA model: {model_id}")
    dtype = torch.float16 if "cuda" in device else torch.float32
    model = DeSTA25AudioModel.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(device).eval()
    encoder_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    processor = AutoFeatureExtractor.from_pretrained(encoder_id)
    return model, processor


def _get_target_layer_ids(encoder_model_id):
    mapping = {
        "openai/whisper-tiny": [0, 1, 2, 3],
        "openai/whisper-small": [2, 5, 8, 11],
        "openai/whisper-medium": [5, 11, 17, 23],
        "openai/whisper-large-v3": [7, 15, 23, 31],
        "openai/whisper-large-v3-turbo": [7, 15, 23, 31],
    }
    if encoder_model_id not in mapping:
        raise NotImplementedError(f"encoder {encoder_model_id} not supported")
    return mapping[encoder_model_id]


@torch.inference_mode()
def extract_query_vectors(model, processor, items, device,
                          use_projected=False):
    """Return query vectors as np.ndarray [N, K, D]."""
    perception = model.perception
    connector = perception.connector
    K = model.config.prompt_size
    encoder_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    target_layer_ids = _get_target_layer_ids(encoder_id)
    is_orca = model.config.connector_mode in ["orca_r1", "orca_hybrid"]

    if is_orca and not use_projected:
        print("ℹ️  ORCA connectors fuse projection — using projected tokens.")

    all_queries = []
    print(f"🎧 Extracting {K} query vectors from {len(items)} samples …")

    for i, item in enumerate(items):
        audio, sr = item["audio"], item["sr"]
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(device)

        whisper_enc = perception.whisper.model.encoder
        target_dtype = whisper_enc.conv1.weight.dtype
        target_dev = whisper_enc.conv1.weight.device
        feats = feats.to(dtype=target_dtype, device=target_dev)

        h = torch.nn.functional.gelu(whisper_enc.conv1(feats))
        h = torch.nn.functional.gelu(whisper_enc.conv2(h))
        h = h.permute(0, 2, 1)
        pos = whisper_enc.embed_positions.weight[
            : whisper_enc.config.max_source_positions
        ].to(dtype=h.dtype, device=h.device)
        hidden = h + pos

        if is_orca:
            all_layer_outputs = []
            for enc_layer in whisper_enc.layers:
                hidden = enc_layer(hidden, attention_mask=None)[0]
                all_layer_outputs.append(hidden)
            conn_out = connector(all_layer_outputs)
            query_vecs = conn_out[0] if isinstance(conn_out, tuple) else conn_out
        else:
            layer_prompt_outputs = []
            for idx, enc_layer in enumerate(whisper_enc.layers):
                hidden = enc_layer(hidden, attention_mask=None)[0]
                if idx in target_layer_ids:
                    lp = connector.layer_prompts[
                        target_layer_ids.index(idx)
                    ].expand(1, -1, -1)
                    qf_out = connector.qformer(
                        lp, encoder_hidden_states=hidden,
                    )
                    layer_prompt_outputs.append(
                        qf_out.last_hidden_state[:, :K, :]
                    )
            stacked = torch.stack(layer_prompt_outputs, dim=0)
            stacked = stacked.permute(1, 2, 0, 3)
            norm_w = torch.softmax(connector.layer_weights, dim=-1).unsqueeze(-1)
            query_vecs = (stacked * norm_w).sum(dim=2)
            if use_projected:
                query_vecs = connector.proj(query_vecs)

        all_queries.append(query_vecs[0].float().cpu().numpy())

        if (i + 1) % 50 == 0:
            free_mem()
            print(f"   [{i+1}/{len(items)}]")

    free_mem()
    queries = np.stack(all_queries, axis=0)
    print(f"   → queries shape: {queries.shape}")
    return queries


# ===================================================================== #
#   3. Cross-speaker Cosine Similarity: S_same & S_random               #
# ===================================================================== #

def compute_s_same(queries: np.ndarray, items: List[Dict]) -> Tuple[
    List[str], np.ndarray, List[int]
]:
    """
    S_same: average pairwise cosine similarity for (same sentence, different speaker).

    For each sentence group we compute all distinct speaker pairs and
    average their per-query cosine similarity.

    Returns
    -------
    sentence_labels : list[str]        — sorted unique sentences
    ssame_matrix    : np.ndarray [S, K] — S_same per (sentence, query)
    count_per_sent  : list[int]        — #speakers per sentence
    """
    N, K, D = queries.shape

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        groups[it["text"]].append(idx)

    sentence_labels = sorted(groups.keys())
    S = len(sentence_labels)
    ssame_matrix = np.zeros((S, K), dtype=np.float64)
    count_per_sent = []

    for s_i, sent in enumerate(sentence_labels):
        idxs = groups[sent]
        count_per_sent.append(len(idxs))
        sub_q = queries[idxs]  # [n_spk, K, D]
        n_spk = len(idxs)

        if n_spk < 2:
            # Cannot compute pairwise similarity with one speaker
            ssame_matrix[s_i] = np.nan
            continue

        # Collect all distinct pairs
        pair_sims = []  # list of [K] arrays
        for a in range(n_spk):
            for b in range(a + 1, n_spk):
                qa = sub_q[a]  # [K, D]
                qb = sub_q[b]  # [K, D]
                # Per-query cosine similarity  [K]
                sim = cosine_similarity_matrix_row(qa, qb)
                pair_sims.append(sim)

        ssame_matrix[s_i] = np.stack(pair_sims, axis=0).mean(axis=0)

    return sentence_labels, ssame_matrix, count_per_sent


def compute_s_random(
    queries: np.ndarray,
    items: List[Dict],
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """
    S_random: null-hypothesis baseline.

    Randomly sample pairs (i, j) where:
      - items[i].text  ≠ items[j].text
      - items[i].speaker ≠ items[j].speaker
      - items[i].emotion ≠ items[j].emotion

    Returns
    -------
    s_random : np.ndarray [K]  — per-query average cosine similarity
    """
    N, K, D = queries.shape
    rng = np.random.RandomState(seed)

    texts    = np.array([it["text"]    for it in items])
    speakers = np.array([it["speaker"] for it in items])
    emotions = np.array([it["emotion"] for it in items])

    collected_sims = []  # list of [K] arrays
    attempts = 0
    max_attempts = n_pairs * 50

    print(f"🎲 Estimating S_random from up to {n_pairs} random pairs …")

    while len(collected_sims) < n_pairs and attempts < max_attempts:
        i, j = rng.randint(0, N, size=2)
        attempts += 1
        if i == j:
            continue
        if texts[i] == texts[j]:
            continue
        if speakers[i] == speakers[j]:
            continue
        if emotions[i] == emotions[j]:
            continue
        sim = cosine_similarity_matrix_row(queries[i], queries[j])  # [K]
        collected_sims.append(sim)

    if len(collected_sims) == 0:
        raise RuntimeError("Could not sample any valid random pairs. "
                           "Check that the dataset has sufficient diversity.")

    actual = len(collected_sims)
    if actual < n_pairs:
        print(f"   ⚠️  Only {actual} valid random pairs found (requested {n_pairs})")
    else:
        print(f"   ✅ Sampled {actual} valid random pairs")

    s_random = np.stack(collected_sims, axis=0).mean(axis=0)  # [K]
    return s_random


# ===================================================================== #
#                        4. Plotting                                    #
# ===================================================================== #

def plot_cosine_sim_per_query(
    ssame_matrix: np.ndarray,
    s_random: np.ndarray,
    sentence_labels: List[str],
    out_dir: str,
    num_groups=None,
    queries_per_group=None,
):
    """
    Three-panel bar chart per query slot:
      top    : S_same  (cross-speaker, same sentence)
      middle : S_random (null-hypothesis baseline)
      bottom : ΔS = S_same − S_random
    """
    K = ssame_matrix.shape[1]
    # Use nanmean to skip single-speaker sentences
    s_same_avg = np.nanmean(ssame_matrix, axis=0)   # [K]
    delta_s    = s_same_avg - s_random               # [K]

    colors = ["#3498db"] * K
    if num_groups and queries_per_group:
        cmap = plt.cm.get_cmap("tab10", num_groups)
        colors = [cmap(i // queries_per_group) for i in range(K)]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, K * 0.22), 11), sharex=True)

    def _bar(ax, values, title, ylabel, color_list, ref_line=None, ref_label=""):
        ax.bar(range(K), values, color=color_list, alpha=0.85, edgecolor="none")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, K - 0.5)
        if num_groups and queries_per_group:
            for g in range(1, num_groups):
                ax.axvline(g * queries_per_group - 0.5, color="gray",
                           ls="--", alpha=0.4, lw=0.8)
        if ref_line is not None:
            ax.axhline(ref_line, ls="--", color="red", alpha=0.6)
            ax.text(K * 0.98, ref_line, f"  {ref_label}={ref_line:.4f}",
                    va="bottom", ha="right", color="red", fontsize=8)

    _bar(axes[0], s_same_avg,
         "S_same — Cross-Speaker Cosine Similarity (same sentence)",
         "Avg Cosine Similarity", colors,
         ref_line=float(s_same_avg.mean()), ref_label="mean")

    _bar(axes[1], s_random,
         "S_random — Null Hypothesis Baseline (random pairs)",
         "Avg Cosine Similarity", ["#e67e22"] * K,
         ref_line=float(s_random.mean()), ref_label="mean")

    delta_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in delta_s]
    _bar(axes[2], delta_s,
         "ΔS = S_same − S_random  (Relative / Corrected Cosine Similarity)",
         "ΔS", delta_colors,
         ref_line=float(delta_s.mean()), ref_label="mean")
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Query Index")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cosine_sim_per_query.png"), dpi=150)
    plt.close(fig)

    return s_same_avg, delta_s


def plot_cosine_sim_heatmap(ssame_matrix, sentence_labels, out_dir):
    """Heatmap: rows=sentences, cols=query index for S_same."""
    fig, ax = plt.subplots(
        figsize=(max(8, ssame_matrix.shape[1] * 0.18),
                 max(4, len(sentence_labels) * 0.5))
    )
    short_labels = [s[:35] + "…" if len(s) > 35 else s for s in sentence_labels]

    im = ax.imshow(ssame_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Sentence")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title("S_same — Cross-Speaker Cosine Similarity by Sentence")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cosine_sim_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_cosine_sim_by_group(
    s_same_avg: np.ndarray,
    s_random: np.ndarray,
    num_groups: int,
    queries_per_group: int,
    out_dir: str,
):
    """For ORCA: grouped bar showing S_same, S_random, ΔS per group."""
    group_ssame  = []
    group_srandom = []
    group_delta  = []
    group_labels = []

    for g in range(num_groups):
        start = g * queries_per_group
        end   = start + queries_per_group
        gs = float(s_same_avg[start:end].mean())
        gr = float(s_random[start:end].mean())
        group_ssame.append(gs)
        group_srandom.append(gr)
        group_delta.append(gs - gr)
        group_labels.append(f"Group {g + 1}")

    x = np.arange(num_groups)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(max(8, num_groups * 1.4), 5))

    # Panel 1: S_same vs S_random
    cmap = plt.cm.get_cmap("tab10", num_groups)
    bars1 = axes[0].bar(x - width / 2, group_ssame,  width, label="S_same",
                        color=[cmap(i) for i in range(num_groups)], alpha=0.85)
    bars2 = axes[0].bar(x + width / 2, group_srandom, width, label="S_random",
                        color=[cmap(i) for i in range(num_groups)],
                        alpha=0.40, hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_labels, fontsize=9)
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("S_same vs S_random per Group")
    axes[0].legend()
    for bar, val in zip(bars1, group_ssame):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, group_srandom):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: ΔS = S_same − S_random
    delta_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in group_delta]
    bars3 = axes[1].bar(x, group_delta, color=delta_colors, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.5)
    mean_delta = float(np.mean(group_delta))
    axes[1].axhline(mean_delta, ls="--", color="red", alpha=0.6)
    axes[1].text(num_groups - 0.5, mean_delta,
                 f"  mean={mean_delta:.4f}",
                 va="bottom", ha="right", color="red", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_labels, fontsize=9)
    axes[1].set_ylabel("ΔS = S_same − S_random")
    axes[1].set_title("Relative Cosine Similarity per Group")
    for bar, val in zip(bars3, group_delta):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.001 if val >= 0 else -0.003),
                     f"{val:.4f}", ha="center",
                     va="bottom" if val >= 0 else "top", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cosine_sim_by_group.png"), dpi=150)
    plt.close(fig)

    return group_ssame, group_srandom, group_delta


# ===================================================================== #
#                            Main                                       #
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
        args.dataset_name, args.dataset_split, args.num_samples,
        args.max_audio_sec, args.trim_strategy, args.seed,
    )

    # ---- 2. Load model & extract queries ----
    model, processor = load_model(args.model_id, device)
    queries = extract_query_vectors(
        model, processor, items, device, use_projected=args.use_projected,
    )

    # Detect ORCA settings
    connector_mode   = model.config.connector_mode
    num_groups       = getattr(model.config, "orca_r1_num_groups", None)
    queries_per_group = getattr(model.config, "orca_r1_queries_per_group", None)
    is_orca = connector_mode in ["orca_r1", "orca_hybrid"]

    del model, processor
    free_mem()

    N, K, D = queries.shape

    # ---- 3a. Compute S_same (cross-speaker, same sentence) ----
    print("\n📊 Computing S_same — cross-speaker cosine similarity …")
    sentence_labels, ssame_matrix, count_per_sent = compute_s_same(queries, items)

    # ---- 3b. Compute S_random (null-hypothesis baseline) ----
    print("\n📊 Computing S_random — null-hypothesis baseline …")
    s_random = compute_s_random(queries, items, args.n_random_pairs, args.seed)

    # ---- 4. Plot ----
    s_same_avg, delta_s = plot_cosine_sim_per_query(
        ssame_matrix, s_random, sentence_labels, out_dir,
        num_groups=num_groups if is_orca else None,
        queries_per_group=queries_per_group if is_orca else None,
    )
    plot_cosine_sim_heatmap(ssame_matrix, sentence_labels, out_dir)

    group_results = None
    if is_orca and num_groups and queries_per_group:
        group_ssame, group_srandom, group_delta = plot_cosine_sim_by_group(
            s_same_avg, s_random, num_groups, queries_per_group, out_dir
        )
        group_results = list(zip(group_ssame, group_srandom, group_delta))

    # ---- 5. Print summary ----
    overall_s_same   = float(np.nanmean(s_same_avg))
    overall_s_random = float(s_random.mean())
    overall_delta    = overall_s_same - overall_s_random

    print(f"\n{'='*60}")
    print(f"  Speaker Invariance Analysis — Cosine Similarity")
    print(f"{'='*60}")
    print(f"  Model            : {args.model_id}")
    print(f"  Connector        : {connector_mode}")
    print(f"  #Samples         : {N}")
    print(f"  #Sentences       : {len(sentence_labels)}")
    print(f"  K (queries)      : {K}   D (dim) : {D}")
    print(f"{'='*60}")
    print(f"\n  S_random  (null hypothesis baseline) : {overall_s_random:.6f}")
    print(f"  S_same    (cross-speaker, same text)  : {overall_s_same:.6f}")
    print(f"  ΔS = S_same − S_random               : {overall_delta:.6f}")

    print(f"\n  Per-sentence S_same:")
    for s_i, sent in enumerate(sentence_labels):
        row = ssame_matrix[s_i]
        sent_ssame = float(np.nanmean(row))
        delta_sent = sent_ssame - overall_s_random
        marker = "✅" if delta_sent > 0.05 else ("⚠️ " if delta_sent > 0 else "❌")
        print(f"    \"{sent[:40]}\"  "
              f"(n={count_per_sent[s_i]:>4})  "
              f"S_same={sent_ssame:.4f}  "
              f"ΔS={delta_sent:+.4f}  {marker}")

    print(f"\n  Overall interpretation:")
    if overall_delta > 0.05:
        print(f"  ✅ Strong speaker invariance  (ΔS={overall_delta:.4f} >> 0)")
    elif overall_delta > 0.01:
        print(f"  ⚠️  Moderate speaker invariance (ΔS={overall_delta:.4f} > 0)")
    elif overall_delta > 0:
        print(f"  ⚠️  Weak speaker invariance  (ΔS={overall_delta:.4f} ≈ 0)")
    else:
        print(f"  ❌ No speaker invariance detected (ΔS={overall_delta:.4f} ≤ 0)")

    if group_results is not None:
        print(f"\n  Per-group results (ORCA):")
        print(f"  {'Group':<10}  {'S_same':>8}  {'S_random':>8}  {'ΔS':>8}")
        print(f"  {'-'*44}")
        for g_i, (gs, gr, gd) in enumerate(group_results):
            marker = "✅" if gd > 0.05 else ("⚠️ " if gd > 0 else "❌")
            print(f"  Group {g_i+1:<5}  {gs:>8.4f}  {gr:>8.4f}  {gd:>+8.4f}  {marker}")

    print(f"{'='*60}\n")

    # ---- 6. Save report ----
    report = dict(
        config=dict(
            model_id=args.model_id,
            connector_mode=connector_mode,
            num_samples=N, K=K, D=D,
            use_projected=args.use_projected,
            seed=args.seed,
            n_random_pairs=args.n_random_pairs,
        ),
        metrics=dict(
            s_random_overall=overall_s_random,
            s_same_overall=overall_s_same,
            delta_s_overall=overall_delta,
        ),
        per_query=dict(
            s_same=s_same_avg.tolist(),
            s_random=s_random.tolist(),
            delta_s=delta_s.tolist(),
        ),
        per_sentence={
            sent: dict(
                n_speakers=count_per_sent[s_i],
                s_same_avg=float(np.nanmean(ssame_matrix[s_i])),
                s_same_per_query=ssame_matrix[s_i].tolist(),
            )
            for s_i, sent in enumerate(sentence_labels)
        },
    )
    if group_results is not None:
        report["per_group"] = [
            dict(group=g_i + 1, s_same=gs, s_random=gr, delta_s=gd)
            for g_i, (gs, gr, gd) in enumerate(group_results)
        ]

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Report saved → {report_path}")


if __name__ == "__main__":
    main()
