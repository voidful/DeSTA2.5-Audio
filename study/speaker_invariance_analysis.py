#!/usr/bin/env python3
"""
Speaker Invariance Analysis
============================
For each sentence in CREMA-D, many different speakers say the same text.
If the connector learns a speaker-invariant semantic representation,
then query vectors for the same sentence (different speakers) should
cluster tightly → **low variance**.

This script:
1. Extracts query vectors from a DeSTA model.
2. Groups samples by sentence (text).
3. Computes per-query variance across speakers within each sentence group.
4. Visualises variance per query index and per sentence.

Outputs
-------
- variance_per_query.png            : bar chart of avg variance per query slot
- variance_heatmap.png              : heatmap (rows=sentence, cols=query index)
- variance_by_group.png             : (ORCA only) variance aggregated by group
- report.json                       : numeric summary

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
#                           Configuration                               #
# ===================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Speaker Invariance Analysis")
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
#         2. Load Model & Extract Query Vectors                        #
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
                hidden = enc_layer(hidden, None, None)[0]
                all_layer_outputs.append(hidden)
            conn_out = connector(all_layer_outputs)
            query_vecs = conn_out[0] if isinstance(conn_out, tuple) else conn_out
        else:
            layer_prompt_outputs = []
            for idx, enc_layer in enumerate(whisper_enc.layers):
                hidden = enc_layer(hidden, None, None)[0]
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
#     3. Variance Analysis: same text, different speakers              #
# ===================================================================== #

def compute_per_sentence_variance(queries, items):
    """
    Group by text, compute per-query-slot variance across speakers.

    Returns
    -------
    sentence_labels : list[str]        — sorted unique sentences
    var_matrix      : np.ndarray [S, K] — variance per (sentence, query)
    count_per_sent  : list[int]        — #speakers per sentence
    """
    N, K, D = queries.shape

    # Group indices by text
    groups = defaultdict(list)
    for idx, it in enumerate(items):
        groups[it["text"]].append(idx)

    sentence_labels = sorted(groups.keys())
    S = len(sentence_labels)
    var_matrix = np.zeros((S, K), dtype=np.float64)
    count_per_sent = []

    for s_i, sent in enumerate(sentence_labels):
        idxs = groups[sent]
        count_per_sent.append(len(idxs))
        sub_q = queries[idxs]  # [n_speakers, K, D]

        # Per-query variance: for each query slot k, compute
        # trace of covariance = sum of per-dimension variances
        # then normalise by D to get a per-dimension average
        for k in range(K):
            vecs = sub_q[:, k, :]  # [n_speakers, D]
            # Average per-dimension variance
            var_matrix[s_i, k] = float(vecs.var(axis=0).mean())

    return sentence_labels, var_matrix, count_per_sent


# ===================================================================== #
#                        4. Plotting                                    #
# ===================================================================== #

def plot_variance_per_query(var_matrix, sentence_labels, out_dir,
                            num_groups=None, queries_per_group=None):
    """Bar chart: average variance per query slot (averaged over sentences)."""
    K = var_matrix.shape[1]
    avg_var = var_matrix.mean(axis=0)  # [K]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.22), 5))
    colors = ["#3498db"] * K

    # If ORCA, color by group
    if num_groups and queries_per_group:
        cmap = plt.cm.get_cmap("tab10", num_groups)
        colors = [cmap(i // queries_per_group) for i in range(K)]
        # Add group boundary lines
        for g in range(1, num_groups):
            ax.axvline(g * queries_per_group - 0.5, color="gray",
                       ls="--", alpha=0.4, lw=0.8)

    ax.bar(range(K), avg_var, color=colors, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Avg Variance (per dimension)")
    ax.set_title("Per-Query Variance Across Speakers (same sentence)")
    ax.set_xlim(-0.5, K - 0.5)

    # Annotate overall mean
    mean_val = avg_var.mean()
    ax.axhline(mean_val, ls="--", color="red", alpha=0.6)
    ax.text(K * 0.98, mean_val, f"  mean={mean_val:.4f}",
            va="bottom", ha="right", color="red", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_per_query.png"), dpi=150)
    plt.close(fig)

    return avg_var


def plot_variance_heatmap(var_matrix, sentence_labels, out_dir):
    """Heatmap: rows=sentences, cols=query index."""
    fig, ax = plt.subplots(figsize=(max(8, var_matrix.shape[1] * 0.18),
                                    max(4, len(sentence_labels) * 0.5)))

    # Truncate long sentence labels for display
    short_labels = [s[:35] + "…" if len(s) > 35 else s for s in sentence_labels]

    im = ax.imshow(var_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Sentence")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title("Query Variance by Sentence (across speakers)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Variance")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_variance_by_group(var_matrix, num_groups, queries_per_group, out_dir):
    """For ORCA: grouped bar showing avg variance per group."""
    K = var_matrix.shape[1]
    avg_var = var_matrix.mean(axis=0)  # [K]

    group_vars = []
    group_labels = []
    for g in range(num_groups):
        start = g * queries_per_group
        end = start + queries_per_group
        group_vars.append(avg_var[start:end].mean())
        group_labels.append(f"Group {g}")

    fig, ax = plt.subplots(figsize=(max(6, num_groups * 0.8), 4.5))
    cmap = plt.cm.get_cmap("tab10", num_groups)
    bars = ax.bar(range(num_groups), group_vars,
                  color=[cmap(i) for i in range(num_groups)], alpha=0.85)
    ax.set_xlabel("Group Index")
    ax.set_ylabel("Avg Variance (per dimension)")
    ax.set_title("Per-Group Variance Across Speakers")
    ax.set_xticks(range(num_groups))
    ax.set_xticklabels(group_labels, fontsize=9)

    # Annotate values on bars
    for bar, val in zip(bars, group_vars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    overall_mean = np.mean(group_vars)
    ax.axhline(overall_mean, ls="--", color="red", alpha=0.6)
    ax.text(num_groups - 0.5, overall_mean, f"  mean={overall_mean:.4f}",
            va="bottom", ha="right", color="red", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_by_group.png"), dpi=150)
    plt.close(fig)

    return group_vars


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
    connector_mode = model.config.connector_mode
    num_groups = getattr(model.config, "orca_r1_num_groups", None)
    queries_per_group = getattr(model.config, "orca_r1_queries_per_group", None)
    is_orca = connector_mode in ["orca_r1", "orca_hybrid"]

    del model, processor
    free_mem()

    N, K, D = queries.shape

    # ---- 3. Compute variance ----
    print("\n📊 Computing per-sentence variance across speakers …")
    sentence_labels, var_matrix, count_per_sent = \
        compute_per_sentence_variance(queries, items)

    # ---- 4. Plot ----
    avg_var = plot_variance_per_query(
        var_matrix, sentence_labels, out_dir,
        num_groups=num_groups if is_orca else None,
        queries_per_group=queries_per_group if is_orca else None,
    )
    plot_variance_heatmap(var_matrix, sentence_labels, out_dir)

    group_vars = None
    if is_orca and num_groups and queries_per_group:
        group_vars = plot_variance_by_group(
            var_matrix, num_groups, queries_per_group, out_dir)

    # ---- 5. Print summary ----
    overall_var = float(avg_var.mean())
    print(f"\n{'='*55}")
    print(f"  Speaker Invariance Analysis")
    print(f"{'='*55}")
    print(f"  Model            : {args.model_id}")
    print(f"  Connector        : {connector_mode}")
    print(f"  #Samples         : {N}")
    print(f"  #Sentences       : {len(sentence_labels)}")
    print(f"  K (queries)      : {K}   D (dim) : {D}")
    print(f"{'='*55}")

    print(f"\n  Per-sentence results:")
    for s_i, sent in enumerate(sentence_labels):
        sent_var = float(var_matrix[s_i].mean())
        print(f"    \"{sent[:45]}\"  "
              f"(n={count_per_sent[s_i]:>4})  "
              f"avg_var = {sent_var:.6f}")

    print(f"\n  Overall avg variance : {overall_var:.6f}")
    if overall_var < 0.01:
        print(f"  ✅ Variance is very low → strong speaker invariance")
    elif overall_var < 0.05:
        print(f"  ⚠️  Moderate variance → partial speaker invariance")
    else:
        print(f"  ❌ High variance → queries encode speaker-specific info")

    if group_vars is not None:
        print(f"\n  Per-group avg variance (ORCA):")
        for g_i, gv in enumerate(group_vars):
            marker = "✅" if gv < 0.01 else ("⚠️ " if gv < 0.05 else "❌")
            print(f"    Group {g_i}: {gv:.6f}  {marker}")

    print(f"{'='*55}\n")

    # ---- 6. Save report ----
    report = dict(
        config=dict(
            model_id=args.model_id,
            connector_mode=connector_mode,
            num_samples=N, K=K, D=D,
            use_projected=args.use_projected,
            seed=args.seed,
        ),
        overall_avg_variance=overall_var,
        per_query_avg_variance=avg_var.tolist(),
        per_sentence={
            sent: dict(
                n_speakers=count_per_sent[s_i],
                avg_variance=float(var_matrix[s_i].mean()),
                per_query_variance=var_matrix[s_i].tolist(),
            )
            for s_i, sent in enumerate(sentence_labels)
        },
    )
    if group_vars is not None:
        report["per_group_avg_variance"] = group_vars

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Report saved → {report_path}")


if __name__ == "__main__":
    main()
