#!/usr/bin/env python3
"""
Query Redundancy Analysis  (Simplified)
========================================
Visualises how the K query vectors from a Q-Former / ORCA connector
relate to each other, **overall** and **per label group** (emotion, text).

Outputs
-------
- similarity_matrix_all.png          : K×K heatmap averaged over ALL samples
- similarity_matrix_by_emotion.png   : one K×K heatmap per emotion class
- similarity_matrix_by_text.png      : one K×K heatmap per text/sentence class
- report.json                        : numeric summary

Usage
-----
python study/query_redundancy_probing.py \
    --model_id <hf_or_local_desta_model_path> \
    --out_dir study/output
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
    p = argparse.ArgumentParser(description="Query Redundancy Analysis")
    p.add_argument("--model_id", type=str, required=True,
                   help="HuggingFace model ID or local path.")
    p.add_argument("--dataset_name", type=str, default="myleslinder/crema-d")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=0,
                   help="0 = use ALL data.")
    p.add_argument("--max_audio_sec", type=float, default=4.0)
    p.add_argument("--trim_strategy", type=str, default="end",
                   choices=["head", "end", "random"])
    p.add_argument("--out_dir", type=str, default="study/output")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_projected", action="store_true",
                   help="Analyse queries *after* LLM projection (default: before).")
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


def trim_audio(arr: np.ndarray, sr: int, max_seconds: float,
               strategy: str, seed: int) -> np.ndarray:
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
                max_audio_seconds, trim_strategy, seed) -> List[Dict[str, Any]]:
    print(f"📦 Loading dataset: {dataset_name} [{dataset_split}]")
    ds = load_dataset(dataset_name, split=dataset_split)
    ds = ds.shuffle(seed=seed)
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    items: List[Dict[str, Any]] = []
    for ex in ds:
        audio_obj = ex.get("audio")
        if audio_obj is None:
            continue
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        arr = trim_audio(arr, sr, max_audio_seconds, trim_strategy, seed)

        text = ex.get("text") or ex.get("sentence") or ex.get("transcription") or str(ex.get("sentence_id", ""))
        emotion = str(ex.get("emotion") or ex.get("label") or "unknown")

        items.append(dict(audio=arr, sr=sr, text=str(text), emotion=emotion))

    print(f"  Loaded {len(items)} samples  |  "
          f"emotions={sorted(set(i['emotion'] for i in items))}  |  "
          f"unique_texts={len(set(i['text'] for i in items))}")
    return items


# ===================================================================== #
#         2. Load Model & Extract Query Vectors                        #
# ===================================================================== #

def load_model(model_id: str, device: str):
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


@torch.inference_mode()
def extract_query_vectors(model, processor, items, device,
                          use_projected=False) -> np.ndarray:
    """Return query vectors as np.ndarray [N, K, D]."""
    perception = model.perception
    connector = perception.connector
    is_groupwise = model.config.connector_mode in ["groupwise_ortho", "orca_desta", "orca_r1"]
    if not is_groupwise:
        raise ValueError("query redundancy probing expects connector_mode=groupwise_ortho")

    K = connector.num_groups * connector.queries_per_group
    if not use_projected:
        print("Groupwise connectors fuse projection; using projected tokens.")

    all_queries: List[np.ndarray] = []
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

        all_layer_outputs = []
        for enc_layer in whisper_enc.layers:
            hidden = enc_layer(hidden, None, None)[0]
            all_layer_outputs.append(hidden)
        conn_out = connector(all_layer_outputs)
        query_vecs = conn_out[0] if isinstance(conn_out, tuple) else conn_out

        all_queries.append(query_vecs[0].float().cpu().numpy())

        if (i + 1) % 50 == 0:
            free_mem()
            print(f"   [{i+1}/{len(items)}]")

    free_mem()
    queries = np.stack(all_queries, axis=0)
    print(f"   → queries shape: {queries.shape}")
    return queries


# ===================================================================== #
#           3. Cosine Similarity Helpers                                #
# ===================================================================== #

def cosine_sim_matrix(queries: np.ndarray) -> np.ndarray:
    """
    queries : [N, K, D]
    Returns average K×K cosine-similarity matrix across N samples.
    """
    N, K, D = queries.shape
    sim_sum = np.zeros((K, K), dtype=np.float64)
    for n in range(N):
        q = queries[n]
        norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        q_norm = q / norms
        sim_sum += q_norm @ q_norm.T
    return (sim_sum / N).astype(np.float32)


def off_diag_stats(sim: np.ndarray) -> Dict:
    K = sim.shape[0]
    mask = ~np.eye(K, dtype=bool)
    od = sim[mask]
    return dict(
        mean=float(od.mean()),
        std=float(od.std()),
        min=float(od.min()),
        max=float(od.max()),
    )


# ===================================================================== #
#           4. Plotting                                                 #
# ===================================================================== #

def plot_single_heatmap(sim: np.ndarray, title: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Query Index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped_heatmaps(queries: np.ndarray, items: List[Dict],
                          label_key: str, out_dir: str) -> Dict:
    """
    For each unique value of items[label_key], compute and plot
    the K×K cosine-similarity heatmap.  Returns per-group stats.
    """
    # Group sample indices by label
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        groups[it[label_key]].append(idx)

    labels_sorted = sorted(groups.keys())
    n_groups = len(labels_sorted)
    ncols = min(4, n_groups)
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    group_stats = {}
    for g_i, label in enumerate(labels_sorted):
        idxs = groups[label]
        sub_q = queries[idxs]          # [n, K, D]
        sim = cosine_sim_matrix(sub_q)  # [K, K]
        stats = off_diag_stats(sim)
        group_stats[label] = stats

        r, c = divmod(g_i, ncols)
        ax = axes[r][c]
        im = ax.imshow(sim, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"{label_key}={label}  (μ={stats['mean']:.3f})", fontsize=10)
        ax.set_xlabel("Query")
        ax.set_ylabel("Query")

    # Hide unused axes
    for g_i in range(n_groups, nrows * ncols):
        r, c = divmod(g_i, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Query Cosine Similarity by {label_key}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"similarity_matrix_by_{label_key}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return group_stats


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
    del model, processor
    free_mem()

    N, K, D = queries.shape

    # ---- 3. Overall similarity ----
    print("\n📊 Computing cosine similarity …")
    sim_all = cosine_sim_matrix(queries)
    stats_all = off_diag_stats(sim_all)
    plot_single_heatmap(
        sim_all,
        f"All Samples  (off-diag μ={stats_all['mean']:.3f})",
        os.path.join(out_dir, "similarity_matrix_all.png"),
    )
    print(f"   ALL  →  mean_off_diag = {stats_all['mean']:.4f}")

    # ---- 4. Per-emotion similarity ----
    emo_stats = plot_grouped_heatmaps(queries, items, "emotion", out_dir)
    for lab, st in sorted(emo_stats.items()):
        print(f"   emotion={lab}  →  mean_off_diag = {st['mean']:.4f}")

    # ---- 5. Per-text similarity ----
    txt_stats = plot_grouped_heatmaps(queries, items, "text", out_dir)
    for lab, st in sorted(txt_stats.items()):
        print(f"   text={lab[:40]}  →  mean_off_diag = {st['mean']:.4f}")

    # ---- 6. Save report ----
    report = dict(
        config=dict(
            model_id=args.model_id,
            num_samples=N, K=K, D=D,
            use_projected=args.use_projected,
            seed=args.seed,
        ),
        similarity_all=stats_all,
        similarity_by_emotion=emo_stats,
        similarity_by_text=txt_stats,
    )
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved → {report_path}")

    # ---- 7. Summary ----
    print("\n" + "=" * 50)
    print("  SUMMARY: Query Similarity Analysis")
    print("=" * 50)
    print(f"  Overall off-diag cosine mean : {stats_all['mean']:.4f}")
    print(f"  #Emotion groups              : {len(emo_stats)}")
    print(f"  #Text groups                 : {len(txt_stats)}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
