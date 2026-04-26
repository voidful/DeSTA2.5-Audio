#!/usr/bin/env python3
"""
Group Visualization & Interpretability
=======================================
Generates figures for Section 5.3 of the paper:

1. **UMAP Projections** (Figure: umap_groups.png)
   - Group 4 (acoustic) and Group 6 (semantic) latent spaces
   - Colored by sentence identity and speaker identity
   - Includes silhouette scores

2. **Layer Attention Heatmap** (Figure: layer_attention_weights.png)
   - Learned α_{g,l} for all groups across Whisper layers
   - Shows which encoder layers each group draws from

Usage
-----
python study/group_visualization.py \
    --model_id voidful/desta25_4b_R2_full \
    --out_dir study/output_visualization

python study/group_visualization.py \
    --model_id voidful/desta25_4b_R2_full \
    --out_dir study/output_visualization \
    --num_samples 500 \
    --groups 4 6
"""

import argparse
import gc
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datasets import load_dataset


# =====================================================================
#                         Configuration
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Group Visualization & Interpretability")
    p.add_argument("--model_id", type=str, required=True,
                   help="Path or HF ID for DeSTA ORCA model")
    p.add_argument("--dataset_name", type=str, default="myleslinder/crema-d")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=0,
                   help="0 = use all samples")
    p.add_argument("--max_audio_sec", type=float, default=4.0)
    p.add_argument("--groups", type=int, nargs="+", default=[4, 6],
                   help="Group indices to visualize (default: 4 6)")
    p.add_argument("--out_dir", type=str, default="study/output_visualization")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    return p.parse_args()


# =====================================================================
#                          Utility helpers
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def trim_audio(arr, sr, max_seconds):
    max_len = int(max_seconds * sr)
    if arr.shape[0] <= max_len:
        return arr.astype(np.float32)
    return arr[-max_len:].astype(np.float32)


# =====================================================================
#                      1. Load CREMA-D Data
# =====================================================================

def load_cremad(dataset_name, dataset_split, num_samples, max_audio_sec, seed):
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

        # Handle both dict and AudioDecoder (torchcodec) formats
        if isinstance(audio_obj, dict):
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate", 16000))
        else:
            arr = np.asarray(
                audio_obj["array"] if hasattr(audio_obj, '__getitem__') else audio_obj.array,
                dtype=np.float32
            )
            sr = int(getattr(audio_obj, "sampling_rate", 16000))

        arr = trim_audio(arr, sr, max_audio_sec)

        text = (ex.get("text") or ex.get("sentence")
                or ex.get("transcription") or str(ex.get("sentence_id", "")))
        speaker = str(ex.get("actor_id") or ex.get("speaker_id")
                      or ex.get("speaker") or "unknown")

        items.append(dict(audio=arr, sr=sr, text=str(text), speaker=speaker))

    texts = set(i["text"] for i in items)
    speakers = set(i["speaker"] for i in items)
    print(f"  Loaded {len(items)} samples | "
          f"unique_texts={len(texts)} | speakers={len(speakers)}")
    return items


# =====================================================================
#      2. Load Model & Extract Per-Group Query Vectors
# =====================================================================

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


@torch.inference_mode()
def extract_per_group_vectors(model, processor, items, device):
    """
    Extract per-group query vectors for each sample.
    
    Returns:
        group_vectors: dict {group_idx: np.ndarray [N, queries_per_group, D]}
    """
    perception = model.perception
    connector = perception.connector
    whisper_enc = perception.whisper.model.encoder

    num_groups = connector.num_groups
    queries_per_group = connector.queries_per_group

    print(f"🎧 Extracting per-group vectors from {len(items)} samples "
          f"({num_groups} groups × {queries_per_group} queries) ...")

    # Accumulators per group: list of [queries_per_group, D] arrays
    per_group_all = {g: [] for g in range(num_groups)}

    for i, item in enumerate(items):
        audio, sr = item["audio"], item["sr"]
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(device)

        target_dtype = whisper_enc.conv1.weight.dtype
        target_dev = whisper_enc.conv1.weight.device
        feats = feats.to(dtype=target_dtype, device=target_dev)

        # Run Whisper encoder
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

        # Run connector forward to get global_tokens
        conn_out = connector(all_layer_outputs)
        global_tokens = conn_out[0] if isinstance(conn_out, tuple) else conn_out
        # global_tokens: [1, num_groups * queries_per_group, D]

        for g in range(num_groups):
            start = g * queries_per_group
            end = start + queries_per_group
            group_vecs = global_tokens[0, start:end, :].float().cpu().numpy()
            per_group_all[g].append(group_vecs)

        if (i + 1) % 100 == 0:
            free_mem()
            print(f"   [{i+1}/{len(items)}]")

    free_mem()

    # Stack into arrays
    group_vectors = {}
    for g in range(num_groups):
        group_vectors[g] = np.stack(per_group_all[g], axis=0)  # [N, K, D]
        print(f"   Group {g}: shape {group_vectors[g].shape}")

    return group_vectors


# =====================================================================
#                     3. UMAP + Silhouette
# =====================================================================

def compute_group_means(group_vectors):
    """Compute per-sample mean vector for each group: [N, D]"""
    return {g: vecs.mean(axis=1) for g, vecs in group_vectors.items()}


def run_umap(vectors, n_neighbors=15, min_dist=0.1, seed=42):
    """Run UMAP on [N, D] vectors → [N, 2]."""
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=seed, metric="cosine"
    )
    return reducer.fit_transform(vectors)


def compute_silhouette(embeddings_2d, labels):
    """Compute silhouette score. Returns float or NaN if <2 clusters."""
    from sklearn.metrics import silhouette_score
    unique = set(labels)
    if len(unique) < 2:
        return float("nan")
    return float(silhouette_score(embeddings_2d, labels))


def plot_umap_grid(group_means, items, groups_to_plot, out_dir, args):
    """
    Create a 2×2 grid:
      Row 0: Group with higher group index (semantic, e.g., Group 6)
      Row 1: Group with lower group index (acoustic, e.g., Group 4)
      Col 0: Colored by sentence
      Col 1: Colored by speaker
    """
    # Extract labels
    texts = [it["text"] for it in items]
    speakers = [it["speaker"] for it in items]

    # Create label→int mappings
    unique_texts = sorted(set(texts))
    unique_speakers = sorted(set(speakers))
    text_to_id = {t: i for i, t in enumerate(unique_texts)}
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}

    text_ids = np.array([text_to_id[t] for t in texts])
    speaker_ids = np.array([speaker_to_id[s] for s in speakers])

    # Sort groups: put semantic (higher index) on top
    g_sorted = sorted(groups_to_plot, reverse=True)

    # Run UMAP for each group
    umap_results = {}
    for g in g_sorted:
        print(f"  Running UMAP on Group {g} ({group_means[g].shape[0]} samples) ...")
        umap_results[g] = run_umap(
            group_means[g],
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed
        )

    # Compute silhouette scores
    silhouette = {}
    for g in g_sorted:
        emb = umap_results[g]
        silhouette[(g, "sentence")] = compute_silhouette(emb, text_ids)
        silhouette[(g, "speaker")] = compute_silhouette(emb, speaker_ids)
        print(f"  Group {g} silhouette: "
              f"by_sentence={silhouette[(g, 'sentence')]:.4f}, "
              f"by_speaker={silhouette[(g, 'speaker')]:.4f}")

    # === Plot 2×2 grid ===
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.25)

    group_labels = {
        g_sorted[0]: f"Group {g_sorted[0]} (Semantic)" if len(g_sorted) > 1 else f"Group {g_sorted[0]}",
        g_sorted[1]: f"Group {g_sorted[1]} (Acoustic)" if len(g_sorted) > 1 else f"Group {g_sorted[1]}",
    } if len(g_sorted) >= 2 else {g_sorted[0]: f"Group {g_sorted[0]}"}

    for row_idx, g in enumerate(g_sorted):
        emb = umap_results[g]

        for col_idx, (label_name, label_ids, unique_labels) in enumerate([
            ("Sentence", text_ids, unique_texts),
            ("Speaker", speaker_ids, unique_speakers),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            n_classes = len(unique_labels)
            if n_classes <= 20:
                cmap = plt.cm.get_cmap("tab20", n_classes)
            else:
                cmap = plt.cm.get_cmap("hsv", n_classes)

            scatter = ax.scatter(
                emb[:, 0], emb[:, 1],
                c=label_ids,
                cmap=cmap,
                s=6, alpha=0.6, edgecolors="none",
                rasterized=True,
            )

            sil = silhouette[(g, label_name.lower())]
            ax.set_title(
                f"{group_labels.get(g, f'Group {g}')} — by {label_name}\n"
                f"(Silhouette = {sil:.3f})",
                fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("UMAP-1", fontsize=9)
            ax.set_ylabel("UMAP-2", fontsize=9)
            ax.tick_params(labelsize=7)

            # Add legend for sentence labels (if few enough)
            if label_name == "Sentence" and n_classes <= 20:
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=cmap(i), markersize=6)
                    for i in range(n_classes)
                ]
                ax.legend(handles, [t[:25] for t in unique_labels],
                          fontsize=5, loc="best", ncol=2, framealpha=0.6)

    fig.suptitle("UMAP Projections of ORCA Group Representations on CREMA-D",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(out_dir, "umap_groups.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved umap_groups.png")

    return silhouette


# =====================================================================
#                   4. Layer Attention Heatmap
# =====================================================================

def plot_layer_attention_weights(connector, out_dir):
    """
    Plot α_{g,l} = softmax(w_g)_l for all groups.
    connector.group_layer_weights: ParameterList of [K, L] tensors per group.
    connector.target_layer_ids: list of int (Whisper layer indices).
    """
    num_groups = connector.num_groups
    target_layers = connector.target_layer_ids
    num_layers = len(target_layers)

    # Collect per-group attention weights, averaged over queries within each group
    weight_matrix = np.zeros((num_groups, num_layers))
    for g in range(num_groups):
        w = connector.group_layer_weights[g].detach().float().cpu()  # [K, L]
        alpha = torch.softmax(w, dim=-1)  # [K, L]
        weight_matrix[g] = alpha.mean(dim=0).numpy()  # [L] (avg over queries)

    # === Plot heatmap ===
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(weight_matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=weight_matrix.max())

    # Annotate cells
    for g in range(num_groups):
        for l in range(num_layers):
            val = weight_matrix[g, l]
            color = "white" if val > weight_matrix.max() * 0.6 else "black"
            ax.text(l, g, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_yticks(range(num_groups))
    ax.set_yticklabels([f"Group {g}" for g in range(num_groups)], fontsize=10)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"Layer {lid}" for lid in target_layers], fontsize=10)
    ax.set_xlabel("Whisper Encoder Layer", fontsize=11)
    ax.set_ylabel("ORCA Group", fontsize=11)
    ax.set_title("Layer Attention Weights  α(g,l)  per Group", fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_attention_weights.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved layer_attention_weights.png")

    return weight_matrix


# =====================================================================
#                           Main
# =====================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"🖥️  device={device}  |  out_dir={out_dir}\n")

    # ---- 1. Load data ----
    items = load_cremad(
        args.dataset_name, args.dataset_split,
        args.num_samples, args.max_audio_sec, args.seed
    )

    # ---- 2. Load model ----
    model, processor = load_model(args.model_id, device)

    # Verify this is an ORCA model
    connector_mode = model.config.connector_mode
    if connector_mode not in ("orca_desta", "orca_r1"):
        raise ValueError(f"Model connector_mode is '{connector_mode}', "
                         f"but this script requires ORCA-DeSTA (orca_desta).")

    connector = model.perception.connector
    num_groups = connector.num_groups
    queries_per_group = connector.queries_per_group
    print(f"  Connector: {connector_mode} | "
          f"groups={num_groups} | queries_per_group={queries_per_group}\n")

    # Validate requested groups
    for g in args.groups:
        if g < 0 or g >= num_groups:
            raise ValueError(f"Group {g} out of range [0, {num_groups-1}]")

    # ---- 3. Plot layer attention weights (no data needed) ----
    print("📊 Plotting layer attention weights ...")
    weight_matrix = plot_layer_attention_weights(connector, out_dir)

    # ---- 4. Extract per-group vectors ----
    group_vectors = extract_per_group_vectors(model, processor, items, device)

    # Free model memory
    del model, processor
    free_mem()

    # ---- 5. Compute group means and UMAP ----
    group_means = compute_group_means(group_vectors)

    print(f"\n📊 UMAP Visualization for Groups {args.groups} ...")
    silhouette_scores = plot_umap_grid(
        group_means, items, args.groups, out_dir, args
    )

    # ---- 6. Print summary ----
    print(f"\n{'='*60}")
    print(f"  Visualization & Interpretability Summary")
    print(f"{'='*60}")
    print(f"  Model             : {args.model_id}")
    print(f"  #Samples          : {len(items)}")
    print(f"  Connector         : {connector_mode}")
    print(f"  Groups visualized : {args.groups}")
    print(f"{'='*60}")

    print(f"\n  Layer Attention Weights (α_{{g,l}}):")
    target_layers = [f"L{lid}" for lid in [7, 15, 23, 31]]  # Default for large-v3
    header = "  " + " ".join(f"{tl:>8}" for tl in target_layers)
    print(header)
    for g in range(num_groups):
        vals = " ".join(f"{weight_matrix[g, l]:>8.4f}" for l in range(weight_matrix.shape[1]))
        print(f"  G{g}: {vals}")

    print(f"\n  Silhouette Scores:")
    print(f"  {'Group':>12}  {'By Sentence':>14}  {'By Speaker':>14}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}")
    for g in sorted(args.groups, reverse=True):
        s_sent = silhouette_scores.get((g, "sentence"), float("nan"))
        s_spk = silhouette_scores.get((g, "speaker"), float("nan"))
        print(f"  {'Group ' + str(g):>12}  {s_sent:>14.4f}  {s_spk:>14.4f}")
    print(f"{'='*60}\n")

    # ---- 7. Save report ----
    report = {
        "config": {
            "model_id": args.model_id,
            "connector_mode": connector_mode,
            "num_samples": len(items),
            "num_groups": num_groups,
            "queries_per_group": queries_per_group,
            "groups_visualized": args.groups,
            "seed": args.seed,
        },
        "layer_attention_weights": {
            f"group_{g}": weight_matrix[g].tolist()
            for g in range(num_groups)
        },
        "silhouette_scores": {
            f"group_{g}_by_sentence": silhouette_scores.get((g, "sentence"), None)
            for g in args.groups
        } | {
            f"group_{g}_by_speaker": silhouette_scores.get((g, "speaker"), None)
            for g in args.groups
        },
    }

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Report saved → {report_path}")
    print(f"📊 Figures saved → {out_dir}/")


if __name__ == "__main__":
    main()
