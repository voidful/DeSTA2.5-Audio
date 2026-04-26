#!/usr/bin/env python3
"""
Mean-Variance Information Partitioning
=======================================
Probes whether ORCA's variational connector learns disentangled μ / log σ:
  H1. μ → semantic (sentence ID)
  H2. log σ → acoustic variation (speaker, emotion, pitch)
  H3. Controlled: same PCA dim, normalisation, classifier, split.

Three experiments:
  1. Linear probing  (sentence, emotion, pitch, speaker-verification)
  2. Matched-pair similarity (semantic / speaker / emotion pairs)
  3. Ablation control (learned σ  vs  global σ₀  vs  shuffled σ)

Dataset: CREMA-D  (12 sentences × 6 emotions × ~91 speakers)

Usage
-----
python study/mean_variance_probe.py \
    --model_id <hf_or_local> \
    --num_samples 0 \
    --out_dir study/output_mv_probe
"""

import argparse, gc, json, os, random, re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────────────────
DATASET_ID = "myleslinder/crema-d"
SPLIT = "train"
EMOTION_CODE = {"ANG":"anger","DIS":"disgust","FEA":"fear",
                "HAP":"happy","NEU":"neutral","SAD":"sad"}
EMOTION_LABELS = {0:"anger",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad"}
SEED = 42
PCA_DIM = 128          # same for every representation
N_VERIFY_PAIRS = 4000  # positive + negative speaker-verification pairs
device = "cuda" if torch.cuda.is_available() else "cpu"
VARIATIONAL_KEY_MARKERS = (
    "mu_proj",
    "logvar_proj",
    "log_var_proj",
    "logsigma_proj",
    "log_sigma_proj",
)

# ── helpers ────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def free_mem():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def trim_audio(arr, sr, max_sec=10.0):
    mx = int(max_sec * sr)
    return arr[:mx].astype(np.float32) if arr.shape[0] > mx else arr.astype(np.float32)

def _source_path(audio_obj):
    if isinstance(audio_obj, dict):
        return str(audio_obj.get("path", "") or "")
    return ""

def _speaker_from_path(audio_obj):
    p = _source_path(audio_obj)
    m = re.match(r"(\d{4})_", os.path.basename(str(p)))
    return m.group(1) if m else None

def _emotion_from_example(ex, audio_obj):
    p = _source_path(audio_obj)
    m = re.search(r"_(ANG|DIS|FEA|HAP|NEU|SAD)_", os.path.basename(str(p)).upper())
    if m:
        return EMOTION_CODE[m.group(1)]
    emo = ex.get("emotion")
    if emo:
        emo = str(emo).strip().lower()
        return EMOTION_CODE.get(emo.upper(), emo)
    label = ex.get("label")
    try:
        return EMOTION_LABELS[int(label)]
    except (TypeError, ValueError, KeyError):
        return str(label or "unknown")

def extract_pitch(arr, sr):
    """Mean F0 via autocorrelation (no librosa.pyin dependency)."""
    try:
        import librosa
        f0, _, _ = librosa.pyin(arr, fmin=50, fmax=500, sr=sr)
        valid = f0[np.isfinite(f0)]
        return float(np.mean(valid)) if valid.size else 0.0
    except Exception:
        return 0.0

# ── data loading ───────────────────────────────────────────────────────
def load_cremad(num_samples, seed):
    print(f"📦 Loading {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split=SPLIT).shuffle(seed=seed)
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    items = []
    for ex in ds:
        ao = ex.get("audio")
        if ao is None: continue
        arr = np.asarray(ao["array"], dtype=np.float32)
        sr  = int(ao["sampling_rate"])
        arr = trim_audio(arr, sr)
        sentence = ex.get("sentence") or ex.get("text") or ""
        emotion  = _emotion_from_example(ex, ao)
        speaker  = str(ex.get("actor_id") or ex.get("speaker_id")
                       or ex.get("speaker") or _speaker_from_path(ao) or "unk")
        items.append(dict(audio=arr, sr=sr, sentence=sentence,
                          emotion=emotion, speaker=speaker))
    print(f"  {len(items)} samples | "
          f"sentences={len(set(i['sentence'] for i in items))} | "
          f"speakers={len(set(i['speaker'] for i in items))}")
    return items

# ── model + representation extraction ──────────────────────────────────
def load_model(model_id, force_variational=False):
    from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config
    from transformers import AutoFeatureExtractor
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Pre-check: if config says variational=False but checkpoint has mu_proj weights,
    # override the config BEFORE model construction so the layers are created.
    cfg = DeSTA25Config.from_pretrained(model_id, cache_dir=os.getenv("HF_HOME"))
    if force_variational and not getattr(cfg, "variational_grouping_enabled", False):
        print("🔧 --force_variational set. Overriding variational_grouping_enabled=True before model construction.")
        cfg.variational_grouping_enabled = True
    if not getattr(cfg, "variational_grouping_enabled", False):
        # Peek at the checkpoint keys to see if variational weights exist
        import glob
        from safetensors import safe_open
        ckpt_dir = model_id
        try:
            from huggingface_hub import hf_hub_download
            ckpt_dir = os.path.dirname(
                hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    cache_dir=os.getenv("HF_HOME"),
                )
            )
        except Exception:
            pass
        st_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
        variational_keys = []
        for sf in st_files:
            with safe_open(sf, framework="pt") as f:
                variational_keys.extend(
                    k for k in f.keys()
                    if any(marker in k for marker in VARIATIONAL_KEY_MARKERS)
                )
                if variational_keys:
                    break
        if variational_keys:
            print("🔧 Config says variational_grouping_enabled=False but mu_proj weights "
                  "found in checkpoint. Overriding to True.")
            cfg.variational_grouping_enabled = True

    model = DeSTA25AudioModel.from_pretrained(model_id, config=cfg, torch_dtype=dtype)
    model.to(device).eval()
    connector = model.perception.connector
    has_variational_params = _connector_has_variational_params(connector)
    loaded_var_keys = getattr(model, "_desta_checkpoint_variational_keys", [])
    if has_variational_params and not getattr(connector, "variational_enabled", False):
        print("🔧 Connector has mu_proj/logvar_proj modules but variational_enabled=False. "
              "Enabling variational extraction for this probe.")
        connector.variational_enabled = True
        model.config.variational_grouping_enabled = True
    print(f"  Connector mode: {model.config.connector_mode}")
    print(f"  ACP / modality_dpo flag: {getattr(model.config, 'modality_dpo_enabled', False)} | "
          f"beta: {getattr(model.config, 'modality_dpo_beta', None)}")
    print(f"  ASR dropout prob: {getattr(model.config, 'asr_dropout_prob', 0.0)}")
    print(f"  Variational flag: {getattr(connector, 'variational_enabled', False)} | "
          f"params present: {has_variational_params}")
    print(f"  Variational checkpoint keys loaded: {len(loaded_var_keys)}")
    if force_variational and not loaded_var_keys:
        raise RuntimeError(
            "--force_variational created mu/logvar modules, but no matching "
            "mu_proj/logvar_proj weights were found in the checkpoint. "
            "Stopping to avoid probing randomly initialized variational heads."
        )
    enc_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    fe = AutoFeatureExtractor.from_pretrained(enc_id)
    return model, fe

def _target_layers(enc_id):
    m = {"openai/whisper-tiny":[0,1,2,3],"openai/whisper-small":[2,5,8,11],
         "openai/whisper-medium":[5,11,17,23],
         "openai/whisper-large-v3":[7,15,23,31],
         "openai/whisper-large-v3-turbo":[7,15,23,31]}
    return m[enc_id]

def _forward_whisper_encoder_layer(model, encoder_layer, hidden):
    perception = getattr(model, "perception", None)
    if perception is not None and hasattr(perception, "_forward_encoder_layer"):
        return perception._forward_encoder_layer(encoder_layer, hidden, attention_mask=None)[0]
    layer_outputs = encoder_layer(hidden, attention_mask=None)
    return layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

def _connector_has_variational_params(connector):
    return hasattr(connector, "mu_proj") and hasattr(connector, "logvar_proj")

@torch.inference_mode()
def extract_representations(model, fe, items):
    """
    Returns dict with keys 'mu', 'logvar', 'z', 'global_tokens'
    each np.ndarray [N, D] (mean-pooled over token dim).
    """
    connector = model.perception.connector
    var_ok = getattr(connector, "variational_enabled", False) or _connector_has_variational_params(connector)
    if var_ok and not getattr(connector, "variational_enabled", False):
        connector.variational_enabled = True
    if not var_ok:
        print("⚠️  variational_enabled=False and no mu_proj/logvar_proj modules were found "
              "— will extract global_tokens only.")
    enc = model.perception.whisper.model.encoder
    enc_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    is_orca = model.config.connector_mode in ("orca_desta", "orca_r1")

    all_mu, all_lv, all_z, all_gt = [], [], [], []
    for i, item in enumerate(tqdm(items, desc="Extracting representations")):
        feats = fe(item["audio"], sampling_rate=item["sr"],
                   return_tensors="pt").input_features.to(device)
        td = enc.conv1.weight.dtype
        feats = feats.to(dtype=td, device=enc.conv1.weight.device)
        h = torch.nn.functional.gelu(enc.conv1(feats))
        h = torch.nn.functional.gelu(enc.conv2(h))
        h = h.permute(0, 2, 1)
        pos = enc.embed_positions.weight[:enc.config.max_source_positions].to(
            dtype=h.dtype, device=h.device)
        hidden = h + pos
        if is_orca:
            layers_out = []
            for el in enc.layers:
                hidden = _forward_whisper_encoder_layer(model, el, hidden)
                layers_out.append(hidden)
            # get global_tokens (before variational)
            saved = connector.variational_enabled
            connector.variational_enabled = False
            gt, _ = connector(layers_out)          # [1, K, D_llm]
            connector.variational_enabled = saved
            all_gt.append(gt[0].float().cpu().mean(0).numpy())
            if var_ok:
                mu = connector.mu_proj(gt)         # [1, K, D_llm]
                lv = connector.logvar_proj(gt)     # [1, K, D_llm]
                std = torch.exp(0.5 * lv)
                eps = torch.randn_like(std)
                z = mu + std * eps
                all_mu.append(mu[0].float().cpu().mean(0).numpy())
                all_lv.append(lv[0].float().cpu().mean(0).numpy())
                all_z.append(z[0].float().cpu().mean(0).numpy())
        else:
            print("⚠️  Non-ORCA connector — extracting Q-Former output only.")
            for el in enc.layers:
                hidden = _forward_whisper_encoder_layer(model, el, hidden)
            all_gt.append(hidden[0].float().cpu().mean(0).numpy())

        if (i+1) % 200 == 0: free_mem()

    out = {"global_tokens": np.stack(all_gt)}
    if var_ok:
        out["mu"]     = np.stack(all_mu)
        out["logvar"] = np.stack(all_lv)
        out["z"]      = np.stack(all_z)
    return out

# ── Experiment 1: Linear probing ───────────────────────────────────────
def _prepare(X, pca_dim):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X)
    return X

def probe_classification(X, y, n_folds=5, pca_dim=PCA_DIM):
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    X = _prepare(X, pca_dim)
    accs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y_enc):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(X[tr], y_enc[tr])
        accs.append(accuracy_score(y_enc[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs))

def probe_regression(X, y, n_folds=5, pca_dim=PCA_DIM):
    X = _prepare(X, pca_dim)
    y = np.asarray(y, dtype=np.float32)
    r2s = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], reg.predict(X[te])))
    return float(np.mean(r2s)), float(np.std(r2s))

def speaker_verification_auc(X, speakers, n_pairs=N_VERIFY_PAIRS, pca_dim=PCA_DIM):
    X = _prepare(X, pca_dim)
    spk_to_idx = defaultdict(list)
    for i, s in enumerate(speakers):
        spk_to_idx[s].append(i)
    valid_spk = [s for s, idxs in spk_to_idx.items() if len(idxs) >= 2]
    rng = np.random.RandomState(SEED)
    pos_scores, neg_scores = [], []
    for _ in range(n_pairs // 2):
        s = rng.choice(valid_spk)
        i, j = rng.choice(spk_to_idx[s], 2, replace=False)
        cos = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-8)
        pos_scores.append(cos)
    all_idx = list(range(len(speakers)))
    for _ in range(n_pairs // 2):
        i = rng.choice(all_idx)
        j = rng.choice(all_idx)
        while speakers[j] == speakers[i]:
            j = rng.choice(all_idx)
        cos = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-8)
        neg_scores.append(cos)
    labels = [1]*len(pos_scores) + [0]*len(neg_scores)
    scores = pos_scores + neg_scores
    return float(roc_auc_score(labels, scores))

def run_linear_probing(reps, items, pitches):
    sentences = [it["sentence"] for it in items]
    emotions  = [it["emotion"]  for it in items]
    speakers  = [it["speaker"]  for it in items]
    results = {}
    for name, X in reps.items():
        print(f"\n  Probing [{name}] …")
        r = {}
        acc, std = probe_classification(X, sentences)
        r["sentence_acc"] = f"{acc:.4f}±{std:.4f}"
        print(f"    Sentence ID  acc={acc:.4f}±{std:.4f}")
        acc, std = probe_classification(X, emotions)
        r["emotion_acc"] = f"{acc:.4f}±{std:.4f}"
        print(f"    Emotion      acc={acc:.4f}±{std:.4f}")
        if pitches is not None:
            r2, std = probe_regression(X, pitches)
            r["pitch_r2"] = f"{r2:.4f}±{std:.4f}"
            print(f"    Pitch        R²={r2:.4f}±{std:.4f}")
        auc = speaker_verification_auc(X, speakers)
        r["spk_verif_auc"] = f"{auc:.4f}"
        print(f"    Speaker verif AUC={auc:.4f}")
        results[name] = r
    return results

# ── Experiment 2: Matched-pair similarity ──────────────────────────────
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def matched_pair_similarity(reps, items, max_pairs=5000):
    """Compute Sim_sent, Sim_spk, Sim_emo for each representation."""
    rng = np.random.RandomState(SEED)
    # build indices
    by_sent = defaultdict(list)
    by_spk  = defaultdict(list)
    by_emo  = defaultdict(list)
    for i, it in enumerate(items):
        by_sent[it["sentence"]].append(i)
        by_spk[it["speaker"]].append(i)
        by_emo[it["emotion"]].append(i)

    def _sample_pairs(groups, filter_fn, n):
        pairs = []
        keys = [k for k, v in groups.items() if len(v) >= 2]
        attempts = 0
        while len(pairs) < n and attempts < n * 10:
            k = rng.choice(keys)
            i, j = rng.choice(groups[k], 2, replace=False)
            if filter_fn(i, j):
                pairs.append((i, j))
            attempts += 1
        return pairs

    # Semantic pairs: same sentence, different speaker & emotion
    sem_pairs = _sample_pairs(by_sent,
        lambda i, j: items[i]["speaker"] != items[j]["speaker"], max_pairs)
    # Speaker pairs: same speaker, different sentence
    spk_pairs = _sample_pairs(by_spk,
        lambda i, j: items[i]["sentence"] != items[j]["sentence"], max_pairs)
    # Emotion pairs: same emotion, different speaker & sentence
    emo_pairs = _sample_pairs(by_emo,
        lambda i, j: (items[i]["speaker"] != items[j]["speaker"] and
                      items[i]["sentence"] != items[j]["sentence"]), max_pairs)

    results = {}
    for name, X in reps.items():
        Xp = _prepare(X, PCA_DIM)
        sim_sent = np.mean([_cos(Xp[i], Xp[j]) for i, j in sem_pairs]) if sem_pairs else 0
        sim_spk  = np.mean([_cos(Xp[i], Xp[j]) for i, j in spk_pairs]) if spk_pairs else 0
        sim_emo  = np.mean([_cos(Xp[i], Xp[j]) for i, j in emo_pairs]) if emo_pairs else 0
        results[name] = {"Sim_sent": round(sim_sent, 4),
                         "Sim_spk":  round(sim_spk, 4),
                         "Sim_emo":  round(sim_emo, 4)}
        print(f"  [{name}]  Sim_sent={sim_sent:.4f}  Sim_spk={sim_spk:.4f}  Sim_emo={sim_emo:.4f}")
    return results

# ── Experiment 3: Ablation control ─────────────────────────────────────
def run_ablation_control(reps_raw, items, pitches):
    """
    Compare three z variants:
      learned σ(x)   — original
      global σ₀      — mean logvar across all samples
      shuffled σ     — permute logvar across samples
    Re-run sentence + emotion probes to show σ(x) encodes info.
    """
    if "mu" not in reps_raw or "logvar" not in reps_raw:
        print("  ⚠️  Variational not enabled — skipping ablation.")
        return {}
    mu = reps_raw["mu"]       # [N, D]
    lv = reps_raw["logvar"]   # [N, D]
    rng = np.random.RandomState(SEED)

    # Learned σ(x): z = mu + exp(0.5*lv) * eps
    eps = rng.randn(*mu.shape).astype(np.float32)
    z_learned  = mu + np.exp(0.5 * lv) * eps

    # Global σ₀: replace per-sample logvar with global mean
    lv_global = np.mean(lv, axis=0, keepdims=True)
    z_global   = mu + np.exp(0.5 * lv_global) * eps

    # Shuffled σ: randomly permute logvar rows
    perm = rng.permutation(len(lv))
    z_shuffled = mu + np.exp(0.5 * lv[perm]) * eps

    sentences = [it["sentence"] for it in items]
    emotions  = [it["emotion"]  for it in items]
    results = {}
    for tag, z in [("learned_sigma", z_learned),
                   ("global_sigma",  z_global),
                   ("shuffled_sigma", z_shuffled)]:
        acc_s, _ = probe_classification(z, sentences)
        acc_e, _ = probe_classification(z, emotions)
        results[tag] = {"sentence_acc": round(acc_s, 4),
                        "emotion_acc":  round(acc_e, 4)}
        print(f"  [{tag}]  sent={acc_s:.4f}  emo={acc_e:.4f}")
    return results

# ── Visualisation ──────────────────────────────────────────────────────
def plot_probe_comparison(probe_results, out_dir):
    """Bar chart comparing μ vs log σ across probe tasks."""
    if "mu" not in probe_results or "logvar" not in probe_results:
        return
    tasks = ["sentence_acc", "emotion_acc", "spk_verif_auc"]
    labels = ["Sentence ID", "Emotion", "Speaker Verif."]
    mu_vals, lv_vals = [], []
    for t in tasks:
        mu_vals.append(float(probe_results["mu"].get(t, "0").split("±")[0]))
        lv_vals.append(float(probe_results["logvar"].get(t, "0").split("±")[0]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, mu_vals, 0.35, label="μ", color="#4c72b0")
    ax.bar(x + 0.18, lv_vals, 0.35, label="log σ", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy / AUC"); ax.set_ylim(0, 1)
    ax.set_title("Linear Probe: μ  vs  log σ")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 Saved probe_comparison.png")

def plot_matched_pair(mp_results, out_dir):
    if "mu" not in mp_results or "logvar" not in mp_results:
        return
    cats = ["Sim_sent", "Sim_spk", "Sim_emo"]
    labels = ["Semantic", "Speaker", "Emotion"]
    mu_vals = [mp_results["mu"][c] for c in cats]
    lv_vals = [mp_results["logvar"][c] for c in cats]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, mu_vals, 0.35, label="μ", color="#4c72b0")
    ax.bar(x + 0.18, lv_vals, 0.35, label="log σ", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Matched-Pair Similarity: μ  vs  log σ")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "matched_pair.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 Saved matched_pair.png")

# ── main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Mean-Variance Information Partitioning")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=0, help="0=all")
    p.add_argument("--out_dir", type=str, default="study/output_mv_probe")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_pitch", action="store_true")
    p.add_argument("--force_variational", action="store_true",
                   help="Force construction of mu/logvar modules even if config disables variational grouping")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data
    items = load_cremad(args.num_samples, args.seed)

    # 2. Extract pitch (optional)
    pitches = None
    if not args.skip_pitch:
        print("🎵 Extracting pitch …")
        pitches = [extract_pitch(it["audio"], it["sr"]) for it in tqdm(items, desc="Pitch")]

    # 3. Load model & extract representations
    model, fe = load_model(args.model_id, force_variational=args.force_variational)
    reps_raw = extract_representations(model, fe, items)
    del model; free_mem()
    print(f"  Representation shapes: { {k: v.shape for k, v in reps_raw.items()} }")

    # Build the dict of representations to probe
    reps = {}
    reps["global_tokens"] = reps_raw["global_tokens"]
    if "mu" in reps_raw:
        reps["mu"]     = reps_raw["mu"]
        reps["logvar"] = reps_raw["logvar"]
        reps["z"]      = reps_raw["z"]

    # ── Experiment 1 ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1: Linear Probing")
    print(f"{'='*60}")
    probe_results = run_linear_probing(reps, items, pitches)

    # ── Experiment 2 ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2: Matched-Pair Similarity")
    print(f"{'='*60}")
    mp_results = matched_pair_similarity(reps, items)

    # ── Experiment 3 ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 3: Ablation Control (σ variants)")
    print(f"{'='*60}")
    ablation_results = run_ablation_control(reps_raw, items, pitches)

    # ── Plots ──
    plot_probe_comparison(probe_results, args.out_dir)
    plot_matched_pair(mp_results, args.out_dir)

    # ── Save report ──
    report = {
        "model_id": args.model_id,
        "n_samples": len(items),
        "pca_dim": PCA_DIM,
        "probe": probe_results,
        "matched_pair": mp_results,
        "ablation": ablation_results,
    }
    rp = os.path.join(args.out_dir, "mv_probe_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary table ──
    print(f"\n{'='*65}")
    print("  MEAN-VARIANCE INFORMATION PARTITIONING — SUMMARY")
    print(f"{'='*65}")
    if "mu" in probe_results and "logvar" in probe_results:
        print(f"  {'Probe':<20s} {'μ':>14s} {'log σ':>14s}  {'Winner':>8s}")
        print(f"  {'-'*58}")
        for task, label in [("sentence_acc","Sentence ID"),
                            ("emotion_acc","Emotion"),
                            ("spk_verif_auc","Speaker Verif")]:
            mv = probe_results["mu"].get(task, "—")
            lv = probe_results["logvar"].get(task, "—")
            mv_f = float(mv.split("±")[0]) if "±" in str(mv) else float(mv)
            lv_f = float(lv.split("±")[0]) if "±" in str(lv) else float(lv)
            w = "μ" if mv_f > lv_f else "log σ"
            print(f"  {label:<20s} {mv:>14s} {lv:>14s}  {w:>8s}")
        if pitches:
            mv = probe_results["mu"].get("pitch_r2", "—")
            lv = probe_results["logvar"].get("pitch_r2", "—")
            mv_f = float(mv.split("±")[0]) if "±" in str(mv) else float(mv)
            lv_f = float(lv.split("±")[0]) if "±" in str(lv) else float(lv)
            w = "μ" if mv_f > lv_f else "log σ"
            print(f"  {'Pitch R²':<20s} {mv:>14s} {lv:>14s}  {w:>8s}")
    print(f"{'='*65}")
    print(f"  📄 Report saved to: {rp}")

if __name__ == "__main__":
    main()
