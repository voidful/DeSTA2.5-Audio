"""
ORCA-DeSTA Ablation Diagnostics — Three Connector-Level Diagnostics

Implements the three diagnostic experiments from Section 3 of the paper:
  1. Query Redundancy Analysis (§3.1)
  2. Acoustic Information Loss / Cross-Speaker Variance (§3.2)
  3. Acoustic Preference Violation (§3.3)

Usage:
    python scripts/ablation_diagnostics.py \
        --model_id /path/to/checkpoint \
        --output_dir /path/to/output \
        --label baseline
"""

import os
import json
import argparse
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CREMA-D dataset for factorial analysis
CREMAD_DATASET = "CaReSi/CremaD"
CREMAD_SPLIT = "test"
MAX_SAMPLES = None  # Use all samples (7442 for full CREMA-D)

# Emotion recognition prompt (same as §3.3)
EMOTION_PROMPT = "What is the emotion of the speaker? Describe the audio."
EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def load_model(model_id: str):
    """Load DeSTA model from checkpoint or HF hub."""
    from desta import DeSTA25AudioModel
    model = DeSTA25AudioModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def get_cremad_dataset(max_samples=MAX_SAMPLES):
    """Load CREMA-D dataset."""
    ds = load_dataset(CREMAD_DATASET, split=CREMAD_SPLIT)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def extract_audio_features(model, audio_array, sr, device):
    """Extract mel features from raw audio."""
    if not hasattr(model, 'processor'):
        model._setup_generation()
    features = model.processor(
        [audio_array], sampling_rate=sr, return_tensors="pt"
    ).input_features
    return features.to(device)


@torch.inference_mode()
def extract_connector_queries(model, input_features, device):
    """
    Run audio through Whisper encoder + connector and return
    the connector output tokens (before LLM projection for diagnostics).

    Returns:
        tokens: [1, M, D] connector output in LLM space
    """
    perception = model.perception
    connector = perception.connector

    # Get Whisper encoder hidden states
    whisper_enc = perception.whisper.model.encoder
    target_dtype = connector.proj[1].weight.dtype
    target_device = connector.proj[1].weight.device
    feats = input_features.to(dtype=target_dtype, device=target_device)

    inputs_embeds = torch.nn.functional.gelu(whisper_enc.conv1(feats))
    inputs_embeds = torch.nn.functional.gelu(whisper_enc.conv2(inputs_embeds))
    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    embed_pos = whisper_enc.embed_positions.weight[
        :whisper_enc.config.max_source_positions, :
    ]
    embed_pos = embed_pos.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    hidden_states = inputs_embeds + embed_pos

    all_layer_outputs = []
    for encoder_layer in whisper_enc.layers:
        layer_out = perception._forward_encoder_layer(
            encoder_layer, hidden_states, attention_mask=None
        )
        hidden_states = layer_out[0]
        all_layer_outputs.append(hidden_states)

    # Run connector
    from desta.models.modeling_desta25 import _is_orca_desta_mode
    if _is_orca_desta_mode(model.config.connector_mode):
        tokens, losses = connector(all_layer_outputs)
    else:
        tokens = connector(all_layer_outputs)

    return tokens  # [1, M, D]


def get_audio_array(item):
    """Robustly extract audio array + sr from dataset item."""
    audio = item.get("audio", None)
    if audio is None:
        return np.zeros(16000, dtype=np.float32), 16000
    if isinstance(audio, dict):
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio.get("sampling_rate", 16000)
    elif hasattr(audio, '__getitem__'):
        arr = np.asarray(audio["array"], dtype=np.float32)
        try:
            sr = int(audio["sampling_rate"])
        except Exception:
            sr = 16000
    else:
        arr = np.zeros(16000, dtype=np.float32)
        sr = 16000
    return arr, sr


# =====================================================================
# Diagnostic 1: Query Redundancy Analysis (§3.1)
# =====================================================================

def diagnostic_query_redundancy(model, ds, device, max_samples=None):
    """
    Compute mean off-diagonal cosine similarity among K connector queries.
    A value near 1.0 = severe redundancy. Near 0.0 = orthogonal queries.
    """
    logger.info("Running Diagnostic 1: Query Redundancy Analysis...")
    n = min(max_samples, len(ds)) if max_samples else len(ds)

    all_offdiag_sims = []
    emotion_sims = defaultdict(list)
    sentence_sims = defaultdict(list)

    for i in tqdm(range(n), desc="D1: Query Redundancy"):
        item = ds[i]
        audio_arr, sr = get_audio_array(item)
        if len(audio_arr) < 100:
            continue

        features = extract_audio_features(model, audio_arr, sr, device)
        tokens = extract_connector_queries(model, features, device)
        tokens_np = tokens[0].cpu().float().numpy()  # [M, D]

        M, D = tokens_np.shape
        norms = np.linalg.norm(tokens_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        tokens_normed = tokens_np / norms
        gram = tokens_normed @ tokens_normed.T  # [M, M]
        mask = ~np.eye(M, dtype=bool)
        offdiag_sim = float(gram[mask].mean())

        all_offdiag_sims.append(offdiag_sim)

        # Per-condition breakdown
        emotion = item.get("label", item.get("emotion", "unknown"))
        sentence = item.get("sentence", item.get("text", "unknown"))
        if isinstance(emotion, int):
            emotion = EMOTION_LABELS[emotion] if emotion < len(EMOTION_LABELS) else str(emotion)
        emotion_sims[emotion].append(offdiag_sim)
        sentence_sims[sentence].append(offdiag_sim)

    result = {
        "overall_mean": float(np.mean(all_offdiag_sims)) if all_offdiag_sims else 0.0,
        "overall_std": float(np.std(all_offdiag_sims)) if all_offdiag_sims else 0.0,
        "n_samples": len(all_offdiag_sims),
        "per_emotion": {
            k: {"mean": float(np.mean(v)), "count": len(v)}
            for k, v in emotion_sims.items()
        },
        "emotion_range": [
            float(min(np.mean(v) for v in emotion_sims.values())) if emotion_sims else 0,
            float(max(np.mean(v) for v in emotion_sims.values())) if emotion_sims else 0,
        ],
    }

    logger.info(f"  Query cosine sim: {result['overall_mean']:.4f} ± {result['overall_std']:.4f}")
    return result


# =====================================================================
# Diagnostic 2: Acoustic Information Loss (§3.2)
# =====================================================================

def diagnostic_acoustic_collapse(model, ds, device, max_samples=None, n_random_pairs=2000):
    """
    Compute cross-speaker discriminative margin.
    S_random: cosine sim of random audio pairs (baseline anisotropy).
    S_same:   cosine sim of same text, different speakers.
    ΔS = S_same - S_random (near 0 = collapsed).
    """
    logger.info("Running Diagnostic 2: Acoustic Information Loss...")
    n = min(max_samples, len(ds)) if max_samples else len(ds)

    # First pass: extract all query vectors + metadata
    all_tokens = []  # [N, D] (mean-pooled across M queries)
    all_sentences = []
    all_speakers = []

    for i in tqdm(range(n), desc="D2: Extract tokens"):
        item = ds[i]
        audio_arr, sr = get_audio_array(item)
        if len(audio_arr) < 100:
            continue

        features = extract_audio_features(model, audio_arr, sr, device)
        tokens = extract_connector_queries(model, features, device)
        # Mean pool across queries for per-sample representation
        mean_token = tokens[0].mean(dim=0).cpu().float().numpy()  # [D]
        all_tokens.append(mean_token)

        sentence = item.get("sentence", item.get("text", f"sent_{i}"))
        speaker = item.get("speaker_id", item.get("speaker", f"spk_{i}"))
        all_sentences.append(str(sentence))
        all_speakers.append(str(speaker))

    all_tokens = np.stack(all_tokens)  # [N, D]
    N = len(all_tokens)

    # Normalize
    norms = np.linalg.norm(all_tokens, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    all_tokens_normed = all_tokens / norms

    # S_random: random pairs
    rng = np.random.default_rng(42)
    random_sims = []
    for _ in range(n_random_pairs):
        i, j = rng.choice(N, size=2, replace=False)
        sim = float(np.dot(all_tokens_normed[i], all_tokens_normed[j]))
        random_sims.append(sim)
    S_random = float(np.mean(random_sims))

    # S_same: same text, different speakers
    sentence_to_indices = defaultdict(list)
    for idx, sent in enumerate(all_sentences):
        sentence_to_indices[sent].append(idx)

    same_text_sims = []
    for sent, indices in sentence_to_indices.items():
        if len(indices) < 2:
            continue
        # Compare pairs with same text but different speakers
        for ii in range(len(indices)):
            for jj in range(ii + 1, min(len(indices), ii + 10)):  # Cap pairs
                idx_a, idx_b = indices[ii], indices[jj]
                if all_speakers[idx_a] != all_speakers[idx_b]:
                    sim = float(np.dot(
                        all_tokens_normed[idx_a],
                        all_tokens_normed[idx_b]
                    ))
                    same_text_sims.append(sim)

    S_same = float(np.mean(same_text_sims)) if same_text_sims else S_random
    delta_S = S_same - S_random

    # Cross-speaker variance (alternative metric)
    # For same sentence, compute variance of representations across speakers
    cross_speaker_vars = []
    for sent, indices in sentence_to_indices.items():
        if len(indices) >= 3:
            sent_tokens = all_tokens[indices]
            var = float(np.var(sent_tokens, axis=0).mean())
            cross_speaker_vars.append(var)

    cross_speaker_var = float(np.mean(cross_speaker_vars)) if cross_speaker_vars else 0.0

    result = {
        "S_random": S_random,
        "S_same": S_same,
        "delta_S": delta_S,
        "cross_speaker_variance": cross_speaker_var,
        "n_random_pairs": len(random_sims),
        "n_same_text_pairs": len(same_text_sims),
        "n_samples": N,
    }

    logger.info(f"  S_random={S_random:.4f}, S_same={S_same:.4f}, ΔS={delta_S:.4f}")
    logger.info(f"  Cross-speaker variance: {cross_speaker_var:.6f}")
    return result


# =====================================================================
# Diagnostic 3: Acoustic Preference Violation (§3.3)
# =====================================================================

def diagnostic_apv(model, ds, device, max_samples=None):
    """
    Compute Acoustic Preference Violation:
      Δ_ACP = (1/|r|) * [log p(r|x,c) - log p(r|0,c)]

    VR = P[Δ_ACP ≤ 0] (violation rate)
    """
    logger.info("Running Diagnostic 3: Acoustic Preference Violation...")
    n = min(max_samples, len(ds)) if max_samples else len(ds)

    import wave
    import tempfile

    if not hasattr(model, 'tokenizer'):
        model._setup_generation()

    margins = []
    emotion_margins = defaultdict(list)

    for i in tqdm(range(n), desc="D3: APV"):
        item = ds[i]
        audio_arr, sr = get_audio_array(item)
        if len(audio_arr) < 100:
            continue

        # Get emotion label
        emotion = item.get("label", item.get("emotion", "unknown"))
        if isinstance(emotion, int):
            emotion = EMOTION_LABELS[emotion] if emotion < len(EMOTION_LABELS) else str(emotion)

        # Build target response
        target_response = f"The emotion of the speaker is {emotion}."

        # Write temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_int16 = (np.clip(audio_arr, -1, 1) * 32767).astype(np.int16)
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(audio_int16.tobytes())

        try:
            # Build messages
            messages = [
                {"role": "system", "content": "You are an audio assistant."},
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n\n{EMOTION_PROMPT}",
                    "audios": [{"audio": tmp_path}]
                }
            ]

            # Prepare full-audio inputs
            prepared = model._prepare_chat_inputs(messages)
            input_ids = prepared["context_input_ids"]
            attention_mask = prepared["context_attention_mask"]
            batch_features = prepared["batch_features"]
            batch_transcription_ids = prepared["batch_transcription_ids"]
            batch_start_positions = prepared["context_batch_start_positions"]

            # Tokenize target
            target_ids = model.tokenizer.encode(
                target_response, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            target_len = target_ids.shape[1]

            # Full input with target appended
            full_ids = torch.cat([input_ids, target_ids], dim=1)
            full_mask = torch.cat([
                attention_mask,
                torch.ones(1, target_len, device=device, dtype=attention_mask.dtype)
            ], dim=1)

            # Labels: -100 for context, target_ids for response
            labels = torch.full_like(full_ids, -100)
            labels[0, -target_len:] = target_ids[0]

            # Forward with full audio
            outputs_full = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                batch_features=batch_features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
                labels=labels,
            )
            log_prob_full = -outputs_full.loss.item()  # NLL -> log prob

            # Forward with zero audio (replace audio embeddings)
            # Re-run but zero out the audio features
            zero_features = torch.zeros_like(batch_features)
            outputs_blind = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                batch_features=zero_features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
                labels=labels,
            )
            log_prob_blind = -outputs_blind.loss.item()

            margin = log_prob_full - log_prob_blind
            margins.append(margin)
            emotion_margins[emotion].append(margin)

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Compute APV metrics
    margins_arr = np.array(margins) if margins else np.array([0.0])
    violation_rate = float(np.mean(margins_arr <= 0)) * 100

    per_emotion = {}
    for emo, emo_margins in emotion_margins.items():
        arr = np.array(emo_margins)
        per_emotion[emo] = {
            "mean_margin": float(arr.mean()),
            "violation_rate": float(np.mean(arr <= 0)) * 100,
            "count": len(arr),
        }

    result = {
        "mean_margin": float(margins_arr.mean()),
        "median_margin": float(np.median(margins_arr)),
        "violation_rate": violation_rate,
        "n_samples": len(margins),
        "per_emotion": per_emotion,
    }

    logger.info(f"  APV mean Δ: {result['mean_margin']:+.4f}")
    logger.info(f"  APV VR: {result['violation_rate']:.2f}%")
    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run connector-level diagnostics for ablation study"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF model ID or local checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./diagnostic_results",
                        help="Output directory for diagnostic results")
    parser.add_argument("--label", type=str, default="model",
                        help="Experiment label for reporting")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for each diagnostic (None = all)")
    parser.add_argument("--skip_apv", action="store_true",
                        help="Skip APV diagnostic (slow due to double forward)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model_id}")
    model, device = load_model(args.model_id)

    # Load CREMA-D
    logger.info("Loading CREMA-D dataset...")
    ds = get_cremad_dataset(args.max_samples)
    logger.info(f"  {len(ds)} samples loaded")

    results = {"label": args.label, "model_id": args.model_id}

    # Diagnostic 1: Query Redundancy
    d1 = diagnostic_query_redundancy(model, ds, device, args.max_samples)
    results["query_redundancy"] = d1
    results["query_cosine_sim"] = d1["overall_mean"]

    # Diagnostic 2: Acoustic Information Loss
    d2 = diagnostic_acoustic_collapse(model, ds, device, args.max_samples)
    results["acoustic_collapse"] = d2
    results["cross_speaker_var"] = d2["cross_speaker_variance"]

    # Diagnostic 3: Acoustic Preference Violation
    if not args.skip_apv:
        d3 = diagnostic_apv(model, ds, device, args.max_samples)
        results["apv"] = d3
        results["apv_mean_margin"] = d3["mean_margin"]
        results["apv_violation_rate"] = d3["violation_rate"]
    else:
        logger.info("Skipping APV diagnostic (--skip_apv)")

    # Save results
    out_path = os.path.join(args.output_dir, "diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Diagnostic Summary: {args.label}")
    print("=" * 60)
    print(f"  D1 Query Cosine Sim:    {results['query_cosine_sim']:.4f}")
    print(f"  D2 Cross-Speaker Var:   {results['cross_speaker_var']:.6f}")
    if 'apv_mean_margin' in results:
        print(f"  D3 APV Mean Margin:     {results['apv_mean_margin']:+.4f}")
        print(f"  D3 APV Violation Rate:  {results['apv_violation_rate']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
