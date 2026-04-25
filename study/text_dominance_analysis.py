#!/usr/bin/env python3
"""
Acoustic Preference Violation (APV) Analysis
=============================================
Directly corresponds to the ACP (Acoustic Contrastive Preference) loss.

For each sample with ground-truth response r (emotion label), compute:

  Δ_ACP = (1/|r|) * [log p(r | audio, text) - log p(r | noise, text)]

Metrics
-------
  - Mean Δ_ACP : Average acoustic preference margin
  - Violation Rate (VR) : P[Δ_ACP ≤ 0]
    Fraction of samples where the model prefers the text-only prediction

Dataset : CREMA-D matched-transcript counterfactual subsets
          Same sentence text, different emotion audio
          → model MUST use audio to distinguish emotions

This is structurally isomorphic to ACP's optimisation objective:
ACP maximises ℓ_full − ℓ_blind; this experiment measures exactly that gap.

Usage
-----
python study/text_dominance_analysis.py \\
    --model_id <hf_or_local_desta_model_path> \\
    --output_dir study/output_apv
"""

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
import random
import re
import wave

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_SEED = 42
DATASET_ID = "myleslinder/crema-d"
DATASET_SPLIT = "train"

# 6-class CREMA-D emotions
EMOTION_LABELS = {0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad"}
EMOTION_CODE_TO_LABEL = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
CANONICAL_EMOTIONS = ("anger", "disgust", "fear", "happy", "neutral", "sad")

QUESTION = "What is the emotion of the speaker?"

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
#  Utility helpers
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


def trim_audio(arr, sr, max_seconds=10.0):
    """Trim audio to max_seconds to keep memory bounded."""
    max_len = int(max_seconds * sr)
    if arr.shape[0] <= max_len:
        return arr.astype(np.float32)
    return arr[:max_len].astype(np.float32)


def generate_noise_like(audio_array, seed=0):
    """Generate pure Gaussian noise with the same length and energy."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.02, len(audio_array)).astype(np.float32)
    return noise


def get_gold_emotion(item) -> str:
    """Extract canonical emotion label from CREMA-D item."""
    audio_obj = item.get("audio", {})
    # Try filename-based code first
    for key in ("path", "source_file", "file"):
        source = item.get(key) or (audio_obj.get("path") if isinstance(audio_obj, dict) else "")
        if source:
            m = re.search(r"_(ANG|DIS|FEA|HAP|NEU|SAD)_", os.path.basename(str(source)).upper())
            if m:
                return EMOTION_CODE_TO_LABEL[m.group(1)]
    # Fallback to label field
    label_int = item.get("label")
    if label_int is not None:
        label_str = EMOTION_LABELS.get(int(label_int), "")
        return label_str.lower() if label_str else str(label_int)
    return "unknown"


def get_sentence(item) -> str:
    """Extract sentence text from CREMA-D item."""
    return item.get("sentence") or item.get("text") or ""


# =====================================================================
#  Model loading (same approach as speaker_invariance_analysis.py)
# =====================================================================

def load_model(model_id):
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    from transformers import AutoFeatureExtractor

    print(f"🔄 Loading DeSTA model: {model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = DeSTA25AudioModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()

    # Load feature extractor (mel spectrogram processor)
    encoder_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    feat_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)

    # Setup tokenizer for generation context
    if not hasattr(model, "tokenizer"):
        model._setup_generation()

    return model, feat_extractor


# =====================================================================
#  Core: compute log p(target | audio, text) via teacher-forced forward
# =====================================================================

def _get_audio_token_size(model):
    """Determine how many tokens the audio connector produces."""
    cfg = model.config
    if cfg.connector_mode == "orca_r1":
        return cfg.orca_r1_num_groups * cfg.orca_r1_queries_per_group
    return cfg.prompt_size


@torch.inference_mode()
def compute_target_log_prob(model, feat_extractor, audio_array, sr,
                            target_text, transcript=" "):
    """
    Compute average per-token log p(target_text | audio, prompt).

    Instead of calling model.generate() (which triggers the buggy Whisper
    decoder), we replicate the audio → encoder → connector → LLM forward
    path and do a teacher-forced log-prob computation.

    Returns:
        avg_log_prob : float  — (1/|r|) * Σ log p(r_t | r_{<t}, audio, text)
        total_log_prob : float
        num_tokens : int — |r|
    """
    from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions

    tokenizer = model.tokenizer
    audio_locator = model.audio_locator
    placeholder_token = model.placeholder_token
    audio_token_size = _get_audio_token_size(model)

    # --- 1. Prepare mel features ---
    audio_float = audio_array.astype(np.float32)
    feats = feat_extractor(audio_float, sampling_rate=sr, return_tensors="pt")
    batch_features = feats.input_features.to(device)

    # --- 2. Prepare transcription ---
    trans_ids = tokenizer.encode(transcript, add_special_tokens=False, return_tensors="pt")
    trans_ids = trans_ids.long().to(device)
    transcription_size = trans_ids.size(1)

    # --- 3. Build chat prompt (without target) ---
    choices = ", ".join(CANONICAL_EMOTIONS)
    user_content = (
        f"<|AUDIO|>\n\n{QUESTION}\n\n"
        f"Choose exactly one emotion from: {choices}.\n"
        'Answer with just the emotion label.'
    )
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": user_content},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_str = prompt_str.replace(
        audio_locator, f"<start_audio>{audio_locator}<end_audio>"
    )

    # Expand audio placeholder tokens
    prompt_tokens, start_positions = _prepare_audio_context_and_start_positions(
        token_list=tokenizer.tokenize(prompt_str),
        audio_locator=audio_locator,
        audio_size_list=[audio_token_size],
        transcription_size_list=[transcription_size],
        placeholder_token=placeholder_token,
    )
    prompt_str_expanded = tokenizer.convert_tokens_to_string(prompt_tokens)

    # --- 4. Tokenize prompt + target together ---
    full_str = prompt_str_expanded + target_text
    prompt_input = tokenizer(prompt_str_expanded, return_tensors="pt",
                             add_special_tokens=False)
    full_input = tokenizer(full_str, return_tensors="pt",
                           add_special_tokens=False)

    context_len = prompt_input["input_ids"].size(1)
    input_ids = full_input["input_ids"].to(device)
    attention_mask = full_input["attention_mask"].to(device)
    num_target_tokens = input_ids.size(1) - context_len

    if num_target_tokens <= 0:
        return 0.0, 0.0, 0

    # --- 5. Prepare inputs_embeds with audio injection ---
    batch_start_positions = [(0, start_positions[0])]
    batch_transcription_ids = [trans_ids]

    prepare_result = model._prepare_inputs_for_llm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_features=batch_features,
        batch_transcription_ids=batch_transcription_ids,
        batch_start_positions=batch_start_positions,
    )

    # Handle ORCA mode
    is_orca = model.config.connector_mode == "orca_hybrid"
    if is_orca and isinstance(prepare_result, tuple) and len(prepare_result) >= 3:
        inputs_embeds = prepare_result[0]
        # Set deep injection tokens if needed
        if len(prepare_result) == 4:
            _, global_tok, local_tok, trans_pos = prepare_result
        else:
            _, global_tok, local_tok = prepare_result
            trans_pos = None
        model._orca_transcription_positions = trans_pos
        if getattr(model.config, 'orca_deep_injection_enabled', True):
            if getattr(model.config, 'orca_global_cross_attn', False):
                if local_tok is not None and global_tok is not None:
                    model._orca_audio_local = torch.cat([global_tok, local_tok], dim=1)
                elif global_tok is not None:
                    model._orca_audio_local = global_tok
                else:
                    model._orca_audio_local = local_tok
            else:
                model._orca_audio_local = local_tok
        else:
            model._orca_audio_local = None
        model._orca_audio_local_mask = None
    elif isinstance(prepare_result, tuple):
        inputs_embeds = prepare_result[0]
    else:
        inputs_embeds = prepare_result

    # --- 6. LLM forward pass (teacher-forced) ---
    try:
        outputs = model.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    finally:
        if hasattr(model, '_orca_audio_local'):
            model._orca_audio_local = None
            model._orca_audio_local_mask = None
        if hasattr(model, '_orca_transcription_positions'):
            model._orca_transcription_positions = None

    # --- 7. Extract log probs for target tokens ---
    logits = outputs.logits  # [1, seq_len, vocab_size]
    # Autoregressive: logit at position t predicts token at position t+1
    target_ids = input_ids[0, context_len:]              # [T]
    target_logits = logits[0, context_len - 1: -1]       # [T, V]

    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    target_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # [T]

    avg_lp = target_log_probs.mean().item()
    total_lp = target_log_probs.sum().item()
    return avg_lp, total_lp, num_target_tokens


# =====================================================================
#  Experiment runner
# =====================================================================

def run_apv_experiment(model, feat_extractor, ds, num_samples=0, seed=42,
                       output_dir="results"):
    """
    Run the Acoustic Preference Violation experiment.

    For each CREMA-D sample:
      1. target_text = correct emotion label (e.g. "anger")
      2. log_full  = log p(target | real_audio, text)
      3. log_blind = log p(target | noise_audio, text)
      4. Δ_ACP = log_full - log_blind   (per-token average)
      5. violation = 1 if Δ_ACP ≤ 0
    """
    total = len(ds)
    if num_samples > 0:
        total = min(num_samples, total)

    results = []
    jsonl_path = os.path.join(output_dir, "apv_results.jsonl")
    with open(jsonl_path, "w") as f:
        pass  # truncate

    for idx in tqdm(range(total), desc="APV"):
        item = ds[idx]
        audio_obj = item["audio"]
        audio_array = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        audio_array = trim_audio(audio_array, sr, max_seconds=10.0)

        gold = get_gold_emotion(item)
        sentence = get_sentence(item)
        target_text = gold  # the correct emotion label

        # Noise with same length
        noise = generate_noise_like(audio_array, seed=seed + idx)

        try:
            # Full audio condition
            avg_lp_full, total_lp_full, n_tok = compute_target_log_prob(
                model, feat_extractor, audio_array, sr,
                target_text=target_text, transcript=sentence or " ",
            )
            # Blind (noise) condition
            avg_lp_blind, total_lp_blind, _ = compute_target_log_prob(
                model, feat_extractor, noise, sr,
                target_text=target_text, transcript=sentence or " ",
            )
        except Exception as e:
            if idx < 3:
                import traceback
                traceback.print_exc()
            print(f"Error on item {idx}: {e}")
            continue

        delta_acp = avg_lp_full - avg_lp_blind
        violation = int(delta_acp <= 0)

        row = {
            "idx": idx,
            "gold": gold,
            "sentence": sentence,
            "n_tokens": n_tok,
            "avg_lp_full": round(avg_lp_full, 6),
            "avg_lp_blind": round(avg_lp_blind, 6),
            "delta_acp": round(delta_acp, 6),
            "violation": violation,
        }
        results.append(row)

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (idx + 1) % 200 == 0:
            free_mem()

    return results


# =====================================================================
#  Analysis & Reporting
# =====================================================================

def analyse_results(results, model_id, output_dir):
    """Compute aggregate metrics and per-sentence / per-emotion breakdowns."""
    if not results:
        print("No results to analyse.")
        return

    deltas = np.array([r["delta_acp"] for r in results])
    violations = np.array([r["violation"] for r in results])

    mean_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))
    std_delta = float(deltas.std())
    vr = float(violations.mean())

    # Per-emotion
    by_emotion = defaultdict(list)
    for r in results:
        by_emotion[r["gold"]].append(r["delta_acp"])

    per_emotion = {}
    for emo in CANONICAL_EMOTIONS:
        vals = by_emotion.get(emo, [])
        if vals:
            arr = np.array(vals)
            per_emotion[emo] = {
                "n": len(vals),
                "mean_delta": round(float(arr.mean()), 6),
                "vr": round(float((arr <= 0).mean()), 4),
            }

    # Per-sentence (matched-transcript analysis)
    by_sentence = defaultdict(list)
    for r in results:
        by_sentence[r["sentence"]].append(r)

    per_sentence = {}
    for sent, rows in sorted(by_sentence.items()):
        arr = np.array([r["delta_acp"] for r in rows])
        per_sentence[sent] = {
            "n": len(rows),
            "emotions": sorted(set(r["gold"] for r in rows)),
            "mean_delta": round(float(arr.mean()), 6),
            "vr": round(float((arr <= 0).mean()), 4),
        }

    # Verdict
    if vr > 0.40:
        verdict = "⚠️  HIGH VIOLATION — model often ignores audio (text-dominant)"
    elif vr > 0.20:
        verdict = "⚠️  Moderate violation — inconsistent audio usage"
    elif vr > 0.10:
        verdict = "Mild violation — mostly audio-aware"
    else:
        verdict = "✅ Low violation — strong acoustic grounding"

    # Print summary
    print(f"\n{'='*65}")
    print(f"  ACOUSTIC PREFERENCE VIOLATION (APV) SUMMARY")
    print(f"  Model: {model_id}")
    print(f"{'='*65}")
    print(f"  N samples           : {len(results)}")
    print(f"  Mean Δ_ACP          : {mean_delta:+.4f}")
    print(f"  Median Δ_ACP        : {median_delta:+.4f}")
    print(f"  Std Δ_ACP           : {std_delta:.4f}")
    print(f"  Violation Rate (VR) : {vr:.4f}  ({int(violations.sum())}/{len(results)})")
    print(f"  Verdict             : {verdict}")
    print(f"{'='*65}")
    print(f"  Δ_ACP > 0 → model prefers audio  (good)")
    print(f"  Δ_ACP ≤ 0 → model prefers text   (violation)")
    print(f"{'='*65}")

    print(f"\n  Per-emotion breakdown:")
    for emo in CANONICAL_EMOTIONS:
        info = per_emotion.get(emo)
        if info:
            print(f"    {emo:>8s}  n={info['n']:>4d}  Δ_ACP={info['mean_delta']:+.4f}  VR={info['vr']:.3f}")

    print(f"\n  Per-sentence (matched-transcript) breakdown:")
    for sent, info in per_sentence.items():
        emos = ",".join(info["emotions"])
        print(f"    \"{sent[:40]:40s}\"  n={info['n']:>4d}  Δ_ACP={info['mean_delta']:+.4f}  VR={info['vr']:.3f}  [{emos}]")

    # Save report
    report = {
        "model_id": model_id,
        "dataset": DATASET_ID,
        "n_samples": len(results),
        "mean_delta_acp": mean_delta,
        "median_delta_acp": median_delta,
        "std_delta_acp": std_delta,
        "violation_rate": vr,
        "verdict": verdict,
        "per_emotion": per_emotion,
        "per_sentence": per_sentence,
    }
    report_path = os.path.join(output_dir, "apv_summary.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Summary saved to: {report_path}")

    return report


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Acoustic Preference Violation (APV) — ACP-aligned ablation"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--output_dir", type=str, default="study/output_apv",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}\n")

    # Load dataset
    print(f"📦 Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=args.seed)
    print(f"   Total samples: {len(ds)}")

    # Load model
    model, feat_extractor = load_model(args.model_id)
    print(f"   Connector mode: {model.config.connector_mode}")
    print(f"   Audio token size: {_get_audio_token_size(model)}")

    # Run APV experiment
    results = run_apv_experiment(
        model, feat_extractor, ds,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Analyse and report
    analyse_results(results, args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
