#!/usr/bin/env python3
"""
Text Dominance Analysis — "Blind Test"
======================================
Tests whether the model is **text-dominant** by comparing emotion-recognition
accuracy when real audio is provided versus when the audio is replaced with
pure noise.

Dataset : myleslinder/crema-d  (6-class emotion: Anger, Disgust, Fear, Happy, Neutral, Sad)
Question: "What is the emotion of the speaker?"

Method
------
For each sample in CREMA-D, run generation twice:
  A) **Full Audio**   : Real audio
  B) **Blind Audio**  : Pure Gaussian noise (same duration)

Metric
------
  Full Audio Accuracy  = correct / total  (condition A)
  Blind Audio Accuracy = correct / total  (condition B)
  Gap = Full - Blind

  - Healthy ALM  : Gap is large  (audio matters)
  - Text-dominant: Gap is small  (audio is ignored)

Usage
-----
python study/text_dominance_analysis.py \\
    --model_id <hf_or_local_desta_model_path> \\
    --output_dir study/output_text_dominance
"""

import argparse
from collections import Counter, defaultdict
import json
import os
import random
import re
import tempfile
import traceback
import atexit
import wave

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from desta import DeSTA25AudioModel
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_SEED = 42
DATASET_ID = "myleslinder/crema-d"
DATASET_SPLIT = "train"

EMOTION_LABELS = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
}
EMOTION_CODE_TO_LABEL = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
CANONICAL_EMOTIONS = ("anger", "disgust", "fear", "happy", "neutral", "sad")
EMOTION_ALIASES = {
    "anger": ("anger", "angry", "mad"),
    "disgust": ("disgust", "disgusted", "disgusting"),
    "fear": ("fear", "fearful", "afraid", "scared", "anxious", "anxiety"),
    "happy": ("happy", "happiness", "joy", "joyful", "cheerful", "pleased"),
    "neutral": ("neutral", "calm", "flat"),
    "sad": ("sad", "sadness", "sorrowful", "unhappy"),
}
ALIAS_TO_EMOTION = {
    alias: emotion
    for emotion, aliases in EMOTION_ALIASES.items()
    for alias in aliases
}
ALIAS_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(ALIAS_TO_EMOTION, key=len, reverse=True)
)

QUESTION = "What is the emotion of the speaker? Describe the audio."

_tmp_wav_fd_full, TMP_WAV_FULL = tempfile.mkstemp(suffix=".wav", prefix=f"blind_test_full_{os.getpid()}_")
os.close(_tmp_wav_fd_full)
_tmp_wav_fd_blind, TMP_WAV_BLIND = tempfile.mkstemp(suffix=".wav", prefix=f"blind_test_blind_{os.getpid()}_")
os.close(_tmp_wav_fd_blind)
atexit.register(lambda: os.remove(TMP_WAV_FULL) if os.path.exists(TMP_WAV_FULL) else None)
atexit.register(lambda: os.remove(TMP_WAV_BLIND) if os.path.exists(TMP_WAV_BLIND) else None)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Utilities
# =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_wav_from_array(audio_array, sample_rate, wav_path):
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())
    return wav_path


def _extract_audio_array_and_sr(audio_obj):
    if isinstance(audio_obj, dict):
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = audio_obj.get("sampling_rate", 16000)
    else:
        arr = np.asarray(
            audio_obj["array"] if hasattr(audio_obj, '__getitem__') else audio_obj.array,
            dtype=np.float32,
        )
        sr = getattr(audio_obj, "sampling_rate", 16000)
    return arr, sr


def generate_noise_like(audio_array, seed=0):
    """Generate pure Gaussian noise with the same length."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.02, len(audio_array)).astype(np.float32)
    return noise


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_emotion(text) -> str | None:
    """Map label strings and common adjective forms to the 6 CREMA-D labels."""
    if text is None:
        return None
    code = str(text).strip().upper()
    if code in EMOTION_CODE_TO_LABEL:
        return EMOTION_CODE_TO_LABEL[code]
    normalized = _normalize_text(text)
    if normalized in ALIAS_TO_EMOTION:
        return ALIAS_TO_EMOTION[normalized]
    if normalized in CANONICAL_EMOTIONS:
        return normalized
    matches = {
        ALIAS_TO_EMOTION[match.group(0)]
        for match in re.finditer(rf"\b(?:{ALIAS_PATTERN})\b", normalized)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


def parse_emotion_prediction(pred_text: str) -> str | None:
    """Extract a canonical emotion label from free-form model output."""
    text = _normalize_text(pred_text)
    if not text:
        return None

    cue_patterns = [
        rf"\b(?:emotion|answer|label|prediction)\s*(?:of the speaker\s*)?(?:is|:|-)?\s*(?:the speaker\s*)?(?:is|sounds|seems|appears)?\s*(?P<label>{ALIAS_PATTERN})\b",
        rf"\bthe speaker\s*(?:is|sounds|seems|appears)\s*(?P<label>{ALIAS_PATTERN})\b",
    ]
    for pattern in cue_patterns:
        match = re.search(pattern, text)
        if match:
            return ALIAS_TO_EMOTION[match.group("label")]

    matches = [
        (match.start(), ALIAS_TO_EMOTION[match.group(0)])
        for match in re.finditer(rf"\b(?:{ALIAS_PATTERN})\b", text)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def match_emotion(pred_text: str, gold_label: str) -> bool:
    """Check whether the parsed prediction matches the gold emotion."""
    return parse_emotion_prediction(pred_text) == canonicalize_emotion(gold_label)


def _extract_source_path(item, audio_obj) -> str:
    for key in ("path", "source_file", "file", "audio_path"):
        if item.get(key):
            return str(item[key])
    if isinstance(audio_obj, dict) and audio_obj.get("path"):
        return str(audio_obj["path"])
    return ""


def get_gold_emotion(item, audio_obj, label_names=None) -> str:
    """Prefer CREMA-D filename codes, then dataset metadata, then label id."""
    source_path = _extract_source_path(item, audio_obj)
    code_match = re.search(r"_(ANG|DIS|FEA|HAP|NEU|SAD)_", os.path.basename(source_path).upper())
    if code_match:
        return EMOTION_CODE_TO_LABEL[code_match.group(1)]

    for key in ("emotion", "emotion_label"):
        label = canonicalize_emotion(item.get(key))
        if label:
            return label

    label_int = item.get("label")
    if label_names is not None and label_int is not None:
        try:
            label = canonicalize_emotion(label_names[int(label_int)])
            if label:
                return label
        except (IndexError, TypeError, ValueError):
            pass
    return canonicalize_emotion(EMOTION_LABELS.get(label_int, str(label_int))) or str(label_int)


def get_label_names(ds):
    label_feature = getattr(ds, "features", {}).get("label") if hasattr(ds, "features") else None
    return getattr(label_feature, "names", None)


def build_user_content(question, prompt_style):
    choices = ", ".join(CANONICAL_EMOTIONS)
    if prompt_style == "label_only":
        return (
            f"<|AUDIO|>\n\n{question}\n\n"
            f"Choose exactly one emotion from: {choices}.\n"
            'Answer in this exact format: "Emotion: <label>".'
        )
    if prompt_style == "mcq":
        return (
            f"<|AUDIO|>\n\n{question}\n\n"
            f"Choose from the following options: {choices}.\n"
            'End with: "The correct answer is: <label>".'
        )
    return (
        f"<|AUDIO|>\n\n{question}\n\n"
        f"The emotion must be one of: {choices}.\n"
        'Start with "Emotion: <label>". Then briefly describe the acoustic evidence.'
    )


# =====================
# DeSTA Inference
# =====================

def run_desta_inference(model, wav_path, question, transcript=" ", prompt_style="case_study", max_new_tokens=96):
    """Run DeSTA model on a single audio + question.
    
    Args:
        transcript: Provide a transcription to bypass internal Whisper ASR.
                    Use the actual sentence for full-audio, or a space for blind.
    """
    audio_entry = {"audio": wav_path, "text": transcript}
    user_content = build_user_content(question, prompt_style)

    messages = [
        {
            "role": "system",
            "content": "Focus on the audio clips and instructions."
        },
        {
            "role": "user",
            "content": user_content,
            "audios": [audio_entry]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, str):
        pred = pred.strip()
    return pred


# =====================
# Single Condition Eval
# =====================

def evaluate_condition(
    desta_model,
    ds,
    condition,  # "full" or "blind"
    output_dir="results",
    seed=DEFAULT_SEED,
    num_samples=0,
    prompt_style="case_study",
    blind_transcript="same",
    max_new_tokens=96,
):
    """Evaluate emotion recognition accuracy under one condition."""
    out_path = os.path.join(output_dir, f"blind_test_{condition}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    total = len(ds)
    if num_samples > 0:
        total = min(num_samples, total)

    num_correct = 0
    results = []
    label_names = get_label_names(ds)
    pred_counter = Counter()
    gold_counter = Counter()
    per_label_total = Counter()
    per_label_correct = Counter()
    confusion = defaultdict(Counter)

    for idx in tqdm(range(total), desc=f"Emotion-{condition}"):
        item = ds[idx]

        audio_obj = item["audio"]
        audio_array, sample_rate = _extract_audio_array_and_sr(audio_obj)

        label_int = item["label"]
        gold = get_gold_emotion(item, audio_obj, label_names=label_names)

        # Get the sentence text to bypass internal Whisper ASR
        sentence = item.get("sentence") or item.get("text") or " "

        if condition == "full":
            write_wav_from_array(audio_array, sample_rate, TMP_WAV_FULL)
            wav_path = TMP_WAV_FULL
            transcript = sentence
        else:  # blind
            noise = generate_noise_like(audio_array, seed=seed + idx)
            write_wav_from_array(noise, sample_rate, TMP_WAV_BLIND)
            wav_path = TMP_WAV_BLIND
            transcript = sentence if blind_transcript == "same" else " "

        try:
            pred = run_desta_inference(
                desta_model,
                wav_path,
                QUESTION,
                transcript=transcript,
                prompt_style=prompt_style,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise  # Stop on first error to see full traceback

        pred_label = parse_emotion_prediction(pred)
        correct = pred_label == gold
        if correct:
            num_correct += 1
            per_label_correct[gold] += 1
        gold_counter[gold] += 1
        pred_counter[pred_label or "unparsed"] += 1
        per_label_total[gold] += 1
        confusion[gold][pred_label or "unparsed"] += 1

        result_item = {
            "idx": idx,
            "condition": condition,
            "label_int": label_int,
            "gold": gold,
            "pred": pred,
            "pred_label": pred_label,
            "correct": correct,
            "sentence": sentence,
            "source_path": _extract_source_path(item, audio_obj),
        }
        results.append(result_item)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    accuracy = num_correct / total if total > 0 else 0.0
    print(f"  [{condition}]: {num_correct}/{total} = {accuracy:.4f}")
    print(f"  [{condition}] prediction distribution: {dict(pred_counter)}")
    per_label_accuracy = {
        label: per_label_correct[label] / per_label_total[label]
        for label in CANONICAL_EMOTIONS
        if per_label_total[label] > 0
    }
    print(f"  [{condition}] per-label accuracy: {per_label_accuracy}")

    return {
        "condition": condition,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total": total,
        "results_path": out_path,
        "gold_distribution": dict(gold_counter),
        "prediction_distribution": dict(pred_counter),
        "per_label_accuracy": per_label_accuracy,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser(
        description="Text Dominance 'Blind Test' — Emotion Recognition on CREMA-D"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--output_dir", type=str, default="study/output_text_dominance",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--prompt_style", type=str, default="case_study",
                        choices=["case_study", "label_only", "mcq"],
                        help="Prompt format for emotion recognition")
    parser.add_argument("--blind_transcript", type=str, default="same",
                        choices=["same", "blank"],
                        help="Use the same transcript for blind noise, or blank it out")
    parser.add_argument("--max_new_tokens", type=int, default=96,
                        help="Maximum generated tokens per example")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}\n")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Blind transcript: {args.blind_transcript}")
    print(f"Max new tokens: {args.max_new_tokens}\n")

    # Load dataset
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=args.seed)
    print(f"  Total samples: {len(ds)}")

    # Load DeSTA model
    print(f"Loading DeSTA model: {args.model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    desta_model = DeSTA25AudioModel.from_pretrained(args.model_id, torch_dtype=dtype)
    desta_model.to(device).eval()

    # Run evaluation: full audio and blind audio
    stats_full = evaluate_condition(
        desta_model=desta_model,
        ds=ds,
        condition="full",
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        prompt_style=args.prompt_style,
        blind_transcript=args.blind_transcript,
        max_new_tokens=args.max_new_tokens,
    )
    stats_blind = evaluate_condition(
        desta_model=desta_model,
        ds=ds,
        condition="blind",
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        prompt_style=args.prompt_style,
        blind_transcript=args.blind_transcript,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute Gap and print summary
    acc_full = stats_full["accuracy"]
    acc_blind = stats_blind["accuracy"]
    gap = acc_full - acc_blind

    if gap < 0.05:
        verdict = "⚠️  TEXT DOMINANT — audio is ignored"
    elif gap < 0.15:
        verdict = "⚠️  Weak audio reliance"
    elif gap < 0.30:
        verdict = "Moderate audio reliance"
    else:
        verdict = "✅ Audio-aware"

    print(f"\n{'='*60}")
    print(f"  TEXT DOMINANCE 'BLIND TEST' SUMMARY")
    print(f"  Model: {args.model_id}")
    print(f"  Dataset: {DATASET_ID}")
    print(f"  Question: {QUESTION}")
    print(f"{'='*60}")
    print(f"  Full Audio Accuracy  : {acc_full:.4f}  ({stats_full['num_correct']}/{stats_full['total']})")
    print(f"  Blind Audio Accuracy : {acc_blind:.4f}  ({stats_blind['num_correct']}/{stats_blind['total']})")
    print(f"  Gap = Full - Blind   : {gap:.4f}")
    print(f"  Verdict              : {verdict}")
    print(f"{'='*60}")
    print(f"  Large Gap  → model relies on audio (good)")
    print(f"  Small Gap  → model ignores audio, text dominant (bad)")
    print(f"{'='*60}")

    # Save summary
    summary_report = {
        "model_id": args.model_id,
        "dataset": DATASET_ID,
        "question": QUESTION,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "blind_transcript": args.blind_transcript,
        "max_new_tokens": args.max_new_tokens,
        "full_audio_accuracy": acc_full,
        "blind_audio_accuracy": acc_blind,
        "gap": gap,
        "verdict": verdict,
        "full": stats_full,
        "blind": stats_blind,
    }

    report_path = os.path.join(args.output_dir, "blind_test_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {report_path}")


if __name__ == "__main__":
    main()
