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
import json
import os
import random
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

QUESTION = "What is the emotion of the speaker?"

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


def match_emotion(pred_text: str, gold_label: str) -> bool:
    """Check if the predicted text contains the gold emotion label."""
    pred_lower = pred_text.lower().strip()
    gold_lower = gold_label.lower().strip()
    return gold_lower in pred_lower


# =====================
# DeSTA Inference
# =====================

def run_desta_inference(model, wav_path, question, transcript=" "):
    """Run DeSTA model on a single audio + question.
    
    Args:
        transcript: Provide a transcription to bypass internal Whisper ASR.
                    Use the actual sentence for full-audio, or a space for blind.
    """
    audio_entry = {"audio": wav_path, "text": transcript}

    messages = [
        {
            "role": "system",
            "content": "You are an audio assistant."
        },
        {
            "role": "user",
            "content": f"<|AUDIO|>\n\n{question}\n\nChoose from: Anger, Disgust, Fear, Happy, Neutral, Sad.\nAnswer with just the emotion label.",
            "audios": [audio_entry]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            max_new_tokens=64,
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

    for idx in tqdm(range(total), desc=f"Emotion-{condition}"):
        item = ds[idx]

        audio_obj = item["audio"]
        audio_array, sample_rate = _extract_audio_array_and_sr(audio_obj)

        label_int = item["label"]
        gold = EMOTION_LABELS.get(label_int, str(label_int))

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
            transcript = " "  # no real transcript for noise

        try:
            pred = run_desta_inference(desta_model, wav_path, QUESTION, transcript=transcript)
        except Exception as e:
            if idx == 0:  # Print full traceback only for first error
                traceback.print_exc()
            print(f"Error on item {idx}: {e}")
            pred = "ERROR"

        correct = match_emotion(pred, gold)
        if correct:
            num_correct += 1

        result_item = {
            "idx": idx,
            "condition": condition,
            "gold": gold,
            "pred": pred,
            "correct": correct,
        }
        results.append(result_item)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    accuracy = num_correct / total if total > 0 else 0.0
    print(f"  [{condition}]: {num_correct}/{total} = {accuracy:.4f}")

    return {
        "condition": condition,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total": total,
        "results_path": out_path,
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
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}\n")

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
    )
    stats_blind = evaluate_condition(
        desta_model=desta_model,
        ds=ds,
        condition="blind",
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
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
