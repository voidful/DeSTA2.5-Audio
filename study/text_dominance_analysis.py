#!/usr/bin/env python3
"""
Text Dominance Analysis — "Blind Test"
======================================
Tests whether the model is **text-dominant** by comparing performance when
real audio is provided versus when the audio is replaced with pure noise.

Method
------
For each sample, run generation twice:
  A) **Full Input**  : Real audio + ASR transcript
  B) **Blind Input**  : Pure Gaussian noise (same duration) + ASR transcript

If the model is text-dominant, Acc(Blind) stays close to Acc(Full) because it
ignores the audio and "guesses" from the transcript alone.

Metric
------
  Delta = Acc(Full) - Acc(Blind)
  - Healthy ALM  : Delta is large (audio matters)
  - Text-dominant : Delta is small (audio is ignored)

Datasets : SAKURA EmotionQA / GenderQA (classification tasks where audio matters)

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
import atexit
import wave

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from desta import DeSTA25AudioModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_SEED = 42

DATASETS = {
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    "GenderQA":  "SLLM-multi-hop/GenderQA",
}

HOP_SPLITS = ["single_", "multi_"]
DATA_SPLIT = "test"

JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

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
    elif hasattr(audio_obj, 'get_all_samples'):
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
    else:
        arr = np.asarray(audio_obj["array"] if hasattr(audio_obj, '__getitem__') else audio_obj.array, dtype=np.float32)
        try:
            sr = int(audio_obj["sampling_rate"])
        except Exception:
            sr = getattr(audio_obj, "sampling_rate", 16000)
    return arr, sr


def generate_noise_like(audio_array, sample_rate, seed=0):
    """Generate pure Gaussian noise with the same duration and sample rate."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.02, len(audio_array)).astype(np.float32)
    return noise


# =====================
# DeSTA Inference
# =====================

def run_desta_on_item(model, item, hop_prefix, wav_path, transcript=None):
    instruction_key = f"{hop_prefix}instruction"

    audio_entry = {"audio": wav_path}
    if transcript is not None:
        audio_entry["text"] = transcript

    messages = [
        {
            "role": "system",
            "content": "You are an audio assistant."
        },
        {
            "role": "user",
            "content": f"<|AUDIO|>\n\nQuestion: {item[instruction_key]}\n\nInstructions:\nListen to the audio and select the correct option from the list.\n\nFormat:\nReasoning: <Brief thoughts>\nAnswer: (x) label",
            "audios": [audio_entry]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            max_new_tokens=512
        )

    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, str):
        pred = pred.strip()
    return pred


# =====================
# Qwen Judge
# =====================

BINARY_PROMPT_TEMPLATE = """You are a strict expert judge for an audio question answering task.

You receive:
1. A question about an audio clip.
2. The ground truth answer.
3. The model's predicted answer.

Decide if the model's answer is semantically correct.
Ignore small wording differences, punctuation, and synonyms.
Focus only on meaning.

Question: {question}
Ground truth answer: {gold}
Model answer: {pred}

If the model's answer is semantically correct or equivalent, output exactly:
CORRECT

Otherwise, output exactly:
INCORRECT
"""


def load_qwen_judge(model_id=JUDGE_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def call_qwen_binary_judge(tokenizer, model, question, gold, pred):
    prompt = BINARY_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)
    messages = [
        {"role": "system", "content": "You are a careful binary judge for QA outputs."},
        {"role": "user", "content": prompt}
    ]
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, temperature=0.0)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().upper()
    if raw_text.startswith("CORRECT"):
        return True, raw_text
    if raw_text.startswith("INCORRECT"):
        return False, raw_text
    return None, raw_text


# =====================
# Single Condition Eval
# =====================

def evaluate_condition(
    desta_model, judge_tokenizer, judge_model,
    dataset_id, dataset_name, hop_prefix,
    condition,  # "full" or "blind"
    split=DATA_SPLIT, output_dir="results", seed=DEFAULT_SEED,
):
    ds = load_dataset(dataset_id, "default")[split]
    instruction_key = f"{hop_prefix}instruction"
    answer_key = f"{hop_prefix}answer"
    hop_tag = hop_prefix.rstrip("_")

    out_path = os.path.join(
        output_dir,
        f"blind_test_{dataset_name.lower()}_{hop_tag}_{condition}.jsonl"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    total = len(ds)
    num_correct = 0
    num_valid_judged = 0
    results = []

    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_tag}-{condition}")):
        question = item[instruction_key]
        gold = item[answer_key]

        audio_obj = item["audio"]
        audio_array, sample_rate = _extract_audio_array_and_sr(audio_obj)

        # Prepare transcript (use item's text field if available)
        transcript = item.get("text") or item.get("sentence") or item.get("transcription")

        if condition == "full":
            write_wav_from_array(audio_array, sample_rate, TMP_WAV_FULL)
            wav_path = TMP_WAV_FULL
        else:  # blind
            noise = generate_noise_like(audio_array, sample_rate, seed=seed + idx)
            write_wav_from_array(noise, sample_rate, TMP_WAV_BLIND)
            wav_path = TMP_WAV_BLIND

        try:
            pred = run_desta_on_item(desta_model, item, hop_prefix, wav_path, transcript=transcript)
        except Exception as e:
            print(f"Error on item {idx}: {e}")
            pred = "ERROR"

        try:
            judge_bool, raw_text = call_qwen_binary_judge(
                judge_tokenizer, judge_model, question, gold, pred
            )
        except Exception as e:
            print(f"Judge error on item {idx}: {e}")
            judge_bool = False
            raw_text = "ERROR"

        if judge_bool is not None:
            num_valid_judged += 1
            if judge_bool:
                num_correct += 1

        result_item = {
            "idx": idx,
            "condition": condition,
            "question": question,
            "gold": gold,
            "pred": pred,
            "judge_correct": judge_bool,
            "judge_raw": raw_text,
        }
        results.append(result_item)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    accuracy = num_correct / num_valid_judged if num_valid_judged > 0 else 0.0
    print(f"  {dataset_name} {hop_tag} [{condition}]: {num_correct}/{num_valid_judged} = {accuracy:.4f}")

    return {
        "dataset_name": dataset_name,
        "hop": hop_tag,
        "condition": condition,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_valid_judged": num_valid_judged,
        "total": total,
        "results_path": out_path,
    }


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser(description="Text Dominance 'Blind Test' for DeSTA/ORCA models")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--output_dir", type=str, default="study/output_text_dominance",
                        help="Directory to save results")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: EmotionQA, GenderQA)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}\n")

    # Load models
    print(f"Loading DeSTA model: {args.model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    desta_model = DeSTA25AudioModel.from_pretrained(args.model_id, torch_dtype=dtype)
    desta_model.to(device).eval()

    print(f"Loading Qwen judge: {JUDGE_MODEL_ID}")
    judge_tokenizer, judge_model = load_qwen_judge(JUDGE_MODEL_ID)

    # Select datasets
    datasets_to_eval = DATASETS
    if args.datasets:
        datasets_to_eval = {k: v for k, v in DATASETS.items() if k in args.datasets}

    # Run evaluation: both conditions for each dataset x hop
    all_stats = []
    for dataset_name, dataset_id in datasets_to_eval.items():
        for hop_prefix in HOP_SPLITS:
            for condition in ["full", "blind"]:
                stats = evaluate_condition(
                    desta_model=desta_model,
                    judge_tokenizer=judge_tokenizer,
                    judge_model=judge_model,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    hop_prefix=hop_prefix,
                    condition=condition,
                    output_dir=args.output_dir,
                    seed=args.seed,
                )
                all_stats.append(stats)

    # Compute Delta and print summary
    print(f"\n{'='*70}")
    print(f"  TEXT DOMINANCE 'BLIND TEST' SUMMARY")
    print(f"  Model: {args.model_id}")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12s} {'Hop':<8s} {'Acc(Full)':<12s} {'Acc(Blind)':<12s} {'Delta':<10s} {'Verdict'}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*20}")

    summary_report = {"model_id": args.model_id, "seed": args.seed, "results": []}

    # Group stats by (dataset, hop)
    from collections import defaultdict
    grouped = defaultdict(dict)
    for s in all_stats:
        key = (s["dataset_name"], s["hop"])
        grouped[key][s["condition"]] = s

    for (ds_name, hop), conds in sorted(grouped.items()):
        acc_full = conds.get("full", {}).get("accuracy", 0.0)
        acc_blind = conds.get("blind", {}).get("accuracy", 0.0)
        delta = acc_full - acc_blind

        if delta < 0.05:
            verdict = "TEXT DOMINANT"
        elif delta < 0.15:
            verdict = "Weak audio use"
        elif delta < 0.30:
            verdict = "Moderate"
        else:
            verdict = "Audio-aware"

        print(f"  {ds_name:<12s} {hop:<8s} {acc_full:<12.4f} {acc_blind:<12.4f} {delta:<10.4f} {verdict}")

        summary_report["results"].append({
            "dataset": ds_name,
            "hop": hop,
            "acc_full": acc_full,
            "acc_blind": acc_blind,
            "delta": delta,
            "verdict": verdict,
        })

    print(f"{'='*70}")
    print(f"  Delta = Acc(Full) - Acc(Blind)")
    print(f"  Large Delta -> model relies on audio (good)")
    print(f"  Small Delta -> model ignores audio, text dominant (bad)")
    print(f"{'='*70}")

    # Save summary
    report_path = os.path.join(args.output_dir, "blind_test_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {report_path}")


if __name__ == "__main__":
    main()
