import os
import json
import wave
import random
import numpy as np
import re
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset
from desta import DeSTA25AudioModel
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SEED = 42

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================
# 基本設定
# =====================

DEFAULT_MODEL_ID = "voidful/desta25_4b_H1_variational_grouping"
DATASET_ID = "yuantuo666/MMSU-full_5k_hf_format.v0"  # MMSU dataset
DEFAULT_SPLIT = "train"
TMP_WAV_PATH = "tmp_mmsu_audio.wav"
RESULT_DIR = "mmsu_results"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Audio 工具函式
# =====================

def write_wav_from_array(audio_array, sample_rate, wav_path):
    """
    將 float [-1, 1] 波形轉成 mono 16-bit PCM WAV 檔。
    """
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())

    return wav_path


def _extract_audio_array_and_sr(audio_obj):
    """
    Robustly extract (audio_array, sample_rate) from an audio object.
    Supports both legacy dict format and new AudioDecoder (torchcodec, datasets >= 4.x).
    """
    if isinstance(audio_obj, dict):
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = audio_obj.get("sampling_rate", 16000)
    elif hasattr(audio_obj, 'get_all_samples'):
        # datasets AudioDecoder: supports __getitem__ but NOT .get() or attribute access
        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
    else:
        arr = np.asarray(audio_obj["array"], dtype=np.float32) if hasattr(audio_obj, '__getitem__') else np.zeros(16000, dtype=np.float32)
        try:
            sr = int(audio_obj["sampling_rate"])
        except Exception:
            sr = 16000
    if sr is None:
        sr = 16000
    return arr, int(sr)


def write_wav_from_dataset_item(item, wav_path):
    """
    從 MMSU dataset item 取出 audio 寫成 wav 檔。
    """
    audio_obj = item["audio"]
    audio_array, sample_rate = _extract_audio_array_and_sr(audio_obj)
    return write_wav_from_array(audio_array, sample_rate, wav_path)


# =====================
# Scoring Logic
# =====================

def extract_answer_choice(response):
    """
    Extract answer choice (A/B/C/D) from model response.
    """
    if not response:
        return None
    
    response = response.strip().replace('\n', '')
    
    # Try first character
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    
    # Try last character/second last
    if len(response) > 1:
        if response[-2] in ['A', 'B', 'C', 'D']:
            return response[-2]
        if response[-1] in ['A', 'B', 'C', 'D']:
            return response[-1]
    
    # Try to find "Answer: X" or "(X)" patterns
    patterns = [
        r'answer[:\s]+([ABCD])',
        r'([ABCD])\)',
        r'\(([ABCD])\)',
        r'option\s+([ABCD])',
        r'correct answer is\s*([ABCD])',
    ]
    for pattern in patterns:
        match = re.search(pattern, response.upper())
        if match:
            return match.group(1)
    
    return None


def check_answer(pred_choice, answer_index, options):
    """
    Check if predicted choice matches ground truth.
    answer_index is 0-3, options is a list of 4 choices.
    """
    if pred_choice is None:
        return False
    
    choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    pred_index = choice_map.get(pred_choice, -1)
    
    return pred_index == answer_index


# =====================
# DeSTA 推論
# =====================

def build_prompt(question, options):
    """
    Build question with choices - using reference format
    """
    prompt = f"{question.strip()} "
    prompt += "Choose from the following options: "
    if options:
        for i, option in enumerate(options[:4]):
            prompt += f'"{option}"'
            if i == len(options[:4]) - 2:
                prompt += " or "
            else:
                prompt += ", "
    prompt = prompt.rstrip(", ")
    return prompt

# =====================
# DeSTA 推論
# =====================

def run_desta_on_item(model, item, wav_path=TMP_WAV_PATH):
    """
    對單一樣本跑 DeSTA. 回傳文字答案。
    """
    write_wav_from_dataset_item(item, wav_path)

    system_prompt = 'Focus on the audio clips and instructions. Provide your answer by first thinking in <think> tags if needed, and then ending with "The correct answer is: "___" " where ___ is the exact choice from the list.'

    # Build question with choices
    prompt = build_prompt(item.get('question', ''), item.get('options', []))

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            # Audio First: <|AUDIO|>\n\n{text}
            "content": f"<|AUDIO|>\n\n{prompt.replace('<|AUDIO|>', '')}", 
            "audios": [{
                "audio": wav_path
            }]
        }
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                max_new_tokens=512
            )

    pred = outputs.text[0] if isinstance(outputs.text, list) else outputs.text
    if isinstance(pred, str):
        # 1) Clean thinking process
        pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        
        # 2) Extract answer following "The correct answer is:"
        match = re.search(r'The correct answer is:\s*["\']?(.*?)["\']?$', pred_no_think, re.IGNORECASE)
        if match:
            cleaned_pred = match.group(1).strip()
        else:
            # Fallback: if no prefix found, just use the think-stripped version
            cleaned_pred = pred_no_think
            
        # Remove surrounding quotes if any
        cleaned_pred = cleaned_pred.strip('"').strip("'")
        return cleaned_pred
    return str(pred)


# =====================
# LLM Judge Logic
# =====================

JUDGE_PROMPT_TEMPLATE = """You are a strict expert judge for an audio multiple-choice question answering task.

You receive:
1. A question about an audio clip.
2. A list of choices (A, B, C, D).
3. The ground truth answer.
4. The model's predicted answer.

Decide if the model's final answer choice is correct according to the ground truth.
Ignore any thinking process in <think> tags. 

Question: {question}
Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Ground truth answer: {gold}
Model answer: {pred}

Output "CORRECT" or "INCORRECT".
"""


def load_judge(model_id=JUDGE_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    model.to(torch.bfloat16)
    return tokenizer, model


def call_judge(tokenizer, model, item, pred):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=item.get('question', ''),
        choice_a=item.get('choice_a', ''),
        choice_b=item.get('choice_b', ''),
        choice_c=item.get('choice_c', ''),
        choice_d=item.get('choice_d', ''),
        gold=item.get('answer_gt', ''),
        pred=pred
    )

    messages = [
        {"role": "system", "content": "You are a careful judge for multiple-choice QA outputs."},
        {"role": "user", "content": prompt}
    ]

    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().upper()

    if raw_text.startswith("CORRECT"):
        return True, raw_text
    if raw_text.startswith("INCORRECT"):
        return False, raw_text
    return None, raw_text


# =====================
# Main Evaluation function
# =====================

def main():
    parser = argparse.ArgumentParser(description="Run MMSU evaluation with DeSTA2.5-Audio")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--use_judge", action="store_true", help="Use LLM judge for evaluation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    # Set global random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 DeSTA
    print(f"Loading DeSTA model from {args.model_id}...")
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    # 載入 Judge (optional)
    judge_tokenizer, judge_model = None, None
    if args.use_judge:
        print(f"Loading Judge model {JUDGE_MODEL_ID}...")
        judge_tokenizer, judge_model = load_judge()

    # 載入資料
    print(f"Loading dataset {args.dataset_id} split {args.split}...")
    ds = load_dataset(args.dataset_id, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    total = 0
    corr = 0
    fail_num = 0

    # Metrics trackers
    category_metrics = {}
    subcategory_metrics = {}

    results = []

    for idx, item in enumerate(tqdm(ds, desc="Evaluating")):
        # Debug: print first item's keys
        if idx == 0:
            print(f"Dataset fields: {list(item.keys())}")
        
        # Using yuantuo666 format: answer_index (0-3), options (list)
        answer_index = item.get("answer_index", 0)
        category = item.get("category", "NA")
        subcategory = item.get("linguistics_sub_discipline", "NA")
        options = item.get('options', [])

        # 1) DeSTA 推論
        pred = run_desta_on_item(model, item, TMP_WAV_PATH)

        # 2) Extract answer choice
        pred_choice = extract_answer_choice(pred)
        
        # 3) Check correctness (answer_index is 0-3)
        is_correct = check_answer(pred_choice, answer_index, options)
        print(f"Pred: {pred_choice}, Answer: {answer_index}, Correct: {is_correct}")
        # 4) LLM Judge as backup (if enabled)
        if args.use_judge and not is_correct:
            is_llm_correct, judge_raw = call_judge(judge_tokenizer, judge_model, item, pred)
            if is_llm_correct:
                is_correct = True

        if pred_choice is None:
            fail_num += 1

        # Update metrics
        if category not in category_metrics:
            category_metrics[category] = [0, 0]
        if subcategory not in subcategory_metrics:
            subcategory_metrics[subcategory] = [0, 0]

        category_metrics[category][1] += 1
        subcategory_metrics[subcategory][1] += 1
        total += 1

        if is_correct:
            category_metrics[category][0] += 1
            subcategory_metrics[subcategory][0] += 1
            corr += 1

        results.append({
            "id": idx,
            "key": item.get("key", ""),
            "question": item.get("question", ""),
            "answer_index": answer_index,
            "prediction": pred,
            "pred_choice": pred_choice,
            "is_correct": is_correct,
            "category": category,
            "subcategory": subcategory
        })

    # Print results
    print("\n" + "=" * 60)
    print("Category-wise Accuracy:")
    print("=" * 60)
    for cat, counts in sorted(category_metrics.items()):
        acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
        print(f"  {cat}: {acc:.2f}% ({counts[0]}/{counts[1]})")

    print("\n" + "=" * 60)
    print("Sub-category-wise Accuracy:")
    print("=" * 60)
    for subcat, counts in sorted(subcategory_metrics.items()):
        acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
        print(f"  {subcat}: {acc:.2f}% ({counts[0]}/{counts[1]})")

    print("\n" + "=" * 60)
    print("Overall Results:")
    print("=" * 60)
    total_acc = (corr / total) * 100 if total > 0 else 0
    print(f"  Total Accuracy: {total_acc:.2f}% ({corr}/{total})")
    print(f"  Failed parsing: {fail_num}")
    print("=" * 60)

    # Save results
    out_path = os.path.join(args.output_dir, f"mmsu_{args.split}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
