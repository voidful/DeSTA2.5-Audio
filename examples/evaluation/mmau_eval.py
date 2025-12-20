import os
import json
import wave
import numpy as np
import re
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset
from desta import DeSTA25AudioModel
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

# =====================
# 基本設定
# =====================

DEFAULT_MODEL_ID = "voidful/QAQ_0.6b_orca_all"
DATASET_ID = "lmms-lab/mmau"
DEFAULT_SPLIT = "test_mini"
TMP_WAV_PATH = "tmp_mmau_audio.wav"
RESULT_DIR = "mmau_results"
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


def write_wav_from_dataset_item(item, wav_path):
    """
    從 MMAU dataset item 取出 audio 寫成 wav 檔。
    """
    audio_obj = item["audio"]
    audio_array = audio_obj["array"]
    sample_rate = audio_obj.get("sampling_rate", 16000)

    # 如果 sampling_rate 是 None 或其它非 int
    if sample_rate is None:
        sample_rate = 16000

    return write_wav_from_array(audio_array, sample_rate, wav_path)


# =====================
# Scoring Logic (from mmau_evaluate.py)
# =====================

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        if not isinstance(text, str):
            text = str(text)
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


# =====================
# DeSTA 推論
# =====================

def run_desta_on_item(model, item, wav_path=TMP_WAV_PATH):
    """
    對單一樣本跑 DeSTA. 回傳文字答案。
    """
    write_wav_from_dataset_item(item, wav_path)

    system_prompt = 'Focus on the audio clips and instructions. Provide your answer by first thinking in <think> tags if needed, and then ending with "The correct answer is: \"___\" " where ___ is the exact choice from the list.'

    # Build question with choices (matching inference_desta25_audio.py logic)
    question = f"{item['question']} Choose from the following options: "
    choices = item["choices"]
    # Handle if choices is a string representation of a list
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except:
            pass

    for i, option in enumerate(choices):
        question += f'"{option}"'
        if i == len(choices) - 2:
            question += " or "
        elif i < len(choices) - 1:
            question += ", "

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"<|AUDIO|>\n\n{question}",
            "audios": [{
                "audio": wav_path
            }]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            top_p=0.85,
            temperature=0.0,
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
2. A list of choices.
3. The ground truth answer.
4. The model's predicted answer (which may include reasoning in <think> tags).

Decide if the model's final answer choice is correct according to the ground truth.
Ignore the thinking process in <think> tags. 
The model's answer is correct if it chooses the same meaning or option as the ground truth.

Question: {question}
Choices: {choices}
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
    return tokenizer, model


def call_judge(tokenizer, model, item, pred):
    question = item['question']
    gold = item['answer']
    choices = item['choices']

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        choices=choices,
        gold=gold,
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
    parser = argparse.ArgumentParser(description="Run MMAU evaluation with DeSTA2.5-Audio")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 DeSTA
    print(f"Loading DeSTA model from {args.model_id}...")
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    # 載入 Judge
    print(f"Loading Judge model {JUDGE_MODEL_ID}...")
    judge_tokenizer, judge_model = load_judge()

    # 載入資料
    print(f"Loading dataset {DATASET_ID} split {args.split}...")
    ds = load_dataset(DATASET_ID, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    total = 0
    corr = 0

    # Metrics trackers
    task_metrics = {}
    diff_metrics = {}
    subcat_metrics = {}

    results = []

    for idx, item in enumerate(tqdm(ds, desc="Evaluating")):
        answer = item["answer"]
        task = item["task"]
        difficulty = item["difficulty"]
        subcat = item.get("sub-category", "NA")
        choices = item["choices"]
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except:
                pass

        # 1) DeSTA 推論
        pred = run_desta_on_item(model, item, TMP_WAV_PATH)

        # 2) Match
        is_string_correct = string_match(answer, pred, choices)
        
        # 3) LLM Judge as a secondary check if string match fails or to be sure
        is_llm_correct, judge_raw = call_judge(judge_tokenizer, judge_model, item, pred)
        
        # Combine results: if either is correct, we consider it correct (usually LLM judge is more reliable for complex output)
        is_correct = is_string_correct or is_llm_correct
        
        print(f"Match: {is_string_correct}, LLM Judge: {is_llm_correct} ({judge_raw}), Ans: {answer}, Pred: {pred}")
        # Update metrics
        if task not in task_metrics: task_metrics[task] = [0, 0]
        if difficulty not in diff_metrics: diff_metrics[difficulty] = [0, 0]
        if subcat not in subcat_metrics: subcat_metrics[subcat] = [0, 0]

        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        subcat_metrics[subcat][1] += 1
        total += 1

        if is_correct:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            subcat_metrics[subcat][0] += 1
            corr += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "prediction": pred,
            "is_correct": is_correct,
            "task": task,
            "difficulty": difficulty,
            "subcat": subcat
        })

    # Print results (similar to mmau_evaluate.py)
    print("\n" + "*" * 30)
    print("Task-wise Accuracy:")
    for task, counts in task_metrics.items():
        acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
        print(f"{task} : {acc:.2f}% over {counts[1]} samples")

    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    for diff, counts in diff_metrics.items():
        acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
        print(f"{diff} : {acc:.2f}% over {counts[1]} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat, counts in subcat_metrics.items():
        acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {counts[1]} samples")

    print("*" * 30)
    total_acc = (corr / total) * 100 if total > 0 else 0
    print(f"Total Accuracy: {total_acc:.2f}% over {total} samples")
    print("*" * 30)

    # Save results
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
