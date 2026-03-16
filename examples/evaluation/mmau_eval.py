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
# Noise Injection
# =====================

def add_gaussian_noise_snr(audio_array: np.ndarray, snr_db, sample_idx: int = 0, base_seed: int = DEFAULT_SEED) -> np.ndarray:
    """
    Add white Gaussian noise to an audio waveform at the specified SNR (dB).
    If snr_db is None or inf, returns the original audio unchanged.
    Uses a deterministic RNG seeded with (base_seed + sample_idx) for reproducibility.
    """
    if snr_db is None or np.isinf(snr_db):
        return audio_array
    audio_array = np.asarray(audio_array, dtype=np.float32)
    sig_power = np.mean(audio_array ** 2)
    if sig_power < 1e-10:
        return audio_array  # silence – nothing to corrupt
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    rng = np.random.default_rng(base_seed + sample_idx)
    noise = rng.normal(0, np.sqrt(noise_power), audio_array.shape).astype(np.float32)
    return audio_array + noise

# =====================
# 基本設定
# =====================

# Expected ORCA Configuration:
# - Whisper: openai/whisper-large-v3 (standard, not turbo)
# - Target layers: [7, 15, 23, 31] (4 selected layers)
# - Local downsample: 2x (not 4x)
# - Local kernel size: 5
# - Audio position scale: 5.0
# - Losses: L_ortho_diversity + L_align_layerwise (simplified)

DEFAULT_MODEL_ID = "voidful/desta25_4b_R2_full"  # Update to your trained model
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


def load_audio_as_array(item, snr_db=None):
    """
    Robustly extract audio array from dataset item.
    Optionally injects additive Gaussian noise at the given SNR (dB).
    Supports both legacy dict format and new AudioDecoder (torchcodec) objects.
    """
    audio_obj = item.get("audio") if isinstance(item, dict) else getattr(item, "audio", None)
    # For MMAU it might be "context" or "audio" depending on version, check both
    if audio_obj is None:
        audio_obj = item.get("context") if isinstance(item, dict) else getattr(item, "context", None)

    if audio_obj is None:
        return np.zeros(16000, dtype=np.float32), 16000

    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
             y = np.asarray(audio_obj["array"], dtype=np.float32)
             sr = audio_obj.get("sampling_rate", 16000)
        elif "path" in audio_obj and audio_obj["path"]:
             y, sr = librosa.load(audio_obj["path"], sr=None)
             y = y.astype(np.float32)
        else:
             return np.zeros(16000, dtype=np.float32), 16000
    elif isinstance(audio_obj, str):
        y, sr = librosa.load(audio_obj, sr=None)
        y = y.astype(np.float32)
    else:
        # New datasets versions use AudioDecoder objects (torchcodec)
        try:
            arr = audio_obj["array"] if hasattr(audio_obj, '__getitem__') else audio_obj.array
            y = np.asarray(arr, dtype=np.float32)
            sr = getattr(audio_obj, "sampling_rate", 16000)
        except Exception:
            return np.zeros(16000, dtype=np.float32), 16000
    
    # Inject noise if requested
    y = add_gaussian_noise_snr(y, snr_db)
    return y, sr

def write_wav_from_dataset_item(item, wav_path, snr_db=None):
    """
    Robustly extract audio and write to WAV.
    Optionally injects additive Gaussian noise at the given SNR (dB).
    """
    y, sr = load_audio_as_array(item, snr_db=snr_db)
    return write_wav_from_array(y, sr, wav_path)

def build_prompt(instr, choices):
    # Reference Implementation format
    prompt = f"{instr.strip()} "
    prompt += "Choose from the following options: "
    if choices:
        # Handle stringified choices if necessary
        if isinstance(choices, str):
            try:
                 choices = json.loads(choices)
            except:
                 pass
        
        for i, option in enumerate(choices):
            prompt += f'"{option}"'
            if i == len(choices) - 2:
                prompt += " or "
            else:
                prompt += ", "
    prompt = prompt.rstrip(", ")
    return prompt




# =====================
# Scoring Logic
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

def extract_answer_choice(response):
    """
    Robustly extract answer choice (A/B/C/D) or full answer text from model response.
    """
    if not response:
        return None
        
    pred = response.strip()
    
    # 1) Clean thinking process
    # 1) Clean thinking process
    pred_no_think = re.sub(r'<(?:think|thinking|analysis|analyze_audio|start_analysis)>.*?(?:</(?:think|thinking|analysis|analyze_audio|start_analysis)>|$)', '', pred, flags=re.DOTALL).strip()
    
    # 2) Extract answer with multiple fallback patterns
    patterns = [
        r'The correct answer is:\s*["\']?(.*?)["\']?$',
        r'Final Answer:\s*["\']?(.*?)["\']?$',
        r'Answer:\s*["\']?(.*?)["\']?$',
        r'Option\s*([A-D])'
    ]
    
    extracted = None
    for pat in patterns:
        match = re.search(pat, pred_no_think, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            break
    
    if not extracted:
         # Fallback: look for last isolated A/B/C/D or (A)/(B)/(C)/(D)
         paren_match = re.findall(r'\(([A-D])\)', pred_no_think)
         if paren_match:
             extracted = paren_match[-1] # Take the last one
         else:
             # Last resort: look for just A-D at the end
             last_char_match = re.search(r'\b([A-D])\b[. ]*$', pred_no_think)
             if last_char_match:
                 extracted = last_char_match.group(1)
             else:
                 extracted = pred_no_think # Return full string
    
    return extracted.strip('"').strip("'").strip()

# =====================
# DeSTA 推論
# =====================

def run_desta_on_item(model, item, wav_path=TMP_WAV_PATH, snr_db=None):
    write_wav_from_dataset_item(item, wav_path, snr_db=snr_db)
    
    system_prompt = "You are an audio assistant."
    
    # Build question with choices (matching inference_desta25_audio.py logic)
    choices = item["choices"]
    # Handle if choices is a string representation of a list
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except:
            pass

    options_str = ""
    for i, option in enumerate(choices):
        options_str += f'"{option}"'
        if i == len(choices) - 2:
            options_str += " or "
        elif i < len(choices) - 1:
            options_str += ", "
            
    question = f"{item['question']}\nChoose from the following options: {options_str}"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            # Audio First: <|AUDIO|>\n\n{text}
            "content": f"<|AUDIO|>\n\n{question}\n\nInstructions:\nListen to the audio and select the correct option from the list.\n\nFormat:\nReasoning: <Brief thoughts>\nAnswer: (x) label", 
            "audios": [{
                "audio": wav_path
            }]
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
    return extract_answer_choice(pred)

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
    parser.add_argument("--snr_db", type=float, default=None,
                        help="SNR in dB for additive Gaussian noise (default: None = clean)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()
    
    # Set global random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    snr_tag = "clean" if args.snr_db is None else f"snr{int(args.snr_db)}"

    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 DeSTA
    print(f"Loading DeSTA model from {args.model_id}...")
    # Use torch_dtype in from_pretrained instead of manual .to(dtype)
    model = DeSTA25AudioModel.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16
    )
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
        pred = run_desta_on_item(model, item, TMP_WAV_PATH, snr_db=args.snr_db)

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
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_{snr_tag}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
