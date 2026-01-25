import os
import json
import wave
import numpy as np
import re
from tqdm import tqdm

import torch
from datasets import load_dataset
from desta import DeSTA25AudioModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

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

DESTA_MODEL_ID = "voidful/desta25_4b_R2_full"  # Update to your trained model

DATASETS = {
    "AnimalQA":  "SLLM-multi-hop/AnimalQA",
    "GenderQA":  "SLLM-multi-hop/GenderQA",
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    "LanguageQA": "SLLM-multi-hop/LanguageQA",
}

HOP_SPLITS = ["single_", "multi_"]     # 兩種 hop 問題
DATA_SPLIT = "test"                    # 目前四個 dataset 都只有 test split

JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

TMP_WAV_PATH = "tmp_audio.wav"
RESULT_DIR = "desta_sakura_results"

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
        wf.setnchannels(1)   # mono
        wf.setsampwidth(2)   # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())

    return wav_path


def load_audio_as_array(item):
    """
    Robustly extract audio array from dataset item.
    """
    audio_obj = item.get("audio")
    # For some datasets it might be "context"
    if audio_obj is None:
        audio_obj = item.get("context")

    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
             return np.asarray(audio_obj["array"], dtype=np.float32), audio_obj.get("sampling_rate", 16000)
        elif "path" in audio_obj and audio_obj["path"]:
             y, sr = librosa.load(audio_obj["path"], sr=None)
             return y.astype(np.float32), sr
    elif isinstance(audio_obj, str):
        y, sr = librosa.load(audio_obj, sr=None)
        return y.astype(np.float32), sr
    
    return np.zeros(16000, dtype=np.float32), 16000

def write_wav_from_dataset_item(item, wav_path):
    """
    從 SAKURA 問答 dataset item 取出 audio 寫成 wav 檔。
    """
    y, sr = load_audio_as_array(item)
    return write_wav_from_array(y, sr, wav_path)

def build_prompt(instr, choices):
    """
    Robust prompts handling both MCQA and Open-ended questions.
    """
    if choices and len(choices) > 0:
        cs = "\n".join(f"({chr(65+i)}) {c.strip()}" for i, c in enumerate(choices))
def build_prompt(instr, choices):
    # Reference Implementation format
    prompt = f"{instr.strip()} "
    prompt += "Choose from the following options: "
    if choices:
        for i, option in enumerate(choices):
            prompt += f'"{option}"'
            if i == len(choices) - 2:
                prompt += " or "
            else:
                prompt += ", "
    prompt = prompt.rstrip(", ")
    return prompt

# =====================
# DeSTA 推論
# =====================

# Global ACD settings (set via command line)
ACD_ENABLED = False
ACD_ALPHA = 0.5

def run_desta_on_item(model, item, hop_prefix, wav_path=TMP_WAV_PATH):
    """
    對單一樣本跑 DeSTA. 回傳文字答案。
    hop_prefix: "single_" 或 "multi_"
    """
    write_wav_from_dataset_item(item, wav_path)

    instruction_key = f"{hop_prefix}instruction"
    question = item[instruction_key]
    choices = item.get("choices")

    # remove System Prompt (Training does not use it)
    # system_prompt = ...

    # Use robust prompt builder
    prompt = build_prompt(question, choices)

    messages = [
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
            if ACD_ENABLED:
                # Use ACD for paralinguistic tasks
                outputs = model.generate_with_acd(
                    messages=messages,
                    acd_alpha=ACD_ALPHA,
                    do_sample=False,
                    top_p=0.85,
                    temperature=0.0,
                    max_new_tokens=64, # Force conciseness
                    repetition_penalty=1.5 # Prevent loops
                )
            else:
                outputs = model.generate(
                    messages=messages,
                    do_sample=False,
                    max_new_tokens=512,
                    repetition_penalty=1.5 # Prevent loops
                )

    pred = outputs.text[0] if isinstance(outputs.text, list) else outputs.text
    if isinstance(pred, str):
        # 1) Clean thinking process
        pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        
        # 2) Extract answer following "The correct answer is:"
        match = re.search(r'The correct answer is:\s*["\']?(.*?)["\']?$', pred_no_think, re.IGNORECASE)
        if match:
            pred = match.group(1).strip()
        else:
            pred = pred_no_think
        
        return pred.strip('"').strip("'")
    return str(pred)



# =====================
# Qwen 評審（二分類）
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


def build_binary_prompt(question, gold, pred):
    return BINARY_PROMPT_TEMPLATE.format(
        question=question,
        gold=gold,
        pred=pred
    )


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
    """
    回傳 (judge_bool, raw_text)
    judge_bool:
        True  -> CORRECT
        False -> INCORRECT
        None  -> 無法解析
    """
    prompt = build_binary_prompt(question, gold, pred)

    messages = [
        {"role": "system", "content": "You are a careful binary judge for QA outputs."},
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

    # 只看前幾個字. 避免多餘空白或其它說明文字
    if raw_text.startswith("CORRECT"):
        return True, raw_text
    if raw_text.startswith("INCORRECT"):
        return False, raw_text
    return None, raw_text


# =====================
# 主評測函式（可重複呼叫）
# =====================

def evaluate_desta_binary_accuracy_on_dataset(
    desta_model,
    judge_tokenizer,
    judge_model,
    dataset_id,
    dataset_name,
    hop_prefix,
    split=DATA_SPLIT,
    output_dir=RESULT_DIR,
):
    """
    對單一 dataset + 單一 hop_prefix (single_ 或 multi_) 做完整評測。
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n========== Evaluating {dataset_name} ({dataset_id}), hop={hop_prefix} ==========")

    # 載入資料
    ds = load_dataset(dataset_id, "default")[split]

    total = len(ds)
    num_correct = 0
    num_valid_judged = 0

    results = []

    instruction_key = f"{hop_prefix}instruction"
    answer_key = f"{hop_prefix}answer"

    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_prefix}{split}")):
        question = item[instruction_key]
        gold = item[answer_key]

        # 1) DeSTA 推論
        pred = run_desta_on_item(desta_model, item, hop_prefix, TMP_WAV_PATH)
        print(gold,pred)
        # 2) Qwen 評審
        judge_bool, raw_text = call_qwen_binary_judge(
            judge_tokenizer,
            judge_model,
            question,
            gold,
            pred
        )

        if judge_bool is not None:
            num_valid_judged += 1
            if judge_bool:
                num_correct += 1

        results.append({
            "idx": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
            "judge_correct": judge_bool,
            "judge_raw": raw_text,
        })

        # 如需 debug 可以開這行
        # print(dataset_name, hop_prefix, idx, pred, gold, judge_bool)

    accuracy = num_correct / num_valid_judged if num_valid_judged > 0 else 0.0

    print(f"\nDeSTA on {dataset_name} ({hop_prefix}{split}), judged by Qwen (binary):")
    print(f"  Valid judged samples: {num_valid_judged}/{total}")
    print(f"  Accuracy: {num_correct}/{num_valid_judged} = {accuracy:.4f}")

    hop_tag = hop_prefix.rstrip("_")  # "single" or "multi"
    out_path = os.path.join(
        output_dir,
        f"desta_{dataset_name.lower()}_{hop_tag}_qwen_binary_results.jsonl"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Results saved to: {out_path}")

    return {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "hop_prefix": hop_prefix,
        "accuracy": accuracy,
        "num_valid_judged": num_valid_judged,
        "num_correct": num_correct,
        "total": total,
        "results_path": out_path,
    }


# =====================
# 主程式：一次跑完 4 個 QA × single/multi
# =====================

def main():
    import argparse
    global ACD_ENABLED, ACD_ALPHA, DESTA_MODEL_ID
    
    parser = argparse.ArgumentParser(description="SAKURA Evaluation for DeSTA/ORCA models")
    parser.add_argument("--model_id", type=str, default=DESTA_MODEL_ID,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--acd_alpha", type=float, default=None,
                        help="ACD contrast strength (enables ACD if set). Try 0.3, 0.5, 0.7, 1.0")
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR,
                        help="Directory to save results")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all)")
    args = parser.parse_args()
    
    # Update globals
    DESTA_MODEL_ID = args.model_id
    if args.acd_alpha is not None:
        ACD_ENABLED = True
        ACD_ALPHA = args.acd_alpha
        print(f"ACD enabled with alpha={ACD_ALPHA}")
    
    # Load DeSTA
    print(f"Loading model from {DESTA_MODEL_ID}...")
    desta_model = DeSTA25AudioModel.from_pretrained(
        DESTA_MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    desta_model.to(device)
    desta_model.eval()

    # 載入 Qwen 評審
    print("Loading Qwen judge model...")
    judge_tokenizer, judge_model = load_qwen_judge(JUDGE_MODEL_ID)

    # 選擇 datasets
    datasets_to_eval = DATASETS
    if args.datasets:
        datasets_to_eval = {k: v for k, v in DATASETS.items() if k in args.datasets}
    
    # 跑所有 dataset × hop 組合
    all_stats = []

    for dataset_name, dataset_id in datasets_to_eval.items():
        for hop_prefix in HOP_SPLITS:
            stats = evaluate_desta_binary_accuracy_on_dataset(
                desta_model=desta_model,
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                hop_prefix=hop_prefix,
                split=DATA_SPLIT,
                output_dir=args.output_dir,
            )
            all_stats.append(stats)

    # 總結表
    print("\n================ Overall summary ================")
    if ACD_ENABLED:
        print(f"ACD alpha: {ACD_ALPHA}")
    for s in all_stats:
        hop_tag = s["hop_prefix"].rstrip("_")
        print(
            f"{s['dataset_name']:12s} | hop={hop_tag:6s} | "
            f"acc={s['accuracy']:.4f} "
            f"({s['num_correct']}/{s['num_valid_judged']} valid; total={s['total']})"
        )


if __name__ == "__main__":
    main()

