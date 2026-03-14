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
logging.basicConfig(level = logging.INFO)

DESTA_MODEL_ID = "voidful/desta25_4b_baseline_full"

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


def write_wav_from_dataset_item(item, wav_path):
    """
    從 SAKURA 問答 dataset item 取出 audio 寫成 wav 檔。
    """
    audio_obj = item["audio"]
    audio_array = audio_obj["array"]
    sample_rate = audio_obj.get("sampling_rate", 16000)

    return write_wav_from_array(audio_array, sample_rate, wav_path)


# =====================
# DeSTA 推論
# =====================

def run_desta_on_item(model, item, hop_prefix, wav_path=TMP_WAV_PATH):
    """
    對單一樣本跑 DeSTA. 回傳文字答案。
    hop_prefix: "single_" 或 "multi_"
    """
    write_wav_from_dataset_item(item, wav_path)

    instruction_key = f"{hop_prefix}instruction"
    # DEBUG input to check for leakage
    print(f"DEBUG_INPUT: {item[instruction_key]}")

    messages = [
        {
            "role": "system",
            "content": "You are an audio assistant."
        },
        {
            "role": "user",
            "content": f"<|AUDIO|>\n\nQuestion: {item[instruction_key]}\n\nInstructions:\nListen to the audio and select the correct option from the list.\n\nFormat:\nReasoning: <Brief thoughts>\nAnswer: (x) label",
            "audios": [{
                "audio": wav_path
            }]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,      # 評測建議關掉 sampling
            max_new_tokens=512
        )

    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, str):
        pred = pred.strip()
    return pred


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

    hop_tag = hop_prefix.rstrip("_")  # "single" or "multi"
    out_path = os.path.join(
        output_dir,
        f"desta_{dataset_name.lower()}_{hop_tag}_qwen_binary_results.jsonl"
    )

    # Clean output file first
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_prefix}{split}")):
        question = item[instruction_key]
        gold = item[answer_key]

        # 1) DeSTA 推论
        try:
            pred = run_desta_on_item(desta_model, item, hop_prefix, TMP_WAV_PATH)
        except Exception as e:
            print(f"Error running DeSTA on item {idx}: {e}")
            pred = "ERROR"

        # 2) Qwen 評審
        try:
            judge_bool, raw_text = call_qwen_binary_judge(
                judge_tokenizer,
                judge_model,
                question,
                gold,
                pred
            )
        except Exception as e:
            print(f"Error running Judge on item {idx}: {e}")
            judge_bool = False
            raw_text = "ERROR"
        
        print(f"Gold: {gold} | Pred: {pred} | Correct: {judge_bool}")

        if judge_bool is not None:
            num_valid_judged += 1
            if judge_bool:
                num_correct += 1

        result_item = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
            "judge_correct": judge_bool,
            "judge_raw": raw_text,
        }
        results.append(result_item)

        # Write incrementally
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print(f"  Results saved to: {out_path}")

    accuracy = num_correct / num_valid_judged if num_valid_judged > 0 else 0.0

    print(f"\nDeSTA on {dataset_name} ({hop_prefix}{split}), judged by Qwen (binary):")
    print(f"  Valid judged samples: {num_valid_judged}/{total}")
    print(f"  Accuracy: {num_correct}/{num_valid_judged} = {accuracy:.4f}")

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
    parser = argparse.ArgumentParser(description="SAKURA Evaluation for DeSTA/ORCA models")
    parser.add_argument("--model_id", type=str, default=DESTA_MODEL_ID,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR,
                        help="Directory to save results")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Update globals
    desta_model_id = args.model_id
    
    # 載入 DeSTA
    print(f"Loading DeSTA model from {desta_model_id}...")
    desta_model = DeSTA25AudioModel.from_pretrained(desta_model_id)
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
    for s in all_stats:
        hop_tag = s["hop_prefix"].rstrip("_")
        print(
            f"{s['dataset_name']:12s} | hop={hop_tag:6s} | "
            f"acc={s['accuracy']:.4f} "
            f"({s['num_correct']}/{s['num_valid_judged']} valid; total={s['total']})"
        )


if __name__ == "__main__":
    main()
