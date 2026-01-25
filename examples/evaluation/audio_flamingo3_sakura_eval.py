import os
import json
import re
import argparse
import numpy as np
import librosa
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict

# =====================
# Basic Configuration
# =====================

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

DATASETS = {
    "AnimalQA":  "SLLM-multi-hop/AnimalQA",
    "GenderQA":  "SLLM-multi-hop/GenderQA",
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    "LanguageQA": "SLLM-multi-hop/LanguageQA",
}

HOP_SPLITS = ["single_", "multi_"]
DATA_SPLIT = "test"
RESULT_DIR = "sakura_results_flamingo3"

# =====================
# Utility Functions
# =====================

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except:
        return {}

def load_audio_as_array(item, target_sr):
    """
    Extract audio array from dataset item.
    Returns (array, sampling_rate)
    """
    audio_obj = item.get("audio")
    
    if isinstance(audio_obj, dict) and "array" in audio_obj:
        return audio_obj["array"], audio_obj["sampling_rate"]
    elif isinstance(audio_obj, str):
        # Path
        y, sr = librosa.load(audio_obj, sr=None)
        return y, sr
    else:
        # Unexpected format or None
        return np.zeros(16000), 16000

def build_prompt(instr, choices):
    """
    Robust prompts handling both MCQA and Open-ended questions.
    """
    if choices and len(choices) > 0:
        cs = "\n".join(f"({chr(65+i)}) {c.strip()}" for i, c in enumerate(choices))
        return (
            f"{instr.strip()}\n\n"
            f"Options:\n{cs}\n\n"
            "Answer with the option letter and text corresponding to the correct answer."
        )
    else:
        return (
            f"{instr.strip()}\n\n"
            "Answer the question directly and concisely."
        )

# =====================
# Qwen Judge Logic (Binary)
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
    """
    Returns (judge_bool, raw_text)
    """
    prompt = BINARY_PROMPT_TEMPLATE.format(
        question=question,
        gold=gold,
        pred=pred
    )

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

    if raw_text.startswith("CORRECT"):
        return True, raw_text
    if raw_text.startswith("INCORRECT"):
        return False, raw_text
    return None, raw_text

# =====================
# Main Evaluation Loop
# =====================

def evaluate_dataset_hop(
    model, processor, device,
    judge_tokenizer, judge_model, 
    dataset_name, dataset_id, hop_prefix, 
    split, output_dir, max_samples, batch_size,
    target_sr, temp_wav_path
):
    print(f"\n========== Evaluating {dataset_name} ({dataset_id}), hop={hop_prefix} ==========")
    
    # Load dataset
    ds = load_dataset(dataset_id, "default")[split]
    
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    total = len(ds)
    num_correct = 0
    num_valid_judged = 0
    results = []
    
    instruction_key = f"{hop_prefix}instruction"
    answer_key = f"{hop_prefix}answer"
    
    # Evaluation Loop (Batch size 1 for safe Flamingo temp file usage)
    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_prefix}{split}")):
        question = item[instruction_key]
        gold = item[answer_key]
        
        # Determine choices if available
        choices = item.get("choices") 
        prompt = build_prompt(question, choices)

        # Prepare Audio
        y, _ = load_audio_as_array(item, target_sr)
            
        # Write to temp file
        sf.write(temp_wav_path, y, target_sr)

        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": temp_wav_path}
            ]}
        ]
        
        # Inference
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        # Cast float inputs to model dtype (AudioFlamingo3 bias type fix)
        dtype = model.dtype
        for key in ["input_features", "pixel_values"]:
            if key in inputs and inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(dtype)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Decode
        prompt_len = inputs.input_ids.shape[1]
        out_ids = outputs[:, prompt_len:]
        pred = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Judge
        judge_bool, raw_text = call_qwen_binary_judge(
            judge_tokenizer, judge_model, 
            question, gold, pred
        )

        if judge_bool is not None:
            num_valid_judged += 1
            if judge_bool:
                num_correct += 1
        
        print(f"Goal: {gold}, Pred: {pred}, Judge: {judge_bool}")

        results.append({
            "idx": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
            "judge_correct": judge_bool,
            "judge_raw": raw_text,
        })

    accuracy = num_correct / num_valid_judged if num_valid_judged > 0 else 0.0

    print(f"\nFlamingo3 on {dataset_name} ({hop_prefix}{split}):")
    print(f"  Valid judged samples: {num_valid_judged}/{total}")
    print(f"  Accuracy: {num_correct}/{num_valid_judged} = {accuracy:.4f}")

    # Save results
    hop_tag = hop_prefix.rstrip("_")
    out_path = os.path.join(
        output_dir,
        f"flamingo3_{dataset_name.lower()}_{hop_tag}_results.jsonl"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset_name": dataset_name,
        "hop_prefix": hop_prefix,
        "accuracy": accuracy,
        "num_valid_judged": num_valid_judged,
        "num_correct": num_correct,
        "total": total,
        "results_path": out_path,
    }

def main():
    parser = argparse.ArgumentParser(description="Run SAKURA evaluation with Audio Flamingo 3")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Specific datasets to evaluate")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per dataset/hop for debugging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_wav_path = os.path.join(args.output_dir, "temp_flamingo_sakura_audio.wav")

    # Load Flamingo 3
    print(f"Loading Audio Flamingo 3 model from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    device = next(model.parameters()).device
    target_sr = int(processor.feature_extractor.sampling_rate)
    print(f"Model loaded on {device}, target_sr={target_sr}")

    # Load Judge
    print(f"Loading Judge model {JUDGE_MODEL_ID}...")
    judge_tokenizer, judge_model = load_qwen_judge()

    # Determine datasets loop
    datasets_to_eval = DATASETS
    if args.datasets:
        datasets_to_eval = {k: v for k, v in DATASETS.items() if k in args.datasets}

    all_stats = []

    for dataset_name, dataset_id in datasets_to_eval.items():
        for hop_prefix in HOP_SPLITS:
            stats = evaluate_dataset_hop(
                model=model,
                processor=processor,
                device=device,
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                hop_prefix=hop_prefix,
                split=DATA_SPLIT,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                batch_size=1, # Fixed to 1 for Flamingo temp file handling
                target_sr=target_sr,
                temp_wav_path=temp_wav_path
            )
            all_stats.append(stats)

    # Summary
    print("\n================ Overall Summary ================")
    for s in all_stats:
        hop_tag = s["hop_prefix"].rstrip("_")
        print(f"{s['dataset_name']:12s} | hop={hop_tag:6s} | acc={s['accuracy']:.4f} ({s['num_correct']}/{s['num_valid_judged']} valid)")
    
    # Cleanup
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

if __name__ == "__main__":
    main()
