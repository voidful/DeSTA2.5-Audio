import os
import json
import wave
import numpy as np
import re
import argparse
from tqdm import tqdm
import torch
import librosa
from datasets import load_dataset
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

logging.basicConfig(level=logging.INFO)

# =====================
# Basic Configuration
# =====================

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

DATASETS = {
    "AnimalQA":  "SLLM-multi-hop/AnimalQA",
    "GenderQA":  "SLLM-multi-hop/GenderQA",
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    "LanguageQA": "SLLM-multi-hop/LanguageQA",
}

HOP_SPLITS = ["single_", "multi_"]
DATA_SPLIT = "test"
RESULT_DIR = "sakura_results_qwen_omni"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Audio Utility Functions
# =====================

def write_wav_from_array(audio_array, sample_rate, wav_path):
    """
    Convert float [-1, 1] waveform to mono 16-bit PCM WAV file.
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
    Extract audio from SAKURA dataset item and write to wav file.
    """
    audio_obj = item["audio"]
    audio_array = audio_obj["array"]
    sample_rate = audio_obj.get("sampling_rate", 16000)

    if sample_rate is None:
        sample_rate = 16000

    return write_wav_from_array(audio_array, sample_rate, wav_path)


def load_audio_as_array(wav_path, target_sr=16000):
    """
    Load audio file and return as numpy array at target sample rate.
    """
    audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    return audio

# =====================
# Qwen2.5-Omni Inference
# =====================

def load_qwen_omni(model_id):
    """
    Load Qwen2.5-Omni model and processor.
    """
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return model, processor


def run_qwen_omni_on_item(model, processor, item, hop_prefix, tmp_wav_path):
    """
    Run inference on a single sample with Qwen2.5-Omni. Returns text answer.
    """
    write_wav_from_dataset_item(item, tmp_wav_path)

    instruction_key = f"{hop_prefix}instruction"
    question = item[instruction_key]

    # Use Qwen2.5-Omni's exact default system prompt to avoid warning
    default_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    # Qwen2.5-Omni conversation format
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": default_system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": tmp_wav_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Ensure audio tokens are present (fallback if template fails)
    if "<|audio_bos|>" not in text and "<|AUDIO|>" not in text:
        # Manually reconstruct prompt with audio tokens if missing
        print(f"Warning: Audio tokens missing in prompt for item. Injecting manually.")
        system_part = f"<|im_start|>system\n{default_system_prompt}<|im_end|>\n"
        user_part = f"<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        text = system_part + user_part

    # Load audio as array for processor
    audio_array = load_audio_as_array(tmp_wav_path, target_sr=16000)


    inputs = processor(
        text=text,
        audio=[audio_array],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        # Use thinker model directly for text-only generation
        text_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode the generated text
    text_ids = text_ids[:, inputs.input_ids.size(1):]
    pred = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Clean thinking process and return
    if isinstance(pred, str):
        # Clean thinking process
        pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        return pred_no_think
    
    return str(pred)

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
# Evaluation Loop
# =====================

def evaluate_dataset_hop(
    model, processor, 
    judge_tokenizer, judge_model, 
    dataset_name, dataset_id, hop_prefix, 
    split, output_dir, max_samples
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
    
    tmp_wav_path = os.path.join(output_dir, f"tmp_{dataset_name}_{hop_prefix}.wav")

    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_prefix}{split}")):
        question = item[instruction_key]
        gold = item[answer_key]

        # 1) Inference
        pred = run_qwen_omni_on_item(model, processor, item, hop_prefix, tmp_wav_path)

        # 2) Judge
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

    print(f"\nQwen2.5-Omni on {dataset_name} ({hop_prefix}{split}):")
    print(f"  Valid judged samples: {num_valid_judged}/{total}")
    print(f"  Accuracy: {num_correct}/{num_valid_judged} = {accuracy:.4f}")

    # Save results
    hop_tag = hop_prefix.rstrip("_")
    out_path = os.path.join(
        output_dir,
        f"qwen_omni_{dataset_name.lower()}_{hop_tag}_results.jsonl"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Cleanup temp file
    if os.path.exists(tmp_wav_path):
        os.remove(tmp_wav_path)

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
    parser = argparse.ArgumentParser(description="Run SAKURA evaluation with Qwen2.5-Omni")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Specific datasets to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset/hop for debugging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Qwen2.5-Omni
    print(f"Loading Qwen2.5-Omni model from {args.model_id}...")
    model, processor = load_qwen_omni(args.model_id)
    model.eval()

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
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                hop_prefix=hop_prefix,
                split=DATA_SPLIT,
                output_dir=args.output_dir,
                max_samples=args.max_samples
            )
            all_stats.append(stats)

    # Summary
    print("\n================ Overall Summary ================")
    for s in all_stats:
        hop_tag = s["hop_prefix"].rstrip("_")
        print(f"{s['dataset_name']:12s} | hop={hop_tag:6s} | acc={s['accuracy']:.4f} ({s['num_correct']}/{s['num_valid_judged']} valid)")

if __name__ == "__main__":
    main()
