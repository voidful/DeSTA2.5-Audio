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
DATASET_ID = "lmms-lab/mmau"
DEFAULT_SPLIT = "test_mini"
RESULT_DIR = "mmau_results_qwen"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

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
    Extract audio from MMAU dataset item and write to wav file.
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
# Scoring Logic (from mmau_evaluate.py)
# =====================

def string_match(answer, prediction, choices):
    def tokenize(text):
        if not isinstance(text, str):
            text = str(text)
        return set(re.findall(r'\b\w+\b', text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


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


def run_qwen_omni_on_item(model, processor, item, tmp_wav_path):
    """
    Run inference on a single sample with Qwen2.5-Omni. Returns text answer.
    Uses thinker model directly for text-only generation (no audio output).
    """
    write_wav_from_dataset_item(item, tmp_wav_path)

    # Build question with choices
    choices = item["choices"]
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except:
            pass

    choice_text = ""
    if choices:
        choice_text = "\nOptions:\n" + "\n".join([f'"{opt}"' for opt in choices])

    # Align with robust prompt format
    full_question = (
        f"Question: {item['question'].strip()}\n"
        f"{choice_text}\n\n"
        "Answer with the text corresponding to the correct answer. The correct answer is"
    )

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
                {"type": "text", "text": full_question}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Load audio as array for processor
    audio_array = load_audio_as_array(tmp_wav_path, target_sr=16000)

    inputs = processor(
        text=text,
        audios=[audio_array],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        # Use thinker model directly for text-only generation (avoids OOM from audio synthesis)
        # The thinker is the language model component without the audio output head
        text_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode the generated text
    text_ids = text_ids[:, inputs.input_ids.size(1):]
    pred = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if isinstance(pred, str):
        # Clean thinking process
        pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()

        # Extract answer following "The correct answer is:"
        match = re.search(r'The correct answer is:\s*["\']?(.*?)["\']?$', pred_no_think, re.IGNORECASE)
        if match:
            cleaned_pred = match.group(1).strip()
        else:
            cleaned_pred = pred_no_think

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
    parser = argparse.ArgumentParser(description="Run MMAU evaluation with Qwen2.5-Omni")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--no_judge", action="store_true", help="Skip LLM judge and use only string matching")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Temporary wav path for processing
    tmp_wav_path = os.path.join(args.output_dir, "tmp_mmau_audio.wav")

    # Load Qwen2.5-Omni
    print(f"Loading Qwen2.5-Omni model from {args.model_id}...")
    model, processor = load_qwen_omni(args.model_id)
    model.eval()

    # Load Judge (optional)
    judge_tokenizer, judge_model = None, None
    if not args.no_judge:
        print(f"Loading Judge model {JUDGE_MODEL_ID}...")
        judge_tokenizer, judge_model = load_judge()

    # Load dataset
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

        # 1) Qwen2.5-Omni inference
        pred = run_qwen_omni_on_item(model, processor, item, tmp_wav_path)

        # 2) String Match
        is_string_correct = string_match(answer, pred, choices)

        # 3) LLM Judge as a secondary check
        is_llm_correct = None
        judge_raw = "N/A"
        if judge_tokenizer and judge_model:
            is_llm_correct, judge_raw = call_judge(judge_tokenizer, judge_model, item, pred)

        # Combine results
        is_correct = is_string_correct or (is_llm_correct if is_llm_correct is not None else False)

        print(f"Match: {is_string_correct}, LLM Judge: {is_llm_correct} ({judge_raw}), Ans: {answer}, Pred: {pred}")

        # Update metrics
        if task not in task_metrics:
            task_metrics[task] = [0, 0]
        if difficulty not in diff_metrics:
            diff_metrics[difficulty] = [0, 0]
        if subcat not in subcat_metrics:
            subcat_metrics[subcat] = [0, 0]

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

    # Print results
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
    model_name = args.model_id.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_{model_name}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Detailed results saved to: {out_path}")

    # Cleanup temp file
    if os.path.exists(tmp_wav_path):
        os.remove(tmp_wav_path)


if __name__ == "__main__":
    main()
