import os
import json
import re
import random
import argparse
import numpy as np
import librosa
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm

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
# Basic Configuration
# =====================

DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

DATASETS = {
    "AnimalQA":  "SLLM-multi-hop/AnimalQA",
    "GenderQA":  "SLLM-multi-hop/GenderQA",
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    "LanguageQA": "SLLM-multi-hop/LanguageQA",
}

HOP_SPLITS = ["single_", "multi_"]
DATA_SPLIT = "test"
RESULT_DIR = "sakura_results_qwen2audio"

# =====================
# Utility Functions
# =====================

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}

def load_audio(audio_obj, target_sr: int) -> np.ndarray:
    """
    Robustly load audio from HF datasets Audio feature.
    audio_obj can be:
      - {"array": ..., "sampling_rate": ...}  (legacy dict)
      - AudioDecoder object (datasets >= 4.x, torchcodec)
      - {"path": ...}
      - a path string
    """
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            y = np.asarray(audio_obj["array"], dtype=np.float32)
            orig_sr = int(audio_obj.get("sampling_rate", target_sr))
        elif "path" in audio_obj and audio_obj["path"]:
            y, orig_sr = librosa.load(audio_obj["path"], sr=None, mono=True)
            y = y.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio dict keys: {list(audio_obj.keys())}")
    elif hasattr(audio_obj, 'get_all_samples'):
        # datasets AudioDecoder: supports __getitem__ but NOT .get() or attribute access
        y = np.asarray(audio_obj["array"], dtype=np.float32)
        orig_sr = int(audio_obj["sampling_rate"])
    elif isinstance(audio_obj, str):
        y, orig_sr = librosa.load(audio_obj, sr=None, mono=True)
        y = y.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio_obj).__name__}")

    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    return y

def build_prompt(instr, choices):
    """
    Optional. Only used if a dataset provides a separate `choices` field
    AND the instruction does not already contain (A)(B)(C)(D)-style options.
    """
    if choices and len(choices) > 0:
        cs = "\n".join(f"({chr(65+i)}) {c.strip()}" for i, c in enumerate(choices))
        return (
            f"{instr.strip()}\n\n"
            f"Options:\n{cs}\n\n"
            "Answer with exactly one option letter (A/B/C/D) and its text."
        )
    return instr.strip()

def _move_batch_to_device(batch, device):
    """
    BatchFeature supports .to(device) in most transformers versions.
    Fallback to moving tensors manually.
    """
    if hasattr(batch, "to"):
        return batch.to(device)

    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch

def _processor_call_with_audio(processor, text, audio_list, target_sr, return_tensors="pt", padding=True):
    inputs = processor(
        text=text,
        audio=audio_list,
        sampling_rate=target_sr,
        return_tensors=return_tensors,
        padding=padding,
    )
    has_audio_feats = any(k in inputs for k in ["input_features", "audio_features", "feature_attention_mask"])
    if has_audio_feats:
        return inputs

    # fallback for older naming
    inputs = processor(
        text=text,
        audios=audio_list,
        sampling_rate=target_sr,
        return_tensors=return_tensors,
        padding=padding,
    )
    return inputs


# =====================
# Qwen Judge Logic (Binary)
# =====================

BINARY_PROMPT_TEMPLATE = """You are a strict expert judge for an audio multiple-choice question answering task.

You receive:
1. The original instruction (it contains the question and the choices).
2. The ground truth answer.
3. The model's predicted answer.

Rules:
- Each question has exactly one correct choice.
- If the model selects zero choices or multiple choices, output INCORRECT.
- If the model selects exactly one choice, judge whether it matches the ground truth in meaning.
- Ignore minor wording differences, punctuation, and synonyms. Focus on meaning.

Instruction:
{question}

Ground truth answer:
{gold}

Model answer:
{pred}

Output exactly one token:
CORRECT
or
INCORRECT
"""

def load_qwen_judge(model_id=JUDGE_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def call_qwen_binary_judge(tokenizer, model, question, gold, pred):
    """
    Returns (judge_bool, raw_text)
    judge_bool is always True/False. Ambiguous outputs default to False.
    """
    prompt = BINARY_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)

    messages = [
        {"role": "system", "content": "You are a strict evaluator that outputs only CORRECT or INCORRECT."},
        {"role": "user", "content": prompt},
    ]

    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().upper()
    first_token = raw_text.split()[0] if raw_text else ""

    if first_token == "CORRECT":
        return True, raw_text
    if first_token == "INCORRECT":
        return False, raw_text
    return False, raw_text

# =====================
# Main Evaluation Loop
# =====================

def evaluate_dataset_hop(
    model, processor, device,
    judge_tokenizer, judge_model,
    dataset_name, dataset_id, hop_prefix,
    split, output_dir, max_samples, batch_size,
    target_sr, verbose=False,
):
    print(f"\n========== Evaluating {dataset_name} ({dataset_id}), hop={hop_prefix} ==========")

    ds = load_dataset(dataset_id, "default")[split]
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    total = len(ds)
    num_correct = 0
    results = []

    instruction_key = f"{hop_prefix}instruction"
    answer_key = f"{hop_prefix}answer"

    batch_convs = []
    batch_audio = []
    batch_meta = []

    def process_batch(convs, audios, metas):
        nonlocal num_correct

        chat_texts = [
            processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in convs
        ]

        inputs = _processor_call_with_audio(
            processor=processor,
            text=chat_texts,
            audio_list=audios,
            target_sr=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
            )

        prompt_len = inputs["input_ids"].size(1)
        out_ids = out_ids[:, prompt_len:]
        preds = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for p, m in zip(preds, metas):
            question = m["question"]
            gold = m["gold"]

            judge_bool, raw_text = call_qwen_binary_judge(
                judge_tokenizer, judge_model,
                question, gold, p,
            )
            if judge_bool:
                num_correct += 1

            if verbose:
                print(f"Gold: {gold} | Pred: {p} | Judge: {judge_bool} | Raw: {raw_text}")

            results.append({
                "idx": m["idx"],
                "question": question,
                "gold": gold,
                "pred": p,
                "judge_correct": judge_bool,
                "judge_raw": raw_text,
            })

    for idx, item in enumerate(tqdm(ds, desc=f"{dataset_name}-{hop_prefix}{split}")):
        question = item[instruction_key]
        gold = item[answer_key]

        y = load_audio(item["audio"], target_sr)

        # If a dataset provides `choices` separately AND the instruction does not already contain options, rebuild.
        if "choices" in item and item["choices"]:
            has_inline_opts = bool(re.search(r"\([A-Da-d]\)", question))
            prompt = question if has_inline_opts else build_prompt(question, item["choices"])
        else:
            prompt = question

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": f"audio_{idx}"},
                {"type": "text", "text": prompt},
            ]},
        ]

        batch_convs.append(conversation)
        batch_audio.append(y)
        batch_meta.append({"idx": idx, "question": question, "gold": gold})

        if len(batch_convs) >= batch_size:
            process_batch(batch_convs, batch_audio, batch_meta)
            batch_convs, batch_audio, batch_meta = [], [], []

    if batch_convs:
        process_batch(batch_convs, batch_audio, batch_meta)

    accuracy = num_correct / total if total > 0 else 0.0

    print(f"\nQwen2-Audio on {dataset_name} ({hop_prefix}{split}):")
    print(f"  Accuracy: {num_correct}/{total} = {accuracy:.4f}")

    hop_tag = hop_prefix.rstrip("_")
    out_path = os.path.join(output_dir, f"qwen2audio_{dataset_name.lower()}_{hop_tag}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset_name": dataset_name,
        "hop_prefix": hop_prefix,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "total": total,
        "results_path": out_path,
    }

def main():
    parser = argparse.ArgumentParser(description="Run SAKURA evaluation with Qwen2-Audio")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Specific datasets to evaluate")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per dataset/hop for debugging. 0 means all.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Qwen2-Audio model from {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    # device_map="auto" can shard the model. Use a real param device for inputs.
    device = next(model.parameters()).device
    target_sr = int(processor.feature_extractor.sampling_rate)
    print(f"Model loaded. device={device}. target_sr={target_sr}")

    print(f"Loading Judge model {JUDGE_MODEL_ID} ...")
    judge_tokenizer, judge_model = load_qwen_judge()

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
                batch_size=args.batch_size,
                target_sr=target_sr,
                verbose=args.verbose,
            )
            all_stats.append(stats)

    print("\n================ Overall Summary ================")
    for s in all_stats:
        hop_tag = s["hop_prefix"].rstrip("_")
        print(
            f"{s['dataset_name']:12s} | hop={hop_tag:6s} | "
            f"acc={s['accuracy']:.4f} ({s['num_correct']}/{s['total']})"
        )

if __name__ == "__main__":
    main()
