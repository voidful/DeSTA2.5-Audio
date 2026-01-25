import os
import json
import re
import argparse
import numpy as np
import librosa
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from tqdm import tqdm
from collections import defaultdict

# =====================
# Basic Configuration
# =====================

DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DEFAULT_DATASET_ID = "gamma-lab-umd/MMAU-test-mini" # User snippet default, can be overridden
DEFAULT_SPLIT = "test"
RESULT_DIR = "mmau_results_qwen2audio"

# =====================
# Utility Functions
# =====================

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except:
        return {}

def strip_mc_prefix(text: str) -> str:
    t = (text or "").strip()
    return re.sub(r"^\(?[A-H]\)?\s*[\)\.\:]\s*", "", t).strip()

def tokenize_words(text: str) -> set:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))

def string_match(answer, pred, choices):
    pred_toks = tokenize_words(pred)
    ans_toks = tokenize_words(answer)
    if not pred_toks:
        return False
    wrong = set()
    for c in choices:
        ct = tokenize_words(c)
        if ct != ans_toks:
            wrong.update(ct - ans_toks)
    return ans_toks.issubset(pred_toks) and pred_toks.isdisjoint(wrong)

def load_audio(audio_obj, target_sr: int) -> np.ndarray:
    """
    Robustly load audio from HF datasets Audio feature.
    audio_obj can be:
      - {"array": ..., "sampling_rate": ...}
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
    else:
        y, orig_sr = librosa.load(audio_obj, sr=None, mono=True)
        y = y.astype(np.float32)

    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    return y

def build_prompt(instr, choices):
    if choices and len(choices) > 0:
        cs = "\n".join(f"({chr(65+i)}) {c.strip()}" for i, c in enumerate(choices))
        return (
            f"{instr.strip()}\n\n"
            f"Options:\n{cs}\n\n"
            "Answer with the option letter and text corresponding to the correct answer."
        )
    else:
        # Fallback for open-ended or missing choices
        return (
            f"{instr.strip()}\n\n"
            "Answer the question directly and concisely."
        )

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
# Main Evaluation
# =====================

def main():
    parser = argparse.ArgumentParser(description="Run MMAU evaluation with Qwen2-Audio")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model & Processor
    print(f"Loading Qwen2-Audio model from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype="auto"
    )
    
    # Use real parameter device logic for safety
    device = next(model.parameters()).device
    # If not on GPU but available, can move it, though normally user does this via script
    if torch.cuda.is_available() and device.type == "cpu":
        model.to("cuda")
        device = torch.device("cuda")

    model.eval()
    
    target_sr = int(processor.feature_extractor.sampling_rate)
    print(f"Model loaded on {device}, target_sr={target_sr}")

    # Load Dataset
    print(f"Loading dataset {args.dataset_id} split {args.split}...")
    ds = load_dataset(args.dataset_id, split=args.split)
    
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Metrics
    by_subcat = defaultdict(lambda: {"correct": 0, "total": 0})
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    
    total = 0
    total_correct = 0
    
    results = []
    
    # Batch Processing buffers
    batch_convs = []
    batch_audio = []
    batch_meta = []

    def process_batch(convs, audios, metas):
        nonlocal total, total_correct
        
        # Consistent chat template calls
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
            padding=True
        )
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        
        prompt_len = inputs["input_ids"].size(1)
        out_ids = out_ids[:, prompt_len:]
        preds = processor.batch_decode(out_ids, skip_special_tokens=True)

        for p, m in zip(preds, metas):
            ans_clean = strip_mc_prefix(m["answer"])
            choices_clean = [strip_mc_prefix(c) for c in m["choices"]]
            got = strip_mc_prefix(p)

            ok = string_match(ans_clean, got, choices_clean)
            
            # Update metrics
            subcat = m["subcat"]
            task = m["task"]
            diff = m["difficulty"]
            
            by_subcat[subcat]["total"] += 1
            by_subcat[subcat]["correct"] += int(ok)
            
            by_task[task]["total"] += 1
            by_task[task]["correct"] += int(ok)
            
            by_difficulty[diff]["total"] += 1
            by_difficulty[diff]["correct"] += int(ok)
            
            total += 1
            total_correct += int(ok)
            
            print(f"Pred: {got} | Ans: {ans_clean} | Correct: {ok}")
            
            results.append({
                "id": m["id"],
                "question": m["question"],
                "answer": m["answer"],
                "prediction": p,
                "is_correct": ok,
                "task": task,
                "difficulty": diff,
                "subcat": subcat
            })

    # Evaluation Loop
    for ex in tqdm(ds, desc="Evaluating"):
        other = safe_json_loads(ex.get("other_attributes", "{}"))
        subcat = other.get("sub-category", "UNKNOWN")
        
        # MMAU dataset fields might vary slightly, ensuring fallback
        task = ex.get("task")
        if not task:
            task = other.get("task", "UNKNOWN")
            
        difficulty = ex.get("difficulty")
        if not difficulty:
            difficulty = other.get("difficulty", "UNKNOWN")
        
        # Load audio
        context = ex.get("context") # or "audio" depending on dataset version
        if context is None and "audio" in ex:
            context = ex["audio"]
            
        y = load_audio(context, target_sr)
        prompt = build_prompt(ex["instruction"], ex["choices"])

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": other.get("id", "audio_placeholder")},
                {"type": "text", "text": prompt},
            ]}
        ]

        batch_convs.append(conversation)
        batch_audio.append(y)
        batch_meta.append({
            "id": ex.get("id", other.get("id", "unknown")),
            "question": ex["instruction"],
            "answer": ex["answer"],
            "choices": ex["choices"],
            "subcat": subcat,
            "task": task,
            "difficulty": difficulty
        })

        if len(batch_convs) >= args.batch_size:
            process_batch(batch_convs, batch_audio, batch_meta)
            batch_convs, batch_audio, batch_meta = [], [], []

    # Process remaining
    if batch_convs:
        process_batch(batch_convs, batch_audio, batch_meta)

    # Printing Results
    print("\n=== Overall Accuracy ===")
    if total > 0:
        print(f"{total_correct}/{total} = {total_correct/total*100:.2f}%\n")
    else:
        print("No samples evaluated.\n")

    print("=== By Task ===")
    for k, m in by_task.items():
        c, n = m["correct"], m["total"]
        if n > 0:
            print(f"{k} : {c}/{n} = {c/n*100:.2f}%")
            
    print("\n=== By Difficulty ===")
    for k, m in by_difficulty.items():
        c, n = m["correct"], m["total"]
        if n > 0:
            print(f"{k} : {c}/{n} = {c/n*100:.2f}%")

    print("\n=== By Sub-Category ===")
    for sc, m in by_subcat.items():
        c, n = m["correct"], m["total"]
        if n > 0:
            print(f"{sc} : {c}/{n} = {c/n*100:.2f}%")

    # Save results
    model_name = args.model_id.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_{model_name}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nDetailed results saved to: {out_path}")

if __name__ == "__main__":
    main()
