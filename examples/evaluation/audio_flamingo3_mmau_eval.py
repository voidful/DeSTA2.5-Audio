import os
import json
import re
import random
import argparse
import numpy as np
import librosa
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
from tqdm import tqdm
from collections import defaultdict

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

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_DATASET_ID = "gamma-lab-umd/MMAU-test-mini"
DEFAULT_SPLIT = "test"
RESULT_DIR = "mmau_results_flamingo3"

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

def load_audio_as_array(item, target_sr):
    """
    Robustly extract audio array from dataset item.
    """
    audio_obj = item.get("context")
    if audio_obj is None and "audio" in item:
        audio_obj = item["audio"]
        
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            y = np.asarray(audio_obj["array"], dtype=np.float32)
            orig_sr = int(audio_obj.get("sampling_rate", target_sr))
        elif "path" in audio_obj and audio_obj["path"]:
            y, orig_sr = librosa.load(audio_obj["path"], sr=None, mono=True)
            y = y.astype(np.float32)
        else:
             # Fallback or error
             return np.zeros(16000), 16000
    elif isinstance(audio_obj, str):
        # Path
        y, orig_sr = librosa.load(audio_obj, sr=None, mono=True)
        y = y.astype(np.float32)
    elif hasattr(audio_obj, 'get_all_samples'):
        # datasets AudioDecoder (torchcodec, datasets >= 4.x)
        y = np.asarray(audio_obj["array"], dtype=np.float32)
        orig_sr = int(audio_obj["sampling_rate"])
    elif isinstance(audio_obj, np.ndarray):
         y = audio_obj.astype(np.float32)
         orig_sr = target_sr # assumption if not provided
    else:
        return np.zeros(16000), 16000

    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    return y, target_sr

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
# Main Evaluation
# =====================

def main():
    parser = argparse.ArgumentParser(description="Run MMAU evaluation with Audio Flamingo 3")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use a unique temp file per process/run to avoid collisions if running parallel
    import uuid
    temp_wav_path = os.path.join(args.output_dir, f"temp_flamingo_{uuid.uuid4()}.wav")

    # Load Model & Processor
    print(f"Loading Audio Flamingo 3 model from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Robust device placement
    # With device_map="auto", the model is already on device(s). 
    # identifying main device for inputs
    device = next(model.parameters()).device
        
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

    # Evaluation Loop
    for ex in tqdm(ds, desc="Evaluating"):
        other = safe_json_loads(ex.get("other_attributes", "{}"))
        subcat = other.get("sub-category", "UNKNOWN")
        
        # Robust metadata extraction
        task = ex.get("task")
        if not task:
            task = other.get("task", "UNKNOWN")
            
        difficulty = ex.get("difficulty")
        if not difficulty:
            difficulty = other.get("difficulty", "UNKNOWN")

        # Prepare Prompt
        prompt = build_prompt(ex["instruction"], ex["choices"])

        # Prepare Audio
        y, _ = load_audio_as_array(ex, target_sr)
        
        # Save to temp file for Flamingo path reference
        sf.write(temp_wav_path, y, target_sr)

        # Build Conversation with Structured Input
        # Use 'audio' type as per user snippet
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

        # Cast float inputs to model dtype
        dtype = model.dtype
        for key in ["input_features", "pixel_values"]:
            if key in inputs and inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(dtype)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )

        # Decode
        prompt_len = inputs.input_ids.shape[1]
        out_ids = outputs[:, prompt_len:]
        pred = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Cleanup prompt echo if any (usually handled by slicing, but just in case)
        pred_clean = strip_mc_prefix(pred)
        
        # Evaluate
        ans_clean = strip_mc_prefix(ex["answer"])
        choices_clean = [strip_mc_prefix(c) for c in ex["choices"]]
        
        ok = string_match(ans_clean, pred_clean, choices_clean)

        # Update Metrics
        by_subcat[subcat]["total"] += 1
        by_subcat[subcat]["correct"] += int(ok)
        
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(ok)
        
        by_difficulty[difficulty]["total"] += 1
        by_difficulty[difficulty]["correct"] += int(ok)
        
        total += 1
        total_correct += int(ok)

        print(f"Pred: {pred_clean} | Ans: {ans_clean} | Correct: {ok}")

        results.append({
            "id": ex.get("id", other.get("id", "unknown")),
            "question": ex["instruction"],
            "answer": ex["answer"],
            "prediction": pred,
            "is_correct": ok,
            "task": task,
            "difficulty": difficulty,
            "subcat": subcat
        })

    # Summary
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

    # Save Results
    model_name = args.model_id.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_{model_name}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nDetailed results saved to: {out_path}")
    
    # Cleanup
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

if __name__ == "__main__":
    main()
