import os
import json
import wave
import random
import gc
import numpy as np
import re
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
from datasets import load_dataset
from desta import DeSTA25AudioModel
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor

DEFAULT_SEED = 42

# Group ablation configurations from the paper (Table group_config)
# Each tuple is (G, K) with M = G*K = 64 fixed
GROUP_ABLATION_CONFIGS = [
    (1, 64),
    (2, 32),
    (4, 16),
    (8, 8),
    (16, 4),
]

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


def load_audio_as_array(item, snr_db=None, sample_idx=0):
    """
    Robustly extract audio array from dataset item.
    Optionally injects additive Gaussian noise at the given SNR (dB).
    Compatible with both legacy dict format and new AudioDecoder (torchcodec) objects.
    """
    # --- Step 1: Get the audio object from item ---
    audio_obj = None
    for key in ("audio", "context"):
        if isinstance(item, dict):
            audio_obj = item.get(key)
        else:
            audio_obj = getattr(item, key, None)
        if audio_obj is not None:
            break

    if audio_obj is None:
        return np.zeros(16000, dtype=np.float32), 16000

    # --- Step 2: Extract array + sample rate from audio_obj ---
    y, sr = None, 16000

    # Strategy A: dict with "array" key (legacy datasets format)
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            y = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = audio_obj.get("sampling_rate", 16000)
        elif "path" in audio_obj and audio_obj["path"]:
            import librosa
            y, sr = librosa.load(audio_obj["path"], sr=None)
            y = y.astype(np.float32)

    # Strategy B: file path string
    elif isinstance(audio_obj, str):
        import librosa
        y, sr = librosa.load(audio_obj, sr=None)
        y = y.astype(np.float32)

    # Strategy C: new datasets AudioDecoder / torchcodec objects
    else:
        obj_type = type(audio_obj).__name__
        # Try multiple access patterns
        for attempt, extractor in enumerate([
            # C1: dict-like ["array"] access
            lambda o: (np.asarray(o["array"], dtype=np.float32), o.get("sampling_rate", 16000) if hasattr(o, "get") else getattr(o, "sampling_rate", 16000)),
            # C2: attribute .array access
            lambda o: (np.asarray(o.array, dtype=np.float32), getattr(o, "sampling_rate", 16000)),
            # C3: numpy() for torch tensors
            lambda o: (o.numpy().astype(np.float32) if hasattr(o, 'numpy') else None, 16000),
        ]):
            try:
                result = extractor(audio_obj)
                if result[0] is not None and len(result[0]) > 0:
                    y, sr = result
                    break
            except Exception:
                continue

        if y is None:
            return np.zeros(16000, dtype=np.float32), 16000

    # --- Step 3: Validate ---
    if y is None or len(y) == 0:
        return np.zeros(16000, dtype=np.float32), 16000

    # Inject noise if requested
    y = add_gaussian_noise_snr(y, snr_db, sample_idx=sample_idx)
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
    pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
    
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
    
    system_prompt = 'You are an audio question answering assistant. You will be given an audio clip and a question with multiple choices. Please think step-by-step in <think> tags, analyzing the audio content and ruling out incorrect options. Then, output the final answer strictly in the format: "The correct answer is: "choice" ".'
    
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
            
    question = f"{item['question']} Choose from the following options: {options_str}"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            # Audio First: <|AUDIO|>\n\n{text}
            "content": f"<|AUDIO|>\n\n{question}", 
            "audios": [{
                "audio": wav_path
            }]
        }
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                max_new_tokens=512,
                repetition_penalty=1.5
            )
    
    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, str):
        pred = pred.strip()
    return extract_answer_choice(pred)


def run_desta_with_latent_sampling(model, item, num_samples, tau,
                                   wav_path=TMP_WAV_PATH, snr_db=None):
    """
    Run DeSTA inference S times with latent sampling (τ > 0),
    then return the majority-voted answer.
    
    Each sample draws z = μ + τ·σ·ε with fresh ε ~ N(0,I),
    giving a different perceptual "view" of the same audio.
    """
    connector = model.perception.connector
    
    # Check if model supports variational sampling
    if not getattr(connector, 'variational_enabled', False):
        # Not variational — just run once
        pred = run_desta_on_item(model, item, wav_path, snr_db=snr_db)
        return pred, [pred]
    
    # Set the sampling temperature
    original_alpha = connector.s1_inference_alpha
    connector.s1_inference_alpha = tau
    
    predictions = []
    for s in range(num_samples):
        pred = run_desta_on_item(model, item, wav_path, snr_db=snr_db)
        predictions.append(pred)
    
    # Restore original alpha
    connector.s1_inference_alpha = original_alpha
    
    # Majority vote
    vote_counts = Counter(p for p in predictions if p is not None)
    if vote_counts:
        final_pred = vote_counts.most_common(1)[0][0]
    else:
        final_pred = predictions[0] if predictions else None
    
    return final_pred, predictions

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
# Group Ablation Diagnostics
# =====================

def compute_mean_offdiag_cosine_similarity(queries):
    """
    For each sample, compute cosine similarity between all pairs of
    query vectors. Return the mean off-diagonal cosine similarity
    averaged across all samples. Lower = less query redundancy.
    
    Args:
        queries: np.ndarray [N, M, D]
    Returns:
        float: mean off-diagonal cosine similarity
    """
    N, M, D = queries.shape
    offdiag_sims = []
    for i in range(N):
        vecs = queries[i]  # [M, D]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs_normed = vecs / norms
        gram = vecs_normed @ vecs_normed.T  # [M, M]
        mask = ~np.eye(M, dtype=bool)
        offdiag_sims.append(float(gram[mask].mean()))
    return float(np.mean(offdiag_sims)) if offdiag_sims else 0.0


@torch.inference_mode()
def extract_query_vectors_from_mmau(model, ds, device, max_samples=None):
    """
    Extract query vectors [N, M, D] from MMAU audio items via the ORCA connector.
    Reuses load_audio_as_array() already defined in this script.
    """
    perception = model.perception
    connector = perception.connector
    whisper_enc = perception.whisper.model.encoder
    encoder_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    processor = AutoFeatureExtractor.from_pretrained(encoder_id)

    n = min(max_samples, len(ds)) if max_samples else len(ds)
    all_queries = []
    print(f"  Extracting query vectors from {n} MMAU samples ...")

    for i in range(n):
        item = ds[i]
        audio_array, sr = load_audio_as_array(item)
        if audio_array is None:
            continue

        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(device)
        target_dtype = whisper_enc.conv1.weight.dtype
        target_dev = whisper_enc.conv1.weight.device
        feats = feats.to(dtype=target_dtype, device=target_dev)

        h = torch.nn.functional.gelu(whisper_enc.conv1(feats))
        h = torch.nn.functional.gelu(whisper_enc.conv2(h))
        h = h.permute(0, 2, 1)
        pos = whisper_enc.embed_positions.weight[
            : whisper_enc.config.max_source_positions
        ].to(dtype=h.dtype, device=h.device)
        hidden = h + pos

        all_layer_outputs = []
        for enc_layer in whisper_enc.layers:
            hidden = enc_layer(hidden, None, None)[0]
            all_layer_outputs.append(hidden)

        conn_out = connector(all_layer_outputs)
        query_vecs = conn_out[0] if isinstance(conn_out, tuple) else conn_out
        all_queries.append(query_vecs[0].float().cpu().numpy())

        if (i + 1) % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    [{i+1}/{n}]")

    queries = np.stack(all_queries, axis=0)  # [N, M, D]
    print(f"  → queries shape: {queries.shape}")
    return queries


# =====================
# Evaluation helpers
# =====================

def evaluate_model_on_mmau(model, ds, judge_tokenizer, judge_model, snr_db=None,
                           latent_configs=None):
    """
    Run MMAU evaluation for a single model. Returns (results, ablation_summary, trackers).
    """
    if latent_configs is None:
        latent_configs = [(0.0, 1)]

    trackers = {cfg: {
        "total": 0, "corr": 0,
        "task": defaultdict(lambda: [0, 0]),
        "diff": defaultdict(lambda: [0, 0]),
        "subcat": defaultdict(lambda: [0, 0])
    } for cfg in latent_configs}

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

        item_result = {
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "task": task,
            "difficulty": difficulty,
            "subcat": subcat,
            "configs": {}
        }

        for cfg in latent_configs:
            tau, S = cfg
            if tau == 0.0 and S == 1:
                pred, sample_preds = run_desta_with_latent_sampling(model, item, 1, 0.0, TMP_WAV_PATH, snr_db=snr_db)
            else:
                pred, sample_preds = run_desta_with_latent_sampling(model, item, S, tau, TMP_WAV_PATH, snr_db=snr_db)

            is_string_correct = string_match(answer, pred, choices)
            is_llm_correct, judge_raw = call_judge(judge_tokenizer, judge_model, item, pred)
            is_correct = is_string_correct or is_llm_correct

            tr = trackers[cfg]
            tr["total"] += 1
            tr["task"][task][1] += 1
            tr["diff"][difficulty][1] += 1
            tr["subcat"][subcat][1] += 1

            if is_correct:
                tr["corr"] += 1
                tr["task"][task][0] += 1
                tr["diff"][difficulty][0] += 1
                tr["subcat"][subcat][0] += 1

            item_result["configs"][f"tau{tau}_S{S}"] = {
                "prediction": pred,
                "is_correct": is_correct,
                "sample_predictions": sample_preds
            }

            if tau == 0.0 and S == 1:
                print(f"Match: {is_string_correct}, Judge: {is_llm_correct}, Ans: {answer}, Pred: {pred}")

        results.append(item_result)

    ablation_summary = []
    for cfg in latent_configs:
        tau, S = cfg
        tr = trackers[cfg]
        total = tr["total"]
        corr = tr["corr"]
        total_acc = (corr / total) * 100 if total > 0 else 0

        print("\n" + "=" * 50)
        print(f"RESULTS FOR CONFIG: tau={tau}, S={S}")
        print("=" * 50)
        print(f"Overall Accuracy: {total_acc:.2f}% ({corr}/{total})")

        ablation_summary.append({
            "tau": tau,
            "S": S,
            "accuracy": round(total_acc, 2),
            "correct": corr,
            "total": total
        })

        if tau == 0.0 and S == 1:
            print("-" * 30)
            print("Task-wise Accuracy:")
            for t, counts in tr["task"].items():
                acc = (counts[0] / counts[1]) * 100 if counts[1] > 0 else 0
                print(f"{t} : {acc:.2f}% over {counts[1]}")

    return results, ablation_summary, trackers


def resolve_group_model_id(base_model_id, G, K, pattern=None):
    """
    Resolve model ID for a given (G, K) configuration.
    If pattern is provided, substitute {G} and {K}.
    Otherwise, use base_model_id_G{G}_K{K}.
    """
    if pattern:
        return pattern.format(G=G, K=K)
    return f"{base_model_id}_G{G}_K{K}"


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
    parser.add_argument("--group_ablation", action="store_true",
                        help="Run group config ablation: iterate over (G,K) in {(1,64),(2,32),(4,16),(8,8),(16,4)}")
    parser.add_argument("--group_model_pattern", type=str, default=None,
                        help="Model ID pattern with {G} and {K} placeholders, e.g. 'voidful/desta25_4b_G{G}_K{K}'. "
                             "If not set, defaults to <model_id>_G{G}_K{K} (base model_id used as-is for its native G,K).")
    parser.add_argument("--latent_ablation", action="store_true",
                        help="Run latent sampling ablation over specific (tau, S) pairs from the paper.")
    args = parser.parse_args()

    # Set global random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    snr_tag = "clean" if args.snr_db is None else f"snr{int(args.snr_db)}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Judge
    print(f"Loading Judge model {JUDGE_MODEL_ID}...")
    judge_tokenizer, judge_model = load_judge()

    # Load dataset
    print(f"Loading dataset {DATASET_ID} split {args.split}...")
    ds = load_dataset(DATASET_ID, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Latent ablation configs
    latent_configs = [(0.0, 1)]
    if args.latent_ablation:
        latent_configs.extend([
            (0.3, 3),
            (0.3, 5),
            (0.5, 5),
            (1.0, 5)
        ])
        print(f"Latent ablation enabled. Running configs (tau, S): {latent_configs}")

    # =======================
    # Group Ablation Mode
    # =======================
    if args.group_ablation:
        print(f"\n{'='*60}")
        print(f"  GROUP CONFIGURATION ABLATION (M=64 fixed)")
        print(f"  Configs: {GROUP_ABLATION_CONFIGS}")
        print(f"{'='*60}\n")

        group_ablation_results = []

        for G, K in GROUP_ABLATION_CONFIGS:
            model_id = resolve_group_model_id(args.model_id, G, K, args.group_model_pattern)
            print(f"\n{'─'*60}")
            print(f"  Loading model for G={G}, K={K}: {model_id}")
            print(f"{'─'*60}")

            try:
                model = DeSTA25AudioModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16
                )
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"  [SKIP] Failed to load model {model_id}: {e}")
                group_ablation_results.append({
                    "G": G, "K": K, "M": G * K,
                    "model_id": model_id,
                    "mmau_accuracy": None,
                    "mean_offdiag_cosine_sim": None,
                    "error": str(e)
                })
                continue

            # Verify connector config matches expected G, K
            connector = model.perception.connector
            actual_G = connector.num_groups
            actual_K = connector.queries_per_group
            if actual_G != G or actual_K != K:
                print(f"  [WARN] Model reports G={actual_G}, K={actual_K} but expected G={G}, K={K}")

            # Reset seed for each config to ensure identical evaluation order
            set_seed(args.seed)

            # Run MMAU evaluation
            results, ablation_summary, trackers = evaluate_model_on_mmau(
                model, ds, judge_tokenizer, judge_model,
                snr_db=args.snr_db, latent_configs=[(0.0, 1)]
            )
            baseline_acc = ablation_summary[0]["accuracy"]

            # Compute query diagnostics
            connector_mode = model.config.connector_mode
            mean_cos_sim = None
            if connector_mode in ("orca_r1", "orca_hybrid"):
                queries = extract_query_vectors_from_mmau(model, ds, device)
                mean_cos_sim = compute_mean_offdiag_cosine_similarity(queries)

            entry = {
                "G": G, "K": K, "M": G * K,
                "model_id": model_id,
                "mmau_accuracy": baseline_acc,
                "mean_offdiag_cosine_sim": round(mean_cos_sim, 4) if mean_cos_sim is not None else None,
            }
            group_ablation_results.append(entry)

            # Save per-config detailed results
            out_path = os.path.join(args.output_dir, f"group_ablation_G{G}_K{K}_results.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Free model memory before loading the next one
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Print summary table
        print(f"\n{'='*60}")
        print(f"  GROUP ABLATION SUMMARY (MMAU-{args.split})")
        print(f"{'='*60}")
        print(f"  {'G':>4}  {'K':>4}  {'MMAU (%)':>10}  {'Cos Sim':>10}  Model")
        print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*30}")
        for entry in group_ablation_results:
            acc_str = f"{entry['mmau_accuracy']:.1f}" if entry['mmau_accuracy'] is not None else "SKIP"
            cos_str = f"{entry['mean_offdiag_cosine_sim']:.4f}" if entry['mean_offdiag_cosine_sim'] is not None else "N/A"
            print(f"  {entry['G']:>4}  {entry['K']:>4}  {acc_str:>10}  {cos_str:>10}  {entry['model_id']}")
        print(f"{'='*60}")

        # Save combined summary
        summary_path = os.path.join(args.output_dir, f"group_ablation_summary_{args.split}_{snr_tag}.json")
        with open(summary_path, "w") as f:
            json.dump({
                "base_model_id": args.model_id,
                "split": args.split,
                "configs": group_ablation_results
            }, f, indent=2)
        print(f"  Summary saved to: {summary_path}")
        return

    # =======================
    # Standard (non-group-ablation) Mode
    # =======================
    print(f"Loading DeSTA model from {args.model_id}...")
    model = DeSTA25AudioModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    results, ablation_summary, trackers = evaluate_model_on_mmau(
        model, ds, judge_tokenizer, judge_model,
        snr_db=args.snr_db, latent_configs=latent_configs
    )

    # Save results
    out_path = os.path.join(args.output_dir, f"mmau_{args.split}_{snr_tag}_latent_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.latent_ablation:
        summary_path = os.path.join(args.output_dir, f"mmau_{args.split}_{snr_tag}_latent_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "model_id": args.model_id,
                "ablation_results": ablation_summary
            }, f, indent=2)
        print(f"\nLatent ablation summary saved to: {summary_path}")

    print(f"Detailed per-item results saved to: {out_path}")


if __name__ == "__main__":
    main()
