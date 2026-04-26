import os
import json
import wave
import random
import gc
import tempfile
import atexit
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
# Use a process-unique temp file to avoid collisions when multiple evals run on the same machine
_tmp_wav_fd, TMP_WAV_PATH = tempfile.mkstemp(suffix=".wav", prefix=f"mmau_eval_pid{os.getpid()}_")
os.close(_tmp_wav_fd)  # close fd, we'll write via wave module
atexit.register(lambda: os.remove(TMP_WAV_PATH) if os.path.exists(TMP_WAV_PATH) else None)
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

    # Strategy C: datasets AudioDecoder (torchcodec, datasets >= 4.x)
    # AudioDecoder supports __getitem__ for "array" and "sampling_rate"
    # but NOT .get() or attribute access. Must use o["key"] syntax.
    elif hasattr(audio_obj, 'get_all_samples'):
        try:
            y = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj["sampling_rate"])
        except Exception:
            y, sr = None, 16000

    # Strategy D: other unknown audio objects — try multiple access patterns
    else:
        def _safe_get_sr(o):
            """Extract sampling_rate via __getitem__, .get(), or getattr()."""
            for accessor in [
                lambda: int(o["sampling_rate"]),
                lambda: int(o.get("sampling_rate", 16000)),
                lambda: int(getattr(o, "sampling_rate", 16000)),
            ]:
                try:
                    return accessor()
                except Exception:
                    continue
            return 16000

        for extractor in [
            # D1: dict-like ["array"] access
            lambda o: (np.asarray(o["array"], dtype=np.float32), _safe_get_sr(o)),
            # D2: attribute .array access
            lambda o: (np.asarray(o.array, dtype=np.float32), _safe_get_sr(o)),
            # D3: numpy() for torch tensors
            lambda o: (o.numpy().astype(np.float32) if hasattr(o, 'numpy') else None, 16000),
        ]:
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

def extract_answer_choice(response, choices=None):
    """
    Robustly extract answer choice from model response.
    Handles: closed <think>...</think>, unclosed <think> (truncated), and choice-matching fallback.
    """
    if not response:
        return None

    pred = response.strip()

    # --- Step 1: Separate thinking content from answer content ---
    think_content = ""

    # Remove closed <think>...</think> blocks
    pred_clean = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()

    # Handle unclosed <think> (model ran out of tokens mid-thinking)
    if '<think>' in pred_clean:
        idx = pred_clean.index('<think>')
        think_content = pred_clean[idx + 7:]  # content after <think>
        pred_clean = pred_clean[:idx].strip()  # content before <think>

    # Also capture think content from closed blocks for fallback
    think_match = re.search(r'<think>(.*?)</think>', pred, flags=re.DOTALL)
    if think_match and not think_content:
        think_content = think_match.group(1)

    # --- Step 2: Try standard answer patterns on clean text ---
    answer_patterns = [
        r'The correct answer is:\s*["\']?(.*?)["\']?\s*$',
        r'Final [Aa]nswer:\s*["\']?(.*?)["\']?\s*$',
        r'[Aa]nswer:\s*["\']?(.*?)["\']?\s*$',
        r'Option\s*([A-D])',
    ]

    extracted = None
    for text_to_search in [pred_clean, think_content]:
        if not text_to_search:
            continue
        for pat in answer_patterns:
            match = re.search(pat, text_to_search, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                break
        if extracted:
            break

    # --- Step 3: Look for A/B/C/D letter patterns ---
    if not extracted:
        for text_to_search in [pred_clean, think_content]:
            if not text_to_search:
                continue
            paren_match = re.findall(r'\(([A-D])\)', text_to_search)
            if paren_match:
                extracted = paren_match[-1]
                break
            last_char_match = re.search(r'\b([A-D])\b[. ]*$', text_to_search)
            if last_char_match:
                extracted = last_char_match.group(1)
                break

    # --- Step 3.5: Short-text / prefix matching against choices ---
    # e.g. pred="b" → "Bonfire", pred="bon" → "Bonfire"
    if extracted and choices and len(extracted) <= 3:
        _choices = choices
        if isinstance(_choices, str):
            try:
                _choices = json.loads(_choices)
            except Exception:
                _choices = []
        extracted_lower = extracted.lower()
        prefix_matches = [c for c in _choices if c.lower().startswith(extracted_lower)]
        if len(prefix_matches) == 1:
            extracted = prefix_matches[0]

    # Also try when extracted is None but pred_clean is very short
    if not extracted and choices and pred_clean and len(pred_clean) <= 3:
        _choices = choices
        if isinstance(_choices, str):
            try:
                _choices = json.loads(_choices)
            except Exception:
                _choices = []
        pred_lower = pred_clean.lower()
        prefix_matches = [c for c in _choices if c.lower().startswith(pred_lower)]
        if len(prefix_matches) == 1:
            extracted = prefix_matches[0]

    # --- Step 4: Choice-matching fallback (for truncated thinking) ---
    # If we have choices and thinking content but no extracted answer,
    # find which choice the model concluded on by looking at the last portion
    if not extracted and choices and think_content:
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except Exception:
                choices = []

        # Use the last 500 chars of thinking where conclusion usually appears
        tail = think_content[-500:].lower()

        # Score each choice: count occurrences + bonus for appearing near the end
        best_choice = None
        best_score = 0
        for choice in choices:
            choice_lower = choice.lower()
            tokens = set(re.findall(r'\b\w+\b', choice_lower))
            # Count token matches in tail
            score = sum(1 for t in tokens if t in tail)
            # Bonus: exact choice string near end (last 200 chars)
            if choice_lower in think_content[-200:].lower():
                score += 3
            # Bonus: bolded/emphasized choice (e.g., **Person**)
            if re.search(r'\*\*' + re.escape(choice_lower) + r'\*\*', tail):
                score += 5
            if score > best_score:
                best_score = score
                best_choice = choice

        if best_choice and best_score >= 2:
            extracted = best_choice

    # --- Step 5: Final fallback ---
    if not extracted:
        extracted = pred_clean if pred_clean else think_content if think_content else pred

    return extracted.strip('"').strip("'").strip() if extracted else None

# =====================
# DeSTA 推論
# =====================

SYSTEM_PROMPT = 'Focus on the audio clips and instructions. Provide your answer by first thinking in <think> tags if needed, and then ending with "The correct answer is: \\"___\\" " where ___ is the exact choice from the list.'

DEFAULT_MAX_NEW_TOKENS = 512


def _build_question(item):
    choices = item["choices"]
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

    return f"{item['question']} Choose from the following options: {options_str}"


def _run_desta_generate(model, wav_path, question, transcription=None,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, choices=None):
    """
    Core DeSTA generate call. If transcription is provided, ASR is skipped.
    choices is passed to extract_answer_choice for fallback matching.
    """
    audio_entry = {"audio": wav_path}
    if transcription is not None:
        audio_entry["text"] = transcription

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<|AUDIO|>\n\n{question}",
            "audios": [audio_entry]
        }
    ]

    with torch.no_grad():
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            top_p=0.85,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    pred = outputs.text
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, str):
        pred = pred.strip()
    return extract_answer_choice(pred, choices=choices)


def _suppress_whisper_max_length_warning(model):
    """
    Whisper's generation_config has max_length=448 by default.
    When we also pass max_new_tokens, HF prints a noisy warning.
    Clear max_length once to suppress it.
    """
    whisper = model.perception.whisper
    if hasattr(whisper, 'generation_config') and whisper.generation_config.max_length is not None:
        whisper.generation_config.max_length = None


def _get_cached_transcription(model, wav_path):
    """
    Run VAD + ASR on an audio file once and return the transcription string.
    This avoids re-running Whisper for every repeated call on the same audio.
    """
    from desta.utils.audio import AudioSegment as DeSTAAudioSegment
    feature = DeSTAAudioSegment.from_file(wav_path, target_sr=16000, channel_selector="average").samples
    model._setup_vad()
    is_speech = model.get_speech_timestamps(feature, model.vad_model)
    if not is_speech:
        return " "

    if not hasattr(model, "processor"):
        model._setup_generation()
    _suppress_whisper_max_length_warning(model)
    feats = model.processor([feature], sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(model.device).half()
    with torch.no_grad():
        trans_ids = model.perception.whisper.generate(
            input_features=feats, attention_mask=None,
            max_new_tokens=128,
        )
    transcription = model.processor.batch_decode(trans_ids, skip_special_tokens=True)[0].strip()
    return transcription


def run_desta_on_item(model, item, wav_path=TMP_WAV_PATH, snr_db=None,
                      transcription=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                      _wav_written=False):
    if not _wav_written:
        write_wav_from_dataset_item(item, wav_path, snr_db=snr_db)
    question = _build_question(item)
    choices = item.get("choices") if isinstance(item, dict) else getattr(item, "choices", None)
    return _run_desta_generate(model, wav_path, question, transcription=transcription,
                               max_new_tokens=max_new_tokens, choices=choices)


def run_all_latent_configs(model, item, latent_configs, wav_path=TMP_WAV_PATH,
                           snr_db=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """
    Run all latent configs for a single item efficiently:
    - Write WAV once
    - Compute ASR transcription once, reuse for all subsequent calls
    - Group configs by tau to avoid redundant samples

    Returns: dict mapping cfg -> (final_pred, sample_preds)
    """
    # Write WAV once for this item
    write_wav_from_dataset_item(item, wav_path, snr_db=snr_db)
    question = _build_question(item)
    choices = item.get("choices") if isinstance(item, dict) else getattr(item, "choices", None)

    connector = model.perception.connector
    is_variational = getattr(connector, 'variational_enabled', False)

    # If not variational, run once and return for all configs
    if not is_variational:
        pred = _run_desta_generate(model, wav_path, question, max_new_tokens=max_new_tokens, choices=choices)
        return {cfg: (pred, [pred]) for cfg in latent_configs}

    # Group configs by tau, find max S needed per tau
    from collections import OrderedDict
    tau_max_s = OrderedDict()
    for tau, S in latent_configs:
        if tau not in tau_max_s or S > tau_max_s[tau]:
            tau_max_s[tau] = S

    # Run ASR once (first call without cached transcription), then cache it
    cached_transcription = None
    tau_samples = {}  # tau -> list of predictions

    for tau, max_s in tau_max_s.items():
        original_alpha = connector.s1_inference_alpha
        connector.s1_inference_alpha = tau

        predictions = []
        for s in range(max_s):
            if cached_transcription is None:
                # First call ever: no cached transcription, ASR will run
                pred = _run_desta_generate(model, wav_path, question,
                                           max_new_tokens=max_new_tokens, choices=choices)
                # Cache transcription for all subsequent calls
                cached_transcription = _get_cached_transcription(model, wav_path)
            else:
                # Subsequent calls: pass cached transcription to skip ASR
                pred = _run_desta_generate(model, wav_path, question,
                                           transcription=cached_transcription,
                                           max_new_tokens=max_new_tokens, choices=choices)
            predictions.append(pred)

        connector.s1_inference_alpha = original_alpha
        tau_samples[tau] = predictions

    # Build results for each config
    results = {}
    for cfg in latent_configs:
        tau, S = cfg
        preds = tau_samples[tau][:S]
        vote_counts = Counter(p for p in preds if p is not None)
        if vote_counts:
            final_pred = vote_counts.most_common(1)[0][0]
        else:
            final_pred = preds[0] if preds else None
        results[cfg] = (final_pred, preds)

    return results


def run_desta_with_latent_sampling(model, item, num_samples, tau,
                                   wav_path=TMP_WAV_PATH, snr_db=None,
                                   max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """
    Run DeSTA inference S times with latent sampling.
    For multi-config evaluation, prefer run_all_latent_configs() instead.
    """
    connector = model.perception.connector

    if not getattr(connector, 'variational_enabled', False):
        pred = run_desta_on_item(model, item, wav_path, snr_db=snr_db,
                                 max_new_tokens=max_new_tokens)
        return pred, [pred]

    original_alpha = connector.s1_inference_alpha
    connector.s1_inference_alpha = tau

    predictions = []
    for s in range(num_samples):
        pred = run_desta_on_item(model, item, wav_path, snr_db=snr_db,
                                 max_new_tokens=max_new_tokens)
        predictions.append(pred)

    connector.s1_inference_alpha = original_alpha

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


def load_judge(model_id=JUDGE_MODEL_ID, use_compile=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    if use_compile and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  Judge model compiled with torch.compile (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile failed for judge, using eager mode: {e}")
    return tokenizer, model


def _build_judge_chat_str(tokenizer, item, pred):
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

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def _parse_judge_output(raw_text):
    # Strip Qwen3 thinking tags (closed or unclosed) before parsing
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    if '<think>' in raw_text:
        raw_text = raw_text[:raw_text.index('<think>')].strip()
    raw_text = raw_text.strip().upper()
    if raw_text.startswith("CORRECT"):
        return True, raw_text
    if raw_text.startswith("INCORRECT"):
        return False, raw_text
    return None, raw_text


def call_judge(tokenizer, model, item, pred):
    chat_str = _build_judge_chat_str(tokenizer, item, pred)
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _parse_judge_output(raw_text)


def call_judge_batch(tokenizer, model, items_and_preds, batch_size=16):
    """
    Batched judge inference. Returns list of (is_correct, raw_text) tuples.
    """
    results = []
    chat_strs = [_build_judge_chat_str(tokenizer, item, pred) for item, pred in items_and_preds]

    for i in range(0, len(chat_strs), batch_size):
        batch_strs = chat_strs[i:i + batch_size]
        inputs = tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        input_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0
            )

        for j in range(len(batch_strs)):
            gen_ids = output_ids[j][input_lengths[j]:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(_parse_judge_output(raw_text))

    return results


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
                           latent_configs=None, judge_batch_size=16,
                           max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """
    Run MMAU evaluation for a single model. Returns (results, ablation_summary, trackers).

    Two-pass strategy for speed:
      Pass 1 – Run DeSTA inference + string_match on all items (GPU-bound on DeSTA).
               Uses run_all_latent_configs() to write WAV once, cache ASR, and
               share samples across configs with the same tau.
      Pass 2 – Batch judge only the items where string_match failed.
    """
    if latent_configs is None:
        latent_configs = [(0.0, 1)]

    # Pre-compute total inference calls for progress info
    tau_max_s = {}
    for tau, S in latent_configs:
        if tau not in tau_max_s or S > tau_max_s[tau]:
            tau_max_s[tau] = S
    total_calls_per_item = sum(tau_max_s.values())
    print(f"  Latent configs: {latent_configs}")
    print(f"  Unique (tau, max_S): {dict(tau_max_s)} → {total_calls_per_item} inference calls/item "
          f"(was {sum(S for _, S in latent_configs)} without sharing)")

    # --- Pass 1: DeSTA inference + string_match ---
    raw_results = []

    for idx, item in enumerate(tqdm(ds, desc="DeSTA inference")):
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

        entry = {
            "item": item,
            "answer": answer,
            "task": task,
            "difficulty": difficulty,
            "subcat": subcat,
            "choices": choices,
            "configs": {}
        }

        # Run all latent configs efficiently for this item
        all_cfg_results = run_all_latent_configs(
            model, item, latent_configs, wav_path=TMP_WAV_PATH,
            snr_db=snr_db, max_new_tokens=max_new_tokens
        )

        for cfg in latent_configs:
            tau, S = cfg
            pred, sample_preds = all_cfg_results[cfg]

            is_string_correct = string_match(answer, pred, choices)
            entry["configs"][cfg] = {
                "pred": pred,
                "sample_preds": sample_preds,
                "is_string_correct": is_string_correct,
            }

            # Print every prediction for debugging
            match_str = "✓" if is_string_correct else "✗"
            print(f"  [{idx}] tau={tau} S={S} | {match_str} | Ans: {answer} | Pred: {pred}")
            if S > 1:
                print(f"         samples: {sample_preds}")

        raw_results.append(entry)

    # --- Pass 2: Batched judge for items where string_match failed ---
    # Collect (index, cfg) pairs that need judging
    judge_requests = []  # (result_idx, cfg, item, pred)
    for i, entry in enumerate(raw_results):
        for cfg, cfg_data in entry["configs"].items():
            if not cfg_data["is_string_correct"]:
                judge_requests.append((i, cfg, entry["item"], cfg_data["pred"]))

    judge_results_map = {}  # (i, cfg) -> (is_correct, raw_text)
    if judge_requests and judge_model is not None:
        print(f"Running batched judge on {len(judge_requests)} items (string_match failed)...")
        items_and_preds = [(item, pred) for _, _, item, pred in judge_requests]
        judge_outputs = call_judge_batch(judge_tokenizer, judge_model, items_and_preds, batch_size=judge_batch_size)
        for (i, cfg, _, _), output in zip(judge_requests, judge_outputs):
            judge_results_map[(i, cfg)] = output
    elif judge_requests:
        print(f"Skipping judge for {len(judge_requests)} items (--no_judge mode).")

    # --- Assemble final results and trackers ---
    trackers = {cfg: {
        "total": 0, "corr": 0,
        "task": defaultdict(lambda: [0, 0]),
        "diff": defaultdict(lambda: [0, 0]),
        "subcat": defaultdict(lambda: [0, 0])
    } for cfg in latent_configs}

    results = []
    for i, entry in enumerate(raw_results):
        item_result = {
            "id": entry["item"]["id"],
            "question": entry["item"]["question"],
            "answer": entry["answer"],
            "task": entry["task"],
            "difficulty": entry["difficulty"],
            "subcat": entry["subcat"],
            "configs": {}
        }

        for cfg in latent_configs:
            tau, S = cfg
            cfg_data = entry["configs"][cfg]
            pred = cfg_data["pred"]
            is_string_correct = cfg_data["is_string_correct"]

            if is_string_correct:
                is_llm_correct = None
            else:
                is_llm_correct, _ = judge_results_map.get((i, cfg), (None, ""))

            is_correct = is_string_correct or bool(is_llm_correct)

            tr = trackers[cfg]
            task = entry["task"]
            difficulty = entry["difficulty"]
            subcat = entry["subcat"]
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
                "sample_predictions": cfg_data["sample_preds"]
            }

            if tau == 0.0 and S == 1:
                print(f"Match: {is_string_correct}, Judge: {is_llm_correct}, Ans: {entry['answer']}, Pred: {pred}")

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
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip LLM judge entirely; use only string_match for scoring (much faster).")
    parser.add_argument("--judge_batch_size", type=int, default=16,
                        help="Batch size for LLM judge inference (default: 16)")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Max new tokens for DeSTA generation (default: {DEFAULT_MAX_NEW_TOKENS})")
    args = parser.parse_args()

    # Set global random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    snr_tag = "clean" if args.snr_db is None else f"snr{int(args.snr_db)}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Judge (skip if --no_judge)
    judge_tokenizer, judge_model = None, None
    if not args.no_judge:
        print(f"Loading Judge model {JUDGE_MODEL_ID}...")
        judge_tokenizer, judge_model = load_judge()
    else:
        print("Judge model disabled (--no_judge). Using string_match only.")

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
                model = DeSTA25AudioModel.from_pretrained(model_id)
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
                snr_db=args.snr_db, latent_configs=[(0.0, 1)],
                judge_batch_size=args.judge_batch_size,
                max_new_tokens=args.max_new_tokens
            )
            baseline_acc = ablation_summary[0]["accuracy"]

            # Compute query diagnostics
            connector_mode = model.config.connector_mode
            mean_cos_sim = None
            if connector_mode in ("orca_desta", "orca_r1"):
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
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    results, ablation_summary, trackers = evaluate_model_on_mmau(
        model, ds, judge_tokenizer, judge_model,
        snr_db=args.snr_db, latent_configs=latent_configs,
        judge_batch_size=args.judge_batch_size,
        max_new_tokens=args.max_new_tokens
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
