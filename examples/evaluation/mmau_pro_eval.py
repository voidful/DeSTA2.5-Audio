"""
MMAU-Pro Evaluation Script for DeSTA2.5-Audio

This script evaluates DeSTA2.5-Audio models on the MMAU-Pro benchmark.
MMAU-Pro contains 3 types of questions with different evaluation methods:
- Open-ended questions: LLM-as-a-judge evaluation
- Instruction following questions: Regex/constraint-based evaluation
- Closed-ended (MCQ) questions: NVEmbed similarity + LLM judge

Dataset: gamma-lab-umd/MMAU-Pro
Reference: https://arxiv.org/abs/2508.13992

Usage:
    python mmau_pro_eval.py --model_id <model_path> --split test --max_samples 100
"""

import os
import json
import wave
import random
import numpy as np
import re
import argparse
from tqdm import tqdm
import tempfile

import torch
import torch.nn.functional as F
from datasets import load_dataset
from desta import DeSTA25AudioModel
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

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
# Configuration
# =====================

DEFAULT_MODEL_ID = "voidful/QAQ_0.6b_orca_all"
DATASET_ID = "gamma-lab-umd/MMAU-Pro"
DEFAULT_SPLIT = "test"
TMP_WAV_PATH = "tmp_mmau_pro_audio.wav"
RESULT_DIR = "mmau_pro_results"
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
NVEMBED_MODEL_ID = "nvidia/NV-Embed-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Audio Utilities
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


def write_wav_from_dataset_item(item, wav_path, audio_key="audio"):
    """
    Extract audio from dataset item and write to WAV file.
    Handles both single audio and multi-audio samples.
    """
    audio_obj = item.get(audio_key)
    if audio_obj is None:
        # Try alternative keys
        for key in ["audio", "audios", "audio_path"]:
            if key in item and item[key] is not None:
                audio_obj = item[key]
                break
    
    if audio_obj is None:
        raise ValueError(f"No audio found in item: {list(item.keys())}")

    # Handle single audio (legacy dict format)
    if isinstance(audio_obj, dict) and "array" in audio_obj:
        audio_array = audio_obj["array"]
        sample_rate = audio_obj.get("sampling_rate", 16000)
        if sample_rate is None:
            sample_rate = 16000
        return write_wav_from_array(audio_array, sample_rate, wav_path)

    # Handle AudioDecoder (torchcodec, datasets >= 4.x)
    if hasattr(audio_obj, 'get_all_samples'):
        audio_array = np.asarray(audio_obj["array"], dtype=np.float32)
        sample_rate = int(audio_obj["sampling_rate"])
        return write_wav_from_array(audio_array, sample_rate, wav_path)

    # Handle list of audios (multi-audio)
    if isinstance(audio_obj, list) and len(audio_obj) > 0:
        all_audio = []
        sample_rate = 16000
        for audio in audio_obj:
            if isinstance(audio, dict) and "array" in audio:
                all_audio.append(audio["array"])
                sample_rate = audio.get("sampling_rate", sample_rate) or sample_rate
            elif hasattr(audio, 'get_all_samples'):
                all_audio.append(np.asarray(audio["array"], dtype=np.float32))
                sample_rate = int(audio["sampling_rate"])
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            return write_wav_from_array(combined_audio, sample_rate, wav_path)

    raise ValueError(f"Unsupported audio format: {type(audio_obj)}")


# =====================
# Instruction Following Evaluation Functions
# =====================

def count_words(text):
    return len(text.split())

def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def count_paragraphs(text):
    paragraphs = text.split("***")
    return len([p for p in paragraphs if p.strip()])

def count_bullet_points(text):
    bullets = re.findall(r'(?:^|\n)\s*\*\s+', text)
    return len(bullets)

def count_highlighted_sections(text):
    highlights = re.findall(r'\*([^*]+)\*', text)
    return len(highlights)

def count_placeholders(text):
    placeholders = re.findall(r'\[[^\]]+\]', text)
    return len(placeholders)

def count_capital_words(text):
    words = text.split()
    return len([w for w in words if w.isupper()])

def count_keyword_frequency(text, keyword):
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    return len(re.findall(pattern, text.lower()))

def has_title(text):
    return bool(re.search(r'<<[^>]+>>', text))

def has_postscript(text, marker):
    text_alpha = re.sub(r'[^a-zA-Z]', '', text).lower()
    marker_alpha = re.sub(r'[^a-zA-Z]', '', marker).lower()
    return marker_alpha in text_alpha

def starts_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.startswith(phrase_alpha)

def ends_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.endswith(phrase_alpha)

def is_wrapped_in_quotes(text):
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')

def has_no_commas(text):
    return ',' not in text

def check_sections(text, num_sections, splitter):
    escaped_splitter = re.escape(splitter)
    sections = re.split(rf'\s*{escaped_splitter}\s*', text.strip())
    actual_sections = [s for s in sections if s.strip()]
    return len(actual_sections) == num_sections


def evaluate_aif_sample(response, sample_data):
    """Evaluate Audio Instruction Following sample based on task_identifier and kwargs."""
    task_identifier = sample_data.get("task_identifier", "")
    kwargs = sample_data.get("kwargs", {}) or {}
    
    # Convert kwargs from string if needed
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except:
            kwargs = {}

    success = False

    if task_identifier == "Include Keywords":
        keywords = kwargs.get("keywords", "").split(", ")
        success = all(keyword.lower() in response.lower() for keyword in keywords if keyword)

    elif task_identifier == "Keyword Frequency":
        keyword = kwargs.get("keyword", "")
        target = kwargs.get("N", 0)
        actual = count_keyword_frequency(response, keyword)
        success = actual == target

    elif task_identifier == "Forbidden Words":
        forbidden_words = kwargs.get("forbidden_words", "").split(", ")
        success = not any(word.lower() in response.lower() for word in forbidden_words if word)

    elif task_identifier == "Number Paragraphs":
        target = kwargs.get("N", 0)
        actual = count_paragraphs(response)
        success = actual == target

    elif task_identifier == "Number Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual >= target

    elif task_identifier == "Number Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual <= target

    elif task_identifier == "Number Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_words(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Number Sentences (at least)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual >= target

    elif task_identifier == "Number Sentences (at most)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual <= target

    elif task_identifier == "Number Sentences (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_sentences(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Postscript":
        marker = kwargs.get("postscript_marker", "")
        success = has_postscript(response, marker)

    elif task_identifier == "Number Placeholder":
        target = kwargs.get("N", 0)
        actual = count_placeholders(response)
        success = actual >= target

    elif task_identifier == "Number Bullets":
        target = kwargs.get("N", 0)
        actual = count_bullet_points(response)
        success = actual == target

    elif task_identifier == "Title":
        success = has_title(response)

    elif task_identifier == "Minimum Number Highlighted Section":
        target = kwargs.get("N", 0)
        actual = count_highlighted_sections(response)
        success = actual >= target

    elif task_identifier == "Multiple Sections":
        target = kwargs.get("N", 0)
        splitter = kwargs.get("section_splitter", "")
        success = check_sections(response, target, splitter)

    elif task_identifier == "Repeat Prompt":
        original_prompt = sample_data.get("prompt_transcription", "")
        success = response.strip().lower().startswith(original_prompt.strip().lower())

    elif task_identifier == "Two Responses":
        separator = "******"
        parts = response.split(separator)
        success = len(parts) == 2 and parts[0].lower().strip() != parts[1].lower().strip()

    elif task_identifier == "All Uppercase":
        success = response.isupper()

    elif task_identifier == "All Lowercase":
        success = response.islower()

    elif task_identifier == "All-capital Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual >= target

    elif task_identifier == "All-capital Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual <= target

    elif task_identifier == "All-capital Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_capital_words(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Start Checker":
        phrase = kwargs.get("start_phrase", "")
        success = starts_with_phrase(response, phrase)

    elif task_identifier == "End Checker":
        phrase = kwargs.get("end_phrase", "")
        success = ends_with_phrase(response, phrase)

    elif task_identifier == "Quotation":
        success = is_wrapped_in_quotes(response)

    elif task_identifier == "No Commas":
        success = has_no_commas(response)

    else:
        # Unknown task identifier, default to False
        logging.warning(f"Unknown task_identifier: {task_identifier}")
        success = False

    return success


# =====================
# LLM Judge for Open-ended
# =====================

OPENENDED_JUDGE_PROMPT = """You are an expert evaluator for audio understanding tasks.

Question: {question}
Reference Answer: {reference}
Model Response: {response}

Evaluate the model response on these criteria (1-5 scale, 5 is best):

1. **Correctness**: How factually accurate compared to reference?
2. **Relevance**: How well does it address the question?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: How clear and well-structured?

Format your response EXACTLY as:
CORRECTNESS: [score]
RELEVANCE: [score]
COMPLETENESS: [score]
CLARITY: [score]
OVERALL: [average]
"""


def load_judge(model_id=JUDGE_MODEL_ID):
    """Load LLM judge model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def evaluate_openended(judge_tokenizer, judge_model, question, reference, response):
    """Evaluate open-ended response using LLM judge."""
    prompt = OPENENDED_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        response=response
    )

    messages = [
        {"role": "system", "content": "You are a careful evaluator for audio QA outputs."},
        {"role": "user", "content": prompt}
    ]

    chat_str = judge_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = judge_tokenizer(chat_str, return_tensors="pt").to(judge_model.device)

    with torch.no_grad():
        output_ids = judge_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = judge_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Extract scores
    scores = {}
    patterns = {
        'correctness': r'CORRECTNESS:\s*(\d+)',
        'relevance': r'RELEVANCE:\s*(\d+)',
        'completeness': r'COMPLETENESS:\s*(\d+)',
        'clarity': r'CLARITY:\s*(\d+)',
        'overall': r'OVERALL:\s*(\d+(?:\.\d+)?)'
    }

    for criterion, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
        else:
            scores[criterion] = 3.0  # Default neutral score

    # Calculate overall if not found
    if scores.get('overall', 3.0) == 3.0:
        criteria_scores = [scores.get(k, 3.0) for k in ['correctness', 'relevance', 'completeness', 'clarity']]
        scores['overall'] = np.mean(criteria_scores)

    return scores, raw_text


# =====================
# NVEmbed for Closed-ended
# =====================

def load_nvembed(model_id=NVEMBED_MODEL_ID):
    """Load NVEmbed model for similarity matching."""
    print(f"Loading NVEmbed model {model_id}...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def evaluate_closedended_nvembed(nvembed_model, prediction, choices):
    """
    Match prediction to choices using embedding similarity.
    Returns the matched choice and confidence score.
    """
    # Encode prediction
    pred_embedding = nvembed_model.encode([prediction], instruction="", max_length=4096)
    pred_embedding = F.normalize(pred_embedding, p=2, dim=1)

    # Encode choices
    choice_embeddings = nvembed_model.encode(choices, instruction="", max_length=4096)
    choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

    # Calculate similarity
    scores = (pred_embedding @ choice_embeddings.T) * 100
    scores = scores.squeeze()

    best_idx = torch.argmax(scores).item()
    matched_choice = choices[best_idx]
    confidence = torch.max(scores).item()

    return matched_choice, confidence


# =====================
# String Match (for MCQ fallback)
# =====================

def string_match(answer, prediction, choices):
    """String matching for MCQ evaluation (from mmau_eval.py)."""
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
# DeSTA Inference
# =====================

def run_desta_inference(model, item, category, wav_path=TMP_WAV_PATH):
    """
    Run DeSTA inference on a single sample.
    Returns the text prediction.
    """
    write_wav_from_dataset_item(item, wav_path)

    question = item.get("question", "")
    choices = item.get("choices", [])
    
    # Convert choices from string if needed
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except:
            choices = []

    # Build prompt based on category
    # Build prompt based on category
    if category in ["open", "instruction following"]:
        # Open-ended or instruction following: direct question
        # Match training format: NO system prompt
        # system_prompt = "You are a helpful audio assistant. Focus on the audio and provide a helpful, complete answer."
        user_content = f"<|AUDIO|>\n\n{question.strip()}"
    else:
        # Closed-ended (MCQ) alignment
        # Match training format: NO system prompt
        # system_prompt = "You are a helpful audio assistant. Select the correct option."
        
        # Format choices robustly
        choice_text = ""
        if choices:
            choice_text = " Choose from the following options: "
            for i, option in enumerate(choices):
                choice_text += f'"{option}"'
                if i == len(choices) - 2:
                    choice_text += " or "
                else:
                    choice_text += ", "
            choice_text = choice_text.rstrip(", ")
            
            user_content = f"<|AUDIO|>\n\n{question.strip().replace('<|AUDIO|>', '')} {choice_text}"
        else:
             user_content = f"<|AUDIO|>\n\n{question.strip().replace('<|AUDIO|>', '')}"
    
    system_prompt = 'Focus on the audio clips and instructions. Provide your answer by first thinking in <think> tags if needed, and then ending with "The correct answer is: "___" " where ___ is the exact choice from the list.'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content, "audios": [{"audio": wav_path}]}
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                max_new_tokens=512,
                repetition_penalty=1.2
            )

    pred = outputs.text[0] if isinstance(outputs.text, list) else outputs.text
    
    # Clean prediction for MCQ
    if category not in ["open", "instruction following"]:
        if isinstance(pred, str):
            # Remove thinking process
            pred_no_think = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
            
            # Extract answer
            match = re.search(r'The correct answer is:\s*["\']?(.*?)["\']?$', pred_no_think, re.IGNORECASE)
            if match:
                pred = match.group(1).strip().strip('"').strip("'")
            else:
                pred = pred_no_think

    return str(pred) if pred else ""


# =====================
# Main Evaluation
# =====================

def main():
    parser = argparse.ArgumentParser(description="Run MMAU-Pro evaluation with DeSTA2.5-Audio")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--skip_nvembed", action="store_true", help="Skip NVEmbed for MCQ, use string match only")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    # Set global random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DeSTA model
    print(f"Loading DeSTA model from {args.model_id}...")
    desta_model = DeSTA25AudioModel.from_pretrained(args.model_id)
    desta_model.to(device)
    desta_model.eval()

    # Load Judge model
    print(f"Loading Judge model {JUDGE_MODEL_ID}...")
    judge_tokenizer, judge_model = load_judge()

    # Load NVEmbed (optional)
    nvembed_model = None
    if not args.skip_nvembed:
        try:
            nvembed_model = load_nvembed()
        except Exception as e:
            print(f"Warning: Could not load NVEmbed model: {e}")
            print("Falling back to string matching for MCQ evaluation.")

    # Load dataset
    print(f"Loading dataset {DATASET_ID} split {args.split}...")
    ds = load_dataset(DATASET_ID, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Initialize metrics
    category_results = {
        "open": {"correct": 0, "total": 0, "scores": []},
        "instruction following": {"correct": 0, "total": 0},
        "closed": {"correct": 0, "total": 0}
    }
    subcategory_metrics = {}
    results = []

    print(f"\nEvaluating {len(ds)} samples...")

    for idx, item in enumerate(tqdm(ds, desc="Evaluating")):
        try:
            # Get category
            category = item.get("category", "closed").lower()
            if category not in category_results:
                category = "closed"  # Default to closed-ended

            # Get metadata
            answer = item.get("answer", "")
            question = item.get("question", "")
            choices = item.get("choices", [])
            if isinstance(choices, str):
                try:
                    choices = json.loads(choices)
                except:
                    choices = []
            
            subcategory = item.get("sub-category", item.get("subcategory", "NA"))

            # Run inference
            pred = run_desta_inference(desta_model, item, category, TMP_WAV_PATH)

            # Evaluate based on category
            is_correct = False
            eval_details = {}

            if category == "open":
                # LLM judge evaluation
                scores, raw_eval = evaluate_openended(judge_tokenizer, judge_model, question, answer, pred)
                category_results["open"]["scores"].append(scores)
                is_correct = scores.get("overall", 0) >= 3.5  # Threshold for "correct"
                eval_details = {"scores": scores, "raw_eval": raw_eval}

            elif category == "instruction following":
                # Constraint-based evaluation
                sample_data = {
                    "task_identifier": item.get("task_identifier", ""),
                    "kwargs": item.get("kwargs", {}),
                    "prompt_transcription": question
                }
                is_correct = evaluate_aif_sample(pred, sample_data)
                eval_details = {"task_identifier": sample_data["task_identifier"]}

            else:
                # Closed-ended (MCQ) evaluation
                if nvembed_model is not None and choices:
                    matched_choice, confidence = evaluate_closedended_nvembed(nvembed_model, pred, choices)
                    is_correct = matched_choice == answer
                    eval_details = {"matched_choice": matched_choice, "confidence": confidence}
                else:
                    # Fallback to string matching
                    is_correct = string_match(answer, pred, choices) if choices else (answer.lower() in pred.lower())
                    eval_details = {"method": "string_match"}

            # Update metrics
            category_results[category]["total"] += 1
            if is_correct:
                category_results[category]["correct"] += 1

            if subcategory not in subcategory_metrics:
                subcategory_metrics[subcategory] = [0, 0]
            subcategory_metrics[subcategory][1] += 1
            if is_correct:
                subcategory_metrics[subcategory][0] += 1

            # Store result
            results.append({
                "id": item.get("id", idx),
                "category": category,
                "subcategory": subcategory,
                "question": question,
                "answer": answer,
                "prediction": pred,
                "is_correct": is_correct,
                "eval_details": eval_details
            })

            # Print progress
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx + 1}/{len(ds)}")
                for cat, data in category_results.items():
                    if data["total"] > 0:
                        acc = data["correct"] / data["total"] * 100
                        print(f"  {cat}: {acc:.1f}% ({data['correct']}/{data['total']})")

        except Exception as e:
            logging.error(f"Error processing sample {idx}: {e}")
            results.append({
                "id": item.get("id", idx),
                "error": str(e)
            })
            continue

    # =====================
    # Print Final Results
    # =====================
    print("\n" + "=" * 60)
    print("MMAU-Pro EVALUATION RESULTS")
    print("=" * 60)

    total_correct = 0
    total_samples = 0

    print("\nCategory-wise Performance:")
    print("-" * 40)
    for cat, data in category_results.items():
        if data["total"] > 0:
            acc = data["correct"] / data["total"] * 100
            print(f"{cat:25s}: {acc:6.2f}% ({data['correct']:4d}/{data['total']:4d})")
            total_correct += data["correct"]
            total_samples += data["total"]

            # For open-ended, also show average scores
            if cat == "open" and data["scores"]:
                avg_overall = np.mean([s.get("overall", 0) for s in data["scores"]])
                print(f"  Average LLM Judge Score: {avg_overall:.2f}/5.0")

    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print("-" * 40)
        print(f"{'Overall':25s}: {overall_acc:6.2f}% ({total_correct:4d}/{total_samples:4d})")

    print("\nSub-category-wise Performance:")
    print("-" * 40)
    for subcat, (corr, tot) in sorted(subcategory_metrics.items()):
        if tot > 0:
            acc = corr / tot * 100
            print(f"{subcat:35s}: {acc:6.2f}% ({corr:4d}/{tot:4d})")

    # Save results
    out_path = os.path.join(args.output_dir, f"mmau_pro_{args.split}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nDetailed results saved to: {out_path}")

    # Save summary
    summary = {
        "model_id": args.model_id,
        "split": args.split,
        "total_samples": total_samples,
        "overall_accuracy": overall_acc if total_samples > 0 else 0,
        "category_results": {
            cat: {
                "accuracy": data["correct"] / data["total"] * 100 if data["total"] > 0 else 0,
                "correct": data["correct"],
                "total": data["total"]
            }
            for cat, data in category_results.items()
            if data["total"] > 0
        },
        "subcategory_results": {
            subcat: {"accuracy": corr / tot * 100, "correct": corr, "total": tot}
            for subcat, (corr, tot) in subcategory_metrics.items()
            if tot > 0
        }
    }

    summary_path = os.path.join(args.output_dir, f"mmau_pro_{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Cleanup temp files
    if os.path.exists(TMP_WAV_PATH):
        os.remove(TMP_WAV_PATH)


if __name__ == "__main__":
    main()
