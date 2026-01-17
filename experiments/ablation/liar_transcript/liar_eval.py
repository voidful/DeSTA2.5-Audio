"""
P0-2: Liar Transcript Evaluation (Optimized for H100)

Evaluates model performance when given intentionally contradictory transcripts.

The test: Model receives audio with a WRONG transcript and must answer correctly
based on what it HEARS, not what the transcript says.

Expected results:
- DeSTA (ASR-reliant): ~35% accuracy (follows wrong transcript)
- ORCA (audio-aware): ~70% accuracy (uses actual audio info)
"""

import os
import json
import argparse
import logging
import wave
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import AutoTokenizer, AutoProcessor

# Try to import DeSTA
try:
    from desta import DeSTA25AudioModel
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from desta import DeSTA25AudioModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    import re
    answer = str(answer).lower().strip()
    answer = re.sub(r'^(the |a |an )', '', answer)
    answer = re.sub(r'[.!?,;:]+$', '', answer)
    return answer


def answer_matches_audio_truth(prediction: str, audio_truth: str) -> bool:
    """Check if prediction matches the audio ground truth."""
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(audio_truth)
    
    if pred_norm == truth_norm:
        return True
    if truth_norm in pred_norm or pred_norm in truth_norm:
        return True
    if pred_norm.split() and truth_norm.split():
        if pred_norm.split()[0] == truth_norm.split()[0]:
            return True
    return False


def answer_matches_liar_transcript(prediction: str, liar_transcript: str) -> bool:
    """Check if prediction was fooled by the liar transcript."""
    pred_norm = normalize_answer(prediction)
    liar_norm = normalize_answer(liar_transcript)
    
    liar_words = set(liar_norm.split())
    pred_words = set(pred_norm.split())
    
    # Key contradiction words
    key_words = {"happy", "sad", "angry", "calm", "male", "female", 
                 "man", "woman", "dog", "cat", "bird", "lion", "sheep"}
    
    liar_keys = liar_words & key_words
    pred_keys = pred_words & key_words
    
    if liar_keys & pred_keys:
        return True
    return False


class LiarDataset(Dataset):
    def __init__(self, samples, tokenizer, processor, audio_locator="<|AUDIO|>", inject_transcript=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = processor
        self.audio_locator = audio_locator
        self.inject_transcript = inject_transcript

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        audio_paths = [b.get("audio_path") for b in batch]
        questions = [b["question"] for b in batch]
        audio_truths = [b["audio_ground_truth"] for b in batch]
        liar_transcripts = [b["liar_transcript"] for b in batch]
        task_types = [b["task_type"] for b in batch]
        sample_ids = [b.get("sample_id") for b in batch]
        
        # Prepare prompts
        texts = []
        for q, l_trans in zip(questions, liar_transcripts):
            if self.inject_transcript:
                content = f"{self.audio_locator}\n{q}\n\n[Transcription]: {l_trans}"
            else:
                content = f"{self.audio_locator}\n{q}"
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    txt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    txt = content
            else:
                txt = content
            texts.append(txt)
                
        # Tokenize prompts (to find start positions)
        inputs_text = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Determine start positions
        # Simple heuristic: find first audio_locator token. 
        # If tokenizer handles special tokens correctly, this works.
        # Otherwise, need to rely on adding tokens.
        audio_id = self.tokenizer.convert_tokens_to_ids(self.audio_locator)
        # If failed to map (returns unknown id usually), we might need logic.
        
        batch_start_positions = []
        for seq in inputs_text.input_ids:
            indices = (seq == audio_id).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                batch_start_positions.append(indices[0].item())
            else:
                batch_start_positions.append(0)
        
        return {
            "audio_paths": audio_paths,
            "inputs_text": inputs_text,
            "batch_start_positions": batch_start_positions,
            "audio_truths": audio_truths,
            "liar_transcripts": liar_transcripts,
            "task_types": task_types,
            "sample_ids": sample_ids
        }


def process_audio_files_batch(audio_paths, processor, device):
    """Load and process multiple audio files."""
    audio_arrays = []
    
    for path in audio_paths:
        try:
            # Load with torchaudio (handles resampling)
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            
            # Mix to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            audio_arrays.append(wav.squeeze().numpy())
        except Exception as e:
            # Fallback for missing/erroneous audio
            audio_arrays.append(np.zeros(16000)) # 1 sec silence
            
    input_features = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True 
    ).input_features.to(device)
    
    return input_features


def run_liar_evaluation(
    model_path: str,
    liar_data_path: str,
    output_dir: str = "liar_eval_results",
    device: str = "cuda",
    inject_transcript: bool = True,
    batch_size: int = 32
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {model_path}")
    model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    model.eval()
    
    # Load processor and tokenizer
    logger.info("Loading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(model.config.encoder_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Ensure audio locator in vocab
    audio_locator = model.config.audio_locator
    if audio_locator not in tokenizer.get_vocab():
        tokenizer.add_tokens([audio_locator], special_tokens=True)
    
    # Load samples
    logger.info(f"Loading samples from: {liar_data_path}")
    samples = []
    with open(liar_data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create Dataset/DataLoader
    dataset = LiarDataset(samples, tokenizer, processor, audio_locator, inject_transcript)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=dataset.collate_fn,
        num_workers=4
    )
    
    results = []
    task_stats = {}
    
    logger.info(f"Evaluating {len(samples)} samples (Batch {batch_size})...")
    
    for batch in tqdm(dataloader, desc="Eval Batches"):
        input_features = process_audio_files_batch(batch["audio_paths"], processor, device)
        inputs_text = batch["inputs_text"].to(device)
        
        with torch.no_grad():
            outputs = model.llm_model.generate(
                input_ids=inputs_text.input_ids,
                attention_mask=inputs_text.attention_mask,
                batch_features=input_features,
                batch_start_positions=batch["batch_start_positions"],
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode
        generated_ids = outputs[:, inputs_text.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Evaluate stats
        for i, pred in enumerate(preds):
            truth = batch["audio_truths"][i]
            liar = batch["liar_transcripts"][i]
            task = batch["task_types"][i]
            sid = batch["sample_ids"][i]
            
            correct = answer_matches_audio_truth(pred, truth)
            fooled = answer_matches_liar_transcript(pred, liar)
            
            if task not in task_stats:
                task_stats[task] = {"total": 0, "correct": 0, "fooled": 0}
            task_stats[task]["total"] += 1
            if correct: task_stats[task]["correct"] += 1
            if fooled: task_stats[task]["fooled"] += 1
            
            results.append({
                "sample_id": sid,
                "task_type": task,
                "truth": truth,
                "liar": liar,
                "pred": pred,
                "correct": correct,
                "fooled": fooled
            })
            
    # Calculate Summary
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    total_fooled = sum(1 for r in results if r["fooled"])
    
    summary = {
        "accuracy": total_correct / total if total > 0 else 0,
        "fooled_rate": total_fooled / total if total > 0 else 0,
        "total": total,
        "per_task": {}
    }
    
    for task, stats in task_stats.items():
        summary["per_task"][task] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "fooled": stats["fooled"] / stats["total"] if stats["total"] > 0 else 0
        }
        
    # Save
    with open(output_path / "liar_eval_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Accuracy: {summary['accuracy']*100:.1f}%, Fooled: {summary['fooled_rate']*100:.1f}%")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--liar-data", type=str, default="liar_transcript_data/liar_samples.jsonl")
    parser.add_argument("--output-dir", type=str, default="liar_eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-inject", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    run_liar_evaluation(
        model_path=args.model,
        liar_data_path=args.liar_data,
        output_dir=args.output_dir,
        device=args.device,
        inject_transcript=not args.no_inject,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
