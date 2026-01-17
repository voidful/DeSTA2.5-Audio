"""
P0-1: Match Rate Quantification (Optimized for H100)

Measures ASR reliance by comparing predictions with/without audio.

Method:
1. Load SAKURA test samples (SLLM-multi-hop/*QA)
2. BATCHED Inference:
   - With audio (DeSTA)
   - Text only (LLM backbone)
3. Calculate match rate

Expected: DeSTA ~78%, ORCA ~60%
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import math
import re

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

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

# SLLM Datasets
SAKURA_TASKS = {
    "AnimalQA": "SLLM-multi-hop/AnimalQA",
    "GenderQA": "SLLM-multi-hop/GenderQA",
    "EmotionQA": "SLLM-multi-hop/EmotionQA",
    # "LanguageQA": "SLLM-multi-hop/LanguageQA" # Optional, sticking to 3 main tasks for consistency
}

class MatchRateDataset(Dataset):
    def __init__(self, samples, tokenizer, processor=None, audio_locator="<|AUDIO|>"):
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = processor
        self.audio_locator = audio_locator

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        questions = [b["question"] for b in batch]
        ground_truths = [b.get("answer", "") for b in batch]
        task_names = [b.get("task_name", "") for b in batch]
        
        # Text Only Batching
        text_inputs = []
        for q in questions:
            # Apply basic chat template if possible, or just raw
            # DeSTA usually expects: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    txt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    txt = q
            else:
                txt = q
            text_inputs.append(txt)
            
        inputs_text = self.tokenizer(
            text_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )

        return {
            "questions": questions,
            "ground_truths": ground_truths,
            "task_names": task_names,
            "input_ids_text": inputs_text.input_ids,
            "attention_mask_text": inputs_text.attention_mask,
            "raw_items": batch
        }

def process_audio_batch(model, processor, batch_items, device):
    audio_arrays = [item["audio"]["array"] for item in batch_items]
    sampling_rates = [item["audio"]["sampling_rate"] for item in batch_items]
    
    input_features = processor(
        audio_arrays, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    return input_features

def run_match_rate_analysis(
    model_path: str,
    samples_per_task: int = 250,
    output_dir: str = "match_rate_results",
    device: str = "cuda",
    hop_type: str = "single_hop", # single_hop or multi_hop
    batch_size: int = 32
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading DeSTA model: {model_path}")
    desta_model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    desta_model.eval()
    
    llm_model_id = desta_model.config.llm_model_id
    logger.info(f"Loading LLM backbone: {llm_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    llm_model.eval()
    
    processor = AutoProcessor.from_pretrained(desta_model.config.encoder_model_id)
    
    # Prepare Data
    all_samples = []
    
    hop_key_instruction = "single_instruction" if hop_type == "single_hop" else "multi_instruction"
    hop_key_answer = "single_answer" if hop_type == "single_hop" else "multi_answer"

    for task_name, dataset_id in SAKURA_TASKS.items():
        try:
            ds = load_dataset(dataset_id, split="test")
            indices = np.random.choice(len(ds), min(samples_per_task, len(ds)), replace=False)
            for idx in indices:
                item = ds[int(idx)]
                
                q = item.get(hop_key_instruction)
                a = item.get(hop_key_answer)
                
                if q and a:
                    # Create a consolidated item dict
                    sample = {
                        "question": q,
                        "answer": a,
                        "audio": item["audio"], # Keep raw audio dict
                        "task_name": task_name,
                        "id": f"{task_name}_{idx}"
                    }
                    all_samples.append(sample)
                    
        except Exception as e:
            logger.warning(f"Failed to load {dataset_id}: {e}")

    dataset = MatchRateDataset(all_samples, tokenizer, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=dataset.collate_fn,
        num_workers=4
    )

    results = []
    
    logger.info(f"Running inference on {len(all_samples)} samples...")
    
    for batch in tqdm(dataloader, desc="Processing"):
        questions = batch["questions"]
        
        # 1. Text-Only Inference
        input_ids_text = batch["input_ids_text"].to(device)
        attention_mask_text = batch["attention_mask_text"].to(device)
        
        with torch.no_grad():
            outputs_text = llm_model.generate(
                input_ids=input_ids_text,
                attention_mask=attention_mask_text,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        preds_text = tokenizer.batch_decode(outputs_text[:, input_ids_text.shape[1]:], skip_special_tokens=True)
            
        # 2. Audio Inference
        input_features = process_audio_batch(desta_model, processor, batch["raw_items"], device)
        
        # Prepare Audio Prompts using Chat Template
        texts_audio = []
        for q in questions:
            # Construct message with audio locator
            # Note: We assume audio_locator is handled by tokenizer if present in string
            content = f"{desta_model.config.audio_locator}\n{q}"
            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    txt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    txt = content
            else:
                txt = content
            texts_audio.append(txt)
        
        inputs_audio = tokenizer(
            texts_audio,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Ensure audio_locator is in vocab for ID search
        if desta_model.config.audio_locator not in tokenizer.get_vocab():
            tokenizer.add_tokens([desta_model.config.audio_locator], special_tokens=True)
            # IMPORTANT: resizing embeddings might be needed if added new token, 
            # but DeSTA should have it. We just ensure our tokenizer wrapper knows it.
        
        audio_token_id = tokenizer.convert_tokens_to_ids(desta_model.config.audio_locator)
        
        batch_start_positions = []
        for seq in inputs_audio.input_ids:
            indices = (seq == audio_token_id).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                batch_start_positions.append(indices[0].item())
            else:
                batch_start_positions.append(0)
                
        with torch.no_grad():
            outputs_audio = desta_model.llm_model.generate(
                input_ids=inputs_audio.input_ids,
                attention_mask=inputs_audio.attention_mask,
                batch_features=input_features,
                batch_start_positions=batch_start_positions,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        preds_audio = tokenizer.batch_decode(outputs_audio[:, inputs_audio.input_ids.shape[1]:], skip_special_tokens=True)
            
        for i in range(len(questions)):
            p_audio = preds_audio[i].strip()
            p_text = preds_text[i].strip()
            
            is_match = normalize_answer(p_audio) == normalize_answer(p_text)
            
            results.append({
                "task": batch["task_names"][i],
                "question": questions[i],
                "pred_with_audio": p_audio,
                "pred_text_only": p_text,
                "is_match": is_match
            })

    total = len(results)
    matches = sum(1 for r in results if r["is_match"])
    rate = matches / total if total > 0 else 0
    
    summary = {
        "overall_match_rate": rate,
        "total": total,
        "matches": matches,
        "per_task": {}
    }
    
    # Per task stats
    task_groups = {}
    for r in results:
        t = r["task"]
        if t not in task_groups: task_groups[t] = {"total": 0, "matches": 0}
        task_groups[t]["total"] += 1
        if r["is_match"]: task_groups[t]["matches"] += 1
        
    for t, s in task_groups.items():
        summary["per_task"][t] = s["matches"] / s["total"] if s["total"] > 0 else 0
    
    with open(output_path / "match_rate_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Final Match Rate: {rate*100:.1f}%")
    return summary

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--samples-per-task", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="match_rate_results")
    parser.add_argument("--hop-type", type=str, default="single_hop", choices=["single_hop", "multi_hop"])
    
    args = parser.parse_args()
    
    run_match_rate_analysis(
        model_path=args.model,
        samples_per_task=args.samples_per_task,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        hop_type=args.hop_type
    )

if __name__ == "__main__":
    main()
