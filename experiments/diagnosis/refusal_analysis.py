"""
P1-2: Refusal Rate Analysis (Optimized for H100)

Analyzes how often models refuse to answer audio-dependent questions.

Hypothesis: DeSTA (ASR-reliant) refuses more because it can't extract info
from audio. ORCA should refuse less (better audio understanding).

Method:
1. Select audio samples (Animal Recognition)
2. BATCHED inference to check responses
3. Count "cannot determine", "unclear", "unable to" responses

Expected:
- DeSTA: ~40% refusal rate
- ORCA: ~18% refusal rate
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor

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

# Patterns that indicate refusal/uncertainty
REFUSAL_PATTERNS = [
    r"cannot determine",
    r"unable to (identify|determine|recognize|classify)",
    r"cannot (identify|tell|recognize|classify)",
    r"not (possible|able) to",
    r"unclear",
    r"(can't|cannot) be (determined|identified)",
    r"i('m| am) not sure",
    r"i don't know",
    r"difficult to (tell|determine|identify)",
    r"impossible to (tell|determine|identify)",
    r"no (clear|definitive) (answer|indication)",
    r"insufficient (information|audio|data)",
    r"hard to (tell|say|determine)",
    r"(audio|sound) (is )?(too )?(unclear|noisy|distorted)",
    r"sorry",
    r"i cannot",
]

def is_refusal(response: str) -> bool:
    """Check if response indicates refusal/uncertainty."""
    response_lower = response.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False

class RefusalDataset(Dataset):
    def __init__(self, samples, tokenizer, processor, audio_locator="<|AUDIO|>"):
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
        raw_items = batch
        
        prompts = [f"{self.audio_locator}\n{q}" for q in questions]
        
        inputs_text = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        audio_id = self.tokenizer.convert_tokens_to_ids(self.audio_locator)
        batch_start_positions = []
        for seq in inputs_text.input_ids:
            indices = (seq == audio_id).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                batch_start_positions.append(indices[0].item())
            else:
                batch_start_positions.append(0)
                
        return {
            "questions": questions,
            "ground_truths": ground_truths,
            "inputs_text": inputs_text,
            "batch_start_positions": batch_start_positions,
            "raw_items": raw_items
        }

def process_audio_batch(processor, raw_items, device):
    audio_arrays = [item["audio"]["array"] for item in raw_items]
    
    input_features = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_features.to(device)
    
    return input_features

def run_refusal_analysis(
    model_path: str,
    num_samples: int = 200,
    batch_size: int = 32,
    output_dir: str = "refusal_analysis_results",
    device: str = "cuda"
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {model_path}")
    model = DeSTA25AudioModel.from_pretrained(model_path, device=device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model.config.encoder_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    audio_locator = model.config.audio_locator
    if audio_locator not in tokenizer.get_vocab():
        tokenizer.add_tokens([audio_locator], special_tokens=True)
        
    logger.info("Loading SLLM-multi-hop/AnimalQA dataset...")
    try:
        ds = load_dataset("SLLM-multi-hop/AnimalQA", split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {}
        
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    samples = []
    for idx in indices:
        item = ds[int(idx)]
        samples.append({
            "question": item["single_instruction"], # Use single hop
            "answer": item["single_answer"],
            "audio": item["audio"]
        })
    
    dataset = RefusalDataset(samples, tokenizer, processor, audio_locator)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=4
    )
    
    results = []
    refusal_count = 0
    
    logger.info(f"Analyzing {len(samples)} samples...")
    
    for batch in tqdm(dataloader, desc="Eval Batches"):
        input_features = process_audio_batch(processor, batch["raw_items"], device)
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
            
        preds = tokenizer.batch_decode(outputs[:, inputs_text.input_ids.shape[1]:], skip_special_tokens=True)
        
        for i, pred in enumerate(preds):
            pred = pred.strip()
            is_refused = is_refusal(pred)
            if is_refused:
                refusal_count += 1
                
            results.append({
                "question": batch["questions"][i],
                "ground_truth": batch["ground_truths"][i],
                "response": pred,
                "is_refusal": is_refused
            })
            
    total = len(results)
    rate = refusal_count / total if total > 0 else 0
    
    example_refusals = [r for r in results if r["is_refusal"]][:5]
    summary = {
        "model_path": model_path,
        "total": total,
        "refusal_count": refusal_count,
        "refusal_rate": rate,
        "refusal_rate_pct": f"{rate*100:.1f}%",
        "example_refusals": example_refusals
    }
    
    with open(output_path / "refusal_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Refusal Rate: {rate*100:.1f}%")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="refusal_analysis_results")
    
    args = parser.parse_args()
    run_refusal_analysis(
        model_path=args.model,
        num_samples=args.samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
