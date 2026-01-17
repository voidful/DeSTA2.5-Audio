"""
P0-2: Liar Transcript Generator

Generates contradictory transcripts for testing audio vs text reliance.

Examples:
- Emotion: Audio shows sad speaker, transcript says "I'm so happy today!"
- Gender: Male voice audio, transcript says "As a woman, I think..."
- Animal: Dog barking audio, transcript says "The cat is meowing"

The goal is to test whether ORCA can correctly identify audio-grounded truth
when the transcript is intentionally misleading.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import wave

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping for generating contradictory information
EMOTION_CONTRADICTIONS = {
    "happy": ["sad", "angry", "depressed", "upset"],
    "sad": ["happy", "excited", "joyful", "cheerful"],
    "angry": ["calm", "peaceful", "happy", "relaxed"],
    "neutral": ["excited", "angry", "sad"],
    "fear": ["confident", "brave", "calm"],
    "surprise": ["bored", "expected", "unsurprised"],
    "disgust": ["pleased", "satisfied", "happy"],
}

GENDER_CONTRADICTIONS = {
    "male": "female",
    "female": "male",
    "man": "woman",
    "woman": "man",
    "boy": "girl",
    "girl": "boy",
    "he": "she",
    "she": "he",
    "his": "her",
    "her": "his",
}

ANIMAL_CONTRADICTIONS = {
    "dog": ["cat", "bird", "horse"],
    "cat": ["dog", "cow", "sheep"],
    "bird": ["dog", "cat", "frog"],
    "cow": ["pig", "sheep", "goat"],
    "pig": ["cow", "horse", "sheep"],
    "horse": ["cow", "dog", "pig"],
    "sheep": ["goat", "cow", "pig"],
    "frog": ["bird", "snake", "lizard"],
    "lion": ["tiger", "bear", "elephant"],
    "elephant": ["lion", "giraffe", "zebra"],
}


def get_emotion_liar_text(true_emotion: str) -> str:
    """Generate transcript that contradicts the true emotion."""
    true_emotion = true_emotion.lower()
    
    emotion_phrases = {
        "happy": [
            "I'm feeling so down today.",
            "Everything makes me sad.",
            "I can't stop crying.",
        ],
        "sad": [
            "I'm so excited and happy!",
            "This is the best day ever!",
            "I feel absolutely wonderful!",
        ],
        "angry": [
            "I'm so calm and relaxed right now.",
            "Everything is perfectly fine.",
            "I couldn't be more at peace.",
        ],
        "neutral": [
            "I'm extremely upset about this!",
            "I'm overjoyed beyond words!",
        ],
    }
    
    if true_emotion in emotion_phrases:
        return np.random.choice(emotion_phrases[true_emotion])
    
    opposite = EMOTION_CONTRADICTIONS.get(true_emotion, ["happy"])[0]
    return f"I'm feeling very {opposite} right now."


def get_gender_liar_text(true_gender: str) -> str:
    """Generate transcript that contradicts the true gender."""
    true_gender = true_gender.lower()
    
    if true_gender == "male" or true_gender == "man":
        return "As a woman, I'd like to share my thoughts. Speaking from a female perspective..."
    elif true_gender == "female" or true_gender == "woman":
        return "As a man, I believe that. Speaking from a male perspective..."
    else:
        return f"The speaker is clearly a {GENDER_CONTRADICTIONS.get(true_gender, 'person')}."


def get_animal_liar_text(true_animal: str) -> str:
    """Generate transcript that contradicts the true animal sound."""
    true_animal = true_animal.lower()
    
    if true_animal in ANIMAL_CONTRADICTIONS:
        fake_animal = np.random.choice(ANIMAL_CONTRADICTIONS[true_animal])
    else:
        fake_animal = "cat" if true_animal != "cat" else "dog"
    
    templates = [
        f"The sound is clearly from a {fake_animal}.",
        f"This is the sound of a {fake_animal} making noise.",
        f"A {fake_animal} can be heard in this audio clip.",
        f"Listen to the {fake_animal} in this recording.",
    ]
    
    return np.random.choice(templates)


def generate_liar_sample(
    question: str,
    ground_truth: str,
    task_type: str,
    original_id: str
) -> Dict:
    """Generate a liar sample with contradictory transcript."""
    
    if task_type == "emotion":
        liar_transcript = get_emotion_liar_text(ground_truth)
    elif task_type == "gender":
        liar_transcript = get_gender_liar_text(ground_truth)
    elif task_type == "animal":
        liar_transcript = get_animal_liar_text(ground_truth)
    else:
        liar_transcript = "This transcript is intentionally misleading."
    
    return {
        "task_type": task_type,
        "question": question,
        "audio_ground_truth": ground_truth,
        "liar_transcript": liar_transcript,
        "original_item_id": original_id,
    }


def write_wav_from_item(item, wav_path: str):
    """Write SAKURA audio to WAV file."""
    audio_data = item["audio"]
    audio_array = np.array(audio_data["array"], dtype=np.float32)
    sample_rate = audio_data["sampling_rate"]
    
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def generate_liar_dataset(
    samples_per_task: int = 150,
    output_dir: str = "liar_transcript_data",
    save_audio: bool = True
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_audio:
        audio_dir = output_path / "audio"
        audio_dir.mkdir(exist_ok=True)
    
    # Updated Dataset Mapping using SLLM-multi-hop
    task_map = {
        "emotion": "SLLM-multi-hop/EmotionQA",
        "gender": "SLLM-multi-hop/GenderQA",
        "animal": "SLLM-multi-hop/AnimalQA"
    }
    
    all_samples = []
    stats = {}
    
    for task_type, dataset_name in task_map.items():
        logger.info(f"Loading {dataset_name} for {task_type} task...")
        
        try:
            dataset = load_dataset(dataset_name, split="test")
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
        
        num_samples = min(samples_per_task, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        task_samples = []
        for idx in tqdm(indices, desc=f"Generating {task_type}"):
            item = dataset[int(idx)]
            
            # Extract fields for Single Hop
            # Note: SLLM datasets have both single_instruction and multi_instruction
            # Liar test focuses on simple contradictions, so use single hop
            question = item.get("single_instruction", "")
            if not question: # Fallback
                question = item.get("multi_instruction", "")
                
            answer = item.get("single_answer", "")
            if not answer:
                answer = item.get("multi_answer", "")
                
            liar_sample = generate_liar_sample(question, answer, task_type, str(idx))
            liar_sample["sample_id"] = f"{task_type}_{idx:04d}"
            
            if save_audio:
                audio_path = audio_dir / f"{liar_sample['sample_id']}.wav"
                write_wav_from_item(item, str(audio_path))
                liar_sample["audio_path"] = str(audio_path)
            
            task_samples.append(liar_sample)
        
        stats[task_type] = len(task_samples)
        all_samples.extend(task_samples)
        logger.info(f"  Generated {len(task_samples)} {task_type} samples")
    
    samples_file = output_path / "liar_samples.jsonl"
    with open(samples_file, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    summary = {
        "total_samples": len(all_samples),
        "per_task": stats,
        "output_dir": str(output_path),
        "samples_file": str(samples_file)
    }
    
    summary_file = output_path / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("LIAR TRANSCRIPT GENERATION COMPLETE")
    logger.info(f"Total samples: {len(all_samples)}")
    for task, count in stats.items():
        logger.info(f"  {task}: {count}")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'='*60}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="P0-2: Generate liar transcripts")
    parser.add_argument("--samples-per-task", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="liar_transcript_data")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    generate_liar_dataset(
        samples_per_task=args.samples_per_task,
        output_dir=args.output_dir,
        save_audio=not args.no_audio
    )

if __name__ == "__main__":
    main()
