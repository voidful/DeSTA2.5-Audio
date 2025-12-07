"""
OCAR-Qwen Data Collator

This module provides the OCARCollator class for handling audio data preprocessing
and prosody target extraction for the OCAR-Qwen model.

Key features:
- Audio loading and padding to 30s (no VAD cutting to preserve rhythm)
- F0 and Energy extraction using librosa
- Global stats computation (mean/std)
- Local contour interpolation for prosody supervision
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from transformers import AutoProcessor, AutoTokenizer


@dataclass
class OCARCollatorConfig:
    """Configuration for OCARCollator."""
    target_sr: int = 16000
    max_audio_length: float = 30.0  # seconds
    local_output_length: int = 375  # ~30s / 80ms per step
    f0_min: float = 50.0
    f0_max: float = 600.0
    hop_length: int = 512
    pad_token_id: int = 0


class OCARCollator:
    """
    Data collator for OCAR-Qwen model.
    
    Handles:
    - Audio loading and padding to 30s
    - Mel spectrogram extraction via Whisper processor
    - F0 and Energy contour extraction
    - Global stats and local contour computation
    """
    
    def __init__(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        config: Optional[OCARCollatorConfig] = None,
    ):
        """
        Initialize OCARCollator.
        
        Args:
            processor: Whisper processor for mel extraction
            tokenizer: Tokenizer for text processing
            config: Collator configuration
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config or OCARCollatorConfig()
        
        self.target_sr = self.config.target_sr
        self.max_samples = int(self.config.max_audio_length * self.target_sr)
        self.local_output_length = self.config.local_output_length
        
    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_samples, valid_length)
        """
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        valid_length = len(audio)
        
        # Pad or truncate to max_samples (30s)
        if len(audio) < self.max_samples:
            # Pad with zeros
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
        else:
            # Truncate
            audio = audio[:self.max_samples]
            valid_length = self.max_samples
            
        return audio, valid_length
    
    def extract_f0(self, audio: np.ndarray, valid_length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 contour using librosa's pyin.
        
        Args:
            audio: Audio samples
            valid_length: Number of valid samples (before padding)
            
        Returns:
            Tuple of (f0_contour, voiced_mask)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=self.target_sr,
            hop_length=self.config.hop_length,
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Create mask for valid regions
        valid_frames = int(valid_length / self.config.hop_length) + 1
        mask = np.zeros(len(f0), dtype=bool)
        mask[:min(valid_frames, len(f0))] = True
        mask = mask & (f0 > 0)  # Only voiced frames
        
        return f0, mask
    
    def extract_energy(self, audio: np.ndarray, valid_length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract energy (RMS) contour.
        
        Args:
            audio: Audio samples
            valid_length: Number of valid samples (before padding)
            
        Returns:
            Tuple of (energy_contour, valid_mask)
        """
        energy = librosa.feature.rms(
            y=audio,
            hop_length=self.config.hop_length,
        )[0]
        
        # Create mask for valid regions
        valid_frames = int(valid_length / self.config.hop_length) + 1
        mask = np.zeros(len(energy), dtype=bool)
        mask[:min(valid_frames, len(energy))] = True
        
        return energy, mask
    
    def compute_global_stats(
        self, 
        f0: np.ndarray, 
        energy: np.ndarray, 
        f0_mask: np.ndarray, 
        energy_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute global statistics: F0 mean/std, Energy mean/std.
        
        Args:
            f0: F0 contour
            energy: Energy contour
            f0_mask: Valid F0 positions
            energy_mask: Valid energy positions
            
        Returns:
            Array of shape (4,): [f0_mean, f0_std, energy_mean, energy_std]
        """
        # F0 stats (only from voiced regions)
        if f0_mask.sum() > 0:
            f0_voiced = f0[f0_mask]
            f0_mean = f0_voiced.mean()
            f0_std = f0_voiced.std() if len(f0_voiced) > 1 else 0.0
        else:
            f0_mean, f0_std = 0.0, 0.0
            
        # Energy stats (from valid regions)
        if energy_mask.sum() > 0:
            energy_valid = energy[energy_mask]
            energy_mean = energy_valid.mean()
            energy_std = energy_valid.std() if len(energy_valid) > 1 else 0.0
        else:
            energy_mean, energy_std = 0.0, 0.0
            
        return np.array([f0_mean, f0_std, energy_mean, energy_std], dtype=np.float32)
    
    def interpolate_contour(
        self, 
        contour: np.ndarray, 
        mask: np.ndarray, 
        target_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate contour to target length.
        
        Args:
            contour: Original contour
            mask: Valid positions mask
            target_length: Target output length
            
        Returns:
            Tuple of (interpolated_contour, interpolated_mask)
        """
        # Interpolate contour
        x_old = np.linspace(0, 1, len(contour))
        x_new = np.linspace(0, 1, target_length)
        
        interpolated = np.interp(x_new, x_old, contour)
        
        # Interpolate mask (use nearest neighbor)
        mask_float = mask.astype(np.float32)
        interpolated_mask = np.interp(x_new, x_old, mask_float) > 0.5
        
        return interpolated.astype(np.float32), interpolated_mask
    
    def process_audio_sample(
        self, 
        audio_path: str
    ) -> Dict[str, np.ndarray]:
        """
        Process a single audio sample.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio features and prosody targets
        """
        # Load and pad audio
        audio, valid_length = self.load_audio(audio_path)
        
        # Extract F0 and Energy
        f0, f0_mask = self.extract_f0(audio, valid_length)
        energy, energy_mask = self.extract_energy(audio, valid_length)
        
        # Compute global stats
        global_stats = self.compute_global_stats(f0, energy, f0_mask, energy_mask)
        
        # Interpolate to local output length
        local_f0, local_f0_mask = self.interpolate_contour(f0, f0_mask, self.local_output_length)
        local_energy, local_energy_mask = self.interpolate_contour(energy, energy_mask, self.local_output_length)
        
        # Combine masks
        local_prosody_mask = local_f0_mask | local_energy_mask
        
        # Compute audio attention mask (for Whisper output: ~1500 frames for 30s)
        # Whisper uses stride of 2 in conv layers, so 3000 mel frames -> 1500 encoder frames
        encoder_frames = 1500
        valid_encoder_frames = int((valid_length / self.max_samples) * encoder_frames)
        audio_attention_mask = np.zeros(encoder_frames, dtype=bool)
        audio_attention_mask[:valid_encoder_frames] = True
        
        return {
            "audio": audio,
            "valid_length": valid_length,
            "global_stats": global_stats,
            "local_f0": local_f0,
            "local_energy": local_energy,
            "local_prosody_mask": local_prosody_mask,
            "audio_attention_mask": audio_attention_mask,
        }
    
    def __call__(
        self, 
        batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Expected batch format (list of dicts):
        [
            {
                "audio_path": str,
                "input_ids": List[int],
                "labels": List[int],
                "attention_mask": List[int],
            },
            ...
        ]
        
        Returns:
            Collated batch dictionary with all tensors
        """
        batch_size = len(batch)
        
        # Process text
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # Process audio
        audio_list = []
        global_stats_list = []
        local_f0_list = []
        local_energy_list = []
        local_prosody_mask_list = []
        audio_attention_mask_list = []
        
        for sample in batch:
            # Text processing
            input_ids_list.append(sample["input_ids"])
            attention_mask_list.append(sample.get("attention_mask", [1] * len(sample["input_ids"])))
            labels_list.append(sample.get("labels", sample["input_ids"]))
            
            # Audio processing
            if "audio_path" in sample:
                audio_data = self.process_audio_sample(sample["audio_path"])
                audio_list.append(audio_data["audio"])
                global_stats_list.append(audio_data["global_stats"])
                local_f0_list.append(audio_data["local_f0"])
                local_energy_list.append(audio_data["local_energy"])
                local_prosody_mask_list.append(audio_data["local_prosody_mask"])
                audio_attention_mask_list.append(audio_data["audio_attention_mask"])
            else:
                # Handle missing audio (use zeros)
                audio_list.append(np.zeros(self.max_samples, dtype=np.float32))
                global_stats_list.append(np.zeros(4, dtype=np.float32))
                local_f0_list.append(np.zeros(self.local_output_length, dtype=np.float32))
                local_energy_list.append(np.zeros(self.local_output_length, dtype=np.float32))
                local_prosody_mask_list.append(np.zeros(self.local_output_length, dtype=bool))
                audio_attention_mask_list.append(np.zeros(1500, dtype=bool))
        
        # Pad text sequences
        max_text_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(batch_size):
            pad_len = max_text_len - len(input_ids_list[i])
            
            # Pad input_ids (left padding)
            padded_input_ids.append([self.config.pad_token_id] * pad_len + input_ids_list[i])
            padded_attention_mask.append([0] * pad_len + attention_mask_list[i])
            padded_labels.append([-100] * pad_len + labels_list[i])
        
        # Process audio through Whisper processor
        audio_values = self.processor(
            audio_list,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        ).input_features
        
        # Build output batch
        output_batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "audio_values": audio_values,
            "audio_attention_mask": torch.tensor(np.stack(audio_attention_mask_list), dtype=torch.bool),
            "global_stats": torch.tensor(np.stack(global_stats_list), dtype=torch.float32),
            "local_f0": torch.tensor(np.stack(local_f0_list), dtype=torch.float32),
            "local_energy": torch.tensor(np.stack(local_energy_list), dtype=torch.float32),
            "local_prosody_mask": torch.tensor(np.stack(local_prosody_mask_list), dtype=torch.bool),
        }
        
        return output_batch


def create_ocar_collator(
    whisper_model_id: str = "openai/whisper-large-v3",
    llm_model_id: str = "Qwen/Qwen2-4B-Instruct",
    **kwargs
) -> OCARCollator:
    """
    Factory function to create an OCARCollator.
    
    Args:
        whisper_model_id: Whisper model ID for processor
        llm_model_id: LLM model ID for tokenizer
        **kwargs: Additional arguments for OCARCollatorConfig
        
    Returns:
        Configured OCARCollator instance
    """
    import os
    
    cache_dir = os.getenv("HF_HOME")
    
    processor = AutoProcessor.from_pretrained(whisper_model_id, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, cache_dir=cache_dir)
    
    config = OCARCollatorConfig(**kwargs)
    
    return OCARCollator(processor=processor, tokenizer=tokenizer, config=config)
