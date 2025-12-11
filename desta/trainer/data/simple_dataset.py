"""
Dataset Module for DeSTA2.5-Audio

This module provides dataset and collation classes for loading and preprocessing
audio-text data for training and evaluation.
"""
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import datasets
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, AutoTokenizer

from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions
from desta.utils.audio import AudioSegment
from lulutils import resolve_filepath


def _get_rank() -> int:
    """Get the current process rank in distributed training. Returns 0 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return _get_rank() == 0


def _barrier():
    """Synchronize all processes in distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _prepare_audio_context_with_start_end_tags(
    text: str,
    audio_size_list: List[int],
    transcription_size_list: List[int],
    placeholder_token: str,
    tokenizer: AutoTokenizer,
    start_tag: str = "<start_audio>",
    end_tag: str = "<end_audio>"
) -> Tuple[str, List[int]]:
    """
    Process text with <start_audio>...<end_audio> format.
    
    Replaces each audio block with placeholder tokens for audio features 
    and transcription.
    
    Args:
        text: Input text containing audio blocks
        audio_size_list: Audio feature sizes for each audio
        transcription_size_list: Transcription token sizes for each audio
        placeholder_token: Token to use as placeholder
        tokenizer: Tokenizer for text processing
        start_tag: Start tag for audio block
        end_tag: End tag for audio block
    
    Returns:
        Tuple of (processed_text, start_positions)
    """
    pattern = re.escape(start_tag) + r'.*?' + re.escape(end_tag)
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if len(matches) != len(audio_size_list):
        logging.warning(
            f"Audio block count ({len(matches)}) != audio_size_list ({len(audio_size_list)})"
        )
    
    result_tokens = []
    start_positions = []
    last_end = 0
    
    for audio_idx, match in enumerate(matches):
        # Add tokens before this match
        prefix_text = text[last_end:match.start()]
        if prefix_text:
            result_tokens.extend(tokenizer.tokenize(prefix_text, add_special_tokens=False))
        
        # Record start position and add placeholders
        start_positions.append(len(result_tokens))
        
        if audio_idx < len(audio_size_list) and audio_idx < len(transcription_size_list):
            total_size = audio_size_list[audio_idx] + transcription_size_list[audio_idx]
            result_tokens.extend([placeholder_token] * total_size)
        
        last_end = match.end()
    
    # Add remaining text
    suffix_text = text[last_end:]
    if suffix_text:
        result_tokens.extend(tokenizer.tokenize(suffix_text, add_special_tokens=False))
    
    return tokenizer.convert_tokens_to_string(result_tokens), start_positions


def _resolve_audio_filepath(audio_filepath: str) -> str:
    """Resolve audio filepath, trying .wav extension if original doesn't exist."""
    if os.path.exists(audio_filepath):
        return audio_filepath
    
    base, ext = os.path.splitext(audio_filepath)
    wav_filepath = base + ".wav"
    
    if os.path.exists(wav_filepath):
        return wav_filepath
    
    raise FileNotFoundError(f"Audio file not found: {audio_filepath}")


class BaseCollateFn:
    """Collate function for batching audio-text samples."""
    
    def __init__(
        self, 
        data_cfg: DictConfig, 
        tokenizer: AutoTokenizer, 
        processor: AutoFeatureExtractor
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = data_cfg.max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare a training batch from list of samples."""
        assert self.tokenizer.padding_side == "left", \
            f"padding_side must be left, got {self.tokenizer.padding_side}"
        
        # Pre-validate audio files and filter out samples with undecodable audio
        valid_batch = []
        valid_audio_features = []  # List of list of features per sample
        
        for item in batch:
            sample_features = []
            audio_valid = True
            
            for audio_dict in item["processed_audios"]:
                try:
                    feature = AudioSegment.from_file(
                        audio_dict["audio"],
                        target_sr=16000,
                        channel_selector="average"
                    ).samples
                    # Convert to list for numpy 2.0 / torch compatibility
                    sample_features.append(feature.tolist())
                except Exception as e:
                    logging.warning(f"Skipping sample due to audio decode error: {audio_dict['audio']} - {e}")
                    audio_valid = False
                    break
            
            if audio_valid:
                valid_batch.append(item)
                valid_audio_features.append(sample_features)
        
        # If no valid samples remain, return an empty batch marker
        # This can happen in rare cases (e.g., file system issues, race conditions)
        if len(valid_batch) == 0:
            failed_paths = [a.get("audio", "unknown") for item in batch for a in item.get("processed_audios", [])]
            logging.warning(
                f"Entire batch skipped due to audio decode errors. "
                f"Failed paths: {failed_paths[:3]}..."
            )
            # Return empty batch marker - trainer should check for this
            return {"_empty_batch": True}
        
        batch = valid_batch
        
        # Tokenize audio_context + target
        audio_text_inputs = self.tokenizer(
            [item["audio_context"] + item["target"] for item in batch],
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )

        # Tokenize audio_context only (for evaluation)
        audio_context_inputs = self.tokenizer(
            [item["audio_context"] for item in batch],
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )
        
        labels = torch.full_like(audio_text_inputs['input_ids'], -100)

        # Collect audio features and positions
        batch_features = []
        batch_start_positions = []
        batch_transcription_ids = []
        audio_context_batch_start_positions = []
        audio_start_answer_positions = []
        
        for i, item in enumerate(batch):
            # Calculate label positions
            total_length = audio_text_inputs["length"][i]
            audio_context_length = len(self.tokenizer.tokenize(item["audio_context"]))
            pad_length = total_length - audio_text_inputs["attention_mask"][i].sum()
            
            start_answer_position = pad_length + audio_context_length
            labels[i, start_answer_position:] = audio_text_inputs['input_ids'][i, start_answer_position:]
            audio_start_answer_positions.append(start_answer_position)

            # Use pre-loaded audio features
            batch_features.extend(valid_audio_features[i])

            # Encode transcriptions
            for transcription in item["transcription_list"]:
                batch_transcription_ids.append(
                    self.tokenizer.encode(
                        transcription, 
                        add_special_tokens=False, 
                        return_tensors="pt"
                    ).long()
                )
            
            # Record start positions with padding offset
            for start_position in item["start_positions"]:
                batch_start_positions.append((i, start_position + pad_length))

            # Context positions for evaluation
            ctx_total = audio_context_inputs["length"][i]
            ctx_pad = ctx_total - audio_context_inputs["attention_mask"][i].sum()
            for start_position in item["start_positions"]:
                audio_context_batch_start_positions.append((i, start_position + ctx_pad))

        # Extract audio features
        batch_features = self.processor(
            batch_features, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features

        assert len(batch_features) == len(batch_start_positions) == len(batch_transcription_ids), \
            f"Length mismatch: features={len(batch_features)}, positions={len(batch_start_positions)}, transcriptions={len(batch_transcription_ids)}"

        result = {
            # Audio-text sequence
            "input_ids": audio_text_inputs['input_ids'],
            "attention_mask": audio_text_inputs['attention_mask'],
            "labels": labels,
            "audio_start_answer_positions": audio_start_answer_positions,
            # Audio features
            "batch_features": batch_features,
            "batch_transcription_ids": batch_transcription_ids,
            "batch_start_positions": batch_start_positions,
            # Context for evaluation
            "context_input_ids": audio_context_inputs['input_ids'],
            "context_attention_mask": audio_context_inputs['attention_mask'],
            "context_batch_start_positions": audio_context_batch_start_positions,
            # Metadata
            "metadata": list(batch)
        }
        
        # === Optional OCAR prosody fields ===
        # Collate f0_energy_global if present in samples
        if any("f0_energy_global" in item for item in batch):
            global_prosody = []
            for item in batch:
                if "f0_energy_global" in item:
                    global_prosody.append(torch.tensor(item["f0_energy_global"], dtype=torch.float32))
                else:
                    global_prosody.append(torch.zeros(4, dtype=torch.float32))
            result["f0_energy_global"] = torch.stack(global_prosody, dim=0)  # [B, 4]
        
        # Collate f0_energy_local if present in samples (with padding)
        if any("f0_energy_local" in item for item in batch):
            local_prosody_list = []
            max_len = 0
            for item in batch:
                if "f0_energy_local" in item:
                    t = torch.tensor(item["f0_energy_local"], dtype=torch.float32)
                    local_prosody_list.append(t)
                    max_len = max(max_len, t.size(0))
                else:
                    local_prosody_list.append(None)
            
            # Pad to max length
            padded = []
            for t in local_prosody_list:
                if t is None:
                    padded.append(torch.zeros(max_len, 2, dtype=torch.float32))
                else:
                    if t.size(0) < max_len:
                        pad = torch.zeros(max_len - t.size(0), 2, dtype=torch.float32)
                        t = torch.cat([t, pad], dim=0)
                    padded.append(t)
            result["f0_energy_local"] = torch.stack(padded, dim=0)  # [B, T_local, 2]
        
        return result


class BaseAudioTextDataset:
    """
    Dataset for audio-text training data.
    
    Expected JSONL format:
    {
        "id": "sample_id",
        "messages": [
            {
                "role": "user",
                "content": "Describe the audio. <|AUDIO|>",
                "audios": [{"audio": "path/to/file.wav", "text": "transcription"}]
            }
        ],
        "response": "This is a description of the audio."
    }
    
    Supports two audio marker formats:
    - <|AUDIO|> single marker
    - <start_audio>...<end_audio> block markers
    """
    
    def __init__(
        self, 
        cfg: DictConfig,
        data_cfg: DictConfig,
        tokenizer: AutoTokenizer,
        processor: AutoFeatureExtractor
    ):
        self.audio_locator = cfg.model.audio_locator
        self.placeholder_token = cfg.model.placeholder_token
        self.data_root = data_cfg.data_root
        self.prompt_size = cfg.model.connector.prompt_size
        self.tokenizer = tokenizer
        self.processor = processor
        
        # OCAR configuration: use global_num_tokens for audio size in OCAR mode
        self.connector_mode = cfg.model.connector.mode
        ocar_cfg = cfg.model.get("ocar", {})
        self.ocar_global_num_tokens = ocar_cfg.get("global_num_tokens", 4)
        
        model_cfg = cfg.model
        if isinstance(model_cfg, DictConfig):
            self.system_prompt = model_cfg.get("system_prompt", None)
        else:
            self.system_prompt = getattr(model_cfg, "system_prompt", None)

        # Load manifest files
        self.manifest_filepaths = (
            [data_cfg.manifest_filepaths] 
            if isinstance(data_cfg.manifest_filepaths, str) 
            else data_cfg.manifest_filepaths
        )

        for filepath in self.manifest_filepaths:
            logging.info(f"Loading manifest: {filepath}")

        # === Robust Dataset Loading with Disk Cache ===
        # Use save_to_disk/load_from_disk to avoid HuggingFace cache race conditions.
        # A lock file coordinates distributed workers.
        
        data_files = [resolve_filepath(fp) for fp in self.manifest_filepaths]
        
        # Create a stable cache path based on manifest files
        import hashlib
        cache_key = hashlib.md5("_".join(sorted(data_files)).encode()).hexdigest()[:12]
        cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "desta_preprocessed",
            cache_key
        )
        lock_file = cache_dir + ".lock"
        ready_file = cache_dir + ".ready"
        
        def _load_and_preprocess():
            """Load raw data and run preprocessing."""
            ds = datasets.load_dataset("json", data_files=data_files)["train"]
            ds = ds.map(
                self._preprocess_function,
                batched=True,
                batch_size=128,
                num_proc=1,
                load_from_cache_file=False,  # Don't use HF cache, we manage our own
                keep_in_memory=False
            )
            return ds
        
        # Check if preprocessed cache already exists and is ready
        if os.path.exists(ready_file) and os.path.isdir(cache_dir):
            logging.info(f"[Rank {_get_rank()}] Loading preprocessed dataset from disk cache: {cache_dir}")
            try:
                self.dataset = datasets.load_from_disk(cache_dir)
                logging.info(f"[Rank {_get_rank()}] Loaded {len(self.dataset)} samples from cache.")
            except Exception as e:
                logging.warning(f"[Rank {_get_rank()}] Cache load failed: {e}. Will reprocess.")
                os.remove(ready_file) if os.path.exists(ready_file) else None
                self.dataset = None
        else:
            self.dataset = None
        
        # If cache doesn't exist or failed to load, process it
        if self.dataset is None:
            if _is_main_process():
                # Rank 0 does the preprocessing
                logging.info(f"[Rank {_get_rank()}] Preprocessing dataset (this may take ~1 hour)...")
                
                # Create lock file to signal we're working
                os.makedirs(os.path.dirname(lock_file), exist_ok=True)
                with open(lock_file, "w") as f:
                    f.write(f"rank0_processing_{os.getpid()}")
                
                try:
                    self.dataset = _load_and_preprocess()
                    
                    # Save to disk
                    logging.info(f"[Rank {_get_rank()}] Saving preprocessed dataset to: {cache_dir}")
                    self.dataset.save_to_disk(cache_dir)
                    
                    # Mark as ready
                    with open(ready_file, "w") as f:
                        f.write("ready")
                    
                    logging.info(f"[Rank {_get_rank()}] Preprocessing complete. Saved {len(self.dataset)} samples.")
                finally:
                    # Remove lock file
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
            
            # Synchronize - rank 0 finishes before others proceed
            _barrier()
            
            # Non-main ranks load from the cache that rank 0 created
            if not _is_main_process():
                # Wait for ready file (with timeout)
                import time
                max_wait = 7200  # 2 hours max
                waited = 0
                while not os.path.exists(ready_file) and waited < max_wait:
                    time.sleep(5)
                    waited += 5
                    if waited % 60 == 0:
                        logging.info(f"[Rank {_get_rank()}] Waiting for rank 0 to finish preprocessing... ({waited}s)")
                
                if os.path.exists(ready_file):
                    logging.info(f"[Rank {_get_rank()}] Loading preprocessed dataset from: {cache_dir}")
                    self.dataset = datasets.load_from_disk(cache_dir)
                    logging.info(f"[Rank {_get_rank()}] Loaded {len(self.dataset)} samples.")
                else:
                    raise RuntimeError(f"[Rank {_get_rank()}] Timeout waiting for preprocessed dataset!")

        # Analyze skip reasons before filtering
        original_len = len(self.dataset)
        
        # Count skip reasons
        no_length = sum(1 for x in self.dataset if x["length"] == 0)
        no_audio_context = sum(1 for x in self.dataset if not x["audio_context"])
        no_processed_audios = sum(1 for x in self.dataset if not x["processed_audios"])
        
        # Filter invalid samples (basic validation)
        self.dataset = self.dataset.filter(
            lambda x: x["length"] > 0 and len(x["audio_context"]) > 0 and len(x["processed_audios"]) > 0
        )
        filtered_len = len(self.dataset)
        skipped = original_len - filtered_len
        
        logging.info("=" * 60)
        logging.info("Dataset Statistics (Phase 1 - Format Validation):")
        logging.info(f"  Total samples: {original_len}")
        logging.info(f"  Valid samples: {filtered_len}")
        logging.info(f"  Skipped samples: {skipped} ({skipped/max(original_len,1)*100:.2f}%)")
        logging.info("Skip Reasons (may overlap):")
        logging.info(f"  - No length (empty response): {no_length}")
        logging.info(f"  - No audio_context: {no_audio_context}")
        logging.info(f"  - No processed_audios: {no_processed_audios}")
        
        # Phase 2: Audio validation is DISABLED for performance.
        # The collate function already handles decode errors gracefully by skipping bad samples.
        # Full audio decoding of 4M samples would take ~20 hours.
        logging.info("-" * 60)
        logging.info("Dataset Statistics (Phase 2 - Audio Validation):")
        logging.info("  ⚠ SKIPPED: Audio validation disabled for performance.")
        logging.info("  → Bad audio files will be skipped at runtime by collate_fn.")
        after_audio_validation = len(self.dataset)
        
        logging.info("-" * 60)
        logging.info("DeSTA Training Mode:")
        logging.info("  ✓ Prompt-only training (messages field ignored)")
        logging.info("  ✓ Audio locator appended automatically to prompts")
        logging.info("  ✓ seed_description is ignored (no transcription text used)")
        logging.info("  → Model learns: audio_features → response")
        logging.info("=" * 60)
        
        # Print sample of first valid and invalid examples for debugging
        if skipped > 0 and original_len > 0:
            # Find first invalid sample
            for i, sample in enumerate(datasets.load_dataset(
                "json", data_files=[resolve_filepath(fp) for fp in self.manifest_filepaths]
            )["train"]):
                if i >= 3:
                    break
                logging.info(f"Sample {i} keys: {list(sample.keys())}")
                if "messages" in sample and sample["messages"]:
                    msg = sample["messages"][0] if sample["messages"] else {}
                    logging.info(f"  First message keys: {list(msg.keys()) if msg else 'empty'}")
                    if msg.get("content"):
                        content_preview = msg["content"][:200] if len(msg.get("content", "")) > 200 else msg.get("content", "")
                        logging.info(f"  Content preview: {content_preview}...")
        
        if after_audio_validation == 0:
            logging.error("No valid samples found! Check data format.")
            logging.error(f"Expected audio markers: '{self.audio_locator}' or '<start_audio>...<end_audio>'")
            logging.error(f"Data root: {self.data_root}")
            logging.error("Common issues:")
            logging.error("  1. data_root path is incorrect")
            logging.error("  2. Audio files don't exist at expected paths")
            logging.error("  3. Audio files cannot be decoded (corrupted or unsupported format)")
            logging.error("  4. JSONL format doesn't match expected schema")

        self.collate_fn = BaseCollateFn(
            data_cfg=data_cfg, 
            tokenizer=self.tokenizer, 
            processor=self.processor
        )
        
        # Sanity check: verify first sample can be loaded
        if after_audio_validation > 0:
            try:
                first_sample = self.dataset[0]
                if first_sample["processed_audios"]:
                    test_audio = first_sample["processed_audios"][0]["audio"]
                    test_segment = AudioSegment.from_file(
                        test_audio, target_sr=16000, channel_selector="average"
                    )
                    logging.info(f"✓ Audio loading sanity check passed: {test_audio}")
                    logging.info(f"  Audio duration: {test_segment.duration:.2f}s, samples: {test_segment.num_samples}")
            except Exception as e:
                logging.error(f"✗ Audio loading sanity check FAILED: {e}")
                logging.error("  This means audio files exist in manifest but cannot be loaded!")

    def _validate_audio_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that all audio files in a sample can be decoded.
        
        Args:
            sample: A dataset sample with 'processed_audios' field
            
        Returns:
            True if all audio files can be decoded, False otherwise
        """
        if not sample.get("processed_audios"):
            return False
        
        for audio_dict in sample["processed_audios"]:
            audio_path = audio_dict.get("audio")
            if not audio_path:
                return False
            
            try:
                # Try to load the audio file
                AudioSegment.from_file(
                    audio_path,
                    target_sr=16000,
                    channel_selector="average"
                )
            except Exception as e:
                logging.debug(f"Audio validation failed for {audio_path}: {e}")
                return False
        
        return True

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess a batch of examples using prompt/id/response only."""
        audio_context_list = []
        start_positions_list = []
        audio_list = []
        transcription_list = []

        ids = examples["id"]
        prompts = examples.get("prompt", [""] * len(ids))
        responses = examples.get("response", [""] * len(ids))

        skip_reasons = {
            "empty_prompt": 0,
            "audio_file_not_found": 0,
            "no_audio_markers": 0,
        }

        is_first_batch = not hasattr(self, "_debug_prompt_logged")

        for idx, (sample_id, prompt) in enumerate(zip(ids, prompts)):
            prompt_text = (prompt or "").strip()
            if not prompt_text:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["empty_prompt"] += 1
                continue

            if self.audio_locator not in prompt_text:
                user_content = f"{prompt_text} {self.audio_locator}"
            else:
                user_content = prompt_text

            user_message = {
                "role": "user",
                "content": user_content,
                "audios": [{
                    "audio": sample_id,
                    "text": "",
                }]
            }

            processed_messages = []
            if getattr(self, "system_prompt", None):
                processed_messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            processed_messages.append(user_message)

            try:
                audio_context = self.tokenizer.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logging.error(f"Error at index {idx}: {processed_messages}")
                raise e

            audios = user_message["audios"]
            new_audios = []
            audio_not_found = False
            missing_audio_path = None
            for audio_dict in audios:
                try:
                    full_path = os.path.join(self.data_root, audio_dict["audio"])
                    audio_dict["audio"] = _resolve_audio_filepath(full_path)
                    new_audios.append(audio_dict)
                except FileNotFoundError:
                    audio_not_found = True
                    missing_audio_path = full_path
                    break

            if audio_not_found:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["audio_file_not_found"] += 1
                if not hasattr(self, '_first_missing_audio_logged'):
                    logging.error(f"[DEBUG] First missing audio file: {missing_audio_path}")
                    logging.error(f"  data_root: {self.data_root}")
                    logging.error(f"  audio id: {audios[0].get('audio', 'N/A') if audios else 'N/A'}")
                    self._first_missing_audio_logged = True
                continue

            # Use appropriate audio size based on connector mode
            if self.connector_mode == "ocar_hybrid":
                audio_size = self.ocar_global_num_tokens
            else:
                audio_size = self.prompt_size
            audio_size_list = [audio_size] * len(new_audios)
            transcriptions = ["" for _ in new_audios]
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False))
                for text in transcriptions
            ]

            start_end_pattern = r'<start_audio>.*?<end_audio>'
            has_start_end_blocks = bool(re.search(start_end_pattern, audio_context, re.DOTALL))
            num_locators = audio_context.count(self.audio_locator)

            if has_start_end_blocks:
                audio_context, start_positions = _prepare_audio_context_with_start_end_tags(
                    text=audio_context,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token,
                    tokenizer=self.tokenizer
                )
            elif num_locators > 0:
                tokens, start_positions = _prepare_audio_context_and_start_positions(
                    token_list=self.tokenizer.tokenize(audio_context),
                    audio_locator=self.audio_locator,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token
                )
                audio_context = self.tokenizer.convert_tokens_to_string(tokens)
            else:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["no_audio_markers"] += 1
                continue

            audio_context_list.append(audio_context)
            start_positions_list.append(start_positions)
            audio_list.append(new_audios)
            transcription_list.append(transcriptions)

            if is_first_batch and not hasattr(self, "_debug_prompt_logged"):
                logging.info("[DEBUG] Prompt-only preprocessing active.")
                logging.info(f"  Example prompt: {prompt_text[:80]}...")
                self._debug_prompt_logged = True

        total_skipped = sum(skip_reasons.values())
        if total_skipped > 0 and is_first_batch:
            logging.info(f"Batch skip reasons (prompt-only mode): {skip_reasons}")
            if skip_reasons["empty_prompt"] > 0:
                logging.warning("  empty_prompt: prompt field missing for some samples")
            if skip_reasons["audio_file_not_found"] > 0:
                logging.warning(f"  audio_file_not_found: Check data_root='{self.data_root}'")
            if skip_reasons["no_audio_markers"] > 0:
                logging.warning(f"  no_audio_markers: '{self.audio_locator}' missing in prompt text")

        # Set outputs
        examples["audio_context"] = audio_context_list
        examples["start_positions"] = start_positions_list
        examples["transcription_list"] = transcription_list
        examples["processed_audios"] = audio_list

        # Calculate targets and lengths
        targets = []
        lengths = []
        for audio_context, response in zip(examples["audio_context"], examples["response"]):
            if audio_context and response:
                targets.append(response + self.tokenizer.eos_token)
                lengths.append(len(self.tokenizer.tokenize(audio_context + response)))
            else:
                targets.append("")
                lengths.append(0)

        examples["target"] = targets
        examples["length"] = lengths

        return examples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]