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
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, AutoTokenizer

from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions
from desta.utils.audio import AudioSegment
from lulutils import resolve_filepath


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

            # Load audio features
            for audio_dict in item["processed_audios"]:
                feature = AudioSegment.from_file(
                    audio_dict["audio"],
                    target_sr=16000,
                    channel_selector="average"
                ).samples
                # Convert to list for numpy 2.0 / torch compatibility
                batch_features.append(feature.tolist())

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

        return {
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

        # Load manifest files
        self.manifest_filepaths = (
            [data_cfg.manifest_filepaths] 
            if isinstance(data_cfg.manifest_filepaths, str) 
            else data_cfg.manifest_filepaths
        )

        for filepath in self.manifest_filepaths:
            logging.info(f"Loading manifest: {filepath}")

        # Load and preprocess dataset
        datasets.disable_caching()
        
        self.dataset = datasets.load_dataset(
            "json", 
            data_files=[resolve_filepath(fp) for fp in self.manifest_filepaths]
        )["train"]

        self.dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=256,
            num_proc=1,
            load_from_cache_file=False,
            keep_in_memory=True
        )

        # Analyze skip reasons before filtering
        original_len = len(self.dataset)
        
        # Count skip reasons
        no_length = sum(1 for x in self.dataset if x["length"] == 0)
        no_audio_context = sum(1 for x in self.dataset if not x["audio_context"])
        no_processed_audios = sum(1 for x in self.dataset if not x["processed_audios"])
        
        # Filter invalid samples
        self.dataset = self.dataset.filter(
            lambda x: x["length"] > 0 and len(x["audio_context"]) > 0 and len(x["processed_audios"]) > 0
        )
        filtered_len = len(self.dataset)
        skipped = original_len - filtered_len
        
        logging.info("=" * 60)
        logging.info("Dataset Statistics:")
        logging.info(f"  Total samples: {original_len}")
        logging.info(f"  Valid samples: {filtered_len}")
        logging.info(f"  Skipped samples: {skipped} ({skipped/max(original_len,1)*100:.2f}%)")
        logging.info("Skip Reasons (may overlap):")
        logging.info(f"  - No length (empty response): {no_length}")
        logging.info(f"  - No audio_context: {no_audio_context}")
        logging.info(f"  - No processed_audios: {no_processed_audios}")
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
        
        if filtered_len == 0:
            logging.error("No valid samples found! Check data format.")
            logging.error(f"Expected audio markers: '{self.audio_locator}' or '<start_audio>...<end_audio>'")

        self.collate_fn = BaseCollateFn(
            data_cfg=data_cfg, 
            tokenizer=self.tokenizer, 
            processor=self.processor
        )

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess a batch of examples."""
        audio_context_list = []
        start_positions_list = []
        audio_list = []
        transcription_list = []
        
        # Skip reason counters for diagnostics
        skip_reasons = {
            "empty_messages": 0,
            "no_audio_markers": 0,
            "audio_file_not_found": 0,
            "no_audios_in_messages": 0,
        }

        # Collect audios from messages
        batch_audios = []
        for messages, sample_id in zip(examples["messages"], examples["id"]):
            audios = []
            if messages:
                for message in messages:
                    if message and message.get("audios"):
                        for audio_item in message["audios"]:
                            item = dict(audio_item)
                            if item.get("audio") is None:
                                item["audio"] = sample_id
                            audios.append(item)
            batch_audios.append(audios)

        # Process each sample
        for idx, (messages, audios) in enumerate(zip(examples["messages"], batch_audios)):
            # Handle empty messages
            if not messages:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["empty_messages"] += 1
                continue

            # Apply chat template
            try:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logging.error(f"Error at index {idx}: {messages}")
                raise e
            
            # Debug: log first sample to check if audio_locator is preserved
            if idx == 0 and len(audio_context_list) == 0:
                logging.info(f"[DEBUG] First sample audio_context preview (first 500 chars):")
                logging.info(f"  {audio_context[:500] if len(audio_context) > 500 else audio_context}")
                logging.info(f"[DEBUG] Looking for audio_locator: '{self.audio_locator}'")
                logging.info(f"[DEBUG] Count of <|AUDIO|> markers: {audio_context.count(self.audio_locator)}")
                logging.info(f"[DEBUG] Has <start_audio>...<end_audio> blocks: {bool(re.search(r'<start_audio>.*?<end_audio>', audio_context, re.DOTALL))}")

            # Check if there are any audios
            if not audios:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["no_audios_in_messages"] += 1
                continue

            # Resolve audio paths
            new_audios = []
            audio_not_found = False
            for audio_dict in audios:
                try:
                    audio_dict["audio"] = _resolve_audio_filepath(
                        os.path.join(self.data_root, audio_dict["audio"])
                    )
                    new_audios.append(audio_dict)
                except FileNotFoundError:
                    audio_not_found = True
                    break
            
            if audio_not_found:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skip_reasons["audio_file_not_found"] += 1
                continue

            # Prepare placeholder sizes
            audio_size_list = [self.prompt_size] * len(new_audios)
            
            # Truncate transcriptions to 100 tokens
            transcriptions = [
                self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.tokenize(audio.get("text", " ") or " ", add_special_tokens=False)[:100]
                ) for audio in new_audios
            ]
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) 
                for text in transcriptions
            ]
            transcription_list.append(transcriptions)

            # Process audio markers
            # Check for actual <start_audio>...<end_audio> blocks (not just mentions in text)
            # A real audio block looks like: <start_audio>some content<end_audio>
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
                # No audio markers - invalid sample
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                skip_reasons["no_audio_markers"] += 1
                continue

            audio_context_list.append(audio_context)
            start_positions_list.append(start_positions)
            audio_list.append(new_audios)
        
        # Log skip reasons (only once per batch to reduce log spam)
        total_skipped = sum(skip_reasons.values())
        if total_skipped > 0:
            logging.debug(f"Batch skip reasons: {skip_reasons}")

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
