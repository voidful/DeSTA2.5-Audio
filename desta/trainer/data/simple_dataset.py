import datasets
import logging
import os
import re
import torch
from typing import List, Dict, Any
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoFeatureExtractor

from desta.utils.audio import AudioSegment
from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions
from lulutils import resolve_filepath


def _prepare_audio_context_with_start_end_tags(
    text: str,
    audio_size_list: List[int],
    transcription_size_list: List[int],
    placeholder_token: str,
    tokenizer,
    start_tag: str = "<start_audio>",
    end_tag: str = "<end_audio>"
):
    """
    Process text with <start_audio>...<end_audio> format.
    Replace each <start_audio>...<end_audio> block with placeholder tokens.
    
    Args:
        text: Input text containing <start_audio>...<end_audio> blocks
        audio_size_list: List of audio feature sizes (prompt_size) for each audio
        transcription_size_list: List of transcription token sizes for each audio
        placeholder_token: Token to use as placeholder
        tokenizer: Tokenizer for tokenizing the text
        start_tag: Start tag for audio block
        end_tag: End tag for audio block
    
    Returns:
        Tuple of (processed_text, start_positions)
    """
    # Find all <start_audio>...<end_audio> blocks
    pattern = re.escape(start_tag) + r'.*?' + re.escape(end_tag)
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if len(matches) != len(audio_size_list):
        logging.warning(f"Number of audio blocks ({len(matches)}) does not match audio_size_list ({len(audio_size_list)})")
    
    # Process text: replace each block with placeholders
    result_tokens = []
    start_positions = []
    last_end = 0
    audio_idx = 0
    
    for match in matches:
        # Add tokens before this match
        prefix_text = text[last_end:match.start()]
        if prefix_text:
            result_tokens.extend(tokenizer.tokenize(prefix_text, add_special_tokens=False))
        
        # Record start position
        start_positions.append(len(result_tokens))
        
        # Add placeholder tokens for audio features + transcription
        if audio_idx < len(audio_size_list) and audio_idx < len(transcription_size_list):
            audio_size = audio_size_list[audio_idx]
            transcription_size = transcription_size_list[audio_idx]
            result_tokens.extend([placeholder_token] * (audio_size + transcription_size))
        
        audio_idx += 1
        last_end = match.end()
    
    # Add remaining text after last match
    suffix_text = text[last_end:]
    if suffix_text:
        result_tokens.extend(tokenizer.tokenize(suffix_text, add_special_tokens=False))
    
    # Convert tokens back to string
    processed_text = tokenizer.convert_tokens_to_string(result_tokens)
    
    return processed_text, start_positions

def _resolve_audio_filepath(audio_filepath):
    """
    There might be some ext "name" mismatch in the audio_filepath
    """

    # First check if the original file exists
    if os.path.exists(audio_filepath):
        return audio_filepath
    
    # If not, try to resolve to .wav version
    base, ext = os.path.splitext(audio_filepath)
    wav_filepath = base + ".wav"
    
    if os.path.exists(wav_filepath):
        return wav_filepath
    
    raise FileNotFoundError(f"Audio file {audio_filepath} does not exist")


class BaseCollateFn(object):
    """
    Collate function for BaseAudioTextDataset.
    """
    def __init__(self, data_cfg: DictConfig, tokenizer: AutoTokenizer, processor: AutoFeatureExtractor):
        """
        Initialize the collate function.

        Args:
            data_cfg (DictConfig): Data configuration.
            tokenizer (AutoTokenizer): Tokenizer.
            processor (AutoFeatureExtractor): Feature extractor.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = data_cfg.max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare training batch.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, Any]: Batched data.
        """
        # ====================
        # Prepare text inputs
        # ====================

        # audio_text_inputs = audio_context + target
        assert self.tokenizer.padding_side == "left", f"padding_side must be left, but got {self.tokenizer.padding_side}"
        audio_text_inputs = self.tokenizer(
            [item["audio_context"] + item["target"] for item in batch],
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )

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

        # ====================
        # Prepare text and audio inputs
        # ====================
        
        batch_features = []
        batch_start_positions = [] # list of tuple (batch_idx, start_position)
        batch_transcription_ids = []

        audio_context_batch_start_positions = []

        audio_start_answer_positions = []
        for i, item in enumerate(batch):
            # Prepare labels
            total_length = audio_text_inputs["length"][i]
            audio_context_length = len(self.tokenizer.tokenize(item["audio_context"]))
            pad_length = total_length - audio_text_inputs["attention_mask"][i].sum()
            
            start_answer_position = pad_length + audio_context_length
            labels[i, start_answer_position:] = audio_text_inputs['input_ids'][i, start_answer_position:]
            
            audio_start_answer_positions.append(start_answer_position)
      

            # Prepare audio inputs
            for audio_dict in item["processed_audios"]:
                feature = AudioSegment.from_file(
                    audio_dict["audio"],
                    target_sr=16000,
                    channel_selector="average" # average two channels
                ).samples
                batch_features.append(feature)

            for transcription in item["transcription_list"]:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long()
                )
            
            for start_position in item["start_positions"]:
                batch_start_positions.append((i, start_position + pad_length))

            # Prepare context
            total_length = audio_context_inputs["length"][i]
            pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()
            for start_position in item["start_positions"]:
                audio_context_batch_start_positions.append((i, start_position + pad_length))


        batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features

        assert len(batch_features) == len(batch_start_positions) == len(batch_transcription_ids), f"batch_features({len(batch_features)}), batch_start_positions({len(batch_start_positions)}), batch_transcription_ids({len(batch_transcription_ids)}), must have the same length."

        assert len(batch) == len(audio_text_inputs["input_ids"]) == len(audio_text_inputs["attention_mask"]) == len(labels), f"batch, audio_text_inputs, labels must have the same length."

        # for inp, lab, att in zip(context_inputs["input_ids"][0], labels[0], context_inputs["attention_mask"][0]):
        #     print(0 if lab == -100 else self.tokenizer.decode(lab), "    ", self.tokenizer.decode(inp), f"({inp.item()})", "    ", att.item())
        
        new_batch = {
            # audio text sequence
            "input_ids": audio_text_inputs['input_ids'],
            "attention_mask": audio_text_inputs['attention_mask'],
            "labels": labels,
            "audio_start_answer_positions": audio_start_answer_positions,

            # audio batch features
            "batch_features": batch_features, # tensor of shape (batch_size, audio_seq)
            "batch_transcription_ids": batch_transcription_ids, # list of list of transcription ids
            "batch_start_positions": batch_start_positions, # list of tuple (batch_idx, start_position)

            # context inputs (for evaluation)
            "context_input_ids": audio_context_inputs['input_ids'],
            "context_attention_mask": audio_context_inputs['attention_mask'],
            "context_batch_start_positions": audio_context_batch_start_positions, # list of tuple (batch_idx, start_position)

            "metadata": [item for item in batch]
        }

        return new_batch
        

class BaseAudioTextDataset(object):
    """
    Dataset format(jsonl):
    {
        "id": "123",
        "messages": [
            {
                "role": "user",
                "content": "Describe the following audio. <|AUDIO|>",
                "audios": [
                    {
                        "audio": "path/to/audio/file.wav",
                        "text": "Hello world", # leave ' ' if no transcription
                    }
                ]
            }
        ],
        "response": "The first audio is more similar to the user's query."
    }
    """
    def __init__(self, 
                 cfg,
                 data_cfg,
                 tokenizer,
                 processor
                ):
        """
        audio_locator: tokens in dataset that will be tokenized as a token.
        placeholder_token: token in the LLM embedding table that will be replaced with audio in forward pass. Use a least used or reserved token to avoid conflicts.

        Here is an audio: <|AUDIO|>. bla bla bla
        -->
        Here is an audio: <|placeholder_token|><|placeholder_token|><|placeholder_token|><|placeholder_token|> bla bla bla <|eos_token|>
        """
        self.audio_locator = cfg.model.audio_locator
        self.placeholder_token = cfg.model.placeholder_token

        self.data_root = data_cfg.data_root
        self.manifest_filepaths = [data_cfg.manifest_filepaths] if isinstance(data_cfg.manifest_filepaths, str) else data_cfg.manifest_filepaths # convert to list if single file

        for manifest_filepath in self.manifest_filepaths:
            logging.info(f"manifest_filepath: {manifest_filepath}")
        
        self.prompt_size = cfg.model.connector.prompt_size
        self.tokenizer = tokenizer
        self.processor = processor

        self.dataset = datasets.load_dataset(
            "json", data_files=[resolve_filepath(manifest_filepath) for manifest_filepath in self.manifest_filepaths]
        )["train"]

        # Disable caching completely to avoid multi-process conflicts
        datasets.disable_caching()
        
        self.dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=256,
            num_proc=1,
            load_from_cache_file=False,
            keep_in_memory=True  # Keep in memory to avoid file conflicts
        )

        # Filter out invalid samples (empty messages or no audio context)
        original_len = len(self.dataset)
        self.dataset = self.dataset.filter(lambda x: x["length"] > 0 and len(x["audio_context"]) > 0 and len(x["processed_audios"]) > 0)
        filtered_len = len(self.dataset)
        skipped_count = original_len - filtered_len
        skip_ratio = (skipped_count / original_len * 100) if original_len > 0 else 0
        
        logging.info(f"="*60)
        logging.info(f"Dataset Statistics:")
        logging.info(f"  Total samples: {original_len}")
        logging.info(f"  Valid samples: {filtered_len}")
        logging.info(f"  Skipped samples: {skipped_count} ({skip_ratio:.2f}%)")
        logging.info(f"="*60)
        
        if filtered_len == 0:
            logging.error("No valid samples found! Please check your data format.")
            logging.error("Expected format: messages should be a non-empty list with audio markers")

        self.collate_fn = BaseCollateFn(data_cfg=data_cfg, tokenizer=self.tokenizer, processor=self.processor)



    def _preprocess_function(self, examples):
        audio_context_list = []
        start_positions_list = []
        audio_list = []
        transcription_list = []
        
        # Counters for logging
        total_samples = len(examples["messages"])
        skipped_empty_messages = 0
        skipped_no_audio_markers = 0
        processed_samples = 0

        # gather all audios from all messages (don't modify original audios column)
        batch_audios = []
        for idx, (messages, sample_id) in enumerate(zip(examples["messages"], examples["id"])):
            _a = []
            if messages:  # Check if messages is not None or empty
                for message in messages:
                    if message and message.get("audios"):
                        for audio_item in message["audios"]:
                            # Make a copy and fix null audio path
                            new_item = dict(audio_item)
                            if new_item.get("audio") is None:
                                new_item["audio"] = sample_id
                            _a.append(new_item)
            batch_audios.append(_a)
        
        # Use a temporary variable, don't overwrite original audios column

        for idx, (messages, audios) in enumerate(zip(examples["messages"], batch_audios)):
            # Skip empty messages
            if not messages or len(messages) == 0:
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                transcription_list.append([])
                skipped_empty_messages += 1
                continue
                
            try:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ) # for audio_locator
            except Exception as e:
                logging.error(f"Error processing messages at index {idx}: {messages}")
                raise e
            
            # modify audio_filepath to be absolute path
            new_audios = []
            for audio_dict in audios:
                audio_dict["audio"] = _resolve_audio_filepath(
                    os.path.join(self.data_root, audio_dict["audio"])
                )
                new_audios.append(audio_dict)

            # Replace audio_locator with [placeholder_token] * (prompt_size + transcription_size) in context and get start positions
            # this serves as placeholder for features and transcription
            # Transcript the speech <features+transcription>
            # Now assume all audios have the same size

            audio_size_list = [self.prompt_size] * len(audios)

            # truncate transcriptions to 100 tokens
            # Handle missing "text" field - default to empty string
            transcriptions = [
                self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.tokenize(audio.get("text", " ") or " ", add_special_tokens=False)[:100]
                ) for audio in audios
            ]
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in transcriptions
            ]
            transcription_list.append(transcriptions)

            # Check for audio format - support both <|AUDIO|> and <start_audio>...<end_audio>
            has_start_end_tags = "<start_audio>" in audio_context and "<end_audio>" in audio_context
            num_audio_locators = audio_context.count(self.audio_locator)
            
            if has_start_end_tags:
                # Use <start_audio>...<end_audio> format
                # Returns processed string directly
                audio_context, start_positions = _prepare_audio_context_with_start_end_tags(
                    text=audio_context,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token,
                    tokenizer=self.tokenizer
                )
            elif num_audio_locators > 0:
                # Use <|AUDIO|> format
                # Returns token list, need to convert to string
                audio_context_tokens, start_positions = _prepare_audio_context_and_start_positions(
                    token_list=self.tokenizer.tokenize(audio_context), 
                    audio_locator=self.audio_locator,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token
                )
                audio_context = self.tokenizer.convert_tokens_to_string(audio_context_tokens)
            else:
                # No audio markers found - skip this sample
                audio_context_list.append("")
                start_positions_list.append([])
                audio_list.append([])
                skipped_no_audio_markers += 1
                continue
            audio_context_list.append(audio_context)
            start_positions_list.append(start_positions)
            audio_list.append(new_audios)
            processed_samples += 1
        
        examples["audio_context"] = audio_context_list
        examples["start_positions"] = start_positions_list # list of list of start positions
        examples["transcription_list"] = transcription_list # list of list of transcription ids
        examples["processed_audios"] = audio_list # list of list of audio dicts (use different name to avoid type conflicts)

        # Handle target and length calculation, skip invalid samples
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
