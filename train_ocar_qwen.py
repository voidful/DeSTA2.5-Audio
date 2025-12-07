"""
OCAR-Qwen Training Script

Training script for OCAR-Qwen model using HuggingFace Trainer.

Features:
- Load Qwen2-4B-Instruct and Whisper-large-v3
- Apply LoRA to all linear layers in Qwen
- Freeze Whisper encoder
- Make Adapter & Cross-Attention layers trainable
- Support OCAR loss weights as arguments
"""

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import load_dataset

# Local imports
from audio_qwen.modeling_ocar_qwen import OCARQwenConfig, OCARQwenForCausalLM
from data.ocar_collator import OCARCollator, OCARCollatorConfig, create_ocar_collator


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    
    llm_model_id: str = field(
        default="Qwen/Qwen2-4B-Instruct",
        metadata={"help": "LLM model ID or path"}
    )
    encoder_model_id: str = field(
        default="openai/whisper-large-v3",
        metadata={"help": "Whisper encoder model ID"}
    )
    global_tokens: int = field(
        default=32,
        metadata={"help": "Number of global Q-Former tokens"}
    )
    local_stride: int = field(
        default=4,
        metadata={"help": "Stride for local Conv1d downsampling"}
    )
    qformer_layers: int = field(
        default=2,
        metadata={"help": "Number of Q-Former layers"}
    )
    cross_attn_num_heads: int = field(
        default=16,
        metadata={"help": "Number of cross-attention heads"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: str = field(
        default="all",
        metadata={"help": "LoRA target modules: 'all' for all linear layers, or comma-separated list"}
    )


@dataclass
class OCARLossArguments:
    """Arguments for OCAR loss weights."""
    
    w_llm: float = field(
        default=1.0,
        metadata={"help": "Weight for LLM (next token prediction) loss"}
    )
    w_ortho_text: float = field(
        default=0.1,
        metadata={"help": "Weight for orthogonality-text loss"}
    )
    w_ortho_self: float = field(
        default=0.1,
        metadata={"help": "Weight for orthogonality-self loss"}
    )
    w_prosody_global: float = field(
        default=0.25,
        metadata={"help": "Weight for global prosody loss"}
    )
    w_prosody_local: float = field(
        default=0.25,
        metadata={"help": "Weight for local prosody loss"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to training data (JSON/JSONL format)"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data (JSON/JSONL format)"}
    )
    max_audio_length: float = field(
        default=30.0,
        metadata={"help": "Maximum audio length in seconds"}
    )
    audio_column: str = field(
        default="audio_path",
        metadata={"help": "Column name for audio file path"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column name for text/transcription"}
    )


class OCARTrainer(Trainer):
    """
    Custom Trainer for OCAR-Qwen model.
    
    Extends HuggingFace Trainer with:
    - Custom loss logging for all OCAR components
    - Proper handling of multi-component loss
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = {
            "loss": [],
            "loss_llm": [],
            "loss_ortho_text": [],
            "loss_ortho_self": [],
            "loss_prosody_global": [],
            "loss_prosody_local": [],
        }
        
    def compute_loss(
        self,
        model: OCARQwenForCausalLM,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        Compute loss with detailed logging of each component.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        if self.state.global_step % self.args.logging_steps == 0:
            # Log individual loss components if available
            if hasattr(outputs, 'loss_dict'):
                for key, value in outputs.loss_dict.items():
                    if key in self.loss_history:
                        self.loss_history[key].append(value)
                        
        if return_outputs:
            return loss, outputs
        return loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log to include OCAR loss components."""
        # Add average losses if available
        for key in list(self.loss_history.keys()):
            if self.loss_history[key]:
                logs[f"train/{key}"] = sum(self.loss_history[key]) / len(self.loss_history[key])
                self.loss_history[key] = []
                
        super().log(logs)


def get_lora_target_modules(model, target_modules_str: str) -> List[str]:
    """
    Get LoRA target modules.
    
    Args:
        model: The model to apply LoRA to
        target_modules_str: 'all' for all linear layers, or comma-separated list
        
    Returns:
        List of module names to target
    """
    if target_modules_str.lower() == "all":
        # Find all linear layers in the LLM
        target_modules = []
        for name, module in model.llm.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Extract the last part of the module name
                target_name = name.split(".")[-1]
                if target_name not in target_modules:
                    target_modules.append(target_name)
        return list(set(target_modules))
    else:
        return [m.strip() for m in target_modules_str.split(",")]


def apply_lora(model: OCARQwenForCausalLM, lora_args: LoraArguments) -> OCARQwenForCausalLM:
    """
    Apply LoRA to the LLM component of the model.
    
    Args:
        model: OCAR-Qwen model
        lora_args: LoRA configuration arguments
        
    Returns:
        Model with LoRA applied to LLM
    """
    target_modules = get_lora_target_modules(model, lora_args.target_modules)
    logger.info(f"Applying LoRA to modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    # Apply LoRA to LLM only
    model.llm = get_peft_model(model.llm, lora_config)
    
    logger.info("LoRA applied successfully")
    logger.info(f"Trainable LLM parameters: {model.llm.print_trainable_parameters()}")
    
    return model


def create_model(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    loss_args: OCARLossArguments,
) -> OCARQwenForCausalLM:
    """
    Create and configure OCAR-Qwen model.
    
    Args:
        model_args: Model configuration
        lora_args: LoRA configuration
        loss_args: Loss weight configuration
        
    Returns:
        Configured model
    """
    logger.info("Creating OCAR-Qwen model...")
    
    config = OCARQwenConfig(
        llm_model_id=model_args.llm_model_id,
        encoder_model_id=model_args.encoder_model_id,
        global_tokens=model_args.global_tokens,
        local_stride=model_args.local_stride,
        qformer_layers=model_args.qformer_layers,
        cross_attn_num_heads=model_args.cross_attn_num_heads,
        w_llm=loss_args.w_llm,
        w_ortho_text=loss_args.w_ortho_text,
        w_ortho_self=loss_args.w_ortho_self,
        w_prosody_global=loss_args.w_prosody_global,
        w_prosody_local=loss_args.w_prosody_local,
    )
    
    model = OCARQwenForCausalLM(config)
    
    # Configure trainable parameters
    model.configure_trainable_parameters()
    
    # Apply LoRA
    model = apply_lora(model, lora_args)
    
    # Setup tokenizer and processor
    model.setup_for_training()
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def load_datasets(data_args: DataArguments, tokenizer):
    """
    Load and prepare datasets.
    
    Args:
        data_args: Data configuration
        tokenizer: Tokenizer for text processing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info("Loading datasets...")
    
    # Load training data
    if data_args.train_data_path:
        if data_args.train_data_path.endswith(".json"):
            train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")
        elif data_args.train_data_path.endswith(".jsonl"):
            train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")
        else:
            train_dataset = load_dataset(data_args.train_data_path, split="train")
    else:
        raise ValueError("train_data_path is required")
        
    # Load evaluation data if provided
    eval_dataset = None
    if data_args.eval_data_path:
        if data_args.eval_data_path.endswith(".json") or data_args.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_dataset("json", data_files=data_args.eval_data_path, split="train")
        else:
            eval_dataset = load_dataset(data_args.eval_data_path, split="test")
            
    # Preprocess function
    def preprocess_function(examples):
        """Tokenize text and prepare audio paths."""
        texts = examples[data_args.text_column]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        
        # Add audio paths
        if data_args.audio_column in examples:
            tokenized["audio_path"] = examples[data_args.audio_column]
            
        # Labels are same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset",
        )
        
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((
        ModelArguments,
        LoraArguments,
        OCARLossArguments,
        DataArguments,
        TrainingArguments,
    ))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from config file
        model_args, lora_args, loss_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, lora_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Log arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"LoRA arguments: {lora_args}")
    logger.info(f"Loss arguments: {loss_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    
    # Create model
    model = create_model(model_args, lora_args, loss_args)
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(data_args, model.tokenizer)
    
    # Create data collator
    collator_config = OCARCollatorConfig(
        max_audio_length=data_args.max_audio_length,
        pad_token_id=model.tokenizer.pad_token_id or 0,
    )
    data_collator = OCARCollator(
        processor=model.processor,
        tokenizer=model.tokenizer,
        config=collator_config,
    )
    
    # Create trainer
    trainer = OCARTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=model.tokenizer,
    )
    
    # Save configuration
    os.makedirs(training_args.output_dir, exist_ok=True)
    model.config.save_pretrained(training_args.output_dir)
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate if eval dataset provided
    if eval_dataset:
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
