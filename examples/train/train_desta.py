"""
DeSTA2.5-Audio Training Script

This script trains the DeSTA2.5-Audio model using HuggingFace Transformers Trainer.
Supports multi-GPU training with SLURM and torchrun.
"""
import os

# Disable wandb and verbose logging on non-main processes BEFORE any other imports
_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
_global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
_is_main_process = (_local_rank == 0 and _global_rank == 0)

if not _is_main_process:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    # Suppress logging on non-main processes
    import logging
    logging.basicConfig(level=logging.WARNING)

import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config
from desta.trainer.desta_trainer import DeSTA25Trainer
from desta.trainer.data.simple_dataset import BaseAudioTextDataset
from desta.utils.utils import run


def setup_logging(is_main_process: bool = True):
    """Configure logging format. Only main process logs at INFO level."""
    root_logger = logging.getLogger()
    
    if not is_main_process:
        root_logger.setLevel(logging.WARNING)
        return
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def load_pretrained_weights(model: DeSTA25AudioModel, checkpoint_path: str):
    """Load pretrained weights from a checkpoint file."""
    logging.info(f"Loading pretrained weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
    
    # Remove "model." prefix if present (from PyTorch Lightning)
    new_state_dict = {
        (k[6:] if k.startswith("model.") else k): v 
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=False)


def log_git_info():
    """Log git information for reproducibility."""
    try:
        logging.info(f"Git commit: {run('git rev-parse HEAD')}")
        logging.info(f"Git branch: {run('git branch --show-current')}")
        logging.info(f"Working directory: {run('pwd')}")
    except Exception:
        pass


def create_model(cfg: DictConfig) -> DeSTA25AudioModel:
    """Create and configure the DeSTA25 model."""
    model_config = DeSTA25Config(
        llm_model_id=cfg.model.llm.model_id,
        encoder_model_id=cfg.model.encoder.model_id,
        connector_mode=cfg.model.connector.mode,
        qformer_num_hidden_layers=cfg.model.connector.num_hidden_layers,
        prompt_size=cfg.model.connector.prompt_size,
        use_lora=getattr(cfg.model.llm, "use_lora", False),
        audio_locator=cfg.model.audio_locator,
        placeholder_token=cfg.model.placeholder_token,
    )
    
    model = DeSTA25AudioModel(model_config)
    model.config.train_id = 30678  # Legacy ID for compatibility
    model._setup_generation()  # Setup tokenizer and processor
    
    return model


def create_training_args(cfg: DictConfig) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=cfg.exp_dir,
        num_train_epochs=cfg.trainer.max_epochs,
        per_device_train_batch_size=cfg.dataset.train_ds.batch_size,
        per_device_eval_batch_size=cfg.dataset.validation_ds.batch_size,
        gradient_accumulation_steps=cfg.trainer.accumulate_grad_batches,
        learning_rate=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        warmup_steps=cfg.optim.sched.warmup_steps,
        logging_steps=cfg.trainer.log_every_n_steps,
        save_strategy="epoch" if cfg.trainer.enable_checkpointing else "no",
        eval_strategy="steps" if isinstance(cfg.trainer.val_check_interval, int) else "epoch",
        eval_steps=cfg.trainer.val_check_interval if isinstance(cfg.trainer.val_check_interval, int) else None,
        bf16="bf16" in cfg.trainer.precision,
        fp16="fp16" in cfg.trainer.precision,
        optim="adafactor",
        report_to="wandb",
        run_name=cfg.name,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=getattr(cfg.trainer, "gradient_checkpointing", False),
        dataloader_num_workers=getattr(cfg.dataset.train_ds, "num_workers", 4),
        dataloader_pin_memory=getattr(cfg.dataset.train_ds, "pin_memory", True),
    )


@hydra.main(config_path="config", config_name="desta25", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Setup
    os.makedirs(cfg.exp_dir, exist_ok=True)
    setup_logging(_is_main_process)
    
    if _is_main_process:
        log_git_info()
    
    # Parse checkpoint configs
    cfg.resume_from_checkpoint = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint != "null" else None
    cfg.init_from_pretrained_weights = cfg.init_from_pretrained_weights if cfg.init_from_pretrained_weights != "null" else None
    
    assert not (cfg.resume_from_checkpoint and cfg.init_from_pretrained_weights), \
        "Cannot provide both resume_from_checkpoint and init_from_pretrained_weights"
    
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create model
    model = create_model(cfg)
    
    # Load pretrained weights if specified
    if cfg.init_from_pretrained_weights:
        load_pretrained_weights(model, cfg.init_from_pretrained_weights)
    
    # Create datasets
    train_dataset = BaseAudioTextDataset(
        cfg=cfg,
        data_cfg=cfg.dataset.train_ds,
        tokenizer=model.tokenizer,
        processor=model.processor
    )
    
    val_dataset = BaseAudioTextDataset(
        cfg=cfg,
        data_cfg=cfg.dataset.validation_ds,
        tokenizer=model.tokenizer,
        processor=model.processor
    )
    
    # Create trainer
    training_args = create_training_args(cfg)
    
    trainer = DeSTA25Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        cfg=cfg
    )
    
    # Save config
    OmegaConf.save(cfg, f"{cfg.exp_dir}/config.yaml")
    
    # Train
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)


if __name__ == "__main__":
    main()
