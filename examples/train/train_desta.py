"""
DeSTA2.5-Audio Training Script

This script trains the DeSTA2.5-Audio model using HuggingFace Transformers Trainer.
Supports multi-GPU training with SLURM and torchrun.
"""
# ============================================================================
# SECURITY PATCH: Bypass torch.load security check for older PyTorch versions
# This MUST be done BEFORE any transformers imports (CVE-2025-32434)
# Safe when loading trusted checkpoints created by the user
# ============================================================================
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# Also patch the trainer module directly since it may have already imported the function
import transformers.trainer
transformers.trainer.check_torch_load_is_safe = lambda: None
# ============================================================================

import os
import sys

# Disable wandb and verbose logging on non-main processes BEFORE any other imports
_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
_global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
_is_main_process = (_local_rank == 0 and _global_rank == 0)

if not _is_main_process:
    # Suppress verbose output on non-main processes (but keep stderr for errors)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["DATASETS_VERBOSITY"] = "error"
    os.environ["WANDB_SILENT"] = "true"
    import logging
    logging.basicConfig(level=logging.ERROR)
    # Only redirect stdout (keep stderr for error messages)
    sys.stdout = open(os.devnull, 'w')

import logging
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from desta.models.modeling_desta25 import (
    GROUPWISE_ORTHO_ALIASES,
    DeSTA25AudioModel,
    DeSTA25Config,
)
from desta.trainer.desta_trainer import DeSTA25Trainer
from desta.trainer.data.simple_dataset import BaseAudioTextDataset
from desta.utils.utils import run


def setup_logging(is_main_process: bool = True):
    """Configure logging format. Only main process logs at INFO level."""
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if not is_main_process:
        root_logger.setLevel(logging.WARNING)
        # Add a null handler or a warning-only handler if needed, 
        # but usually we just want silence on non-main
        return
    
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


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
    groupwise_cfg = cfg.model.get(
        "groupwise_ortho",
        cfg.model.get("orca_desta", cfg.model.get("orca_r1", {})),
    )
    
    model_config = DeSTA25Config(
        llm_model_id=cfg.model.llm.model_id,
        encoder_model_id=cfg.model.encoder.model_id,
        connector_mode=cfg.model.connector.mode,
        qformer_num_hidden_layers=cfg.model.connector.num_hidden_layers,
        prompt_size=cfg.model.connector.prompt_size,
        use_lora=getattr(cfg.model.llm, "use_lora", False),
        use_flash_attention=cfg.model.connector.get("use_flash_attention", True),
        audio_locator=cfg.model.audio_locator,
        placeholder_token=cfg.model.placeholder_token,
        # DeSTA25Config keeps the old orca_r1_* attribute names for checkpoint
        # compatibility, but the active method is groupwise_ortho only.
        orca_r1_num_groups=groupwise_cfg.get("num_groups", 8),
        orca_r1_queries_per_group=groupwise_cfg.get("queries_per_group", 8),
        orca_r1_inter_group_weight=groupwise_cfg.get("inter_group_weight", 0.1),
        orca_r1_intra_group_weight=groupwise_cfg.get("intra_group_weight", 0.01),
    )
    logging.info(
        "Groupwise-orthogonal connector config: groups=%s, queries_per_group=%s, "
        "inter_group_weight=%s, intra_group_weight=%s",
        model_config.orca_r1_num_groups,
        model_config.orca_r1_queries_per_group,
        model_config.orca_r1_inter_group_weight,
        model_config.orca_r1_intra_group_weight,
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
        max_grad_norm=getattr(cfg.trainer, "gradient_clip_val", 1.0),
        logging_steps=cfg.trainer.log_every_n_steps,
        save_strategy="epoch" if cfg.trainer.enable_checkpointing else "no",
        eval_strategy="steps" if isinstance(cfg.trainer.val_check_interval, int) else "epoch",
        eval_steps=cfg.trainer.val_check_interval if isinstance(cfg.trainer.val_check_interval, int) else None,
        bf16="bf16" in cfg.trainer.precision,
        fp16="fp16" in cfg.trainer.precision,
        optim="adafactor",
        report_to="wandb" if _is_main_process else "none",
        run_name=cfg.name,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=cfg.model.connector.mode in GROUPWISE_ORTHO_ALIASES,
        gradient_checkpointing=getattr(cfg.trainer, "gradient_checkpointing", False),
        dataloader_num_workers=getattr(cfg.dataset.train_ds, "num_workers", 4),
        dataloader_pin_memory=getattr(cfg.dataset.train_ds, "pin_memory", True),
        # Disabled: auto_find_batch_size can cause DDP parameter desync issues
        auto_find_batch_size=False,
    )


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
        processing_class=model.tokenizer,  # Pass tokenizer for new transformers API
        cfg=cfg
    )
    
    # Save config
    OmegaConf.save(cfg, f"{cfg.exp_dir}/config.yaml")
    
    # === Safety First: Eval before train to catch logic errors early ===
    if not cfg.resume_from_checkpoint:
        logging.info("Running initial evaluation to verify model and trainer logic...")
        trainer.evaluate()
        
        logging.info(f"Saving initial checkpoint to {cfg.exp_dir}/checkpoint-initial")
        trainer.save_model(os.path.join(cfg.exp_dir, "checkpoint-initial"))
    
    # Train
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)


def _load_config_with_defaults(config_path, config_dir, _seen=None):
    """Load a YAML config, resolving a top-level Hydra-style `defaults:` list.

    Each entry in `defaults` is a string naming another config file in
    `config_dir` (without the `.yaml` suffix). Parents are merged in list order,
    then the current file's own keys override them. Nesting is supported.
    """
    config_path = os.path.abspath(config_path)
    _seen = _seen or set()
    if config_path in _seen:
        raise ValueError(f"Cyclic defaults reference detected at {config_path}")
    _seen = _seen | {config_path}

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        return cfg

    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg

    merged = OmegaConf.create({})
    for entry in defaults:
        if not isinstance(entry, str):
            raise ValueError(
                f"defaults entries must be strings (got {entry!r} in {config_path})"
            )
        parent_path = os.path.join(config_dir, f"{entry}.yaml")
        if not os.path.exists(parent_path):
            raise FileNotFoundError(
                f"defaults entry '{entry}' in {config_path} -> missing {parent_path}"
            )
        parent_cfg = _load_config_with_defaults(parent_path, config_dir, _seen)
        merged = OmegaConf.merge(merged, parent_cfg)

    return OmegaConf.merge(merged, cfg)


if __name__ == "__main__":
    import argparse
    import warnings
    # Suppress UserWarning about interpolations when initially loading
    warnings.filterwarnings("ignore", category=UserWarning, module="omegaconf")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config")
    parser.add_argument("--config-name", type=str, default="desta25")
    args, unknown = parser.parse_known_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, args.config_path)
    main_config_path = os.path.join(config_dir, f"{args.config_name}.yaml")

    if not os.path.exists(main_config_path):
        raise FileNotFoundError(f"Config file not found: {main_config_path}")

    cfg = _load_config_with_defaults(main_config_path, config_dir)

    # Process unknown args to find overrides and additional configs (like +dataset=)
    cli_args = []
    for arg in unknown:
        # Strip hydra's + or ++ prefixes
        clean_arg = arg.lstrip("+")

        # Check if it's loading a sub-config, e.g. dataset=DestaAQA-5M_4b_ablation
        if "=" in clean_arg:
            key, val = clean_arg.split("=", 1)
            # Check if this maps to a yaml file in the subfolder (e.g. config/dataset/XYZ.yaml)
            sub_config_path = os.path.join(config_dir, key, f"{val}.yaml")
            if os.path.exists(sub_config_path):
                sub_cfg = _load_config_with_defaults(
                    sub_config_path, os.path.join(config_dir, key)
                )
                # Assign sub-config to the key in the main config
                cfg[key] = sub_cfg
                continue

        cli_args.append(clean_arg)

    # Apply CLI overrides
    if cli_args:
        cli_cfg = OmegaConf.from_cli(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    main(cfg)
