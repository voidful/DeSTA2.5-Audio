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
    # Extract ORCA config if present
    orca_cfg = cfg.model.get("orca", {})
    
    # Extract ORCA-R1 config if present
    orca_r1_cfg = cfg.model.get("orca_r1", {})

    # Extract variational/S1 config. Older experiment YAMLs used a few names,
    # so accept all of them and map to DeSTA25Config's flat fields.
    variational_cfg = cfg.model.get("variational_grouping", cfg.model.get("variational", {}))
    orca_variational_cfg = orca_r1_cfg.get("variational_grouping", orca_r1_cfg.get("variational", {}))
    s1_cfg = cfg.model.get("s1", cfg.model.get("s1_variational", {}))
    orca_s1_cfg = orca_r1_cfg.get("s1", orca_r1_cfg.get("s1_variational", {}))
    variational_grouping_enabled = cfg.model.get(
        "variational_grouping_enabled",
        variational_cfg.get(
            "enabled",
            orca_variational_cfg.get(
                "enabled",
                orca_r1_cfg.get("variational_grouping_enabled", False),
            ),
        ),
    )
    variational_kl_weight = cfg.model.get(
        "variational_kl_weight",
        variational_cfg.get(
            "kl_weight",
            orca_variational_cfg.get(
                "kl_weight",
                orca_r1_cfg.get("variational_kl_weight", 0.01),
            ),
        ),
    )
    s1_kl_annealing_enabled = cfg.model.get(
        "s1_kl_annealing_enabled",
        s1_cfg.get(
            "kl_annealing_enabled",
            orca_s1_cfg.get(
                "kl_annealing_enabled",
                orca_r1_cfg.get("s1_kl_annealing_enabled", False),
            ),
        ),
    )
    s1_kl_annealing_warmup_steps = cfg.model.get(
        "s1_kl_annealing_warmup_steps",
        s1_cfg.get(
            "kl_annealing_warmup_steps",
            orca_s1_cfg.get(
                "kl_annealing_warmup_steps",
                orca_r1_cfg.get("s1_kl_annealing_warmup_steps", 2000),
            ),
        ),
    )
    s1_kl_annealing_cycle_steps = cfg.model.get(
        "s1_kl_annealing_cycle_steps",
        s1_cfg.get(
            "kl_annealing_cycle_steps",
            orca_s1_cfg.get(
                "kl_annealing_cycle_steps",
                orca_r1_cfg.get("s1_kl_annealing_cycle_steps", 0),
            ),
        ),
    )
    s1_free_bits = cfg.model.get(
        "s1_free_bits",
        s1_cfg.get(
            "free_bits",
            orca_s1_cfg.get(
                "free_bits",
                orca_r1_cfg.get("s1_free_bits", 0.0),
            ),
        ),
    )
    s1_mu_invariance_enabled = cfg.model.get(
        "s1_mu_invariance_enabled",
        s1_cfg.get(
            "mu_invariance_enabled",
            orca_s1_cfg.get(
                "mu_invariance_enabled",
                orca_r1_cfg.get("s1_mu_invariance_enabled", False),
            ),
        ),
    )
    s1_mu_invariance_weight = cfg.model.get(
        "s1_mu_invariance_weight",
        s1_cfg.get(
            "mu_invariance_weight",
            orca_s1_cfg.get(
                "mu_invariance_weight",
                orca_r1_cfg.get("s1_mu_invariance_weight", 0.1),
            ),
        ),
    )
    s1_inference_alpha = cfg.model.get(
        "s1_inference_alpha",
        s1_cfg.get(
            "inference_alpha",
            orca_s1_cfg.get(
                "inference_alpha",
                orca_r1_cfg.get("s1_inference_alpha", 0.5),
            ),
        ),
    )
    s1_augment_freq_mask = cfg.model.get(
        "s1_augment_freq_mask",
        s1_cfg.get(
            "augment_freq_mask",
            orca_s1_cfg.get(
                "augment_freq_mask",
                orca_r1_cfg.get("s1_augment_freq_mask", 0.1),
            ),
        ),
    )
    s1_augment_time_mask = cfg.model.get(
        "s1_augment_time_mask",
        s1_cfg.get(
            "augment_time_mask",
            orca_s1_cfg.get(
                "augment_time_mask",
                orca_r1_cfg.get("s1_augment_time_mask", 0.1),
            ),
        ),
    )
    modality_dpo_cfg = cfg.model.get("modality_dpo", {})
    modality_dpo_enabled = cfg.model.get(
        "modality_dpo_enabled",
        modality_dpo_cfg.get("enabled", orca_r1_cfg.get("modality_dpo_enabled", False)),
    )
    modality_dpo_beta = cfg.model.get(
        "modality_dpo_beta",
        modality_dpo_cfg.get("beta", orca_r1_cfg.get("modality_dpo_beta", 0.1)),
    )
    asr_dropout_cfg = cfg.model.get("asr_dropout", {})
    asr_dropout_prob = cfg.model.get(
        "asr_dropout_prob",
        asr_dropout_cfg.get("prob", orca_r1_cfg.get("asr_dropout_prob", 0.0)),
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
        # ORCA-DeSTA configuration
        orca_enabled=orca_cfg.get("enabled", False),
        orca_local_enabled=orca_cfg.get("local_enabled", True),
        orca_global_cross_attn=orca_cfg.get("global_cross_attn", False),
        orca_deep_injection_enabled=orca_cfg.get("deep_injection_enabled", True),
        orca_audio_position_scale=orca_cfg.get("audio_position_scale", 5.0),
        orca_global_num_tokens=orca_cfg.get("global_num_tokens", 4),
        orca_local_downsample=orca_cfg.get("local_downsample", 4),
        orca_local_kernel_size=orca_cfg.get("local_kernel_size", 7),
        orca_gate_init=orca_cfg.get("gate_init", 0.1),
        orca_ortho_weight_global=orca_cfg.get("ortho_weight_global", 0.01),
        orca_ortho_diversity_weight=orca_cfg.get("ortho_diversity_weight", 0.01),
        orca_ortho_weight_qformer_local=orca_cfg.get("ortho_weight_qformer_local", 0.01),
        orca_align_weight_local=orca_cfg.get("align_weight_local", 0.05),
        # ORCA-R1 configuration
        orca_r1_num_groups=orca_r1_cfg.get("num_groups", 8),
        orca_r1_queries_per_group=orca_r1_cfg.get("queries_per_group", 8),
        orca_r1_inter_group_weight=orca_r1_cfg.get("inter_group_weight", 0.1),
        orca_r1_intra_group_weight=orca_r1_cfg.get("intra_group_weight", 0.01),
        orca_r1_iv_weight=orca_r1_cfg.get("iv_weight", 0.1),
        orca_r1_acd_alpha=orca_r1_cfg.get("acd_alpha", 0.5),
        # Variational grouping / S1 configuration
        variational_grouping_enabled=variational_grouping_enabled,
        variational_kl_weight=variational_kl_weight,
        s1_kl_annealing_enabled=s1_kl_annealing_enabled,
        s1_kl_annealing_warmup_steps=s1_kl_annealing_warmup_steps,
        s1_kl_annealing_cycle_steps=s1_kl_annealing_cycle_steps,
        s1_free_bits=s1_free_bits,
        s1_mu_invariance_enabled=s1_mu_invariance_enabled,
        s1_mu_invariance_weight=s1_mu_invariance_weight,
        s1_inference_alpha=s1_inference_alpha,
        s1_augment_freq_mask=s1_augment_freq_mask,
        s1_augment_time_mask=s1_augment_time_mask,
        modality_dpo_enabled=modality_dpo_enabled,
        modality_dpo_beta=modality_dpo_beta,
        asr_dropout_prob=asr_dropout_prob,
    )
    logging.info(
        "Variational grouping config: enabled=%s, kl_weight=%s, "
        "s1_mu_invariance=%s, s1_kl_annealing=%s, modality_dpo=%s, asr_dropout=%s",
        model_config.variational_grouping_enabled,
        model_config.variational_kl_weight,
        model_config.s1_mu_invariance_enabled,
        model_config.s1_kl_annealing_enabled,
        model_config.modality_dpo_enabled,
        model_config.asr_dropout_prob,
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
        report_to="wandb" if _is_main_process else "none",
        run_name=cfg.name,
        remove_unused_columns=False,
        label_names=["labels"],
        # Enable find_unused_parameters for ORCA/ORCA-R1 mode (some heads may not be used)
        ddp_find_unused_parameters=cfg.model.connector.mode in ["orca_hybrid", "orca_r1"],
        gradient_checkpointing=getattr(cfg.trainer, "gradient_checkpointing", False),
        dataloader_num_workers=getattr(cfg.dataset.train_ds, "num_workers", 4),
        dataloader_pin_memory=getattr(cfg.dataset.train_ds, "pin_memory", True),
        # Disabled: auto_find_batch_size can cause DDP parameter desync issues
        auto_find_batch_size=False,
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


if __name__ == "__main__":
    main()
