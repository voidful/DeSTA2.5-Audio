import hydra
from omegaconf import DictConfig, OmegaConf
from desta.trainer.desta_trainer import DeSTA25Trainer
from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config
import logging
from lulutils import get_unique_filepath
import os
import torch
from desta.utils.utils import run
from transformers import TrainingArguments
from desta.trainer.data.simple_dataset import BaseAudioTextDataset

@hydra.main(config_path="config", config_name="desta25")
def main(cfg: DictConfig):
    # Disable wandb on non-main processes BEFORE any wandb import
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        os.environ["WANDB_DISABLED"] = "true"
    
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # resume training from checkpoint or init from pretrained weights
    cfg.resume_from_checkpoint = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint != "null" else None
    cfg.init_from_pretrained_weights = cfg.init_from_pretrained_weights if cfg.init_from_pretrained_weights != "null" else None
    assert cfg.resume_from_checkpoint is None or cfg.init_from_pretrained_weights is None, "Cannot provide both resume_from_checkpoint and init_from_pretrained_weights"

    root_logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    logging.info(f"Config: {cfg}")

    working_dir = os.getcwd()

    # Model Config
    model_config = DeSTA25Config(
        llm_model_id=cfg.model.llm.model_id,
        encoder_model_id=cfg.model.encoder.model_id,
        connector_mode=cfg.model.connector.mode,
        qformer_num_hidden_layers=cfg.model.connector.num_hidden_layers,
        prompt_size=cfg.model.connector.prompt_size,
        use_lora=cfg.model.llm.use_lora if hasattr(cfg.model.llm, "use_lora") else False,
        audio_locator=cfg.model.audio_locator,
        placeholder_token=cfg.model.placeholder_token,
    )

    print("="*100)
    model = DeSTA25AudioModel(model_config)
    model.config.train_id = 30678 # Keep legacy ID
    
    # remove whisper decoder (we only use Whisper decoder during inference/validation if needed, but for training we might not need it if we only use encoder)
    # However, the original code removed it. 
    # Note: If validation uses generate(), it might need decoder if it uses WhisperForConditionalGeneration.
    # The original code: del self.model.perception.whisper.model.decoder
    # But validation uses self.perception.whisper.generate() which needs decoder?
    # Let's check modeling_desta25.py again.
    # In generate(): self.perception.whisper.generate(...)
    # If decoder is deleted, this will fail.
    # But original code deleted it: 
    # del self.model.perception.whisper.model.decoder
    # del self.model.perception.whisper.proj_out
    # Maybe original validation didn't use whisper generation?
    # In modeling_desta25.py:
    # if asr_features: ... transcriptions = self.perception.whisper.generate(...)
    # So it DOES use it.
    # This implies the original code might have been broken for ASR generation during validation if it deleted the decoder, 
    # OR the original code re-initialized it or something?
    # Or maybe the original code only ran ASR if not deleted?
    # Actually, let's keep it safe and NOT delete it for now, or check if we can offload it.
    # The user request is to refactor.
    # I will comment out the deletion to be safe, or delete it if I'm sure.
    # Given the user wants to "support whisper large v3 turbo", maybe they want ASR.
    # I'll keep it.
    
    # Tokenizer & Processor
    model._setup_generation() # This sets up tokenizer and processor

    if cfg.init_from_pretrained_weights is not None:
        logging.info(f"Loading pretrained weights from {cfg.init_from_pretrained_weights}")
        state_dict = torch.load(cfg.init_from_pretrained_weights)["state_dict"]
        # Remove "model." prefix if present (from PL)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    try:
        logging.info(f"{run('git rev-parse HEAD')}")
        logging.info(f"{run('git branch --show-current')}")
        logging.info(f"{run('pwd')}")
    except:
        pass
    
    # Datasets
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

    # Training Arguments
    training_args = TrainingArguments(
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
        eval_strategy="steps" if isinstance(cfg.trainer.val_check_interval, int) else "epoch", # Simplified
        eval_steps=cfg.trainer.val_check_interval if isinstance(cfg.trainer.val_check_interval, int) else None,
        bf16="bf16" in cfg.trainer.precision,
        fp16="fp16" in cfg.trainer.precision,
        optim="adafactor", # Requested by user
        report_to="wandb" if hasattr(cfg, "wandb") else "none",
        run_name=cfg.name,
        remove_unused_columns=False, # Important for custom datasets
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    trainer = DeSTA25Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        cfg=cfg # Pass cfg for custom trainer logic
    )
    
    OmegaConf.save(cfg, f"{cfg.exp_dir}/config.yaml")

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

if __name__ == "__main__":
    main()