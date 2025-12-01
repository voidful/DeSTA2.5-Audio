"""
DeSTA2.5-Audio Custom Trainer

This module provides a custom HuggingFace Trainer for the DeSTA2.5-Audio model
with evaluation and prediction capabilities.
"""
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer

from desta.models.modeling_desta25 import DeSTA25AudioModel
from desta.utils.metrics import ConsecutiveWordsAccuracyMetric
from desta.utils.utils import run
from lulutils import get_unique_filepath


class DeSTA25Trainer(Trainer):
    """
    Custom Trainer for DeSTA2.5-Audio model.
    
    Extends HuggingFace Transformers Trainer with:
    - Custom loss computation with perplexity logging
    - Evaluation with generation and accuracy metrics
    - Prediction reports in JSONL format
    """
    
    def __init__(self, model: DeSTA25AudioModel, cfg: DictConfig, **kwargs):
        """
        Initialize the trainer.

        Args:
            model: The DeSTA25AudioModel to train
            cfg: Configuration object with model and training settings
            **kwargs: Additional arguments for parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.cfg = cfg
        self.metrics = ConsecutiveWordsAccuracyMetric()
        self.prediction_step_outputs: List[Dict[str, Any]] = []

    def compute_loss(
        self, 
        model: DeSTA25AudioModel, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """Compute loss with perplexity logging."""
        outputs = model(**inputs)
        loss = outputs.loss
        
        perplexity = torch.exp(loss)
        self.log({"train/loss": loss.item(), "train/ppl": perplexity.item()})
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self, 
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Run evaluation with generation and accuracy metrics."""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        
        all_losses = []
        all_ppls = []
        self.prediction_step_outputs = []
        
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                all_losses.append(loss.item())
                all_ppls.append(perplexity.item())
                
                self._predict_step(batch)

        # Compute average metrics
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        avg_ppl = sum(all_ppls) / len(all_ppls) if all_ppls else 0
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_ppl": avg_ppl,
        }
        
        # Generate report
        report = self._write_report()
        metrics[f"{metric_key_prefix}_accuracy"] = report.get("accuracy_by_sample", 0)
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return metrics

    def _predict_step(self, batch: Dict[str, Any]):
        """Perform a single prediction step with generation."""
        gen_kwargs = self.cfg.model.generation_kwargs
        
        generated_ids = self.model._generate_step(
            batch,
            pad_token_id=self.processing_class.eos_token_id,
            temperature=gen_kwargs.temperature,
            top_p=gen_kwargs.top_p,
            max_new_tokens=gen_kwargs.max_new_tokens,
            do_sample=gen_kwargs.do_sample,
        )

        # Replace -100 with eos_token_id for decoding
        batch["context_input_ids"][batch["context_input_ids"] == -100] = self.processing_class.eos_token_id
        batch["labels"][batch["labels"] == -100] = self.processing_class.eos_token_id
        generated_ids[generated_ids == -100] = self.processing_class.eos_token_id

        # Decode
        contexts = self.processing_class.batch_decode(batch["context_input_ids"], skip_special_tokens=False)
        labels = self.processing_class.batch_decode(batch["labels"], skip_special_tokens=True)
        preds = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Record predictions
        for context, label, pred, metadata in zip(contexts, labels, preds, batch["metadata"]):
            metadata.update({
                "context": context,
                "prediction": pred,
                "label": label,
            })
            self.prediction_step_outputs.append(metadata)

    def _write_report(self) -> Dict[str, Any]:
        """Write evaluation report to file."""
        dataset_name = "val"
        results_dir = f"{self.cfg.exp_dir}/results/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = f"{results_dir}/val@ep={self.state.epoch}-{self.state.global_step}.jsonl"

        return self._save_results(
            results=self.prediction_step_outputs,
            filepath=output_path,
            ckpt=f"ep={self.state.epoch}-{self.state.global_step}"
        )

    def _save_results(
        self, 
        results: List[Dict[str, Any]], 
        filepath: Union[str, Path],
        ckpt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save prediction results to JSONL and generate accuracy report.
        
        Args:
            results: List of prediction results
            filepath: Output file path
            ckpt: Checkpoint identifier
            
        Returns:
            Report dictionary with accuracy metrics
        """
        filepath = Path(filepath)
        categories_accuracy = defaultdict(list)
        
        # Create predictions file
        jsonl_path = Path(get_unique_filepath(filepath.parent / "preds" / filepath.name))
        os.makedirs(jsonl_path.parent, exist_ok=True)

        with open(jsonl_path, "w") as f:
            for i, result in enumerate(results):
                result["correct"] = self.metrics(result["prediction"], result["label"])
                result["index"] = i
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                categories_accuracy[result.get("category", "all")].append(result["correct"])

        # Create report
        report_path = jsonl_path.parent.parent / jsonl_path.name.replace(".jsonl", "-report.json")
        
        # Clean up results for report (remove verbose fields)
        cleaned_results = []
        for result in results:
            clean = {k: v for k, v in result.items() if k not in ["context", "audio_context"]}
            cleaned_results.append(clean)

        total_correct = sum(r["correct"] for r in cleaned_results) if cleaned_results else 0
        total_samples = len(cleaned_results) if cleaned_results else 1
        
        report = {
            "metric": self.metrics.metric_name,
            "preds_path": str(jsonl_path),
            "accuracy_by_sample": total_correct / total_samples,
            "avg_accuracy_by_category": (
                sum(sum(v) / len(v) for v in categories_accuracy.values()) / len(categories_accuracy)
                if categories_accuracy else 0
            ),
            "categories_accuracy": {k: sum(v) / len(v) for k, v in categories_accuracy.items()},
            "ckpt": str(ckpt),
            "results": cleaned_results,
            "exp_dir": self.cfg.exp_dir,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "commit": run("git rev-parse HEAD"),
            "name": "DeSTA2.5-Audio",
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info(f"Report saved to {report_path}")
        return report
