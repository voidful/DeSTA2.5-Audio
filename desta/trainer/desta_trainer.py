import torch
from transformers import Trainer
from desta.utils.utils import run
from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config, GenerationOutput
import logging
from desta.trainer.data.simple_dataset import BaseAudioTextDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
import os
import json
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict, defaultdict
from pathlib import Path
from lulutils import get_unique_filepath
from desta.utils.metrics import ConsecutiveWordsAccuracyMetric
import time
import math
from typing import Dict, List, Optional, Union, Any

class DeSTA25Trainer(Trainer):
    """
    Custom Trainer for DeSTA2.5-Audio model.
    Inherits from HuggingFace Transformers Trainer.
    """
    def __init__(self, model: DeSTA25AudioModel, cfg: DictConfig, **kwargs):
        """
        Initialize the trainer.

        Args:
            model (DeSTA25AudioModel): The model to train.
            cfg (DictConfig): The configuration object.
            **kwargs: Additional arguments passed to the parent Trainer class.
        """
        super().__init__(model=model, **kwargs)
        self.cfg = cfg
        self.metrics = ConsecutiveWordsAccuracyMetric()
        self.prediction_step_outputs = []

    def compute_loss(self, model: DeSTA25AudioModel, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: int = None):
        """
        Compute the loss for a batch of inputs.

        Args:
            model (DeSTA25AudioModel): The model.
            inputs (Dict[str, torch.Tensor]): The inputs to the model.
            return_outputs (bool, optional): Whether to return the outputs. Defaults to False.
            num_items_in_batch (int, optional): Number of items in the batch. Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: The loss or a tuple of (loss, outputs).
        """
        # HF Trainer passes labels in inputs if the collator includes them
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log perplexity
        perplexity = torch.exp(loss)
        self.log({"train/loss": loss.item(), "train/ppl": perplexity.item()})
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset: Optional[Any] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        Args:
            eval_dataset (Optional[Any], optional): The evaluation dataset. Defaults to None.
            ignore_keys (Optional[List[str]], optional): Keys to ignore. Defaults to None.
            metric_key_prefix (str, optional): Prefix for metric keys. Defaults to "eval".

        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        # Custom evaluation loop to match the original PL logic
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        
        all_losses = []
        all_ppls = []
        self.prediction_step_outputs = []
        
        start_time = time.time()
        
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                all_losses.append(loss.item())
                all_ppls.append(perplexity.item())
                
                # Prediction step
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
        metrics[f"{metric_key_prefix}_accuracy"] = report["accuracy_by_sample"]
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return metrics

    def _predict_step(self, batch: Dict[str, Any]):
        """
        Perform a prediction step.

        Args:
            batch (Dict[str, Any]): The batch of data.
        """
        # Logic from original predict_step
        generated_ids = self.model._generate_step(batch, 
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  temperature=self.cfg.model.generation_kwargs.temperature,
                                                  top_p=self.cfg.model.generation_kwargs.top_p,
                                                  max_new_tokens=self.cfg.model.generation_kwargs.max_new_tokens,
                                                  do_sample=self.cfg.model.generation_kwargs.do_sample,
                                                  ) # batched

        batch["context_input_ids"][batch["context_input_ids"] == -100] = self.tokenizer.eos_token_id
        batch["labels"][batch["labels"] == -100] = self.tokenizer.eos_token_id
        generated_ids[generated_ids == -100] = self.tokenizer.eos_token_id

        contexts = self.tokenizer.batch_decode(batch["context_input_ids"], skip_special_tokens=False)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Record predictions
        for context, label, pred, metadata in zip(contexts, labels, preds, batch["metadata"]):
            metadata.update({
                "context": context,
                "prediction": pred,
                "label": label,
            })
            self.prediction_step_outputs.append(metadata)

    def _write_report(self) -> Dict[str, Any]:
        """
        Write the evaluation report to a file.

        Returns:
            Dict[str, Any]: The report data.
        """
        dataset_name = "val"
        os.makedirs(f"{self.cfg.exp_dir}/results/{dataset_name}", exist_ok=True)
        output_path = f"{self.cfg.exp_dir}/results/{dataset_name}/val@ep={self.state.epoch}-{self.state.global_step}.jsonl"

        return self.write_to_file(
            results=self.prediction_step_outputs,
            filepath=output_path, 
            cfg=self.cfg,
            ckpt=f"ep={self.state.epoch}-{self.state.global_step}",
            write_report=True
        )

    def write_to_file(self, results: List[Dict[str, Any]], filepath: Union[str, Path], cfg: Optional[DictConfig] = None, ckpt: Optional[str] = None, write_report: bool = True) -> Dict[str, Any]:
        """
        Write results to a JSONL file and optionally generate a report.

        Args:
            results (List[Dict[str, Any]]): The results to write.
            filepath (Union[str, Path]): The path to the output file.
            cfg (Optional[DictConfig], optional): The configuration. Defaults to None.
            ckpt (Optional[str], optional): The checkpoint name. Defaults to None.
            write_report (bool, optional): Whether to write a report file. Defaults to True.

        Returns:
            Dict[str, Any]: The report data if write_report is True, else None.
        """
        filepath = Path(filepath)
        
        categories_accuracy = defaultdict(list)
        
        jsonl_path = Path(get_unique_filepath(filepath.parent / "preds" / filepath.name))
        os.makedirs(jsonl_path.parent, exist_ok=True)

        with open(jsonl_path, "w") as f:
            for i, result in enumerate(results):
                result["correct"] = self.metrics(result["prediction"], result["label"])
                result["index"] = i
                f.write(json.dumps(result) + "\n")
                categories_accuracy[result.get("category", "all")].append(result["correct"])

        if write_report:
            # Report
            report_path = jsonl_path.parent.parent / jsonl_path.name.replace(".jsonl", "-report.json")

            reported_results = []
            for i, result in enumerate(results):
                # remove context and audio_context for better readability (too long!)
                if "context" in result: del result["context"]
                if "audio_context" in result: del result["audio_context"]
                reported_results.append(result)

            report = {
                "metric": self.metrics.metric_name,
                "preds_path": str(jsonl_path),
                "accuracy_by_sample": sum([reported_results[i]["correct"] for i in range(len(reported_results))]) / len(reported_results) if reported_results else 0,
                "avg_accuracy_by_category": sum([sum(v) / len(v) for v in categories_accuracy.values()]) if categories_accuracy else 0,
                "categories_accuracy": dict([ (k, sum(v) / len(v)) for k, v in categories_accuracy.items()]),
                "ckpt": str(ckpt),
                "results": reported_results,
                "exp_dir": self.cfg.exp_dir,
                "config": OmegaConf.to_container(cfg, resolve=True) if cfg else {},
                "commit": run("git rev-parse HEAD"),
                "name": "DeSTA2.5-Audio",
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logging.info(f"Report saved to\n{report_path}\n")
            print(f"Report saved to\n{report_path}\n")

            return report
