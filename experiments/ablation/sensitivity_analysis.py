#!/usr/bin/env python
"""
Sensitivity Analysis for ORCA-DeSTA

Analyzes sensitivity to key hyperparameters:
- Number of groups M (with fixed total tokens)
- Loss weights (inter-group, intra-group, IV)
- ACD alpha values

Usage:
    python experiments/ablation/sensitivity_analysis.py \
        --base_checkpoint /path/to/checkpoint \
        --output_dir ./sensitivity_results \
        --analysis groups  # or 'loss_weights' or 'acd_alpha'
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def analyze_group_sensitivity(
    checkpoints: Dict[str, str],
    data_path: str,
    output_dir: str,
    num_samples: int = 500
) -> Dict[str, Any]:
    """
    Analyze sensitivity to number of groups M.
    
    Args:
        checkpoints: Dict mapping "MxN" config name to checkpoint path
        data_path: Path to evaluation data
        output_dir: Directory to save results
        num_samples: Samples per checkpoint
        
    Returns:
        Dict with accuracy and metrics for each configuration
    """
    from desta import DeSTA25AudioModel
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diagnosis'))
    from feature_analysis import analyze_feature_collapse, extract_audio_representations
    from mutual_information import estimate_mutual_information
    
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for config_name, checkpoint in checkpoints.items():
        print(f"\n=== Analyzing {config_name} ===")
        
        # Load model
        model = DeSTA25AudioModel.from_pretrained(checkpoint)
        model.to(device)
        model.eval()
        
        # Extract config
        num_groups = model.config.struct_orca_num_groups
        queries_per_group = model.config.struct_orca_queries_per_group
        
        # Run evaluation
        # This would run SAKURA eval - simplified here
        # In practice, call sakura_eval functions
        
        # Extract representations for analysis
        # Placeholder - would use actual dataloader
        
        results[config_name] = {
            "num_groups": num_groups,
            "queries_per_group": queries_per_group,
            "total_tokens": num_groups * queries_per_group,
            "checkpoint": checkpoint,
            # Add metrics after eval
        }
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "group_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_loss_weight_sensitivity(
    checkpoints: Dict[str, str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Analyze sensitivity to loss weights.
    
    Args:
        checkpoints: Dict mapping weight config to checkpoint path
        output_dir: Directory to save results
        
    Returns:
        Dict with metrics for each weight configuration
    """
    results = {}
    
    for config_name, checkpoint in checkpoints.items():
        print(f"\n=== Analyzing {config_name} ===")
        
        # Parse weights from config name (e.g., "inter0.1_intra0.01_iv0.05")
        # Load checkpoint and extract metrics
        
        results[config_name] = {
            "checkpoint": checkpoint,
            # Add metrics after eval
        }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "loss_weight_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_acd_alpha_sensitivity(
    checkpoint: str,
    alphas: List[float],
    data_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Analyze sensitivity to ACD alpha parameter.
    
    Args:
        checkpoint: Path to trained model
        alphas: List of alpha values to test
        data_path: Path to evaluation data
        output_dir: Directory to save results
        
    Returns:
        Dict with accuracy for each alpha
    """
    from desta import DeSTA25AudioModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model once
    print(f"Loading model from {checkpoint}")
    model = DeSTA25AudioModel.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    
    results = {}
    
    for alpha in alphas:
        print(f"\n=== Testing ACD alpha={alpha} ===")
        
        # Run evaluation with this alpha
        # In practice, modify sakura_eval to accept model instance and alpha
        
        results[f"alpha_{alpha}"] = {
            "acd_alpha": alpha,
            # Add accuracy metrics after eval
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(os.path.join(output_dir, "acd_alpha_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    if len(results) > 1:
        plot_acd_sensitivity(results, output_dir)
    
    del model
    torch.cuda.empty_cache()
    
    return results


def plot_acd_sensitivity(results: Dict[str, Any], output_dir: str):
    """Plot ACD alpha sensitivity curve."""
    alphas = []
    accuracies = []
    
    for key, val in sorted(results.items(), key=lambda x: x[1]["acd_alpha"]):
        alphas.append(val["acd_alpha"])
        accuracies.append(val.get("accuracy", 0))
    
    if not accuracies or all(a == 0 for a in accuracies):
        print("No accuracy data to plot")
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, 'bo-', markersize=8)
    plt.xlabel("ACD Alpha")
    plt.ylabel("Accuracy (%)")
    plt.title("Sensitivity to ACD Alpha Parameter")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "acd_alpha_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ORCA Sensitivity Analysis")
    parser.add_argument("--analysis", type=str, required=True,
                        choices=["groups", "loss_weights", "acd_alpha"],
                        help="Type of sensitivity analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Base checkpoint for ACD analysis")
    parser.add_argument("--checkpoints_json", type=str, default=None,
                        help="JSON file with checkpoint paths for groups/loss_weights analysis")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default="./sensitivity_results",
                        help="Output directory")
    parser.add_argument("--alphas", type=float, nargs="+", 
                        default=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
                        help="Alpha values for ACD analysis")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.analysis == "groups":
        if args.checkpoints_json is None:
            print("Error: --checkpoints_json required for groups analysis")
            return
        with open(args.checkpoints_json) as f:
            checkpoints = json.load(f)
        analyze_group_sensitivity(checkpoints, args.data_path, args.output_dir)
        
    elif args.analysis == "loss_weights":
        if args.checkpoints_json is None:
            print("Error: --checkpoints_json required for loss_weights analysis")
            return
        with open(args.checkpoints_json) as f:
            checkpoints = json.load(f)
        analyze_loss_weight_sensitivity(checkpoints, args.output_dir)
        
    elif args.analysis == "acd_alpha":
        if args.checkpoint is None:
            print("Error: --checkpoint required for acd_alpha analysis")
            return
        analyze_acd_alpha_sensitivity(
            args.checkpoint, args.alphas, args.data_path, args.output_dir
        )


if __name__ == "__main__":
    main()
