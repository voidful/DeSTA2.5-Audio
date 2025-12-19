#!/bin/bash

# DeSTA 2.5 Audio Ablation Study Script
# Use this script to launch multiple experiments comparing different components.

# Set your variables
ROOT_DIR=$(pwd)
CONFIG_NAME="desta25_qwen3-0.6b_ORCAHybrid"  # Choose your base config
DATASET_CONFIG="DestaAQA-5M_0.6b_orca"
NUM_GPUS=4
PROJECT="desta25_ablation"

# Define the Ablation Matrix
# Format: "experiment_name;overrides"
EXPERIMENTS=(
    "full;model.orca.enabled=true,model.orca.local_enabled=true,model.orca.global_cross_attn=true,model.orca.deep_injection_enabled=true"
    "no_local;model.orca.local_enabled=false"
    "no_global_xattn;model.orca.global_cross_attn=false"
    "no_deep_injection;model.orca.deep_injection_enabled=false"
    "baseline_desta2.0;connector_mode=qformer_1,model.orca.enabled=false"
    "no_aux_losses;model.orca.ortho_weight_global=0,model.orca.ortho_diversity_weight=0,model.orca.ortho_weight_qformer_local=0,model.orca.align_weight_local=0"
)

# Create outputs directory
mkdir -p ./outputs/ablation
mkdir -p ./slurm-report

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=";" read -r EXP_NAME OVERRIDES <<< "$EXP"
    
    echo "===================================================="
    echo "Launching Experiment: $EXP_NAME"
    echo "Overrides: $OVERRIDES"
    echo "===================================================="
    
    EXP_FULL_NAME=$(date +%y%m%d-%H%M)_${EXP_NAME}
    EXP_DIR="./outputs/ablation/${EXP_FULL_NAME}"
    
    # Replace commas with spaces for hydra
    HYDRA_OVERRIDES=$(echo $OVERRIDES | tr ',' ' ')
    
    # Run the training command
    # NOTE: In a real cluster, you might want to use sbatch instead of running directly.
    # To use sbatch, wrap this in a sbatch command or use the template below.
    
    torchrun --nproc_per_node=${NUM_GPUS} \
        ${ROOT_DIR}/examples/train/train_desta.py \
        --config-path=config \
        --config-name=${CONFIG_NAME} \
        +dataset=${DATASET_CONFIG} \
        ++exp_dir=${EXP_DIR} \
        project=${PROJECT} \
        name=${EXP_NAME} \
        ${HYDRA_OVERRIDES}
        
    echo "Experiment $EXP_NAME finished."
done
