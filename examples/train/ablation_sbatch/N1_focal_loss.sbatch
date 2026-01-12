#!/bin/bash
#SBATCH --job-name=N1_focal
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/N1_focal_%j.out
#SBATCH --error=./slurm-report/N1_focal_%j.out

# N1: Focal Loss (based on I1)

module load anaconda 2>/dev/null || true
conda activate desta25-qwen3
cd /work/voidful2nlp/DeSTA2.5-Audio
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface

ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-4B_ORCA
dataset_config=DestaAQA-5M_4b_orca
project=struct_orca_ablation
name="N1_focal_loss"
exp_dir="/work/voidful2nlp/desta/outputs/${project}/$(date +%y%m%d-%H%M)_${name}"
mkdir -p ${exp_dir} ./slurm-report

torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config --config-name=${config} \
    +dataset=${dataset_config} ++exp_dir=${exp_dir} project=${project} name=${name} \
    ++model.struct_orca.num_groups=8 ++model.struct_orca.queries_per_group=8 \
    ++model.struct_orca.inter_group_weight=0.1 ++model.struct_orca.intra_group_weight=0.03 \
    ++model.orca.enabled=false ++model.orca.orthogonal_projection=true \
    ++model.modality_dpo_enabled=true ++model.modality_dpo_beta=0.15 \
    ++model.variational_grouping_enabled=true ++model.variational_kl_weight=0.005 \
    ++model.focal_loss_enabled=true ++model.focal_loss_gamma=2.0

./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_N1_focal"
