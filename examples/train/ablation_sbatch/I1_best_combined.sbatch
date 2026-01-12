#!/bin/bash
#SBATCH --job-name=I1_best
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/I1_best_%j.out
#SBATCH --error=./slurm-report/I1_best_%j.out

# ==========================================
# I1: Premium Configuration (Updated based on analysis)
# = G1 (Modality-DPO) - best overall
# + H1 (Variational) - best Speech (lower weight to avoid Music drop)
# + F1 (Local Branch) - best multi-speaker handling
# + D1 (Orthogonal Projection) - stable base
#
# 預期效果:
# - Speech: ~71% (H1 優勢)
# - Music: ~54% (G1 優勢)
# - Sound: ~59% (F1 優勢)
# - Total: ~62%
# ==========================================

# ===== 環境設定 =====
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

cd /work/voidful2nlp/DeSTA2.5-Audio
pip install "numpy<2.0" -q 2>/dev/null || true

export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# ===== 實驗設定 =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-4B_ORCA
dataset_config=DestaAQA-5M_4b_orca

project=struct_orca_ablation
name="I1_premium"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="/work/voidful2nlp/desta/outputs/${project}/${exp_name}"

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=4
MASTER_PORT=$((29500 + RANDOM % 1000))

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    ++model.struct_orca.num_groups=8 \
    ++model.struct_orca.queries_per_group=8 \
    ++model.struct_orca.inter_group_weight=0.1 \
    ++model.struct_orca.intra_group_weight=0.03 \
    ++model.struct_orca.iv_weight=0 \
    ++model.orca.enabled=false \
    ++model.orca.align_weight_local=0.1 \
    ++model.orca.orthogonal_projection=true \
    ++model.modality_dpo_enabled=true \
    ++model.modality_dpo_beta=0.15 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=0.003 \
    ++model.orca.local_branch_true=true \
    ++model.orca.local_ortho_weight=0.05

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_I1_premium"
