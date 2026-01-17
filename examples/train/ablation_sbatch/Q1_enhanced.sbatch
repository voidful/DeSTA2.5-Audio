#!/bin/bash
#SBATCH --job-name=Q1_enhanced
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/Q1_enhanced_%j.out
#SBATCH --error=./slurm-report/Q1_enhanced_%j.out

# ==========================================
# Q1: Enhanced Configuration (I1 + 2 improvements)
# 
# 基於 I1 (8項有效技巧) 新增:
# ✅ Layer-wise Weighted Sum: 加權融合 encoder 各層輸出
# ✅ Noisy ASR Augmentation: 模擬真實 ASR 錯誤
#
# Pillar 1 強化: Layer-wise Weighted Sum
# Pillar 3 強化: Noisy ASR Augmentation
#
# 預期效果:
# - SAKURA: ~73% (層次特徵提升)
# - MMAU: ~64% (ASR 魯棒性提升)
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

project=orca_r1_ablation
name="Q1_enhanced"
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
    ++model.orca_r1.num_groups=8 \
    ++model.orca_r1.queries_per_group=8 \
    ++model.orca_r1.inter_group_weight=0.1 \
    ++model.orca_r1.intra_group_weight=0.03 \
    ++model.orca_r1.iv_weight=0 \
    ++model.orca.enabled=false \
    ++model.orca.align_weight_local=0.1 \
    ++model.orca.orthogonal_projection=true \
    ++model.modality_dpo_enabled=true \
    ++model.modality_dpo_beta=0.15 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=0.005 \
    ++model.task_aware_routing_enabled=true \
    ++model.temporal_attention_enabled=true \
    ++model.focal_loss_enabled=true \
    ++model.focal_loss_gamma=2.0 \
    ++model.asr_dropout_prob=0.3 \
    ++model.multi_scale_local_enabled=true \
    ++model.layer_weighted_sum_enabled=true \
    ++model.layer_weighted_learnable=true \
    ++model.asr_noise_augment_enabled=true \
    ++model.asr_noise_prob=0.15

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_Q1_enhanced"
