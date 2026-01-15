#!/bin/bash
#SBATCH --job-name=R1_lean
#SBATCH --partition=normal1
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/R1_lean_%j.out
#SBATCH --error=./slurm-report/R1_lean_%j.out

# ==========================================
# R1: Lean & Mean Configuration (Resolving Negative Transfer)
# 
# 核心理念: Acoustic-First + Structured Disentanglement
# 去除冗餘的正則化，保留最強效且互不衝突的組件。
#
# 保留組件 (The Core 4):
# 1. ✅ P1 (ASR Dropout): MMAU 王者 (61.60%) - 核心 ASR 魯棒性
# 2. ✅ M1 (Multi-Scale Local): Sound 王者 (61.56%) - 核心時序特徵
# 3. ✅ Q1 (Layer-wise Weighted Sum): 提升 SAKURA - 核心層次特徵
# 4. ✅ J1 (Task-Aware Routing): 提升 SAKURA (70.75%) - 核心語義路由
#
# 移除組件 (Removing Conflicts):
# ❌ D1 (Orthogonal Projection): 可能破壞 M1 特徵結構，導致 Sound 下降
# ❌ H1 (Variational Grouping): 增加優化難度
# ❌ N1 (Focal Loss): 可能改變 Loss Landscape 導致不穩定
# ❌ G1 (Modality-DPO): 避免與 P1 過度疊加
# ❌ C2 (Align Loss): 去除輔助 Loss 干擾
#
# 預期效果:
# - SAKURA: ~71-72% (J1 + Q1)
# - MMAU: ~62-63% (P1 + M1 無干擾)
# - Sound: 回升至 ~60% (移除 D1 干擾)
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
name="R1_lean"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="/work/voidful2nlp/desta/outputs/${project}/${exp_name}"

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=3
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
    ++model.orca.align_weight_local=0.0 \
    ++model.orca.orthogonal_projection=false \
    ++model.modality_dpo_enabled=false \
    ++model.variational_grouping_enabled=false \
    ++model.task_aware_routing_enabled=true \
    ++model.temporal_attention_enabled=false \
    ++model.focal_loss_enabled=false \
    ++model.asr_dropout_prob=0.3 \
    ++model.multi_scale_local_enabled=true \
    ++model.layer_weighted_sum_enabled=true \
    ++model.layer_weighted_learnable=true \
    ++model.asr_noise_augment_enabled=false

# 註: Q1 的 Noisy ASR 也暫時移除，先只用 P1 確認 ASR Dropout 效果，避免過度 augment
# 註: K1 (Temporal Attention) 也暫時移除，專注於 M1 的時序特徵

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_R1_lean"
