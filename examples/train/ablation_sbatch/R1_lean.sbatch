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
# R1: G1 + H1 + P1 組合 (J1 的真正核心)
# 
# 發現: J1 的 task_router 未實際使用!
# J1 效果好是因為它包含 G1 + H1，不是 Task Routing
#
# 組合:
# ✅ G1 (Modality-DPO): MMAU 60.90% - 偏好優化
# ✅ H1 (Variational Grouping): Speech 71.47% - 概率分組
# ✅ P1 (ASR Dropout 0.3): MMAU 61.60% - ASR 魯棒性
#
# 預期效果:
# - SAKURA: ~70% (G1+H1)
# - MMAU: ~62% (P1+G1)
# - Speech: ~71% (H1)
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
name="R1_G1H1P1"
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
    ++model.modality_dpo_enabled=true \
    ++model.modality_dpo_beta=0.15 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=0.005 \
    ++model.task_aware_routing_enabled=false \
    ++model.temporal_attention_enabled=false \
    ++model.focal_loss_enabled=false \
    ++model.asr_dropout_prob=0.3 \
    ++model.multi_scale_local_enabled=false \
    ++model.layer_weighted_sum_enabled=false \
    ++model.asr_noise_augment_enabled=false

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_R1_G1H1P1"


