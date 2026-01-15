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
# R1v2: Robust G1 + H1 + P1 配置
# 
# 衝突修復:
# 1. P1 Dropout: 0.3 → 0.2 (減少雙重隨機)
# 2. KL Weight: 0.005 → 0.01 (增強 H1 效果)
# 3. DPO Beta: 0.15 → 0.1 (減少梯度拉扯)
#
# 安全新增:
# ✅ K1 (Temporal Attention): 與 H1/G1 無衝突
#
# 預期效果:
# - SAKURA: ~71% (更穩定)
# - MMAU: ~61-62% (減少衝突)
# - Speech: ~71% (保持 H1 優勢)
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
name="R1v2_robust"
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
    ++model.modality_dpo_beta=0.1 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=0.01 \
    ++model.task_aware_routing_enabled=false \
    ++model.temporal_attention_enabled=true \
    ++model.focal_loss_enabled=false \
    ++model.asr_dropout_prob=0.2 \
    ++model.multi_scale_local_enabled=false \
    ++model.layer_weighted_sum_enabled=false \
    ++model.asr_noise_augment_enabled=false

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_R1v2_robust"



