#!/bin/bash
#SBATCH --job-name=R2_full
#SBATCH --partition=normal
#SBATCH --account=MST114563
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/R2_full_%j.out
#SBATCH --error=./slurm-report/R2_full_%j.out

# ==========================================
# R2: R1v2 Robust + Full Dataset
# 
# 基於 R1v2 的穩健配置，使用完整數據集訓練
# 數據集: qwen3_4b_desta_fixed.jsonl (全部資料)
#
# 技巧組合:
# ✅ G1 (Modality-DPO): β=0.1
# ✅ H1 (Variational Grouping): KL=0.01
# ✅ P1 (ASR Dropout): prob=0.2
# ✅ K1 (Temporal Attention)
#
# 預期效果 (完整數據):
# - SAKURA: ~72%
# - MMAU: ~62%
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
dataset_config=DestaAQA-5M_4b_orca_full  # 使用完整數據集

project=orca_r1_ablation
name="R2_full"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="/work/voidful2nlp/desta/outputs/${project}/${exp_name}"

mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# ===== 啟動訓練 =====
NUM_GPUS=2
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
    ++model.modality_dpo_enabled=true \
    ++model.modality_dpo_beta=0.1 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=0.01 \
    ++model.asr_dropout_prob=0.2

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_R2_full"
