#!/bin/bash
#SBATCH --job-name=S1_3
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --exclude=hgpn21
#SBATCH --output=./slurm-report/S1_3_%j.out
#SBATCH --error=./slurm-report/S1_3_%j.out

# ==========================================
# S1_3: + μ Invariance (SpecAugment)
# 
# 加入 Contrastive μ Invariance Loss
# 預期：μ 對聲學擾動不變，編碼純語義
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

project=orca_s1_ablation
name="S1_3_mu_invariance"
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
    ++model.orca_r1.intra_group_weight=0.05 \
    ++model.variational_grouping_enabled=true \
    ++model.variational_kl_weight=1.0 \
    ++model.s1_kl_annealing_enabled=true \
    ++model.s1_kl_annealing_warmup_steps=2000 \
    ++model.s1_free_bits=0.1 \
    ++model.modality_dpo_enabled=false \
    ++model.s1_mu_invariance_enabled=true \
    ++model.s1_mu_invariance_weight=0.1

# ===== 上傳到 HuggingFace =====
echo 'Training complete. Uploading to HuggingFace...'
./examples/train/ablation_sbatch/upload_to_hf.sh ${exp_dir} "desta25_4b_S1_3_mu_invariance"
