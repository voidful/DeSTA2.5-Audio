#!/bin/bash

#SBATCH --job-name=desta_qwen3_4b_orca_resume
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-report/desta_qwen3_4b_orca_resume_%j.out
#SBATCH --error=./slurm-report/desta_qwen3_4b_orca_resume_%j.out

# ===== RESUME 設定 =====
# 要 resume 的 checkpoint 路徑 (修改這裡來指定不同的 checkpoint)
CHECKPOINT_PATH="/work/voidful2nlp/desta/outputs/desta25_qwen3_4b_ocar/251212-2151_qwen3-4b-instruct-ocar/checkpoint-210129"
# 原始實驗目錄 (logs 會繼續寫入這裡)
RESUME_EXP_DIR="/work/voidful2nlp/desta/outputs/desta25_qwen3_4b_ocar/251212-2151_qwen3-4b-instruct-ocar"

# ===== 環境設定 =====
# 載入 Anaconda/Miniconda module (依 cluster 調整)
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true

# 啟動 conda 環境
source activate desta25 2>/dev/null || conda activate desta25 2>/dev/null || true
conda activate desta25-qwen3

# 切換到工作目錄
cd /work/voidful2nlp/DeSTA2.5-Audio

# 確保 numpy < 2.0 (避免 torch.from_numpy 相容性問題)
pip install "numpy<2.0" -q 2>/dev/null || true

# 設定環境變數
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1

# 如果需要存取 gated model，取消下行註解並填入 token
# export HF_TOKEN=your_huggingface_token_here

# ===== 實驗設定 =====
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
config=desta25_qwen3-4b_ORCAHybrid
dataset_config=DestaAQA-5M_local

project=desta25_qwen3_4b_orca
name="qwen3-4b-instruct-orca"

# 使用原始實驗目錄
exp_dir="${RESUME_EXP_DIR}"

# 建立輸出目錄 (應該已存在)
mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# 記錄 resume 資訊
echo "" >> ${exp_dir}/run_info.txt
echo "========================================" >> ${exp_dir}/run_info.txt
echo "RESUMED Job ID: $SLURM_JOB_ID" >> ${exp_dir}/run_info.txt
echo "Resume Time: $(date)" >> ${exp_dir}/run_info.txt
echo "Checkpoint: ${CHECKPOINT_PATH}" >> ${exp_dir}/run_info.txt
echo "========================================" >> ${exp_dir}/run_info.txt

# ===== 啟動訓練 (使用 torchrun 進行 DDP 分布式訓練) =====
NUM_GPUS=4
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "Resuming training from checkpoint: ${CHECKPOINT_PATH}"
echo "Output directory: ${exp_dir}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    ++dataset.train_ds.data_root=/work/voidful2nlp/desta \
    ++dataset.validation_ds.data_root=/work/voidful2nlp/desta \
    ++resume_from_checkpoint=${CHECKPOINT_PATH} \
    ++init_from_pretrained_weights=null

echo "Training finished at $(date)" >> ${exp_dir}/run_info.txt
