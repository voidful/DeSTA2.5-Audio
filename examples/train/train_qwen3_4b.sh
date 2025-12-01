#!/bin/bash
# DeSTA2.5-Audio Qwen3-4B 本地訓練腳本
# 用法: bash examples/train/train_qwen3_4b.sh

set -e

# ===== 配置區 =====
# 根據你的環境修改以下路徑
ROOT_DIR="/work/voidful2nlp/DeSTA2.5-Audio"
DATA_ROOT="/work/voidful2nlp/desta"
OUTPUT_BASE="/work/voidful2nlp/desta/outputs"

# 模型和數據集配置
config=desta25_qwen3-4B_Qformer6L
dataset_config=DestaAQA-5M_local

# GPU 設定 (可用逗號分隔多個 GPU，如 "0,1,2,3")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 實驗名稱
project="desta25_qwen3_4b"
name="qwen3-4b-instruct"
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="${OUTPUT_BASE}/${project}/${exp_name}"

# ===== 環境設定 =====
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# 如果需要存取 gated model，取消下行註解
# export HF_TOKEN="your_token_here"

# ===== 建立目錄 =====
mkdir -p "${exp_dir}"

echo "=========================================="
echo "DeSTA2.5-Audio Qwen3-4B Training"
echo "=========================================="
echo "Config: ${config}"
echo "Dataset: ${dataset_config}"
echo "Output: ${exp_dir}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# 記錄 git 和環境資訊
git diff > "${exp_dir}/git-diff.txt" 2>/dev/null || true
nvidia-smi > "${exp_dir}/nvidia-smi.txt" 2>/dev/null || true
pip list > "${exp_dir}/pip-list.txt" 2>/dev/null || true

# ===== 計算 GPU 數量 =====
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# 生成 devices 列表 (從 0 開始的索引)
DEVICES="["
for ((i=0; i<NUM_GPUS; i++)); do
    DEVICES+="$i"
    if [ $i -lt $((NUM_GPUS-1)) ]; then
        DEVICES+=","
    fi
done
DEVICES+="]"

echo "Using ${NUM_GPUS} GPUs with devices=${DEVICES}"

# ===== 啟動訓練 =====
cd "${ROOT_DIR}"

python examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    trainer.devices=${DEVICES} \
    +dataset=${dataset_config} \
    ++exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    ++dataset.train_ds.data_root=${DATA_ROOT} \
    ++dataset.validation_ds.data_root=${DATA_ROOT} \
    ++resume_from_checkpoint=null \
    ++init_from_pretrained_weights=null \
    "$@"  # 傳遞額外參數

echo "Training finished at $(date)"
echo "Output saved to: ${exp_dir}"

