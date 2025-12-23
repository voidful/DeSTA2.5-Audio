#!/bin/bash

#SBATCH --job-name=desta_qwen3_0.6b_orca
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-report/desta_qwen3_0.6b_orca_%j.out
#SBATCH --error=./slurm-report/desta_qwen3_0.6b_orca_%j.out

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
config=desta25_qwen3-0.6b_ORCAHybrid
dataset_config=DestaAQA-5M_0.6b_orca

project=desta25_qwen3_0.6b_orca
name="qwen3-0.6b-instruct-orca"
output_root="/work/voidful2nlp/desta/outputs/${project}"

# ===== 自動檢測並 Resume Checkpoint =====
exp_name=$(date +%y%m%d-%H%M)_${name}
exp_dir="${output_root}/${exp_name}"
resume_args=""

# 尋找最新的實驗目錄
latest_dir=$(ls -td ${output_root}/*_${name} 2>/dev/null | head -n 1)
if [ -d "$latest_dir" ]; then
    # 尋找該目錄中最新的 checkpoint (可能是 checkpoint-latest 或 checkpoint-XXXXX)
    latest_checkpoint=$(ls -td "$latest_dir"/checkpoint-* 2>/dev/null | head -n 1)
    if [ -d "$latest_checkpoint" ]; then
        exp_dir="$latest_dir"
        resume_args="++resume_from_checkpoint=$latest_checkpoint"
        echo "Found existing checkpoint, will resume from: $latest_checkpoint"
    else
        echo "No checkpoint found in $latest_dir, starting new training run"
    fi
else
    echo "No existing experiment directory found, starting new training run"
fi

# 建立輸出目錄
mkdir -p ${exp_dir}
mkdir -p ./slurm-report

# 記錄 git diff 和 GPU 資訊
git diff > ${exp_dir}/git-diff.txt 2>/dev/null || true
nvidia-smi > ${exp_dir}/nvidia-smi.txt 2>/dev/null || true

# 記錄配置
echo "========================================" > ${exp_dir}/run_info.txt
echo "Job ID: $SLURM_JOB_ID" >> ${exp_dir}/run_info.txt
echo "Start Time: $(date)" >> ${exp_dir}/run_info.txt
echo "Config: ${config}" >> ${exp_dir}/run_info.txt
echo "Dataset Config: ${dataset_config}" >> ${exp_dir}/run_info.txt
echo "Output Dir: ${exp_dir}" >> ${exp_dir}/run_info.txt
echo "ORCA Mode: enabled" >> ${exp_dir}/run_info.txt
if [ -n "$resume_args" ]; then
    echo "Resume: Yes (from ${latest_checkpoint})" >> ${exp_dir}/run_info.txt
else
    echo "Resume: No (fresh start)" >> ${exp_dir}/run_info.txt
fi
echo "========================================" >> ${exp_dir}/run_info.txt

# ===== 啟動訓練 (使用 torchrun 進行 DDP 分布式訓練) =====
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
    ++dataset.train_ds.data_root=/work/voidful2nlp/desta \
    ++dataset.validation_ds.data_root=/work/voidful2nlp/desta \
    ${resume_args} \
    ++init_from_pretrained_weights=null

echo "Training finished at $(date)" >> ${exp_dir}/run_info.txt
