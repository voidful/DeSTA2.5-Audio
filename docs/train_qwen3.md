# DeSTA2.5-Audio with Qwen3-4B-Instruct 訓練指南

本文件說明如何使用 Qwen3-4B-Instruct-2507 作為 LLM backbone 來訓練 DeSTA2.5-Audio 模型。

## 📁 檔案結構

```
examples/train/
├── config/
│   ├── desta25_qwen3-4B_groupwise_ortho.yaml         # Qwen3-4B groupwise_ortho 配置
│   └── dataset/
│       └── DestaAQA-5M_local.yaml          # 本地數據集配置
├── run_desta_qwen3_4b.sbatch               # SLURM 批次腳本
└── train_desta.py                          # 訓練腳本
```

## 🔧 配置說明

### 模型配置 (`desta25_qwen3-4B_groupwise_ortho.yaml`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `model.llm.model_id` | `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 4B 指令微調模型 |
| `model.encoder.model_id` | `openai/whisper-large-v3` | Whisper Large V3 音訊編碼器 |
| `model.connector.mode` | `groupwise_ortho` | Groupwise-orthogonal Q-Former |
| `model.connector.num_hidden_layers` | 6 | Q-Former 層數 |
| `model.connector.prompt_size` | 64 | 音訊 prompt 長度 |
| `model.groupwise_ortho.num_groups` | 8 | group 數 |
| `model.groupwise_ortho.queries_per_group` | 8 | 每個 group 的 query 數 |
| `model.placeholder_token` | `<\|video_pad\|>` | Qwen3 的 placeholder token |
| `model.audio_locator` | `<\|AUDIO\|>` | 音訊位置標記 |

### 數據集配置 (`DestaAQA-5M_local.yaml`)

```yaml
train_ds:
  data_root: "/work/voidful2nlp/desta"
  manifest_filepaths:
    - "/work/voidful2nlp/desta/qwen3_desta_v4.jsonl"
  batch_size: 12
  max_seq_length: 300
  num_workers: 4

validation_ds:
  data_root: "/work/voidful2nlp/desta"
  manifest_filepaths:
    - "/work/voidful2nlp/desta/val_v4.jsonl"
  batch_size: 4
```

## 📊 數據格式

訓練數據為 JSONL 格式，每行一個 JSON 物件：

```json
{
  "id": "WavCaps_AudioSetSL/Y-1YwpJxxfNU.flac",
  "dataset": "WavCaps_AudioSetSL",
  "seed_description": "[00:00-00:10] (Background noise and ticking...)",
  "prompt": "In a sentence, explain what happened first in this audio sequence.",
  "response": "The background noise and ticking began while the music was playing.",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<|AUDIO|>\n{prompt}"},
    {"role": "assistant", "content": "{response}"}
  ]
}
```

### 關鍵欄位說明

| 欄位 | 說明 |
|------|------|
| `id` | 音訊檔案相對路徑（相對於 `data_root`）|
| `dataset` | 數據集來源名稱 |
| `seed_description` | 音訊內容描述（用於生成訓練數據）|
| `prompt` | 使用者提問 |
| `response` | 模型回答 |
| `messages` | 完整的對話格式（用於 chat template）|

## 🚀 執行訓練

### 方法一：使用 SLURM（推薦用於 HPC）

```bash
# 確保 slurm-report 目錄存在
mkdir -p slurm-report

# 提交任務
sbatch examples/train/run_desta_qwen3_4b.sbatch
```

### 方法二：直接執行（單機多卡）

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio

# 設定環境變數
export HF_HOME=/work/voidful2nlp/.cache/huggingface
export PYTHONPATH="/work/voidful2nlp/DeSTA2.5-Audio:$PYTHONPATH"

# 執行訓練
python examples/train/train_desta.py \
    --config-path=config \
    --config-name=desta25_qwen3-4B_groupwise_ortho \
    trainer.devices=[0,1,2,3,4,5,6,7] \
    +dataset=DestaAQA-5M_local \
    +exp_dir=/work/voidful2nlp/desta/outputs/qwen3-4b
```

### 方法三：單卡測試

```bash
python examples/train/train_desta.py \
    --config-path=config \
    --config-name=desta25_qwen3-4B_groupwise_ortho \
    trainer.devices=[0] \
    +dataset=DestaAQA-5M_local \
    +exp_dir=./test_output \
    +dataset.train_ds.batch_size=2
```

## ⚙️ SLURM 配置說明

```bash
#SBATCH --job-name=desta_qwen3_4b      # 任務名稱
#SBATCH --partition=normal              # 分區（依 cluster 調整）
#SBATCH --account=MST111038             # 帳號（依 cluster 調整）
#SBATCH --nodes=1                       # 節點數
#SBATCH --ntasks-per-node=8             # 每節點任務數
#SBATCH --gpus-per-node=8               # 每節點 GPU 數
#SBATCH --cpus-per-task=12              # 每任務 CPU 數
#SBATCH --mem=200G                      # 記憶體
#SBATCH --time=48:00:00                 # 最大執行時間
```

## 📈 訓練參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `max_epochs` | 5 | 訓練輪數 |
| `learning_rate` | 1e-4 | 學習率 |
| `warmup_steps` | 5000 | 預熱步數 |
| `batch_size` | 12 | 每 GPU batch size |
| `precision` | bf16-mixed | 混合精度訓練 |
| `gradient_clip_val` | 1.0 | 梯度裁剪 |

## 🔍 監控訓練

### 查看 SLURM 任務狀態

```bash
# 查看任務狀態
squeue -u $USER

# 查看即時輸出
tail -f slurm-report/desta_qwen3_4b_<job_id>.out
```

### Weights & Biases 監控

訓練會自動記錄到 W&B，可在 https://wandb.ai 查看：
- 訓練/驗證 loss
- 學習率變化
- GPU 使用率

## 🐛 常見問題

### 1. CUDA Out of Memory

```bash
# 減少 batch size
+dataset.train_ds.batch_size=8

# 使用 gradient accumulation
trainer.accumulate_grad_batches=2
```

### 2. 找不到音訊檔案

確認 `data_root` 路徑正確，且音訊檔案路徑格式為：
```
{data_root}/{id}
例如：/work/voidful2nlp/desta/WavCaps_AudioSetSL/Y-1YwpJxxfNU.flac
```

### 3. Placeholder Token 錯誤

Qwen3 模型使用 `<|video_pad|>` 作為 placeholder token，不要使用 Llama 的 `<|reserved_special_token_87|>`。

## 📚 參考資源

- [DeSTA2.5-Audio Paper](https://arxiv.org/abs/2507.02768)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3)
