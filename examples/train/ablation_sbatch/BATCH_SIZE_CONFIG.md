# ORCA Ablation Study - Batch Size Configuration

## 問題

使用所有 32 層 Whisper encoder 時，記憶體需求顯著增加：

```
4 層 Q-Former:  4 × Q-Former forward = baseline memory
32 層 Q-Former: 32 × Q-Former forward = 8x memory
```

原始配置 `batch_size=32` 導致 OOM (Out of Memory)。

## 解決方案

### 新增配置: `DestaAQA-5M_0.6b_ablation32.yaml`

**訓練**:
- `batch_size`: 32 → 8 (減少 4 倍)
- 使用 `gradient_accumulation_steps=4` 保持有效 batch size = 32

**驗證**:
- `batch_size`: 4 → 2 (減少 2 倍)

### 記憶體計算

```
32 層實驗:
- Per-device batch size: 8
- Gradient accumulation: 4
- Effective batch size: 8 × 4 = 32 ✓ (與原始相同)
- 記憶體使用: ~減少 75%
```

### 使用方式

**Exp 1-6 (32 層實驗)**:
```yaml
dataset_config: DestaAQA-5M_0.6b_ablation32
```

**Exp 0 (Baseline, 4 層)**:
```yaml
dataset_config: DestaAQA-5M_0.6b_orca  # 保持原始 batch_size=32
```

## 訓練時間影響

```
Gradient accumulation 會略微增加訓練時間:
- 原始: 1 step = 32 samples
- 新配置: 1 step = 4 micro-steps × 8 samples = 32 samples
- 時間增加: ~5-10% (gradient accumulation overhead)
```

## 環境變數

添加到所有 ablation 腳本:
```bash
export TOKENIZERS_PARALLELISM=false  # 避免 fork warning
```
