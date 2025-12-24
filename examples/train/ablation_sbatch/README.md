# ORCA Ablation Study v2

## 實驗總覽

基於當前 ORCA 架構（新增 Global-Local 正交性損失、使用所有 32 層、4x downsample）的系統性 ablation study。

---

## 實驗列表

| Exp | 名稱 | Global (32L) | Local (4x) | Deep Inj | Diversity | GL-Ortho | Alignment | 腳本 |
|-----|------|-------------|-----------|----------|-----------|----------|-----------|------|
| 0 | Baseline | ❌ (Q-Former) | ❌ | ❌ | ❌ | ❌ | ❌ | `exp0_baseline.sbatch` |
| 1 | Global 32L | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | `exp1_global32.sbatch` |
| 2 | + Local | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | `exp2_add_local.sbatch` |
| 3 | + Deep Inj | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `exp3_add_deep_inj.sbatch` |
| 4 | + Diversity | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | `exp4_add_diversity.sbatch` |
| 5 | + GL-Ortho | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | `exp5_add_gl_ortho.sbatch` |
| 6 | + Alignment | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `exp6_add_alignment.sbatch` |
| 7 | Full ORCA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `exp7_full_orca.sbatch` |

---

## 快速開始

### 提交單個實驗

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio/examples/train/ablation_sbatch
sbatch exp0_baseline.sbatch
```

### 提交所有實驗

```bash
for i in {0..6}; do
    sbatch exp${i}_*.sbatch
done
```

### 檢查實驗狀態

```bash
squeue -u $USER
```

---

## 實驗配置

### Exp 0: DeSTA2.5 Baseline

- **目的**: 建立基準線
- **配置**: Q-Former only, no ORCA
- **預期**: Hmean ~48.38

### Exp 1: ORCA Base (Global 32 Layers)

- **目的**: 評估 32 層 global branch
- **配置**: Global only, 32 layers, no local/deep injection/losses
- **預期**: Hmean ~49-50

### Exp 2: + Local Branch

- **目的**: 評估 local branch 貢獻
- **配置**: + Local (4x downsample), no deep injection/losses
- **預期**: Hmean ~50-51, Multi-speaker 改善

### Exp 3: + Deep Injection

- **目的**: 評估 deep injection 效果
- **配置**: + Gated cross-attention, no losses
- **預期**: Hmean ~51-52

### Exp 4: + Diversity Loss

- **目的**: 評估 L_ortho_diversity
- **配置**: + Diversity loss (0.05)
- **預期**: Hmean ~51.5-52.5

### Exp 5: + Alignment Loss

- **目的**: 評估 L_align_layerwise
- **配置**: + Alignment loss (0.05)
- **預期**: Hmean ~52-53, Language 改善

### Exp 6: Full ORCA

- **目的**: 完整配置參考
- **配置**: 使用預設 config
- **預期**: Hmean ~52-53

---

## 輸出目錄結構

```
/work/voidful2nlp/desta/outputs/ablation_v2/
├── YYMMDD-HHMM_exp0_baseline/
│   ├── checkpoint-latest/
│   ├── checkpoint-1000/
│   └── ...
├── YYMMDD-HHMM_exp1_global32/
├── YYMMDD-HHMM_exp2_add_local/
├── YYMMDD-HHMM_exp3_add_deep_inj/
├── YYMMDD-HHMM_exp4_add_diversity/
├── YYMMDD-HHMM_exp5_add_alignment/
└── YYMMDD-HHMM_exp6_full_orca/
```

---

## 日誌位置

```
/work/voidful2nlp/DeSTA2.5-Audio/examples/train/slurm-report/
├── ablation_exp0_baseline_<job_id>.out
├── ablation_exp1_global32_<job_id>.out
├── ...
```

---

## 評估

訓練完成後，在 Sakura benchmark 上評估所有實驗：

```bash
# 需要用戶提供評估腳本
for exp_dir in /work/voidful2nlp/desta/outputs/ablation_v2/*; do
    python evaluate_sakura.py --checkpoint ${exp_dir}/checkpoint-latest
done
```

---

## 預期結果

### 組件貢獻排序（預測）

1. **Local Branch**: +1~2 Hmean
2. **Deep Injection**: +1~2 Hmean
3. **Alignment Loss**: +0.5~1 Hmean
4. **32 Layers**: +0.5~1 Hmean
5. **Diversity Loss**: +0.3~0.5 Hmean

### 關鍵發現（預期）

- Local branch 對 multi-speaker 最重要
- Deep injection 提升整體性能
- Alignment loss 改善 language 識別
- 32 層比 4 層略好但成本高

---

## 注意事項

1. **Resume 功能**: 所有腳本支持自動 resume
2. **實驗命名**: 使用時間戳避免衝突
3. **資源需求**: 每個實驗需要 4 GPUs, 200GB RAM
4. **訓練時間**: 約 2 天/實驗 (5 epochs)

---

## 聯絡

如有問題請參考 `implementation_plan.md` 或聯絡實驗負責人。
