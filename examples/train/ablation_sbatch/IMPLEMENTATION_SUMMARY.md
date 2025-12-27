# Minimal Ablation Study - Implementation Summary

## 完成狀態: ✅ 已完成

實施日期: 2025-12-27

---

## 改動總結

### 1. 新增實驗腳本 (3 個)

| 文件 | 目的 | 配置重點 |
|------|------|---------|
| `exp1_orca_architecture.sbatch` | 測試 ORCA 架構 | 雙分支 + 深度注入,無 loss |
| `exp2_add_orthogonality.sbatch` | 測試正交性損失 | 架構 + 3 個正交性 loss |
| `exp3_full_orca.sbatch` | 驗證完整系統 | 使用默認配置 |

### 2. 保留實驗腳本 (1 個)

| 文件 | 目的 |
|------|------|
| `exp0_baseline.sbatch` | DeSTA2.5 基準線 |

### 3. 歸檔舊腳本 (6 個)

移至 `archive/` 目錄:

- `exp1_global32.sbatch`
- `exp2_add_local.sbatch`
- `exp3_add_deep_inj.sbatch`
- `exp4_add_diversity.sbatch`
- `exp5_add_alignment.sbatch`
- `exp6_full_orca.sbatch`

### 4. 輔助腳本 (2 個)

| 文件 | 功能 |
|------|------|
| `submit_all.sh` | 一鍵提交所有 4 個實驗 |
| `verify_setup.sh` | 驗證實驗配置正確性 |

### 5. 文檔更新

| 文件 | 更新內容 |
|------|---------|
| `README.md` | 完整重寫,詳細說明 4 實驗設計 |

---

## 實驗設計對比

### 舊設計 (8 實驗)

```
Exp 0: Baseline
Exp 1: Global 32L
Exp 2: + Local
Exp 3: + Deep Injection
Exp 4: + Diversity Loss
Exp 5: + Alignment Loss
Exp 6: Full ORCA
Exp 7: Full ORCA (duplicate)
```

**問題**:

- ❌ 太多增量步驟
- ❌ 故事線不清晰
- ❌ 時間成本高 (16 天)
- ❌ 有重複實驗

### 新設計 (4 實驗)

```
Exp 0: DeSTA2.5 Baseline
Exp 1: ORCA Architecture (Dual-branch + Deep Inj)
Exp 2: + Orthogonality (All 3 losses)
Exp 3: Full ORCA (Validation)
```

**優勢**:

- ✅ 清晰的兩階段改進 (架構 → 正交性)
- ✅ 符合論文核心概念
- ✅ 節省 50% 時間 (8 天)
- ✅ 無冗餘實驗

---

## 配置差異矩陣

| Component | Exp 0 | Exp 1 | Exp 2 | Exp 3 |
|-----------|-------|-------|-------|-------|
| **Connector** | qformer_1 | orca_hybrid | orca_hybrid | orca_hybrid |
| **ORCA Enabled** | ❌ | ✅ | ✅ | ✅ |
| **Global Cross-Attn** | - | ✅ | ✅ | ✅ |
| **Local Branch** | - | ✅ | ✅ | ✅ |
| **Deep Injection** | - | ✅ | ✅ | ✅ |
| **L_ortho_diversity** | - | 0.0 | 0.05 | 0.05 |
| **L_ortho_qformer_local** | - | 0.0 | 0.05 | 0.05 |
| **L_align_layerwise** | - | 0.0 | 0.05 | 0.05 |

---

## 預期結果

### 性能改善路徑

```
Exp 0 (48-49) 
   ↓ +2-3 (Architecture: 60%)
Exp 1 (50-51)
   ↓ +1-2 (Orthogonality: 40%)
Exp 2 (51-52)
   ↓ ±0 (Validation)
Exp 3 (51-52)
```

### 組件貢獻分析

| 改進 | Hmean Δ | 貢獻比例 | 關鍵組件 |
|------|---------|---------|---------|
| **Architecture** | +2-3 | ~60% | Dual-branch + Deep Injection |
| **Orthogonality** | +1-2 | ~40% | 3 Orthogonality Losses |
| **Total** | +3-4 | 100% | Complete ORCA-DeSTA |

---

## 使用指南

### 快速開始

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio/examples/train/ablation_sbatch

# 1. 驗證設置
bash verify_setup.sh

# 2. 提交所有實驗
bash submit_all.sh

# 3. 監控狀態
squeue -u $USER
```

### 查看日誌

```bash
# 實時監控
tail -f slurm-report/ablation_exp1_architecture_*.out

# 檢查損失
grep "loss" slurm-report/ablation_exp2_orthogonality_*.out | tail -20
```

### 評估結果

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio/examples/evaluation

# 評估單個實驗
python sakura_eval.py --model_id /path/to/checkpoint-latest

# 批量評估
for exp_dir in /work/voidful2nlp/desta/outputs/ablation_minimal/*; do
    python sakura_eval.py --model_id ${exp_dir}/checkpoint-latest
done
```

---

## 驗證清單

### 實施前驗證 ✅

- [x] 4 個實驗腳本已創建
- [x] 舊腳本已歸檔
- [x] README 已更新
- [x] 輔助腳本已創建
- [x] 權限已設置 (chmod +x)
- [x] 配置差異已驗證

### 運行時驗證 (待完成)

- [ ] 所有實驗成功提交
- [ ] Exp 0: 只有 LLM loss
- [ ] Exp 1: 只有 LLM loss (無 ORCA losses)
- [ ] Exp 2: LLM loss + 3 個 ORCA losses
- [ ] Exp 3: 與 Exp 2 相同
- [ ] 正交性損失逐漸降低

### 評估後驗證 (待完成)

- [ ] Exp 0→1: Hmean 提升 2-3 分
- [ ] Exp 1→2: Hmean 提升 1-2 分
- [ ] Exp 2≈3: 結果一致 (±0.5)
- [ ] Multi-speaker: 主要在 Exp 1 改善
- [ ] Language: 主要在 Exp 2 改善

---

## 論文對應

### 題目: "ORCA-DeSTA: Orthogonal Residual Complementary Acoustics"

| 論文關鍵詞 | 對應實驗 | 驗證內容 |
|-----------|---------|---------|
| **Complementary Acoustics** | Exp 1 | Global (style) + Local (prosody) |
| **Residual** | Exp 1 | Gated residual in deep injection |
| **Orthogonal** | Exp 2 | 3 orthogonality losses |

### 核心貢獻展示

1. **架構創新** (Exp 0→1):
   - 雙分支互補聲學特徵
   - 深度跨模態融合

2. **正交性約束** (Exp 1→2):
   - 確保特徵真正互補
   - 改善跨模態對齊

---

## 時間與資源

### 計算資源

- **每個實驗**: 4 × A100 (40GB), 200GB RAM
- **總需求**: 4 實驗 × 4 GPUs = 16 GPU-days
- **並行運行**: 可同時運行 4 個實驗

### 時間估算

- **訓練**: ~48 小時/實驗 (5 epochs)
- **評估**: ~2 小時/實驗
- **總時間**:
  - 串行: 8 天
  - 並行: 2 天 (如果有足夠 GPU)

---

## 後續步驟

1. **提交實驗**: 使用 `submit_all.sh`
2. **監控訓練**: 定期檢查日誌和損失
3. **評估結果**: 訓練完成後運行 Sakura benchmark
4. **分析結果**: 對比 4 個實驗的性能差異
5. **撰寫論文**: 使用結果支撐論文論點

---

## 聯絡與支援

- **文檔**: `README.md` (詳細說明)
- **計劃**: `implementation_plan.md` (設計理念)
- **驗證**: `verify_setup.sh` (配置檢查)

---

**狀態**: ✅ 已完成並驗證,準備提交實驗
