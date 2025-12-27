# Minimal ORCA Ablation Study

## 實驗總覽

精簡的 4 實驗設計,清晰展示 ORCA-DeSTA 相對於 DeSTA2.5 的核心改進:

1. **架構創新**: 雙分支 (Global + Local) + 深度注入
2. **正交性約束**: 3 個正交性損失確保互補特徵

**優勢**:

- ⏱️ 節省 50% 時間 (8 天 vs 16 天)
- 📊 清晰故事線
- 🎯 符合論文 "Orthogonal Residual Complementary Acoustics"

---

## 實驗列表

| Exp | 名稱 | Dual-Branch | Deep Inj | Ortho Losses | 預期 Hmean | 腳本 |
|-----|------|------------|----------|--------------|-----------|------|
| **0** | DeSTA2.5 Baseline | ❌ | ❌ | ❌ | 48-49 | `exp0_baseline.sbatch` |
| **1** | ORCA Architecture | ✅ | ✅ | ❌ | 50-51 | `exp1_orca_architecture.sbatch` |
| **2** | + Orthogonality | ✅ | ✅ | ✅ | 51-52 | `exp2_add_orthogonality.sbatch` |
| **3** | Full ORCA | ✅ | ✅ | ✅ | 51-52 | `exp3_full_orca.sbatch` |

**組件說明**:

- **Dual-Branch**: Global (Q-Former, 8 tokens) + Local (Conv1d 4x downsample)
- **Deep Inj**: Gated cross-attention in all LLM decoder layers
- **Ortho Losses**: L_ortho_diversity + L_ortho_qformer_local + L_align_layerwise

---

## 快速開始

### 提交所有實驗

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio/examples/train/ablation_sbatch

# 提交 4 個實驗
sbatch exp0_baseline.sbatch
sbatch exp1_orca_architecture.sbatch
sbatch exp2_add_orthogonality.sbatch
sbatch exp3_full_orca.sbatch
```

### 檢查實驗狀態

```bash
squeue -u $USER
```

### 查看日誌

```bash
tail -f slurm-report/ablation_exp0_baseline_*.out
tail -f slurm-report/ablation_exp1_architecture_*.out
tail -f slurm-report/ablation_exp2_orthogonality_*.out
tail -f slurm-report/ablation_exp3_full_orca_*.out
```

---

## 實驗詳細配置

### Exp 0: DeSTA2.5 Baseline

**目的**: 建立基準線

**配置**:

```yaml
connector:
  mode: qformer_1
  num_hidden_layers: 6
  prompt_size: 64

orca:
  enabled: false
```

**預期**: Hmean ~48-49

---

### Exp 1: ORCA Architecture

**目的**: 評估雙分支架構 + 深度注入的貢獻

**配置**:

```yaml
connector:
  mode: orca_hybrid

orca:
  enabled: true
  # Architecture
  global_cross_attn: true
  local_enabled: true
  deep_injection_enabled: true
  
  # Disable all losses
  ortho_diversity_weight: 0.0
  ortho_weight_qformer_local: 0.0
  align_weight_local: 0.0
```

**關鍵特性**:

- ✅ Global branch: Q-Former cross-attention (8 tokens)
- ✅ Local branch: Conv1d 4x downsample (prosody tokens)
- ✅ Deep injection: Gated cross-attention in all LLM layers
- ❌ 無正交性約束

**預期**: Hmean ~50-51 (+2-3 from architecture)

**展示**:

- 互補聲學特徵 (global style + local prosody)
- 深度跨模態融合的效果

---

### Exp 2: + Orthogonality Losses

**目的**: 評估正交性約束的貢獻

**配置**:

```yaml
orca:
  # Architecture (same as Exp 1)
  enabled: true
  global_cross_attn: true
  local_enabled: true
  deep_injection_enabled: true
  
  # Enable all 3 orthogonality losses
  ortho_diversity_weight: 0.05      # L_ortho_diversity
  ortho_weight_qformer_local: 0.05  # L_ortho_qformer_local
  align_weight_local: 0.05          # L_align_layerwise
```

**關鍵特性**:

- ✅ 所有架構組件
- ✅ L_ortho_diversity: Global tokens 內部多樣性
- ✅ L_ortho_qformer_local: Global-Local 正交性 (新!)
- ✅ L_align_layerwise: 逐層音頻-文本對齊

**預期**: Hmean ~51-52 (+1-2 from orthogonality)

**展示**:

- 正交性確保真正互補的特徵
- 對齊損失改善跨模態理解

---

### Exp 3: Full ORCA (Validation)

**目的**: 驗證完整系統的一致性和可重現性

**配置**: 使用默認 ORCA config (與 Exp 2 相同)

**預期**: Hmean ~51-52 (與 Exp 2 一致)

**展示**: 系統穩定性和可重現性

---

## 預期結果

### 組件貢獻分析

| 改進 | Δ Hmean | 貢獻比例 | 關鍵發現 |
|------|---------|---------|---------|
| **Architecture** (Exp 0→1) | +2-3 | ~60% | 雙分支 + 深度注入是主要貢獻 |
| **Orthogonality** (Exp 1→2) | +1-2 | ~40% | 正交性確保互補特徵 |
| **Total** | +3-4 | 100% | ORCA-DeSTA 總改善 |

### 細分指標預期

| Metric | Exp 0 | Exp 1 | Exp 2 | Exp 3 | 主要改善來源 |
|--------|-------|-------|-------|-------|------------|
| Multi-speaker | 39-40 | 41-42 | 42-43 | 42-43 | Local branch (韻律) |
| Language-Single | 65-66 | 67-68 | 68-70 | 68-70 | Alignment loss |
| Language-Multi | 39-40 | 40-41 | 42-44 | 42-44 | Deep injection + Alignment |
| **Overall Hmean** | **48-49** | **50-51** | **51-52** | **51-52** | 架構 + 正交性 |

---

## 輸出目錄結構

```
/work/voidful2nlp/desta/outputs/ablation_minimal/
├── YYMMDD-HHMM_exp0_baseline/
│   ├── checkpoint-latest/
│   ├── checkpoint-1000/
│   └── ...
├── YYMMDD-HHMM_exp1_architecture/
├── YYMMDD-HHMM_exp2_orthogonality/
└── YYMMDD-HHMM_exp3_full_orca/
```

---

## 評估

訓練完成後,在 Sakura benchmark 上評估:

```bash
cd /work/voidful2nlp/DeSTA2.5-Audio/examples/evaluation

# 評估所有實驗
for exp_dir in /work/voidful2nlp/desta/outputs/ablation_minimal/*; do
    exp_name=$(basename $exp_dir)
    echo "Evaluating: $exp_name"
    python sakura_eval.py --model_id ${exp_dir}/checkpoint-latest
done
```

---

## 訓練日誌監控

### 關鍵指標

**Exp 0 (Baseline)**:

- 只有 `loss` (LLM loss)

**Exp 1 (Architecture)**:

- 只有 `loss` (無 ORCA losses)
- 應該比 Exp 0 收斂更快

**Exp 2 & 3 (Orthogonality)**:

- `loss` + `L_ortho_diversity` + `L_ortho_qformer_local` + `L_align_layerwise`
- 正交性損失應該逐漸降低

### 監控命令

```bash
# 實時監控損失
grep "loss" slurm-report/ablation_exp2_orthogonality_*.out | tail -20

# 檢查正交性損失趨勢
grep "L_ortho" slurm-report/ablation_exp2_orthogonality_*.out | tail -20
```

---

## 資源需求

- **GPU**: 4 × A100 (40GB) per experiment
- **RAM**: 200GB per experiment
- **Time**: ~48 hours per experiment (5 epochs)
- **Total**: 8 days for all 4 experiments (可並行)

---

## 舊版實驗 (已歸檔)

舊的 8 實驗設計已移至 `archive/` 目錄:

- `exp1_global32.sbatch`
- `exp2_add_local.sbatch`
- `exp3_add_deep_inj.sbatch`
- `exp4_add_diversity.sbatch`
- `exp5_add_alignment.sbatch`
- `exp6_full_orca.sbatch`

如需參考舊設計,請查看 `archive/` 目錄。

---

## 論文對應

這個精簡設計完美對應論文題目:

**"ORCA-DeSTA: Orthogonal Residual Complementary Acoustics for Audio-Language Models"**

| 論文概念 | 對應實驗 | 展示內容 |
|---------|---------|---------|
| **Complementary Acoustics** | Exp 1 | Dual-branch (Global + Local) |
| **Residual** | Exp 1 | Deep injection (gated residual) |
| **Orthogonal** | Exp 2 | 3 orthogonality losses |

---

## 常見問題

**Q: 為什麼從 8 個實驗減少到 4 個?**
A: 舊設計過於細緻,新設計聚焦於兩大創新 (架構 + 正交性),故事更清晰。

**Q: Exp 2 和 Exp 3 有什麼區別?**
A: Exp 3 是驗證實驗,確保完整配置與 Exp 2 一致且可重現。

**Q: 可以跳過某些實驗嗎?**
A: 建議全部運行。如果時間緊迫,最低要求是 Exp 0, 1, 2。

**Q: 如何解讀結果?**
A:

- Exp 0→1 的改善 = 架構貢獻
- Exp 1→2 的改善 = 正交性貢獻
- Exp 2≈3 = 驗證系統穩定性

---

## 聯絡

如有問題請參考 `implementation_plan.md` 或查看代碼註釋。
