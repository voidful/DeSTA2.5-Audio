# ORCA-DeSTA 改進總結

## 修改日期

2025-12-24

## 修改內容

### 1. ✅ 新增 Global-Local 正交性損失函數

**文件**: `desta/models/modeling_desta25.py`

**修改位置**: `compute_orca_losses()` 方法 (Line 1072-1118)

**新增功能**:

```python
# Orthogonality between global and local tokens (ensure complementary features)
if global_tokens is not None and local_tokens is not None:
    # Normalize tokens
    g_norm = F.normalize(global_tokens, dim=-1)  # [B, K_g, H]
    l_norm = F.normalize(local_tokens, dim=-1)   # [B, K_l, H]
    
    # Compute cross-similarity: should be close to 0 for orthogonality
    # Sample local tokens if too many to reduce computation
    max_local_samples = 100  # Limit local tokens for efficiency
    if l_norm.size(1) > max_local_samples:
        # Uniformly sample local tokens
        indices = torch.linspace(0, l_norm.size(1) - 1, max_local_samples, dtype=torch.long, device=l_norm.device)
        l_norm = l_norm[:, indices, :]
    
    cross_sim = torch.einsum("bgh,blh->bgl", g_norm, l_norm)  # [B, K_g, K_l]
    L_ortho_gl = (cross_sim ** 2).mean()  # Minimize squared similarity
    losses["L_ortho_qformer_local"] = self.config.orca_ortho_weight_qformer_local * L_ortho_gl
```

**設計理念**:

- 確保 global tokens (風格特徵) 和 local tokens (韻律特徵) 之間正交
- 避免兩個分支學習到重疊的信息
- 通過最小化 cross-similarity 的平方來實現正交性
- 為了計算效率,當 local tokens 超過 100 個時進行均勻採樣

**對應論文題目**: 完全符合 "**Orthogonal** Residual **Complementary** Acoustics" 中的 "Orthogonal" 概念

---

### 2. ✅ 將 Local Branch 下採樣從 2x 改為 4x

**目的**: 減少計算量,提高訓練和推理效率

**修改文件**:

#### A. 模型配置默認值

**文件**: `desta/models/modeling_desta25.py` (Line 640-660)

```python
# 修改前
orca_audio_position_scale=5.0,  # Position interpolation scale for audio tokens (adjusted for 2x downsample)
orca_local_downsample=2,

# 修改後
orca_audio_position_scale=2.5,  # Position interpolation scale for audio tokens (adjusted for 4x downsample)
orca_local_downsample=4,
```

#### B. 訓練配置文件

修改了以下 3 個配置文件:

1. **`examples/train/config/desta25_llama31-8B_ORCAHybrid.yaml`**
2. **`examples/train/config/desta25_qwen3-0.6b_ORCAHybrid.yaml`**
3. **`examples/train/config/desta25_qwen3-4b_ORCAHybrid.yaml`**

```yaml
# 修改前
orca:
  local_downsample: 2
  audio_position_scale: 5.0  # RoPE position interpolation (adjusted for 2x downsample)

# 修改後
orca:
  local_downsample: 4
  audio_position_scale: 2.5  # RoPE position interpolation (adjusted for 4x downsample)
```

**RoPE 位置縮放調整說明**:

- 原來 2x downsample 使用 `audio_position_scale=5.0`
- 改為 4x downsample 後,調整為 `audio_position_scale=2.5`
- 這是因為下採樣倍數增加,需要相應調整 RoPE 的位置插值比例
- 計算邏輯: `5.0 / 2 = 2.5` (下採樣倍數翻倍,位置縮放減半)

---

## 效果預期

### 1. Global-Local 正交性損失

- ✅ **更強的互補性**: 確保 global 和 local 分支學習到真正不同的特徵
- ✅ **符合論文題目**: 完整實現 "Orthogonal Complementary Acoustics" 的概念
- ✅ **可控的訓練**: 通過 `orca_ortho_weight_qformer_local` 參數調整損失權重

### 2. 4x 下採樣

- ✅ **減少計算量**: Local tokens 數量減半,cross-attention 計算量降低約 50%
- ✅ **加快訓練速度**: 特別是在 deep injection 的每一層都會受益
- ✅ **保持效果**: 4x 下採樣仍能保留足夠的韻律信息
- ✅ **節省顯存**: 更少的 tokens 意味著更低的顯存佔用

---

## 訓練監控

訓練時可以觀察以下新增的損失項:

```python
# 訓練日誌中會出現
losses = {
    "L_ortho_diversity": ...,      # Global tokens 內部正交性
    "L_ortho_qformer_local": ...,  # Global-Local 正交性 (新增!)
    "L_align_layerwise": ...,      # 逐層對齊損失
}
```

**預期行為**:

- `L_ortho_qformer_local` 應該逐漸降低
- 理想情況下收斂到接近 0 的值
- 如果損失過大,可以調整 `orca_ortho_weight_qformer_local` 權重

---

## 向後兼容性

✅ **完全向後兼容**:

- 舊的 checkpoint 可以正常加載
- 如果 checkpoint 中沒有 `L_ortho_qformer_local`,會自動跳過
- 配置文件可以靈活調整 downsample 倍數

---

## 測試建議

1. **單元測試**: 驗證新損失函數的計算正確性
2. **小規模訓練**: 使用 debug 配置測試幾個 steps
3. **監控損失**: 確保 `L_ortho_qformer_local` 正常收斂
4. **消融實驗**: 對比有無 global-local 正交損失的效果

---

## 論文對應關係

| 論文題目關鍵詞 | 實現狀態 | 對應代碼 |
|--------------|---------|---------|
| **Orthogonal** | ✅ **完整實現** | `L_ortho_diversity` + `L_ortho_qformer_local` |
| **Residual** | ✅ 完整實現 | `h + gate * cross_out` |
| **Complementary Acoustics** | ✅ 完整實現 | Global (Q-Former) + Local (Conv1d) |

**結論**: 修改後的實現完全符合論文題目 "ORCA-DeSTA: Orthogonal Residual Complementary Acoustics for Audio-Language Models" 的核心概念! 🎉
