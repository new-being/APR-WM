# RTWX-O0G1b 报告 — Object Reference-Frame Alignment

日期：2026-08-29  
状态：**正式冻结** `geometry_reference_aligned`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G1B_PREREG.md`  
产物：`runs/rtwx_o0g1b/{run.json,metrics.json,cache_o0g1b.npz}`  
相机：**`head_camera`**；seed **23601**；8×32；**无训练**  
\(\delta_O\)：`021_cup/model_data0.json` 的 `center*scale`（**known geometry prior**；**非** O0G \(+5.11\,\mathrm{cm}\) 拟合）  
**不改 O0G / O0R**；**不解锁 O1**；**允许预注册 G2**（本格不跑 G2）。

## 一句话

B0（表面质心 vs pose）仍失败（median \(5.42\,\mathrm{cm}\)）。  
B1（object-frame \(\delta_O\) 校正）**通过**：median **\(1.65\,\mathrm{cm}\le2\,\mathrm{cm}\)**，\(E_p=0.021\)。

\[
\boxed{\texttt{geometry\_reference\_aligned}}
\]

## Gates

| Gate | 条件 | B1 |
|------|------|-----|
| Primary | median \(\le5\,\mathrm{cm}\)，\(E_p\le0.20\) | **PASS**（1.65 cm / 0.021） |
| Strong | median \(\le2\,\mathrm{cm}\) | **PASS** |

## 主表

| 仪器 | \(E_p\) | median \(\|e\|\) | bias \(xyz\) (cm) | primary | strong |
|------|--------:|-----------------:|-------------------:|:-------:|:------:|
| **B0** \(\hat p^{\mathrm{surf}}\) | 0.069 | **5.42 cm** | — | ✗ | ✗ |
| **B1** \(\hat p^{\mathrm{surf}}-R_{BO}\delta_O\) | **0.021** | **1.65 cm** | (−0.62, +0.43, +0.80) | ✓ | ✓ |
| B1 world-\(z\) secondary | 0.021 | 1.65 cm | （与 B1 同量级；直立杯） | audit only | — |

\(\delta_O\approx[0,\,4.475\,\mathrm{cm},\,0]\)（mesh AABB）；\(R_{BO}\) 来自每帧 cup quat（非常数 world-\(z\) 拟合）。

## 读数

1. O0G 冻结不变：失败机制是 **reference-frame offset**，不是相机几何坏掉。  
2. 已知资产先验 \(\delta_O\) **足够**把 oracle mask+depth 位置拉到 strong 门内。  
3. **G2 可预注册**：\(RGB\to\hat M\to p^{\mathrm{surf}}\to p^{\mathrm{pose}}\)；网络只学 localization。  
4. O1 仍 LOCKED（须 learned 视觉格 PASS）。  
5. 状态语义：\(s^O\) 须标明 \(O=\) pose origin（本格目标），区别于 surface centroid。

## Pattern

```text
B0 = FAIL  (replicates O0G/G1)
B1 = PASS + strong
pattern = geometry_reference_aligned
unlocks_g2_prereg = true
unlocks_o1 = false
```
