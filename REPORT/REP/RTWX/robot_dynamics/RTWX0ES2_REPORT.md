# RTWX-X0ES2 报告 — Stability-Constrained Increment Model

日期：2026-08-28  
状态：**正式冻结** `instability_persists`  
预注册：`REPORT/REG/RTWX/RTWX0ES2_PREREG.md`  
产物：`runs/rtwx_x0es2/{selection.json,run.json,fresh_splits.npz,W_s1.npy,metrics.json}`  
同一 X0E-M2 basis（\(P=1752\)）；仅改 \(W\) 拟合约束。**不改** X0E/X0E1/X0ES/X0ES1。无 \(R_P\)。

## 一句话

在 val 上，**最小合格 \(\lambda=0.01\)** 能同时满足零 catastrophic 与 X0E1 reference-match 阈值；  
但在 **fresh 96×120（seed 10601）** 上，contractive M2（S1）**未能通过 G1/G2/G3**：  
稀有递归爆炸有所降低（\(62\) vs B0 \(105\) 窗，约 \(1.7\times\)），但远未达到预注册要求的 **\(10\times\)**，且仍有 **62** 个 nonfinite 窗。  
Pattern：**`instability_persists`** — 按预注册 **STOP** 此 Jacobian contraction 路线；**不解锁** X0ES3。

## Val 选择（仅原 X0E val）

| \(\lambda\) | V1 有限 | V2 \(N_{\mathrm{cat}}=0\) | V3 E1/Er10/Er50 | 合格 |
|---|:---:|:---:|:---:|:---:|
| \(10^{-4}\) | ✗（1 cat） | ✗ | ✓ | ✗ |
| \(10^{-3}\) | ✗（1 cat） | ✗ | ✓ | ✗ |
| **\(10^{-2}\)** | ✓ | ✓ | ✓（0.719 / 0.616 / 0.712） | **✓** |
| \(10^{-1}\) | ✓ | ✓ | ✓ | ✓ |
| \(1\) | ✓ | ✓ | ✓ | ✓ |

选 **最小** 合格 \(\lambda=0.01\)（不是 accuracy 最优）。

Val 机制侧：\(P(\rho_J>1)=2.4\%\)，\(Q_{99}[\sigma_{\max}(J)]=2.38\)。

## Fresh 确认（B0=冻结 LS-M2，S1=\(\lambda=0.01\) contractive）

6720 窗（\(H=1\ldots50\)），catastrophic \(=\max E_H>10^6\) 或 nonfinite。

### 预测

| 模型 | \(E_1\) | \(E_{\mathrm{roll}10}\) | \(E_{\mathrm{roll}50}\) |
|------|--------:|------------------------:|------------------------:|
| B0（M2） | 0.863 | inf | inf |
| S1（contractive） | 0.844 | inf | inf |

### 稳定性

| 模型 | \(N_{\mathrm{cat}}\) | \(r_{\mathrm{cat}}\) | \(N_{\mathrm{nonfinite}}\) |
|------|---------------------:|---------------------:|---------------------------:|
| B0 | 105 | **1.56%** | 105 |
| S1 | 62 | **0.92%** | 62 |

\(r_{\mathrm{cat}}^{S1}/r_{\mathrm{cat}}^{B0}\approx 0.59\)（约 \(1.7\times\) 下降，**未达** \(10\times\)）。

### 机制（起点 Jacobian）

| 量 | B0 | S1 | 方向 |
|----|----|----|------|
| \(P(\rho_J>1)\) | 4.81% | 3.10% | ↓ 但未减半（需 \(\le 2.40\%\)） |
| \(Q_{99}[\rho_J]\) | 2.13 | 1.79 | ↓ |
| \(Q_{99}[\sigma_{\max}(J)]\) | 5.30 | 4.24 | ↓ |

机制统计朝收缩方向移动，但幅度不足以满足 G3；稳定性门 G1 因 nonfinite 与降幅不足而失败。

## 门

| 门 | 要求 | 结果 |
|----|------|------|
| **G1** | \(N_{\mathrm{nonfinite}}^{S1}=0\) 且 \(r_{\mathrm{cat}}^{S1}\le 0.1\,r_{\mathrm{cat}}^{B0}\) | **未过**。62 nonfinite；\(0.92\% > 0.156\%\) |
| **G2** | \(E_1\le 0.802\)，\(E_{\mathrm{roll}10}\le 0.763\)，\(E_{\mathrm{roll}50}\le 0.835\) | **未过**。\(E_1=0.844\)；roll 因爆炸为 inf |
| **G3** | \(P_{S1}(\rho>1)\le 0.5\,P_{B0}(\rho>1)\) | **未过**。3.10% \(>\) 2.40% |

Pattern：**`instability_persists`**（候选已在 val 选定，fresh 后 G1 或 G3 不过）。

## 读数

1. **Val→fresh 泛化缺口：** \(\sigma_{\max}\) 软约束在 val 上可清零 catastrophic，但 fresh 上 B0 基线率（1.56%）高于 X0ES1 fresh（0.61%），S1 仍保留大量爆炸；说明 **点态 Jacobian 惩罚不足以消除稀有递归不稳定**。
2. **不是纯 accuracy tradeoff：** G2 也失败，故不是 pattern B（`stability_accuracy_tradeoff`）；S1 在 fresh 上既未足够稳定，也未守住 reference-match。
3. **机制部分一致：** \(Q_{99}[\sigma_{\max}]\)、\(P(\rho>1)\) 均下降，支持“扩张 Jacobian 与爆炸相关”，但 **当前 \(\mathcal L_{\mathrm{contract}}\) 强度/形式不够**。
4. 无 \(R_P\)；不改 X0E1 `structure_not_in_robust_set`；**不扫额外 \(\lambda\)**（预注册 STOP）。

## 账本

```text
RTWX-X0ES1 = RAN / rare_instability_intrinsic_map     不改
RTWX-X0ES2 = RAN / instability_persists
             val: lambda=0.01 selected (smallest ok)
             fresh: G1/G2/G3 fail; contraction penalty insufficient
X0ES3 robust capacity = LOCKED
TASK-XL / R10 = LOCKED
```
