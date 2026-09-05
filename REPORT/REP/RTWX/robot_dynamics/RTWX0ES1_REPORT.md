# RTWX-X0ES1 报告 — Rare Local Instability Confirmation

日期：2026-08-28  
状态：**正式冻结** `rare_instability_intrinsic_map`  
预注册：`REPORT/REG/RTWX/RTWX0ES1_PREREG.md`  
产物：`runs/rtwx_x0es1/{run.json,fresh_splits.npz,metrics.json}`  
冻结 M2；96×120 fresh（seed 9601）。**不改** X0E/X0E1/X0ES。无容量声称。

## 一句话

在全新数据上，**高 \(\rho_J>1\) 能预测稀有 catastrophe**（G0/G1 通过）。  
短历史 \(z^{(1)}\) 的增益 \(G_H=0.040<0.20\)（CI \((0.003,0.076)\)，下界虽 \(>0\) 但幅度不够），因此 **不是** 充分的 hidden-state aliasing。  
按预注册：稀有爆炸更像 **冻结 M2 映射本身的局部不稳定递归算子**，而不是“再补一步 \(\Delta q_{t-1}\) 就能补上的 Markov 缺口”。

## 门

| 门 | 结果 |
|----|------|
| G0 | **通过**。\(N_{\mathrm{cat}}=41\ge 5\)（6720 窗，约 0.61%） |
| G1 | **通过**。recall \(=0.854\ge 0.80\)；\(RR=141.5\ge 10\)；Fisher \(p=3.2\times 10^{-44}\) |
| G2 | **未过** \(G_H\ge 0.20\)。\(n=265\) 个 \(\rho>1\) 样本；\(E_{\mathrm{NN}}^{(0)}=1.54\to E_{\mathrm{NN}}^{(1)}=1.47\) |

Pattern：`rare_instability_intrinsic_map`

## \(2\times 2\)（fresh，起点 \(\rho_J>1\)）

| | catastrophe | stable |
|---|---:|---:|
| \(\rho_J>1\) | 35 | 231 |
| \(\rho_J\le 1\) | 6 | 6448 |

precision \(=0.132\)（高风险仍多为稳定窗）；AUPRC \(=0.50\)（极不平衡下不作主指标）。高风险窗约占 4.0%。

## 读数

1. **G1 在独立数据上确认了 X0ES 的 Jacobian 观察：** \(\rho>1\) 是可复现的 catastrophe 风险标记，不是原 split 上的事后相关。
2. **precision 低：** \(\rho>1\) 是敏感而非特异。多数高 \(\rho\) 窗并不炸；但几乎所有爆炸都来自高 \(\rho\)（6 个漏报）。
3. **一步历史几乎不消歧。** \(G_H\approx 4\%\)，达不到 20%。下一步若建模，预注册方向是 **contraction / spectral-radius 约束**，而不是先加 \(h_t^{ctrl}\) 或加 residual capacity。
4. 无 \(R_P\)；不改 X0E1 的 `structure_not_in_robust_set`。

## 账本

```text
RTWX-X0ES  = pathology_not_reproduced     不改
RTWX-X0ES1 = RAN / rare_instability_intrinsic_map
             G1 confirmed on fresh data; history gain insufficient
next allowed: contraction/stable structured model cell
TASK-XL / R10 / capacity rewrite = LOCKED
```
