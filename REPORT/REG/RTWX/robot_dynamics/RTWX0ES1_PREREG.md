# RTWX-X0ES1 预注册 — Rare Local Instability Confirmation

日期：2026-08-28  
状态：**已冻结**；**正式 RAN 2026-08-28**；**`rare_instability_intrinsic_map`**。报告：`REPORT/REP/RTWX/RTWX0ES1_REPORT.md`。  
依赖：X0ES = **`pathology_not_reproduced`**（已冻结，不改判）。X0E-M2 **参数冻结**。  
禁止：重训 M2；改 basis；residual；latent；改 catastrophe 阈值；改 \(\rho>1\) 阈值；改 X0E/X0E1/X0ES 账本；容量声称；TASK-XL/R10。  
数据：**全新** 96×120，与 X0E 同激励/同机器人，**disjoint seeds**。旧 X0E1 cache **仅**作 NN reference bank 与冻结 \(W\)。

## 问题

\[
\boxed{
\text{冻结 M2 的高局部 }\rho_J\text{ 能否在全新数据上预测稀有 rollout catastrophe？}
}
\]

以及：高 \(\rho\) 是映射固有不稳，还是 \((q,\dot q,q^{tar})\) 非充分 Markov、存在隐藏 controller 历史？

## 规模

96 fresh episodes × 120 native qpos。事件 \(<5\) **不**加采。

## Catastrophe / 风险（与 X0ES 相同，不改）

\(H=1,\ldots,50\)。\(C=1\) iff \(\max E_H>10^6\) 或 nonfinite。  
\(E_H=\|\hat x-x\|_2/\mathrm{rms}(x_{\mathrm{old\ train}})\)。  
高风险：窗起点 \(\rho_J>1\)（不搜索阈值）。

## G0

\(N_{\mathrm{cat}}\ge 5\)，否则 `rare_event_insufficient`。

## G1

\(\mathrm{Recall}_\rho=P(\rho>1\mid C=1)\ge 0.8\)，  
\(RR=P(C\mid\rho>1)/P(C\mid\rho\le 1)\ge 10\)（分母为 0 且分子 \(>0\) 视为通过），  
Fisher exact \(p<0.01\)。  
否则 `rare_instability_not_confirmed`。通过 → `rare_local_expansion_confirmed` 仅作为 G1 内部标签，最终 pattern 见下。

## G2（仅 G1 通过后）

\(z^{(0)}=[q,\dot q,e]\)，\(z^{(1)}=[z^{(0)},\Delta q_{t-1},\Delta\dot q_{t-1},e_{t-1}]\)。  
旧 train 为 k=5 NN bank，**不**拟合网络。只在 fresh \(\rho>1\) 且 \(t\ge 1\) 上评。  
\(G_H=1-E_{\mathrm{NN}}^{(1)}/E_{\mathrm{NN}}^{(0)}\)。  
\(G_H\ge 0.20\) 且 paired bootstrap 95% CI 下界 \(>0\) → `rare_instability_history_aliasing`。  
否则 `rare_instability_intrinsic_map`。

## 最终 pattern（仅四类）

`rare_event_insufficient` | `rare_instability_not_confirmed` | `rare_instability_history_aliasing` | `rare_instability_intrinsic_map`

无容量；不解锁 contraction/X0E2，除非本格给出对应机制 pattern。
