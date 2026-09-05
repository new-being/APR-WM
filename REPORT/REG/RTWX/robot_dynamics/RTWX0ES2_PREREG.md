# RTWX-X0ES2 预注册 — Stability-Constrained Increment Model

日期：2026-08-28  
状态：**已冻结（fresh 后）** — pattern `instability_persists`  
依赖：X0ES1 = **`rare_instability_intrinsic_map`**。X0E-M2 basis **不变**。  
禁止：新 basis；residual；history；latent；改 X0E/X0E1/X0ES/X0ES1 账本；本格写 \(R_P\) / 23×。

## 问题

\[
\boxed{
\text{约束同一套 X0E-M2 increment 的状态 Jacobian，能否消掉稀有递归爆炸并保住预测资格？}
}
\]

\(F_W(x,u)=x+W\phi(x,u)+b\)，推理参数量必须仍为 **1752**。只改 \(W\) 的拟合约束。

## 训练

仅原 X0E **train**。\(\gamma=0.98\)。

\[
\mathcal L=\mathcal L_{\Delta x}+\lambda\,\mathrm{mean}([\max(0,\sigma_{\max}(J)-0.98)]^2)
\]

\(\lambda\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}\)（禁止事后加格点）。从 LS \(W\) 初始化。

## Val 选择（仅原 X0E val）

V1 全有限；V2 \(N_{\mathrm{cat}}=0\)（定义同 X0ES：\(\max_H E_H>10^6\) 或 nonfinite，\(H=1..50\)）；  
V3 \(E_1\le 0.802\)，\(E_{\mathrm{roll}10}\le 0.763\)，\(E_{\mathrm{roll}50}\le 0.835\)。  
选 **最小** 合格 \(\lambda\)。无一合格 → `constraint_selection_failure`，**不采** fresh。

## Fresh 确认

seed **10601**，96×120，同激励。B0=冻结原 M2，S1=选定 contractive \(W\)。不按 fresh 重选 \(\lambda\)。

G1：\(N_{\mathrm{nonfinite}}^{S1}=0\) 且 \(r_{\mathrm{cat}}^{S1}\le 0.1\,r_{\mathrm{cat}}^{B0}\)（分母为 **本 fresh 上 B0**）。  
G2：上述 V3 阈值。  
G3：\(P_{S1}(\rho>1)\le 0.5\,P_{B0}(\rho>1)\)。

## Patterns

`stable_structure_supported` | `stability_accuracy_tradeoff` | `instability_persists` | `constraint_selection_failure`

仅 A 解锁 X0ES3 robust capacity。无 \(R_P\)。
