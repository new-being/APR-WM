# RTWX-X0EH 预注册 — Receding-Horizon Sufficiency Audit

日期：2026-08-28  
状态：**已冻结（fresh 后）** — pattern `short_horizon_sufficient`  
依赖：X0ES1 = `rare_instability_intrinsic_map`；X0ES2 = `instability_persists`（contraction STOP）。  
本格：**纯诊断**。冻结 X0E-M2；禁止重训 / 改 basis / residual / history / contraction / latent / \(R_P\) / 改先前 ledger。

## 问题

\[
\boxed{
\text{周期性真实状态重置能否在不修改 M2 的情况下抑制 rare recursive catastrophe？}
}
\]

估计 deployment-relevant \(K_{\max}\)（两次真实观测间允许的 native action 数）。

## 数据

96×120 native-qpos，同 X0E 激励；seed family **11601+**（与 8601/9601/10601 不相交）。

## Part A — open-loop \(T_{\mathrm{div}}\)

冻结 M2；每合法起点 \(H=1..50\)。  
catastrophe：\(\max E_h>10^6\) 或 nonfinite。  
\(T_{\mathrm{div}}=\min\{h:E_h>10^6\}\)（否则 \(\infty\)）。描述性 \(T_1,T_{10},T_{100}\)。  
生存：\(P(T_{\mathrm{div}}\le\{1,2,4,8,16,32,50\})\)；\(K_{99}=\max\{K:P(T_{\mathrm{div}}>K)\ge0.99\}\)。

G0：\(N_{\mathrm{cat}}\ge5\)，否则 `rare_event_insufficient` STOP。

## Part B — receding horizon

\(K\in\{1,2,4,8,16,50\}\)。每 \(K\) 步用真实 \(x\) reset；只评估执行的前 \(K\) 步。概念 \(H_{\mathrm{predict}}=16\)（\(K\le16\)）。  
指标：\(r_{\mathrm{cat}}(K)\)、\(N_{\mathrm{nonfinite}}\)、\(E_{\mathrm{exec}}(K)\)、\(E_{\mathrm{end}}(K)\)、\(R_\rho(K)=P(\exists h\le K:\rho_J>1)\)。

主候选 **\(K=4\)**（预注册，不事后改）。

## Gates

- **G1**：\(r_{\mathrm{cat}}(4)\le0.1\,r_{\mathrm{cat}}(50)\) 且 \(N_{\mathrm{nonfinite}}(4)=0\)
- **G2**：\(E_{\mathrm{exec}}(4)\le0.75\,E_{\mathrm{identity}}(4)\)
- **G3**：\(\mathrm{Spearman}(K,E_{\mathrm{end}})\ge0.7\) 且 \(E_{\mathrm{end}}(16)>E_{\mathrm{end}}(4)\)

## Patterns

`short_horizon_sufficient` | `short_horizon_stable_but_inaccurate` | `replanning_insufficient` | `rare_event_insufficient`

## Shadow（不进 pattern）

adaptive：\(\rho_J>1\Rightarrow K=1\)，否则 \(K=8\)；对比 fixed 4 / 8。
